"""
PyCBC Data Generator for Gravitational Wave Waveforms

This module generates gravitational wave waveforms using PyCBC, projects them to
detectors (H1, L1, V1, etc.), and returns PyTorch DataLoaders ready for training.

Usage:
    from data_generator import pycbc_data_generator
    
    config = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
    }
    
    result = pycbc_data_generator(config, num_samples=1000)
    train_loader = result['train_loader']
"""

import numpy as np
import torch
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, Callable, List, Tuple, Union
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.distributions as dist
import itertools
import random
from torch.utils.data import TensorDataset, DataLoader, random_split
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.psd import welch, interpolate
from pycbc.filter import highpass_fir, lowpass_fir, resample_to_delta_t
from torch.utils.data import Subset
import warnings
import sys, os
from time import perf_counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Real Data'))
try:
    from catalog_utils import *
    from real_data_loader import *
    from event_processor import *
    from comparison_utils import *
except ImportError:
    pass
from scipy.integrate import quad


################### Cosmological / Modified Gravity Constants ###################
_C = 2.998e8                          # speed of light, m/s
_G = 6.674e-11                        # gravitational constant, m^3 kg^-1 s^-2
_M_SUN = 1.989e30                     # solar mass, kg
_MPC = 3.086e22                       # megaparsec, m
_M_SUN_SEC = _G * _M_SUN / _C**3     # solar mass in seconds (~4.926e-6 s)
_H0 = 67.4e3 / _MPC                  # Hubble constant in 1/s
_OMEGA_M = 0.315
_OMEGA_LAMBDA = 0.685
_HIGHPASS_FC        = 35  # high-pass filter cutoff (Hz) applied to all generated signals
_M_OMEGA_PN = 0.1       # PN breakdown: (m1+m2)·omega = 0.1 (geometric units G=c=1)
_TAPER_FRACTION = 0.50  # phase taper: smooth to zero over top 50% of f_pn_cutoff
_RINGDOWN_TAPER_LEN = 128    # samples cosine-tapered to zero at array end before highpass


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time for readable runtime logging."""
    total_seconds = max(0.0, float(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.2f}s"
    return f"{secs:.2f}s"

def _validate_config(config: Dict[str, Callable]) -> None:
    """Validate configuration dictionary."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    
    if not config:
        raise ValueError("config cannot be empty")
    
    for key, value in config.items():
        if not callable(value):
            raise TypeError(f"config['{key}'] must be callable (e.g., a distribution function)")
        
        try:
            test = value(size=2)
            if not isinstance(test, np.ndarray):
                raise TypeError(f"config['{key}']() must return numpy array")
        except Exception as e:
            raise ValueError(f"config['{key}'] failed test call: {e}")


def _generate_parameter_sets(config: Dict[str, Callable], num_samples: int) -> List[Dict]:
    """Generate all parameter combinations upfront."""
    param_arrays = {}
    for param_name, dist_func in config.items():
        param_arrays[param_name] = dist_func(size=num_samples)
    
    param_dicts = []
    for i in range(num_samples):
        param_dict = {name: float(values[i]) for name, values in param_arrays.items()}
        param_dicts.append(param_dict)
    
    return param_dicts


def _generate_single_waveform(params: Dict, time_resolution: float, approximant: str,
                              f_lower: float, detectors: List[str], target_length: int,
                              add_noise: bool = True, whiten: bool = False) -> Dict:
    """Worker function to generate a single waveform and project to detectors at fixed length."""
    try:
        hp, hc = get_td_waveform(
            approximant=approximant,
            mass1=params['mass1'],
            mass2=params['mass2'],
            spin1z=params.get('spin1z', 0.0),
            spin2z=params.get('spin2z', 0.0),
            inclination=params.get('inclination', 0.0),
            coa_phase=params.get('coa_phase', 0.0),
            distance=params.get('distance', 410.0),  # Luminosity distance in Mpc, default 410 Mpc (GW150914)
            delta_t=time_resolution,
            f_lower=f_lower
        )

        # Shift waveform epoch to desired GPS time (affects antenna pattern and time delays)
        gps_time = params.get('gps_time', 1126259462.4)  # Default: GW150914 merger time
        hp.start_time += gps_time
        hc.start_time += gps_time

        # Get sky location parameters (defaults to north pole and zero polarization)
        ra = params.get('ra', 0.0)
        dec = params.get('dec', np.pi/2)  # North pole
        polarization = params.get('polarization', 0.0)

        # Project to detectors and fix to target length
        detector_signals = {}
        for det_name in detectors:
            detector = Detector(det_name)
            signal = detector.project_wave(hp, hc, ra, dec, polarization, method='lal')

            # Fix signal to target_length (crop or pad)
            signal_len = len(signal)
            print(f"[TD] {det_name}: signal_len={signal_len}, target_length={target_length}")
            if signal_len < 0 or signal_len > target_length:
                raise ValueError(f"[TD] Invalid signal_len: {signal_len} (target_length={target_length})")
            if signal_len >= target_length:
                # Crop from the end (keeps merger which is at the end)
                signal = signal[-target_length:]
            else:
                pad_len = target_length - signal_len
                print(f"[TD] {det_name}: pad_len={pad_len}, signal.data.shape={signal.data.shape}")
                padded_data = np.zeros(target_length, dtype=signal.dtype)
                if signal_len > 0 and pad_len >= 0:
                    padded_data[pad_len:] = signal.data[:]
                elif signal_len == 0:
                    print(f"[TD] {det_name}: signal_len is 0, using all zeros.")
                else:
                    raise ValueError(f"[TD] {det_name}: pad_len < 0 or signal_len < 0!")
                padded_epoch = signal.start_time - pad_len * signal.delta_t
                signal = TimeSeries(padded_data, delta_t=signal.delta_t, epoch=padded_epoch)

            detector_signals[det_name] = signal

        # Add noise to each detector signal (if enabled)
        if add_noise:
            for det_name in detectors:
                signal = detector_signals[det_name]

                # All signals now have exactly target_length, so use that for PSD calculation
                delta_t = signal.delta_t
                duration = target_length * delta_t  # Use target_length for consistency
                delta_f = 1.0 / duration

                # flen is number of frequency bins needed for PSD
                # For a real time series of length N, FFT produces N//2 + 1 frequency bins
                flen = target_length // 2 + 1

                # Create PSD with correct frequency resolution
                psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

                # Generate noise at exactly target_length
                noise = noise_from_psd(target_length, delta_t, psd)

                # Set noise epoch to match signal epoch for proper alignment
                noise._epoch = signal._epoch

                # Inject noise into signal (both guaranteed to be target_length)
                detector_signals[det_name] = signal.inject(noise)

        # Whiten signals (if enabled) - should be done AFTER adding noise
        if whiten:
            for det_name in detectors:
                signal = detector_signals[det_name]
                # whiten_waveform returns (whitened_numpy_array, psd, freqs)
                whitened_data, _, _ = whiten_waveform(
                    signal, 
                    delta_t=time_resolution, 
                    f_lower=f_lower,
                    apply_bandpass=True,
                    apply_tukey=True,
                    tukey_side='left'  # Preserve merger at end
                )
                # Store numpy array directly (no need to convert back to TimeSeries)
                detector_signals[det_name] = whitened_data

        result = {
            'success': True,
            'detectors': detector_signals,
            'params': params
        }

        return result

    except Exception as e:
        return {'success': False, 'error': str(e), 'params': params}


def _generate_waveforms_parallel(param_dicts: List[Dict],
                                time_resolution: float,
                                approximant: str,
                                f_lower: float,
                                num_workers: int,
                                show_progress: bool,
                                detectors: List[str],
                                target_length: int,
                                add_noise: bool,
                                whiten: bool) -> List[Dict]:
    """Generate waveforms in parallel using multiprocessing."""
    worker_func = partial(_generate_single_waveform,
                          time_resolution=time_resolution,
                          approximant=approximant,
                          f_lower=f_lower,
                          detectors=detectors,
                          target_length=target_length,
                          add_noise=add_noise,
                          whiten=whiten)
    
    with Pool(processes=num_workers) as pool:
        if show_progress:
            results = list(tqdm(
                pool.imap_unordered(worker_func, param_dicts, chunksize=100),
                total=len(param_dicts),
                desc="Generating waveforms"
            ))
        else:
            results = list(pool.imap_unordered(worker_func, param_dicts, chunksize=100))
    
    return results

def _chirp_mass(m1, m2):
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def _D_alpha(alpha, z):
    """
    Compute cosmological distance D_alpha for modified dispersion relations.

    For massive graviton (alpha=0), returns the effective distance
    appearing in the phase modification.

    Parameters
    ----------
    alpha : float
        Power-law index for modified dispersion relation.
        alpha=0 for massive graviton.
    z : float
        Source redshift.

    Returns
    -------
    float
        Cosmological distance in metres.
    """
    integrand_result = quad(
        lambda z_prime: (1 + z_prime)**(alpha - 2) /
            np.sqrt(_OMEGA_M * (1 + z_prime)**3 + _OMEGA_LAMBDA),
        0, z
    )
    return _C * (1 + z) / _H0 * integrand_result[0]

def _additional_phase(freqs, chirp_mass, z, lambda_g, f_c):
    """
    Compute massive graviton phase shift for a frequency array.

    Implements delta_Psi = -beta * u^{-1} + constant_terms where beta
    encodes the graviton Compton wavelength and cosmological distance.
    The constant terms ensure the phase shift vanishes at the cutoff
    frequency f_c. PN correction terms are NOT included -- they are
    already in IMRPhenomD/IMRPhenomXP.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz (must not contain zeros).
    chirp_mass : float
        Chirp mass in solar masses.
    z : float
        Source redshift.
    lambda_g : float
        Graviton Compton wavelength in metres.
    f_c : float
        Cutoff frequency in Hz (typically the maximum frequency).

    Returns
    -------
    np.ndarray
        Phase shift in radians, same shape as freqs.
    """
    M = chirp_mass * _M_SUN_SEC * (1 + z)   # chirp mass in seconds
    u = np.pi * M * freqs                    # dimensionless PN parameter
    beta = np.pi**2 * _C * _D_alpha(0, z) * M / (lambda_g**2 * (1 + z))
    constant_terms = (-1 * np.pi * _D_alpha(0, z) / ((1 + z) * lambda_g**2 * f_c**2)
                      + np.pi * _D_alpha(0, z) / (lambda_g**2 * (1 + z) * f_c))
    delta_psi = -beta * u**(-1) + constant_terms
    return delta_psi

# ---------------------------------------------------------------------------
#  Frequency-domain waveform generator with massive-graviton phase shift
# ---------------------------------------------------------------------------


def _generate_single_modified_waveform(params: Dict, time_resolution: float,
                                        approximant: str, f_lower: float,
                                        detectors: List[str], target_length: int,
                                        add_noise: bool, whiten: bool, lambda_g: float,
                                        f_final: float) -> Dict:
    """
    Worker function to generate a single modified (massive graviton) waveform.

    Generates a frequency-domain waveform, applies the massive graviton phase
    shift, IFFTs to time domain, projects onto detectors, and optionally adds noise.

    Parameters
    ----------
    params : dict
        Waveform parameters. Required: 'mass1', 'mass2'.
        Optional: 'redshift' (default 0.1), plus all standard sky/orientation params.
    time_resolution : float
        Time step delta_t in seconds.
    approximant : str
        FD-capable waveform approximant (e.g. 'IMRPhenomD').
    f_lower : float
        Lower frequency cutoff in Hz.
    detectors : list of str
        Detector names, e.g. ['H1', 'L1'].
    target_length : int
        Desired number of output samples.
    add_noise : bool
        Whether to add aLIGO noise.
    whiten : bool
        Whether to whiten the waveform.
    lambda_g : float
        Graviton Compton wavelength in metres.
    f_final : float
        Upper frequency cutoff for FD waveform generation.

    Returns
    -------
    dict
        Keys: 'success', 'detectors', 'params', optionally 'error'.
    """
    try:
        delta_f = 1.0 / 256  # Fine frequency resolution for accurate waveform

        hp_fd, hc_fd = get_fd_waveform(
            approximant=approximant,
            mass1=params['mass1'],
            mass2=params['mass2'],
            spin1z=params.get('spin1z', 0.0),
            spin2z=params.get('spin2z', 0.0),
            inclination=params.get('inclination', 0.0),
            coa_phase=params.get('coa_phase', 0.0),
            distance=params.get('distance', 410.0),
            delta_f=delta_f,
            f_lower=f_lower,
            f_final=f_final
        )

        m1 = params['mass1']
        m2 = params['mass2']
        chirp_mass = (m1 * m2)**(3.0 / 5.0) / (m1 + m2)**(1.0 / 5.0)
        z = params.get('redshift', 0.1)

        freqs = hp_fd.sample_frequencies.numpy()[1:]
        lg = params.get('lambda_g', lambda_g)  # per-sample overrides function-level
        hp_fd_amp = np.abs(hp_fd.numpy()[1:])
        f_c = float(np.max(freqs[np.nonzero(hp_fd_amp)]))  # max freq with non-zero amplitude
        phase_shift = _additional_phase(freqs, chirp_mass, z, lg, f_c)

        hp_array = hp_fd.numpy().copy()
        hc_array = hc_fd.numpy().copy()
        hp_array[1:] = hp_array[1:] * np.exp(1j * phase_shift)
        hc_array[1:] = hc_array[1:] * np.exp(1j * phase_shift)

        # IRFFT to time domain. The output is circular:
        #   [ringdown | zeros | inspiral → merger]
        # Merger peak is near the END; ringdown wraps to the BEGINNING.
        # We stitch end (inspiral+merger) with beginning (ringdown).
        hp_raw = np.fft.irfft(hp_array)
        hc_raw = np.fft.irfft(hc_array)
        N = len(hp_raw)
        hp_raw *= delta_f * N  # correct amplitude (IRFFT uses 1/N convention)
        hc_raw *= delta_f * N

        n_ringdown = 500  # ~0.12s of post-merger ringdown at 4096 Hz
        if target_length <= N:
            n_pre = target_length - n_ringdown
            hp_arr = np.concatenate([hp_raw[N - n_pre:], hp_raw[:n_ringdown]])
            hc_arr = np.concatenate([hc_raw[N - n_pre:], hc_raw[:n_ringdown]])
        else:
            hp_arr = np.concatenate([np.zeros(target_length - N), hp_raw])
            hc_arr = np.concatenate([np.zeros(target_length - N), hc_raw])

        hp_ts = TimeSeries(hp_arr.astype(np.float64), delta_t=time_resolution)
        hc_ts = TimeSeries(hc_arr.astype(np.float64), delta_t=time_resolution)

        gps_time = params.get('gps_time', 1126259462.4)
        hp_ts.start_time += gps_time
        hc_ts.start_time += gps_time

        ra = params.get('ra', 0.0)
        dec = params.get('dec', np.pi / 2)
        polarization = params.get('polarization', 0.0)

        detector_signals = {}
        for det_name in detectors:
            detector = Detector(det_name)
            signal = detector.project_wave(hp_ts, hc_ts, ra, dec, polarization, method='lal')

            # Fix signal to target_length using numpy (PyCBC doesn't support negative indexing)
            sig_array = np.array(signal)
            signal_len = len(sig_array)
            if signal_len > target_length:
                sig_array = sig_array[signal_len - target_length:]
            elif signal_len < target_length:
                sig_array = np.concatenate([np.zeros(target_length - signal_len, dtype=sig_array.dtype), sig_array])

            detector_signals[det_name] = TimeSeries(
                sig_array, delta_t=signal.delta_t, epoch=signal.start_time)

        if add_noise:
            for det_name in detectors:
                signal = detector_signals[det_name]
                delta_t = signal.delta_t
                duration = target_length * delta_t
                delta_f_noise = 1.0 / duration
                flen = target_length // 2 + 1
                psd = aLIGOZeroDetHighPower(flen, delta_f_noise, f_lower)
                noise = noise_from_psd(target_length, delta_t, psd)
                noise._epoch = signal._epoch
                detector_signals[det_name] = signal.inject(noise)

        # Whiten signals (if enabled) - should be done AFTER adding noise
        if whiten:
            for det_name in detectors:
                signal = detector_signals[det_name]
                # whiten_waveform returns (whitened_numpy_array, psd, freqs)
                whitened_data, _, _ = whiten_waveform(
                    signal, 
                    delta_t=time_resolution, 
                    f_lower=f_lower,
                    apply_bandpass=True,
                    apply_tukey=True,
                    tukey_side='left'  # Preserve merger at end
                )
                # Store numpy array directly (no need to convert back to TimeSeries)
                detector_signals[det_name] = whitened_data

        return {
            'success': True,
            'detectors': detector_signals,
            'params': params
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'params': params}


def _generate_modified_waveforms_parallel(param_dicts: List[Dict],
                                           time_resolution: float,
                                           approximant: str,
                                           f_lower: float,
                                           num_workers: int,
                                           show_progress: bool,
                                           detectors: List[str],
                                           target_length: int,
                                           add_noise: bool,
                                           whiten: bool,
                                           lambda_g: float,
                                           f_final: float) -> List[Dict]:
    """Generate modified waveforms in parallel using multiprocessing."""
    worker_func = partial(
        _generate_single_modified_waveform,
        time_resolution=time_resolution,
        approximant=approximant,
        f_lower=f_lower,
        detectors=detectors,
        target_length=target_length,
        add_noise=add_noise,
        whiten=whiten,
        lambda_g=lambda_g,
        f_final=f_final
    )

    with Pool(processes=num_workers) as pool:
        if show_progress:
            results = list(tqdm(
                pool.imap_unordered(worker_func, param_dicts, chunksize=100),
                total=len(param_dicts),
                desc="Generating modified waveforms"
            ))
        else:
            results = list(pool.imap_unordered(worker_func, param_dicts, chunksize=100))

    return results


def normalize_waveforms(signal_array: np.ndarray, method: str = 'global_standardize') -> Tuple[np.ndarray, List[float]]:
    """
    Normalize waveform data using various strategies.
    
    Parameters
    ----------
    signal_array : np.ndarray
        Shape: (num_samples, num_detectors, time_steps)
    method : str
        Normalization method:
        - 'per_sample_minmax': Per-sample min-max to [0, 100] (PROBLEMATIC - loses amplitude info)
        - 'global_standardize': Global z-score (mean=0, std=1) across all data (RECOMMENDED)
        - 'per_sample_standardize': Per-sample z-score (preserves relative structure)
        - 'global_minmax': Global min-max to [0, 100] (preserves relative amplitudes)
        - 'scale_constant': Divide by scale (preserves all relationships, simple)
        - 'none': No normalization
    
    Returns
    -------
    normalized_array : np.ndarray
        Normalized waveforms
    amplitude_stats : list
        Average amplitude per sample (for diagnostics)
    """
    num_samples = signal_array.shape[0]
    amplitude_stats = []
    
    if method == 'per_sample_minmax':
        print(f"  Normalizing: per-sample min-max to [0, 100]")
        print(f"  ⚠ WARNING: This removes global amplitude information!")
        for i in range(num_samples):
            sample_min = signal_array[i].min()
            sample_max = signal_array[i].max()
            
            if sample_max > sample_min:
                signal_array[i] = (signal_array[i] - sample_min) / (sample_max - sample_min) * 100
            else:
                signal_array[i] = 50.0
            
            amplitude_stats.append(signal_array[i].mean())
    
    elif method == 'global_standardize':
        print(f"  Normalizing: global z-score (mean=0, std=1)")
        print(f"  ✓ RECOMMENDED: Preserves amplitude variation between samples")
        global_mean = signal_array.mean()
        global_std = signal_array.std()
        
        if global_std > 0:
            signal_array = (signal_array - global_mean) / global_std
        
        for i in range(num_samples):
            amplitude_stats.append(signal_array[i].std())  # Track variation
        
        print(f"  Global stats: mean={global_mean:.2e}, std={global_std:.2e}")
    
    elif method == 'per_sample_standardize':
        print(f"  Normalizing: per-sample z-score (mean=0, std=1)")
        print(f"  Note: Preserves relative structure within each sample")
        for i in range(num_samples):
            sample_mean = signal_array[i].mean()
            sample_std = signal_array[i].std()
            
            if sample_std > 0:
                signal_array[i] = (signal_array[i] - sample_mean) / sample_std
            else:
                signal_array[i] = 0.0
            
            amplitude_stats.append(sample_std)  # Track original std
    
    elif method == 'global_minmax':
        print(f"  Normalizing: global min-max to [0, 100]")
        print(f"  ✓ Preserves relative amplitude differences between samples")
        global_min = signal_array.min()
        global_max = signal_array.max()
        
        if global_max > global_min:
            signal_array = (signal_array - global_min) / (global_max - global_min) * 100
        else:
            signal_array = 50.0
        
        for i in range(num_samples):
            amplitude_stats.append(signal_array[i].mean())
        
        print(f"  Global range: [{global_min:.2e}, {global_max:.2e}]")
    
    elif method == 'scale_constant':
        print(f"  Normalizing: constant scaling by 1e-21")
        print(f"  ✓ Preserves ALL relationships, simple and physically meaningful")
        
        scale_factor = 1e-21
        signal_array = signal_array / scale_factor * 10
        
        for i in range(num_samples):
            amplitude_stats.append(np.abs(signal_array[i]).max())  # Track peak values
        
        print(f"  Scale factor: {scale_factor:.2e}")
    
    elif method == 'none':
        print(f"  No waveform normalization applied")
        for i in range(num_samples):
            amplitude_stats.append(signal_array[i].mean())
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if amplitude_stats:
        print(f"  Amplitude stats: mean={np.mean(amplitude_stats):.2f} ± {np.std(amplitude_stats):.2f}")
    
    return signal_array, amplitude_stats


def normalize_parameters(param_array: np.ndarray, param_names: List[str], method: str = 'zscore') -> Tuple[np.ndarray, Dict]:
    """
    Normalize parameter data.
    
    Parameters
    ----------
    param_array : np.ndarray
        Shape: (num_samples, num_params)
    param_names : list of str
        Parameter names
    method : str
        Normalization method:
        - 'zscore': Z-score normalization (mean=0, std=1) - RECOMMENDED for flows
        - 'minmax': Min-max to [-1, 1]
        - 'none': No normalization
    
    Returns
    -------
    normalized_array : np.ndarray
        Normalized parameters
    norm_info : dict
        Normalization statistics for each parameter
    """
    num_params = param_array.shape[1]
    param_means = param_array.mean(axis=0)
    param_stds = param_array.std(axis=0)
    param_mins = param_array.min(axis=0)
    param_maxs = param_array.max(axis=0)
    
    # Store normalization info
    param_norm_info = {}
    for j, name in enumerate(param_names):
        param_norm_info[name] = {
            'mean': float(param_means[j]),
            'std': float(param_stds[j]),
            'min': float(param_mins[j]),
            'max': float(param_maxs[j]),
            'method': method
        }
    
    if method == 'zscore':
        print(f"  Normalizing parameters: z-score (mean=0, std=1)")
        for j in range(num_params):
            if param_stds[j] > 0:
                param_array[:, j] = (param_array[:, j] - param_means[j]) / param_stds[j]
            else:
                param_array[:, j] = 0
        
        print(f"  Parameters normalized:")
        for name, info in param_norm_info.items():
            print(f"    {name}: mean={info['mean']:.4f}, std={info['std']:.4f}")
    
    elif method == 'minmax':
        print(f"  Normalizing parameters: min-max to [-1, 1]")
        for j in range(num_params):
            if param_maxs[j] > param_mins[j]:
                # Scale to [-1, 1]
                param_array[:, j] = 2 * (param_array[:, j] - param_mins[j]) / (param_maxs[j] - param_mins[j]) - 1
            else:
                param_array[:, j] = 0
        
        print(f"  Parameters normalized:")
        for name, info in param_norm_info.items():
            print(f"    {name}: [{info['min']:.4f}, {info['max']:.4f}]")
    
    elif method == 'none':
        print(f"  No parameter normalization applied")
    
    else:
        raise ValueError(f"Unknown parameter normalization method: {method}")
    
    return param_array, param_norm_info

def pycbc_data_generator(config: Dict[str, Callable],
                        num_samples: int,
                        time_resolution: float = 1/4096,
                        approximant: str = 'IMRPhenomXP',
                        f_lower: float = 40.0,
                        num_workers: int = None,
                        signal_length: float = 2.0,
                        batch_size: int = 256,
                        chunk_size: int = 10000,
                        train_split: float = 0.8,
                        val_split: float = 0.1,
                        show_progress: bool = True,
                        detectors: List[str] = None,
                        add_noise: bool = True,
                        whiten: bool = False,
                        allow_padding: bool = True) -> Dict:
    """
    Generate PyCBC waveforms projected to detectors.
    Returns PyTorch DataLoaders for training, validation, and testing.
    
    Parameters
    ----------
    config : dict
        Dictionary mapping parameter names to numpy distribution functions.
        
        Required parameters:
        - 'mass1': Primary mass (solar masses)
        - 'mass2': Secondary mass (solar masses)
        
        Optional parameters:
        - 'distance': Luminosity distance in Mpc - default: 410.0 (GW150914)
        - 'spin1z', 'spin2z': Spin components - default: 0.0
        - 'inclination', 'coa_phase': Orientation angles - default: 0.0
        - 'ra': Right ascension (radians) - default: 0.0
        - 'dec': Declination (radians) - default: π/2 (north pole)
        - 'polarization': Polarization angle (radians) - default: 0.0
        - 'gps_time': GPS time of merger (seconds) - default: 1126259462.4 (GW150914)
          Affects detector antenna pattern and inter-detector time delays.
        - 'tc': Coalescence time - default: 0.0
        
    num_samples : int
        Total number of waveforms to generate
    time_resolution : float
        Time step (delta_t). Default: 1/4096
    approximant : str
        Waveform approximant. Default: 'IMRPhenomXP'
    f_lower : float
        Lower frequency cutoff (Hz). Default: 40.0
    num_workers : int
        Parallel processes. Default: min(cpu_count(), 8)
    batch_size : int
        DataLoader batch size. Default: 256
    chunk_size : int
        Process in chunks for memory. Default: 10000
    signal_length : float. Default: 2
        Length in seconds that each signal should be
    train_split : float
        Training fraction. Default: 0.8
    val_split : float
        Validation fraction. Default: 0.1
    detectors : list of str
        Detector names. Default: ['H1', 'L1']
    add_noise : bool
        Whether to add detector noise to signals. Default: True
    whiten : bool
        Whether to whiten signals using PSD-based whitening. Default: False
        Should be True when add_noise=True for realistic training.
        Whitening is applied AFTER noise injection.
    allow_padding : bool
        Backward-compatible argument retained for older call sites.
        Signal length normalization is always applied in this implementation.
    use_fd : bool
        If True, use the frequency-domain waveform generator
        (_generate_waveforms_parallel_fd) which supports massive-graviton
        phase modification via the 'lambda_g' config key.  Default: False
        (time-domain generator).

    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'
        
    Notes
    -----
    If ra, dec, or polarization are not provided in config, they default to:
    - ra = 0.0
    - dec = π/2 (north pole)
    - polarization = 0.0
    """
    # Validate inputs
    _validate_config(config)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < train_split < 1 or not 0 < val_split < 1:
        raise ValueError("train_split and val_split must be between 0 and 1")
    if train_split + val_split >= 1:
        raise ValueError("train_split + val_split must be < 1")

    if num_workers is None:
        num_workers = 1
    
    # Set default detectors
    if detectors is None:
        detectors = ['H1', 'L1']
    
    # Check which sky/time parameters are provided
    sky_params_provided = {
        'ra': 'ra' in config,
        'dec': 'dec' in config,
        'polarization': 'polarization' in config,
        'gps_time': 'gps_time' in config
    }
    
    if any(sky_params_provided.values()):
        print(f"Generating {num_samples} waveforms with projection to {detectors}")
        print(f"  Sky parameters: ra={'provided' if sky_params_provided['ra'] else 'default (0.0)'}, "
              f"dec={'provided' if sky_params_provided['dec'] else 'default (π/2)'}, "
              f"psi={'provided' if sky_params_provided['polarization'] else 'default (0.0)'}, "
              f"gps_time={'provided' if sky_params_provided['gps_time'] else 'default (1126259462.4)'}")
    else:
        print(f"Generating {num_samples} waveforms with projection to {detectors}")
        print(f"  Using default sky location: ra=0.0, dec=π/2 (north pole), psi=0.0")
        print(f"  Using default GPS time: 1126259462.4 (GW150914)")
    
    # Calculate target length from signal_length parameter
    target_length = int(signal_length / time_resolution)
    overall_start_time = perf_counter()
    print(f"Target signal length: {target_length} samples ({signal_length}s at {time_resolution}s resolution)")
    print(f"Noise injection: {'enabled' if add_noise else 'disabled'}")
    print(f"Whitening: {'enabled' if whiten else 'disabled'}")
    print(f"Waveform domain: Time (TD)")

    # Generate all parameters upfront
    param_dicts = _generate_parameter_sets(config, num_samples)

    # No FD generator or MG phase shift in TD mode

    # Process in chunks for memory efficiency
    all_successful = []
    all_failed = []
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_params = param_dicts[chunk_start:chunk_end]
        chunk_start_time = perf_counter()

        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")

        # Generate waveforms with detector projection at fixed length
        _parallel_func = _generate_waveforms_parallel
        chunk_results = _parallel_func(
            chunk_params, time_resolution, approximant, f_lower, num_workers, show_progress, detectors, target_length, add_noise, whiten
        )
        
        # Single pass: separate and accumulate
        for r in chunk_results:
            if r['success']:
                all_successful.append(r)
            else:
                all_failed.append(r)

        chunk_successes = sum(1 for r in chunk_results if r['success'])
        chunk_elapsed = perf_counter() - chunk_start_time
        total_elapsed = perf_counter() - overall_start_time

        print(f"  Chunk: {chunk_successes} successful")
        print(f"  Chunk generation time: {_format_elapsed_time(chunk_elapsed)}")
        print(f"  Total runtime so far: {_format_elapsed_time(total_elapsed)}")
    
    if not all_successful:
        # Print first few error messages to help diagnose
        print(f"\n❌ ALL {len(all_failed)} waveforms failed!")
        for i, fail in enumerate(all_failed[:5]):
            print(f"  Error #{i+1}: {fail.get('error', 'unknown')}")
        raise RuntimeError("No waveforms were successfully generated!")
    
    num_success = len(all_successful)
    num_failed = len(all_failed)
    print(f"\nGeneration complete: {num_success} successful, {num_failed} failed")
    print(f"Total waveform generation runtime: {_format_elapsed_time(perf_counter() - overall_start_time)}")
    
    # Extract everything in one pass with pre-allocated arrays
    print(f"\nProcessing {num_success} waveforms...")

    # Get parameter names and detector names
    param_names = list(all_successful[0]['params'].keys())
    num_params = len(param_names)
    detector_names = list(all_successful[0]['detectors'].keys())
    num_detectors = len(detector_names)

    print(f"  Detector channels: {detector_names}")
    print(f"  All signals fixed to: {target_length} samples ({signal_length}s)")

    # Pre-allocate arrays
    signal_array = np.empty((num_success, num_detectors, target_length), dtype=np.float32)
    param_array = np.empty((num_success, num_params), dtype=np.float32)

    print(f"  Extracting signals and parameters...")

    # Fill arrays - all signals already at target_length
    for i, waveform_data in enumerate(all_successful):
        # Extract parameters directly into array
        for j, param_name in enumerate(param_names):
            param_array[i, j] = waveform_data['params'][param_name]

        # Extract detector signals directly (already at correct length, already numpy arrays)
        for k, det_name in enumerate(detector_names):
            # TimeSeries objects support array protocol, direct assignment is efficient
            signal_array[i, k, :] = waveform_data['detectors'][det_name]

    print(f"  Converting to PyTorch tensors...")

    # Normalize parameters before converting to tensors
    print(f"  Normalizing parameters...")
    param_array, param_norm_info = normalize_parameters(param_array, param_names, method='zscore')

    # Convert to PyTorch efficiently using from_numpy (zero-copy view)
    X = torch.from_numpy(signal_array)  # (N, num_detectors, T)
    y = torch.from_numpy(param_array)    # (N, num_params)
    
    print(f"  Tensors: X={X.shape}, y={y.shape}")
    
    # Create dataset and split
    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print(f"\nReady! DataLoaders with batch_size={batch_size}")

    total_runtime = perf_counter() - overall_start_time
    print(f"Total generator runtime: {_format_elapsed_time(total_runtime)}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': {
            'parameter_names': param_names,
            'parameter_normalization': param_norm_info,
            'num_samples': num_success,
            'num_failed': num_failed,
            'waveform_shape': tuple(X.shape[1:]),
            'channels': detector_names,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'batch_size': batch_size,
            'time_resolution': time_resolution,
            'approximant': approximant,
            'f_lower': f_lower,
            'detectors': detector_names,
            'target_length': target_length,
            'signal_length': signal_length,
            'chunk_size': chunk_size,
            'sky_params_provided': sky_params_provided,
            'add_noise': add_noise,
            'whiten': whiten,
            'total_runtime_seconds': total_runtime,
            'preprocessing': {}
        }
    }


def pycbc_modified_data_generator(config: Dict[str, Callable],
                                   num_samples: int,
                                   lambda_g: float = None,
                                   time_resolution: float = 1/4096,
                                   approximant: str = 'IMRPhenomD',
                                   f_lower: float = 30.0,
                                   f_final: float = 2048.0,
                                   num_workers: int = None,
                                   signal_length: float = 2.0,
                                   batch_size: int = 256,
                                   chunk_size: int = 10000,
                                   train_split: float = 0.8,
                                   val_split: float = 0.1,
                                   show_progress: bool = True,
                                   detectors: List[str] = None,
                                   add_noise: bool = True,
                                   whiten: bool = True) -> Dict:
    """
    Generate modified (massive graviton) waveforms projected to detectors.
    Returns PyTorch DataLoaders for training, validation, and testing.

    Mirrors pycbc_data_generator but applies a massive graviton phase
    modification in the frequency domain before converting to time domain.

    Parameters
    ----------
    config : dict
        Dictionary mapping parameter names to numpy distribution functions.
        Required: 'mass1', 'mass2'. Optional: 'redshift' (default 0.1),
        plus all standard sky/orientation params.
    num_samples : int
        Total number of waveforms to generate.
    lambda_g : float, optional
        Graviton Compton wavelength in metres (dataset-level constant).
        If None, 'lambda_g' must be provided in config as a per-sample
        distribution. If both are given, the config (per-sample) takes
        precedence.
    time_resolution : float
        Time step delta_t. Default: 1/4096
    approximant : str
        FD waveform approximant. Default: 'IMRPhenomD'
    f_lower : float
        Lower frequency cutoff in Hz. Default: 30.0
    f_final : float
        Upper frequency cutoff in Hz. Default: 2048.0
    num_workers : int
        Parallel processes. Default: 1
    signal_length : float
        Duration in seconds. Default: 2.0
    batch_size : int
        DataLoader batch size. Default: 256
    chunk_size : int
        Chunk size for memory. Default: 10000
    train_split : float
        Training fraction. Default: 0.8
    val_split : float
        Validation fraction. Default: 0.1
    show_progress : bool
        Show tqdm progress bar. Default: True
    detectors : list of str
        Detector names. Default: ['H1', 'L1']
    add_noise : bool
        Whether to add detector noise. Default: True

    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'
    """
    _validate_config(config)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < train_split < 1 or not 0 < val_split < 1:
        raise ValueError("train_split and val_split must be between 0 and 1")
    if train_split + val_split >= 1:
        raise ValueError("train_split + val_split must be < 1")
    lambda_g_in_config = 'lambda_g' in config
    if lambda_g is None and not lambda_g_in_config:
        raise ValueError("lambda_g must be provided either as an argument or in config")
    if lambda_g is not None and lambda_g <= 0:
        raise ValueError("lambda_g must be positive")

    if num_workers is None:
        num_workers = 1

    if detectors is None:
        detectors = ['H1', 'L1']

    sky_params_provided = {
        'ra': 'ra' in config,
        'dec': 'dec' in config,
        'polarization': 'polarization' in config,
        'gps_time': 'gps_time' in config
    }

    target_length = int(signal_length / time_resolution)
    overall_start_time = perf_counter()
    if lambda_g_in_config:
        print(f"Generating {num_samples} MODIFIED waveforms (lambda_g=per-sample from config)")
    else:
        print(f"Generating {num_samples} MODIFIED waveforms (lambda_g={lambda_g:.2e} m)")
    print(f"  Approximant: {approximant} (frequency domain)")
    print(f"  Frequency range: {f_lower}-{f_final} Hz")
    print(f"  Detectors: {detectors}")
    print(f"  Target signal length: {target_length} samples ({signal_length}s at {time_resolution}s resolution)")
    print(f"  Noise injection: {'enabled' if add_noise else 'disabled'}")

    param_dicts = _generate_parameter_sets(config, num_samples)

    all_successful = []
    all_failed = []
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_params = param_dicts[chunk_start:chunk_end]
        chunk_start_time = perf_counter()

        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")

        chunk_results = _generate_modified_waveforms_parallel(
            chunk_params, time_resolution, approximant, f_lower,
            num_workers, show_progress, detectors, target_length,
            add_noise, whiten, lambda_g, f_final
        )

        for r in chunk_results:
            if r['success']:
                all_successful.append(r)
            else:
                all_failed.append(r)

        chunk_successes = sum(1 for r in chunk_results if r['success'])
        chunk_elapsed = perf_counter() - chunk_start_time
        total_elapsed = perf_counter() - overall_start_time

        print(f"  Chunk: {chunk_successes} successful")
        print(f"  Chunk generation time: {_format_elapsed_time(chunk_elapsed)}")
        print(f"  Total runtime so far: {_format_elapsed_time(total_elapsed)}")

    if not all_successful:
        raise RuntimeError("No waveforms were successfully generated!")

    num_success = len(all_successful)
    num_failed = len(all_failed)
    print(f"\nGeneration complete: {num_success} successful, {num_failed} failed")
    print(f"Total waveform generation runtime: {_format_elapsed_time(perf_counter() - overall_start_time)}")

    print(f"\nProcessing {num_success} waveforms...")

    param_names = list(all_successful[0]['params'].keys())
    num_params = len(param_names)
    detector_names = list(all_successful[0]['detectors'].keys())
    num_detectors = len(detector_names)

    print(f"  Detector channels: {detector_names}")
    print(f"  All signals fixed to: {target_length} samples ({signal_length}s)")

    signal_array = np.empty((num_success, num_detectors, target_length), dtype=np.float32)
    param_array = np.empty((num_success, num_params), dtype=np.float32)

    print(f"  Extracting signals and parameters...")

    for i, waveform_data in enumerate(all_successful):
        for j, param_name in enumerate(param_names):
            param_array[i, j] = waveform_data['params'][param_name]
        for k, det_name in enumerate(detector_names):
            signal_array[i, k, :] = waveform_data['detectors'][det_name]

    print(f"  Converting to PyTorch tensors...")

    print(f"  Normalizing parameters...")
    param_array, param_norm_info = normalize_parameters(param_array, param_names, method='zscore')
    
    X = torch.from_numpy(signal_array)
    y = torch.from_numpy(param_array)

    print(f"  Tensors: X={X.shape}, y={y.shape}")

    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"\nReady! DataLoaders with batch_size={batch_size}")

    total_runtime = perf_counter() - overall_start_time
    print(f"Total generator runtime: {_format_elapsed_time(total_runtime)}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': {
            'parameter_names': param_names,
            'parameter_normalization': param_norm_info,
            'num_samples': num_success,
            'num_failed': num_failed,
            'waveform_shape': tuple(X.shape[1:]),
            'channels': detector_names,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'batch_size': batch_size,
            'time_resolution': time_resolution,
            'approximant': approximant,
            'f_lower': f_lower,
            'f_final': f_final,
            'detectors': detector_names,
            'target_length': target_length,
            'signal_length': signal_length,
            'chunk_size': chunk_size,
            'sky_params_provided': sky_params_provided,
            'add_noise': add_noise,
            'lambda_g': lambda_g,
            'lambda_g_varied': lambda_g_in_config,
            'modified': True,
            'total_runtime_seconds': total_runtime,
            'preprocessing': {}
        }
    }

def _cos_taper(freqs, f_low, f_high):
    """
    Cosine (Hann-like) spectral taper from 1 at f_low to 0 at f_high.

    Returns 1 for f <= f_low, 0 for f >= f_high, and a smooth half-cosine
    transition in between.  Used to suppress Gibbs ringing that would otherwise
    result from abrupt step discontinuities in the frequency-domain phase
    modification before IRFFT.
    """
    result = np.ones_like(freqs, dtype=float)
    mask = (freqs > f_low) & (freqs < f_high)
    xi = (freqs[mask] - f_low) / (f_high - f_low)
    result[mask] = 0.5 * (1.0 + np.cos(np.pi * xi))
    result[freqs >= f_high] = 0.0
    return result

def _apply_end_taper(arr):
    """Cosine-taper the last _RINGDOWN_TAPER_LEN samples of arr to zero.

    The ringdown does not fully decay within the n_ringdown=500 sample window,
    so the highpass FIR filter sees a hard non-zero step at the end of the array
    and produces edge-effect ringing in the post-coalescence region.  Tapering
    the tail to zero before filtering eliminates that discontinuity.
    """
    arr = arr.copy()
    xi = np.linspace(0, 1, _RINGDOWN_TAPER_LEN)
    arr[-_RINGDOWN_TAPER_LEN:] *= 0.5 * (1.0 + np.cos(np.pi * xi))
    return arr



def _additional_phase_lv(freqs, chirp_mass, z, lambda_g, alpha_lv, A_lv, f_c,
                          f_pn_cutoff=None):
    """
    Compute the generalized Lorentz-violating (LV) phase shift.

    Implements the parametrized modified dispersion relation of Mirshekari,
    Yunes & Will (2011), arXiv:1110.2720:

        E² = p²c² + m_g²c⁴ + A_physical · p^α · c^α          [Eq. 1]

    where A_physical has units [energy]^{2−α}.  The paper re-packages this
    as the LV Compton wavelength (Eq. 13):

        A  ≡  A_physical^{1/(α−2)}       [always has units of metres]

    The total phase correction (Eq. 28, α ≠ 1, 2) is:

        δΨ = −β u^{−1} − ζ u^{α−1}

    where the massive graviton part (β term) is identical to _additional_phase
    and the Lorentz-violating ζ term is computed from A_lv via Eqs. 30, 32:

        ζ = π^{2−α}/(1−α) · c^{1−α} · D_α · M^{1−α}
                          / (A^{2−α} · (1+Z)^{1−α})     [α ≠ 1, 2]
        ζ = D_1 / A                                       [α = 1, Eq. 32]

    The ppE mapping (Eq. 34):  β_ppE = −ζ,  b_ppE = α − 1.

    Special cases:
        α = 0   : degenerate with massive graviton (A → λ_g)
        α = 1   : logarithmic correction (Eq. 31)
        α = 2   : degenerate with time of coalescence — no observable effect
        α = 2.5 : non-commutative geometry
        α = 3   : doubly special relativity (DSR)
        α = 4   : extra dimensions / Horava-Lifshitz gravity

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz (must not contain zeros).
    chirp_mass : float
        Chirp mass in solar masses.
    z : float
        Source redshift.
    lambda_g : float
        Graviton Compton wavelength in metres. Use np.inf to suppress the
        massive graviton term.
    alpha_lv : float
        LV dispersion exponent α in the modified dispersion relation.
    A_lv : float
        LV Compton wavelength in metres (A ≡ A_physical^{1/(α−2)}).
        Use np.inf to suppress the LV term (GR + massive-graviton limit).
    f_c : float
        Cutoff frequency in Hz (typically the maximum waveform frequency).
        The LV phase is normalised to zero at f_c. The massive graviton term
        is purely frequency-dependent and does not use f_c.
    f_pn_cutoff : float or None, optional
        Post-Newtonian breakdown frequency in Hz, from (m1+m2)·omega = 0.1.
        If given, the total phase is forced to zero above this frequency — the
        PN dispersion formula is not valid beyond the inspiral regime.
        Default None (no cutoff applied, backward-compatible).

    Returns
    -------
    np.ndarray
        Total phase shift in radians, same shape as freqs.
    """
    M   = chirp_mass * _M_SUN_SEC * (1 + z)   # detector-frame chirp mass [s]
    u   = np.pi * M * freqs                    # dimensionless PN frequency
    u_c = np.pi * M * f_c

    # ── Massive graviton term (same as _additional_phase) ─────────────────────
    if np.isfinite(lambda_g) and lambda_g > 0:
        beta_g = np.pi**2 * _C * _D_alpha(0, z) * M / (lambda_g**2 * (1 + z))
        delta_psi_mg = -beta_g * u**(-1)
    else:
        delta_psi_mg = np.zeros_like(freqs)

    # ── Lorentz-violating term (Mirshekari et al. 2011, Eqs. 28–32) ──────────
    if not np.isfinite(A_lv) or A_lv <= 0 or alpha_lv == 2.0:
        # A_lv = inf  → GR / massive-graviton-only limit
        # alpha_lv = 2 → degenerate with time of coalescence
        delta_psi_lv = np.zeros_like(freqs)
    elif alpha_lv == 1.0:
        # Eq. 32:  ζ = D_1 / A  (both in metres → dimensionless)
        # Eq. 31:  δΨ_LV = +ζ · (ln u − ln u_c)
        zeta = _D_alpha(1, z) / A_lv
        delta_psi_lv = zeta * (np.log(u) - np.log(u_c))
    else:
        # Eq. 30 (SI):  ζ = π^{2−α}/(1−α) · c^{1−α} · D_α · M^{1−α}
        #                             / (A^{2−α} · (1+Z)^{1−α})
        zeta = (
            np.pi**(2 - alpha_lv) / (1 - alpha_lv)
            * _C**(1 - alpha_lv)
            * _D_alpha(alpha_lv, z)
            * M**(1 - alpha_lv)
            / (A_lv**(2 - alpha_lv) * (1 + z)**(1 - alpha_lv))
        )
        # Eq. 28:  δΨ_LV = −ζ · u^{α−1},  normalised to zero at u_c
        delta_psi_lv = -zeta * (u**(alpha_lv - 1) - u_c**(alpha_lv - 1))

    total_phase = delta_psi_mg + delta_psi_lv
    if f_pn_cutoff is not None:
        taper = _cos_taper(freqs, f_pn_cutoff * (1.0 - _TAPER_FRACTION), f_pn_cutoff)
        total_phase = total_phase * taper
    return total_phase

def _generate_single_lv_waveform(params: dict, time_resolution: float,
                                  approximant: str, f_lower: float,
                                  detectors: list, target_length: int,
                                  add_noise: bool, lambda_g: float,
                                  alpha_lv: float, A_lv: float,
                                  f_final: float,
                                  highpass_fc: float = _HIGHPASS_FC) -> dict:
    """
    Worker function to generate a single Lorentz-violating (LV) GW waveform.

    Identical pipeline to _generate_single_modified_waveform but applies the
    generalized LV phase from Mirshekari, Yunes & Will (2011), arXiv:1110.2720,
    instead of the pure massive graviton phase from Will (1997), arXiv:9709011.

    Parameters
    ----------
    params : dict
        Waveform parameters. Required: 'mass1', 'mass2'.
        Optional: 'redshift' (default 0.1), plus sky/orientation params.
    time_resolution : float
        Time step delta_t in seconds.
    approximant : str
        FD-capable waveform approximant (e.g. 'IMRPhenomD').
    f_lower : float
        Lower frequency cutoff in Hz.
    detectors : list of str
        Detector names, e.g. ['H1', 'L1'].
    target_length : int
        Desired number of output samples.
    add_noise : bool
        Whether to add aLIGO noise.
    lambda_g : float
        Graviton Compton wavelength in metres. Use np.inf for GR (no mass term).
    alpha_lv : float
        LV dispersion exponent (α_LV). Key values: 3 (DSR), 4 (Horava-Lifshitz).
    A_lv : float
        LV Compton wavelength in metres (A ≡ A_physical^{1/(α−2)}).
        Use np.inf to suppress the LV term (pure massive-graviton or GR).
    f_final : float
        Upper frequency cutoff in Hz.

    Returns
    -------
    dict
        Keys: 'success', 'detectors', 'params', optionally 'error'.
    """
    try:
        delta_f = 1.0 / 256

        hp_fd, hc_fd = get_fd_waveform(
            approximant=approximant,
            mass1=params['mass1'],
            mass2=params['mass2'],
            spin1z=params.get('spin1z', 0.0),
            spin2z=params.get('spin2z', 0.0),
            inclination=params.get('inclination', 0.0),
            coa_phase=params.get('coa_phase', 0.0),
            distance=params.get('distance', 410.0),
            delta_f=delta_f,
            f_lower=f_lower,
            f_final=f_final
        )

        m1 = params['mass1']
        m2 = params['mass2']
        chirp_mass = (m1 * m2)**(3.0 / 5.0) / (m1 + m2)**(1.0 / 5.0)
        z = params.get('redshift', 0.1)

        freqs = hp_fd.sample_frequencies.numpy()[1:]
        hp_fd_amp = np.abs(hp_fd.numpy()[1:])
        f_c = float(np.max(freqs[np.nonzero(hp_fd_amp)]))

        # Per-sample overrides for lambda_g / A_lv
        lg    = params.get('lambda_g', lambda_g)
        a_lv  = params.get('A_lv', A_lv)

        # PN breakdown frequency: phase modification is only valid in the inspiral
        f_pn = _M_OMEGA_PN / (np.pi * (m1 + m2) * _M_SUN_SEC)

        phase_shift = _additional_phase_lv(freqs, chirp_mass, z, lg,
                                            alpha_lv, a_lv, f_c,
                                            f_pn_cutoff=f_pn)

        hp_array = hp_fd.numpy().copy()
        hc_array = hc_fd.numpy().copy()
        hp_array[1:] = hp_array[1:] * np.exp(1j * phase_shift)
        hc_array[1:] = hc_array[1:] * np.exp(1j * phase_shift)

        hp_raw = np.fft.irfft(hp_array)
        hc_raw = np.fft.irfft(hc_array)
        N = len(hp_raw)
        hp_raw *= delta_f * N
        hc_raw *= delta_f * N

        n_ringdown = 500
        if target_length <= N:
            n_pre = target_length - n_ringdown
            hp_arr = np.concatenate([hp_raw[N - n_pre:], hp_raw[:n_ringdown]])
            hc_arr = np.concatenate([hc_raw[N - n_pre:], hc_raw[:n_ringdown]])
        else:
            hp_arr = np.concatenate([np.zeros(target_length - N), hp_raw])
            hc_arr = np.concatenate([np.zeros(target_length - N), hc_raw])

        hp_arr = _apply_end_taper(hp_arr)
        hc_arr = _apply_end_taper(hc_arr)
        hp_ts = TimeSeries(hp_arr.astype(np.float64), delta_t=time_resolution)
        hc_ts = TimeSeries(hc_arr.astype(np.float64), delta_t=time_resolution)
        hp_ts = highpass_fir(hp_ts, highpass_fc, 128)
        hc_ts = highpass_fir(hc_ts, highpass_fc, 128)

        gps_time = params.get('gps_time', 1126259462.4)
        hp_ts.start_time += gps_time
        hc_ts.start_time += gps_time

        ra = params.get('ra', 0.0)
        dec = params.get('dec', np.pi / 2)
        polarization = params.get('polarization', 0.0)

        detector_signals = {}
        for det_name in detectors:
            detector = Detector(det_name)
            signal = detector.project_wave(hp_ts, hc_ts, ra, dec, polarization, method='lal')

            sig_array = signal.numpy()
            signal_len = len(sig_array)
            if signal_len > target_length:
                sig_array = sig_array[signal_len - target_length:]
            elif signal_len < target_length:
                sig_array = np.concatenate([
                    np.zeros(target_length - signal_len, dtype=sig_array.dtype),
                    sig_array
                ])

            detector_signals[det_name] = TimeSeries(
                sig_array, delta_t=signal.delta_t, epoch=signal.start_time)

        if add_noise:
            for det_name in detectors:
                signal = detector_signals[det_name]
                delta_t = signal.delta_t
                duration = target_length * delta_t
                delta_f_noise = 1.0 / duration
                flen = target_length // 2 + 1
                psd = load_random_o4_psd(flen, delta_f_noise, f_lower, det_name, int(round(1.0 / delta_t)))
                noise = noise_from_psd(target_length, delta_t, psd)
                noise._epoch = signal._epoch
                detector_signals[det_name] = signal.inject(noise)

        return {
            'success': True,
            'detectors': detector_signals,
            'params': params
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'params': params}
    

def _generate_lv_waveforms_parallel(param_dicts: List[Dict],
                                     time_resolution: float,
                                     approximant: str,
                                     f_lower: float,
                                     num_workers: int,
                                     show_progress: bool,
                                     detectors: List[str],
                                     target_length: int,
                                     add_noise: bool,
                                     lambda_g: float,
                                     alpha_lv: float,
                                     A_lv: float,
                                     f_final: float,
                                     highpass_fc: float = _HIGHPASS_FC) -> List[Dict]:
    """Generate Lorentz-violating waveforms in parallel using multiprocessing."""
    worker_func = partial(
        _generate_single_lv_waveform,
        time_resolution=time_resolution,
        approximant=approximant,
        f_lower=f_lower,
        detectors=detectors,
        target_length=target_length,
        add_noise=add_noise,
        lambda_g=lambda_g,
        alpha_lv=alpha_lv,
        A_lv=A_lv,
        f_final=f_final,
        highpass_fc=highpass_fc,
    )

    if num_workers == 1:
        iterable = map(worker_func, param_dicts)
        if show_progress:
            results = list(tqdm(iterable, total=len(param_dicts), desc="Generating LV waveforms"))
        else:
            results = list(iterable)
    else:
        with Pool(processes=num_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap_unordered(worker_func, param_dicts, chunksize=100),
                    total=len(param_dicts),
                    desc="Generating LV waveforms"
                ))
            else:
                results = list(pool.imap_unordered(worker_func, param_dicts, chunksize=100))

    return results


def pycbc_lorentz_violation_data_generator(config: Dict[str, Callable],
                                            num_samples: int,
                                            alpha_lv: float,
                                            A_lv: float = None,
                                            lambda_g: float = None,
                                            time_resolution: float = 1/4096,
                                            approximant: str = 'IMRPhenomD',
                                            f_lower: float = 30.0,
                                            f_final: float = 2048.0,
                                            highpass_fc: float = _HIGHPASS_FC,
                                            num_workers: int = None,
                                            signal_length: float = 2.0,
                                            batch_size: int = 256,
                                            chunk_size: int = 10000,
                                            train_split: float = 0.8,
                                            val_split: float = 0.1,
                                            show_progress: bool = True,
                                            detectors: List[str] = None,
                                            add_noise: bool = True) -> Dict:
    """
    Generate Lorentz-violating (LV) waveforms projected to detectors.
    Returns PyTorch DataLoaders for training, validation, and testing.

    Mirrors pycbc_massive_gravity_data_generator but applies the generalised
    LV phase from Mirshekari, Yunes & Will (2011), arXiv:1110.2720, which
    combines a massive graviton term (lambda_g) with a power-law LV term
    (alpha_lv, A_lv).

    Parameters
    ----------
    config : dict
        Dictionary mapping parameter names to numpy distribution functions.
        Required: 'mass1', 'mass2'. Optional: 'redshift' (default 0.1),
        'lambda_g' (per-sample override), 'A_lv' (per-sample override),
        plus all standard sky/orientation params.
    num_samples : int
        Total number of waveforms to generate.
    alpha_lv : float
        LV dispersion exponent. Key values: 2.5 (non-commutative geometry),
        3.0 (doubly special relativity), 4.0 (extra dimensions / Horava-Lifshitz).
    A_lv : float, optional
        LV Compton wavelength in metres (dataset-level constant).
        If None, 'A_lv' must be provided in config as a per-sample distribution.
        If both are given, the config (per-sample) takes precedence.
        Set to np.inf to suppress the LV term (pure massive-graviton limit).
    lambda_g : float, optional
        Graviton Compton wavelength in metres. Use np.inf to suppress the mass
        term (pure LV limit). If None, 'lambda_g' must be in config or
        lambda_g defaults to np.inf.
    time_resolution : float
        Time step delta_t. Default: 1/4096
    approximant : str
        FD waveform approximant. Default: 'IMRPhenomD'
    f_lower : float
        Lower frequency cutoff in Hz. Default: 30.0
    f_final : float
        Upper frequency cutoff in Hz. Default: 2048.0
    num_workers : int
        Parallel processes. Default: 1
    signal_length : float
        Duration in seconds. Default: 2.0
    batch_size : int
        DataLoader batch size. Default: 256
    chunk_size : int
        Chunk size for memory. Default: 10000
    train_split : float
        Training fraction. Default: 0.8
    val_split : float
        Validation fraction. Default: 0.1
    show_progress : bool
        Show tqdm progress bar. Default: True
    detectors : list of str
        Detector names. Default: ['H1', 'L1']
    add_noise : bool
        Whether to add detector noise. Default: True

    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'
    """
    _validate_config(config)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < train_split < 1 or not 0 < val_split < 1:
        raise ValueError("train_split and val_split must be between 0 and 1")
    if train_split + val_split >= 1:
        raise ValueError("train_split + val_split must be < 1")

    A_lv_in_config = 'A_lv' in config
    if A_lv is None and not A_lv_in_config:
        raise ValueError("A_lv must be provided either as an argument or in config")
    if A_lv is not None and A_lv <= 0:
        raise ValueError("A_lv must be positive")

    lambda_g_in_config = 'lambda_g' in config
    # lambda_g defaults to np.inf (suppress mass term) if not given
    if lambda_g is None and not lambda_g_in_config:
        lambda_g = np.inf
    if lambda_g is not None and lambda_g <= 0:
        raise ValueError("lambda_g must be positive")

    if num_workers is None:
        num_workers = 1

    if detectors is None:
        detectors = ['H1', 'L1']

    sky_params_provided = {
        'ra': 'ra' in config,
        'dec': 'dec' in config,
        'polarization': 'polarization' in config,
        'gps_time': 'gps_time' in config
    }

    target_length = int(signal_length / time_resolution)
    print(f"Generating {num_samples} LORENTZ-VIOLATING waveforms (alpha_lv={alpha_lv})")
    if A_lv_in_config:
        print(f"  A_lv=per-sample from config")
    else:
        print(f"  A_lv={A_lv:.2e} m")
    if lambda_g_in_config:
        print(f"  lambda_g=per-sample from config")
    elif np.isinf(lambda_g):
        print(f"  lambda_g=inf (mass term suppressed)")
    else:
        print(f"  lambda_g={lambda_g:.2e} m")
    print(f"  Approximant: {approximant} (frequency domain)")
    print(f"  Frequency range: {f_lower}-{f_final} Hz")
    print(f"  Detectors: {detectors}")
    print(f"  Target signal length: {target_length} samples ({signal_length}s at {time_resolution}s resolution)")
    print(f"  Noise injection: {'enabled' if add_noise else 'disabled'}")

    if add_noise:
        _sample_rate = int(round(1.0 / time_resolution))
        for _det in detectors:
            build_o4_psd_cache(_det, n_segments=_DEFAULT_N_SEGS, sample_rate=_sample_rate, cache_dir=_DEFAULT_CACHE)

    param_dicts = _generate_parameter_sets(config, num_samples)

    all_successful = []
    all_failed = []
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_params = param_dicts[chunk_start:chunk_end]

        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")

        chunk_results = _generate_lv_waveforms_parallel(
            chunk_params, time_resolution, approximant, f_lower,
            num_workers, show_progress, detectors, target_length,
            add_noise, lambda_g, alpha_lv, A_lv, f_final, highpass_fc
        )

        for r in chunk_results:
            if r['success']:
                all_successful.append(r)
            else:
                all_failed.append(r)

        if num_chunks > 1:
            print(f"  Chunk: {len([r for r in chunk_results if r['success']])} successful")

    if not all_successful:
        raise RuntimeError("No waveforms were successfully generated!")

    num_success = len(all_successful)
    num_failed = len(all_failed)
    print(f"\nGeneration complete: {num_success} successful, {num_failed} failed")

    print(f"\nProcessing {num_success} waveforms...")

    param_names = list(all_successful[0]['params'].keys())
    num_params = len(param_names)
    detector_names = list(all_successful[0]['detectors'].keys())
    num_detectors = len(detector_names)

    print(f"  Detector channels: {detector_names}")
    print(f"  All signals fixed to: {target_length} samples ({signal_length}s)")

    signal_array = np.empty((num_success, num_detectors, target_length), dtype=np.float32)
    param_array = np.empty((num_success, num_params), dtype=np.float32)

    print(f"  Extracting signals and parameters...")

    for i, waveform_data in enumerate(all_successful):
        for j, param_name in enumerate(param_names):
            param_array[i, j] = waveform_data['params'][param_name]
        for k, det_name in enumerate(detector_names):
            signal_array[i, k, :] = waveform_data['detectors'][det_name]

    print(f"  Converting to PyTorch tensors...")

    X = torch.from_numpy(signal_array)
    y = torch.from_numpy(param_array)

    print(f"  Tensors: X={X.shape}, y={y.shape}")

    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"\nReady! DataLoaders with batch_size={batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': {
            'parameter_names': param_names,
            'num_samples': num_success,
            'num_failed': num_failed,
            'waveform_shape': tuple(X.shape[1:]),
            'channels': detector_names,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'batch_size': batch_size,
            'time_resolution': time_resolution,
            'approximant': approximant,
            'f_lower': f_lower,
            'f_final': f_final,
            'highpass_fc': highpass_fc,
            'detectors': detector_names,
            'target_length': target_length,
            'signal_length': signal_length,
            'chunk_size': chunk_size,
            'sky_params_provided': sky_params_provided,
            'add_noise': add_noise,
            'alpha_lv': alpha_lv,
            'A_lv': A_lv,
            'A_lv_varied': A_lv_in_config,
            'lambda_g': lambda_g,
            'lambda_g_varied': lambda_g_in_config,
            'waveform_type': 'lorentz_violation',
            'preprocessing': {}
        }
    }


def whiten_waveform(waveform, delta_t=1/4096, f_lower=20.0, apply_bandpass=True,
                    apply_tukey=True, tukey_alpha=0.1, tukey_side='left'):
    """
    Whiten a single waveform using PSD-based whitening.

    Follows the PyCBC GW150914 tutorial approach:
    https://pycbc.org/pycbc/latest/html/gw150914.html

    The whitening process:
    1. Optionally apply Tukey window to prevent edge effects
    2. Compute PSD using Welch's method
    3. Interpolate PSD to smooth frequency grid
    4. Divide frequency-domain data by sqrt(PSD)
    5. Convert back to time domain
    6. Optionally apply Butterworth bandpass filter (35-350 Hz, order 8)

    Parameters
    ----------
    waveform : np.ndarray or TimeSeries
        Single waveform to whiten (1D array)
    delta_t : float, optional
        Time resolution in seconds (default: 1/4096)
    f_lower : float, optional
        Lower frequency cutoff in Hz (default: 20.0)
    apply_bandpass : bool, optional
        Apply Butterworth bandpass filter 35-350 Hz, order 8 (default: True)
        Matches real GW processing in Untitled-1.py
    apply_tukey : bool, optional
        Apply Tukey window before whitening to prevent edge effects (default: True)
        Smoothly tapers signal edges to zero, eliminating filter ringing
    tukey_alpha : float, optional
        Tukey window alpha parameter - fraction of signal to taper (default: 0.1)
        0.1 means 5% tapered on each edge, 90% flat in middle
    tukey_side : str, optional
        Which side(s) to apply the Tukey taper (default: 'left')
        - 'left': Only taper the beginning (preserves merger at end)
        - 'right': Only taper the end
        - 'both': Taper both sides (standard Tukey window)

    Returns
    -------
    whitened : np.ndarray
        Whitened waveform as numpy array (same length as input)
    psd : np.ndarray
        Power spectral density used for whitening
    freqs : np.ndarray
        Frequency array for PSD

    Notes
    -----
    Edge effects (amplitude spikes at start/end) can occur with IIR filters like
    the Butterworth bandpass. The Tukey window prevents these by smoothly tapering
    the signal edges to zero before processing, eliminating discontinuities that
    cause filter transients.

    For GW signals where the merger is at the end of the waveform, use
    tukey_side='left' to only taper the beginning and preserve the merger.

    The output length always matches the input length.

    Examples
    --------
    >>> waveform = np.random.randn(8192)
    >>> whitened, psd, freqs = whiten_waveform(waveform, delta_t=1/4096)
    >>> print(f"Input: {waveform.shape}, Output: {whitened.shape}")
    Input: (8192,), Output: (8192,)

    >>> # Taper only the beginning (preserve merger at end)
    >>> whitened, psd, freqs = whiten_waveform(waveform, tukey_side='left')

    >>> # Disable Tukey window (will have edge effects)
    >>> whitened, psd, freqs = whiten_waveform(waveform, apply_tukey=False)
    """
    from scipy.signal.windows import tukey
    # Convert to numpy array if TimeSeries
    if isinstance(waveform, TimeSeries):
        data = np.array(waveform)
    else:
        data = np.array(waveform)
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")

    # Convert to TimeSeries
    strain = TimeSeries(data, delta_t=delta_t)

    # Compute PSD
    n_samples = len(strain)
    if n_samples >= 8192:
        # Long signals: use Welch method with 4096-sample segments
        seg_len = 4096
        seg_stride = 2048
        psd_welch = welch(strain, seg_len=seg_len, seg_stride=seg_stride)
        psd = interpolate(psd_welch, 1.0 / strain.duration)
    else:
        # Short signals (<2s at 4096Hz): Welch gives noisy PSD with too few
        # segments, causing whitening artifacts/ringing. Use analytical aLIGO
        # PSD instead, which is smooth and well-characterised.
        delta_f = 1.0 / strain.duration
        flen = n_samples // 2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

    # Convert strain to frequency domain
    freq_series = strain.to_frequencyseries()

    # Resize PSD to match frequency series length
    psd.resize(len(freq_series))

    # Add small epsilon to PSD to avoid division by zero
    # This handles cases where PSD might be zero or very small
    psd_safe = psd.copy()
    psd_array = np.array(psd_safe)
    epsilon = 1e-40  # Very small value to prevent division by zero
    psd_array[psd_array <= 0] = epsilon
    psd_safe = FrequencySeries(psd_array, delta_f=psd.delta_f, epoch=psd.epoch)

    # Whiten: divide by sqrt(PSD) in frequency domain
    white_strain = (freq_series / (psd_safe ** 0.5)).to_timeseries()

    # Apply Tukey window BEFORE bandpass filtering to prevent edge effects
    # The window tapers the whitened signal to zero at edges, preventing
    # the FIR filter from ringing at discontinuities
    if apply_tukey and apply_bandpass:
        n = len(white_strain)
        if tukey_side == 'both':
            # Standard symmetric Tukey window
            window = tukey(n, alpha=tukey_alpha)
        elif tukey_side == 'left':
            # Only taper the beginning - create half Tukey window
            # Use a full Tukey window but only take the left taper + flat portion
            full_window = tukey(n, alpha=tukey_alpha * 2)  # Double alpha since we only use half
            window = np.ones(n)
            taper_len = int(n * tukey_alpha)
            window[:taper_len] = full_window[:taper_len]
        elif tukey_side == 'right':
            # Only taper the end - create half Tukey window
            full_window = tukey(n, alpha=tukey_alpha * 2)
            window = np.ones(n)
            taper_len = int(n * tukey_alpha)
            window[-taper_len:] = full_window[-taper_len:]
        else:
            raise ValueError(f"tukey_side must be 'left', 'right', or 'both', got '{tukey_side}'")

        white_strain = TimeSeries(np.array(white_strain) * window, delta_t=delta_t)

    # Apply optional bandpass filtering using Butterworth (order 8, 35-350 Hz)
    # Matches processing in real GW analysis (Untitled-1.py)
    if apply_bandpass:
        from scipy.signal import butter, sosfiltfilt
        whitened_array = np.array(white_strain)
        sample_rate = 1.0 / delta_t
        f_low = 35.0  # Hz (match Untitled-1.py)
        f_high = 350.0  # Hz (match Untitled-1.py)
        sos = butter(8, [f_low, f_high], btype='bandpass', fs=sample_rate, output='sos')
        whitened = sosfiltfilt(sos, whitened_array)
    else:
        whitened = np.array(white_strain)

    # Prepare outputs
    psd_array = np.array(psd)
    freqs = np.arange(len(psd)) * psd.delta_f

    return whitened, psd_array, freqs



if __name__ == "__main__":
    # Example 1: Basic usage with default sky location (north pole, zero polarization)
    print("=" * 80)
    print("Example 1: Basic usage with default sky location")
    print("=" * 80)
    
    config = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
    }

    # Generate with H1 and L1 projection (default detectors)
    result = pycbc_data_generator(
        config, 
        num_samples=1000, 
        batch_size=16, 
        num_workers=2,
        allow_padding=True
    )

    # Access the loaders
    train_loader = result['train_loader']
    val_loader = result['val_loader']
    test_loader = result['test_loader']

    # Print metadata
    print(f"\nMetadata:")
    for key, value in result['metadata'].items():
        print(f"  {key}: {value}")

    # Show first batch
    print(f"\nCollecting all training data into one array:")
    all_combined = []
    for waveforms, params in train_loader:
        print(f"  Waveforms shape: {waveforms.shape}")
        print(f"  Parameters shape: {params.shape}")
        print(f"  H1 channel (first 5): {waveforms[0, 0, :5]}")
        print(f"  L1 channel (first 5): {waveforms[0, 1, :5]}")
        
        # Convert to torch tensors and concatenate
        train_params = torch.FloatTensor(params)
        train_data = torch.FloatTensor(waveforms)
        print(train_data.shape)
        
        combined = torch.cat([train_params, train_data[:, 0, :]], dim=1)    
        all_combined.append(combined)
    

    data_test = []
    params_test = []
    for waveforms, params in test_loader:
        test_params = torch.FloatTensor(params)
        test_data = torch.FloatTensor(waveforms)
        params_test.append(test_params)
        data_test.append(test_data)
    all_test_data = torch.cat(data_test, dim=0)
    all_test_params = torch.cat(params_test, dim=0)
    print(f"\nTest data shape: {len(all_test_data[:, 0])}")

    # Concatenate all batches into one big array
    all_data = torch.cat(all_combined, dim=0)
    print(f"  Total combined shape: {all_data.shape[1]}")
    #print(all_data)

'''
    # Example 2: Including sky location parameters
    print("\n" + "=" * 80)
    print("Example 2: Including sky location parameters (random positions)")
    print("=" * 80)
    
    config_with_sky = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
        'ra': lambda size: np.random.uniform(0, 2*np.pi, size=size),
        'dec': lambda size: np.random.uniform(-np.pi/2, np.pi/2, size=size),
        'polarization': lambda size: np.random.uniform(0, 2*np.pi, size=size),
    }

    result_sky = pycbc_data_generator(
        config_with_sky, 
        num_samples=100, 
        batch_size=16,
        num_workers=2,
        allow_padding=True
    )

    print(f"\nWith random sky locations:")
    print(f"  Sky params provided: {result_sky['metadata']['sky_params_provided']}")
    for waveforms, params in result_sky['train_loader']:
        print(f"  Waveforms shape: {waveforms.shape}")
        print(f"  Parameters include: {result_sky['metadata']['parameter_names']}")
        break

    # Example 3: Using three detectors
    print("\n" + "=" * 80)
    print("Example 3: Using three detectors (H1, L1, V1)")
    print("=" * 80)
    
    config_three = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
        'ra': lambda size: np.random.uniform(0, 2*np.pi, size=size),
        'dec': lambda size: np.random.uniform(-np.pi/2, np.pi/2, size=size),
        'polarization': lambda size: np.random.uniform(0, 2*np.pi, size=size),
    }

    result_three = pycbc_data_generator(
        config_three, 
        num_samples=50, 
        batch_size=16,
        detectors=['H1', 'L1', 'V1'],
        num_workers=2,
        allow_padding=True
    )

    print(f"\nWith three detectors:")
    for waveforms, params in result_three['train_loader']:
        print(f"  Waveforms shape: {waveforms.shape}")
        print(f"  Channels: {result_three['metadata']['channels']}")
        print(f"  H1 (first 5): {waveforms[0, 0, :5]}")
        print(f"  L1 (first 5): {waveforms[0, 1, :5]}")
        print(f"  V1 (first 5): {waveforms[0, 2, :5]}")
        break
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
'''