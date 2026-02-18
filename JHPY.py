"""
Welcome to John-Hamza py.
This is a library containing the neural networks we are using for our project.
A lot of this is written by Claude.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import itertools
import random
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from typing import Dict, List, Callable
from functools import partial
from multiprocessing import Pool
from pycbc.waveform import get_td_waveform, get_fd_waveform
from scipy.integrate import quad
from pycbc.detector import Detector
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.types import TimeSeries
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.psd import welch, interpolate
from pycbc.filter import highpass_fir, lowpass_fir, resample_to_delta_t
from torch.utils.data import Subset
import warnings
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Real Data'))



################### Cosmological / Modified Gravity Constants ###################
_C = 2.998e8                          # speed of light, m/s
_G = 6.674e-11                        # gravitational constant, m^3 kg^-1 s^-2
_M_SUN = 1.989e30                     # solar mass, kg
_MPC = 3.086e22                       # megaparsec, m
_M_SUN_SEC = _G * _M_SUN / _C**3     # solar mass in seconds (~4.926e-6 s)
_H0 = 67.4e3 / _MPC                  # Hubble constant in 1/s
_OMEGA_M = 0.315
_OMEGA_LAMBDA = 0.685


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


################### Miscellaneous functions ###################
def normalize_waveform(waveform, scale_factor=1e21):
    """
    Normalize a single waveform by multiplying by a fixed scale factor.

    This scales very small gravitational wave strain values (typically ~10^-21)
    to order 1-10 range for easier processing and visualization.

    Parameters
    ----------
    waveform : np.ndarray or TimeSeries
        Single waveform to normalize (1D array)
    scale_factor : float, optional
        Fixed scaling factor to multiply the waveform by.
        Default: 1e21 (appropriate for typical GW strains of ~1e-21)

    Returns
    -------
    normalized : np.ndarray
        Scaled waveform as numpy array

    Examples
    --------
    >>> waveform = np.random.randn(1000) * 1e-21  # Typical GW strain
    >>> normalized = normalize_waveform(waveform)  # Scale by 1e21
    >>> print(f"Typical amplitude: {normalized.std():.1f}")
    Typical amplitude: 1.0
    """
    # Convert to numpy array if needed
    if isinstance(waveform, TimeSeries):
        data = np.array(waveform)
    else:
        data = np.array(waveform)

    # Validate input
    if data.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {data.shape}")

    # Apply fixed scaling
    normalized = data * scale_factor

    return normalized


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
    6. Optionally apply bandpass filter (35-300 Hz)

    Parameters
    ----------
    waveform : np.ndarray or TimeSeries
        Single waveform to whiten (1D array)
    delta_t : float, optional
        Time resolution in seconds (default: 1/4096)
    f_lower : float, optional
        Lower frequency cutoff in Hz (default: 20.0)
    apply_bandpass : bool, optional
        Apply bandpass filter 35-300 Hz (default: True)
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
    Edge effects (amplitude spikes at start/end) are caused by FIR filter
    transients from the bandpass filter. The Tukey window prevents these by
    smoothly tapering the signal edges to zero before processing, eliminating
    the discontinuities that cause filter ringing.

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

    # Apply optional bandpass filtering
    if apply_bandpass:
        white_strain = highpass_fir(white_strain, f_lower, 8)
        white_strain = lowpass_fir(white_strain, 300, 8)

    # Prepare outputs
    whitened = np.array(white_strain)
    psd_array = np.array(psd)
    freqs = np.arange(len(psd)) * psd.delta_f

    return whitened, psd_array, freqs


def resample_waveform(waveform, original_delta_t, target_delta_t,
                      apply_tukey=True, tukey_alpha=0.1, tukey_side='left'):
    """
    Resample a single waveform to a different sampling rate.

    Uses PyCBC's resample_to_delta_t function which applies proper
    anti-aliasing filtering.

    Parameters
    ----------
    waveform : np.ndarray or TimeSeries
        Single waveform to resample (1D array)
    original_delta_t : float
        Original time resolution in seconds
    target_delta_t : float
        Target time resolution in seconds
    apply_tukey : bool, optional
        Apply Tukey window before resampling to prevent edge effects (default: True)
        The anti-aliasing filter in resampling can cause edge transients
    tukey_alpha : float, optional
        Tukey window alpha parameter - fraction of signal to taper (default: 0.1)
    tukey_side : str, optional
        Which side(s) to apply the Tukey taper (default: 'left')
        - 'left': Only taper the beginning (preserves merger at end)
        - 'right': Only taper the end
        - 'both': Taper both sides

    Returns
    -------
    resampled : np.ndarray
        Resampled waveform as numpy array

    Examples
    --------
    >>> waveform = np.random.randn(8192)
    >>> # Downsample from 4096 Hz to 2048 Hz
    >>> resampled = resample_waveform(waveform,
    ...                               original_delta_t=1/4096,
    ...                               target_delta_t=1/2048)
    >>> print(f"Original length: {len(waveform)}, Resampled length: {len(resampled)}")
    Original length: 8192, Resampled length: 4096
    """
    from scipy.signal.windows import tukey

    # Convert to numpy array if TimeSeries
    if isinstance(waveform, TimeSeries):
        data = np.array(waveform)
    else:
        data = np.array(waveform)
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")

    # Apply Tukey window before resampling to prevent edge effects
    if apply_tukey:
        n = len(data)
        if tukey_side == 'both':
            window = tukey(n, alpha=tukey_alpha)
        elif tukey_side == 'left':
            full_window = tukey(n, alpha=tukey_alpha * 2)
            window = np.ones(n)
            taper_len = int(n * tukey_alpha)
            window[:taper_len] = full_window[:taper_len]
        elif tukey_side == 'right':
            full_window = tukey(n, alpha=tukey_alpha * 2)
            window = np.ones(n)
            taper_len = int(n * tukey_alpha)
            window[-taper_len:] = full_window[-taper_len:]
        else:
            raise ValueError(f"tukey_side must be 'left', 'right', or 'both', got '{tukey_side}'")
        data = data * window

    # Convert to TimeSeries
    strain = TimeSeries(data, delta_t=original_delta_t)

    # Resample using PyCBC function
    resampled_strain = resample_to_delta_t(strain, target_delta_t)

    # Convert to numpy array
    resampled = np.array(resampled_strain)

    return resampled

def simulate_sine_wave(frequency, num_points=1000, noise_std=0.1, amplitude=1.0, phase=0):
    """
    Generate noisy sine wave with given frequency, amplitude, and phase.

    Args:
        frequency: sine wave frequency
        num_points: number of time points. Defaults to 1000.
        noise_std: Gaussian noise std dev. Defaults to 0.1.
        amplitude: sine amplitude. Defaults to 1.0.
        phase: phase shift. Defaults to 0.

    Returns:
        ndarray: Noisy sine wave observations.
    """
    t = np.linspace(0, 6*np.pi, num_points)
    signal = amplitude * np.sin(2*np.pi*frequency * t + phase)
    noise = np.random.normal(0, noise_std, num_points)
    observed_data = signal + noise
    return observed_data

def generate_sine_data(num_simulations=10000, freq_low=0.5, freq_high=5.0, phase_low=-3, phase_high=3, amplitude_low=0.5, amplitude_high=3.0, num_points=1000, noise_std=0.1, batch_size=256):
    """
    Generate vectorized training dataset with sine signals and DataLoaders.

    Args:
        num_simulations: number of samples. Defaults to 10000.
        freq_low, freq_high: frequency range. Defaults to 0.5, 5.0.
        phase_low, phase_high: phase range. Defaults to -3, 3.
        amplitude_low, amplitude_high: amplitude range. Defaults to 0.5, 3.0.
        num_points: time points per signal. Defaults to 1000.
        noise_std: noise std dev. Defaults to 0.1.
        batch_size: DataLoader batch size. Defaults to 256.

    Returns:
        dict: Amplitudes, Phases, Frequencies tensors and Train/Test/Val_Loader DataLoaders.
    """
    print(f"generating {num_simulations} samples for training")
    
    # Sample parameter ranges
    frequencies = np.random.uniform(freq_low, freq_high, num_simulations)
    phases = np.random.uniform(phase_low, phase_high, num_simulations)
    amplitudes = np.random.uniform(amplitude_low, amplitude_high, num_simulations)

    t = np.linspace(0, 6*np.pi, num_points)

    # Generate signals with broadcasting
    signal = amplitudes[:, np.newaxis] * np.sin(
        2*np.pi*frequencies[:, np.newaxis] * t + phases[:, np.newaxis]
    )

    noise = np.random.normal(0, noise_std, (num_simulations, num_points))
    X = torch.FloatTensor(signal + noise)
    y = torch.FloatTensor(np.column_stack([amplitudes, frequencies, phases]))

    frequencies = torch.FloatTensor(frequencies).unsqueeze(1)
    phases = torch.FloatTensor(phases).unsqueeze(1)
    amplitudes = torch.FloatTensor(amplitudes).unsqueeze(1)

    data = TensorDataset(X, y)  
    
    train_data, val_data, test_data = random_split(data, lengths=[0.8,0.1,0.1])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print("data generated")
    
    return {"Amplitudes": amplitudes, "Phases": phases, "Frequencies":frequencies,
            "Train_Loader": train_loader, "Test_Loader": test_loader, "Val_Loader": val_loader}

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
                              add_noise: bool = True) -> Dict:
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
            if signal_len >= target_length:
                # Crop from the end (keeps merger which is at the end)
                signal = signal[-target_length:]
            else:
                # Zero-pad at the beginning (keeps merger at the end)
                padded = TimeSeries(np.zeros(target_length, dtype=signal.dtype),
                                   delta_t=signal.delta_t,
                                   epoch=signal.start_time - (target_length - signal_len) * signal.delta_t)
                padded[-signal_len:] = signal
                signal = padded

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
                                add_noise: bool) -> List[Dict]:
    """Generate waveforms in parallel using multiprocessing."""
    worker_func = partial(_generate_single_waveform,
                          time_resolution=time_resolution,
                          approximant=approximant,
                          f_lower=f_lower,
                          detectors=detectors,
                          target_length=target_length,
                          add_noise=add_noise)
    
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
                        add_noise: bool = True) -> Dict:
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
    print(f"Target signal length: {target_length} samples ({signal_length}s at {time_resolution}s resolution)")
    print(f"Noise injection: {'enabled' if add_noise else 'disabled'}")

    # Generate all parameters upfront
    param_dicts = _generate_parameter_sets(config, num_samples)

    # Process in chunks for memory efficiency
    all_successful = []
    all_failed = []
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_params = param_dicts[chunk_start:chunk_end]

        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")

        # Generate waveforms with detector projection at fixed length
        chunk_results = _generate_waveforms_parallel(
            chunk_params, time_resolution, approximant, f_lower, num_workers, show_progress, detectors, target_length, add_noise
        )
        
        # Single pass: separate and accumulate
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
            'detectors': detector_names,
            'target_length': target_length,
            'signal_length': signal_length,
            'chunk_size': chunk_size,
            'sky_params_provided': sky_params_provided,
            'add_noise': add_noise,
            'preprocessing': {}
        }
    }


def _generate_single_modified_waveform(params: Dict, time_resolution: float,
                                        approximant: str, f_lower: float,
                                        detectors: List[str], target_length: int,
                                        add_noise: bool, lambda_g: float,
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
                                   add_noise: bool = True) -> Dict:
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

        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")

        chunk_results = _generate_modified_waveforms_parallel(
            chunk_params, time_resolution, approximant, f_lower,
            num_workers, show_progress, detectors, target_length,
            add_noise, lambda_g, f_final
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
            'detectors': detector_names,
            'target_length': target_length,
            'signal_length': signal_length,
            'chunk_size': chunk_size,
            'sky_params_provided': sky_params_provided,
            'add_noise': add_noise,
            'lambda_g': lambda_g,
            'lambda_g_varied': lambda_g_in_config,
            'modified': True,
            'preprocessing': {}
        }
    }


def save_dataloaders(result: Dict, save_path: str) -> None:
    """
    Save the datasets from a pycbc_data_generator result.
    
    Parameters
    ----------
    result : dict
        The result dictionary from pycbc_data_generator containing 
        train_loader, val_loader, test_loader, and metadata
    save_path : str
        Path where to save the data (e.g., 'my_data.pt')
    
    Examples
    --------
    >>> result = pycbc_data_generator(config, num_samples=1000)
    >>> save_dataloaders(result, 'my_waveforms.pt')
    """
    print(f"Saving datasets to {save_path}...")
    
    # Extract the underlying datasets and indices from the DataLoaders
    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset
    
    # Get the full tensors from the base dataset
    # The subsets have .dataset (base) and .indices attributes
    base_dataset = train_dataset.dataset
    X = base_dataset.tensors[0]
    y = base_dataset.tensors[1]
    
    save_data = {
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'test_indices': test_dataset.indices,
        'X': X,
        'y': y,
        'metadata': result['metadata']
    }
    
    torch.save(save_data, save_path)
    print(f"  Saved successfully!")


def load_dataloaders(load_path: str, batch_size: int = None, shuffle_train: bool = True) -> Dict:
    """
    Load previously saved datasets and create DataLoaders.
    
    Parameters
    ----------
    load_path : str
        Path to the saved data file (created with save_dataloaders)
    batch_size : int, optional
        Batch size for DataLoaders. If None, uses the batch_size from metadata.
    shuffle_train : bool
        Whether to shuffle training data. Default: True
    
    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'
    
    Examples
    --------
    >>> # Save after generation
    >>> result = pycbc_data_generator(config, num_samples=1000)
    >>> save_dataloaders(result, 'my_data.pt')
    >>> 
    >>> # Later, load the data
    >>> loaded = load_dataloaders('my_data.pt')
    >>> train_loader = loaded['train_loader']
    >>> val_loader = loaded['val_loader']
    >>> test_loader = loaded['test_loader']
    """
    print(f"Loading datasets from {load_path}...")
    
    # Load the saved data
    save_data = torch.load(load_path, weights_only=False)
    
    X = save_data['X']
    y = save_data['y']
    train_indices = save_data['train_indices']
    val_indices = save_data['val_indices']
    test_indices = save_data['test_indices']
    metadata = save_data['metadata']
    
    # Use saved batch_size if not provided
    if batch_size is None:
        batch_size = metadata['batch_size']
    else:
        # Update metadata with new batch_size
        metadata = metadata.copy()
        metadata['batch_size'] = batch_size
    
    print(f"  Tensors: X={X.shape}, y={y.shape}")
    print(f"  Splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Recreate the dataset
    full_dataset = TensorDataset(X, y)
    
    # Create subsets using the saved indices
    train_data = Subset(full_dataset, train_indices)
    val_data = Subset(full_dataset, val_indices)
    test_data = Subset(full_dataset, test_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print(f"\nReady! DataLoaders with batch_size={batch_size}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': metadata
    }

def resample_dataloaders(result: Dict,
                        target_sample_rate: float,
                        batch_size: int = None,
                        preserve_splits: bool = True) -> Dict:
    """
    Resample all waveforms in a DataLoader result to a different sampling rate.

    This function extracts waveforms from existing DataLoaders, resamples them
    using PyCBC's anti-aliasing resample function, and repackages them into
    new DataLoaders with updated metadata.

    Parameters
    ----------
    result : dict
        Result dictionary from pycbc_data_generator() or load_dataloaders()
        Must contain 'train_loader', 'val_loader', 'test_loader', 'metadata'
    target_sample_rate : float
        Target sampling rate in Hz (e.g., 2048 for downsampling from 4096)
    batch_size : int, optional
        Batch size for new DataLoaders. If None, uses original batch_size
    preserve_splits : bool
        If True, maintains original train/val/test splits. Default: True

    Returns
    -------
    dict
        New DataLoader result with resampled data, same structure as input

    Examples
    --------
    >>> # Generate data at 4096 Hz
    >>> result = pycbc_data_generator(config, num_samples=1000)
    >>> # Resample to 2048 Hz
    >>> result_2048 = resample_dataloaders(result, target_sample_rate=2048)
    >>> # Or load from disk and resample
    >>> loaded = load_dataloaders('data.pt')
    >>> resampled = resample_dataloaders(loaded, target_sample_rate=1024)
    """

    # Extract metadata
    metadata = result['metadata'].copy()
    original_delta_t = metadata['time_resolution']
    original_rate = 1.0 / original_delta_t
    target_delta_t = 1.0 / target_sample_rate

    print(f"Resampling DataLoaders from {original_rate:.0f} Hz to {target_sample_rate:.0f} Hz...")

    # Extract indices from Subsets
    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset

    # Get base dataset (unwrap from Subset)
    base_dataset = train_dataset.dataset
    X_full = base_dataset.tensors[0]  # (N, num_detectors, time_length)
    y_full = base_dataset.tensors[1]  # (N, num_params)

    # Get split indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    # Compute new length
    num_samples = X_full.shape[0]
    num_detectors = X_full.shape[1]
    original_length = X_full.shape[2]
    original_duration = original_length * original_delta_t
    new_length = int(original_duration / target_delta_t)

    print(f"  Processing {num_samples} waveforms...")
    print(f"  Original: {original_length} samples/waveform")
    print(f"  New: {new_length} samples/waveform")

    # Pre-allocate new tensor
    X_resampled = torch.zeros(num_samples, num_detectors, new_length, dtype=torch.float32)

    # Resample all waveforms
    for i in range(num_samples):
        for j in range(num_detectors):
            # Extract waveform as numpy array
            waveform = X_full[i, j, :].numpy()

            # Resample using PyCBC
            resampled = resample_waveform(
                waveform,
                original_delta_t=original_delta_t,
                target_delta_t=target_delta_t
            )

            # Store back in tensor
            X_resampled[i, j, :] = torch.from_numpy(resampled)

    print(f"  Resampling complete!")

    # Create new dataset
    new_dataset = TensorDataset(X_resampled, y_full)

    # Create subsets with preserved indices
    if preserve_splits:
        train_data = Subset(new_dataset, train_indices)
        val_data = Subset(new_dataset, val_indices)
        test_data = Subset(new_dataset, test_indices)
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
    else:
        # Re-split with same proportions
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        train_data, val_data, test_data = random_split(
            new_dataset, [train_size, val_size, test_size]
        )

    # Determine batch size
    if batch_size is None:
        batch_size = metadata.get('batch_size', 256)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Update metadata to reflect resampling
    new_metadata = metadata.copy()
    new_metadata['time_resolution'] = target_delta_t
    new_metadata['waveform_shape'] = (num_detectors, new_length)
    new_metadata['target_length'] = new_length
    new_metadata['batch_size'] = batch_size
    new_metadata['train_size'] = train_size
    new_metadata['val_size'] = val_size
    new_metadata['test_size'] = test_size

    # Update preprocessing info
    if 'preprocessing' not in new_metadata:
        new_metadata['preprocessing'] = {}

    new_metadata['preprocessing'] = new_metadata['preprocessing'].copy()
    new_metadata['preprocessing']['resampled'] = True
    new_metadata['preprocessing']['original_rate'] = original_rate
    new_metadata['preprocessing']['resample_rate'] = target_sample_rate
    new_metadata['preprocessing']['final_length_samples'] = new_length

    print(f"\nNew DataLoaders created:")
    print(f"  Sampling rate: {target_sample_rate:.0f} Hz")
    print(f"  Waveform shape: {new_metadata['waveform_shape']}")
    print(f"  Batch size: {batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': new_metadata
    }


def truncate_dataloaders(result: Dict,
                         target_length: int = None,
                         target_duration: float = None,
                         keep_end: bool = True,
                         batch_size: int = None,
                         preserve_splits: bool = True) -> Dict:
    """
    Truncate all waveforms in a DataLoader result to a specified length.

    This function extracts waveforms from existing DataLoaders, cuts them
    to the specified length, and repackages them into new DataLoaders.
    Useful for keeping only the merger portion of signals or standardizing
    signal lengths.

    Parameters
    ----------
    result : dict
        Result dictionary from pycbc_data_generator() or load_dataloaders()
        Must contain 'train_loader', 'val_loader', 'test_loader', 'metadata'
    target_length : int, optional
        Target length in samples. Either this or target_duration must be specified.
    target_duration : float, optional
        Target duration in seconds. Will be converted to samples using metadata.
        Either this or target_length must be specified.
    keep_end : bool
        If True (default), keeps the END of the signal (where merger is).
        If False, keeps the START of the signal.
    batch_size : int, optional
        Batch size for new DataLoaders. If None, uses original batch_size.
    preserve_splits : bool
        If True, maintains original train/val/test splits. Default: True

    Returns
    -------
    dict
        New DataLoader result with truncated data, same structure as input

    Examples
    --------
    >>> # Generate 2-second signals
    >>> result = pycbc_data_generator(config, num_samples=1000, signal_length=2.0)
    >>> # Keep only last 1 second (merger)
    >>> result_1s = truncate_dataloaders(result, target_duration=1.0, keep_end=True)
    >>> # Or specify in samples
    >>> result_512 = truncate_dataloaders(result, target_length=512, keep_end=True)
    """
    # Extract metadata
    metadata = result['metadata'].copy()
    delta_t = metadata['time_resolution']
    sample_rate = 1.0 / delta_t

    # Determine target length
    if target_length is None and target_duration is None:
        raise ValueError("Must specify either target_length or target_duration")
    if target_length is not None and target_duration is not None:
        raise ValueError("Specify only one of target_length or target_duration")

    if target_duration is not None:
        target_length = int(target_duration * sample_rate)

    # Extract indices from Subsets
    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset

    # Get base dataset (unwrap from Subset)
    base_dataset = train_dataset.dataset
    X_full = base_dataset.tensors[0]  # (N, num_detectors, time_length)
    y_full = base_dataset.tensors[1]  # (N, num_params)

    # Get split indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    # Get dimensions
    num_samples = X_full.shape[0]
    num_detectors = X_full.shape[1]
    original_length = X_full.shape[2]

    if target_length > original_length:
        raise ValueError(f"target_length ({target_length}) cannot be greater than "
                        f"original length ({original_length})")

    original_duration = original_length * delta_t
    new_duration = target_length * delta_t

    print(f"Truncating DataLoaders...")
    print(f"  Original: {original_length} samples ({original_duration:.3f}s)")
    print(f"  Target: {target_length} samples ({new_duration:.3f}s)")
    print(f"  Keeping: {'END' if keep_end else 'START'} of signal")

    # Truncate waveforms
    if keep_end:
        # Keep the end (where merger is)
        X_truncated = X_full[:, :, -target_length:]
    else:
        # Keep the start
        X_truncated = X_full[:, :, :target_length]

    # Make a contiguous copy
    X_truncated = X_truncated.clone()

    print(f"  Processing {num_samples} waveforms... done!")

    # Create new dataset
    new_dataset = TensorDataset(X_truncated, y_full)

    # Create subsets with preserved indices
    if preserve_splits:
        train_data = Subset(new_dataset, train_indices)
        val_data = Subset(new_dataset, val_indices)
        test_data = Subset(new_dataset, test_indices)
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
    else:
        # Re-split with same proportions
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        train_data, val_data, test_data = random_split(
            new_dataset, [train_size, val_size, test_size]
        )

    # Determine batch size
    if batch_size is None:
        batch_size = metadata.get('batch_size', 256)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Update metadata
    new_metadata = metadata.copy()
    new_metadata['waveform_shape'] = (num_detectors, target_length)
    new_metadata['target_length'] = target_length
    new_metadata['signal_length'] = new_duration
    new_metadata['batch_size'] = batch_size
    new_metadata['train_size'] = train_size
    new_metadata['val_size'] = val_size
    new_metadata['test_size'] = test_size

    # Update preprocessing info
    if 'preprocessing' not in new_metadata:
        new_metadata['preprocessing'] = {}

    new_metadata['preprocessing'] = new_metadata['preprocessing'].copy()
    new_metadata['preprocessing']['truncated'] = True
    new_metadata['preprocessing']['original_length'] = original_length
    new_metadata['preprocessing']['truncated_length'] = target_length
    new_metadata['preprocessing']['keep_end'] = keep_end

    print(f"\nNew DataLoaders created:")
    print(f"  Waveform shape: {new_metadata['waveform_shape']}")
    print(f"  Duration: {new_duration:.3f}s")
    print(f"  Batch size: {batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': new_metadata
    }


def _whiten_single_waveform(args):
    """Worker function for parallel whitening."""
    from JHPY import whiten_waveform
    waveform, delta_t, f_lower, apply_bandpass, apply_tukey, tukey_alpha, tukey_side = args
    whitened, _, _ = whiten_waveform(
        waveform,
        delta_t=delta_t,
        f_lower=f_lower,
        apply_bandpass=apply_bandpass,
        apply_tukey=apply_tukey,
        tukey_alpha=tukey_alpha,
        tukey_side=tukey_side
    )
    return whitened


def whiten_dataloaders(result: Dict,
                       f_lower: float = 20.0,
                       apply_bandpass: bool = True,
                       apply_tukey: bool = True,
                       tukey_alpha: float = 0.1,
                       tukey_side: str = 'left',
                       batch_size: int = None,
                       preserve_splits: bool = True,
                       num_workers: int = 1,
                       show_progress: bool = True) -> Dict:
    """
    Whiten all waveforms in a DataLoader result.

    This function extracts waveforms from existing DataLoaders, applies PSD-based
    whitening using the whiten_waveform() function, and repackages them into
    new DataLoaders with updated metadata.

    Parameters
    ----------
    result : dict
        Result dictionary from pycbc_data_generator() or load_dataloaders()
        Must contain 'train_loader', 'val_loader', 'test_loader', 'metadata'
    f_lower : float
        Lower frequency cutoff in Hz. Default: 20.0
    apply_bandpass : bool
        Apply 35-300 Hz bandpass filter after whitening. Default: True
    apply_tukey : bool
        Apply Tukey window before whitening to prevent edge effects. Default: True
    tukey_alpha : float
        Tukey window alpha parameter - fraction of signal to taper. Default: 0.1
    tukey_side : str
        Which side(s) to apply the Tukey taper. Default: 'left'
        - 'left': Only taper the beginning (preserves merger at end)
        - 'right': Only taper the end
        - 'both': Taper both sides (standard Tukey window)
    batch_size : int, optional
        Batch size for new DataLoaders. If None, uses original batch_size
    preserve_splits : bool
        If True, maintains original train/val/test splits. Default: True
    num_workers : int
        Number of parallel workers for whitening. Default: 1
    show_progress : bool
        Show progress bar during whitening. Default: True

    Returns
    -------
    dict
        New DataLoader result with whitened data, same structure as input
    """
    # Extract metadata
    metadata = result['metadata'].copy()
    delta_t = metadata['time_resolution']

    print(f"Whitening DataLoaders...")
    print(f"  f_lower: {f_lower} Hz")
    print(f"  Bandpass: {apply_bandpass}")
    print(f"  Tukey window: {apply_tukey} (alpha={tukey_alpha}, side={tukey_side})")

    # Extract data from DataLoaders
    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset

    # Get base dataset (unwrap from Subset)
    base_dataset = train_dataset.dataset
    X_full = base_dataset.tensors[0]  # (N, num_detectors, time_length)
    y_full = base_dataset.tensors[1]  # (N, num_params)

    # Get split indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    num_samples = X_full.shape[0]
    num_detectors = X_full.shape[1]
    time_length = X_full.shape[2]

    print(f"  Processing {num_samples} waveforms x {num_detectors} detectors...")

    # Pre-allocate output tensor
    X_whitened = torch.zeros_like(X_full)

    # Prepare arguments for parallel processing
    all_args = []
    for i in range(num_samples):
        for j in range(num_detectors):
            waveform = X_full[i, j, :].numpy()
            all_args.append((waveform, delta_t, f_lower, apply_bandpass, apply_tukey, tukey_alpha, tukey_side))

    # Process in parallel or single-threaded
    if num_workers > 1:
        import multiprocessing as mp
        # Use spawn to avoid issues with forking and PyCBC/scipy
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(_whiten_single_waveform, all_args, chunksize=10),
                    total=len(all_args),
                    desc="Whitening"
                ))
            else:
                results = list(pool.imap(_whiten_single_waveform, all_args, chunksize=10))
    else:
        # Single-threaded
        if show_progress:
            results = [_whiten_single_waveform(args) for args in tqdm(all_args, desc="Whitening")]
        else:
            results = [_whiten_single_waveform(args) for args in all_args]

    # Reshape results back into tensor
    idx = 0
    for i in range(num_samples):
        for j in range(num_detectors):
            X_whitened[i, j, :] = torch.from_numpy(results[idx])
            idx += 1

    print(f"  Whitening complete!")

    # Create new dataset
    new_dataset = TensorDataset(X_whitened, y_full)

    # Create subsets with preserved indices
    if preserve_splits:
        train_data = Subset(new_dataset, train_indices)
        val_data = Subset(new_dataset, val_indices)
        test_data = Subset(new_dataset, test_indices)
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
    else:
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        train_data, val_data, test_data = random_split(
            new_dataset, [train_size, val_size, test_size]
        )

    # Determine batch size
    if batch_size is None:
        batch_size = metadata.get('batch_size', 256)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Update metadata
    new_metadata = metadata.copy()
    new_metadata['batch_size'] = batch_size
    new_metadata['train_size'] = train_size
    new_metadata['val_size'] = val_size
    new_metadata['test_size'] = test_size

    # Update preprocessing info
    if 'preprocessing' not in new_metadata:
        new_metadata['preprocessing'] = {}

    new_metadata['preprocessing'] = new_metadata['preprocessing'].copy()
    new_metadata['preprocessing']['whitened'] = True
    new_metadata['preprocessing']['whiten_f_lower'] = f_lower
    new_metadata['preprocessing']['whiten_bandpass'] = apply_bandpass
    new_metadata['preprocessing']['whiten_tukey'] = apply_tukey
    new_metadata['preprocessing']['whiten_tukey_alpha'] = tukey_alpha
    new_metadata['preprocessing']['whiten_tukey_side'] = tukey_side

    print(f"\nNew DataLoaders created:")
    print(f"  Waveform shape: {new_metadata['waveform_shape']}")
    print(f"  Batch size: {batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': new_metadata
    }


def normalize_dataloaders(result: Dict,
                          scale_factor: float = 1e21,
                          batch_size: int = None,
                          preserve_splits: bool = True) -> Dict:
    """
    Normalize all waveforms in a DataLoader result by multiplying by a scale factor.

    This function extracts waveforms from existing DataLoaders, applies fixed-scale
    normalization, and repackages them into new DataLoaders with updated metadata.

    Parameters
    ----------
    result : dict
        Result dictionary from pycbc_data_generator() or load_dataloaders()
        Must contain 'train_loader', 'val_loader', 'test_loader', 'metadata'
    scale_factor : float
        Fixed scaling factor to multiply waveforms by. Default: 1e21
        (appropriate for typical GW strains of ~1e-21)
    batch_size : int, optional
        Batch size for new DataLoaders. If None, uses original batch_size
    preserve_splits : bool
        If True, maintains original train/val/test splits. Default: True

    Returns
    -------
    dict
        New DataLoader result with normalized data, same structure as input
    """
    # Extract metadata
    metadata = result['metadata'].copy()

    print(f"Normalizing DataLoaders...")
    print(f"  Scale factor: {scale_factor:.2e}")

    # Extract data from DataLoaders
    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset

    # Get base dataset (unwrap from Subset)
    base_dataset = train_dataset.dataset
    X_full = base_dataset.tensors[0]  # (N, num_detectors, time_length)
    y_full = base_dataset.tensors[1]  # (N, num_params)

    # Get split indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    num_samples = X_full.shape[0]

    print(f"  Processing {num_samples} waveforms...")

    # Apply normalization (simple multiplication, very fast)
    X_normalized = X_full * scale_factor

    print(f"  Normalization complete!")

    # Create new dataset
    new_dataset = TensorDataset(X_normalized, y_full)

    # Create subsets with preserved indices
    if preserve_splits:
        train_data = Subset(new_dataset, train_indices)
        val_data = Subset(new_dataset, val_indices)
        test_data = Subset(new_dataset, test_indices)
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
    else:
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        train_data, val_data, test_data = random_split(
            new_dataset, [train_size, val_size, test_size]
        )

    # Determine batch size
    if batch_size is None:
        batch_size = metadata.get('batch_size', 256)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Update metadata
    new_metadata = metadata.copy()
    new_metadata['batch_size'] = batch_size
    new_metadata['train_size'] = train_size
    new_metadata['val_size'] = val_size
    new_metadata['test_size'] = test_size

    # Update preprocessing info
    if 'preprocessing' not in new_metadata:
        new_metadata['preprocessing'] = {}

    new_metadata['preprocessing'] = new_metadata['preprocessing'].copy()
    new_metadata['preprocessing']['normalized'] = True
    new_metadata['preprocessing']['normalize_scale'] = scale_factor

    print(f"\nNew DataLoaders created:")
    print(f"  Waveform shape: {new_metadata['waveform_shape']}")
    print(f"  Batch size: {batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': new_metadata
    }


################### Neural Network Layers ###################

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows

    Splits input, transforms one half conditioned on the other:
    x2_new = x2 * exp(s(x1, context)) + t(x1, context)
    """
    def __init__(self, dim, context_dim, hidden_dim=128, mask_type='half'):
        super().__init__()
        self.dim = dim

        # Create alternating mask for coupling layer
        self.register_buffer('mask', torch.zeros(dim))
        if mask_type in ['half', 'even']:
            self.mask[::2] = 1
        elif mask_type == 'odd':
            self.mask[1::2] = 1

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()  # Stabilize training
        )

        self.translation_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, context, reverse=False):
        """
        Forward (data -> latent) or reverse (latent -> data) transformation

        Args:
            x: input tensor [batch_size, dim]
            context: conditioning context (embedded data) [batch_size, context_dim]
            reverse: if True, compute inverse transformation

        Returns:
            output: transformed tensor
            log_det: log determinant of Jacobian
        """
        masked_x = x * self.mask

        scale_input = torch.cat([masked_x, context], dim=1)
        translation_input = torch.cat([masked_x, context], dim=1)

        # Compute scale and translation
        s = self.scale_net(scale_input)
        t = self.translation_net(translation_input)

        # Only apply to unmasked dimensions
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            # Forward: x -> z
            y = x * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            # Reverse: z -> x
            y = (x - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)

        return y, log_det

class EmbeddingNetwork(nn.Module):
    """
    Neural network to embed observed data into context vector.
    Similar to DINGO's data compression network.
    Supports multi-detector data with multiple processing strategies.
    """
    def __init__(self, data_dim=100, context_dim=64, hidden_dim=128, num_detectors=1, multi_detector_mode='concatenate'):
        super().__init__()

        self.num_detectors = num_detectors
        self.multi_detector_mode = multi_detector_mode
        self.data_dim = data_dim
        self.context_dim = context_dim

        if multi_detector_mode == 'concatenate':
            # Concatenate all detector data and process together
            self.network = nn.Sequential(
                nn.Linear(data_dim * num_detectors, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),  # Dropout layer (rate set by training function)
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),  # Dropout layer (rate set by training function)
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),  # Dropout layer (rate set by training function)
                nn.Linear(hidden_dim, context_dim)
            )

        elif multi_detector_mode == 'separate':
            # Process each detector separately, then combine
            self.detector_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(data_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.0),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.0),
                    nn.Linear(hidden_dim, context_dim // num_detectors)
                )
                for _ in range(num_detectors)
            ])

            # Combine detector embeddings
            self.combine_network = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, context_dim)
            )

        elif multi_detector_mode == 'shared':
            # Shared network for all detectors, then combine
            self.shared_network = nn.Sequential(
                nn.Linear(data_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, context_dim // num_detectors)
            )

            self.combine_network = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, context_dim)
            )
        else:
            raise ValueError(f"Unknown multi_detector_mode: {multi_detector_mode}")

    def forward(self, data):
        """
        Args:
            data: observed data
                - If num_detectors == 1: [batch_size, data_dim]
                - If num_detectors > 1: [batch_size, num_detectors, data_dim]

        Returns:
            context: embedded representation [batch_size, context_dim]
        """
        if self.num_detectors == 1:
            # Single detector case
            if self.multi_detector_mode == 'concatenate':
                return self.network(data)
            else:
                # For consistency with multi-detector modes
                return self.network(data) if hasattr(self, 'network') else self.shared_network(data)

        # Multi-detector case
        batch_size = data.shape[0]

        if self.multi_detector_mode == 'concatenate':
            # Flatten detectors: [batch, num_detectors, data_dim] -> [batch, num_detectors * data_dim]
            data_flat = data.reshape(batch_size, -1)
            return self.network(data_flat)

        elif self.multi_detector_mode == 'separate':
            # Process each detector separately
            detector_embeddings = []
            for i in range(self.num_detectors):
                embedding = self.detector_networks[i](data[:, i, :])
                detector_embeddings.append(embedding)

            # Concatenate and combine
            combined = torch.cat(detector_embeddings, dim=1)
            return self.combine_network(combined)

        elif self.multi_detector_mode == 'shared':
            # Process each detector with shared weights
            detector_embeddings = []
            for i in range(self.num_detectors):
                embedding = self.shared_network(data[:, i, :])
                detector_embeddings.append(embedding)

            # Concatenate and combine
            combined = torch.cat(detector_embeddings, dim=1)
            return self.combine_network(combined)

################### Model Classes ###################

class ParameterPredictor(nn.Module):
    """
    LSTM-based neural network for predicting scalar parameters from time series.

    Configurable model with LSTM layers followed by fully connected layers.
    Config options: lstm_hidden_size (256), lstm_num_layers (1), fc_layer_sizes ([128, 64]),
    activation ('silu'/'relu'/'tanh'), dropout (0.0).
    """
    def __init__(self, config=None):
        """
        Initialize model with optional config overrides.

        Args:
            config (dict, optional): Config dict with lstm_hidden_size, lstm_num_layers, fc_layer_sizes, activation, dropout.
        """
        super().__init__()

        # Default configuration
        default_config = {
            'lstm_hidden_size': 256,
            'lstm_num_layers': 1,
            'fc_layer_sizes': [128, 64],  # Sizes of fully connected layers before output
            'activation': 'silu',  # 'silu', 'relu', 'tanh'
            'dropout': 0.0,  # Dropout probability
        }

        if config is None:
            config = {}
        self.config = {**default_config, **config}
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.config['lstm_hidden_size'],
            num_layers=self.config['lstm_num_layers'],
            batch_first=True,
            dropout=self.config['dropout'] if self.config['lstm_num_layers'] > 1 else 0.0
        )

        fc_layers = []
        input_size = self.config['lstm_hidden_size']

        for hidden_size in self.config['fc_layer_sizes']:
            fc_layers.append(nn.Linear(input_size, hidden_size))

            if self.config['activation'] == 'silu':
                fc_layers.append(nn.SiLU())
            elif self.config['activation'] == 'relu':
                fc_layers.append(nn.ReLU())
            elif self.config['activation'] == 'tanh':
                fc_layers.append(nn.Tanh())

            if self.config['dropout'] > 0:
                fc_layers.append(nn.Dropout(self.config['dropout']))

            input_size = hidden_size

        fc_layers.append(nn.Linear(input_size, 3))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Forward pass: process time series through LSTM and FC layers.

        Args:
            x (torch.Tensor): Input shape [batch, sequence_length]

        Returns:
            torch.Tensor: Output shape [batch, 3] with predictions for amplitude, frequency, phase
        """
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

class NormalizingFlow(nn.Module):
    """
    Normalizing flow: stack of coupling layers
    Transforms base distribution into complex posterior
    """
    def __init__(self, param_dim=1, context_dim=64, num_layers=6, hidden_dim=128, config=None):
        super().__init__()

        # Support both positional args and config dict
        if config is not None:
            param_dim = config.get('param_dim', param_dim)
            context_dim = config.get('context_dim', context_dim)
            num_layers = config.get('num_flow_layers', config.get('num_layers', num_layers))
            hidden_dim = config.get('hidden_dim', hidden_dim)

        # Store config for checkpointing
        self.config = {
            'param_dim': param_dim,
            'context_dim': context_dim,
            'num_flow_layers': num_layers,
            'hidden_dim': hidden_dim
        }

        self.param_dim = param_dim
        self.context_dim = context_dim

        # Stack of coupling layers with alternating masks
        self.layers = nn.ModuleList([
            AffineCouplingLayer(
                dim=param_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                mask_type='even' if i % 2 == 0 else 'odd'
            )
            for i in range(num_layers)
        ])

        # Base distribution: standard Gaussian using PyTorch distributions
        self.base_dist = dist.Normal(loc=0.0, scale=1.0)

    def forward(self, params, context):
        """
        Forward pass: compute log probability of parameters given context

        Args:
            params: parameter values [batch_size, param_dim]
            context: embedded observed data [batch_size, context_dim]

        Returns:
            log_prob: log p(params | context)
        """
        z = params
        log_det_sum = 0

        # Apply flow transformations
        for layer in self.layers:
            z, log_det = layer(z, context, reverse=False)
            log_det_sum += log_det

        # Compute log probability under base distribution using PyTorch distributions
        log_prob_base = self.base_dist.log_prob(z).sum(dim=1)

        # Apply change of variables
        log_prob = log_prob_base + log_det_sum

        return log_prob

    def sample(self, context, num_samples=1):
        """
        Sample from posterior p(params | context)

        Args:
            context: embedded observed data [batch_size, context_dim]
            num_samples: number of samples per context

        Returns:
            samples: parameter samples [batch_size * num_samples, param_dim]
        """
        batch_size = context.shape[0]

        context_repeated = context.repeat_interleave(num_samples, dim=0)

        # Sample from base distribution using PyTorch distributions
        z = self.base_dist.sample((batch_size * num_samples, self.param_dim)).to(context.device)

        # Apply inverse flow transformations
        for layer in reversed(self.layers):
            z, _ = layer(z, context_repeated, reverse=True)

        return z

class DINGOModel(nn.Module):
    """
    Complete DINGO-style neural posterior estimation model

    Architecture:
    observed_data -> EmbeddingNet -> context -> NormalizingFlow -> log p(params | data)

    Supports multi-detector data with configurable processing strategies.
    """
    def __init__(self, data_dim=100, param_dim=1, context_dim=64,
                 num_flow_layers=6, hidden_dim=128, num_detectors=1,
                 multi_detector_mode='concatenate', config=None):
        super().__init__()

        # Support both positional args and config dict
        if config is not None:
            data_dim = config.get('data_dim', data_dim)
            param_dim = config.get('param_dim', param_dim)
            context_dim = config.get('context_dim', context_dim)
            num_flow_layers = config.get('num_flow_layers', num_flow_layers)
            hidden_dim = config.get('hidden_dim', hidden_dim)
            num_detectors = config.get('num_detectors', num_detectors)
            multi_detector_mode = config.get('multi_detector_mode', multi_detector_mode)

        # Store config for checkpointing
        self.config = {
            'data_dim': data_dim,
            'param_dim': param_dim,
            'context_dim': context_dim,
            'num_flow_layers': num_flow_layers,
            'hidden_dim': hidden_dim,
            'num_detectors': num_detectors,
            'multi_detector_mode': multi_detector_mode
        }

        self.embedding_net = EmbeddingNetwork(
            data_dim=data_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_detectors=num_detectors,
            multi_detector_mode=multi_detector_mode
        )

        self.flow = NormalizingFlow(
            param_dim=param_dim,
            context_dim=context_dim,
            num_layers=num_flow_layers,
            hidden_dim=hidden_dim
        )

    def forward(self, params, data):
        """
        Compute log probability of parameters given data

        Args:
            params: parameter values [batch_size, param_dim]
            data: observed data
                - Single detector: [batch_size, data_dim]
                - Multi-detector: [batch_size, num_detectors, data_dim]

        Returns:
            log_prob: log p(params | data)
        """
        context = self.embedding_net(data)
        log_prob = self.flow(params, context)
        return log_prob

    def sample_posterior(self, data, num_samples=1000):
        """
        Sample from posterior p(params | data)

        Args:
            data: observed data
                - Single detector: [batch_size, data_dim]
                - Multi-detector: [batch_size, num_detectors, data_dim]
            num_samples: number of samples to draw

        Returns:
            samples: posterior samples [batch_size * num_samples, param_dim]
        """
        self.eval()
        with torch.no_grad():
            context = self.embedding_net(data)
            samples = self.flow.sample(context, num_samples=num_samples)
        return samples

################### Training Functions ###################
    
def train_predictor_model(model, optimizer, loss_fcn, n_epochs, train_dloader, val_dloader, start_epoch=0, patience=8, scheduler=None, save_best_model=True, model_path='best_predictor_model.pt', grad_clip_norm=5.0, dropout_rate=None):
    """
    Train model with early stopping, validation monitoring, gradient clipping, and optional checkpointing.

    Args:
        model (nn.Module): PyTorch model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fcn (callable): Loss function
        n_epochs (int): Number of epochs to train
        train_dloader (DataLoader): Training data loader
        val_dloader (DataLoader): Validation data loader
        start_epoch (int, optional): Starting epoch for resume. Defaults to 0.
        patience (int, optional): Epochs to wait before early stopping. Defaults to 8.
        scheduler (torch.optim.lr_scheduler, optional): LR scheduler. Defaults to None.
        save_best_model (bool, optional): Save best model checkpoint. Defaults to True.
        model_path (str, optional): Path for checkpoint. Defaults to 'best_predictor_model.pt'.
        grad_clip_norm (float, optional): Max gradient norm for clipping. Set to None to disable. Defaults to 5.0.
        dropout_rate (float, optional): Dropout probability (0.0-1.0). None uses model's default. Defaults to None.

    Returns:
        dict: Contains 'train_losses', 'val_losses', 'train_metrics', 'val_metrics', 'best_val_loss', 'best_val_epoch'.
    """
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    best_val_loss = float('inf')
    best_val_epoch = 0

    # Set dropout rate if specified
    if dropout_rate is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout_rate

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()
        tloss, vloss = 0, 0
        train_predictions = []
        train_targets = []

        for X_train, y_train in tqdm(train_dloader, desc='Epoch {}, training'.format(epoch+1)):
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fcn(pred, y_train)
            tloss += loss.item()
            loss.backward()

            # Gradient clipping
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            train_predictions.extend(pred.detach().numpy())
            train_targets.extend(y_train.numpy())

            
        model.eval()
        vloss = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for X_valid, y_valid in tqdm(val_dloader, desc='Epoch {}, validation'.format(epoch+1)):
                pred = model(X_valid)
                loss = loss_fcn(pred, y_valid)
                vloss += loss.item()
                val_predictions.extend(pred.numpy())
                val_targets.extend(y_valid.numpy())

        # Calculate metrics
        train_metrics_dict = calculate_metrics(
            np.array(train_predictions), 
            np.array(train_targets)
        )
        val_metrics_dict = calculate_metrics(
            np.array(val_predictions), 
            np.array(val_targets)
        )

        # Store losses
        avg_train_loss = tloss / len(train_dloader)
        avg_val_loss = vloss / len(val_dloader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Store metrics
        train_metrics.append(train_metrics_dict)
        val_metrics.append(val_metrics_dict)

        # Print epoch results
        print(f"\n[Epoch {epoch+1:2d}]")
        print(f"Training - Loss: {avg_train_loss:.4f}, MAE: {train_metrics_dict['mae']:.4f}, "
              f"RMSE: {train_metrics_dict['rmse']:.4f}, R²: {train_metrics_dict['r2']:.4f}")
        print(f"Validation - Loss: {avg_val_loss:.4f}, MAE: {val_metrics_dict['mae']:.4f}, "
              f"RMSE: {val_metrics_dict['rmse']:.4f}, R²: {val_metrics_dict['r2']:.4f} \n")

        # Learning rate scheduling
        if scheduler is not None:
            old_lr = [param_group['lr'] for param_group in optimizer.param_groups]
            scheduler.step(avg_val_loss)
            new_lr = [param_group['lr'] for param_group in optimizer.param_groups]

            # If learning rate changed, load best model and continue training
            if old_lr != new_lr and save_best_model:
                print(f"Learning rate reduced. Loading best model from {model_path}")
                checkpoint = torch.load(model_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Keep the reduced learning rate by reapplying it to all param groups
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr[optimizer.param_groups.index(param_group)]
                print("Best model loaded, resuming training\n")

            print(f"Current learning rates: {new_lr}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            print("New best validation performance \n")
            best_val_loss = avg_val_loss
            best_val_epoch = epoch
            
            # Save the best model
            if save_best_model:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                torch.save(checkpoint, model_path)
                print(f"Model checkpoint saved to {model_path}\n")
                
        elif best_val_epoch <= epoch - patience:
            print(f'No improvement in validation loss in last {patience} epochs \n')
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss,
        'best_val_epoch': best_val_epoch
    }
    
def train_npe_model(model, optimizer, n_epochs, train_dloader, val_dloader, start_epoch=0, patience=15, scheduler=None, save_best_model=True, model_path='best_npe_model.pt', grad_clip_norm=5.0, dropout_rate=None, device='cpu'):
    """
    Train NPE model with log probability, early stopping, validation monitoring, and optional checkpointing.

    Args:
        model (nn.Module): PyTorch model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        n_epochs (int): Number of epochs to train
        train_dloader (DataLoader): Training data loader
        val_dloader (DataLoader): Validation data loader
        start_epoch (int, optional): Starting epoch for resume. Defaults to 0.
        patience (int, optional): Epochs to wait for improvement before early stopping. Defaults to 15.
        scheduler (torch.optim.lr_scheduler, optional): LR scheduler. Defaults to None.
        save_best_model (bool, optional): Save best model checkpoint. Defaults to True.
        model_path (str, optional): Path for checkpoint. Defaults to 'best_npe_model.pt'.
        grad_clip_norm (float, optional): Max gradient norm for clipping. Set to None to disable. Defaults to 5.0.
        dropout_rate (float, optional): Dropout probability (0.0-1.0). None uses model's default. Defaults to None.

    Returns:
        dict: Contains 'train_log_probs', 'val_log_probs', 'best_val_log_prob', 'best_val_epoch'.
    """
    train_log_probs = []
    val_log_probs = []
    best_val_log_prob = float('-inf')  # Higher is better for log probability
    best_val_epoch = 0

    model.to(device)

    # Set dropout rate if specified
    if dropout_rate is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout_rate

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()
        train_log_prob_sum = 0

        for X_train, y_train in tqdm(train_dloader, desc='Epoch {}, training'.format(epoch+1)):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            log_prob = model(y_train, X_train)
            loss = -log_prob.mean()  # Negative log prob for gradient descent
            
            loss.backward()
            
            # Gradient clipping
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
            optimizer.step()
            
            train_log_prob_sum += log_prob.mean().item()

        model.eval()
        val_log_prob_sum = 0

        with torch.no_grad():
            for X_valid, y_valid in tqdm(val_dloader, desc='Epoch {}, validation'.format(epoch+1)):
                X_valid = X_valid.to(device)
                y_valid = y_valid.to(device)
                log_prob = model(y_valid, X_valid)
                val_log_prob_sum += log_prob.mean().item()

        # Compute averages
        avg_train_log_prob = train_log_prob_sum / len(train_dloader)
        avg_val_log_prob = val_log_prob_sum / len(val_dloader)
        train_log_probs.append(avg_train_log_prob)
        val_log_probs.append(avg_val_log_prob)

        # Print epoch results
        print(f"\n[Epoch {epoch+1:2d}]")
        print(f"Training - Log Prob: {avg_train_log_prob:.4f}")
        print(f"Validation - Log Prob: {avg_val_log_prob:.4f}")

        # Learning rate scheduling (higher log prob is better)
        if scheduler is not None:
            old_lr = [param_group['lr'] for param_group in optimizer.param_groups]
            scheduler.step(avg_val_log_prob)
            new_lr = [param_group['lr'] for param_group in optimizer.param_groups]

            # If learning rate changed, load best model and continue training
            if old_lr != new_lr and save_best_model:
                print(f"Learning rate reduced. Loading best model from {model_path}")
                checkpoint = torch.load(model_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Keep the reduced learning rate by reapplying it to all param groups
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr[optimizer.param_groups.index(param_group)]
                print("Best model loaded, resuming training\n")

            print(f"Current learning rates: {new_lr}")

        # Early stopping check (higher log prob is better)
        if avg_val_log_prob > best_val_log_prob:
            print("New best validation performance \n")
            best_val_log_prob = avg_val_log_prob
            best_val_epoch = epoch

            # Save the best model
            if save_best_model:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_log_prob': best_val_log_prob,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_log_probs': train_log_probs,
                    'val_log_probs': val_log_probs
                }
                torch.save(checkpoint, model_path)
                print(f"Model checkpoint saved to {model_path}\n")

        elif best_val_epoch <= epoch - patience:
            print(f'No improvement in validation log prob in last {patience} epochs \n')
            break

    return {
        'train_log_probs': train_log_probs,
        'val_log_probs': val_log_probs,
        'best_val_log_prob': best_val_log_prob,
        'best_val_epoch': best_val_epoch
    }
    
################### Other Neural Network Functions ###################

def calculate_metrics(predictions, targets):
    """
    Calculate MAE, RMSE, and R² metrics.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Target values

    Returns:
        dict: Contains 'mae', 'rmse', 'r2'.
    """
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def load_predictor(model_path='best_predictor_model.pt'):
    """
    Load saved ParameterPredictor checkpoint.

    Args:
        model_path (str, optional): Path to checkpoint. Defaults to 'best_predictor_model.pt'.

    Returns:
        tuple: (model, checkpoint dict)
    """
    checkpoint = torch.load(model_path, weights_only=False)

    model = ParameterPredictor(checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded ParameterPredictor from {model_path}")
    print(f"  Best epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")

    return model, checkpoint

def load_npe(model_path='best_npe_model.pt', model_class=DINGOModel):
    """
    Load saved NPE model checkpoint (NormalizingFlow or DINGOModel).

    Args:
        model_path (str, optional): Path to checkpoint. Defaults to 'best_npe_model.pt'.
        model_class: NPE model class (DINGOModel or NormalizingFlow). Defaults to DINGOModel.

    Returns:
        tuple: (model, checkpoint dict)
    """
    checkpoint = torch.load(model_path, weights_only=False)

    model = model_class(config=checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded {model_class.__name__} from {model_path}")
    print(f"  Best epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best validation log prob: {checkpoint['best_val_log_prob']:.4f}")

    return model, checkpoint

def create_dingo_from_data(metadata, param_dim=None, context_dim=64,
                           num_flow_layers=6, hidden_dim=128, multi_detector_mode='concatenate'):
    """
    Create a DINGOModel with dimensions automatically inferred from metadata.

    Args:
        metadata: Metadata dict from pycbc_data_generator or load_dataloaders result
        param_dim: Number of parameters to infer. If None, inferred from metadata
        context_dim: Context dimension. Defaults to 64
        num_flow_layers: Number of flow layers. Defaults to 6
        hidden_dim: Hidden dimension. Defaults to 128
        multi_detector_mode: 'concatenate', 'separate', or 'shared'. Defaults to 'concatenate'

    Returns:
        DINGOModel: Model configured with correct dimensions

    Examples:
        >>> data = load_dataloaders('my_data.pt')
        >>> model = create_dingo_from_data(data['metadata'], context_dim=128, num_flow_layers=8)
    """
    # Infer dimensions from metadata (waveform_shape is always present in new format)
    if 'waveform_shape' not in metadata:
        raise ValueError("Metadata missing 'waveform_shape'. This data may be from an old version.")

    # waveform_shape format: (num_detectors, time_length)
    num_detectors, data_dim = metadata['waveform_shape']

    # Infer param_dim from parameter_names if not specified
    if param_dim is None:
        if 'parameter_names' not in metadata:
            raise ValueError("Cannot infer param_dim from metadata. Please specify explicitly.")
        param_dim = len(metadata['parameter_names'])

    # Display inferred configuration
    print(f"Creating DINGOModel with inferred dimensions:")
    print(f"  data_dim (time length per detector): {data_dim}")
    print(f"  num_detectors: {num_detectors}")
    print(f"  param_dim: {param_dim}")
    print(f"  Parameter names: {metadata.get('parameter_names', 'N/A')}")
    print(f"  Detectors: {metadata.get('detectors', 'N/A')}")
    print(f"  Signal length: {metadata.get('signal_length', 'N/A')}s")
    print(f"  Time resolution: {metadata.get('time_resolution', 'N/A')}s")
    print(f"\nModel architecture:")
    print(f"  context_dim: {context_dim}")
    print(f"  num_flow_layers: {num_flow_layers}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  multi_detector_mode: {multi_detector_mode}")

    model = DINGOModel(
        data_dim=data_dim,
        param_dim=param_dim,
        context_dim=context_dim,
        num_flow_layers=num_flow_layers,
        hidden_dim=hidden_dim,
        num_detectors=num_detectors,
        multi_detector_mode=multi_detector_mode
    )

    return model

def predictor_hyperparameter_search(param_grid, train_loader, val_loader, n_epochs=20, n_trials=None, model_path='best_predictor_model.pt'):
    """
    Search for best hyperparameter configuration.

    Args:
        param_grid: Dict of parameter names to value lists
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Epochs per configuration. Defaults to 20.
        n_trials: Random trials to try; None = all combinations. Defaults to None.
        model_path: Path to save best model. Defaults to 'best_predictor_model.pt'.

    Returns:
        tuple: (best_config dict, results list)
    """

    results = []
    best_val_loss = float('inf')
    best_config = None

    # Generate all combinations or sample randomly
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    if n_trials is None:
        # Try all combinations (grid search)
        all_combinations = list(itertools.product(*param_values))
    else:
        # Random search: sample n_trials random combinations
        all_combinations = []
        for _ in range(n_trials):
            combo = tuple(random.choice(values) for values in param_values)
            all_combinations.append(combo)
    
    print(f"Testing {len(all_combinations)} configurations...\n")
    
    for i, combo in enumerate(all_combinations):
        config = dict(zip(param_names, combo))

        print(f"{'='*60}")
        print(f"Trial {i+1}/{len(all_combinations)}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        model = ParameterPredictor(config)
        lr = config.get('learning_rate', 0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )

        loss_fcn = nn.MSELoss()

        try:
            outputs = train_predictor_model(
                model,
                optimizer,
                loss_fcn,
                n_epochs,
                train_loader,
                val_loader,
                patience=8,
                scheduler=scheduler,
                save_best_model=False
            )

            final_val_loss = min(outputs['val_losses'])
            final_val_metrics = outputs['val_metrics'][outputs['val_losses'].index(final_val_loss)]

            result = {
                'config': config.copy(),
                'best_val_loss': final_val_loss,
                'best_val_mae': final_val_metrics['mae'],
                'best_val_rmse': final_val_metrics['rmse'],
                'best_val_r2': final_val_metrics['r2'],
                'n_epochs_trained': len(outputs['val_losses'])
            }
            results.append(result)

            print(f"\nFinal validation loss: {final_val_loss:.4f}")
            print(f"Best validation R²: {final_val_metrics['r2']:.4f}\n")

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_config = config.copy()
                checkpoint = {
                    'epoch': len(outputs['val_losses']) - 1,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_losses': outputs['train_losses'],
                    'val_losses': outputs['val_losses'],
                    'train_metrics': outputs['train_metrics'],
                    'val_metrics': outputs['val_metrics']
                }
                torch.save(checkpoint, model_path)
                print(f"*** New best configuration found! ***\n")
        
        except Exception as e:
            print(f"Error training with config {config}: {e}\n")
            continue

    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest validation loss: {best_val_loss:.4f}")

    results.sort(key=lambda x: x['best_val_loss'])
    
    return best_config, results

def npe_hyperparameter_search(param_grid, train_loader, val_loader, model_class=DINGOModel, n_epochs=20, n_trials=None, model_path='best_npe_model.pt'):
    """
    Hyperparameter search optimized for NPE models (NormalizingFlow, DINGOModel).

    Args:
        param_grid: Dict of parameter names to value lists
        train_loader: Training data loader
        val_loader: Validation data loader
        model_class: NPE model class (DINGOModel or NormalizingFlow). Defaults to DINGOModel.
        n_epochs: Epochs per configuration. Defaults to 20.
        n_trials: Random trials to try; None = all combinations. Defaults to None.
        model_path: Path to save best model. Defaults to 'best_npe_model.pt'.

    Returns:
        tuple: (best_config dict, results list sorted by validation log prob)
    """
    results = []
    best_val_log_prob = float('-inf')
    best_config = None

    # Generate all combinations or sample randomly
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    if n_trials is None:
        # Try all combinations (grid search)
        all_combinations = list(itertools.product(*param_values))
    else:
        # Random search: sample n_trials random combinations
        all_combinations = []
        for _ in range(n_trials):
            combo = tuple(random.choice(values) for values in param_values)
            all_combinations.append(combo)

    print(f"Testing {len(all_combinations)} configurations with {model_class.__name__}...\n")

    for i, combo in enumerate(all_combinations):
        config = dict(zip(param_names, combo))

        print(f"{'='*60}")
        print(f"Trial {i+1}/{len(all_combinations)}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        model = model_class(config=config)
        lr = config.get('learning_rate', 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        )

        grad_clip_norm = config.get('grad_clip_norm', 5.0)
        dropout_rate = config.get('dropout_rate', None)

        try:
            outputs = train_npe_model(
                model,
                optimizer,
                n_epochs,
                train_loader,
                val_loader,
                patience=8,
                scheduler=scheduler,
                grad_clip_norm=grad_clip_norm,
                dropout_rate=dropout_rate,
                save_best_model=False
            )

            final_val_log_prob = max(outputs['val_log_probs'])

            result = {
                'config': config.copy(),
                'best_val_log_prob': final_val_log_prob,
                'n_epochs_trained': len(outputs['val_log_probs'])
            }
            results.append(result)

            print(f"\nFinal validation log prob: {final_val_log_prob:.4f}\n")

            if final_val_log_prob > best_val_log_prob:
                best_val_log_prob = final_val_log_prob
                best_config = config.copy()
                checkpoint = {
                    'epoch': len(outputs['val_log_probs']) - 1,
                    'model_state_dict': model.state_dict(),
                    'best_val_log_prob': best_val_log_prob,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_log_probs': outputs['train_log_probs'],
                    'val_log_probs': outputs['val_log_probs']
                }
                torch.save(checkpoint, model_path)
                print(f"*** New best configuration found! ***\n")

        except Exception as e:
            print(f"Error training with config {config}: {e}\n")
            continue

    print(f"\n{'='*60}")
    print("NPE HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest validation log prob: {best_val_log_prob:.4f}")

    results.sort(key=lambda x: x['best_val_log_prob'], reverse=True)

    return best_config, results

def infer_NPE(model, observed_data, num_samples=5000):
    """
    Perform inference of NPE
    
    Args:
        model: trained model
        observed_data: observed sine wave [data_dim]
        num_samples: number of posterior samples
    
    Returns:
        samples: posterior samples [num_samples, param_dim]
        statistics: dict with mean, median, std, quantiles
    """
    model.eval()

    data_tensor = torch.FloatTensor(observed_data).unsqueeze(0)

    with torch.no_grad():
        samples = model.sample_posterior(data_tensor, num_samples=num_samples)
        samples = samples.numpy()

    if samples.shape[1] == 1:
        samples = samples.flatten()
        statistics = {
            'mean': np.mean(samples),
            'median': np.median(samples),
            'std': np.std(samples),
            'q05': np.percentile(samples, 5),
            'q95': np.percentile(samples, 95),
        }
    else:
        statistics = None
    
    return samples, statistics


################### Data Representation Functions ###################

def to_frequency_domain(waveform, sample_rate=4096, f_low=20.0, f_high=500.0):
    """
    Convert a time-domain waveform to its amplitude spectrum in a useful frequency band.

    Parameters
    ----------
    waveform : np.ndarray
        1D time-domain waveform
    sample_rate : float
        Sampling rate in Hz. Default: 4096
    f_low : float
        Lower frequency cutoff in Hz. Default: 20.0
    f_high : float
        Upper frequency cutoff in Hz. Default: 500.0

    Returns
    -------
    freqs : np.ndarray
        Frequency array (Hz) within [f_low, f_high]
    amplitudes : np.ndarray
        Amplitude spectrum |FFT(h)| in that band
    """
    data = np.asarray(waveform, dtype=np.float64)
    n = len(data)
    freqs_full = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    fft_vals = np.fft.rfft(data)
    amplitudes_full = np.abs(fft_vals)

    mask = (freqs_full >= f_low) & (freqs_full <= f_high)
    return freqs_full[mask], amplitudes_full[mask]


def to_spectrogram(waveform, sample_rate=4096, nperseg=256, noverlap=224):
    """
    Compute a time-frequency spectrogram (STFT) of a waveform.

    Parameters
    ----------
    waveform : np.ndarray
        1D time-domain waveform
    sample_rate : float
        Sampling rate in Hz. Default: 4096
    nperseg : int
        FFT segment length. Default: 256
    noverlap : int
        Overlap between segments. Default: 224

    Returns
    -------
    times : np.ndarray
        Time bins (seconds)
    freqs : np.ndarray
        Frequency bins (Hz)
    Sxx : np.ndarray
        Spectrogram power, shape (len(freqs), len(times))
    """
    from scipy.signal import spectrogram as scipy_spectrogram

    data = np.asarray(waveform, dtype=np.float64)
    freqs, times, Sxx = scipy_spectrogram(
        data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
        window='hann', scaling='spectrum'
    )
    return times, freqs, Sxx


def to_wavelet_scalogram(waveform, sample_rate=4096, num_scales=64, f_low=20.0, f_high=500.0):
    """
    Compute a continuous wavelet transform (Morlet) scalogram.

    Parameters
    ----------
    waveform : np.ndarray
        1D time-domain waveform
    sample_rate : float
        Sampling rate in Hz. Default: 4096
    num_scales : int
        Number of wavelet scales (frequency resolution). Default: 64
    f_low : float
        Lowest frequency of interest in Hz. Default: 20.0
    f_high : float
        Highest frequency of interest in Hz. Default: 500.0

    Returns
    -------
    freqs : np.ndarray
        Pseudo-frequencies for each scale (Hz), shape (num_scales,)
    times : np.ndarray
        Time array (seconds), shape (N,)
    coefficients : np.ndarray
        Scalogram |CWT|^2, shape (num_scales, N)
    """
    from scipy.signal import fftconvolve

    data = np.asarray(waveform, dtype=np.float64)
    n = len(data)

    # Morlet wavelet center angular frequency
    w0 = 6.0

    # Map desired frequencies to scales: scale = w0 * sample_rate / (2*pi*freq)
    target_freqs = np.geomspace(f_low, f_high, num_scales)
    scales = w0 * sample_rate / (2.0 * np.pi * target_freqs)

    # Compute CWT with Morlet wavelet via FFT convolution
    coefficients = np.zeros((num_scales, n), dtype=np.complex128)
    for i, s in enumerate(scales):
        # Build Morlet wavelet: psi(t) = pi^(-1/4) * exp(iw0*t) * exp(-t^2/2)
        M = min(int(10 * s), n)
        half = M // 2
        t_wav = np.arange(-half, half + 1) / s
        wavelet = (np.pi ** -0.25) * np.exp(1j * w0 * t_wav) * np.exp(-0.5 * t_wav ** 2)
        wavelet = wavelet / np.sqrt(s)  # L2 normalize for scale

        conv = fftconvolve(data, np.conj(wavelet[::-1]), mode='same')
        coefficients[i, :] = conv

    power = np.abs(coefficients) ** 2
    times = np.arange(n) / sample_rate

    return target_freqs, times, power
