
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
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.psd import aLIGOZeroDetHighPower, welch, interpolate
from pycbc.noise import noise_from_psd
from pycbc.types import TimeSeries
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.catalog import Merger
from torch.utils.data import Subset
from scipy.signal import welch as scipy_welch
import matplotlib.pyplot as plt


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


        
    return {
        'success': True,
        'detectors': detector_signals,
        'params': params
    }


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
                        add_noise: bool = True,
                        normalize_waveforms: bool = True) -> Dict:
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
    normalize_waveforms : bool
        Normalize each waveform by its maximum absolute value. Default: True.
        This scales waveforms to [-1, 1] range, helping the network learn better.
        If True, also scales by sqrt(2) to bring waveforms to ~[-0.7, 0.7] range.

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
    if num_workers is None:
        num_workers = 1
    
    # Set default detectors
    if detectors is None:
        detectors = ['H1', 'L1']
    
    # Check which sky parameters are provided
    sky_params_provided = {
        'ra': 'ra' in config,
        'dec': 'dec' in config,
        'polarization': 'polarization' in config
    }
    
    if any(sky_params_provided.values()):
        print(f"Generating {num_samples} waveforms with projection to {detectors}")
        print(f"  Sky parameters: ra={'provided' if sky_params_provided['ra'] else 'default (0.0)'}, "
              f"dec={'provided' if sky_params_provided['dec'] else 'default (π/2)'}, "
              f"psi={'provided' if sky_params_provided['polarization'] else 'default (0.0)'}")
    else:
        print(f"Generating {num_samples} waveforms with projection to {detectors}")
        print(f"  Using default sky location: ra=0.0, dec=π/2 (north pole), psi=0.0")
    
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

    # Pre-allocate arrays (all signals are already at target_length)
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

    # Normalize waveforms if requested
    if normalize_waveforms:
        print(f"  Normalizing waveforms...")
        for i in range(num_success):
            for j in range(num_detectors):
                # Get maximum absolute value for this waveform
                max_val = np.abs(signal_array[i, j]).max()
                # Avoid division by zero (empty waveforms)
                if max_val > 0:
                    signal_array[i, j] /= max_val
                    # Also scale by sqrt(2) to approximately normalize power
                    signal_array[i, j] /= np.sqrt(2)

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
            'normalize_waveforms': normalize_waveforms,
        }
    }


def whiten_waveform(strain_timeseries, psd_freqs: np.ndarray, psd_values: np.ndarray, ifo: str):
    """
    Whiten a waveform using the PSD to remove colored noise.
    
    This follows the exact PyCBC GW150914 tutorial approach:
    white_strain = (strain_fft / sqrt(psd)).to_timeseries()
    
    Reference: https://pycbc.org/pycbc/latest/html/gw150914.html#plotting-the-whitened-strain
    
    Parameters
    ----------
    strain_timeseries : TimeSeries or array-like
        Time-domain waveform to whiten
    psd_freqs : np.ndarray
        Frequency array from PSD calculation
    psd_values : np.ndarray
        PSD values corresponding to frequencies
    
    Returns
    -------
    whitened : TimeSeries
        Whitened waveform in time domain
    """
    if isinstance(strain_timeseries, TimeSeries):
        strain = strain_timeseries
    else:
        strain = TimeSeries(strain_timeseries, delta_t=1.0/4096)
    
    # Calculate raw Welch PSD
    psd_welch = welch(strain)
    
    # Interpolate to smooth frequency grid
    psd_interp = interpolate(psd_welch, 1.0 / strain.duration)

    # Whiten: divide by sqrt(PSD) and convert back to time domain
    white_strain = (strain.to_frequencyseries() / (psd_interp ** 0.5)).to_timeseries()
    
    # remove some of the high and low
    #smooth = highpass_fir(white_strain, 35, 8)
    #smooth = lowpass_fir(smooth, 300, 8)

    # time shift and flip L1
    #if ifo == 'L1':
    #    smooth *= -1
    #    smooth.roll(int(.007 / smooth.delta_t))


    return white_strain


def compute_psd_from_waveform(waveform: np.ndarray, delta_t: float) -> tuple:
    """
    Compute Power Spectral Density from time-domain waveform using FFT.
    
    Parameters
    ----------
    waveform : np.ndarray
        Time-domain waveform
    delta_t : float
        Time resolution (seconds)
    
    Returns
    -------
    frequencies : np.ndarray
        Frequency array (Hz)
    psd : np.ndarray
        Power spectral density (one-sided)
    """
    # Compute FFT
    fft = np.fft.rfft(waveform)
    
    # Power spectrum (magnitude squared, normalized)
    power = np.abs(fft) ** 2
    
    # Frequency array
    n = len(waveform)
    freqs = np.fft.rfftfreq(n, delta_t)
    
    # Normalize by sampling rate to get power spectral density
    # (power per unit frequency)
    psd = power / (1.0 / delta_t)
    
    return freqs, psd


def compute_psd_pycbc(waveform_or_timeseries, delta_t: float = None) -> tuple:
    """
    Compute Power Spectral Density using PyCBC's Welch method with interpolation.
    
    This follows the standard PyCBC approach from their GW150914 tutorial:
    https://pycbc.org/pycbc/latest/html/gw150914.html#plotting-the-whitened-strain
    
    Uses Welch's method to estimate PSD, then interpolates to a smooth frequency grid.
    This is the method recommended in PyCBC documentation and used in real LIGO analysis.
    
    Parameters
    ----------
    waveform_or_timeseries : np.ndarray or pycbc.types.TimeSeries
        Input strain data. Can be numpy array or PyCBC TimeSeries object.
    delta_t : float, optional
        Time resolution (seconds). Required if input is numpy array.
        Ignored if input is TimeSeries (uses TimeSeries.delta_t).
    
    Returns
    -------
    frequencies : np.ndarray
        Frequency array (Hz)
    psd : np.ndarray
        Power spectral density values
    
    Notes
    -----
    This uses the official PyCBC approach:
    1. Compute raw Welch PSD (may be sparse)
    2. Interpolate to smooth frequency grid with delta_f = 1.0 / duration
    3. This gives a well-defined PSD even for short signals
    
    Reference: https://pycbc.org/pycbc/latest/html/gw150914.html#plotting-the-whitened-strain
    """
    from pycbc.psd import interpolate
    
    # Convert to TimeSeries if needed
    if isinstance(waveform_or_timeseries, TimeSeries):
        strain = waveform_or_timeseries
    else:
        strain = TimeSeries(waveform_or_timeseries, delta_t=delta_t)
    
    # Calculate raw Welch PSD
    psd_welch = welch(strain)
    
    # Interpolate to smooth frequency grid
    psd_interp = interpolate(psd_welch, 1.0 / strain.duration)
    
    # Convert to numpy arrays
    delta_f = psd_interp.delta_f
    frequencies = np.arange(len(psd_interp)) * delta_f
    psd_values = np.array(psd_interp)
    
    return frequencies, psd_values




def plot_example_waveforms(result: Dict, figsize: tuple = (36, 8)):
    """
    Plot single example waveform with FFT, PSD, whitened timestream, whitened PSD, and clean waveform.
    
    For each detector, shows:
    - Column 1: Time-domain waveform (with noise)
    - Column 2: PSD using raw FFT method
    - Column 3: PSD using PyCBC Welch method
    - Column 4: Whitened timestream
    - Column 5: PSD of whitened data
    - Column 6: Waveform without noise
    
    Layout:
    - Row 1: H1 detector
    - Row 2: L1 detector
    
    Parameters
    ----------
    result : dict
        Output from pycbc_data_generator containing 'train_loader' and 'metadata'
    figsize : tuple
        Figure size (width, height). Default: (36, 8)))
    """
    train_loader = result['train_loader']
    metadata = result['metadata']
    param_names = metadata['parameter_names']
    channels = metadata['channels']  # e.g., ['H1', 'L1']
    time_resolution = metadata['time_resolution']
    f_lower = metadata.get('f_lower', 40.0)
    
    # Get first batch
    waveforms, params = next(iter(train_loader))
    
    # Use only first example
    ex_idx = 0
    
    # Create subplots: 2 rows (H1, L1) x 6 columns (time, FFT, PyCBC, whitened, whitened PSD, clean)
    fig, axes = plt.subplots(2, 6, figsize=figsize)
    
    # Time axis in seconds
    waveform_length = metadata['waveform_shape'][1]
    time_axis = np.arange(waveform_length) * time_resolution
    
    # Get parameters for this example
    param_values = params[ex_idx].numpy()
    param_str = '\n'.join([f'{name}: {val:.4f}' for name, val in zip(param_names, param_values)])
    
    for ch_idx, channel in enumerate(channels):
        if ch_idx == 0:
            detector_name = "Hanford (H1)"
        else:
            detector_name = "Livingston (L1)"
        
        # Get waveform data
        waveform = waveforms[ex_idx, ch_idx, :].numpy()
        
        # ===== COLUMN 1: Time-domain waveform =====
        ax_time = axes[ch_idx, 0]
        ax_time.plot(time_axis, waveform, linewidth=0.8, color='steelblue')
        ax_time.set_xlabel('Time (s)', fontsize=10)
        ax_time.set_ylabel('Strain', fontsize=10)
        ax_time.set_title(f'{detector_name} - Time Domain', fontsize=11, fontweight='bold')
        ax_time.grid(True, alpha=0.3)
        
        # Add parameters as text (only on first detector)
        if ch_idx == 0:
            ax_time.text(0.02, 0.98, param_str, 
                       transform=ax_time.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontsize=8, family='monospace')
        
        # ===== COLUMN 2: Raw FFT PSD =====
        ax_fft = axes[ch_idx, 1]
        freqs_fft, psd_fft = compute_psd_from_waveform(waveform, time_resolution)
        mask_fft = freqs_fft >= f_lower
        ax_fft.loglog(freqs_fft[mask_fft], psd_fft[mask_fft], linewidth=0.8, color='darkred', label='FFT')
        ax_fft.set_xlabel('Frequency (Hz)', fontsize=10)
        ax_fft.set_ylabel('PSD', fontsize=10)
        ax_fft.set_title(f'{detector_name} - FFT Method', fontsize=11, fontweight='bold')
        ax_fft.grid(True, alpha=0.3, which='both')
        ax_fft.set_xlim([max(1, f_lower/2), freqs_fft.max()])
        
        # ===== COLUMN 3: PyCBC Welch PSD =====
        ax_pycbc = axes[ch_idx, 2]
        freqs_pycbc, psd_pycbc = compute_psd_pycbc(waveform, delta_t=time_resolution)
        mask_pycbc = freqs_pycbc >= f_lower
        ax_pycbc.loglog(freqs_pycbc[mask_pycbc], np.abs(psd_pycbc[mask_pycbc]), linewidth=0.8, color='darkgreen', label='Welch')
        
        ax_pycbc.set_xlabel('Frequency (Hz)', fontsize=10)
        ax_pycbc.set_ylabel('PSD', fontsize=10)
        ax_pycbc.set_title(f'{detector_name} - PyCBC Welch Method', fontsize=11, fontweight='bold')
        ax_pycbc.grid(True, alpha=0.3, which='both')
        ax_pycbc.set_xlim([max(1, f_lower/2), freqs_pycbc.max()])
        
        # ===== COLUMN 4: Whitened Timestream =====
        ax_white = axes[ch_idx, 3]
        whitened = whiten_waveform(waveform, freqs_pycbc, psd_pycbc, ifo =channel)
        white_data = whitened.numpy() if hasattr(whitened, 'numpy') else np.array(whitened)
        ax_white.plot(time_axis, white_data, linewidth=0.8, color='purple')
        ax_white.set_xlabel('Time (s)', fontsize=10)
        ax_white.set_ylabel('Whitened Strain', fontsize=10)
        ax_white.set_title(f'{detector_name} - Whitened', fontsize=11, fontweight='bold')
        ax_white.grid(True, alpha=0.3)        
        # ===== COLUMN 5: PSD of Whitened Data =====
        ax_white_psd = axes[ch_idx, 4]
        freqs_white, psd_white = compute_psd_pycbc(white_data, delta_t=time_resolution)
        mask_white = freqs_white >= f_lower
        ax_white_psd.loglog(freqs_white[mask_white], psd_white[mask_white], linewidth=0.8, color='orange', label='Whitened PSD')
        ax_white_psd.set_xlabel('Frequency (Hz)', fontsize=10)
        ax_white_psd.set_ylabel('PSD', fontsize=10)
        ax_white_psd.set_title(f'{detector_name} - Whitened PSD', fontsize=11, fontweight='bold')
        ax_white_psd.grid(True, alpha=0.3, which='both')
        ax_white_psd.set_xlim([max(1, f_lower/2), freqs_white.max()])
        
        # ===== COLUMN 6: Clean Waveform (no noise) =====
        ax_clean = axes[ch_idx, 5]
        hp, hc = get_td_waveform(
            approximant='IMRPhenomXP',
            mass1=param_values[0],
            mass2=param_values[1],
            spin1z=param_values[2],
            spin2z=param_values[3],
            f_lower=f_lower,
            delta_t=time_resolution
        )
        detector = Detector(channel)
        clean_signal = detector.project_wave(hp, hc, 0.0, np.pi/2, 0.0, method='lal')
        signal_len = len(clean_signal)
        target_length = len(waveform)
        if signal_len >= target_length:
            clean_signal = clean_signal[-target_length:]
        else:
            padded = TimeSeries(np.zeros(target_length, dtype=clean_signal.dtype),
                               delta_t=clean_signal.delta_t,
                               epoch=clean_signal.start_time - (target_length - signal_len) * clean_signal.delta_t)
            padded[-signal_len:] = clean_signal
            clean_signal = padded
        clean_data = clean_signal.numpy() if hasattr(clean_signal, 'numpy') else np.array(clean_signal)
        ax_clean.plot(time_axis, clean_data, linewidth=0.8, color='green')
        ax_clean.set_xlabel('Time (s)', fontsize=10)
        ax_clean.set_ylabel('Strain', fontsize=10)
        ax_clean.set_title(f'{detector_name} - Clean (No Noise)', fontsize=11, fontweight='bold')
        ax_clean.grid(True, alpha=0.3)    
    plt.tight_layout()
    return fig


def plot_pycbc_tutorial_method(result: Dict, figsize: tuple = (16, 14)):
    """
    Plot using pure PyCBC tutorial method from GW150914 example.
    
    Follows exactly: https://pycbc.org/pycbc/latest/html/gw150914.html#plotting-the-whitened-strain
    
    For each detector, shows:
    - Row 1: Original strain (with noise)
    - Row 2: Calculated PSD
    - Row 3: Whitened strain
    - Row 4: PSD of whitened strain
    
    Parameters
    ----------
    result : dict
        Output from pycbc_data_generator containing 'train_loader' and 'metadata'
    figsize : tuple
        Figure size (width, height). Default: (16, 14)
    """
    from pycbc.filter import highpass_fir, lowpass_fir
    
    train_loader = result['train_loader']
    metadata = result['metadata']
    channels = metadata['channels']
    time_resolution = metadata['time_resolution']
    f_lower = metadata.get('f_lower', 40.0)
    
    waveforms, params = next(iter(train_loader))
    
    # Use first example
    waveform_length = metadata['waveform_shape'][1]
    time_axis = np.arange(waveform_length) * time_resolution
    
    # Create subplots: 4 rows (original, PSD, whitened, whitened PSD) x 2 columns (H1, L1)
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    
    for ch_idx, channel in enumerate(channels):
        waveform = waveforms[0, ch_idx, :].numpy()
        
        # Convert to TimeSeries for PyCBC methods
        strain = TimeSeries(waveform, delta_t=time_resolution)
        
        # Step 1: High-pass filter to remove low frequency content (optional)
        filtered = highpass_fir(strain, f_lower, 8)
        
        # Step 2: Calculate PSD using Welch + interpolate (pure PyCBC method)
        psd_welch = welch(filtered)
        psd = interpolate(psd_welch, 1.0 / filtered.duration)
        
        # Step 3: Whiten the strain (pure PyCBC method)
        white_strain = (filtered.to_frequencyseries() / (psd ** 0.5)).to_timeseries()
        
        # Step 4: Apply optional smoothing filters
        smooth = highpass_fir(white_strain, 35, 8)
        smooth = lowpass_fir(smooth, 300, 8)
        
        detector_name = "Hanford (H1)" if ch_idx == 0 else "Livingston (L1)"
        
        # Row 1: Original strain
        ax1 = axes[0, ch_idx]
        ax1.plot(time_axis, filtered.numpy(), linewidth=0.8, color='steelblue')
        ax1.set_ylabel('Strain', fontsize=10)
        ax1.set_title(f'{detector_name} - Original (HP filtered)', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Row 2: PSD
        ax2 = axes[1, ch_idx]
        psd_freqs = np.arange(len(psd)) * psd.delta_f
        mask = psd_freqs >= f_lower
        ax2.loglog(psd_freqs[mask], np.abs(psd[mask]), linewidth=0.8, color='darkred')
        ax2.set_ylabel('PSD', fontsize=10)
        ax2.set_title(f'{detector_name} - PSD (Welch+Interp)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        
        # Row 3: Whitened and smoothed
        ax3 = axes[2, ch_idx]
        ax3.plot(time_axis[:len(smooth)], smooth.numpy(), linewidth=0.8, color='purple')
        ax3.set_ylabel('Whitened Strain', fontsize=10)
        ax3.set_title(f'{detector_name} - Whitened & Smoothed', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Row 4: PSD of whitened strain
        ax4 = axes[3, ch_idx]
        freqs_white, psd_white = compute_psd_pycbc(smooth.numpy(), delta_t=time_resolution)
        mask_white = freqs_white >= f_lower
        ax4.loglog(freqs_white[mask_white], psd_white[mask_white], linewidth=0.8, color='orange')
        ax4.set_xlabel('Frequency (Hz)', fontsize=10)
        ax4.set_ylabel('PSD', fontsize=10)
        ax4.set_title(f'{detector_name} - Whitened PSD', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    """Example usage: Generate waveforms and plot them with noise spectra."""
    
    print("=" * 80)
    print("PyCBC Data Generator - Example Waveforms with Noise Spectra")
    print("=" * 80)
    
    # Configuration: Generate simple binaries with varied masses and spins
    config = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
        'spin2z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
    }
    
    # Generate dataset
    print("\nGenerating 20 waveforms for visualization...")
    result = pycbc_data_generator(
        config,
        num_samples=20,
        batch_size=32,
        num_workers=2,
        normalize_waveforms=True,
        detectors=['H1', 'L1'],
        add_noise=True,
        signal_length=2.0
    )
    
    # Plot example waveforms with FFT and PyCBC PSD comparison
    print("\nPlotting waveforms with FFT, PyCBC Welch PSD, whitened timestream, whitened PSD, and clean waveform...")
    print("  Layout: Time Domain | FFT PSD | PyCBC Welch PSD | Whitened | Whitened PSD | Clean") 
    print("          H1          | H1      | H1              | H1       | H1           | H1")
    print("          L1          | L1      | L1              | L1       | L1           | L1")
    fig = plot_example_waveforms(result, figsize=(18, 8))
    
    # Save figure
    output_file = '/workspace/example_waveforms.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_file}")
    
    # Show first batch statistics
    print("\nFirst batch statistics:")
    train_loader = result['train_loader']
    waveforms, params = next(iter(train_loader))
    print(f"  Waveform batch shape: {waveforms.shape}")
    print(f"  Parameters batch shape: {params.shape}")
    print(f"  H1 amplitude range: [{waveforms[:, 0, :].min():.4f}, {waveforms[:, 0, :].max():.4f}]")
    print(f"  L1 amplitude range: [{waveforms[:, 1, :].min():.4f}, {waveforms[:, 1, :].max():.4f}]")
    print(f"\n  Example parameters (first sample in batch):")
    metadata = result['metadata']
    for i, name in enumerate(metadata['parameter_names']):
        print(f"    {name}: {params[0, i]:.4f}")
    
    # Generate and process waveforms using pure PyCBC tutorial method
    print("\n" + "=" * 80)
    print("Generating GW150914 strain using Merger class...")
    print("=" * 80)
    
    # Load GW150914 strain directly using Merger
    detectors = ['H1', 'L1']
    strains = {}
    
    for ifo in detectors:
        h1 = Merger("GW150914").strain(ifo)
        h1 = highpass_fir(h1, 15, 8)
        strains[ifo] = h1
        print(f"{ifo}: Loaded GW150914 strain with {len(h1)} samples")
    
    # Create figure for tutorial method results
    fig_tutorial_gen, axes_tutorial = plt.subplots(4, 2, figsize=(16, 14))
    
    for col, ifo in enumerate(detectors):
        strain = strains[ifo]
        
        # Step 1: High-pass filter
        filtered = highpass_fir(strain, 20.0, 8)
        
        # Step 2: Calculate PSD - use Welch with interpolation (handles short signals better now)
        psd_welch = welch(filtered)
        psd = interpolate(psd_welch, 1.0 / filtered.duration)
        
        # Resize PSD to match frequency series length
        freq_series = filtered.to_frequencyseries()
        psd.resize(len(freq_series))
        
        # Step 3: Whiten the strain
        white_strain = (freq_series / (psd ** 0.5)).to_timeseries()
        
        # Step 4: Apply smoothing filters
        smooth = highpass_fir(white_strain, 35, 8)
        smooth = lowpass_fir(smooth, 300, 8)
        
        # Plot results
        time_axis = np.arange(len(filtered)) * filtered.delta_t
        
        # Row 1: Original filtered strain
        axes_tutorial[0, col].plot(time_axis, filtered.numpy(), linewidth=0.8, color='steelblue')
        axes_tutorial[0, col].set_ylabel('Strain', fontsize=10)
        axes_tutorial[0, col].set_title(f'{ifo} - High-Pass Filtered', fontsize=11, fontweight='bold')
        axes_tutorial[0, col].grid(True, alpha=0.3)
        
        # Row 2: PSD
        psd_freqs = np.arange(len(psd)) * psd.delta_f
        mask = psd_freqs >= 20.0
        axes_tutorial[1, col].loglog(psd_freqs[mask], np.abs(psd[mask]), linewidth=0.8, color='darkred')
        axes_tutorial[1, col].set_ylabel('PSD', fontsize=10)
        axes_tutorial[1, col].set_title(f'{ifo} - PSD (Welch+Interpolate)', fontsize=11, fontweight='bold')
        axes_tutorial[1, col].grid(True, alpha=0.3, which='both')
        
        # Row 3: Whitened and smoothed
        time_smooth = np.arange(len(smooth)) * smooth.delta_t
        axes_tutorial[2, col].plot(time_smooth, smooth.numpy(), linewidth=0.8, color='purple')
        axes_tutorial[2, col].set_ylabel('Whitened Strain', fontsize=10)
        axes_tutorial[2, col].set_title(f'{ifo} - Whitened & Smoothed', fontsize=11, fontweight='bold')
        axes_tutorial[2, col].grid(True, alpha=0.3)
        
        # Row 4: PSD of whitened strain
        axes_tutorial[3, col].set_xlabel('Frequency (Hz)', fontsize=10)
        freqs_white, psd_white = compute_psd_pycbc(smooth.numpy(), delta_t=smooth.delta_t)
        mask_white = freqs_white >= 20.0
        axes_tutorial[3, col].loglog(freqs_white[mask_white], psd_white[mask_white], linewidth=0.8, color='orange')
        axes_tutorial[3, col].set_ylabel('PSD', fontsize=10)
        axes_tutorial[3, col].set_title(f'{ifo} - Whitened PSD', fontsize=11, fontweight='bold')
        axes_tutorial[3, col].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_tutorial_gen = '/workspace/pycbc_tutorial_generated.png'
    fig_tutorial_gen.savefig(output_tutorial_gen, dpi=150, bbox_inches='tight')
    print(f"\n✓ Tutorial-method generated waveforms saved to {output_tutorial_gen}")


