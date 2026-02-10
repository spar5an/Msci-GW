"""
Signal processing for real GW detector strain.

Real data requires additional processing not needed for simulations:
- Notch filtering for power line harmonics (60 Hz)
- Data quality checking
- Handling missing data/glitches
- PSD estimation from off-source data
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.signal.windows import tukey
from scipy.signal import iirnotch, filtfilt
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, Subset

from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.psd import welch, interpolate


def notch_filter_powerlines(
    strain: np.ndarray,
    sample_rate: int = 4096,
    harmonics: List[float] = None,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filters to remove power line harmonics.

    Parameters
    ----------
    strain : np.ndarray
        Input detector strain (1D array)
    sample_rate : int
        Sample rate in Hz. Default: 4096
    harmonics : List[float], optional
        Frequencies to notch. Default: [60, 120, 180] Hz (US power grid)
    quality_factor : float
        Quality factor for notch filters. Higher = narrower notch.
        Default: 30.0

    Returns
    -------
    np.ndarray
        Notch-filtered strain
    """
    if harmonics is None:
        harmonics = [60.0, 120.0, 180.0]  # US power grid harmonics

    filtered = strain.copy()

    for freq in harmonics:
        # Skip if frequency is above Nyquist
        if freq >= sample_rate / 2:
            continue

        # Design notch filter
        b, a = iirnotch(freq, quality_factor, sample_rate)

        # Apply filter (forward-backward for zero phase)
        filtered = filtfilt(b, a, filtered)

    return filtered


def estimate_psd_from_offsource(
    strain: TimeSeries,
    coalescence_time: float,
    psd_duration: float = 16.0,
    psd_offset: float = 4.0,
    method: str = 'welch'
) -> FrequencySeries:
    """
    Estimate PSD from off-source data (away from the signal).

    For real data, PSD should be estimated from data NOT containing
    the signal to avoid signal leakage into the noise estimate.

    Parameters
    ----------
    strain : TimeSeries
        Full detector strain (should be longer than signal window)
    coalescence_time : float
        GPS time of coalescence
    psd_duration : float
        Duration of data to use for PSD. Default: 16.0 seconds
    psd_offset : float
        Gap between PSD region and signal. Default: 4.0 seconds
    method : str
        'welch' (default) or 'median'

    Returns
    -------
    FrequencySeries
        Estimated PSD
    """
    start_time = float(strain.start_time)
    end_time = float(strain.end_time)

    # Use data BEFORE the signal for PSD estimation
    psd_end_time = coalescence_time - psd_offset
    psd_start_time = psd_end_time - psd_duration

    # Ensure we're within the strain data
    if psd_start_time < start_time:
        # Not enough data before signal, try after
        psd_start_time = coalescence_time + psd_offset
        psd_end_time = psd_start_time + psd_duration

        if psd_end_time > end_time:
            # Not enough data, use what we have
            print("Warning: Limited off-source data for PSD estimation")
            psd_start_time = start_time
            psd_end_time = min(start_time + psd_duration, coalescence_time - 1.0)

    # Extract off-source segment
    offsource = strain.time_slice(psd_start_time, psd_end_time)

    # Compute PSD
    n_samples = len(offsource)
    if n_samples >= 4096:
        seg_len = 4096
        seg_stride = 2048
    elif n_samples >= 1024:
        seg_len = max(256, n_samples // 4)
        seg_stride = seg_len // 2
    else:
        seg_len = max(64, n_samples // 4)
        seg_stride = seg_len // 2

    psd = welch(offsource, seg_len=seg_len, seg_stride=seg_stride)

    return psd


def whiten_real_strain(
    strain: np.ndarray,
    sample_rate: int = 4096,
    psd: FrequencySeries = None,
    f_lower: float = 20.0,
    apply_bandpass: bool = True,
    bandpass_low: float = 35.0,
    bandpass_high: float = 300.0,
    apply_tukey: bool = True,
    tukey_alpha: float = 0.1,
    tukey_side: str = 'left'
) -> Tuple[np.ndarray, FrequencySeries]:
    """
    Whiten real detector strain.

    Similar to JHPY.whiten_waveform() but optimized for real data:
    - Uses pre-computed off-source PSD when provided
    - Handles real data artifacts better

    Parameters
    ----------
    strain : np.ndarray
        Input detector strain (1D array)
    sample_rate : int
        Sample rate in Hz. Default: 4096
    psd : FrequencySeries, optional
        Pre-computed PSD. If None, computed from strain itself.
    f_lower : float
        Lower frequency cutoff. Default: 20.0 Hz
    apply_bandpass : bool
        Apply bandpass filter. Default: True
    bandpass_low : float
        Low frequency cutoff for bandpass. Default: 35.0 Hz
    bandpass_high : float
        High frequency cutoff for bandpass. Default: 300.0 Hz
    apply_tukey : bool
        Apply Tukey window to prevent edge effects. Default: True
    tukey_alpha : float
        Tukey window alpha parameter. Default: 0.1
    tukey_side : str
        Which side to taper ('left', 'right', 'both'). Default: 'left'

    Returns
    -------
    Tuple[np.ndarray, FrequencySeries]
        Whitened strain as numpy array, and the PSD used
    """
    delta_t = 1.0 / sample_rate

    # Convert to TimeSeries
    ts = TimeSeries(strain.astype(np.float64), delta_t=delta_t)

    # Compute PSD if not provided
    if psd is None:
        n_samples = len(ts)
        if n_samples >= 4096:
            seg_len = 4096
            seg_stride = 2048
        elif n_samples >= 1024:
            seg_len = max(256, n_samples // 4)
            seg_stride = seg_len // 2
        else:
            seg_len = max(64, n_samples // 4)
            seg_stride = seg_len // 2

        psd = welch(ts, seg_len=seg_len, seg_stride=seg_stride)

    # Interpolate PSD to smooth frequency grid
    psd_interp = interpolate(psd, 1.0 / ts.duration)

    # Convert strain to frequency domain
    freq_series = ts.to_frequencyseries()

    # Resize PSD to match frequency series length
    psd_interp.resize(len(freq_series))

    # Add epsilon to avoid division by zero
    psd_array = np.array(psd_interp)
    epsilon = 1e-40
    psd_array[psd_array <= 0] = epsilon
    psd_safe = FrequencySeries(psd_array, delta_f=psd_interp.delta_f, epoch=psd_interp.epoch)

    # Whiten: divide by sqrt(PSD) in frequency domain
    white_strain = (freq_series / (psd_safe ** 0.5)).to_timeseries()

    # Apply Tukey window before bandpass filtering
    if apply_tukey and apply_bandpass:
        n = len(white_strain)
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
            raise ValueError(f"tukey_side must be 'left', 'right', or 'both'")

        white_strain = TimeSeries(np.array(white_strain) * window, delta_t=delta_t)

    # Apply bandpass filtering
    if apply_bandpass:
        white_strain = highpass_fir(white_strain, bandpass_low, 8)
        white_strain = lowpass_fir(white_strain, bandpass_high, 8)

    return np.array(white_strain), psd


def check_data_quality(
    strain: np.ndarray,
    sample_rate: int = 4096,
    max_gap_duration: float = 0.1,
    saturation_threshold: float = 1e-16
) -> Dict:
    """
    Check data quality flags and report issues.

    Parameters
    ----------
    strain : np.ndarray
        Input strain data
    sample_rate : int
        Sample rate in Hz. Default: 4096
    max_gap_duration : float
        Maximum allowed data gap in seconds. Default: 0.1
    saturation_threshold : float
        Threshold for detecting saturation. Default: 1e-16

    Returns
    -------
    Dict containing:
        - is_valid: bool
        - has_gaps: bool
        - gap_locations: List of (start_idx, end_idx) tuples
        - saturation_detected: bool
        - glitch_count: int (number of potential glitches)
        - quality_score: float (0-1)
    """
    result = {
        'is_valid': True,
        'has_gaps': False,
        'gap_locations': [],
        'saturation_detected': False,
        'glitch_count': 0,
        'quality_score': 1.0
    }

    # Check for NaN or Inf values
    if np.any(np.isnan(strain)) or np.any(np.isinf(strain)):
        result['is_valid'] = False
        result['quality_score'] = 0.0
        return result

    # Check for data gaps (zeros or constant values)
    zero_mask = np.abs(strain) < saturation_threshold
    if np.any(zero_mask):
        # Find contiguous regions of zeros
        diff = np.diff(zero_mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if zero_mask[0]:
            starts = np.concatenate([[0], starts])
        if zero_mask[-1]:
            ends = np.concatenate([ends, [len(strain)]])

        gap_samples = int(max_gap_duration * sample_rate)
        for start, end in zip(starts, ends):
            if end - start > gap_samples:
                result['has_gaps'] = True
                result['gap_locations'].append((start, end))

    # Check for potential glitches (sudden large amplitude changes)
    diff = np.abs(np.diff(strain))
    threshold = 10 * np.std(diff)
    glitch_mask = diff > threshold
    result['glitch_count'] = np.sum(glitch_mask)

    # Compute quality score
    quality_penalties = 0.0
    if result['has_gaps']:
        quality_penalties += 0.3
    if result['glitch_count'] > 0:
        quality_penalties += min(0.3, result['glitch_count'] * 0.05)

    result['quality_score'] = max(0.0, 1.0 - quality_penalties)
    result['is_valid'] = result['quality_score'] > 0.5

    return result


def interpolate_gaps(
    strain: np.ndarray,
    gap_locations: List[Tuple[int, int]],
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate across small data gaps.

    Parameters
    ----------
    strain : np.ndarray
        Input strain with gaps
    gap_locations : List[Tuple]
        List of (start_idx, end_idx) for gaps
    method : str
        'linear', 'cubic', or 'noise'. Default: 'linear'

    Returns
    -------
    np.ndarray
        Strain with gaps interpolated
    """
    result = strain.copy()

    for start, end in gap_locations:
        if start <= 0 or end >= len(strain):
            continue

        if method == 'linear':
            # Linear interpolation
            result[start:end] = np.linspace(
                strain[start-1], strain[end],
                end - start
            )
        elif method == 'noise':
            # Fill with Gaussian noise matching local statistics
            local_std = np.std(strain[max(0, start-100):start])
            result[start:end] = np.random.normal(0, local_std, end - start)
        elif method == 'cubic':
            from scipy.interpolate import interp1d
            # Use surrounding points for cubic interpolation
            x_known = np.concatenate([
                np.arange(max(0, start-10), start),
                np.arange(end, min(len(strain), end+10))
            ])
            y_known = strain[x_known]
            f = interp1d(x_known, y_known, kind='cubic', fill_value='extrapolate')
            x_gap = np.arange(start, end)
            result[start:end] = f(x_gap)

    return result


def process_real_strain(
    strain: np.ndarray,
    sample_rate: int = 4096,
    coalescence_time: float = None,
    strain_start_time: float = None,
    signal_length: float = 2.0,
    apply_notch: bool = True,
    apply_whitening: bool = True,
    apply_bandpass: bool = True,
    normalize_scale: float = None,
    tukey_alpha: float = 0.1
) -> Dict:
    """
    Apply full processing pipeline to real detector strain.

    Pipeline order:
    1. Apply notch filters for power lines
    2. Estimate PSD from off-source data (if possible)
    3. Whiten using PSD
    4. Apply bandpass filter
    5. Optionally normalize

    Parameters
    ----------
    strain : np.ndarray
        Detector strain (1D array)
    sample_rate : int
        Sample rate in Hz. Default: 4096
    coalescence_time : float, optional
        GPS time of coalescence (for off-source PSD)
    strain_start_time : float, optional
        GPS start time of strain array
    signal_length : float
        Expected signal duration (seconds). Default: 2.0
    apply_notch : bool
        Apply notch filters. Default: True
    apply_whitening : bool
        Apply whitening. Default: True
    apply_bandpass : bool
        Apply bandpass (during whitening). Default: True
    normalize_scale : float, optional
        Scale factor for normalization. If None, no normalization.
    tukey_alpha : float
        Tukey window alpha. Default: 0.1

    Returns
    -------
    Dict containing:
        - processed_strain: np.ndarray
        - psd: FrequencySeries (if whitening applied)
        - processing_steps: List[str]
    """
    processing_steps = []
    psd = None

    processed = strain.copy()

    # Step 1: Notch filter power lines
    if apply_notch:
        processed = notch_filter_powerlines(processed, sample_rate)
        processing_steps.append('notch_filter')

    # Step 2 & 3: Whiten (includes PSD estimation)
    if apply_whitening:
        processed, psd = whiten_real_strain(
            processed,
            sample_rate=sample_rate,
            apply_bandpass=apply_bandpass,
            apply_tukey=True,
            tukey_alpha=tukey_alpha,
            tukey_side='left'
        )
        processing_steps.append('whiten')
        if apply_bandpass:
            processing_steps.append('bandpass')

    # Step 4: Normalize
    if normalize_scale is not None:
        processed = processed * normalize_scale
        processing_steps.append('normalize')

    return {
        'processed_strain': processed,
        'psd': psd,
        'processing_steps': processing_steps
    }


def apply_real_data_processing_pipeline(
    result: Dict,
    apply_notch: bool = True,
    apply_whitening: bool = True,
    apply_bandpass: bool = True,
    normalize_scale: float = 100.0,
    show_progress: bool = True
) -> Dict:
    """
    Apply processing pipeline to real data DataLoaders.

    Compatible with JHPY's processing functions pattern.

    Parameters
    ----------
    result : Dict
        Output from create_real_data_dataloaders()
    apply_notch : bool
        Apply notch filters. Default: True
    apply_whitening : bool
        Apply whitening. Default: True
    apply_bandpass : bool
        Apply bandpass (with whitening). Default: True
    normalize_scale : float
        Normalization scale. Default: 100.0
    show_progress : bool
        Show progress bar. Default: True

    Returns
    -------
    Dict with processed DataLoaders and updated metadata
    """
    # Extract tensors from DataLoaders
    train_dataset = result['train_loader'].dataset
    base_dataset = train_dataset.dataset
    X_full = base_dataset.tensors[0].numpy()
    y_full = base_dataset.tensors[1].numpy()

    metadata = result['metadata'].copy()
    sample_rate = int(1.0 / metadata['time_resolution'])
    num_samples, num_detectors, time_length = X_full.shape

    # Process each waveform
    X_processed = np.zeros_like(X_full)

    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Processing real data")

    for i in iterator:
        for d in range(num_detectors):
            proc_result = process_real_strain(
                X_full[i, d],
                sample_rate=sample_rate,
                apply_notch=apply_notch,
                apply_whitening=apply_whitening,
                apply_bandpass=apply_bandpass,
                normalize_scale=normalize_scale
            )
            X_processed[i, d] = proc_result['processed_strain']

    # Convert back to tensors
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y_full, dtype=torch.float32)

    # Get original indices
    train_indices = result['train_loader'].dataset.indices
    val_indices = result['val_loader'].dataset.indices
    test_indices = result['test_loader'].dataset.indices

    # Create new dataset and DataLoaders
    full_dataset = TensorDataset(X_tensor, y_tensor)
    train_data = Subset(full_dataset, train_indices)
    val_data = Subset(full_dataset, val_indices)
    test_data = Subset(full_dataset, test_indices)

    batch_size = metadata['batch_size']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Update preprocessing metadata
    metadata['preprocessing'] = {
        'whitened': apply_whitening,
        'normalized': normalize_scale is not None,
        'notch_filtered': apply_notch,
        'bandpass_filtered': apply_bandpass,
        'normalize_scale': normalize_scale
    }

    print(f"\nProcessed {num_samples} samples")
    print(f"  Notch filter: {apply_notch}")
    print(f"  Whitening: {apply_whitening}")
    print(f"  Bandpass: {apply_bandpass}")
    print(f"  Normalization: {normalize_scale}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': metadata
    }
