"""
Waveform Processing Utilities using PyCBC

This module provides simple functions for processing single gravitational wave waveforms:
- normalize_waveform(): Normalize a waveform using peak, RMS, or energy methods
- whiten_waveform(): Whiten a waveform using PSD-based whitening (follows PyCBC GW150914 tutorial)
- resample_waveform(): Resample a waveform to a different sampling rate

All functions work on single 1D waveforms (not batches).
"""

from JHPY import *
import numpy as np
import matplotlib.pyplot as plt
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.psd import welch, interpolate
from pycbc.filter import highpass_fir, lowpass_fir, resample_to_delta_t


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

    # Compute PSD using Welch method (PyCBC tutorial approach)
    # Adjust segment length for short signals - need at least 2 segments
    n_samples = len(strain)
    if n_samples >= 4096:
        # Default: use 4096 sample segments
        seg_len = 4096
        seg_stride = 2048
    elif n_samples >= 1024:
        # Short signals: use smaller segments (1/4 of signal length)
        seg_len = max(256, n_samples // 4)
        seg_stride = seg_len // 2
    else:
        # Very short signals: use minimal segments
        seg_len = max(64, n_samples // 4)
        seg_stride = seg_len // 2

    psd_welch = welch(strain, seg_len=seg_len, seg_stride=seg_stride)

    # Interpolate to smooth frequency grid
    psd = interpolate(psd_welch, 1.0 / strain.duration)

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
        white_strain = highpass_fir(white_strain, 35, 8)
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


if __name__ == "__main__":
    """
    Example usage: Generate waveforms and demonstrate processing functions.
    """

    print("=" * 80)
    print("Waveform Processing Functions Demo")
    print("=" * 80)

    # Configuration for waveform generation
    config = {
        'mass1': lambda size: np.random.uniform(20, 50, size=size),
        'mass2': lambda size: np.random.uniform(20, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
        'spin2z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
    }

    # Generate a few waveforms using JHPY with realistic noise
    print("\nGenerating 5 waveforms using JHPY's pycbc_data_generator...")
    result = pycbc_data_generator(
        config,
        num_samples=5,
        batch_size=5,
        num_workers=1,
        detectors=['H1', 'L1'],
        add_noise=True,  # Add realistic detector noise
        signal_length=2.0,
        show_progress=False
    )

    # Extract a single waveform from the first sample, H1 detector
    print("\nExtracting single waveform from batch...")
    train_loader = result['train_loader']
    waveforms, params = next(iter(train_loader))
    single_waveform = waveforms[0, 0, :].numpy()  # Shape: (time_samples,)
    delta_t = result['metadata']['time_resolution']

    print(f"  Single waveform shape: {single_waveform.shape}")
    print(f"  Time resolution: {delta_t} s")
    print(f"  Duration: {len(single_waveform) * delta_t:.2f} s")

    # Demonstrate all three functions
    print("\n" + "=" * 80)
    print("Testing Functions")
    print("=" * 80)

    # 1. Normalize
    print("\n1. Normalizing waveform (multiply by 1e21)...")
    normalized = normalize_waveform(single_waveform)
    print(f"   Before: std={single_waveform.std():.2e}, range=[{single_waveform.min():.2e}, {single_waveform.max():.2e}]")
    print(f"   After:  std={normalized.std():.2f}, range=[{normalized.min():.2f}, {normalized.max():.2f}]")

    # 2. Whiten
    print("\n2. Whitening waveform...")
    whitened, psd, freqs = whiten_waveform(single_waveform, delta_t=delta_t, f_lower=40.0)
    print(f"   Whitened shape: {whitened.shape}")
    print(f"   PSD shape: {psd.shape}")
    print(f"   Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")

    # 3. Resample
    print("\n3. Resampling waveform (4096 Hz -> 2048 Hz)...")
    target_delta_t = 1/2048  # Downsample by factor of 2
    resampled = resample_waveform(single_waveform,
                                  original_delta_t=delta_t,
                                  target_delta_t=target_delta_t)
    print(f"   Original length: {len(single_waveform)} samples")
    print(f"   Resampled length: {len(resampled)} samples")
    print(f"   Original rate: {1/delta_t:.0f} Hz")
    print(f"   Target rate: {1/target_delta_t:.0f} Hz")

    # 4. Combined pipeline: Whiten -> Normalize -> Resample
    # Why this order?
    # 1. Whiten first: Remove colored noise, flatten spectrum (ensures same noise amplitude!)
    # 2. Normalize second: Scale whitened data to O(1-10) range with FIXED factor
    #    - CRITICAL: Use fixed scale factor to preserve relative peak amplitudes!
    #    - This allows LSTM to learn distance variations from peak heights
    # 3. Resample third: Downsample after processing
    print("\n4. Combined pipeline (Whiten -> Normalize -> Resample)...")

    # Step 1: Whiten to remove colored noise
    whitened_combined, psd_combined, freqs_combined = whiten_waveform(
        single_waveform,
        delta_t=delta_t,
        f_lower=40.0,
        apply_bandpass=True
    )
    print(f"   After whiten: std={whitened_combined.std():.2e}, range=[{whitened_combined.min():.2e}, {whitened_combined.max():.2e}]")

    # Step 2: Normalize to bring to reasonable range
    # Use FIXED scale factor to preserve relative peak amplitudes between waveforms
    # This is critical for LSTM to learn distance-dependent amplitude changes
    fixed_scale = 100.0  # Scale whitened data (~0.01-0.1) to O(1-10) range
    normalized_combined = normalize_waveform(whitened_combined, scale_factor=fixed_scale)
    print(f"   After normalize (x{fixed_scale:.1f}): std={normalized_combined.std():.2f}, range=[{normalized_combined.min():.2f}, {normalized_combined.max():.2f}]")

    # Step 3: Resample to reduce data size
    resampled_combined = resample_waveform(normalized_combined,
                                          original_delta_t=delta_t,
                                          target_delta_t=target_delta_t)
    print(f"   After resample: length {len(normalized_combined)} -> {len(resampled_combined)}")
    print(f"   Final output: std={resampled_combined.std():.2f}, range=[{resampled_combined.min():.2f}, {resampled_combined.max():.2f}]")
    print(f"   Combined pipeline complete!")

    # Create comparison plots
    print("\n" + "=" * 80)
    print("Creating Plots")
    print("=" * 80)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # Time arrays
    time_original = np.arange(len(single_waveform)) * delta_t
    time_normalized = time_original
    time_whitened = np.arange(len(whitened)) * delta_t
    time_resampled = np.arange(len(resampled)) * target_delta_t

    # Row 1, Col 1: Original waveform
    axes[0, 0].plot(time_original, single_waveform, linewidth=0.8, color='steelblue')
    axes[0, 0].set_xlabel('Time (s)', fontsize=10)
    axes[0, 0].set_ylabel('Strain', fontsize=10)
    axes[0, 0].set_title('Original Waveform', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Row 1, Col 2: Normalized waveform
    axes[0, 1].plot(time_normalized, normalized, linewidth=0.8, color='green')
    axes[0, 1].set_xlabel('Time (s)', fontsize=10)
    axes[0, 1].set_ylabel('Normalized Strain', fontsize=10)
    axes[0, 1].set_title('Normalized (Peak Method)', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Row 1, Col 3: Whitened waveform
    axes[0, 2].plot(time_whitened, whitened, linewidth=0.8, color='purple')
    axes[0, 2].set_xlabel('Time (s)', fontsize=10)
    axes[0, 2].set_ylabel('Whitened Strain', fontsize=10)
    axes[0, 2].set_title('Whitened Waveform', fontsize=11, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2, Col 1: Original PSD
    psd_array = np.abs(psd)
    mask = (freqs >= 40.0) & (psd_array > 0) & np.isfinite(psd_array)

    if np.any(mask):
        axes[1, 0].loglog(freqs[mask], psd_array[mask], linewidth=0.8, color='darkred')
    else:
        # Fallback to linear scale if no valid data for log plot
        mask = freqs >= 40.0
        axes[1, 0].plot(freqs[mask], psd_array[mask], linewidth=0.8, color='darkred')

    axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=10)
    axes[1, 0].set_ylabel('PSD', fontsize=10)
    axes[1, 0].set_title('Power Spectral Density', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, which='both')

    # Row 2, Col 2: PSD of whitened waveform
    whitened_ts = TimeSeries(whitened, delta_t=delta_t)
    psd_white_welch = welch(whitened_ts)
    psd_white = interpolate(psd_white_welch, 1.0 / whitened_ts.duration)
    freqs_white = np.arange(len(psd_white)) * psd_white.delta_f
    psd_white_array = np.abs(np.array(psd_white))

    # Filter out invalid values for log plotting
    mask_white = (freqs_white >= 40.0) & (psd_white_array > 0) & np.isfinite(psd_white_array)

    if np.any(mask_white):
        axes[1, 1].loglog(freqs_white[mask_white], psd_white_array[mask_white],
                         linewidth=0.8, color='orange')
    else:
        # If no valid data for log plot, use linear scale
        mask_white = freqs_white >= 40.0
        axes[1, 1].plot(freqs_white[mask_white], psd_white_array[mask_white],
                       linewidth=0.8, color='orange')

    axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=10)
    axes[1, 1].set_ylabel('PSD', fontsize=10)
    axes[1, 1].set_title('Whitened PSD (Flattened)', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, which='both')

    # Row 2, Col 3: Resampled waveform
    axes[1, 2].plot(time_resampled, resampled, linewidth=0.8, color='darkgreen')
    axes[1, 2].set_xlabel('Time (s)', fontsize=10)
    axes[1, 2].set_ylabel('Strain', fontsize=10)
    axes[1, 2].set_title(f'Resampled ({1/target_delta_t:.0f} Hz)', fontsize=11, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)

    # Row 1, Col 4: Combined pipeline result (normalized -> whitened -> resampled)
    time_combined = np.arange(len(resampled_combined)) * target_delta_t
    axes[0, 3].plot(time_combined, resampled_combined, linewidth=0.8, color='darkmagenta')
    axes[0, 3].set_xlabel('Time (s)', fontsize=10)
    axes[0, 3].set_ylabel('Strain', fontsize=10)
    axes[0, 3].set_title('Combined: White+Norm+Resamp', fontsize=11, fontweight='bold')
    axes[0, 3].grid(True, alpha=0.3)

    # Row 2, Col 4: PSD of combined pipeline result (final resampled data)
    combined_ts = TimeSeries(resampled_combined, delta_t=target_delta_t)
    psd_combined_welch = welch(combined_ts)
    psd_combined_interp = interpolate(psd_combined_welch, 1.0 / combined_ts.duration)
    freqs_combined_plot = np.arange(len(psd_combined_interp)) * psd_combined_interp.delta_f
    psd_combined_array = np.abs(np.array(psd_combined_interp))

    # Filter out invalid values for log plotting
    mask_combined = (freqs_combined_plot >= 40.0) & (psd_combined_array > 0) & np.isfinite(psd_combined_array)

    if np.any(mask_combined):
        axes[1, 3].loglog(freqs_combined_plot[mask_combined], psd_combined_array[mask_combined],
                         linewidth=0.8, color='darkmagenta')
    else:
        # Fallback to linear scale
        mask_combined = freqs_combined_plot >= 40.0
        axes[1, 3].plot(freqs_combined_plot[mask_combined], psd_combined_array[mask_combined],
                       linewidth=0.8, color='darkmagenta')

    axes[1, 3].set_xlabel('Frequency (Hz)', fontsize=10)
    axes[1, 3].set_ylabel('PSD', fontsize=10)
    axes[1, 3].set_title('Combined Pipeline PSD', fontsize=11, fontweight='bold')
    axes[1, 3].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_file = 'waveform_processing_demo.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nThree simple functions demonstrated:")
    print("  1. normalize_waveform() - Scales waveform to order 1-10 range")
    print("  2. whiten_waveform() - Whitens using PSD (PyCBC tutorial method)")
    print("  3. resample_waveform() - Resamples to different rate")
    print("\nCombined pipeline demonstrated:")
    print("  4. Whiten -> Normalize -> Resample")
    print("     Remove noise, scale to O(1-10), then downsample")
    print("\nAll functions:")
    print("  - Work on single 1D waveforms")
    print("  - Accept numpy arrays or PyCBC TimeSeries")
    print("  - Return numpy arrays")
    print("  - Can be called in a loop for multiple waveforms")
    print("  - Can be easily combined in pipelines")
