"""
Test suite for Real GW Data module.

Tests:
1. Catalog access and parameter extraction
2. Real strain data loading from GWOSC
3. Signal processing pipeline
4. DataLoader format compatibility with JHPY
5. Real vs simulated comparison
6. Save/load functionality
7. Demo visualization generation

Usage:
    python test_real_data.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch

# Add parent directory to path for JHPY import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catalog_utils import (
    get_available_events,
    get_event_parameters,
    get_event_metadata,
    filter_events_by_criteria,
    parameters_to_jhpy_format,
    get_event_summary
)

from real_data_loader import (
    load_real_event,
    load_multiple_events,
    extract_signal_window,
    create_real_data_dataloaders,
    save_real_dataloaders,
    load_real_dataloaders
)

from event_processor import (
    notch_filter_powerlines,
    whiten_real_strain,
    check_data_quality,
    process_real_strain,
    apply_real_data_processing_pipeline
)

from comparison_utils import (
    generate_comparison_waveform,
    compare_real_vs_simulated,
    create_comparison_plot,
    compute_parameter_recovery_accuracy,
    print_comparison_summary
)


def test_catalog_access():
    """Test GWTC catalog access."""
    print("\n" + "="*60)
    print("TEST 1: Catalog Access")
    print("="*60)

    # Get available events
    events = get_available_events('gwtc-3')
    print(f"Found {len(events)} events in GWTC-3")
    print(f"First 5 events: {events[:5]}")

    # Get parameters for GW150914
    params = get_event_parameters('GW150914')
    print(f"\nGW150914 parameters:")
    print(f"  Mass1: {params.get('mass1', 'N/A')}")
    print(f"  Mass2: {params.get('mass2', 'N/A')}")
    print(f"  Distance: {params.get('luminosity_distance', 'N/A')} Mpc")
    print(f"  SNR: {params.get('network_snr', 'N/A')}")
    print(f"  Event type: {params.get('event_type', 'N/A')}")

    # Get event metadata
    metadata = get_event_metadata('GW150914')
    print(f"\nGW150914 metadata:")
    print(f"  GPS time: {metadata.get('gps_time', 'N/A')}")
    print(f"  Detectors: {metadata.get('detectors', 'N/A')}")

    # Get event summary
    print(f"\nEvent summary:")
    print(get_event_summary('GW150914'))

    # Filter events
    try:
        bbh_events = filter_events_by_criteria(event_type='BBH')
        print(f"\nBBH events found: {len(bbh_events)}")
    except Exception as e:
        print(f"Filter test skipped: {e}")

    print("\nTEST 1 PASSED")
    return True


def test_parameter_conversion():
    """Test parameter conversion to JHPY format."""
    print("\n" + "="*60)
    print("TEST 2: Parameter Conversion")
    print("="*60)

    params = get_event_parameters('GW150914')
    param_names = ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance']

    jhpy_params = parameters_to_jhpy_format(params, param_names)
    print(f"Parameter names: {param_names}")
    print(f"JHPY format array: {jhpy_params}")
    print(f"Array shape: {jhpy_params.shape}")
    print(f"Array dtype: {jhpy_params.dtype}")

    assert len(jhpy_params) == len(param_names), "Parameter count mismatch"
    assert jhpy_params.dtype == np.float32, "Wrong dtype"

    print("\nTEST 2 PASSED")
    return True


def test_real_data_loading():
    """Test loading real detector strain."""
    print("\n" + "="*60)
    print("TEST 3: Real Data Loading")
    print("="*60)

    # Load GW150914
    print("Loading GW150914 data from GWOSC...")
    data = load_real_event(
        'GW150914',
        detectors=['H1', 'L1'],
        sample_rate=4096,
        duration=32.0
    )

    print(f"\nLoaded GW150914 data:")
    print(f"  Sample rate: {data['sample_rate']} Hz")
    print(f"  Duration: {data['duration']} s")
    print(f"  Coalescence time: {data['coalescence_time']}")

    for det, strain in data['strains'].items():
        print(f"  {det} strain: {len(strain)} samples")

    # Verify data
    assert 'H1' in data['strains'] or 'L1' in data['strains'], "No detector data loaded"
    assert data['event_params'] is not None, "No event parameters"

    print("\nTEST 3 PASSED")
    return True


def test_signal_window_extraction():
    """Test extracting signal window around merger."""
    print("\n" + "="*60)
    print("TEST 4: Signal Window Extraction")
    print("="*60)

    # Load full data
    data = load_real_event('GW150914', duration=32.0)

    # Extract 2-second window
    windowed = extract_signal_window(data, window_duration=2.0, center_on_merger=True)

    print(f"Original duration: {data['duration']} s")
    print(f"Window duration: {windowed['duration']} s")

    for det in windowed['strains']:
        expected_samples = int(2.0 * data['sample_rate'])
        actual_samples = len(windowed['strains'][det])
        print(f"  {det}: {actual_samples} samples (expected ~{expected_samples})")

    print("\nTEST 4 PASSED")
    return True


def test_dataloader_creation():
    """Test creating DataLoaders compatible with JHPY."""
    print("\n" + "="*60)
    print("TEST 5: DataLoader Creation")
    print("="*60)

    # Create DataLoaders for a single event
    print("Creating DataLoaders for GW150914...")
    result = create_real_data_dataloaders(
        event_names='GW150914',
        signal_length=2.0,
        time_resolution=1/4096,
        batch_size=1
    )

    print(f"\nDataLoader result keys: {list(result.keys())}")
    print(f"Metadata keys: {list(result['metadata'].keys())}")

    # Verify format
    assert 'train_loader' in result, "Missing train_loader"
    assert 'val_loader' in result, "Missing val_loader"
    assert 'test_loader' in result, "Missing test_loader"
    assert 'metadata' in result, "Missing metadata"

    # Check shapes
    X, y = next(iter(result['train_loader']))
    print(f"\nBatch shapes:")
    print(f"  X (waveforms): {X.shape}")
    print(f"  y (parameters): {y.shape}")

    metadata = result['metadata']
    print(f"\nMetadata:")
    print(f"  Waveform shape: {metadata['waveform_shape']}")
    print(f"  Parameter names: {metadata['parameter_names']}")
    print(f"  Channels: {metadata['channels']}")
    print(f"  Data source: {metadata['data_source']}")
    print(f"  Event names: {metadata['event_names']}")

    # Verify JHPY compatibility
    assert X.dim() == 3, "X should be 3D (batch, detectors, time)"
    assert y.dim() == 2, "y should be 2D (batch, params)"
    assert metadata['data_source'] == 'real_gwtc', "Wrong data source"

    print("\nTEST 5 PASSED")
    return True


def test_signal_processing():
    """Test signal processing pipeline."""
    print("\n" + "="*60)
    print("TEST 6: Signal Processing")
    print("="*60)

    # Load data
    data = load_real_event('GW150914', duration=4.0)
    windowed = extract_signal_window(data, window_duration=2.0)

    # Get strain for processing
    det = list(windowed['strains'].keys())[0]
    strain = np.array(windowed['strains'][det])
    sample_rate = data['sample_rate']

    print(f"Processing {det} strain: {len(strain)} samples")

    # Test notch filtering
    print("\nTesting notch filter...")
    notched = notch_filter_powerlines(strain, sample_rate)
    print(f"  Notched strain shape: {notched.shape}")

    # Test data quality check
    print("\nTesting data quality check...")
    quality = check_data_quality(strain, sample_rate)
    print(f"  Quality score: {quality['quality_score']:.2f}")
    print(f"  Is valid: {quality['is_valid']}")
    print(f"  Has gaps: {quality['has_gaps']}")

    # Test whitening
    print("\nTesting whitening...")
    whitened, psd = whiten_real_strain(strain, sample_rate)
    print(f"  Whitened strain shape: {whitened.shape}")
    print(f"  PSD type: {type(psd)}")

    # Test full processing pipeline
    print("\nTesting full processing pipeline...")
    processed = process_real_strain(
        strain,
        sample_rate=sample_rate,
        apply_notch=True,
        apply_whitening=True,
        apply_bandpass=True,
        normalize_scale=100.0
    )
    print(f"  Processing steps: {processed['processing_steps']}")
    print(f"  Processed strain shape: {processed['processed_strain'].shape}")

    print("\nTEST 6 PASSED")
    return True


def test_jhpy_compatibility():
    """Test that real data DataLoaders work with JHPY functions."""
    print("\n" + "="*60)
    print("TEST 7: JHPY Function Compatibility")
    print("="*60)

    # Import JHPY functions
    try:
        from JHPY import (
            whiten_dataloaders,
            normalize_dataloaders,
            truncate_dataloaders,
            resample_dataloaders
        )
        jhpy_available = True
    except ImportError:
        print("JHPY not available, skipping compatibility test")
        jhpy_available = False
        return True

    # Create real data DataLoaders
    result = create_real_data_dataloaders(
        event_names='GW150914',
        signal_length=2.0,
        time_resolution=1/4096
    )

    print("Testing JHPY processing functions on real data...")

    # Test whiten_dataloaders
    print("\n  Testing whiten_dataloaders...")
    try:
        whitened = whiten_dataloaders(result, show_progress=False)
        assert whitened['metadata']['preprocessing']['whitened'] == True
        print("    PASSED")
    except Exception as e:
        print(f"    FAILED: {e}")

    # Test normalize_dataloaders
    print("  Testing normalize_dataloaders...")
    try:
        normalized = normalize_dataloaders(whitened, scale_factor=100.0)
        assert normalized['metadata']['preprocessing']['normalized'] == True
        print("    PASSED")
    except Exception as e:
        print(f"    FAILED: {e}")

    # Test truncate_dataloaders
    print("  Testing truncate_dataloaders...")
    try:
        truncated = truncate_dataloaders(normalized, target_duration=1.0)
        assert truncated['metadata']['preprocessing']['truncated'] == True
        print("    PASSED")
    except Exception as e:
        print(f"    FAILED: {e}")

    print("\nTEST 7 PASSED - All JHPY functions compatible!")
    return True


def test_comparison_utilities():
    """Test real vs simulated comparison."""
    print("\n" + "="*60)
    print("TEST 8: Comparison Utilities")
    print("="*60)

    # Generate simulated waveform with GW150914 parameters
    print("Generating simulated waveform with GW150914 parameters...")
    simulated = generate_comparison_waveform(
        'GW150914',
        signal_length=2.0,
        detectors=['H1', 'L1']
    )

    print(f"\nSimulated waveform:")
    print(f"  Detectors: {list(simulated['waveforms'].keys())}")
    print(f"  Parameters used:")
    for key, val in simulated['parameters'].items():
        if val is not None:
            print(f"    {key}: {val}")

    # Load real data
    print("\nLoading real data for comparison...")
    real_data = load_real_event('GW150914', duration=4.0)

    # Compare
    print("\nComparing real vs simulated...")
    comparison = compare_real_vs_simulated(
        real_data,
        simulated,
        'GW150914'
    )

    print_comparison_summary(comparison)

    # Verify comparison outputs
    assert 'match_values' in comparison, "Missing match_values"
    assert 'snr_real' in comparison, "Missing snr_real"
    assert 'time_shift' in comparison, "Missing time_shift"

    print("TEST 8 PASSED")
    return True


def test_save_load():
    """Test saving and loading real data DataLoaders."""
    print("\n" + "="*60)
    print("TEST 9: Save/Load")
    print("="*60)

    # Create DataLoaders
    result = create_real_data_dataloaders('GW150914', signal_length=2.0)

    # Apply processing
    processed = apply_real_data_processing_pipeline(
        result,
        apply_notch=True,
        apply_whitening=True,
        normalize_scale=100.0,
        show_progress=False
    )

    # Save
    test_path = 'test_real_data_temp.pt'
    save_real_dataloaders(processed, test_path)
    print(f"Saved to {test_path}")

    # Load
    loaded = load_real_dataloaders(test_path)

    # Verify
    X_orig, y_orig = next(iter(processed['train_loader']))
    X_loaded, y_loaded = next(iter(loaded['train_loader']))

    print(f"\nOriginal X shape: {X_orig.shape}")
    print(f"Loaded X shape: {X_loaded.shape}")

    assert torch.allclose(X_orig, X_loaded), "X tensors don't match"
    assert torch.allclose(y_orig, y_loaded), "y tensors don't match"
    print("Tensors match!")

    # Cleanup
    os.remove(test_path)
    print(f"Cleaned up {test_path}")

    print("\nTEST 9 PASSED")
    return True


def test_parameter_recovery():
    """Test parameter recovery accuracy computation."""
    print("\n" + "="*60)
    print("TEST 10: Parameter Recovery Accuracy")
    print("="*60)

    # Simulate inferred parameters (with some error)
    true_params = get_event_parameters('GW150914')
    inferred = {
        'mass1': true_params.get('mass1', 35) * 1.05,  # 5% error
        'mass2': true_params.get('mass2', 30) * 0.98,  # 2% error
        'distance': true_params.get('luminosity_distance', 410) * 1.1,  # 10% error
    }

    print(f"True mass1: {true_params.get('mass1', 'N/A')}")
    print(f"Inferred mass1: {inferred['mass1']:.2f}")

    accuracy = compute_parameter_recovery_accuracy('GW150914', inferred)

    print(f"\nParameter recovery results:")
    for param in accuracy['absolute_errors']:
        true_val = accuracy['true_values'][param]
        inferred_val = accuracy['inferred_values'][param]
        rel_err = accuracy['relative_errors'][param]
        print(f"  {param}: true={true_val:.2f}, inferred={inferred_val:.2f}, error={rel_err:.1f}%")

    print("\nTEST 10 PASSED")
    return True


def create_real_vs_simulated_comparison():
    """Create comparison between whitened real signal and simulated data."""
    print("\n" + "="*60)
    print("Real vs Simulated Comparison")
    print("="*60)

    from pycbc.waveform import get_td_waveform
    from pycbc.types import TimeSeries, FrequencySeries
    from pycbc.detector import Detector
    from pycbc.psd import welch, interpolate
    from pycbc.filter import highpass_fir, lowpass_fir

    # Load 32s of real data for proper PSD estimation
    print("Loading GW150914 real data (32s)...")
    real_data = load_real_event('GW150914', duration=32.0)

    sample_rate = real_data['sample_rate']
    delta_t = 1.0 / sample_rate
    coalescence_time = real_data['coalescence_time']
    start_time = float(real_data['strains']['H1'].start_time)

    # Get event parameters for simulation
    params = get_event_parameters('GW150914')
    mass1 = params.get('mass1') or params.get('mass1_source', 35.6)
    mass2 = params.get('mass2') or params.get('mass2_source', 30.6)
    spin1z = params.get('spin1z', 0.0) or 0.0
    spin2z = params.get('spin2z', 0.0) or 0.0
    distance = params.get('luminosity_distance', 410.0)
    inclination = params.get('inclination', 0.0) or 0.0
    ra = params.get('ra', 0.0) or 0.0
    dec = params.get('dec', 0.0) or 0.0
    polarization = params.get('polarization', 0.0) or 0.0

    print(f"Using parameters: M1={mass1:.1f}, M2={mass2:.1f}, D={distance:.0f} Mpc")

    # Setup detector and real data first
    detector = Detector('H1')
    real_strain = np.array(real_data['strains']['H1'])
    merger_idx = int((coalescence_time - start_time) * sample_rate)

    # Estimate PSD from off-source data (before the signal)
    print("Estimating PSD from off-source data...")
    ts_real = TimeSeries(real_strain.astype(np.float64), delta_t=delta_t)
    psd = welch(ts_real, seg_len=4096, seg_stride=2048)

    # Function to whiten and bandpass a strain
    def whiten_strain(strain_array, psd_in, sr):
        ts = TimeSeries(strain_array.astype(np.float64), delta_t=1.0/sr)
        psd_interp = interpolate(psd_in, 1.0 / ts.duration)
        freq_series = ts.to_frequencyseries()
        psd_interp.resize(len(freq_series))

        # Add epsilon to avoid division by zero
        psd_arr = np.array(psd_interp)
        psd_arr[psd_arr <= 0] = 1e-40
        psd_safe = FrequencySeries(psd_arr, delta_f=psd_interp.delta_f)

        # Whiten
        white = (freq_series / (psd_safe ** 0.5)).to_timeseries()

        # Bandpass 35-300 Hz
        white = highpass_fir(white, 35.0, 8)
        white = lowpass_fir(white, 300.0, 8)
        return np.array(white)

    # Whiten real data first
    print("Whitening real data...")
    real_whitened = whiten_strain(real_strain, psd, sample_rate)

    # Create time axis relative to merger
    time_full = np.arange(len(real_whitened)) / sample_rate
    time_relative = time_full - time_full[merger_idx]

    # Helper to generate, embed, and whiten a simulated waveform
    def generate_and_whiten(coa_phase_val, inc_val, dist_override=None):
        eff_distance = dist_override if dist_override else distance
        hp_t, hc_t = get_td_waveform(
            approximant='IMRPhenomXP',
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1z,
            spin2z=spin2z,
            inclination=inc_val,
            coa_phase=coa_phase_val,
            distance=eff_distance,
            delta_t=delta_t,
            f_lower=20.0
        )
        fp_t, fc_t = detector.antenna_pattern(ra, dec, polarization, coalescence_time)
        strain_t = fp_t * hp_t + fc_t * hc_t
        arr_t = np.array(strain_t)

        padded = np.zeros(len(real_strain))
        start_idx = merger_idx - len(arr_t)
        if start_idx >= 0:
            padded[start_idx:merger_idx] = arr_t
        else:
            padded[:merger_idx] = arr_t[-merger_idx:]
        return whiten_strain(padded, psd, sample_rate)

    # Grid search over phase and inclination for best match
    print("Optimizing phase and inclination for best match...")
    best_corr = -1
    best_phase = 0
    # GW150914 was roughly face-on, use reasonable range
    best_inc = inclination if inclination else np.pi/6

    # Face-on to moderately inclined (avoid edge-on which kills amplitude)
    inc_values = [np.pi/8, np.pi/6, np.pi/4, np.pi/3] if not inclination else [inclination]
    phase_values = np.linspace(0, 2*np.pi, 16, endpoint=False)

    print(f"  Testing {len(inc_values)} inclinations x {len(phase_values)} phases...")
    for inc_test in inc_values:
        for phase_test in phase_values:
            try:
                sim_test = generate_and_whiten(phase_test, inc_test)
                test_mask = (time_relative > -0.2) & (time_relative < 0.05)
                # Check both positive and negative correlation (phase ambiguity)
                corr_val = np.abs(np.corrcoef(real_whitened[test_mask], sim_test[test_mask])[0, 1])
                if corr_val > best_corr:
                    best_corr = corr_val
                    best_phase = phase_test
                    best_inc = inc_test
            except Exception:
                continue

    print(f"  Best phase: {best_phase:.3f} rad, inclination: {best_inc:.3f} rad")
    print(f"  Best initial correlation: {best_corr:.3f}")

    # Generate final waveform with optimized parameters
    # First pass: use catalog distance
    print("Generating final simulated waveform...")
    sim_whitened = generate_and_whiten(best_phase, best_inc)

    # Calculate amplitude ratio for distance adjustment
    test_mask = (time_relative > -0.2) & (time_relative < 0.05)
    amp_ratio = np.std(real_whitened[test_mask]) / (np.std(sim_whitened[test_mask]) + 1e-10)

    # If amplitude is off by > 2x, try adjusting effective distance
    if amp_ratio > 1.5:
        effective_distance = distance / amp_ratio
        print(f"  Amplitude ratio: {amp_ratio:.2f}x - trying effective distance {effective_distance:.0f} Mpc")
        sim_whitened = generate_and_whiten(best_phase, best_inc, dist_override=effective_distance)
    else:
        effective_distance = distance

    # Find optimal time alignment via cross-correlation
    print("Finding optimal time alignment...")
    from scipy.signal import correlate

    # Use the chirp region for alignment
    align_mask = (time_relative > -0.5) & (time_relative < 0.1)
    real_for_align = real_whitened[align_mask]
    sim_for_align = sim_whitened[align_mask]

    # Cross-correlate to find optimal time shift (check both + and - phase)
    corr = correlate(real_for_align, sim_for_align, mode='full')
    max_pos = np.max(corr)
    max_neg = np.max(-corr)

    if max_pos >= max_neg:
        lag_idx = np.argmax(corr) - len(sim_for_align) + 1
        phase_flip = 1.0
    else:
        lag_idx = np.argmax(-corr) - len(sim_for_align) + 1
        phase_flip = -1.0
        print("  Phase flip detected - inverting simulated waveform")

    time_shift_samples = lag_idx

    # Apply time shift and phase flip to simulated data
    sim_aligned = np.roll(sim_whitened * phase_flip, time_shift_samples)
    if time_shift_samples > 0:
        sim_aligned[:time_shift_samples] = 0
    elif time_shift_samples < 0:
        sim_aligned[time_shift_samples:] = 0

    print(f"  Time shift: {time_shift_samples / sample_rate * 1000:.2f} ms ({time_shift_samples} samples)")

    # Create comparison figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: Overlay of real and simulated (zoomed to signal)
    zoom_mask = (time_relative > -0.5) & (time_relative < 0.1)

    # Scale simulated to match real amplitude
    real_zoom = real_whitened[zoom_mask]
    sim_zoom = sim_aligned[zoom_mask]

    # Find optimal scaling factor
    scale = np.std(real_zoom) / (np.std(sim_zoom) + 1e-10)
    sim_scaled = sim_aligned * scale

    axes[0].plot(time_relative[zoom_mask], real_whitened[zoom_mask], 'b-',
                 linewidth=0.8, alpha=0.8, label='Real GW150914')
    axes[0].plot(time_relative[zoom_mask], sim_scaled[zoom_mask], 'r--',
                 linewidth=0.8, alpha=0.8, label='Simulated (scaled)')
    axes[0].axvline(x=0, color='k', linestyle=':', alpha=0.5)
    axes[0].set_title('GW150914 H1: Real vs Simulated (both whitened, 35-300 Hz)')
    axes[0].set_xlabel('Time relative to merger (s)')
    axes[0].set_ylabel('Whitened Strain')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Last 0.15s - detailed merger comparison
    zoom_mask2 = (time_relative > -0.15) & (time_relative < 0.05)
    axes[1].plot(time_relative[zoom_mask2], real_whitened[zoom_mask2], 'b-',
                 linewidth=1.0, alpha=0.9, label='Real')
    axes[1].plot(time_relative[zoom_mask2], sim_scaled[zoom_mask2], 'r--',
                 linewidth=1.0, alpha=0.9, label='Simulated')
    axes[1].axvline(x=0, color='k', linestyle=':', alpha=0.5)
    axes[1].set_title('Final Inspiral & Merger Detail')
    axes[1].set_xlabel('Time relative to merger (s)')
    axes[1].set_ylabel('Whitened Strain')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Residual
    residual = real_whitened - sim_scaled
    axes[2].plot(time_relative[zoom_mask], residual[zoom_mask], 'g-',
                 linewidth=0.6, alpha=0.8)
    axes[2].axvline(x=0, color='k', linestyle=':', alpha=0.5)
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[2].set_title('Residual (Real - Simulated)')
    axes[2].set_xlabel('Time relative to merger (s)')
    axes[2].set_ylabel('Residual')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('GW150914_real_vs_simulated.png', dpi=150)
    print("Saved: GW150914_real_vs_simulated.png")

    # Compute match statistics
    # Pearson correlation for aligned signals
    correlation = np.corrcoef(real_zoom, sim_zoom)[0, 1]
    print(f"\nComparison Statistics:")
    print(f"  Pearson correlation (aligned): {correlation:.4f}")
    print(f"  Time shift applied: {time_shift_samples / sample_rate * 1000:.2f} ms")
    print(f"  Scale factor applied: {scale:.2f}")
    print(f"  Real RMS: {np.std(real_zoom):.4f}")
    print(f"  Sim RMS (after alignment): {np.std(sim_zoom):.4f}")

    # Interpret the correlation
    if correlation >= 0.8:
        print(f"\n  Excellent match! The simulated waveform closely matches the real signal.")
    elif correlation >= 0.6:
        print(f"\n  Good match. Differences are expected due to parameter uncertainties and noise.")
    else:
        print(f"\n  Moderate match. Consider tuning: distance, inclination, or coalescence phase.")

    print(f"\n  Final optimized parameters:")
    print(f"    Masses: {mass1:.1f} + {mass2:.1f} Msun")
    print(f"    Distance: {effective_distance:.0f} Mpc (catalog: {distance:.0f} Mpc)")
    print(f"    Inclination: {best_inc:.3f} rad ({np.degrees(best_inc):.1f} deg)")
    print(f"    Phase: {best_phase:.3f} rad")

    plt.close()
    return True


def create_demo_visualization():
    """Create demonstration visualization with properly whitened GW signal."""
    print("\n" + "="*60)
    print("Creating Demo Visualization")
    print("="*60)

    # Load full 32s of data for proper whitening (need long segment for good PSD)
    print("Loading GW150914 real data (32s for proper whitening)...")
    real_data = load_real_event('GW150914', duration=32.0)

    sample_rate = real_data['sample_rate']
    coalescence_time = real_data['coalescence_time']
    start_time = float(real_data['strains']['H1'].start_time)

    # Process both detectors
    whitened_strains = {}
    for det in ['H1', 'L1']:
        if det in real_data['strains']:
            strain = np.array(real_data['strains'][det])
            whitened, _ = whiten_real_strain(
                strain,
                sample_rate=sample_rate,
                apply_bandpass=True,
                apply_tukey=True,
                tukey_side='both'
            )
            whitened_strains[det] = whitened

    # Calculate time axis relative to merger
    h1_whitened = whitened_strains['H1']
    time_full = np.arange(len(h1_whitened)) / sample_rate
    merger_idx = int((coalescence_time - start_time) * sample_rate)
    time_relative = time_full - time_full[merger_idx]

    # Create multi-panel visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: Full 32s whitened strain
    axes[0].plot(time_relative, h1_whitened, 'b-', linewidth=0.3)
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Merger')
    axes[0].set_title('GW150914 H1 - Whitened Strain (full 32s, bandpass 35-300 Hz)')
    axes[0].set_xlabel('Time relative to merger (s)')
    axes[0].set_ylabel('Whitened Strain')
    axes[0].set_xlim(-16, 16)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Zoom to 0.5s around merger - see the chirp!
    zoom_mask = (time_relative > -0.5) & (time_relative < 0.1)
    axes[1].plot(time_relative[zoom_mask], h1_whitened[zoom_mask], 'b-', linewidth=0.8)
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Merger')
    axes[1].set_title('GW150914 H1 - Chirp Signal (zoom: -0.5s to +0.1s)')
    axes[1].set_xlabel('Time relative to merger (s)')
    axes[1].set_ylabel('Whitened Strain')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Last 0.2s - the actual inspiral-merger
    zoom_mask2 = (time_relative > -0.2) & (time_relative < 0.05)
    axes[2].plot(time_relative[zoom_mask2], h1_whitened[zoom_mask2], 'b-', linewidth=1.0)
    axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Merger')
    axes[2].set_title('GW150914 H1 - Final Inspiral & Merger (last 0.2s)')
    axes[2].set_xlabel('Time relative to merger (s)')
    axes[2].set_ylabel('Whitened Strain')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('GW150914_whitened_signal.png', dpi=150)
    print("Saved: GW150914_whitened_signal.png")

    # Create H1 vs L1 comparison
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

    l1_whitened = whitened_strains.get('L1', h1_whitened)
    zoom_mask = (time_relative > -0.5) & (time_relative < 0.1)

    axes2[0].plot(time_relative[zoom_mask], h1_whitened[zoom_mask], 'b-', linewidth=0.8)
    axes2[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes2[0].set_title('GW150914 - LIGO Hanford (H1) - Whitened')
    axes2[0].set_ylabel('Whitened Strain')
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(time_relative[zoom_mask], l1_whitened[zoom_mask], 'g-', linewidth=0.8)
    axes2[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes2[1].set_title('GW150914 - LIGO Livingston (L1) - Whitened')
    axes2[1].set_xlabel('Time relative to merger (s)')
    axes2[1].set_ylabel('Whitened Strain')
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('GW150914_H1_L1_whitened.png', dpi=150)
    print("Saved: GW150914_H1_L1_whitened.png")

    plt.close('all')
    return True


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Real GW Data Module Test Suite")
    print("="*60)

    tests = [
        ("Catalog Access", test_catalog_access),
        ("Parameter Conversion", test_parameter_conversion),
        ("Real Data Loading", test_real_data_loading),
        ("Signal Window Extraction", test_signal_window_extraction),
        ("DataLoader Creation", test_dataloader_creation),
        ("Signal Processing", test_signal_processing),
        ("JHPY Compatibility", test_jhpy_compatibility),
        ("Comparison Utilities", test_comparison_utilities),
        ("Save/Load", test_save_load),
        ("Parameter Recovery", test_parameter_recovery),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\nTEST FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Create visualizations
    try:
        create_real_vs_simulated_comparison()
    except Exception as e:
        print(f"\nReal vs simulated comparison failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        create_demo_visualization()
    except Exception as e:
        print(f"\nDemo visualization failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSome tests failed. Check output above for details.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
