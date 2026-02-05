"""
Test file for DataLoader processing functions.

Tests the separated data processing pipeline:
1. pycbc_data_generator() - raw waveform generation
2. whiten_dataloaders() - PSD-based whitening
3. normalize_dataloaders() - fixed-scale normalization
4. truncate_dataloaders() - signal truncation
5. resample_dataloaders() - resampling
"""

import numpy as np
import matplotlib.pyplot as plt
from JHPY import (
    pycbc_data_generator,
    whiten_dataloaders,
    normalize_dataloaders,
    truncate_dataloaders,
    resample_dataloaders,
    save_dataloaders,
    load_dataloaders
)
import os
import time


def test_raw_generation():
    """Test raw waveform generation without any processing."""
    print("\n" + "="*60)
    print("TEST 1: Raw Waveform Generation")
    print("="*60)

    config = {
        'mass1': lambda size: np.random.uniform(20, 50, size=size),
        'mass2': lambda size: np.random.uniform(20, 50, size=size),
    }

    start = time.time()
    output = pycbc_data_generator(
        config, 50,
        num_workers=4,
        add_noise=True,
        time_resolution=1/1024,
        signal_length=1,
        show_progress=False
    )
    elapsed = time.time() - start

    print(f"\n✓ Generated 50 waveforms in {elapsed:.2f}s")
    print(f"  Shape: {output['metadata']['waveform_shape']}")
    print(f"  Preprocessing: {output['metadata']['preprocessing']}")

    # Verify no preprocessing applied
    assert output['metadata']['preprocessing'] == {}, "Raw generation should have empty preprocessing"
    print("✓ Preprocessing is empty (as expected)")

    return output


def test_whitening(raw_output):
    """Test whitening on DataLoaders."""
    print("\n" + "="*60)
    print("TEST 2: Whitening DataLoaders")
    print("="*60)

    start = time.time()
    whitened = whiten_dataloaders(
        raw_output,
        num_workers=4,
        apply_tukey=True,
        tukey_alpha=0.1,
        tukey_side='left',
        show_progress=False
    )
    elapsed = time.time() - start

    print(f"\n✓ Whitened in {elapsed:.2f}s")
    print(f"  Preprocessing: {whitened['metadata']['preprocessing']}")

    # Verify whitening metadata
    assert whitened['metadata']['preprocessing']['whitened'] == True
    assert whitened['metadata']['preprocessing']['whiten_tukey'] == True
    print("✓ Whitening metadata correct")

    return whitened


def test_normalization(whitened_output):
    """Test normalization on DataLoaders."""
    print("\n" + "="*60)
    print("TEST 3: Normalizing DataLoaders")
    print("="*60)

    start = time.time()
    normalized = normalize_dataloaders(whitened_output, scale_factor=100.0)
    elapsed = time.time() - start

    print(f"\n✓ Normalized in {elapsed:.2f}s")
    print(f"  Scale factor: {normalized['metadata']['preprocessing']['normalize_scale']}")

    # Verify normalization metadata
    assert normalized['metadata']['preprocessing']['normalized'] == True
    assert normalized['metadata']['preprocessing']['normalize_scale'] == 100.0
    print("✓ Normalization metadata correct")

    return normalized


def test_truncation(normalized_output):
    """Test truncation on DataLoaders."""
    print("\n" + "="*60)
    print("TEST 4: Truncating DataLoaders")
    print("="*60)

    original_length = normalized_output['metadata']['waveform_shape'][1]

    start = time.time()
    truncated = truncate_dataloaders(normalized_output, target_duration=0.5)
    elapsed = time.time() - start

    new_length = truncated['metadata']['waveform_shape'][1]

    print(f"\n✓ Truncated in {elapsed:.2f}s")
    print(f"  Original: {original_length} samples")
    print(f"  Truncated: {new_length} samples")

    # Verify truncation
    assert new_length < original_length
    assert truncated['metadata']['preprocessing']['truncated'] == True
    print("✓ Truncation metadata correct")

    return truncated


def test_resampling(raw_output):
    """Test resampling on DataLoaders."""
    print("\n" + "="*60)
    print("TEST 5: Resampling DataLoaders")
    print("="*60)

    original_rate = 1.0 / raw_output['metadata']['time_resolution']
    target_rate = original_rate / 2  # Downsample by 2x

    start = time.time()
    resampled = resample_dataloaders(raw_output, target_sample_rate=target_rate)
    elapsed = time.time() - start

    print(f"\n✓ Resampled in {elapsed:.2f}s")
    print(f"  Original rate: {original_rate:.0f} Hz")
    print(f"  New rate: {target_rate:.0f} Hz")
    print(f"  Original shape: {raw_output['metadata']['waveform_shape']}")
    print(f"  Resampled shape: {resampled['metadata']['waveform_shape']}")

    # Verify resampling
    assert resampled['metadata']['preprocessing']['resampled'] == True
    print("✓ Resampling metadata correct")

    return resampled


def test_full_pipeline():
    """Test the complete processing pipeline."""
    print("\n" + "="*60)
    print("TEST 6: Full Pipeline")
    print("="*60)

    config = {
        'mass1': lambda size: np.random.uniform(20, 50, size=size),
        'mass2': lambda size: np.random.uniform(20, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
    }

    print("\nStep 1: Generate...")
    start = time.time()
    output = pycbc_data_generator(
        config, 30,
        num_workers=4,
        add_noise=True,
        time_resolution=1/1024,
        signal_length=2,
        show_progress=False
    )

    print("Step 2: Whiten...")
    output = whiten_dataloaders(output, num_workers=4, show_progress=False)

    print("Step 3: Normalize...")
    output = normalize_dataloaders(output, scale_factor=100.0)

    print("Step 4: Truncate...")
    output = truncate_dataloaders(output, target_duration=1.0)

    elapsed = time.time() - start

    print(f"\n✓ Full pipeline completed in {elapsed:.2f}s")
    print(f"  Final shape: {output['metadata']['waveform_shape']}")
    print(f"  Preprocessing steps: {list(output['metadata']['preprocessing'].keys())}")

    return output


def test_save_load(output):
    """Test saving and loading processed DataLoaders."""
    print("\n" + "="*60)
    print("TEST 7: Save and Load")
    print("="*60)

    test_file = "test_processed_data.pt"

    # Save
    save_dataloaders(output, test_file)
    print(f"✓ Saved to {test_file}")

    # Load
    loaded = load_dataloaders(test_file)
    print(f"✓ Loaded from {test_file}")

    # Verify
    assert loaded['metadata']['waveform_shape'] == output['metadata']['waveform_shape']
    assert loaded['metadata']['preprocessing'] == output['metadata']['preprocessing']
    print("✓ Loaded data matches original")

    # Cleanup
    os.remove(test_file)
    print(f"✓ Cleaned up {test_file}")

    return loaded


def create_comparison_plot(raw_output, whitened_output, normalized_output):
    """Create visualization comparing processing stages."""
    print("\n" + "="*60)
    print("Creating Comparison Plot")
    print("="*60)

    # Get sample waveforms
    raw_loader = raw_output['train_loader']
    whitened_loader = whitened_output['train_loader']
    normalized_loader = normalized_output['train_loader']

    raw_batch, _ = next(iter(raw_loader))
    whitened_batch, _ = next(iter(whitened_loader))
    normalized_batch, _ = next(iter(normalized_loader))

    # Extract single waveform from H1 detector
    raw_wave = raw_batch[0, 0, :].numpy()
    whitened_wave = whitened_batch[0, 0, :].numpy()
    normalized_wave = normalized_batch[0, 0, :].numpy()

    delta_t = raw_output['metadata']['time_resolution']
    time = np.arange(len(raw_wave)) * delta_t

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Raw waveform
    axes[0, 0].plot(time, raw_wave, linewidth=0.5, color='steelblue')
    axes[0, 0].set_title('1. Raw Waveform (with noise)', fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Strain')
    axes[0, 0].grid(True, alpha=0.3)

    # Whitened waveform
    axes[0, 1].plot(time, whitened_wave, linewidth=0.5, color='purple')
    axes[0, 1].set_title('2. Whitened Waveform', fontweight='bold')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Whitened Strain')
    axes[0, 1].grid(True, alpha=0.3)

    # Normalized waveform
    axes[1, 0].plot(time, normalized_wave, linewidth=0.5, color='green')
    axes[1, 0].set_title('3. Normalized Waveform (x100)', fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Normalized Strain')
    axes[1, 0].grid(True, alpha=0.3)

    # Statistics comparison
    stats_text = (
        f"Raw:\n"
        f"  std = {raw_wave.std():.2e}\n"
        f"  range = [{raw_wave.min():.2e}, {raw_wave.max():.2e}]\n\n"
        f"Whitened:\n"
        f"  std = {whitened_wave.std():.2e}\n"
        f"  range = [{whitened_wave.min():.2e}, {whitened_wave.max():.2e}]\n\n"
        f"Normalized:\n"
        f"  std = {normalized_wave.std():.2f}\n"
        f"  range = [{normalized_wave.min():.2f}, {normalized_wave.max():.2f}]"
    )
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].set_title('Statistics Comparison', fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    output_file = 'dataloader_processing_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("DataLoader Processing Functions Test Suite")
    print("="*60)

    # Run tests
    raw = test_raw_generation()
    whitened = test_whitening(raw)
    normalized = test_normalization(whitened)
    truncated = test_truncation(normalized)

    # Test resampling separately (on raw data)
    resampled = test_resampling(raw)

    # Test full pipeline
    full_output = test_full_pipeline()

    # Test save/load
    test_save_load(full_output)

    # Create comparison plot
    create_comparison_plot(raw, whitened, normalized)

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
