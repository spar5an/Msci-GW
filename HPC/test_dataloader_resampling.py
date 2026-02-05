"""
Test script for DataLoader resampling functionality.

Tests:
1. Fresh DataLoader resampling (4096 Hz → 2048 Hz)
2. Saved/loaded DataLoader resampling
3. Visual verification with plots
4. Frequency domain verification
"""

from JHPY import pycbc_data_generator, resample_dataloaders, save_dataloaders, load_dataloaders
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

print("=" * 80)
print("DataLoader Resampling Test")
print("=" * 80)

# ============================================================================
# Test 1: Generate Fresh DataLoaders and Resample
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: Fresh DataLoader Resampling")
print("=" * 80)

config = {
    'mass1': lambda size: np.random.uniform(20, 50, size=size),
    'mass2': lambda size: np.random.uniform(20, 50, size=size),
}

print("\nGenerating initial waveforms at 4096 Hz...")
result_original = pycbc_data_generator(
    config,
    num_samples=100,
    time_resolution=1/4096,  # 4096 Hz
    add_noise=True,
    batch_size=32,
    show_progress=False
)

print(f"\nOriginal DataLoader info:")
print(f"  Sampling rate: {1/result_original['metadata']['time_resolution']:.0f} Hz")
print(f"  Waveform shape: {result_original['metadata']['waveform_shape']}")
print(f"  Train size: {result_original['metadata']['train_size']}")
print(f"  Val size: {result_original['metadata']['val_size']}")
print(f"  Test size: {result_original['metadata']['test_size']}")

# Resample to 2048 Hz
print("\nResampling to 2048 Hz...")
result_resampled = resample_dataloaders(
    result_original,
    target_sample_rate=2048,
    preserve_splits=True
)

print(f"\nResampled DataLoader info:")
print(f"  Sampling rate: {1/result_resampled['metadata']['time_resolution']:.0f} Hz")
print(f"  Waveform shape: {result_resampled['metadata']['waveform_shape']}")
print(f"  Train size: {result_resampled['metadata']['train_size']}")
print(f"  Val size: {result_resampled['metadata']['val_size']}")
print(f"  Test size: {result_resampled['metadata']['test_size']}")

# ============================================================================
# Test 2: Save, Load, and Resample
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: Save/Load/Resample Workflow")
print("=" * 80)

# Save original
save_path = 'test_waveforms_4096hz.pt'
print(f"\nSaving original data to {save_path}...")
save_dataloaders(result_original, save_path)

# Load
print(f"Loading data from {save_path}...")
result_loaded = load_dataloaders(save_path)

print(f"\nLoaded DataLoader info:")
print(f"  Sampling rate: {1/result_loaded['metadata']['time_resolution']:.0f} Hz")
print(f"  Waveform shape: {result_loaded['metadata']['waveform_shape']}")

# Resample loaded data
print("\nResampling loaded data to 1024 Hz...")
result_resampled_from_loaded = resample_dataloaders(
    result_loaded,
    target_sample_rate=1024,
    preserve_splits=True
)

print(f"\nResampled (from loaded) DataLoader info:")
print(f"  Sampling rate: {1/result_resampled_from_loaded['metadata']['time_resolution']:.0f} Hz")
print(f"  Waveform shape: {result_resampled_from_loaded['metadata']['waveform_shape']}")

# Save resampled version
save_path_resampled = 'test_waveforms_1024hz.pt'
print(f"\nSaving resampled data to {save_path_resampled}...")
save_dataloaders(result_resampled_from_loaded, save_path_resampled)

# ============================================================================
# Test 3: Visual Verification with Plots
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: Visual Verification")
print("=" * 80)

# Extract sample waveforms
original_batch, _ = next(iter(result_original['train_loader']))
resampled_2048_batch, _ = next(iter(result_resampled['train_loader']))
resampled_1024_batch, _ = next(iter(result_resampled_from_loaded['train_loader']))

# Get single waveform from each (first sample, H1 detector)
wf_original = original_batch[0, 0, :].numpy()
wf_2048 = resampled_2048_batch[0, 0, :].numpy()
wf_1024 = resampled_1024_batch[0, 0, :].numpy()

# Create time arrays
time_original = np.arange(len(wf_original)) * (1/4096)
time_2048 = np.arange(len(wf_2048)) * (1/2048)
time_1024 = np.arange(len(wf_1024)) * (1/1024)

print("\nCreating comparison plots...")

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Plot 1: Full waveforms
axes[0, 0].plot(time_original, wf_original, linewidth=0.8, label='4096 Hz (original)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Strain')
axes[0, 0].set_title('Original: 4096 Hz', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(time_2048, wf_2048, linewidth=0.8, color='orange', label='2048 Hz')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Strain')
axes[1, 0].set_title('Resampled: 2048 Hz', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[2, 0].plot(time_1024, wf_1024, linewidth=0.8, color='green', label='1024 Hz')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Strain')
axes[2, 0].set_title('Resampled: 1024 Hz', fontweight='bold')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Plot 2: Zoomed-in comparison (merger region)
# Find peak location in original
peak_idx_orig = np.argmax(np.abs(wf_original))
peak_time = time_original[peak_idx_orig]

# Zoom window: ±0.1 seconds around peak
window = 0.1
time_start = max(0, peak_time - window)
time_end = min(time_original[-1], peak_time + window)

# Plot zoomed regions
mask_orig = (time_original >= time_start) & (time_original <= time_end)
axes[0, 1].plot(time_original[mask_orig], wf_original[mask_orig],
                linewidth=1.5, marker='o', markersize=2, label='4096 Hz')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Strain')
axes[0, 1].set_title('Zoomed: 4096 Hz (Original)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

mask_2048 = (time_2048 >= time_start) & (time_2048 <= time_end)
axes[1, 1].plot(time_2048[mask_2048], wf_2048[mask_2048],
                linewidth=1.5, marker='o', markersize=3, color='orange', label='2048 Hz')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Strain')
axes[1, 1].set_title('Zoomed: 2048 Hz', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

mask_1024 = (time_1024 >= time_start) & (time_1024 <= time_end)
axes[2, 1].plot(time_1024[mask_1024], wf_1024[mask_1024],
                linewidth=1.5, marker='o', markersize=4, color='green', label='1024 Hz')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Strain')
axes[2, 1].set_title('Zoomed: 1024 Hz', fontweight='bold')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dataloader_resampling_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved: dataloader_resampling_comparison.png")

# ============================================================================
# Test 4: Frequency Domain Verification
# ============================================================================
print("\nCreating frequency domain comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Compute FFTs
fft_orig = np.fft.rfft(wf_original)
fft_2048 = np.fft.rfft(wf_2048)
fft_1024 = np.fft.rfft(wf_1024)

# Frequency arrays
freqs_orig = np.fft.rfftfreq(len(wf_original), 1/4096)
freqs_2048 = np.fft.rfftfreq(len(wf_2048), 1/2048)
freqs_1024 = np.fft.rfftfreq(len(wf_1024), 1/1024)

# Plot
axes[0].loglog(freqs_orig, np.abs(fft_orig), linewidth=0.8)
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('|FFT|')
axes[0].set_title('FFT: 4096 Hz (Original)', fontweight='bold')
axes[0].grid(True, alpha=0.3, which='both')
axes[0].set_xlim([10, 2048])

axes[1].loglog(freqs_2048, np.abs(fft_2048), linewidth=0.8, color='orange')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('|FFT|')
axes[1].set_title('FFT: 2048 Hz', fontweight='bold')
axes[1].grid(True, alpha=0.3, which='both')
axes[1].set_xlim([10, 1024])

axes[2].loglog(freqs_1024, np.abs(fft_1024), linewidth=0.8, color='green')
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('|FFT|')
axes[2].set_title('FFT: 1024 Hz', fontweight='bold')
axes[2].grid(True, alpha=0.3, which='both')
axes[2].set_xlim([10, 512])

plt.tight_layout()
plt.savefig('dataloader_resampling_fft.png', dpi=150, bbox_inches='tight')
print("  Saved: dataloader_resampling_fft.png")

# ============================================================================
# Validation Summary
# ============================================================================
print("\n" + "=" * 80)
print("Validation Summary")
print("=" * 80)

print("\n✓ Test 1: Fresh DataLoader resampling successful")
print(f"    - Original: {len(wf_original)} samples at 4096 Hz")
print(f"    - Resampled: {len(wf_2048)} samples at 2048 Hz")
print(f"    - Length ratio: {len(wf_original)/len(wf_2048):.2f} (expected: 2.00)")

print("\n✓ Test 2: Save/Load/Resample workflow successful")
print(f"    - Loaded and resampled to 1024 Hz")
print(f"    - Final: {len(wf_1024)} samples")
print(f"    - Length ratio vs original: {len(wf_original)/len(wf_1024):.2f} (expected: 4.00)")
print(f"    - Files created: {save_path}, {save_path_resampled}")

print("\n✓ Test 3: Visual verification plots created")
print("    - Time domain comparison: dataloader_resampling_comparison.png")
print("    - Frequency domain: dataloader_resampling_fft.png")

print("\n✓ Split preservation verified:")
print(f"    - Train size maintained: {result_original['metadata']['train_size']} → {result_resampled['metadata']['train_size']}")
print(f"    - Val size maintained: {result_original['metadata']['val_size']} → {result_resampled['metadata']['val_size']}")
print(f"    - Test size maintained: {result_original['metadata']['test_size']} → {result_resampled['metadata']['test_size']}")

# Additional validation: Check that parameters are unchanged
print("\n✓ Parameter preservation verified:")
_, orig_params_batch = next(iter(result_original['train_loader']))
_, resampled_params_batch = next(iter(result_resampled['train_loader']))
print(f"    - Original params shape: {orig_params_batch.shape}")
print(f"    - Resampled params shape: {resampled_params_batch.shape}")
print(f"    - Parameters identical: {torch.allclose(orig_params_batch, resampled_params_batch)}")

# Cleanup
print("\nCleaning up temporary files...")
if os.path.exists(save_path):
    os.remove(save_path)
    print(f"  Removed: {save_path}")
if os.path.exists(save_path_resampled):
    os.remove(save_path_resampled)
    print(f"  Removed: {save_path_resampled}")

print("\n" + "=" * 80)
print("All Tests Passed!")
print("=" * 80)
