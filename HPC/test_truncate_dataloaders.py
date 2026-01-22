"""
Test script for truncate_dataloaders function.

This function cuts waveforms to a specified length, keeping either
the start or end of the signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from JHPY import pycbc_data_generator, truncate_dataloaders, save_dataloaders, load_dataloaders

print("=" * 70)
print("Test: truncate_dataloaders function")
print("=" * 70)

config = {
    'mass1': lambda size: np.random.uniform(20, 40, size),
    'mass2': lambda size: np.random.uniform(20, 40, size),
}

# Generate 2-second waveforms
print("\n1. Generating 2-second waveforms at 1024 Hz...")
result_original = pycbc_data_generator(
    config,
    num_samples=50,
    signal_length=2.0,
    time_resolution=1/1024,
    add_noise=True,
    whiten=True,
    whiten_tukey=True,
    whiten_tukey_side='left',
    normalize=True,
    batch_size=10,
    show_progress=False
)

wf_original, params = next(iter(result_original['train_loader']))
print(f"   Original shape: {wf_original.shape}")
print(f"   Original length: {wf_original.shape[2]} samples ({wf_original.shape[2]/1024:.2f}s)")

# ============================================================================
# Test 1: Truncate by duration (keep end - where merger is)
# ============================================================================
print("\n2. Truncating to 1 second (keeping END - merger region)...")
result_1s_end = truncate_dataloaders(
    result_original,
    target_duration=1.0,
    keep_end=True
)

wf_1s_end, _ = next(iter(result_1s_end['train_loader']))
print(f"   New shape: {wf_1s_end.shape}")

# ============================================================================
# Test 2: Truncate by samples (keep start)
# ============================================================================
print("\n3. Truncating to 512 samples (keeping START)...")
result_512_start = truncate_dataloaders(
    result_original,
    target_length=512,
    keep_end=False
)

wf_512_start, _ = next(iter(result_512_start['train_loader']))
print(f"   New shape: {wf_512_start.shape}")

# ============================================================================
# Test 3: Save and load truncated data
# ============================================================================
print("\n4. Testing save/load with truncated data...")
save_dataloaders(result_1s_end, 'test_truncated.pt')
result_loaded = load_dataloaders('test_truncated.pt')
wf_loaded, _ = next(iter(result_loaded['train_loader']))
print(f"   Loaded shape: {wf_loaded.shape}")

# Verify data matches
if np.allclose(wf_1s_end.numpy(), wf_loaded.numpy()):
    print("   ✓ Data matches after save/load!")
else:
    print("   ✗ Data mismatch!")

# Clean up test file
import os
os.remove('test_truncated.pt')

# ============================================================================
# Visualization
# ============================================================================
print("\n5. Creating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Get single waveforms for plotting
wf_orig = wf_original[0, 0, :].numpy()
wf_end = wf_1s_end[0, 0, :].numpy()
wf_start = wf_512_start[0, 0, :].numpy()

time_orig = np.arange(len(wf_orig)) / 1024
time_end = np.arange(len(wf_end)) / 1024
time_start = np.arange(len(wf_start)) / 1024

# Original
axes[0, 0].plot(time_orig, wf_orig, lw=0.8, color='blue')
axes[0, 0].set_title(f'Original: {len(wf_orig)} samples ({len(wf_orig)/1024:.2f}s)', fontweight='bold')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1s mark')
axes[0, 0].legend()

# Truncated (keep end)
axes[0, 1].plot(time_end, wf_end, lw=0.8, color='green')
axes[0, 1].set_title(f'Keep END: {len(wf_end)} samples ({len(wf_end)/1024:.2f}s)', fontweight='bold')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].grid(True, alpha=0.3)

# Truncated (keep start)
axes[1, 0].plot(time_start, wf_start, lw=0.8, color='orange')
axes[1, 0].set_title(f'Keep START: {len(wf_start)} samples ({len(wf_start)/1024:.2f}s)', fontweight='bold')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].grid(True, alpha=0.3)

# Overlay comparison
axes[1, 1].plot(time_orig, wf_orig, lw=0.8, color='blue', alpha=0.5, label='Original (2s)')
# Shift truncated end to align with original
time_end_shifted = time_end + (len(wf_orig) - len(wf_end)) / 1024
axes[1, 1].plot(time_end_shifted, wf_end, lw=1, color='green', label='Keep END (1s)')
axes[1, 1].plot(time_start, wf_start, lw=1, color='orange', label='Keep START (0.5s)')
axes[1, 1].set_title('Overlay Comparison', fontweight='bold')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('truncate_dataloaders_test.png', dpi=150, bbox_inches='tight')
print("   Saved: truncate_dataloaders_test.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"""
Original:        {result_original['metadata']['waveform_shape']}
Truncated (end): {result_1s_end['metadata']['waveform_shape']}
Truncated (start): {result_512_start['metadata']['waveform_shape']}

Usage:
------
from JHPY import pycbc_data_generator, truncate_dataloaders

# Generate longer signals
result = pycbc_data_generator(config, num_samples=1000, signal_length=2.0)

# Keep only the last 1 second (merger region)
result_truncated = truncate_dataloaders(
    result,
    target_duration=1.0,  # or target_length=1024 for samples
    keep_end=True         # Keep merger at end
)

# Or keep the start (inspiral region)
result_inspiral = truncate_dataloaders(
    result,
    target_duration=0.5,
    keep_end=False
)
""")
