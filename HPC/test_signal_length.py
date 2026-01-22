"""
Test script to verify signal_length parameter works correctly.

The signal_length parameter controls the duration of each waveform in seconds.
"""

import numpy as np
import matplotlib.pyplot as plt
from JHPY import pycbc_data_generator

print("=" * 70)
print("Test: signal_length parameter (waveform duration in seconds)")
print("=" * 70)

config = {
    'mass1': lambda size: np.full(size, 30.0),
    'mass2': lambda size: np.full(size, 30.0),
}

# Test different signal lengths
# Note: Very short signals (<2s at low sample rates) may fail PSD estimation
lengths = [2.0, 4.0, 6.0, 8.0]
sample_rate = 2048  # Hz
results = {}

for length in lengths:
    print(f"\nGenerating {length}s waveforms at {sample_rate} Hz...")
    result = pycbc_data_generator(
        config,
        num_samples=5,
        signal_length=length,  # Duration in seconds
        time_resolution=1/sample_rate,
        add_noise=True,
        whiten=True,
        whiten_bandpass=True,
        whiten_tukey=True,
        whiten_tukey_side='left',
        normalize=True,
        normalize_scale=100.0,
        batch_size=5,
        show_progress=False
    )

    wf, params = next(iter(result['train_loader']))
    results[length] = {
        'waveform': wf[0, 0, :].numpy(),
        'metadata': result['metadata']
    }

    expected_samples = int(length * sample_rate)
    actual_samples = wf.shape[2]

    print(f"  Expected samples: {expected_samples}")
    print(f"  Actual samples:   {actual_samples}")
    print(f"  Duration: {actual_samples / sample_rate:.2f}s")

    if actual_samples == expected_samples:
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL - sample count mismatch!")

# ============================================================================
# Create visualization
# ============================================================================
print("\n" + "=" * 70)
print("Creating plots...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

colors = ['red', 'blue', 'green', 'purple']

for idx, (length, data) in enumerate(results.items()):
    wf = data['waveform']
    time = np.arange(len(wf)) / sample_rate

    axes[idx].plot(time, wf, lw=0.8, color=colors[idx])
    axes[idx].set_title(f'Signal Length = {length}s ({len(wf)} samples)',
                        fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Time (s)')
    axes[idx].set_ylabel('Amplitude')
    axes[idx].grid(True, alpha=0.3)

    # Mark the merger location (should be near the end)
    merger_idx = np.argmax(np.abs(wf))
    merger_time = merger_idx / sample_rate
    axes[idx].axvline(x=merger_time, color='orange', linestyle='--', alpha=0.7,
                      label=f'Peak at {merger_time:.2f}s')
    axes[idx].legend(loc='upper left')

fig.suptitle(f'Waveform Duration Test (Sample Rate: {sample_rate} Hz)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('signal_length_test.png', dpi=150, bbox_inches='tight')
print("  Saved: signal_length_test.png")

# ============================================================================
# Summary table
# ============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"\n{'Length (s)':<12} {'Samples':<10} {'Duration (s)':<14} {'Peak Location':<14}")
print("-" * 50)

for length, data in results.items():
    wf = data['waveform']
    samples = len(wf)
    duration = samples / sample_rate
    peak_idx = np.argmax(np.abs(wf))
    peak_time = peak_idx / sample_rate
    print(f"{length:<12} {samples:<10} {duration:<14.2f} {peak_time:<14.2f}")

print("\n" + "=" * 70)
print("Usage example:")
print("=" * 70)
print("""
from JHPY import pycbc_data_generator

# Generate 4-second waveforms at 2048 Hz
result = pycbc_data_generator(
    config,
    num_samples=1000,
    signal_length=4.0,      # Duration in seconds
    time_resolution=1/2048,  # Sample rate
    ...
)

# This will produce waveforms with shape (num_samples, num_detectors, 8192)
# where 8192 = 4.0 seconds * 2048 Hz
""")
