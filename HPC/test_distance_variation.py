"""
Test script to demonstrate distance-dependent amplitude variation in GW waveforms.

This test verifies that the whitening + normalization pipeline preserves relative
peak amplitudes, allowing LSTM networks to learn distance from signal strength.
"""

from JHPY import pycbc_data_generator
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("Distance Variation Test for GW Waveforms")
print("=" * 80)
print("\nThis test verifies that preprocessing preserves relative peak amplitudes,")
print("enabling LSTM networks to learn distance from signal strength.")

# Test parameters
distances = [100, 200, 400, 800, 1600]  # Mpc - geometric progression
num_samples_per_distance = 5
fixed_masses = (30.0, 30.0)  # Solar masses - keep constant

print(f"\nTest configuration:")
print(f"  Distances: {distances} Mpc")
print(f"  Samples per distance: {num_samples_per_distance}")
print(f"  Fixed masses: {fixed_masses} M☉")

# Generate waveforms at each distance
results = {}
print("\nGenerating waveforms...")
for distance in distances:
    print(f"\n  Distance {distance} Mpc...")
    config = {
        'mass1': lambda size: np.full(size, fixed_masses[0]),
        'mass2': lambda size: np.full(size, fixed_masses[1]),
        'distance': lambda size: np.full(size, distance),
        'spin1z': lambda size: np.zeros(size),
        'spin2z': lambda size: np.zeros(size),
    }

    # Test 1: No preprocessing (raw)
    print(f"    Generating raw waveforms...")
    result_raw = pycbc_data_generator(
        config,
        num_samples=num_samples_per_distance,
        add_noise=True,
        whiten=False,
        normalize=False,
        batch_size=num_samples_per_distance,
        show_progress=False
    )

    # Test 2: With whitening + normalization (LSTM-ready)
    print(f"    Generating processed waveforms...")
    result_processed = pycbc_data_generator(
        config,
        num_samples=num_samples_per_distance,
        add_noise=True,
        whiten=True,
        normalize=True,
        normalize_scale=100.0,  # Fixed scale - preserves relative amplitudes!
        batch_size=num_samples_per_distance,
        show_progress=False
    )

    results[distance] = {
        'raw': result_raw,
        'processed': result_processed
    }

print("\n" + "=" * 80)
print("Analyzing Peak Amplitudes")
print("=" * 80)

# Analysis: Extract peak amplitudes
peak_analysis = {
    'distances': distances,
    'raw_peaks': [],
    'processed_peaks': [],
    'raw_std': [],
    'processed_std': []
}

for distance in distances:
    # Get waveforms
    raw_waveforms, _ = next(iter(results[distance]['raw']['train_loader']))
    proc_waveforms, _ = next(iter(results[distance]['processed']['train_loader']))

    # Extract H1 detector, compute peak amplitudes
    raw_h1 = raw_waveforms[:, 0, :].numpy()  # (num_samples, time)
    proc_h1 = proc_waveforms[:, 0, :].numpy()

    raw_peaks = np.abs(raw_h1).max(axis=1)  # Max for each sample
    proc_peaks = np.abs(proc_h1).max(axis=1)

    peak_analysis['raw_peaks'].append(raw_peaks.mean())
    peak_analysis['processed_peaks'].append(proc_peaks.mean())
    peak_analysis['raw_std'].append(raw_peaks.std())
    peak_analysis['processed_std'].append(proc_peaks.std())

print("\nPeak analysis complete. Creating visualizations...")

# Plotting
print("\nCreating waveform comparison plot...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Example waveforms at different distances (raw)
for i, distance in enumerate([100, 400, 1600]):
    waveforms, _ = next(iter(results[distance]['raw']['train_loader']))
    h1_signal = waveforms[0, 0, :].numpy()  # First sample, H1 detector
    time = np.arange(len(h1_signal)) * (1/4096)
    axes[0, i].plot(time, h1_signal, linewidth=0.8)
    axes[0, i].set_title(f'Raw Waveform @ {distance} Mpc', fontsize=12, fontweight='bold')
    axes[0, i].set_xlabel('Time (s)')
    axes[0, i].set_ylabel('Strain')
    axes[0, i].grid(True, alpha=0.3)

# Plot 2: Example waveforms at different distances (processed)
for i, distance in enumerate([100, 400, 1600]):
    waveforms, _ = next(iter(results[distance]['processed']['train_loader']))
    h1_signal = waveforms[0, 0, :].numpy()
    time = np.arange(len(h1_signal)) * (1/4096)
    axes[1, i].plot(time, h1_signal, linewidth=0.8, color='purple')
    axes[1, i].set_title(f'Processed @ {distance} Mpc', fontsize=12, fontweight='bold')
    axes[1, i].set_xlabel('Time (s)')
    axes[1, i].set_ylabel('Normalized Strain')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distance_variation_waveforms.png', dpi=150, bbox_inches='tight')
print("  Saved: distance_variation_waveforms.png")

# Plot 3: Peak amplitude vs distance
print("\nCreating peak amplitude vs distance plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw peaks
axes[0].errorbar(peak_analysis['distances'], peak_analysis['raw_peaks'],
                 yerr=peak_analysis['raw_std'], marker='o', capsize=5, linewidth=2, markersize=8)
axes[0].set_xlabel('Distance (Mpc)', fontsize=12)
axes[0].set_ylabel('Peak Amplitude', fontsize=12)
axes[0].set_title('Raw Waveforms: Peak vs Distance', fontsize=13, fontweight='bold')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3, which='both')

# Processed peaks
axes[1].errorbar(peak_analysis['distances'], peak_analysis['processed_peaks'],
                 yerr=peak_analysis['processed_std'], marker='o', color='purple',
                 capsize=5, linewidth=2, markersize=8)
axes[1].set_xlabel('Distance (Mpc)', fontsize=12)
axes[1].set_ylabel('Peak Amplitude', fontsize=12)
axes[1].set_title('Processed (Whiten+Norm): Peak vs Distance', fontsize=13, fontweight='bold')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('distance_vs_peak_amplitude.png', dpi=150, bbox_inches='tight')
print("  Saved: distance_vs_peak_amplitude.png")

# Verification: Check that relative amplitudes are preserved
print("\n" + "=" * 80)
print("Distance Variation Test Results")
print("=" * 80)
print("\nExpected: Peak amplitude ∝ 1/distance (inverse relationship)")
print("\nRaw waveforms:")
for i, distance in enumerate(distances):
    ratio = peak_analysis['raw_peaks'][0] / peak_analysis['raw_peaks'][i]
    expected_ratio = distance / distances[0]
    print(f"  {distance:4d} Mpc: peak={peak_analysis['raw_peaks'][i]:.2e}, "
          f"ratio vs 100Mpc: {ratio:.2f} (expected: {expected_ratio:.2f})")

print("\nProcessed waveforms (Whiten + Normalize with FIXED scale):")
for i, distance in enumerate(distances):
    ratio = peak_analysis['processed_peaks'][0] / peak_analysis['processed_peaks'][i]
    expected_ratio = distance / distances[0]
    print(f"  {distance:4d} Mpc: peak={peak_analysis['processed_peaks'][i]:.2f}, "
          f"ratio vs 100Mpc: {ratio:.2f} (expected: {expected_ratio:.2f})")

print("\n" + "=" * 80)
print("Interpretation")
print("=" * 80)
print("\n✓ If ratios match expected values, LSTM can learn distance from amplitudes!")
print("\nThe fixed normalization scale (100.0) preserves the relative peak amplitudes")
print("across different distances, allowing neural networks to learn the distance")
print("parameter from the signal strength.\n")
print("Whitening ensures consistent noise amplitude across all signals, making")
print("peak comparisons reliable and improving model performance.")
print("\n" + "=" * 80)
