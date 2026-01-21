"""
Test script to verify edge effect fix in whitening pipeline.

This test demonstrates that the edge cropping parameter successfully eliminates
the amplitude spikes caused by filter transients at the start/end of whitened waveforms.
"""

from JHPY import pycbc_data_generator
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("Edge Effects Fix Verification Test")
print("=" * 80)

# Configuration: Fixed masses for consistency
config = {
    'mass1': lambda size: np.full(size, 30.0),
    'mass2': lambda size: np.full(size, 30.0),
}

# ============================================================================
# Test 1: Without Edge Cropping (shows edge effects)
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: Whitening WITHOUT Edge Cropping")
print("=" * 80)

result_with_edges = pycbc_data_generator(
    config,
    num_samples=10,
    add_noise=True,
    whiten=True,
    whiten_bandpass=True,
    whiten_crop_edges=False,  # Disable edge cropping
    normalize=True,
    normalize_scale=100.0,
    batch_size=10,
    show_progress=False
)

waveforms_with_edges, _ = next(iter(result_with_edges['train_loader']))
wf_with_edges = waveforms_with_edges[0, 0, :].numpy()  # First sample, H1 detector

print(f"\nWaveform statistics (WITH edge effects):")
print(f"  Length: {len(wf_with_edges)} samples")
print(f"  Peak at start (first 100 samples): {np.abs(wf_with_edges[:100]).max():.2f}")
print(f"  Peak in middle (samples 4000-4100): {np.abs(wf_with_edges[4000:4100]).max():.2f}")
print(f"  Overall max: {np.abs(wf_with_edges).max():.2f}")

# ============================================================================
# Test 2: With Edge Cropping (default - removes edge effects)
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: Whitening WITH Edge Cropping (Default)")
print("=" * 80)

result_cropped = pycbc_data_generator(
    config,
    num_samples=10,
    add_noise=True,
    whiten=True,
    whiten_bandpass=True,
    whiten_crop_edges=True,   # Enable edge cropping (default)
    whiten_crop_samples=100,  # Crop 100 samples from each edge (default)
    normalize=True,
    normalize_scale=100.0,
    batch_size=10,
    show_progress=False
)

waveforms_cropped, _ = next(iter(result_cropped['train_loader']))
wf_cropped = waveforms_cropped[0, 0, :].numpy()

print(f"\nWaveform statistics (CROPPED edges):")
print(f"  Length: {len(wf_cropped)} samples")
print(f"  Peak at start (first 100 samples): {np.abs(wf_cropped[:100]).max():.2f}")
print(f"  Peak in middle (samples 3900-4000): {np.abs(wf_cropped[3900:4000]).max():.2f}")
print(f"  Overall max: {np.abs(wf_cropped).max():.2f}")

# ============================================================================
# Test 3: Custom Crop Amount
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: Custom Crop Amount (50 samples)")
print("=" * 80)

result_custom = pycbc_data_generator(
    config,
    num_samples=10,
    add_noise=True,
    whiten=True,
    whiten_bandpass=True,
    whiten_crop_edges=True,
    whiten_crop_samples=50,  # Less aggressive cropping
    normalize=True,
    normalize_scale=100.0,
    batch_size=10,
    show_progress=False
)

waveforms_custom, _ = next(iter(result_custom['train_loader']))
wf_custom = waveforms_custom[0, 0, :].numpy()

print(f"\nWaveform statistics (CUSTOM crop - 50 samples):")
print(f"  Length: {len(wf_custom)} samples")
print(f"  Peak at start (first 100 samples): {np.abs(wf_custom[:100]).max():.2f}")
print(f"  Peak in middle (samples 4000-4100): {np.abs(wf_custom[4000:4100]).max():.2f}")
print(f"  Overall max: {np.abs(wf_custom).max():.2f}")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "=" * 80)
print("Creating Comparison Plots")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Time arrays
delta_t = result_with_edges['metadata']['time_resolution']
time_with_edges = np.arange(len(wf_with_edges)) * delta_t
time_cropped = np.arange(len(wf_cropped)) * delta_t
time_custom = np.arange(len(wf_custom)) * delta_t

# Row 1: Full waveforms
axes[0, 0].plot(time_with_edges, wf_with_edges, linewidth=0.8, alpha=0.7, color='red', label='No cropping')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Normalized Strain')
axes[0, 0].set_title('Without Edge Cropping (Edge Effects Present)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(time_cropped, wf_cropped, linewidth=0.8, alpha=0.7, color='green', label='Cropped (100 samples)')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Normalized Strain')
axes[0, 1].set_title('With Edge Cropping (Default: 100 samples)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Row 2: Zoomed to first 0.1 seconds (where edge effects are visible)
zoom_samples_with = int(0.1 / delta_t)
zoom_samples_cropped = int(0.1 / delta_t)

axes[1, 0].plot(time_with_edges[:zoom_samples_with], wf_with_edges[:zoom_samples_with],
                linewidth=1.5, marker='o', markersize=2, color='red')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Normalized Strain')
axes[1, 0].set_title('Zoomed: First 0.1s (NO cropping)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

axes[1, 1].plot(time_cropped[:zoom_samples_cropped], wf_cropped[:zoom_samples_cropped],
                linewidth=1.5, marker='o', markersize=2, color='green')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Normalized Strain')
axes[1, 1].set_title('Zoomed: First 0.1s (WITH cropping)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Row 3: Custom crop comparison and statistics
axes[2, 0].plot(time_custom, wf_custom, linewidth=0.8, alpha=0.7, color='purple', label='Custom (50 samples)')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Normalized Strain')
axes[2, 0].set_title('Custom Crop Amount (50 samples)', fontweight='bold')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Comparison plot: All three overlaid on zoomed region
zoom_end = min(len(wf_with_edges), len(wf_cropped), len(wf_custom))
zoom_samples = min(int(0.15 / delta_t), zoom_end)

axes[2, 1].plot(time_with_edges[:zoom_samples], wf_with_edges[:zoom_samples],
                linewidth=1.2, alpha=0.7, color='red', label='No crop')
axes[2, 1].plot(time_cropped[:zoom_samples], wf_cropped[:zoom_samples],
                linewidth=1.2, alpha=0.7, color='green', label='Crop 100')
axes[2, 1].plot(time_custom[:zoom_samples], wf_custom[:zoom_samples],
                linewidth=1.2, alpha=0.7, color='purple', label='Crop 50')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Normalized Strain')
axes[2, 1].set_title('Overlay Comparison: First 0.15s', fontweight='bold')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('edge_effects_fix_verification.png', dpi=150, bbox_inches='tight')
print("  Saved: edge_effects_fix_verification.png")

# ============================================================================
# Statistical Analysis
# ============================================================================
print("\n" + "=" * 80)
print("Statistical Analysis: Edge vs Middle Peak Ratios")
print("=" * 80)

# Compute ratio of edge peak to middle peak
def compute_edge_to_middle_ratio(waveform):
    edge_peak = np.abs(waveform[:100]).max()
    middle_start = len(waveform) // 2 - 50
    middle_end = len(waveform) // 2 + 50
    middle_peak = np.abs(waveform[middle_start:middle_end]).max()
    return edge_peak / middle_peak if middle_peak > 0 else float('inf')

ratio_no_crop = compute_edge_to_middle_ratio(wf_with_edges)
ratio_crop_100 = compute_edge_to_middle_ratio(wf_cropped)
ratio_crop_50 = compute_edge_to_middle_ratio(wf_custom)

print(f"\nEdge-to-Middle Peak Ratio (lower is better):")
print(f"  No cropping:      {ratio_no_crop:.2f}x")
print(f"  Crop 100 samples: {ratio_crop_100:.2f}x")
print(f"  Crop 50 samples:  {ratio_crop_50:.2f}x")

print("\nInterpretation:")
if ratio_crop_100 < ratio_no_crop * 0.5:
    print("  ✓ Edge cropping (100 samples) successfully reduces edge effects!")
else:
    print("  ⚠ Edge effects still present after cropping")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)

print(f"\n✓ Test 1: Generated {len(wf_with_edges)} sample waveform without edge cropping")
print(f"    - Edge spike detected: {np.abs(wf_with_edges[:100]).max():.2f}")

print(f"\n✓ Test 2: Generated {len(wf_cropped)} sample waveform with default edge cropping")
print(f"    - Length reduced by: {len(wf_with_edges) - len(wf_cropped)} samples")
print(f"    - Edge spike after cropping: {np.abs(wf_cropped[:100]).max():.2f}")

print(f"\n✓ Test 3: Generated {len(wf_custom)} sample waveform with custom cropping (50 samples)")
print(f"    - Length reduced by: {len(wf_with_edges) - len(wf_custom)} samples")

print("\n✓ Visualization created: edge_effects_fix_verification.png")

print("\n" + "=" * 80)
print("Recommendations for LSTM Training")
print("=" * 80)
print("\n1. Use default edge cropping (whiten_crop_edges=True, whiten_crop_samples=100)")
print("   - Eliminates filter transients")
print("   - Minimal signal loss (~0.024s at 4096 Hz)")
print("   - Merger signal (centered) is preserved")

print("\n2. If edge effects are still visible:")
print("   - Increase whiten_crop_samples to 150 or 200")
print("   - Or disable bandpass entirely (whiten_bandpass=False)")

print("\n3. For maximum signal preservation:")
print("   - Reduce to whiten_crop_samples=50")
print("   - Accept small residual edge effects")

print("\n" + "=" * 80)
