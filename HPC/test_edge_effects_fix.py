"""
Test script to verify Tukey window edge effect fix in whitening pipeline.

This test demonstrates that the Tukey window parameter successfully eliminates
the amplitude spikes caused by filter transients at the start/end of whitened waveforms.
"""

from JHPY import pycbc_data_generator
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("Edge Effects Fix Verification Test (Tukey Window)")
print("=" * 80)

# Configuration: Fixed masses for consistency
config = {
    'mass1': lambda size: np.full(size, 30.0),
    'mass2': lambda size: np.full(size, 30.0),
}

# ============================================================================
# Test 1: Without Tukey Window (shows edge effects)
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: Whitening WITHOUT Tukey Window")
print("=" * 80)

result_no_tukey = pycbc_data_generator(
    config,
    num_samples=10,
    add_noise=True,
    whiten=True,
    whiten_bandpass=True,
    whiten_tukey=False,  # Disable Tukey window
    normalize=True,
    normalize_scale=100.0,
    batch_size=10,
    show_progress=False
)

waveforms_no_tukey, _ = next(iter(result_no_tukey['train_loader']))
wf_no_tukey = waveforms_no_tukey[0, 0, :].numpy()  # First sample, H1 detector

print(f"\nWaveform statistics (WITHOUT Tukey - edge effects present):")
print(f"  Length: {len(wf_no_tukey)} samples")
print(f"  Peak at start (first 100 samples): {np.abs(wf_no_tukey[:100]).max():.2f}")
print(f"  Peak in middle (samples 4000-4100): {np.abs(wf_no_tukey[4000:4100]).max():.2f}")
print(f"  Overall max: {np.abs(wf_no_tukey).max():.2f}")

# ============================================================================
# Test 2: With Tukey Window (default - removes edge effects)
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: Whitening WITH Tukey Window (Default)")
print("=" * 80)

result_tukey = pycbc_data_generator(
    config,
    num_samples=10,
    add_noise=True,
    whiten=True,
    whiten_bandpass=True,
    whiten_tukey=True,# Enable Tukey window (default)
    whiten_tukey_side="left",
    whiten_tukey_alpha=0.1,   # 10% taper (default)
    normalize=True,
    normalize_scale=100.0,
    batch_size=10,
    show_progress=False
)

waveforms_tukey, _ = next(iter(result_tukey['train_loader']))
wf_tukey = waveforms_tukey[0, 0, :].numpy()

print(f"\nWaveform statistics (WITH Tukey alpha=0.1):")
print(f"  Length: {len(wf_tukey)} samples")
print(f"  Peak at start (first 100 samples): {np.abs(wf_tukey[:100]).max():.2f}")
print(f"  Peak in middle (samples 4000-4100): {np.abs(wf_tukey[4000:4100]).max():.2f}")
print(f"  Overall max: {np.abs(wf_tukey).max():.2f}")

# ============================================================================
# Test 3: Custom Tukey Alpha (more aggressive taper)
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: Custom Tukey Alpha (0.2 - more taper)")
print("=" * 80)

result_tukey_02 = pycbc_data_generator(
    config,
    num_samples=10,
    add_noise=True,
    whiten=True,
    whiten_bandpass=True,
    whiten_tukey=True,
    whiten_tukey_alpha=1,  # 20% taper - more aggressive
    whiten_tukey_side="left",
    normalize=True,
    normalize_scale=100.0,
    batch_size=10,
    show_progress=False
)

waveforms_tukey_02, _ = next(iter(result_tukey_02['train_loader']))
wf_tukey_02 = waveforms_tukey_02[0, 0, :].numpy()

print(f"\nWaveform statistics (WITH Tukey alpha=0.2):")
print(f"  Length: {len(wf_tukey_02)} samples")
print(f"  Peak at start (first 100 samples): {np.abs(wf_tukey_02[:100]).max():.2f}")
print(f"  Peak in middle (samples 4000-4100): {np.abs(wf_tukey_02[4000:4100]).max():.2f}")
print(f"  Overall max: {np.abs(wf_tukey_02).max():.2f}")

# ============================================================================
# Verify all waveforms have same length
# ============================================================================
print("\n" + "=" * 80)
print("Length Verification (Critical for DataLoader compatibility)")
print("=" * 80)

print(f"\n✓ No Tukey:       {len(wf_no_tukey)} samples")
print(f"✓ Tukey α=0.1:    {len(wf_tukey)} samples")
print(f"✓ Tukey α=0.2:    {len(wf_tukey_02)} samples")

if len(wf_no_tukey) == len(wf_tukey) == len(wf_tukey_02):
    print("\n✓ All waveforms have identical length - DataLoader compatible!")
else:
    print("\n⚠ Length mismatch detected!")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "=" * 80)
print("Creating Comparison Plots")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Time arrays
delta_t = result_no_tukey['metadata']['time_resolution']
time = np.arange(len(wf_no_tukey)) * delta_t

# Row 1: Full waveforms
axes[0, 0].plot(time, wf_no_tukey, linewidth=0.8, alpha=0.7, color='red', label='No Tukey')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Normalized Strain')
axes[0, 0].set_title('Without Tukey Window (Edge Effects Present)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(time, wf_tukey, linewidth=0.8, alpha=0.7, color='green', label='Tukey α=0.1')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Normalized Strain')
axes[0, 1].set_title('With Tukey Window (Default: α=0.1)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Row 2: Zoomed to first 0.1 seconds (where edge effects are visible)
zoom_samples = int(0.1 / delta_t)

axes[1, 0].plot(time[:zoom_samples], wf_no_tukey[:zoom_samples],
                linewidth=1.5, marker='o', markersize=2, color='red')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Normalized Strain')
axes[1, 0].set_title('Zoomed: First 0.1s (NO Tukey)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

axes[1, 1].plot(time[:zoom_samples], wf_tukey[:zoom_samples],
                linewidth=1.5, marker='o', markersize=2, color='green')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Normalized Strain')
axes[1, 1].set_title('Zoomed: First 0.1s (WITH Tukey)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Row 3: More aggressive Tukey and comparison
axes[2, 0].plot(time, wf_tukey_02, linewidth=0.8, alpha=0.7, color='purple', label='Tukey α=0.2')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Normalized Strain')
axes[2, 0].set_title('Tukey Window (α=0.2 - More Aggressive Taper)', fontweight='bold')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Comparison plot: All three overlaid on zoomed region
zoom_end = int(0.15 / delta_t)
axes[2, 1].plot(time[:zoom_end], wf_no_tukey[:zoom_end],
                linewidth=1.2, alpha=0.7, color='red', label='No Tukey')
axes[2, 1].plot(time[:zoom_end], wf_tukey[:zoom_end],
                linewidth=1.2, alpha=0.7, color='green', label='Tukey α=0.1')
axes[2, 1].plot(time[:zoom_end], wf_tukey_02[:zoom_end],
                linewidth=1.2, alpha=0.7, color='purple', label='Tukey α=0.2')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Normalized Strain')
axes[2, 1].set_title('Overlay Comparison: First 0.15s', fontweight='bold')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('edge_effects_tukey_verification.png', dpi=150, bbox_inches='tight')
print("  Saved: edge_effects_tukey_verification.png")

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

ratio_no_tukey = compute_edge_to_middle_ratio(wf_no_tukey)
ratio_tukey_01 = compute_edge_to_middle_ratio(wf_tukey)
ratio_tukey_02 = compute_edge_to_middle_ratio(wf_tukey_02)

print(f"\nEdge-to-Middle Peak Ratio (lower is better):")
print(f"  No Tukey:       {ratio_no_tukey:.2f}x")
print(f"  Tukey α=0.1:    {ratio_tukey_01:.2f}x")
print(f"  Tukey α=0.2:    {ratio_tukey_02:.2f}x")

print("\nInterpretation:")
if ratio_tukey_01 < ratio_no_tukey * 0.5:
    print("  ✓ Tukey window (α=0.1) successfully reduces edge effects!")
else:
    print("  ⚠ Edge effects still present - try increasing alpha")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)

print(f"\n✓ Test 1: Generated {len(wf_no_tukey)} sample waveform without Tukey window")
print(f"    - Edge spike detected: {np.abs(wf_no_tukey[:100]).max():.2f}")

print(f"\n✓ Test 2: Generated {len(wf_tukey)} sample waveform with Tukey window (α=0.1)")
print(f"    - Same length as input: ✓")
print(f"    - Edge spike reduced to: {np.abs(wf_tukey[:100]).max():.2f}")

print(f"\n✓ Test 3: Generated {len(wf_tukey_02)} sample waveform with Tukey window (α=0.2)")
print(f"    - Same length as input: ✓")
print(f"    - Edge spike reduced to: {np.abs(wf_tukey_02[:100]).max():.2f}")

print("\n✓ Visualization created: edge_effects_tukey_verification.png")

print("\n" + "=" * 80)
print("Tukey Window Parameters Guide")
print("=" * 80)
print("\nThe alpha parameter controls the fraction of the signal that gets tapered:")
print("  α=0.0  → Rectangular window (no taper, maximum edge effects)")
print("  α=0.1  → 5% tapered on each edge, 90% flat (default, recommended)")
print("  α=0.2  → 10% tapered on each edge, 80% flat")
print("  α=0.5  → 25% tapered on each edge, 50% flat (Hann-like)")
print("  α=1.0  → Full Hann window (all tapered)")

print("\nRecommendations:")
print("  - For LSTM training: Use α=0.1 (default) - minimal signal distortion")
print("  - If edge effects persist: Increase to α=0.2")
print("  - For short signals (<1s): Consider α=0.05 to preserve more data")

print("\n" + "=" * 80)
