"""
Test script to verify GPS time variation in GW waveform generation.

GPS time affects the detector antenna pattern (via Earth's rotation) and
inter-detector time delays. Varying GPS time should produce measurably
different detector responses for the same source parameters.

Also tests that varying ra/dec alongside GPS time produces further variation.
"""

from JHPY import pycbc_data_generator
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("GPS Time Variation Test for GW Waveforms")
print("=" * 80)

# ─── Test 1: Direct waveform comparison at different GPS times ───
print("\n--- Test 1: Are waveforms actually different at different GPS times? ---")
print("Generating two sets with identical params except GPS time...\n")

fixed_mass1 = 30.0
fixed_mass2 = 30.0
fixed_distance = 410.0
gw150914_gps = 1126259462.4

def make_config(gps_time, ra=1.95, dec=-1.27):
    return {
        'mass1': lambda size: np.full(size, fixed_mass1),
        'mass2': lambda size: np.full(size, fixed_mass2),
        'distance': lambda size: np.full(size, fixed_distance),
        'spin1z': lambda size: np.zeros(size),
        'spin2z': lambda size: np.zeros(size),
        'ra': lambda size, r=ra: np.full(size, r),
        'dec': lambda size, d=dec: np.full(size, d),
        'polarization': lambda size: np.full(size, 0.82),
        'gps_time': lambda size, t=gps_time: np.full(size, t),
    }

# Generate at two GPS times 12 hours apart
result_A = pycbc_data_generator(
    make_config(gw150914_gps),
    num_samples=3, add_noise=False, batch_size=3, show_progress=False,
)
result_B = pycbc_data_generator(
    make_config(gw150914_gps + 12 * 3600),
    num_samples=3, add_noise=False, batch_size=3, show_progress=False,
)

wf_A, _ = next(iter(result_A['train_loader']))
wf_B, _ = next(iter(result_B['train_loader']))

h1_A = wf_A[0, 0, :].numpy()
h1_B = wf_B[0, 0, :].numpy()
l1_A = wf_A[0, 1, :].numpy()
l1_B = wf_B[0, 1, :].numpy()

# Direct array comparison
h1_identical = np.allclose(h1_A, h1_B, atol=1e-30)
l1_identical = np.allclose(l1_A, l1_B, atol=1e-30)
h1_max_diff = np.max(np.abs(h1_A - h1_B))
l1_max_diff = np.max(np.abs(l1_A - l1_B))
h1_corr = np.corrcoef(h1_A, h1_B)[0, 1]
l1_corr = np.corrcoef(l1_A, l1_B)[0, 1]

print(f"  H1 arrays identical?  {h1_identical}  (max diff: {h1_max_diff:.4e})")
print(f"  L1 arrays identical?  {l1_identical}  (max diff: {l1_max_diff:.4e})")
print(f"  H1 correlation:       {h1_corr:.6f}")
print(f"  L1 correlation:       {l1_corr:.6f}")

if not h1_identical and not l1_identical:
    print("\n  PASS: Waveforms are genuinely different at different GPS times.")
else:
    print("\n  FAIL: Waveforms appear identical - GPS time may not be applied.")

# ─── Test 2: GPS time sweep with varying ra/dec ───
print("\n\n--- Test 2: GPS time sweep + ra/dec variation ---")

gps_times = [
    gw150914_gps,
    gw150914_gps + 6 * 3600,
    gw150914_gps + 12 * 3600,
    gw150914_gps + 18 * 3600,
]
gps_labels = ["GW150914", "+6h", "+12h", "+18h"]

# Also vary ra/dec slightly (within ~0.3 rad ~ 17 degrees)
ra_values = [1.95, 2.10, 1.80, 2.25]
dec_values = [-1.27, -1.10, -1.40, -0.95]

print(f"\n  {'Label':>10s}  {'GPS time':>16s}  {'RA':>6s}  {'Dec':>6s}")
print("  " + "-" * 50)
for label, gps, ra, dec in zip(gps_labels, gps_times, ra_values, dec_values):
    print(f"  {label:>10s}  {gps:>16.1f}  {ra:>6.2f}  {dec:>6.2f}")

results = {}
for gps_time, label, ra, dec in zip(gps_times, gps_labels, ra_values, dec_values):
    print(f"\n  Generating: {label} (GPS={gps_time:.1f}, ra={ra:.2f}, dec={dec:.2f})...")
    result = pycbc_data_generator(
        make_config(gps_time, ra=ra, dec=dec),
        num_samples=3, add_noise=False, batch_size=3, show_progress=False,
    )
    results[label] = result

# ─── Plot: overlay all on same axes for direct comparison ───
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
waveforms_h1 = {}
waveforms_l1 = {}

for (label, result), color in zip(results.items(), colors):
    wf, _ = next(iter(result['train_loader']))
    h1 = wf[0, 0, :].numpy()
    l1 = wf[0, 1, :].numpy()
    waveforms_h1[label] = h1
    waveforms_l1[label] = l1
    time = np.arange(len(h1)) * (1/4096)

    axes[0].plot(time, h1, label=label, linewidth=0.8, alpha=0.85, color=color)
    axes[1].plot(time, l1, label=label, linewidth=0.8, alpha=0.85, color=color)

axes[0].set_ylabel('Strain')
axes[0].set_title('H1 Detector - Same Source, Different GPS Times + Sky Locations', fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel('Strain')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('L1 Detector - Same Source, Different GPS Times + Sky Locations', fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gps_time_variation_waveforms.png', dpi=150, bbox_inches='tight')
print("\n  Saved: gps_time_variation_waveforms.png")

# ─── Zoom into merger region ───
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

# Last 0.15s (merger + ringdown)
zoom_samples = int(0.15 * 4096)
for (label, h1), color in zip(waveforms_h1.items(), colors):
    t_zoom = np.arange(zoom_samples) * (1/4096)
    axes[0].plot(t_zoom, h1[-zoom_samples:], label=label, linewidth=1.0, color=color)
for (label, l1), color in zip(waveforms_l1.items(), colors):
    t_zoom = np.arange(zoom_samples) * (1/4096)
    axes[1].plot(t_zoom, l1[-zoom_samples:], label=label, linewidth=1.0, color=color)

axes[0].set_ylabel('Strain')
axes[0].set_title('H1 Merger Region (last 0.15s)', fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[1].set_ylabel('Strain')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('L1 Merger Region (last 0.15s)', fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gps_time_variation_merger_zoom.png', dpi=150, bbox_inches='tight')
print("  Saved: gps_time_variation_merger_zoom.png")

# ─── Quantitative summary ───
print("\n" + "=" * 80)
print("Quantitative Comparison")
print("=" * 80)

labels = list(waveforms_h1.keys())
baseline_h1 = waveforms_h1[labels[0]]
baseline_l1 = waveforms_l1[labels[0]]

print(f"\n  {'Case':>10s}  {'H1 peak':>11s}  {'L1 peak':>11s}  {'H1/L1':>7s}  "
      f"{'H1 corr vs base':>16s}  {'L1 corr vs base':>16s}")
print("  " + "-" * 80)

for label in labels:
    h1 = waveforms_h1[label]
    l1 = waveforms_l1[label]
    h1_peak = np.abs(h1).max()
    l1_peak = np.abs(l1).max()
    ratio = h1_peak / l1_peak if l1_peak > 0 else float('inf')
    h1_corr = np.corrcoef(baseline_h1, h1)[0, 1]
    l1_corr = np.corrcoef(baseline_l1, l1)[0, 1]
    print(f"  {label:>10s}  {h1_peak:>11.4e}  {l1_peak:>11.4e}  {ratio:>7.3f}  "
          f"{h1_corr:>16.6f}  {l1_corr:>16.6f}")

# ─── Pairwise difference check ───
print(f"\n  Pairwise max absolute difference:")
print(f"  {'Pair':>20s}  {'H1 max diff':>12s}  {'L1 max diff':>12s}")
print("  " + "-" * 50)
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        h1_diff = np.max(np.abs(waveforms_h1[labels[i]] - waveforms_h1[labels[j]]))
        l1_diff = np.max(np.abs(waveforms_l1[labels[i]] - waveforms_l1[labels[j]]))
        print(f"  {labels[i]+' vs '+labels[j]:>20s}  {h1_diff:>12.4e}  {l1_diff:>12.4e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
all_h1_peaks = [np.abs(waveforms_h1[l]).max() for l in labels]
all_l1_peaks = [np.abs(waveforms_l1[l]).max() for l in labels]
h1_var = max(all_h1_peaks) / min(all_h1_peaks)
l1_var = max(all_l1_peaks) / min(all_l1_peaks)
print(f"\n  H1 peak variation (max/min): {h1_var:.3f}x")
print(f"  L1 peak variation (max/min): {l1_var:.3f}x")

any_diff = not np.allclose(waveforms_h1[labels[0]], waveforms_h1[labels[1]], atol=1e-30)
if any_diff and h1_var > 1.01:
    print("\n  PASS: GPS time + sky location variation produces genuinely different waveforms.")
else:
    print("\n  FAIL: Waveforms not sufficiently different.")
print("=" * 80)
