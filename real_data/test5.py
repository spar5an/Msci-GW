"""
test5.py — Diagnose whitening artifacts in the real-data pipeline.

Downloads GW150914, processes it with different whitening configurations,
and produces diagnostic plots to identify the source of edge spikes and
ringing. Compares JHPY's custom whitening with gwpy's native whitening.
"""

from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from JHPY import whiten_waveform
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──
SAMPLE_RATE = 4096
DOWNLOAD_WINDOW = 16     # seconds before merger
POST_MERGER = 1.0        # seconds after merger we want to keep
EDGE_BUFFER = 3.0        # extra seconds after POST_MERGER (cropped after whitening)
TOTAL_DOWNLOAD = DOWNLOAD_WINDOW + POST_MERGER + EDGE_BUFFER  # 20s
DOWNLOAD_LENGTH = int(TOTAL_DOWNLOAD * SAMPLE_RATE)
PRE_MERGER = 1.5         # seconds before merger in final output
FINAL_DURATION = PRE_MERGER + POST_MERGER  # 2.5s
FINAL_LENGTH = int(FINAL_DURATION * SAMPLE_RATE)

# GW150914 parameters
EVENT = 'GW150914'
GPS = 1126259462.408332
DET = 'H1'

# ── 1. Download GW150914 with extra buffer ──
print("=" * 70)
print("test5.py — Whitening artifact diagnosis")
print("=" * 70)

start = GPS - DOWNLOAD_WINDOW
end = GPS + POST_MERGER + EDGE_BUFFER
print(f"\n[1] Downloading {EVENT} {DET}: {TOTAL_DOWNLOAD}s "
      f"[GPS-{DOWNLOAD_WINDOW}, GPS+{POST_MERGER + EDGE_BUFFER}]...")

ts = GWpyTimeSeries.fetch_open_data(DET, start, end, sample_rate=SAMPLE_RATE)
raw = ts.value.astype(np.float32)
if len(raw) >= DOWNLOAD_LENGTH:
    raw = raw[:DOWNLOAD_LENGTH]
else:
    raw = np.pad(raw, (0, DOWNLOAD_LENGTH - len(raw)), mode='constant')
print(f"  Got {len(raw)} samples ({len(raw)/SAMPLE_RATE:.1f}s)")

delta_t = 1.0 / SAMPLE_RATE

# Merger is at exactly DOWNLOAD_WINDOW seconds into the signal
merger_sample = int(DOWNLOAD_WINDOW * SAMPLE_RATE)

# ── 2. Run whitening variants ──
print("\n[2] Running whitening variants...")

# Helper: crop to [-PRE_MERGER, +POST_MERGER] around merger from full signal
def crop_around_merger(signal, merger_idx, pre_samples, post_samples):
    """Crop signal to [merger_idx - pre_samples, merger_idx + post_samples]."""
    return signal[merger_idx - pre_samples:merger_idx + post_samples]

pre_samp = int(PRE_MERGER * SAMPLE_RATE)
post_samp = int(POST_MERGER * SAMPLE_RATE)

# Also define "no-buffer" version of the signal (what test3 currently downloads)
no_buffer_len = int((DOWNLOAD_WINDOW + POST_MERGER) * SAMPLE_RATE)
raw_no_buffer = raw[:no_buffer_len]

variants = {}

# A. Current approach: tukey_side='left', no buffer, crop keeps right edge
print("  A. Current (left taper, no buffer)...")
wA, _, _ = whiten_waveform(raw_no_buffer, delta_t=delta_t, f_lower=40.0,
                           apply_bandpass=True, apply_tukey=True,
                           tukey_alpha=0.1, tukey_side='left')
# truncate_dataloaders(keep_end=True) keeps last FINAL_LENGTH samples
variants['A'] = {
    'full': wA,
    'cropped': wA[-FINAL_LENGTH:],
    'label': 'A: Current (left taper, no buffer)',
    'short_label': 'A: Current'
}

# B. Both-side taper, no buffer
print("  B. Both-side taper, no buffer...")
wB, _, _ = whiten_waveform(raw_no_buffer, delta_t=delta_t, f_lower=40.0,
                           apply_bandpass=True, apply_tukey=True,
                           tukey_alpha=0.1, tukey_side='both')
variants['B'] = {
    'full': wB,
    'cropped': wB[-FINAL_LENGTH:],
    'label': 'B: Both taper, no buffer',
    'short_label': 'B: Both taper'
}

# C. FIXED: Both-side taper + buffer, crop away buffer then keep final window
print("  C. Fixed (both taper + buffer, crop buffer away)...")
wC, _, _ = whiten_waveform(raw, delta_t=delta_t, f_lower=40.0,
                           apply_bandpass=True, apply_tukey=True,
                           tukey_alpha=0.1, tukey_side='both')
# Step 1: drop right buffer (keep first DOWNLOAD_WINDOW + POST_MERGER seconds)
keep_len = int((DOWNLOAD_WINDOW + POST_MERGER) * SAMPLE_RATE)
wC_no_buffer = wC[:keep_len]
# Step 2: keep last FINAL_DURATION seconds
wC_final = wC_no_buffer[-FINAL_LENGTH:]
variants['C'] = {
    'full': wC,
    'cropped': wC_final,
    'label': 'C: Fixed (both taper + buffer)',
    'short_label': 'C: Fixed'
}

# D. gwpy native whitening
print("  D. gwpy native whitening...")
ts_for_gwpy = GWpyTimeSeries(raw, sample_rate=SAMPLE_RATE)
ts_whitened = ts_for_gwpy.whiten(4, 2)  # 4s FFT, 2s overlap
wD = ts_whitened.value.astype(np.float32)
wD_cropped = crop_around_merger(wD, merger_sample, pre_samp, post_samp)
variants['D'] = {
    'full': wD,
    'cropped': wD_cropped,
    'label': 'D: gwpy native',
    'short_label': 'D: gwpy'
}

# E. Whiten only (no bandpass) — isolate FIR filter contribution
print("  E. Whiten only (no bandpass)...")
wE, _, _ = whiten_waveform(raw_no_buffer, delta_t=delta_t, f_lower=40.0,
                           apply_bandpass=False, apply_tukey=False)
variants['E'] = {
    'full': wE,
    'cropped': wE[-FINAL_LENGTH:],
    'label': 'E: Whiten only (no bandpass/taper)',
    'short_label': 'E: No bandpass'
}

# F. No Tukey — bandpass without windowing to show raw edge artifacts
print("  F. No Tukey (bandpass without windowing)...")
wF, _, _ = whiten_waveform(raw_no_buffer, delta_t=delta_t, f_lower=40.0,
                           apply_bandpass=True, apply_tukey=False)
variants['F'] = {
    'full': wF,
    'cropped': wF[-FINAL_LENGTH:],
    'label': 'F: No Tukey (shows raw FIR artifact)',
    'short_label': 'F: No Tukey'
}

print("  All variants computed.")

# ── 3. Figure 1: Stage-by-stage for the FIXED approach ──
print("\n[3] Plotting Figure 1: Stage-by-stage (fixed approach)...")

# Re-run approach C step-by-step to get intermediate stages
# Stage 1: raw signal
# Stage 2: whitened only (no taper, no bandpass)
w_step1_raw = raw.copy()
w_step2, psd_arr, freqs = whiten_waveform(raw, delta_t=delta_t, f_lower=40.0,
                                           apply_bandpass=False, apply_tukey=False)
# Stage 3: whitened + both-side taper + bandpass (full signal with buffer)
w_step3 = wC  # already computed above
# Stage 4: after dropping buffer + final crop
w_step4 = wC_final

t_full = np.linspace(-DOWNLOAD_WINDOW, POST_MERGER + EDGE_BUFFER, len(raw))
t_final = np.linspace(-PRE_MERGER, POST_MERGER, FINAL_LENGTH)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle(f'{EVENT} {DET} — Stage-by-stage whitening (fixed approach)', fontsize=14)

# Row 1: Raw signal
ax = axes[0, 0]
ax.plot(t_full, w_step1_raw, color='black', alpha=0.7, linewidth=0.3)
ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='merger')
ax.set_title('Raw strain (full 20s)')
ax.set_ylabel('Strain')
ax.legend(fontsize=8)

ax = axes[0, 1]
zoom_pre, zoom_post = 1.5, 1.0
zm_start = merger_sample - int(zoom_pre * SAMPLE_RATE)
zm_end = merger_sample + int(zoom_post * SAMPLE_RATE)
t_zoom = np.linspace(-zoom_pre, zoom_post, zm_end - zm_start)
ax.plot(t_zoom, w_step1_raw[zm_start:zm_end], color='black', alpha=0.7, linewidth=0.5)
ax.axvline(0, color='red', linestyle='--', alpha=0.5)
ax.set_title('Raw strain (zoom ±1.5s around merger)')

# Row 2: After whitening (before taper/bandpass)
ax = axes[1, 0]
ax.plot(t_full, w_step2, color='blue', alpha=0.7, linewidth=0.3)
ax.axvline(0, color='red', linestyle='--', alpha=0.5)
ax.set_title('After whitening (no taper, no bandpass)')
ax.set_ylabel('Whitened strain')

ax = axes[1, 1]
ax.plot(t_zoom, w_step2[zm_start:zm_end], color='blue', alpha=0.7, linewidth=0.5)
ax.axvline(0, color='red', linestyle='--', alpha=0.5)
ax.set_title('After whitening (zoom)')

# Row 3: After full pipeline + final crop
ax = axes[2, 0]
ax.plot(t_full, w_step3, color='green', alpha=0.7, linewidth=0.3)
ax.axvline(0, color='red', linestyle='--', alpha=0.5)
ax.axvline(POST_MERGER, color='orange', linestyle=':', alpha=0.7, label=f'crop boundary (+{POST_MERGER}s)')
ax.axvline(-PRE_MERGER, color='orange', linestyle=':', alpha=0.7, label=f'crop boundary (-{PRE_MERGER}s)')
ax.set_title('After taper + bandpass (full 20s, before crop)')
ax.set_ylabel('Processed strain')
ax.set_xlabel('Time relative to merger (s)')
ax.legend(fontsize=8)

ax = axes[2, 1]
ax.plot(t_final, w_step4, color='green', alpha=0.7, linewidth=0.5)
ax.axvline(0, color='red', linestyle='--', alpha=0.5)
ax.set_title(f'Final output [{-PRE_MERGER}s, +{POST_MERGER}s]')
ax.set_xlabel('Time relative to merger (s)')

plt.tight_layout()
plt.savefig('whitening_stages.png', dpi=150)
print("  Saved whitening_stages.png")

# ── 4. Figure 2: Approach comparison in final window ──
print("\n[4] Plotting Figure 2: Approach comparison...")

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle(f'{EVENT} {DET} — Whitening approach comparison '
             f'[{-PRE_MERGER}s, +{POST_MERGER}s]', fontsize=14)

colors = {'A': 'red', 'B': 'orange', 'C': 'green', 'D': 'purple', 'E': 'blue', 'F': 'gray'}
for i, (key, var) in enumerate(variants.items()):
    row, col = divmod(i, 2)
    ax = axes[row, col]
    cropped = var['cropped']
    t = np.linspace(-PRE_MERGER, POST_MERGER, len(cropped))
    ax.plot(t, cropped, color=colors[key], alpha=0.7, linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4)
    ax.set_title(var['label'], fontsize=10)
    if col == 0:
        ax.set_ylabel('Whitened strain')
    if row == 2:
        ax.set_xlabel('Time relative to merger (s)')
    # Show max absolute value in corner
    max_val = np.max(np.abs(cropped))
    ax.text(0.02, 0.95, f'max |y| = {max_val:.1f}', transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('whitening_comparison.png', dpi=150)
print("  Saved whitening_comparison.png")

# ── 5. Figure 3: Edge artifact diagnosis ──
print("\n[5] Plotting Figure 3: Edge artifact diagnosis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle(f'{EVENT} {DET} — Right-edge artifact diagnosis '
             f'(last 0.5s of pre-truncation signal)', fontsize=14)

edge_samples = int(0.5 * SAMPLE_RATE)

for i, (key, var) in enumerate(variants.items()):
    row, col = divmod(i, 3)
    ax = axes[row, col]
    full = var['full']
    edge = full[-edge_samples:]
    t_edge = np.linspace(-0.5, 0, edge_samples)
    ax.plot(t_edge, edge, color=colors[key], alpha=0.8, linewidth=0.5)
    ax.set_title(var['short_label'], fontsize=10)
    ax.set_xlabel('Time from signal end (s)')
    if col == 0:
        ax.set_ylabel('Whitened strain')
    max_edge = np.max(np.abs(edge))
    ax.text(0.02, 0.95, f'max |y| = {max_edge:.1f}', transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('whitening_edge_diagnosis.png', dpi=150)
print("  Saved whitening_edge_diagnosis.png")

# ── 6. Quantitative summary ──
print("\n" + "=" * 70)
print("SUMMARY — Max absolute values in final [-1.5s, +1.0s] window")
print("=" * 70)
for key, var in variants.items():
    cropped = var['cropped']
    max_val = np.max(np.abs(cropped))
    rms = np.sqrt(np.mean(cropped**2))
    # Check last 100 samples for edge spike
    edge_max = np.max(np.abs(cropped[-100:]))
    interior_max = np.max(np.abs(cropped[100:-100]))
    print(f"  {var['short_label']:20s}  max={max_val:8.1f}  rms={rms:6.2f}  "
          f"edge_max={edge_max:8.1f}  interior_max={interior_max:8.1f}  "
          f"edge/interior={'SPIKE' if edge_max > 3*interior_max else 'ok':>6s}")

print("\n" + "=" * 70)
print("Diagnostic plots saved: whitening_stages.png, whitening_comparison.png,")
print("                        whitening_edge_diagnosis.png")
print("=" * 70)
