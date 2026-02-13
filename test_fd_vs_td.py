"""
Diagnostic: understand circular wrapping in FD->IRFFT conversion.
Compare with get_td_waveform to verify the fix.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform, get_fd_waveform

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'test_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

M1, M2 = 30.0, 30.0
DIST = 410.0
F_LOWER = 30.0
APPROX = 'IMRPhenomD'
DT = 1.0 / 4096
DELTA_F = 1.0 / 256
TARGET_LENGTH = 8192  # 2 seconds at 4096 Hz

# ─── Generate reference TD waveform ──────────────────────────────────────────
print("=== TD reference ===")
hp_td_pycbc, _ = get_td_waveform(
    approximant=APPROX, mass1=M1, mass2=M2,
    delta_t=DT, f_lower=F_LOWER, distance=DIST
)
hp_td_ref = hp_td_pycbc.numpy().copy()
td_peak = np.argmax(np.abs(hp_td_ref))
print(f"  Length: {len(hp_td_ref)}, peak at {td_peak}, peak amp: {np.max(np.abs(hp_td_ref)):.4e}")

# ─── Generate FD waveform + IRFFT ────────────────────────────────────────────
print("\n=== FD + IRFFT ===")
hp_fd, _ = get_fd_waveform(
    approximant=APPROX, mass1=M1, mass2=M2,
    delta_f=DELTA_F, f_lower=F_LOWER, f_final=2048.0, distance=DIST
)

hp_fd_array = hp_fd.numpy()
hp_raw = np.fft.irfft(hp_fd_array)
N = len(hp_raw)
hp_raw *= DELTA_F * N  # correct amplitude scaling

peak_raw = np.argmax(np.abs(hp_raw))
print(f"  IRFFT length: {N}, peak at {peak_raw} (={N - peak_raw} from end)")
print(f"  Peak amp: {np.max(np.abs(hp_raw)):.4e}")
print(f"  First 10 samples abs: {np.abs(hp_raw[:10])}")
print(f"  Last 10 samples abs:  {np.abs(hp_raw[-10:])}")

# ─── Plot 1: Show circular structure ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 1a: First 2000 samples (ringdown region)
n_show = 2000
t_start = np.arange(n_show) * DT
axes[0].plot(t_start, hp_raw[:n_show], 'b-', linewidth=0.8)
axes[0].set_title(f'IRFFT output: First {n_show} samples (RINGDOWN wraps here)')
axes[0].set_ylabel('Strain')
axes[0].set_xlabel('Time from index 0 (s)')

# 1b: Last 2000 samples (inspiral → merger)
t_end = (np.arange(N - n_show, N)) * DT
axes[1].plot(t_end, hp_raw[-n_show:], 'r-', linewidth=0.8)
axes[1].axvline(peak_raw * DT, color='gray', linestyle='--', alpha=0.5, label=f'peak at {peak_raw}')
axes[1].set_title(f'IRFFT output: Last {n_show} samples (INSPIRAL → MERGER)')
axes[1].set_ylabel('Strain')
axes[1].set_xlabel('Time (s)')
axes[1].legend()

# 1c: Schematic of full array
axes[2].semilogy(np.abs(hp_raw[::100]), 'k-', linewidth=0.5)
axes[2].axvline(peak_raw // 100, color='red', linestyle='--', label='merger peak')
axes[2].set_title(f'Full IRFFT array (every 100th sample, log scale). N={N}')
axes[2].set_xlabel('Sample / 100')
axes[2].set_ylabel('|Strain|')
axes[2].legend()

plt.tight_layout()
path = os.path.join(PLOT_DIR, 'fd_circular_structure.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"\n  Saved: {path}")


# ─── Apply the stitch fix ────────────────────────────────────────────────────
print("\n=== Stitching fix ===")
n_ringdown = 500  # ~0.12s of post-merger ringdown

pre_merger = hp_raw[N - TARGET_LENGTH + n_ringdown:]  # inspiral through merger
post_merger = hp_raw[:n_ringdown]                      # ringdown from start
hp_stitched = np.concatenate([pre_merger, post_merger])
print(f"  pre_merger: {len(pre_merger)} samples (from index {N - TARGET_LENGTH + n_ringdown})")
print(f"  post_merger: {len(post_merger)} samples (indices 0-{n_ringdown})")
print(f"  Stitched length: {len(hp_stitched)} (target: {TARGET_LENGTH})")

stitch_peak = np.argmax(np.abs(hp_stitched))
print(f"  Stitched peak at sample {stitch_peak} ({stitch_peak * DT:.4f}s)")


# ─── Plot 2: Stitched vs TD reference ────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Align by merger peak
def align_around_peak(sig, peak_idx, half_window):
    s = max(0, peak_idx - half_window)
    e = min(len(sig), peak_idx + half_window)
    t = (np.arange(s, e) - peak_idx) * DT
    return t, sig[s:e]

# 2a: Full 2-second window
hw = TARGET_LENGTH // 2
t_td, s_td = align_around_peak(hp_td_ref, td_peak, hw)
t_st, s_st = align_around_peak(hp_stitched, stitch_peak, hw)

axes[0].plot(t_td, s_td, 'k-', label='TD reference', linewidth=1.5)
axes[0].plot(t_st, s_st, 'r--', label='FD stitched', alpha=0.8)
axes[0].set_title('Full window (aligned by merger)')
axes[0].set_ylabel('Strain')
axes[0].legend()

# 2b: Coalescence zoom
hw2 = int(0.03 / DT)
t_td2, s_td2 = align_around_peak(hp_td_ref, td_peak, hw2)
t_st2, s_st2 = align_around_peak(hp_stitched, stitch_peak, hw2)

axes[1].plot(t_td2, s_td2, 'k-', label='TD reference', linewidth=1.5)
axes[1].plot(t_st2, s_st2, 'r--', label='FD stitched', alpha=0.8)
axes[1].set_title('Coalescence + Ringdown zoom')
axes[1].set_ylabel('Strain')
axes[1].legend()

# 2c: Ringdown only (post-merger)
ring_samples = int(0.02 / DT)
td_ring_start = td_peak
td_ring_end = min(len(hp_td_ref), td_peak + ring_samples)
st_ring_start = stitch_peak
st_ring_end = min(len(hp_stitched), stitch_peak + ring_samples)

t_td_r = (np.arange(td_ring_start, td_ring_end) - td_peak) * DT
t_st_r = (np.arange(st_ring_start, st_ring_end) - stitch_peak) * DT

axes[2].plot(t_td_r, hp_td_ref[td_ring_start:td_ring_end], 'k-', label='TD reference', linewidth=1.5)
axes[2].plot(t_st_r, hp_stitched[st_ring_start:st_ring_end], 'r--', label='FD stitched', alpha=0.8)
axes[2].set_title('Ringdown only (post-merger)')
axes[2].set_xlabel('Time relative to merger (s)')
axes[2].set_ylabel('Strain')
axes[2].legend()

plt.tight_layout()
path = os.path.join(PLOT_DIR, 'fd_stitched_vs_td.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Saved: {path}")


# ─── Correlation ──────────────────────────────────────────────────────────────
# Align both to merger peak and compute correlation over common window
win = min(TARGET_LENGTH // 2, td_peak, stitch_peak)
td_slice = hp_td_ref[td_peak - win:td_peak + win]
st_slice = hp_stitched[stitch_peak - win:stitch_peak + win]
n_common = min(len(td_slice), len(st_slice))
td_slice = td_slice[:n_common]
st_slice = st_slice[:n_common]

corr = np.corrcoef(td_slice, st_slice)[0, 1]
amp_ratio = np.max(np.abs(st_slice)) / np.max(np.abs(td_slice))
print(f"\n=== Comparison ===")
print(f"  Correlation (TD vs stitched FD): {corr:.6f}")
print(f"  Amplitude ratio: {amp_ratio:.4f}")
print(f"  TD peak amp: {np.max(np.abs(td_slice)):.4e}")
print(f"  FD peak amp: {np.max(np.abs(st_slice)):.4e}")
