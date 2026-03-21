"""
plot_waveform.py — Plot clean GW waveforms using the same pipeline as gw_datagen.py.

Pipeline (mirrors _generate_single_waveform exactly):
  1. get_fd_waveform  (delta_f = 1/256, IMRPhenomD)
  2. np.fft.irfft + scale by delta_f * N
  3. Assemble: last (target_length - n_ringdown) samples + first n_ringdown samples
  4. _apply_end_taper  (cosine-taper last 128 samples)
  5. highpass_fir at 35 Hz, 128 taps
  6. Detector.project_wave  (H1, face-on overhead)

Run:
    python plot_waveform.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_fd_waveform
from pycbc.types import TimeSeries
from pycbc.filter import highpass_fir
from pycbc.detector import Detector

# Import helpers directly from gw_datagen so the pipeline is identical
from gw_datagen import _apply_end_taper, _HIGHPASS_FC

# ---------------------------------------------------------------------------
# Parameters (match gw_datagen defaults)
# ---------------------------------------------------------------------------
SAMPLE_RATE  = 4096
DELTA_T      = 1.0 / SAMPLE_RATE
F_LOWER      = 20.0
F_FINAL      = 2048.0
DELTA_F_FD   = 1.0 / 256        # FD resolution used throughout gw_datagen
N_RINGDOWN   = 500               # merger + early ringdown samples kept
TARGET_LEN   = 16384             # 4 s at 4096 Hz
DETECTOR     = 'H1'
GPS_TIME     = 1126259462.4      # GW150914 epoch (same default as gw_datagen)

# (label, mass1, mass2, colour)
CONFIGS = [
    ("10 + 10 M\u2609",  10,  10, "#e05c00"),
    ("30 + 30 M\u2609",  30,  30, "#1f6fbf"),
    ("60 + 40 M\u2609",  60,  40, "#2ca02c"),
]

# Merger sits at this sample index after assembly
_MERGER_IDX = TARGET_LEN - N_RINGDOWN


def generate_clean_signal(m1, m2):
    """
    Generate a clean (noise-free) detector signal using the same FD->IRFFT
    pipeline as _generate_single_waveform in gw_datagen.py.

    Returns
    -------
    times : np.ndarray
        Time in seconds relative to merger (merger = 0).
    signal : np.ndarray
        Strain projected onto H1.
    """
    # Step 1: frequency-domain waveform
    hp_fd, hc_fd = get_fd_waveform(
        approximant='IMRPhenomD',
        mass1=m1,
        mass2=m2,
        spin1z=0.0,
        spin2z=0.0,
        inclination=0.0,
        coa_phase=0.0,
        distance=410.0,
        delta_f=DELTA_F_FD,
        f_lower=F_LOWER,
        f_final=F_FINAL,
    )

    # Step 2: IRFFT + correct normalisation
    hp_raw = np.fft.irfft(hp_fd.numpy())
    hc_raw = np.fft.irfft(hc_fd.numpy())
    N = len(hp_raw)
    hp_raw *= DELTA_F_FD * N
    hc_raw *= DELTA_F_FD * N

    # Step 3: assemble to TARGET_LEN
    if TARGET_LEN <= N:
        n_pre  = TARGET_LEN - N_RINGDOWN
        hp_arr = np.concatenate([hp_raw[N - n_pre:], hp_raw[:N_RINGDOWN]])
        hc_arr = np.concatenate([hc_raw[N - n_pre:], hc_raw[:N_RINGDOWN]])
    else:
        hp_arr = np.concatenate([np.zeros(TARGET_LEN - N), hp_raw])
        hc_arr = np.concatenate([np.zeros(TARGET_LEN - N), hc_raw])

    # Step 4: cosine-taper the tail (128 samples) to avoid filter edge effects
    hp_arr = _apply_end_taper(hp_arr)
    hc_arr = _apply_end_taper(hc_arr)

    # Step 5: highpass at 35 Hz, 128 taps
    hp_ts = TimeSeries(hp_arr.astype(np.float64), delta_t=DELTA_T)
    hc_ts = TimeSeries(hc_arr.astype(np.float64), delta_t=DELTA_T)
    hp_ts.start_time += GPS_TIME
    hc_ts.start_time += GPS_TIME
    hp_ts = highpass_fir(hp_ts, _HIGHPASS_FC, 128)
    hc_ts = highpass_fir(hc_ts, _HIGHPASS_FC, 128)

    # Step 6: project onto detector (face-on, directly overhead)
    det    = Detector(DETECTOR)
    signal = det.project_wave(hp_ts, hc_ts, ra=0.0, dec=np.pi / 2,
                               polarization=0.0, method='lal')

    sig = signal.numpy()
    sig_len = len(sig)
    if sig_len >= TARGET_LEN:
        sig = sig[sig_len - TARGET_LEN:]
    else:
        sig = np.concatenate([np.zeros(TARGET_LEN - sig_len, dtype=sig.dtype), sig])

    # Time axis relative to merger
    times = (np.arange(TARGET_LEN) - _MERGER_IDX) * DELTA_T
    return times, sig


# ---------------------------------------------------------------------------
# Generate waveforms
# ---------------------------------------------------------------------------
print(f"Generating {len(CONFIGS)} waveforms via FD->IRFFT pipeline ({DETECTOR})...")
results = []
for label, m1, m2, colour in CONFIGS:
    print(f"  {label} ...")
    t, s = generate_clean_signal(m1, m2)
    results.append((label, m1, m2, colour, t, s))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(CONFIGS) + 1, 1, figsize=(12, 10))
fig.suptitle(
    f"Clean IMRPhenomD Waveforms — {DETECTOR} projection\n"
    r"($f_{\rm lower}=20\,{\rm Hz}$, $d=410\,{\rm Mpc}$, face-on, no noise)",
    fontsize=12,
)

SHOW_WINDOW = (-1.5, 0.15)   # seconds around merger

for ax, (label, m1, m2, colour, t, s) in zip(axes[:-1], results):
    mask = (t >= SHOW_WINDOW[0]) & (t <= SHOW_WINDOW[1])
    ax.plot(t[mask], s[mask] * 1e21, color=colour, linewidth=0.85, label=label)
    ax.axvline(0, color='grey', linewidth=0.6, linestyle='--', alpha=0.6)
    ax.set_xlim(*SHOW_WINDOW)
    ax.set_ylabel(r'$h\times10^{21}$', fontsize=9)
    ax.legend(loc='upper left', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xticklabels([])

# Bottom panel: normalised overlay zoomed to ±150 ms
ax_ov = axes[-1]
for label, m1, m2, colour, t, s in results:
    mask = (t >= -0.15) & (t <= 0.12)
    norm = np.max(np.abs(s[mask])) or 1.0
    ax_ov.plot(t[mask], s[mask] / norm, color=colour,
               linewidth=1.0, alpha=0.85, label=label)

ax_ov.axvline(0, color='grey', linewidth=0.6, linestyle='--', alpha=0.6)
ax_ov.set_xlim(-0.15, 0.12)
ax_ov.set_xlabel('Time from merger (s)', fontsize=9)
ax_ov.set_ylabel('Normalised strain', fontsize=9)
ax_ov.set_title('Merger comparison (normalised)', fontsize=10)
ax_ov.legend(loc='upper left', fontsize=9)
ax_ov.tick_params(labelsize=8)

# Shared x-label for upper panels
axes[-2].set_xticklabels(
    [f'{v:.1f}' for v in axes[-2].get_xticks()]
)
axes[-2].set_xlabel('Time from merger (s)', fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_waveform.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")
try:
    plt.show()
except Exception:
    pass
