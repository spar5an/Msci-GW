"""
Plots whitened simulated and real GW150914 waveforms side by side, both
centred on coalescence at t = 0.

Processing applied to each:
  Simulated : FD→TD (merger centred) → highpass 35 Hz → O4 noise injected
              → whiten using the SAME O4 PSD draw → bandpass 35–300 Hz
              (noise injection and whitening share one PSD per detector so
              the whitening is perfectly matched and the noise is truly flat)
  Real      : full 32 s HDF5 whitened (Welch PSD from 32 s), then cropped to
              the 2 s window [merger−1 s, merger+1 s]

Layout: 2 rows (H1, L1) × 2 cols (simulated, real)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries as GWpyTS
from pycbc.noise import noise_from_psd

from gw_datagen import _generate_single_waveform, whiten_waveform, load_random_o4_psd

REAL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Real Data")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "o4_psd_cache")

# ── GW150914 parameters ───────────────────────────────────────────────────────
PARAMS = dict(
    mass1=36.2, mass2=29.1,
    spin1z=-0.01, spin2z=0.05,
    distance=410.0,
    inclination=2.78, coa_phase=1.46,
    ra=1.95, dec=-1.27, polarization=0.82,
    gps_time=1126259462.4,
)
DT        = 1.0 / 4096
TARGET    = 8192
DETECTORS = ["H1", "L1"]
t_sim     = (np.arange(TARGET) - TARGET // 2) * DT   # −1 → +1 s
delta_f   = 1.0 / (TARGET * DT)
flen      = TARGET // 2 + 1

# ── load one O4 PSD per detector — used for BOTH noise injection and whitening
# so they are perfectly matched (avoids mismatched random draws from the cache)
print("Loading O4 PSDs …")
det_psds = {det: load_random_o4_psd(flen, delta_f, 35.0, det, cache_dir=CACHE_DIR)
            for det in DETECTORS}
for det, psd in det_psds.items():
    print(f"  {det}: median PSD = {np.median(psd.numpy()):.3e}")

# ── generate clean signal (no noise), then inject noise with the loaded PSDs ──
print("Generating clean simulated signal …")
res = _generate_single_waveform(
    params=PARAMS, time_resolution=DT, approximant="IMRPhenomD",
    f_lower=40.0, f_final=2048.0, detectors=DETECTORS,
    target_length=TARGET, add_noise=False,
)
assert res["success"], res.get("error")

print("Injecting O4 noise and whitening (matched PSD per detector) …")
from pycbc.types import TimeSeries as PyCBCTS
sim_whitened = {}
for det in DETECTORS:
    psd  = det_psds[det]
    sig  = res["detectors"][det]                   # clean PyCBC TimeSeries
    noise = noise_from_psd(TARGET, DT, psd)
    noise._epoch = sig._epoch
    noisy = sig.inject(noise).numpy().astype(np.float64)
    w, _, _ = whiten_waveform(noisy, delta_t=DT, f_lower=35.0,
                               apply_bandpass=True, apply_tukey=True,
                               tukey_alpha=0.1, tukey_side="both",
                               psd=psd)
    sim_whitened[det] = w
    print(f"  {det} done")

# ── real: whiten full 32 s then crop ─────────────────────────────────────────
meta   = torch.load(os.path.join(REAL_DIR, "GW150914_2s_real.pt"),
                    weights_only=False)["metadata"]
GPS_M  = meta["gps_merger"]
T_START = meta["window_start"]
N      = meta["target_length"]

print("Whitening real GW150914 data (full 32 s window) …")
real_whitened = {}
for det in DETECTORS:
    full     = GWpyTS.read(os.path.join(REAL_DIR, f"GW150914_{det}_strain.hdf5"))
    full_arr = full.value.astype(np.float64)
    w_full, _, _ = whiten_waveform(full_arr, delta_t=DT, f_lower=35.0,
                                    apply_bandpass=True, apply_tukey=True,
                                    tukey_alpha=0.1, tukey_side="both")
    t0      = float(full.t0.value)
    i_start = int(round((T_START - t0) / DT))
    real_whitened[det] = w_full[i_start:i_start + N].astype(np.float32)
    print(f"  {det} done")

t_real = np.arange(N) * DT + (T_START - GPS_M)   # −1 → +1 s

# ── plot ──────────────────────────────────────────────────────────────────────
SIM_COL   = "#0571b0"
REAL_COL  = "#d6604d"
VLINE_COL = "#b2182b"

fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                         gridspec_kw={"hspace": 0.48, "wspace": 0.28})

for row, det in enumerate(DETECTORS):
    ax = axes[row, 0]
    ax.plot(t_sim, sim_whitened[det], color=SIM_COL, lw=0.6)
    ax.axvline(0.0, color=VLINE_COL, lw=1.5, ls="--", label="t = 0  (coalescence)")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Time relative to coalescence  (s)")
    ax.set_ylabel("Whitened strain")
    ax.set_title(f"({'ab'[row]})  Simulated + noise, whitened — {det}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[row, 1]
    ax.plot(t_real, real_whitened[det], color=REAL_COL, lw=0.6)
    ax.axvline(0.0, color=VLINE_COL, lw=1.5, ls="--", label="t = 0  (coalescence)")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Time relative to coalescence  (s)")
    ax.set_ylabel("Whitened strain")
    ax.set_title(f"({'cd'[row]})  Real GW150914, whitened — {det}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

fig.suptitle(
    "Coalescence alignment: whitened simulated vs real GW150914\n"
    "Both windows span  t ∈ [−1 s, +1 s]  with coalescence at  t = 0\n"
    "Processing: O4 PSD whitening · Tukey window · bandpass 35–300 Hz",
    fontsize=10,
)

out = "coalescence_alignment.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
