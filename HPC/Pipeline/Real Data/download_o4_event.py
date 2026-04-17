#!/usr/bin/env python3.11
"""
Download, trim to 2 s, and save GW230601_224134 (O4) in the same .pt format as simulated data.
"""

import os
import sys
import numpy as np
import torch
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Data Generation"))
from gw_datagen import whiten_waveform

EVENT       = "GW230601_224134-v1"
DETECTORS   = ["H1", "L1"]
DURATION    = 32
SAMPLE_RATE = 4096
DT          = 1 / SAMPLE_RATE
N_2S        = 8192   # 2 s at 4096 Hz

DIR = os.path.dirname(os.path.abspath(__file__))

GPS_MERGER = event_gps(EVENT)
T_START_32 = GPS_MERGER - DURATION / 2
T_END_32   = GPS_MERGER + DURATION / 2
T_START_2  = GPS_MERGER - 1.0
T_END_2    = GPS_MERGER + 1.0

print(f"Event  : {EVENT}")
print(f"GPS    : {GPS_MERGER}")
print(f"32s window: {T_START_32} – {T_END_32}")
print()

# ── Download 32 s HDF5 ────────────────────────────────────────────────────────
for det in DETECTORS:
    out = os.path.join(DIR, f"{EVENT}_{det}_strain.hdf5")
    if os.path.exists(out):
        print(f"{det} HDF5 already exists, skipping download.")
        continue
    print(f"Fetching {det} …")
    ts = TimeSeries.fetch_open_data(det, T_START_32, T_END_32, sample_rate=SAMPLE_RATE, cache=True)
    ts.write(out, overwrite=True)
    print(f"  Saved → {out}  ({len(ts)} samples)")

# ── Whiten full 32 s, crop to 2 s ────────────────────────────────────────────
print("\nWhitening …")
strains_raw      = []
strains_whitened = []

for det in DETECTORS:
    path = os.path.join(DIR, f"{EVENT}_{det}_strain.hdf5")
    full     = TimeSeries.read(path)
    full_arr = full.value.astype(np.float64)

    w_full, _, _ = whiten_waveform(
        full_arr, delta_t=DT, f_lower=20.0,
        apply_bandpass=True, apply_tukey=True,
        tukey_alpha=0.1, tukey_side="both",
    )

    t0      = float(full.t0.value)
    i_start = int(round((T_START_2 - t0) / DT))
    i_end   = i_start + N_2S

    strains_raw.append(full_arr[i_start:i_end].astype(np.float32))
    strains_whitened.append(w_full[i_start:i_end].astype(np.float32))
    print(f"  {det} done")

# ── Save .pt ──────────────────────────────────────────────────────────────────
PARAM_NAMES = ["mass1","mass2","spin1z","spin2z","distance",
               "inclination","coa_phase","ra","dec","polarization",
               "m_g","alpha_lv","A"]

X = torch.tensor(np.stack(strains_raw)).unsqueeze(0)          # (1, 2, 8192)
y = torch.zeros(1, len(PARAM_NAMES), dtype=torch.float32)     # params not yet known

metadata = {
    "parameter_names": PARAM_NAMES,
    "channels":        DETECTORS,
    "sample_rate":     SAMPLE_RATE,
    "time_resolution": DT,
    "signal_length":   2.0,
    "target_length":   N_2S,
    "waveform_shape":  (2, N_2S),
    "event":           EVENT,
    "gps_merger":      GPS_MERGER,
    "window_start":    T_START_2,
    "window_end":      T_END_2,
    "source":          "GWOSC",
    "normalised":      False,
    "num_samples":     1,
}

pt_path = os.path.join(DIR, f"{EVENT}_2s_real.pt")
torch.save({"X": X, "y": y, "metadata": metadata}, pt_path)
print(f"\nSaved .pt → {pt_path}  ({os.path.getsize(pt_path)//1024} KB)")

# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

t_axis = np.arange(N_2S) * DT + (T_START_2 - GPS_MERGER)
det_labels = ["H1 (Hanford)", "L1 (Livingston)"]
colors_raw = ["steelblue", "darkorange"]
colors_w   = ["navy", "saddlebrown"]

fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)

for row, (raw, w, label, cr, cw) in enumerate(
        zip(strains_raw, strains_whitened, det_labels, colors_raw, colors_w)):
    axes[row, 0].plot(t_axis, raw, lw=0.5, color=cr)
    axes[row, 0].set_ylabel("Strain", fontsize=9)
    axes[row, 0].set_title(f"{label} — raw", fontsize=10)
    axes[row, 0].axvline(0, color="red", lw=0.8, ls="--", label="Merger")
    axes[row, 0].legend(fontsize=8)
    axes[row, 0].grid(True, alpha=0.3)

    axes[row, 1].plot(t_axis, w, lw=0.6, color=cw)
    axes[row, 1].set_ylabel("Whitened strain", fontsize=9)
    axes[row, 1].set_title(f"{label} — whitened + bandpass (35–300 Hz)", fontsize=10)
    axes[row, 1].axvline(0, color="red", lw=0.8, ls="--", label="Merger")
    axes[row, 1].legend(fontsize=8)
    axes[row, 1].grid(True, alpha=0.3)

for ax in axes[-1]:
    ax.set_xlabel("Time relative to merger (s)", fontsize=9)

fig.suptitle(f"{EVENT} — raw vs processed (window · whiten · bandpass)", fontsize=13, fontweight="bold")
fig.tight_layout()

plot_path = os.path.join(DIR, f"{EVENT}_strain.png")
fig.savefig(plot_path, dpi=150)
print(f"Plot   → {plot_path}")
