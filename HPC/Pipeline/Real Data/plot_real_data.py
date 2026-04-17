#!/usr/bin/env python3.11
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Data Generation"))
from gw_datagen import whiten_waveform
from gwpy.timeseries import TimeSeries

DIR = os.path.dirname(os.path.abspath(__file__))
meta_pt    = torch.load(os.path.join(DIR, "GW150914_2s_real.pt"), weights_only=False)["metadata"]
dt         = meta_pt["time_resolution"]
gps_merger = meta_pt["gps_merger"]
T_START    = meta_pt["window_start"]

N = int(round(2.0 / dt))
t = np.arange(N) * dt + (T_START - gps_merger)

det_labels = ["H1 (Hanford)", "L1 (Livingston)"]
colors_raw = ["steelblue", "darkorange"]
colors_w   = ["navy", "saddlebrown"]

print("Whitening using full 32 s window for PSD estimation …")
raw_2s   = []
whitened = []
for det, label in zip(["H1", "L1"], det_labels):
    full     = TimeSeries.read(os.path.join(DIR, f"GW150914_{det}_strain.hdf5"))
    full_arr = full.value.astype(np.float64)

    w_full, _, _ = whiten_waveform(
        full_arr,
        delta_t=dt,
        f_lower=20.0,
        apply_bandpass=True,
        apply_tukey=True,
        tukey_alpha=0.1,
        tukey_side="both",
    )

    t0_full = float(full.t0.value)
    i_start = int(round((T_START - t0_full) / dt))
    i_end   = i_start + N

    raw_2s.append(full_arr[i_start:i_end])
    whitened.append(w_full[i_start:i_end])
    print(f"  {label} done")

fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)

for col, (raw, w, det, cr, cw) in enumerate(zip(raw_2s, whitened, det_labels, colors_raw, colors_w)):
    axes[col, 0].plot(t, raw, lw=0.5, color=cr)
    axes[col, 0].set_ylabel("Strain", fontsize=9)
    axes[col, 0].set_title(f"{det} — raw", fontsize=10)
    axes[col, 0].axvline(0, color="red", lw=0.8, ls="--", label="Merger")
    axes[col, 0].legend(fontsize=8)
    axes[col, 0].grid(True, alpha=0.3)

    axes[col, 1].plot(t, w, lw=0.6, color=cw)
    axes[col, 1].set_ylabel("Whitened strain", fontsize=9)
    axes[col, 1].set_title(f"{det} — whitened + bandpass (35–300 Hz)", fontsize=10)
    axes[col, 1].axvline(0, color="red", lw=0.8, ls="--", label="Merger")
    axes[col, 1].legend(fontsize=8)
    axes[col, 1].grid(True, alpha=0.3)

for ax in axes[-1]:
    ax.set_xlabel("Time relative to merger (s)", fontsize=9)

fig.suptitle("GW150914 — raw vs processed (window · whiten · bandpass)", fontsize=13, fontweight="bold")
fig.tight_layout()

out = os.path.join(DIR, "GW150914_strain.png")
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
