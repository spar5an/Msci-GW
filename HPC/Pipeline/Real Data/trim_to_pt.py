#!/usr/bin/env python3.11
"""
Trim GW150914 strain data to a 2-second window centred on merger
and save as a .pt file matching the simulated dataset format.
"""

import os
import numpy as np
import torch
from gwpy.timeseries import TimeSeries

GPS_MERGER = 1126259462.4
T_START    = GPS_MERGER - 1.0   # 1126259461.4
T_END      = GPS_MERGER + 1.0   # 1126259463.4
SAMPLE_RATE = 4096
N_SAMPLES   = 8192              # 2 s × 4096 Hz

DIR = os.path.dirname(os.path.abspath(__file__))

KNOWN_PARAMS = [
    36.2,    # mass1       (M_sun)
    29.1,    # mass2       (M_sun)
    -0.01,   # spin1z
     0.05,   # spin2z
   410.0,    # distance    (Mpc)
     2.78,   # inclination (rad)
     1.46,   # coa_phase   (rad)
     1.95,   # ra          (rad)
    -1.27,   # dec         (rad)
     0.82,   # polarization (rad)
     0.0,    # m_g
     0.0,    # alpha_lv
     0.0,    # A
]

PARAM_NAMES = [
    "mass1", "mass2", "spin1z", "spin2z", "distance",
    "inclination", "coa_phase", "ra", "dec", "polarization",
    "m_g", "alpha_lv", "A",
]

strains = []
for det in ["H1", "L1"]:
    path = os.path.join(DIR, f"GW150914_{det}_strain.hdf5")
    print(f"Loading {det} …")
    ts = TimeSeries.read(path)
    ts = ts.crop(T_START, T_END)
    arr = np.array(ts.value, dtype=np.float32)
    if len(arr) != N_SAMPLES:
        raise ValueError(f"{det}: expected {N_SAMPLES} samples, got {len(arr)}")
    strains.append(arr)

X = torch.tensor(np.stack(strains, axis=0)).unsqueeze(0)   # (1, 2, 8192)
y = torch.tensor(KNOWN_PARAMS, dtype=torch.float32).unsqueeze(0)  # (1, 13)

metadata = {
    "parameter_names": PARAM_NAMES,
    "channels":        ["H1", "L1"],
    "sample_rate":     SAMPLE_RATE,
    "time_resolution": 1 / SAMPLE_RATE,
    "signal_length":   2.0,
    "target_length":   N_SAMPLES,
    "waveform_shape":  (2, N_SAMPLES),
    "event":           "GW150914",
    "gps_merger":      GPS_MERGER,
    "window_start":    T_START,
    "window_end":      T_END,
    "source":          "GWOSC",
    "normalised":      False,
    "num_samples":     1,
}

out_path = os.path.join(DIR, "GW150914_2s_real.pt")
torch.save({"X": X, "y": y, "metadata": metadata}, out_path)

size_kb = os.path.getsize(out_path) / 1024
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Saved   → {out_path}  ({size_kb:.0f} KB)")
