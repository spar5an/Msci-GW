#!/usr/bin/env python3.11
"""
Download real gravitational wave strain data for GW150914
(first detected GW event, seen by both H1 and L1)
from GWOSC and save as HDF5 files.
"""

import os
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

# GW150914: first BBH detection, clearly seen in both detectors
EVENT = "GW150914"
DETECTORS = ["H1", "L1"]
DURATION = 32        # seconds around the event
SAMPLE_RATE = 4096   # Hz

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

gps = event_gps(EVENT)
t_start = gps - DURATION / 2
t_end   = gps + DURATION / 2

print(f"Event : {EVENT}")
print(f"GPS   : {gps}")
print(f"Window: {t_start} – {t_end}  ({DURATION} s)")
print()

for det in DETECTORS:
    print(f"Fetching {det} data from GWOSC …")
    strain = TimeSeries.fetch_open_data(det, t_start, t_end, sample_rate=SAMPLE_RATE, cache=True)

    out_path = os.path.join(OUTPUT_DIR, f"{EVENT}_{det}_strain.hdf5")
    strain.write(out_path, overwrite=True)
    print(f"  Saved → {out_path}")
    print(f"  Duration : {strain.duration.value:.2f} s")
    print(f"  Samples  : {len(strain)}")
    print(f"  t0       : {strain.t0.value}")
    print()

print("Done.")
