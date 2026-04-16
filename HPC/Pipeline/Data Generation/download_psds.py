"""
download_psds.py — Download O4a GWOSC PSDs and save them locally.

Edit the CONFIG section below, then run:
    python download_psds.py

PSDs are saved to the o4_psd_cache/ folder next to this file.
Once downloaded, the test suite and generate_dataset.py use them
automatically with no network access required.
"""

from gw_datagen import build_o4_psd_cache, _DEFAULT_CACHE, _cache_path
import numpy as np
import logging
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────

DETECTORS   = ['H1', 'L1']   # detectors to download PSDs for
N_SEGMENTS  = 10             # number of O4a science segments per detector
SAMPLE_RATE = 4096            # Hz
FORCE       = False           # set True to re-download even if cache exists

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')

os.makedirs(_DEFAULT_CACHE, exist_ok=True)
print(f"Cache directory : {_DEFAULT_CACHE}\n")

for det in DETECTORS:
    out_path = _cache_path(det, SAMPLE_RATE, _DEFAULT_CACHE)

    if os.path.isfile(out_path) and not FORCE:
        n = np.load(out_path)['psds'].shape[0]
        print(f"[{det}] Already cached — {n} PSDs at {out_path}")
        print(f"       (set FORCE = True to re-download)\n")
        continue

    print(f"[{det}] Fetching {N_SEGMENTS} segments from GWOSC ...")
    build_o4_psd_cache(det, n_segments=N_SEGMENTS, sample_rate=SAMPLE_RATE,
                       cache_dir=_DEFAULT_CACHE, force=FORCE)

    data   = np.load(out_path)
    n_psds = data['psds'].shape[0]
    n_freq = data['freqs'].shape[0]
    print(f"[{det}] Saved {n_psds} PSDs  ({n_freq} frequency bins)  →  {out_path}\n")

print("Done.")
