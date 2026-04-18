#!/usr/bin/env python3.11
"""
reprocess_cached.py — rebuild the combined O4 .pt straight from cached HDF5,
no network. Drops any event whose H1 or L1 strain contains NaN/Inf.

GPS of each event is recovered from the HDF5 ``t0`` attribute
(gps_merger = t0 + DURATION/2). The output matches
``download_real_data.COMBINED_NAME`` so existing consumers pick it up.

Run with:
    python reprocess_cached.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from gwpy.timeseries import TimeSeries

from download_real_data import (
    COMBINED_NAME,
    DIR,
    DURATION,
    build_combined,
    layout,
    process_to_pt,
)


def discover_cached_events(hdf5_dir: Path) -> list[tuple[str, float]]:
    """Return [(event, gps_merger), ...] for every H1+L1 pair on disk."""
    pairs: list[tuple[str, float]] = []
    for h1 in sorted(hdf5_dir.glob("*_H1_strain.hdf5")):
        event = h1.name[: -len("_H1_strain.hdf5")]
        l1 = hdf5_dir / f"{event}_L1_strain.hdf5"
        if not l1.exists():
            continue
        ts = TimeSeries.read(str(h1))
        gps = float(ts.t0.value) + DURATION / 2
        pairs.append((event, gps))
    return pairs


def main() -> None:
    dirs = layout(DIR)
    hdf5_dir = Path(dirs["hdf5"])
    pt_dir   = Path(dirs["pt"])
    pt_dir.mkdir(parents=True, exist_ok=True)

    events = discover_cached_events(hdf5_dir)
    print(f"Found {len(events)} cached H1+L1 pairs under {hdf5_dir}\n")

    per_event: list[tuple[str, dict]] = []
    dropped:  list[tuple[str, str]]  = []

    for event, gps in events:
        paths = {det: str(hdf5_dir / f"{event}_{det}_strain.hdf5")
                 for det in ("H1", "L1")}
        bad = {}
        for det, p in paths.items():
            arr = TimeSeries.read(p).value
            n_nan = int(np.isnan(arr).sum()) + int(np.isinf(arr).sum())
            if n_nan > 0:
                bad[det] = n_nan
        if bad:
            msg = ", ".join(f"{det}: {n} non-finite" for det, n in bad.items())
            print(f"[{event}]  DROP  ({msg})")
            dropped.append((event, msg))
            continue

        try:
            data = process_to_pt(event, gps, paths)
        except Exception as exc:
            print(f"[{event}]  FAIL  {exc}")
            dropped.append((event, str(exc)))
            continue

        per_event.append((event, data))
        print(f"[{event}]  OK")

    if not per_event:
        print("\nNo events processed — nothing to save.")
        return

    combined = build_combined(per_event, batch_size=8)
    combined_path = os.path.join(dirs["pt"], COMBINED_NAME)
    torch.save(combined, combined_path)

    kb = os.path.getsize(combined_path) // 1024
    print(f"\nCombined → {combined_path}  ({kb} KB)")
    print(f"  X shape : {tuple(combined['X'].shape)}")
    print(f"{'─'*60}")
    print(f"Processed: {len(per_event)}   Dropped: {len(dropped)}")
    if dropped:
        print("  dropped:")
        for ev, reason in dropped:
            print(f"    {ev}: {reason}")


if __name__ == "__main__":
    main()
