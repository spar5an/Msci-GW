#!/usr/bin/env python3.11
"""
Download, whiten, crop to 2 s, and save real LIGO events as .pt files.

Default target: all O4 events where both H1 and L1 have open data on GWOSC.
Pass --events GW150914 GW230601_224134-v1 ... to process specific events instead.

Outputs (per event, written to this directory by default):
    <EVENT>_<DET>_strain.hdf5   – raw 32 s strain, cached
    <EVENT>_2s_real.pt          – trimmed + whitened 2 s.
                                   X:(1, 2, 8192)      – whitened
                                   X_raw:(1, 2, 8192)  – trimmed raw
                                   y:(1, 13)
    <EVENT>_strain.png          – 2×2 plot (raw vs whitened) if --plot

Combined (when --combine or default O4 run):
    o4_all_events_2s_real.pt    – stacked X / X_raw with shape (N, 2, 8192)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np
import torch
from gwpy.timeseries import TimeSeries
from gwosc.datasets import find_datasets, event_gps
from gwosc.locate import get_urls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Data Generation"))
from gw_datagen import whiten_waveform  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
DETECTORS    = ["H1", "L1"]
DURATION     = 32              # s, downloaded + whitened for PSD estimation
SAMPLE_RATE  = 4096
DT           = 1 / SAMPLE_RATE
N_2S         = 8192            # 2 s at 4096 Hz
O4_GPS_START = 1369166418      # 2023-05-24 18:00 UTC

PARAM_NAMES = [
    "mass1", "mass2", "spin1z", "spin2z", "distance",
    "inclination", "coa_phase", "ra", "dec", "polarization",
    "m_g", "alpha_lv", "A",
]

DIR = os.path.dirname(os.path.abspath(__file__))


# ── Event discovery ───────────────────────────────────────────────────────────

def list_o4_events() -> list[tuple[str, float]]:
    """Return all GWOSC events with GPS ≥ O4 start, sorted chronologically."""
    events: list[tuple[str, float]] = []
    for ev in find_datasets(type="events"):
        try:
            gps = event_gps(ev)
        except Exception:
            continue
        if gps >= O4_GPS_START:
            events.append((ev, gps))
    events.sort(key=lambda x: x[1])
    return events


def both_detectors_available(gps: float, detectors: Sequence[str] = DETECTORS,
                              duration: float = DURATION) -> bool:
    """True iff GWOSC has open-data URLs for every detector around gps."""
    t0, t1 = gps - duration / 2, gps + duration / 2
    for det in detectors:
        try:
            urls = get_urls(det, t0, t1)
        except Exception:
            return False
        if not urls:
            return False
    return True


# ── Download ──────────────────────────────────────────────────────────────────

def download_strain_hdf5(event: str, gps: float, output_dir: str,
                         detectors: Sequence[str] = DETECTORS,
                         duration: float = DURATION) -> dict[str, str]:
    """Fetch raw `duration` s strain for each detector. Skip if HDF5 cached.
    Returns {detector: hdf5_path}."""
    os.makedirs(output_dir, exist_ok=True)
    t0, t1 = gps - duration / 2, gps + duration / 2
    paths: dict[str, str] = {}
    for det in detectors:
        path = os.path.join(output_dir, f"{event}_{det}_strain.hdf5")
        if not os.path.exists(path):
            ts = TimeSeries.fetch_open_data(
                det, t0, t1, sample_rate=SAMPLE_RATE, cache=True,
            )
            ts.write(path, overwrite=True)
        paths[det] = path
    return paths


# ── Whiten + crop to 2 s .pt ──────────────────────────────────────────────────

def process_to_pt(event: str, gps: float, hdf5_paths: dict[str, str]) -> dict:
    """Whiten the full 32 s, crop to 2 s around merger, build a .pt-ready dict.
    Returns {"X", "y", "metadata"} with X shape (1, N_detectors, N_2S)."""
    t_start_2 = gps - 1.0
    t_end_2   = gps + 1.0

    strains_raw: list[np.ndarray] = []
    strains_w:   list[np.ndarray] = []
    detectors = list(hdf5_paths.keys())

    for det in detectors:
        full     = TimeSeries.read(hdf5_paths[det])
        full_arr = full.value.astype(np.float64)

        w_full, _, _ = whiten_waveform(
            full_arr, delta_t=DT, f_lower=20.0,
            apply_bandpass=True, apply_tukey=True,
            tukey_alpha=0.1, tukey_side="both",
        )

        i_start = int(round((t_start_2 - float(full.t0.value)) / DT))
        i_end   = i_start + N_2S
        if i_end > len(full_arr):
            raise ValueError(f"{event} [{det}]: not enough samples to crop 2 s window")

        strains_raw.append(full_arr[i_start:i_end].astype(np.float32))
        strains_w.append(w_full[i_start:i_end].astype(np.float32))

    X     = torch.tensor(np.stack(strains_w)).unsqueeze(0)      # (1, N_det, 8192) whitened
    X_raw = torch.tensor(np.stack(strains_raw)).unsqueeze(0)    # (1, N_det, 8192) raw trimmed
    y     = torch.zeros(1, len(PARAM_NAMES), dtype=torch.float32)

    metadata = {
        "parameter_names": PARAM_NAMES,
        "channels":        detectors,
        "sample_rate":     SAMPLE_RATE,
        "time_resolution": DT,
        "signal_length":   2.0,
        "target_length":   N_2S,
        "waveform_shape":  (len(detectors), N_2S),
        "event":           event,
        "gps_merger":      gps,
        "window_start":    t_start_2,
        "window_end":      t_end_2,
        "source":          "GWOSC",
        "processing":      "whitened + bandpass 35–300 Hz (Tukey α=0.1)",
        "num_samples":     1,
    }
    return {"X": X, "X_raw": X_raw, "y": y, "metadata": metadata}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_raw_strain(event: str, gps: float, hdf5_paths: dict[str, str],
                    output_path: str) -> None:
    """Plot the raw 32 s strain for each detector, with the merger marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detectors = list(hdf5_paths.keys())
    fig, axes = plt.subplots(len(detectors), 1,
                             figsize=(12, 2.5 * len(detectors)), sharex=True)
    if len(detectors) == 1:
        axes = [axes]

    for ax, det in zip(axes, detectors):
        ts = TimeSeries.read(hdf5_paths[det])
        t  = np.arange(len(ts)) * DT + (float(ts.t0.value) - gps)
        ax.plot(t, ts.value, lw=0.4)
        ax.axvline(0, color="red", lw=0.8, ls="--", label="Merger")
        ax.set_ylabel(f"{det} strain", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time relative to merger (s)", fontsize=9)
    fig.suptitle(f"{event} — raw 32 s strain", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_processed(event: str, data: dict, output_path: str) -> None:
    """2×2 plot: raw 2 s (left) vs whitened+bandpassed 2 s (right) per detector."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta      = data["metadata"]
    detectors = meta["channels"]
    gps       = meta["gps_merger"]
    t         = np.arange(N_2S) * DT + (meta["window_start"] - gps)

    raw_2s  = [data["X_raw"][0, i].numpy() for i in range(len(detectors))]
    whit_2s = [data["X"][0, i].numpy()     for i in range(len(detectors))]

    fig, axes = plt.subplots(len(detectors), 2,
                             figsize=(14, 3 * len(detectors)), sharex=True)
    if len(detectors) == 1:
        axes = axes[np.newaxis, :]

    for row, det in enumerate(detectors):
        axes[row, 0].plot(t, raw_2s[row], lw=0.5)
        axes[row, 0].set_title(f"{det} — raw", fontsize=10)
        axes[row, 0].set_ylabel("Strain", fontsize=9)

        axes[row, 1].plot(t, whit_2s[row], lw=0.6)
        axes[row, 1].set_title(f"{det} — whitened + bandpass (35–300 Hz)", fontsize=10)
        axes[row, 1].set_ylabel("Whitened strain", fontsize=9)

        for ax in axes[row]:
            ax.axvline(0, color="red", lw=0.8, ls="--", label="Merger")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Time relative to merger (s)", fontsize=9)

    fig.suptitle(f"{event} — raw vs processed", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Orchestration ─────────────────────────────────────────────────────────────

def _resolve_events(requested: Sequence[str] | None) -> list[tuple[str, float]]:
    if not requested:
        return list_o4_events()
    resolved: list[tuple[str, float]] = []
    for name in requested:
        resolved.append((name, event_gps(name)))
    return resolved


def run(events: Sequence[str] | None = None, output_dir: str = DIR,
        do_plot: bool = False, do_combine: bool = True) -> list[str]:
    """Process each event end-to-end. Returns list of event names successfully saved."""
    print("Resolving events …", flush=True)
    ev_list = _resolve_events(events)
    print(f"Target: {len(ev_list)} event(s): {[e for e, _ in ev_list]}\n")

    saved: list[tuple[str, dict]] = []
    skipped: list[str] = []
    failed:  list[str] = []

    for event, gps in ev_list:
        pt_path = os.path.join(output_dir, f"{event}_2s_real.pt")

        if os.path.exists(pt_path):
            cached = torch.load(pt_path, weights_only=False)
            if "X_raw" in cached:
                print(f"[{event}] .pt already exists, loading …")
                saved.append((event, cached))
                continue
            print(f"[{event}] existing .pt missing X_raw — regenerating")

        print(f"[{event}]  GPS={gps:.1f}")
        if not both_detectors_available(gps):
            print("  Skipping – H1 or L1 not available on GWOSC")
            skipped.append(event)
            continue

        try:
            hdf5_paths = download_strain_hdf5(event, gps, output_dir)
            data = process_to_pt(event, gps, hdf5_paths)
        except Exception as exc:
            print(f"  Failed: {exc}")
            failed.append(event)
            continue

        torch.save(data, pt_path)
        kb = os.path.getsize(pt_path) // 1024
        print(f"  Saved → {pt_path}  ({kb} KB)")

        if do_plot:
            plot_path = os.path.join(output_dir, f"{event}_strain.png")
            plot_processed(event, data, plot_path)
            print(f"  Plot  → {plot_path}")

        saved.append((event, data))

    # Combined file
    if do_combine and saved:
        X_all     = torch.cat([d["X"]     for _, d in saved], dim=0)
        X_raw_all = torch.cat([d["X_raw"] for _, d in saved], dim=0)
        y_all     = torch.cat([d["y"]     for _, d in saved], dim=0)
        combined_meta = {
            "parameter_names": PARAM_NAMES,
            "channels":        DETECTORS,
            "sample_rate":     SAMPLE_RATE,
            "time_resolution": DT,
            "signal_length":   2.0,
            "target_length":   N_2S,
            "waveform_shape":  (len(DETECTORS), N_2S),
            "events":          [e for e, _ in saved],
            "gps_mergers":     {e: d["metadata"]["gps_merger"] for e, d in saved},
            "source":          "GWOSC",
            "processing":      "whitened + bandpass 35–300 Hz (Tukey α=0.1)",
            "num_samples":     len(saved),
        }
        combined_path = os.path.join(output_dir, "o4_all_events_2s_real.pt")
        torch.save({"X": X_all, "X_raw": X_raw_all, "y": y_all,
                    "metadata": combined_meta}, combined_path)
        print(f"\nCombined → {combined_path}  X:{tuple(X_all.shape)}")

    print(f"\n{'─'*60}")
    print(f"Processed : {len(saved)}   Skipped : {len(skipped)}   Failed : {len(failed)}")
    if skipped: print(f"  skipped: {skipped}")
    if failed:  print(f"  failed : {failed}")
    return [e for e, _ in saved]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", nargs="*", default=None,
                   help="Specific event names; default = all O4 events with H1+L1")
    p.add_argument("--output-dir", default=DIR,
                   help=f"Output directory (default: {DIR})")
    p.add_argument("--plot", action="store_true",
                   help="Save a per-event raw-vs-whitened .png")
    p.add_argument("--no-combine", action="store_true",
                   help="Skip writing the combined o4_all_events_2s_real.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        events=args.events,
        output_dir=args.output_dir,
        do_plot=args.plot,
        do_combine=not args.no_combine,
    )
