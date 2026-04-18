#!/usr/bin/env python3.11
"""
Download, whiten, crop to 2 s, and bundle real LIGO events into a single
DataLoader-ready .pt file that matches the simulated-data schema produced by
``gw_datagen.save_dataloaders``.

Default target: all O4 events where both H1 and L1 have open data on GWOSC.
Pass --events GW150914 GW230601_224134-v1 ... to process specific events.

Layout (under --output-dir, defaults to this script's directory):
    hdf5/<EVENT>_<DET>_strain.hdf5     – raw 32 s strain, cached across runs
    pt/o4_all_events_2s_real.pt        – combined dataset (see schema below)
    plots/<EVENT>_strain.png           – optional, with --plot

Combined .pt schema (compatible with ``gw_datagen.load_dataloaders``)::

    {
        "X":            (N, 2, 8192)  raw trimmed 2 s strain  (float32)
        "X_whitened":   (N, 2, 8192)  whitened + bandpassed    (float32)
        "y":            (N, 13)        parameter labels (zeros — unknown)
        "train_indices": list[int]    (empty by default)
        "val_indices":   list[int]    (empty by default)
        "test_indices":  list[int]    (defaults to all N events)
        "metadata": {
            "parameter_names", "channels", "sample_rate", "time_resolution",
            "signal_length", "target_length", "waveform_shape",
            "num_samples", "batch_size",
            "train_size", "val_size", "test_size",
            "events", "gps_mergers", "source", "processing",
        }
    }
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
COMBINED_NAME = "o4_all_events_2s_real.pt"

PARAM_NAMES = [
    "mass1", "mass2", "spin1z", "spin2z", "distance",
    "inclination", "coa_phase", "ra", "dec", "polarization",
    "m_g", "alpha_lv", "A",
]

DIR = os.path.dirname(os.path.abspath(__file__))


def layout(output_dir: str) -> dict[str, str]:
    """Return the canonical sub-directories under ``output_dir``."""
    return {
        "hdf5":  os.path.join(output_dir, "hdf5"),
        "pt":    os.path.join(output_dir, "pt"),
        "plots": os.path.join(output_dir, "plots"),
    }


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

def download_strain_hdf5(event: str, gps: float, hdf5_dir: str,
                         detectors: Sequence[str] = DETECTORS,
                         duration: float = DURATION) -> dict[str, str]:
    """Fetch raw `duration` s strain for each detector. Skip if HDF5 cached.
    Returns {detector: hdf5_path}."""
    os.makedirs(hdf5_dir, exist_ok=True)
    t0, t1 = gps - duration / 2, gps + duration / 2
    paths: dict[str, str] = {}
    for det in detectors:
        path = os.path.join(hdf5_dir, f"{event}_{det}_strain.hdf5")
        if not os.path.exists(path):
            ts = TimeSeries.fetch_open_data(
                det, t0, t1, sample_rate=SAMPLE_RATE, cache=True,
            )
            ts.write(path, overwrite=True)
        paths[det] = path
    return paths


# ── Whiten + crop to 2 s ─────────────────────────────────────────────────────

def process_to_pt(event: str, gps: float, hdf5_paths: dict[str, str]) -> dict:
    """Whiten the full 32 s, crop to 2 s around merger. Returns a per-event
    dict with X (raw) and X_whitened (processed), each shape (1, N_det, 8192)."""
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

    X          = torch.tensor(np.stack(strains_raw)).unsqueeze(0)    # (1, D, 8192) raw
    X_whitened = torch.tensor(np.stack(strains_w)).unsqueeze(0)      # (1, D, 8192) whitened
    y          = torch.zeros(1, len(PARAM_NAMES), dtype=torch.float32)

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
    return {"X": X, "X_whitened": X_whitened, "y": y, "metadata": metadata}


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

    raw_2s  = [data["X"][0, i].numpy()          for i in range(len(detectors))]
    whit_2s = [data["X_whitened"][0, i].numpy() for i in range(len(detectors))]

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
    return [(name, event_gps(name)) for name in requested]


def _build_splits(n: int, train_frac: float, val_frac: float) -> tuple[list[int], list[int], list[int]]:
    """Contiguous train/val/test split. Test gets the remainder."""
    if train_frac < 0 or val_frac < 0 or train_frac + val_frac > 1:
        raise ValueError("Invalid split: train_frac and val_frac must be ≥ 0 and sum ≤ 1")
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val   = min(n_val, n - n_train)
    train = list(range(0, n_train))
    val   = list(range(n_train, n_train + n_val))
    test  = list(range(n_train + n_val, n))
    return train, val, test


def build_combined(per_event_data: list[tuple[str, dict]],
                   batch_size: int = 8,
                   train_frac: float = 0.0,
                   val_frac: float = 0.0) -> dict:
    """Stack per-event tensors and attach the split indices + metadata.
    Output schema matches ``gw_datagen.save_dataloaders``."""
    X_all     = torch.cat([d["X"]          for _, d in per_event_data], dim=0)
    Xw_all    = torch.cat([d["X_whitened"] for _, d in per_event_data], dim=0)
    y_all     = torch.cat([d["y"]          for _, d in per_event_data], dim=0)

    n = X_all.shape[0]
    train_idx, val_idx, test_idx = _build_splits(n, train_frac, val_frac)

    metadata = {
        "parameter_names": PARAM_NAMES,
        "channels":        DETECTORS,
        "sample_rate":     SAMPLE_RATE,
        "time_resolution": DT,
        "signal_length":   2.0,
        "target_length":   N_2S,
        "waveform_shape":  (len(DETECTORS), N_2S),
        "num_samples":     n,
        "batch_size":      batch_size,
        "train_size":      len(train_idx),
        "val_size":        len(val_idx),
        "test_size":       len(test_idx),
        "events":          [e for e, _ in per_event_data],
        "gps_mergers":     {e: d["metadata"]["gps_merger"] for e, d in per_event_data},
        "source":          "GWOSC",
        "processing":      "whitened + bandpass 35–300 Hz (Tukey α=0.1)",
    }

    return {
        "X":             X_all,
        "X_whitened":    Xw_all,
        "y":             y_all,
        "train_indices": train_idx,
        "val_indices":   val_idx,
        "test_indices":  test_idx,
        "metadata":      metadata,
    }


def run(events: Sequence[str] | None = None,
        output_dir: str = DIR,
        do_plot: bool = False,
        batch_size: int = 8,
        train_frac: float = 0.0,
        val_frac: float = 0.0) -> str | None:
    """Download, process and bundle. Returns the combined .pt path (or None)."""
    dirs = layout(output_dir)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print("Resolving events …", flush=True)
    ev_list = _resolve_events(events)
    print(f"Target: {len(ev_list)} event(s): {[e for e, _ in ev_list]}\n")

    per_event: list[tuple[str, dict]] = []
    skipped: list[str] = []
    failed:  list[str] = []

    for event, gps in ev_list:
        print(f"[{event}]  GPS={gps:.1f}")
        if not both_detectors_available(gps):
            print("  Skipping – H1 or L1 not available on GWOSC")
            skipped.append(event)
            continue

        try:
            hdf5_paths = download_strain_hdf5(event, gps, dirs["hdf5"])
            data = process_to_pt(event, gps, hdf5_paths)
        except Exception as exc:
            print(f"  Failed: {exc}")
            failed.append(event)
            continue

        if do_plot:
            plot_path = os.path.join(dirs["plots"], f"{event}_strain.png")
            plot_processed(event, data, plot_path)
            print(f"  Plot  → {plot_path}")

        per_event.append((event, data))

    if not per_event:
        print("\nNo events processed — nothing to save.")
        return None

    combined = build_combined(per_event, batch_size=batch_size,
                              train_frac=train_frac, val_frac=val_frac)
    combined_path = os.path.join(dirs["pt"], COMBINED_NAME)
    torch.save(combined, combined_path)

    meta = combined["metadata"]
    kb = os.path.getsize(combined_path) // 1024
    print(f"\nCombined → {combined_path}  ({kb} KB)")
    print(f"  X shape     : {tuple(combined['X'].shape)}")
    print(f"  splits      : train={meta['train_size']}, val={meta['val_size']}, test={meta['test_size']}")
    print(f"{'─'*60}")
    print(f"Processed : {len(per_event)}   Skipped : {len(skipped)}   Failed : {len(failed)}")
    if skipped: print(f"  skipped: {skipped}")
    if failed:  print(f"  failed : {failed}")
    return combined_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", nargs="*", default=None,
                   help="Specific event names; default = all O4 events with H1+L1")
    p.add_argument("--output-dir", default=DIR,
                   help=f"Parent directory; hdf5/ and pt/ are created below it (default: {DIR})")
    p.add_argument("--plot", action="store_true",
                   help="Save a per-event raw-vs-whitened .png to plots/")
    p.add_argument("--batch-size", type=int, default=8,
                   help="DataLoader batch size written into metadata (default: 8)")
    p.add_argument("--train-frac", type=float, default=0.0,
                   help="Fraction of events in train split (default: 0.0)")
    p.add_argument("--val-frac", type=float, default=0.0,
                   help="Fraction of events in val split (default: 0.0 — test gets all)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        events=args.events,
        output_dir=args.output_dir,
        do_plot=args.plot,
        batch_size=args.batch_size,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
