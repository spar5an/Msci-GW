#!/usr/bin/env python3.11
"""
plot_o4_real_pt.py — standalone plot script for the combined O4 .pt file
produced by ``download_real_data.py``.

Loads ``Real Data/pt/o4_all_events_2s_real.pt`` and writes a multi-page PDF
showing every event's H1 + L1 raw and whitened strain (4 events per page).
Also drops per-event PNGs under plots/o4_events/ for quick browsing.

Run with:
    python plot_o4_real_pt.py                 # default .pt path + plots/ dir
    python plot_o4_real_pt.py --pt <path>     # custom .pt
    python plot_o4_real_pt.py --no-pngs       # PDF only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

_HERE = Path(__file__).resolve().parent
_REAL_PT = (_HERE.parent.parent
            / "HPC" / "Pipeline" / "Real Data" / "pt"
            / "o4_all_events_2s_real.pt")
_PLOTS_DIR = _HERE / "plots"

EVENTS_PER_PAGE = 4
DETECTORS = ["H1", "L1"]


def _load_blob(pt_path: Path) -> dict:
    if not pt_path.is_file():
        raise FileNotFoundError(f"No combined .pt at {pt_path}")
    return torch.load(str(pt_path), weights_only=False)


def _plot_event_row(axes, t, X_ev, Xw_ev, channels, event, gps):
    """Fill one 4-column row: H1 raw, H1 whit, L1 raw, L1 whit."""
    for det_idx, det in enumerate(channels):
        raw  = X_ev[det_idx].numpy()
        whit = Xw_ev[det_idx].numpy()

        ax_raw = axes[2 * det_idx]
        ax_raw.plot(t, raw, lw=0.4, color="C3")
        ax_raw.set_title(f"{event} — {det} raw", fontsize=8)
        ax_raw.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax_raw.grid(True, alpha=0.3)
        ax_raw.tick_params(labelsize=7)

        ax_whit = axes[2 * det_idx + 1]
        ax_whit.plot(t, whit, lw=0.4, color="C3")
        ax_whit.set_title(f"{event} — {det} whitened", fontsize=8)
        ax_whit.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax_whit.grid(True, alpha=0.3)
        ax_whit.tick_params(labelsize=7)


def write_pdf(blob: dict, pdf_path: Path) -> None:
    X         = blob["X"]
    Xw        = blob["X_whitened"]
    meta      = blob["metadata"]
    events    = meta.get("events") or [f"event_{i}" for i in range(X.shape[0])]
    channels  = meta.get("channels", DETECTORS)
    sr        = meta.get("sample_rate", 4096)
    n_samp    = X.shape[-1]
    gps_map   = meta.get("gps_mergers", {})

    # 2 s window centred on merger: t in [-1, +1]
    t = np.arange(n_samp) / sr - (n_samp / sr) / 2

    n_events = X.shape[0]
    n_pages  = (n_events + EVENTS_PER_PAGE - 1) // EVENTS_PER_PAGE

    print(f"Writing PDF → {pdf_path}  ({n_events} events across {n_pages} pages)")
    with PdfPages(str(pdf_path)) as pdf:
        for page in range(n_pages):
            i0 = page * EVENTS_PER_PAGE
            i1 = min(i0 + EVENTS_PER_PAGE, n_events)
            n_rows = i1 - i0

            fig, axes = plt.subplots(n_rows, 4,
                                     figsize=(16, 2.4 * n_rows),
                                     sharex=True)
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            for row, ev_idx in enumerate(range(i0, i1)):
                _plot_event_row(
                    axes[row], t,
                    X[ev_idx], Xw[ev_idx],
                    channels,
                    events[ev_idx],
                    gps_map.get(events[ev_idx]),
                )
            for ax in axes[-1]:
                ax.set_xlabel("time (s, relative to merger)", fontsize=8)

            fig.suptitle(
                f"O4 real events — page {page+1}/{n_pages} "
                f"(events {i0+1}–{i1} of {n_events})",
                fontsize=11, fontweight="bold",
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig, dpi=120)
            plt.close(fig)
            print(f"  page {page+1}/{n_pages}")


def write_per_event_pngs(blob: dict, out_dir: Path) -> None:
    """One 2×2 PNG per event: rows=detectors, cols=raw/whitened."""
    X, Xw = blob["X"], blob["X_whitened"]
    meta  = blob["metadata"]
    events   = meta["events"]
    channels = meta.get("channels", DETECTORS)
    sr       = meta.get("sample_rate", 4096)
    n_samp   = X.shape[-1]
    t        = np.arange(n_samp) / sr - (n_samp / sr) / 2

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing per-event PNGs → {out_dir}")
    for i, ev in enumerate(events):
        fig, axes = plt.subplots(len(channels), 2,
                                 figsize=(12, 2.6 * len(channels)),
                                 sharex=True)
        if len(channels) == 1:
            axes = axes[np.newaxis, :]
        for d, det in enumerate(channels):
            axes[d, 0].plot(t, X[i, d].numpy(),  lw=0.5, color="C3")
            axes[d, 0].set_title(f"{det} — raw", fontsize=9)
            axes[d, 1].plot(t, Xw[i, d].numpy(), lw=0.5, color="C3")
            axes[d, 1].set_title(f"{det} — whitened", fontsize=9)
            for ax in axes[d]:
                ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
                ax.grid(True, alpha=0.3)
        for ax in axes[-1]:
            ax.set_xlabel("time (s, relative to merger)", fontsize=9)
        fig.suptitle(ev, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(str(out_dir / f"{ev}.png"), dpi=120)
        plt.close(fig)
        if (i + 1) % 20 == 0 or i + 1 == len(events):
            print(f"  {i+1}/{len(events)} events plotted")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt", default=str(_REAL_PT),
                   help=f"Combined .pt path (default: {_REAL_PT})")
    p.add_argument("--out-dir", default=str(_PLOTS_DIR),
                   help=f"Output directory for PDF + PNGs (default: {_PLOTS_DIR})")
    p.add_argument("--no-pngs", action="store_true",
                   help="Skip per-event PNGs; only write the PDF summary")
    args = p.parse_args()

    pt_path  = Path(args.pt)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = _load_blob(pt_path)
    meta = blob["metadata"]
    print(f"Loaded {pt_path}")
    print(f"  X shape       : {tuple(blob['X'].shape)}")
    print(f"  X_whit shape  : {tuple(blob['X_whitened'].shape)}")
    print(f"  events        : {meta['num_samples']} "
          f"({meta['channels']}, sr={meta['sample_rate']} Hz)")

    write_pdf(blob, out_dir / "o4_all_events.pdf")
    if not args.no_pngs:
        write_per_event_pngs(blob, out_dir / "o4_events")

    print("Done.")


if __name__ == "__main__":
    main()
