"""
download_o4_psds.py — Download O4a GWOSC data and save Welch PSDs locally.

Run this once on a machine with internet access (e.g. a login node) before
generating datasets on compute nodes that may have no network connectivity.

Usage
-----
    python download_o4_psds.py --detectors H1 L1 --n-segments 100

The saved .npz files are then read by gw_datagen when NOISE_BACKEND = 'o4_psd'
and PSD_CACHE_DIR is pointed at the same directory.
"""

import argparse
import logging
import os
import sys

import numpy as np

from gw_datagen import _cache_path, build_o4_psd_cache, _DEFAULT_CACHE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Download O4a GWOSC segments and save Welch PSDs as .npz files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--detectors',
        nargs='+',
        default=['H1', 'L1'],
        metavar='DET',
        help='Detector names to download PSDs for (e.g. H1 L1 V1).',
    )
    parser.add_argument(
        '--n-segments',
        type=int,
        default=100,
        metavar='N',
        help='Number of O4a science segments to fetch per detector.',
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=4096,
        metavar='HZ',
        help='Target sample rate in Hz. GWOSC data is resampled if needed.',
    )
    parser.add_argument(
        '--output-dir',
        default=_DEFAULT_CACHE,
        metavar='DIR',
        help='Directory to write the .npz cache files into.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download and overwrite existing cache files.',
    )
    return parser.parse_args()


def _configure_logging() -> None:
    """Silence gwpy/gwosc per-segment chatter; keep our warnings visible.

    gwpy's TimeSeries.fetch_open_data logs "Found N possible sources" +
    "Attemping access with 'gwosc'" at INFO for every single 256 s fetch
    (plus retries), which drowns out the actual failure warnings when a
    job makes many fetches. We keep root at WARNING so the real failure
    lines from gw_datagen still surface, and restore INFO on gw_datagen
    itself so per-detector progress messages still print.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger('gw_datagen').setLevel(logging.INFO)
    for noisy in ('gwpy', 'gwosc', 'urllib3', 'requests'):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    args = parse_args()
    _configure_logging()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Output directory : {os.path.abspath(args.output_dir)}")
    print(f"Detectors        : {args.detectors}")
    print(f"Segments per det : {args.n_segments}")
    print(f"Sample rate      : {args.sample_rate} Hz")
    print(f"Force re-download: {args.force}")
    print()

    per_det = {}   # det -> (requested, saved, status)
    failed_fetch = []
    for det in args.detectors:
        out_path = _cache_path(det, args.sample_rate, args.output_dir)
        if os.path.exists(out_path) and not args.force:
            data = np.load(out_path)
            n_psds = data['psds'].shape[0]
            print(f"[{det}] Cache already exists — {n_psds} PSDs at {out_path}  (use --force to re-download)")
            per_det[det] = (args.n_segments, n_psds, 'skipped (cached)')
            continue

        print(f"[{det}] Fetching {args.n_segments} segments from GWOSC …")
        try:
            build_o4_psd_cache(
                detector=det,
                n_segments=args.n_segments,
                sample_rate=args.sample_rate,
                cache_dir=args.output_dir,
                force=args.force,
            )
            data = np.load(out_path)
            n_psds = data['psds'].shape[0]
            n_freqs = data['freqs'].shape[0]
            print(f"[{det}] Saved {n_psds}/{args.n_segments} PSDs  ({n_freqs} frequency bins)  →  {out_path}")
            per_det[det] = (args.n_segments, n_psds, 'ok')
        except Exception as exc:
            logging.error('[%s] Download failed: %s', det, exc)
            failed_fetch.append(det)
            per_det[det] = (args.n_segments, 0, f'error: {type(exc).__name__}: {exc}')

    print()
    print('=' * 60)
    print('PSD download summary')
    print('=' * 60)
    print(f"{'Detector':<10}{'Requested':>12}{'Saved':>10}{'Failed':>10}   Status")
    for det, (req, saved, status) in per_det.items():
        missed = req - saved
        print(f"{det:<10}{req:>12}{saved:>10}{missed:>10}   {status}")
    print()

    if failed_fetch:
        print(f"ERROR: download completely failed for: {failed_fetch}")
        sys.exit(1)

    thin = [d for d, (req, saved, _) in per_det.items() if saved < 0.5 * req]
    if thin:
        print(f"WARNING: detectors with <50% yield: {thin}")
        print(f"         resubmit with a higher --n-segments or retry to top up.")

    print(f"\nCache dir: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
