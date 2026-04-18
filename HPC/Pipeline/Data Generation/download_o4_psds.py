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


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Output directory : {os.path.abspath(args.output_dir)}")
    print(f"Detectors        : {args.detectors}")
    print(f"Segments per det : {args.n_segments}")
    print(f"Sample rate      : {args.sample_rate} Hz")
    print(f"Force re-download: {args.force}")
    print()

    failed = []
    for det in args.detectors:
        out_path = _cache_path(det, args.sample_rate, args.output_dir)
        if os.path.exists(out_path) and not args.force:
            data = np.load(out_path)
            n_psds = data['psds'].shape[0]
            print(f"[{det}] Cache already exists — {n_psds} PSDs at {out_path}  (use --force to re-download)")
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
            print(f"[{det}] Saved {n_psds} PSDs  ({n_freqs} frequency bins)  →  {out_path}")
        except Exception as exc:
            logging.error('[%s] Download failed: %s', det, exc)
            failed.append(det)

    print()
    if failed:
        print(f"WARNING: download failed for: {failed}")
        sys.exit(1)
    else:
        print("All detectors downloaded successfully.")
        print(f"\nSet PSD_CACHE_DIR = '{os.path.abspath(args.output_dir)}' in generate_dataset.py")


if __name__ == '__main__':
    main()
