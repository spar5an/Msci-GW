"""
plot_noise_comparison.py — Plot one waveform with aLIGO analytic noise vs O4 real noise,
at three stages of signal processing: raw, normalised, and whitened.

Usage:
    python plot_noise_comparison.py [--cache-dir ./o4_psd_cache] [--output comparison.png]
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gw_datagen import (
    build_o4_psd_cache,
    load_random_o4_psd,
    normalize_waveform,
    whiten_waveform,
    _cache_path,
    _DEFAULT_CACHE,
    _DEFAULT_N_SEGS,
    _HIGHPASS_FC,
    _generate_single_waveform,
)
from pycbc.psd import aLIGOZeroDetHighPower

# ── Fixed waveform parameters ─────────────────────────────────────────────────
PARAMS = {
    'mass1':        35.0,
    'mass2':        30.0,
    'spin1z':       0.0,
    'spin2z':       0.0,
    'distance':     400.0,
    'inclination':  0.0,
    'coa_phase':    0.0,
    'ra':           1.375,
    'dec':         -1.211,
    'polarization': 2.659,
    'redshift':     0.09,
}

DETECTOR        = 'H1'
TIME_RESOLUTION = 1 / 4096
SIGNAL_LENGTH   = 2.0
F_LOWER         = 30.0
F_FINAL         = 2048.0
APPROXIMANT     = 'IMRPhenomD'
TARGET_LENGTH   = int(SIGNAL_LENGTH / TIME_RESOLUTION)
DELTA_F         = 1.0 / SIGNAL_LENGTH
FLEN            = TARGET_LENGTH // 2 + 1

ALIGO_COLOR = '#1f77b4'
O4_COLOR    = '#d62728'


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cache-dir', default=_DEFAULT_CACHE,
                   help='O4 PSD cache directory (populated by download_o4_psds.py).')
    p.add_argument('--n-segments', type=int, default=_DEFAULT_N_SEGS,
                   help='Segments to fetch if cache is missing.')
    p.add_argument('--output', default='noise_comparison.png',
                   help='Output image path.')
    return p.parse_args()


def ensure_cache(cache_dir, n_segments):
    path = _cache_path(DETECTOR, int(round(1 / TIME_RESOLUTION)), cache_dir)
    if not os.path.exists(path):
        print(f'Cache not found — downloading {n_segments} O4 segments for {DETECTOR} …')
        build_o4_psd_cache(
            detector=DETECTOR,
            n_segments=n_segments,
            sample_rate=int(round(1 / TIME_RESOLUTION)),
            cache_dir=cache_dir,
        )
    else:
        print(f'Using existing cache: {path}')


def generate(noise_backend, cache_dir):
    result = _generate_single_waveform(
        params=PARAMS,
        time_resolution=TIME_RESOLUTION,
        approximant=APPROXIMANT,
        f_lower=F_LOWER,
        detectors=[DETECTOR],
        target_length=TARGET_LENGTH,
        add_noise=True,
        f_final=F_FINAL,
        noise_backend=noise_backend,
        psd_cache_dir=cache_dir,
        highpass_fc=_HIGHPASS_FC,
    )
    if not result['success']:
        raise RuntimeError(f'Waveform generation failed: {result["error"]}')
    return result['detectors'][DETECTOR].numpy()


def get_psd_curve(noise_backend, cache_dir):
    if noise_backend == 'aligo':
        psd = aLIGOZeroDetHighPower(FLEN, DELTA_F, F_LOWER)
    else:
        psd = load_random_o4_psd(
            FLEN, DELTA_F, F_LOWER, DETECTOR,
            sample_rate=int(round(1 / TIME_RESOLUTION)),
            cache_dir=cache_dir,
            rng=np.random.default_rng(0),
        )
    freqs = np.arange(FLEN) * DELTA_F
    return freqs, psd.numpy()


def plot_strain(ax, times, data, color, ylabel):
    ax.plot(times, data, lw=0.5, color=color, alpha=0.85)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_xlim(times[0], times[-1])


def main():
    args = parse_args()
    ensure_cache(args.cache_dir, args.n_segments)

    print('Generating aLIGO waveform …')
    strain_aligo = generate('aligo',  args.cache_dir)
    print('Generating O4 PSD waveform …')
    strain_o4    = generate('o4_psd', args.cache_dir)

    times = np.arange(TARGET_LENGTH) * TIME_RESOLUTION

    # ── Signal processing ─────────────────────────────────────────────────────
    print('Normalising …')
    norm_aligo = normalize_waveform(strain_aligo)   # ×1e21 → O(1) values
    norm_o4    = normalize_waveform(strain_o4)

    print('Whitening …')
    white_aligo, _, _ = whiten_waveform(strain_aligo, delta_t=TIME_RESOLUTION,
                                         f_lower=F_LOWER)
    white_o4,    _, _ = whiten_waveform(strain_o4,    delta_t=TIME_RESOLUTION,
                                         f_lower=F_LOWER)

    # ── PSD curves for the bottom row ─────────────────────────────────────────
    freqs_aligo, psd_aligo = get_psd_curve('aligo',  args.cache_dir)
    freqs_o4,    psd_o4    = get_psd_curve('o4_psd', args.cache_dir)

    # ── Figure: 4 rows × 2 columns ───────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(
        'Noise comparison: aLIGO analytic vs O4 real PSD\n'
        f'm₁={PARAMS["mass1"]:.0f} M☉,  m₂={PARAMS["mass2"]:.0f} M☉,  '
        f'd={PARAMS["distance"]:.0f} Mpc',
        fontsize=13,
    )

    col_titles = ['aLIGO analytic noise', 'O4 real noise']
    col_colors = [ALIGO_COLOR, O4_COLOR]

    for col, (strain, norm, white, freqs, psd_vals, title, color) in enumerate([
        (strain_aligo, norm_aligo, white_aligo, freqs_aligo, psd_aligo,
         col_titles[0], col_colors[0]),
        (strain_o4,   norm_o4,    white_o4,    freqs_o4,    psd_o4,
         col_titles[1], col_colors[1]),
    ]):
        # Row 0 — raw strain
        axes[0, col].set_title(title, fontsize=11, color=color, fontweight='bold')
        plot_strain(axes[0, col], times, strain, color, 'Strain')

        # Row 1 — normalised (×1e21)
        plot_strain(axes[1, col], times, norm, color, 'Normalised strain  (×10²¹)')

        # Row 2 — whitened + bandpassed
        plot_strain(axes[2, col], times, white, color, 'Whitened strain')

        # Row 3 — PSD
        f_mask = freqs >= F_LOWER
        axes[3, col].loglog(freqs[f_mask], psd_vals[f_mask], lw=1.0, color=color)
        axes[3, col].set_xlabel('Frequency (Hz)')
        axes[3, col].set_ylabel('PSD (strain² / Hz)')
        axes[3, col].set_xlim(F_LOWER, 2048)
        axes[3, col].grid(True, which='both', ls=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
