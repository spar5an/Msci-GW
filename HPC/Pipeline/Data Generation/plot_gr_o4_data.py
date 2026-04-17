"""
plot_gr_o4_data.py — Generate GR waveforms with O4 real noise and plot a selection of samples.

Usage:
    python plot_gr_o4_data.py [--n-samples 10] [--n-plot 4] [--output gr_o4_samples.png]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gw_datagen import (
    pycbc_data_generator,
    _DEFAULT_CACHE,
)

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES     = 10
N_PLOT        = 4
TIME_RES      = 1 / 4096
SIGNAL_LENGTH = 2.0
F_LOWER       = 10.0
F_FINAL       = 2048.0
APPROXIMANT   = 'IMRPhenomD'
DETECTORS     = ['H1', 'L1']

H1_COLOR = '#1f77b4'
L1_COLOR = '#d62728'


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--n-samples', type=int, default=N_SAMPLES,
                   help='Total waveforms to generate.')
    p.add_argument('--n-plot', type=int, default=N_PLOT,
                   help='Number of examples to plot.')
    p.add_argument('--cache-dir', default=_DEFAULT_CACHE,
                   help='O4 PSD cache directory.')
    p.add_argument('--output', default='gr_o4_samples.png',
                   help='Output image path.')
    return p.parse_args()


def main():
    args = parse_args()

    config = {
        'mass1':        lambda size: np.random.uniform(10, 60, size=size),
        'mass2':        lambda size: np.random.uniform(10, 60, size=size),
        'spin1z':       lambda size: np.random.uniform(-0.99, 0.99, size=size),
        'spin2z':       lambda size: np.random.uniform(-0.99, 0.99, size=size),
        'distance':     lambda size: np.random.uniform(200, 800, size=size),
        'inclination':  lambda size: np.random.uniform(0, np.pi, size=size),
        'coa_phase':    lambda size: np.random.uniform(0, 2 * np.pi, size=size),
        'ra':           lambda size: np.random.uniform(0, 2 * np.pi, size=size),
        'dec':          lambda size: np.arcsin(np.random.uniform(-1, 1, size=size)),
        'polarization': lambda size: np.random.uniform(0, np.pi, size=size),
    }

    print(f'Generating {args.n_samples} GR waveforms with O4 noise...')
    result = pycbc_data_generator(
        config=config,
        num_samples=args.n_samples,
        time_resolution=TIME_RES,
        approximant=APPROXIMANT,
        f_lower=F_LOWER,
        f_final=F_FINAL,
        signal_length=SIGNAL_LENGTH,
        detectors=DETECTORS,
        add_noise=True,
        noise_backend='o4_psd',
        psd_cache_dir=args.cache_dir,
        batch_size=args.n_samples,
        train_split=0.8,
        val_split=0.1,
        show_progress=True,
    )

    meta         = result['metadata']
    param_names  = meta['parameter_names']
    m1_idx       = param_names.index('mass1')
    m2_idx       = param_names.index('mass2')
    d_idx        = param_names.index('distance')

    # Pull one full batch from the train loader
    waveforms, params = next(iter(result['train_loader']))
    # waveforms: [batch, n_detectors, time_samples]
    # params:    [batch, n_params]  (raw, unnormalized)

    n_plot = min(args.n_plot, waveforms.shape[0])
    times  = np.arange(waveforms.shape[-1]) * TIME_RES

    fig, axes = plt.subplots(n_plot, 2, figsize=(12, 3 * n_plot))
    if n_plot == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['H1 (raw strain)', 'L1 (raw strain)']
    col_colors = [H1_COLOR, L1_COLOR]

    for col, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, col].set_title(title, fontsize=11, color=color, fontweight='bold')

    for i in range(n_plot):
        strain = waveforms[i].numpy()   # [n_detectors, time_samples]
        p      = params[i].numpy()

        m1 = p[m1_idx]
        m2 = p[m2_idx]
        d  = p[d_idx]

        row_label = f'm₁={m1:.1f} M☉\nm₂={m2:.1f} M☉\nd={d:.0f} Mpc'

        axes[i, 0].plot(times, strain[0], lw=0.5, color=H1_COLOR, alpha=0.85)
        axes[i, 1].plot(times, strain[1], lw=0.5, color=L1_COLOR, alpha=0.85)

        for col in range(2):
            ax = axes[i, col]
            ax.set_xlim(times[0], times[-1])
            ax.set_ylabel('Strain', fontsize=8)
            if i == n_plot - 1:
                ax.set_xlabel('Time (s)')
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        axes[i, 0].text(
            0.02, 0.97, row_label,
            transform=axes[i, 0].transAxes,
            fontsize=7, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
        )

    fig.suptitle('GR Waveforms — O4 Real Noise  (IMRPhenomD)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
