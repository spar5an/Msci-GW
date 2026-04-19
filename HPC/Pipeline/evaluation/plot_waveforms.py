# plot_waveforms.py — sanity-check a few waveforms from Data/dataset.pt.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = str(PIPELINE_ROOT / 'Data' / 'dataset.pt')
PLOT_DIR = PIPELINE_ROOT / 'plots'
PLOT_DIR.mkdir(exist_ok=True)
NUM_EVENTS = 4
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    d = torch.load(DATASET_PATH, weights_only=False)
    Xw, y = d['X_whitened'].numpy(), d['y'].numpy()
    meta = d['metadata']
    param_names = meta['parameter_names']
    dt = meta['time_resolution']
    T = Xw.shape[-1]
    t = np.arange(T) * dt

    idxs = rng.choice(Xw.shape[0], size=NUM_EVENTS, replace=False)

    fig, axes = plt.subplots(NUM_EVENTS, 1, figsize=(10, 2.4 * NUM_EVENTS),
                             sharex=True, squeeze=False)
    for row, idx in enumerate(idxs):
        p = dict(zip(param_names, y[idx]))
        label = (f"idx {idx}: m1={p['mass1']:.1f}, m2={p['mass2']:.1f} M⊙, "
                 f"d_L={p['distance']:.0f} Mpc, ι={p['inclination']:.2f}")

        ax = axes[row, 0]
        for det, name, color in [(0, 'H1', 'tab:blue'), (1, 'L1', 'tab:orange')]:
            ax.plot(t, Xw[idx, det], color=color, lw=0.6, alpha=0.8, label=name)
        ax.set_ylabel('whitened strain')
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[-1, 0].set_xlabel('time [s]')

    fig.suptitle(f"Whitened sample waveforms from {DATASET_PATH} "
                 f"(add_noise={meta['add_noise']}, {meta['approximant']})",
                 fontsize=11)
    fig.tight_layout()
    out = PLOT_DIR / 'waveform_samples.png'
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    # Zoomed view on the last ~0.5 s where the merger lives.
    fig2, axes2 = plt.subplots(NUM_EVENTS, 1, figsize=(10, 2.4 * NUM_EVENTS),
                               sharex=True, squeeze=False)
    mask = t > t[-1] - 0.5
    for row, idx in enumerate(idxs):
        ax = axes2[row, 0]
        for det, name, color in [(0, 'H1', 'tab:blue'), (1, 'L1', 'tab:orange')]:
            ax.plot(t[mask], Xw[idx, det][mask], color=color, lw=0.8, label=name)
        ax.set_ylabel('whitened strain')
        ax.grid(alpha=0.3)
        ax.set_title(f'idx {idx} (zoom on merger)', fontsize=9)
        if row == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes2[-1, 0].set_xlabel('time [s]')
    fig2.suptitle('Last 0.5 s (merger / ringdown region)', fontsize=11)
    fig2.tight_layout()
    out2 = PLOT_DIR / 'waveform_samples_zoom.png'
    fig2.savefig(out2, dpi=150)
    print(f"saved {out2}")


if __name__ == '__main__':
    main()
