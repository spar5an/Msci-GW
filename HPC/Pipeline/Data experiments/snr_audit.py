"""
snr_audit.py — compute optimal matched-filter SNR for every event in the
simulated training set and report how many land in a "detectable" regime.

For each event we build the frequency-domain template at its true parameters,
project onto each detector using the antenna patterns at the training GPS
time, and compute sigma() (= the SNR a perfect matched filter would achieve
against stationary Gaussian noise with the representative O4 PSD).

Outputs
-------
Data/snr_audit.csv       columns: idx, snr_H1, snr_L1, snr_network,
                                  chirp_mass, mass1, mass2, distance, inclination
plots/snr_audit.png      3 panels: network-SNR histogram,
                                   SNR vs chirp mass,
                                   SNR vs distance
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from pycbc.waveform import get_fd_waveform
from pycbc.detector import Detector
from pycbc.filter import sigma
from pycbc.types import FrequencySeries
from scipy.interpolate import interp1d


# Training reference time — source of (F+, Fx) per event.
T_REF = 1126259462.4

# SNR integration band (standard LIGO analysis choice, matches bandpass).
F_LOWER_INTEG = 20.0
F_UPPER_INTEG = 300.0

# Template delta_f. 1/16 Hz ↔ 16 s effective duration — plenty for the inspiral
# of a 5+5 M⊙ binary from 10 Hz, and ~250× faster than the 1/256 Hz used for
# the actual signal generation.
DELTA_F = 1.0 / 16
F_LOWER_TEMPLATE = 10.0
F_UPPER_TEMPLATE = 2048.0
APPROXIMANT = 'IMRPhenomD'

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
DATASET_PATH = PIPELINE / 'Data' / 'dataset.pt'
PSD_CACHE_DIR = PIPELINE / 'Data Generation' / 'o4_psd_cache'
CSV_OUT = HERE / 'outputs' / 'snr_audit.csv'
PNG_OUT = HERE / 'plots' / 'snr_audit.png'


def load_average_psd(detector: str, delta_f: float, flen: int) -> FrequencySeries:
    """Return a representative detector PSD: mean of cached O4 segments,
    interpolated onto the (delta_f, flen) grid."""
    path = PSD_CACHE_DIR / f'o4_psds_{detector}_4096Hz.npz'
    data = np.load(path)
    freqs_cache = data['freqs']           # (M,)
    psd_mean = data['psds'].mean(axis=0)  # (M,) — average across segments

    f_out = np.arange(flen) * delta_f
    safe = freqs_cache > 0
    interp = interp1d(
        np.log10(freqs_cache[safe]),
        np.log10(psd_mean[safe]),
        bounds_error=False,
        fill_value=(np.log10(psd_mean[safe][0]),
                    np.log10(psd_mean[safe][-1])),
    )
    psd_out = np.empty(flen, dtype=np.float64)
    psd_out[0] = 1.0
    psd_out[1:] = 10 ** interp(np.log10(f_out[1:]))
    # Wall below f_lower so sigma integration ignores it cleanly.
    above = f_out >= F_LOWER_INTEG
    wall = psd_out[above][0] if above.any() else 1e-44
    psd_out[(f_out > 0) & (f_out < F_LOWER_INTEG)] = wall
    return FrequencySeries(psd_out, delta_f=delta_f)


def optimal_snr_one_event(
    params: dict,
    psd_h1: FrequencySeries,
    psd_l1: FrequencySeries,
) -> tuple[float, float]:
    hp, hc = get_fd_waveform(
        approximant=APPROXIMANT,
        mass1=params['mass1'], mass2=params['mass2'],
        spin1z=params['spin1z'], spin2z=params['spin2z'],
        inclination=params['inclination'],
        coa_phase=params['coa_phase'],
        distance=params['distance'],
        delta_f=DELTA_F,
        f_lower=F_LOWER_TEMPLATE,
        f_final=F_UPPER_TEMPLATE,
    )

    snrs = []
    for det_name, psd in [('H1', psd_h1), ('L1', psd_l1)]:
        fp, fc = Detector(det_name).antenna_pattern(
            params['ra'], params['dec'], params['polarization'], T_REF)
        h_det = fp * hp + fc * hc

        # Truncate/pad both series to the PSD length so sigma's inner product
        # can line them up bin-for-bin.
        n = min(len(h_det), len(psd))
        h_det = FrequencySeries(h_det.numpy()[:n], delta_f=h_det.delta_f)
        psd_cut = FrequencySeries(psd.numpy()[:n], delta_f=psd.delta_f)

        s = sigma(h_det, psd=psd_cut,
                  low_frequency_cutoff=F_LOWER_INTEG,
                  high_frequency_cutoff=F_UPPER_INTEG)
        snrs.append(float(s))

    return snrs[0], snrs[1]


def main():
    print(f"Loading dataset from {DATASET_PATH} …")
    d = torch.load(DATASET_PATH, weights_only=False)
    y = d['y'].numpy()
    meta = d['metadata']
    names = meta['parameter_names']
    col = {n: i for i, n in enumerate(names)}
    N = y.shape[0]

    # flen sized so f_max = 2048 Hz at the template's delta_f.
    flen = int(round(F_UPPER_TEMPLATE / DELTA_F)) + 1
    print(f"Building average PSDs (flen={flen}, delta_f={DELTA_F}) …")
    psd_h1 = load_average_psd('H1', DELTA_F, flen)
    psd_l1 = load_average_psd('L1', DELTA_F, flen)

    print(f"Computing optimal SNR for {N} events …")
    rows = []
    for i in tqdm(range(N)):
        p = {n: float(y[i, col[n]]) for n in
             ('mass1', 'mass2', 'spin1z', 'spin2z', 'distance',
              'inclination', 'coa_phase', 'ra', 'dec', 'polarization')}
        try:
            snr_h1, snr_l1 = optimal_snr_one_event(p, psd_h1, psd_l1)
        except Exception as e:
            print(f"\n  warning: event {i} failed ({e}); SNR set to NaN")
            snr_h1 = snr_l1 = np.nan
        snr_net = float(np.sqrt(snr_h1**2 + snr_l1**2))
        chirp = (p['mass1'] * p['mass2']) ** 0.6 / (p['mass1'] + p['mass2']) ** 0.2
        rows.append((i, snr_h1, snr_l1, snr_net, chirp,
                     p['mass1'], p['mass2'], p['distance'], p['inclination']))

    df = pd.DataFrame(rows, columns=[
        'idx', 'snr_H1', 'snr_L1', 'snr_network',
        'chirp_mass', 'mass1', 'mass2', 'distance', 'inclination'])

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"\nWrote {CSV_OUT}")

    # Summary
    print("\nSNR summary:")
    print(f"  network SNR  median = {df['snr_network'].median():.2f}")
    print(f"  network SNR  mean   = {df['snr_network'].mean():.2f}")
    print(f"  fraction below 8    = {(df['snr_network'] < 8).mean():.1%}")
    print(f"  fraction above 50   = {(df['snr_network'] > 50).mean():.1%}")

    # Plots
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    bins = np.logspace(np.log10(0.1), np.log10(df['snr_network'].max() * 1.1), 60)
    ax.hist(df['snr_network'].dropna(), bins=bins, color='tab:blue', alpha=0.8)
    ax.axvline(8, color='k', ls='--', lw=0.8, label='SNR = 8')
    ax.set_xscale('log')
    ax.set_xlabel('network SNR (log scale)')
    ax.set_ylabel('events')
    med = df['snr_network'].median()
    frac8 = (df['snr_network'] < 8).mean()
    ax.set_title(f'Network SNR distribution (N={len(df)})\n'
                 f'median={med:.1f}, {frac8:.0%} below SNR 8')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ax.scatter(df['chirp_mass'], df['snr_network'], s=2, alpha=0.25,
               color='tab:blue')
    ax.set_xlabel('chirp mass [M⊙]')
    ax.set_ylabel('network SNR')
    ax.set_title('SNR vs chirp mass')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(df['distance'], df['snr_network'], s=2, alpha=0.25,
               color='tab:orange')
    ax.set_xlabel('luminosity distance [Mpc]')
    ax.set_ylabel('network SNR')
    ax.set_title('SNR vs distance')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=150)
    print(f"Wrote {PNG_OUT}")


if __name__ == '__main__':
    sys.exit(main())
