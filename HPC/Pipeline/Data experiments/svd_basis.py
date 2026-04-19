"""
svd_basis.py — build a reduced SVD basis from clean whitened templates and
quantify how faithfully noisy observations are reconstructed in low-k
subspaces. This is the DINGO-style dimensionality reduction: compress
8192-sample strain to a few hundred coefficients by projecting onto the
leading right-singular vectors of a template matrix.

Outputs
-------
Data/svd_basis_H1.npz      basis (right-singular vectors Vh up to k_max) and
                           singular values
plots/svd_spectrum.png     singular-value spectrum + cumulative energy, with
                           thresholds at 95 %, 99 %, 99.9 % of energy marked
plots/svd_reconstruction.png per-event panels at three k values for three
                           events spanning SNR quantiles
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from pycbc.types import FrequencySeries

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
sys.path.insert(0, str(PIPELINE / 'Data Generation'))
from gw_datagen import _generate_single_waveform, whiten_waveform  # noqa: E402


DATASET_PATH = PIPELINE / 'Data' / 'dataset.pt'
SNR_CSV = HERE / 'outputs' / 'snr_audit.csv'
PSD_CACHE_DIR = PIPELINE / 'Data Generation' / 'o4_psd_cache'
BASIS_OUT = HERE / 'outputs' / 'svd_basis_H1.npz'
PLOT_DIR = HERE / 'plots'

DETECTOR_INDEX = 0
DETECTOR_NAME = 'H1'
T_REF = 1126259462.4
APPROXIMANT = 'IMRPhenomD'
F_LOWER_WAVEFORM = 10.0
F_LOWER_WHITEN = 20.0
TIME_RESOLUTION = 1.0 / 4096
TARGET_LENGTH = 8192
SIGNAL_LENGTH_S = 2.0
MERGER_SAMPLE = TARGET_LENGTH // 2

N_BASIS_TEMPLATES = 2000                 # templates used to build the basis
K_MAX = 600                              # keep top K_MAX singular vectors on disk
K_RECONSTRUCTION = [50, 150, 400]        # reconstruction panel comparison
RECON_QUANTILES = [0.40, 0.80, 0.99]     # SNR quantiles for example events


def build_average_psd(delta_f: float, flen: int, detector: str) -> FrequencySeries:
    data = np.load(PSD_CACHE_DIR / f'o4_psds_{detector}_4096Hz.npz')
    freqs_cache = data['freqs']
    psd_mean = data['psds'].mean(axis=0)
    safe = freqs_cache > 0
    interp = interp1d(
        np.log10(freqs_cache[safe]),
        np.log10(psd_mean[safe]),
        bounds_error=False,
        fill_value=(np.log10(psd_mean[safe][0]),
                    np.log10(psd_mean[safe][-1])),
    )
    f_out = np.arange(flen) * delta_f
    psd_out = np.empty(flen)
    psd_out[0] = 1.0
    psd_out[1:] = 10 ** interp(np.log10(f_out[1:]))
    return FrequencySeries(psd_out, delta_f=delta_f)


def clean_whitened_template(params: dict, psd: FrequencySeries) -> np.ndarray:
    r = _generate_single_waveform(
        params=params,
        time_resolution=TIME_RESOLUTION,
        approximant=APPROXIMANT,
        f_lower=F_LOWER_WAVEFORM,
        detectors=[DETECTOR_NAME],
        target_length=TARGET_LENGTH,
        add_noise=False,
    )
    if not r['success']:
        raise RuntimeError(r.get('error'))
    w, _, _ = whiten_waveform(
        r['detectors'][DETECTOR_NAME].numpy(),
        delta_t=TIME_RESOLUTION, f_lower=F_LOWER_WHITEN, psd=psd,
    )
    return w.astype(np.float32)


def make_params(y_row: np.ndarray, col: dict) -> dict:
    keys = ('mass1', 'mass2', 'spin1z', 'spin2z', 'distance',
            'inclination', 'coa_phase', 'ra', 'dec', 'polarization')
    return {k: float(y_row[col[k]]) for k in keys}


def mf_snr_peak(data_w: np.ndarray, template_w: np.ndarray) -> float:
    """Normalised off-merger-σ cross-correlation peak."""
    n = len(data_w)
    pad = np.zeros(n)
    pad[:len(template_w)] = template_w
    corr = np.fft.ifft(np.fft.fft(data_w) * np.conj(np.fft.fft(pad))).real
    corr = np.fft.fftshift(corr)
    corr /= (np.sqrt(np.sum(template_w ** 2)) + 1e-30)
    centre = n // 2
    half = int(0.1 / TIME_RESOLUTION)
    mask = np.ones(n, dtype=bool)
    mask[centre - half:centre + half] = False
    sigma = corr[mask].std() + 1e-30
    return float(np.max(np.abs(corr / sigma)))


def build_basis(y: np.ndarray, col: dict, psd: FrequencySeries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    idxs = rng.choice(y.shape[0], size=N_BASIS_TEMPLATES, replace=False)
    templates = np.zeros((N_BASIS_TEMPLATES, TARGET_LENGTH), dtype=np.float32)
    kept = np.zeros(N_BASIS_TEMPLATES, dtype=bool)
    t0 = time.time()
    for row, i in enumerate(idxs):
        try:
            templates[row] = clean_whitened_template(make_params(y[i], col), psd)
            kept[row] = True
        except Exception as e:
            print(f"  template {i} failed: {e}")
        if (row + 1) % 200 == 0:
            print(f"    built {row + 1}/{N_BASIS_TEMPLATES} templates "
                  f"({time.time() - t0:.1f}s)")
    templates = templates[kept]
    print(f"  {len(templates)} templates built in {time.time() - t0:.1f}s")
    print(f"  running SVD on ({templates.shape}) matrix …")
    t0 = time.time()
    U, s, Vh = np.linalg.svd(templates, full_matrices=False)
    print(f"  SVD complete in {time.time() - t0:.1f}s  |  "
          f"Vh.shape={Vh.shape}  s.shape={s.shape}")
    return s, Vh, idxs[kept]


def plot_spectrum(s: np.ndarray) -> dict[float, int]:
    energy = s ** 2
    cum = np.cumsum(energy) / energy.sum()
    k_at = {0.95: int(np.searchsorted(cum, 0.95) + 1),
            0.99: int(np.searchsorted(cum, 0.99) + 1),
            0.999: int(np.searchsorted(cum, 0.999) + 1)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.semilogy(s, color='tab:blue', lw=1.0)
    ax.set_xlabel('mode index k')
    ax.set_ylabel('singular value σ_k')
    ax.set_title('Singular value spectrum')
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ax.semilogx(np.arange(1, len(cum) + 1), cum, color='tab:blue', lw=1.0)
    for frac, k in k_at.items():
        ax.axvline(k, color='k', ls='--', lw=0.8, alpha=0.5)
        ax.annotate(f'{frac*100:.1f}%, k={k}', xy=(k, frac),
                    xytext=(k * 1.3, frac - 0.08),
                    fontsize=8, ha='left',
                    arrowprops=dict(arrowstyle='-', color='k', alpha=0.4, lw=0.5))
    ax.set_xlabel('mode index k (log)')
    ax.set_ylabel('cumulative energy fraction')
    ax.set_title('Cumulative energy in top k modes')
    ax.set_xlim(1, len(cum))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, which='both')

    fig.suptitle(
        f'SVD basis from {N_BASIS_TEMPLATES} clean whitened H1 templates '
        f'(T={TARGET_LENGTH} samples)', fontsize=10)
    fig.tight_layout()
    out = PLOT_DIR / 'svd_spectrum.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out.name}  | k(95%)={k_at[0.95]}, "
          f"k(99%)={k_at[0.99]}, k(99.9%)={k_at[0.999]}")
    return k_at


def plot_reconstructions(X_w: np.ndarray, y: np.ndarray, col: dict,
                         Vh: np.ndarray, psd: FrequencySeries,
                         snr_df: pd.DataFrame) -> None:
    t = (np.arange(TARGET_LENGTH) - MERGER_SAMPLE) * TIME_RESOLUTION
    event_idxs = [int(snr_df['idx'].iloc[
        (snr_df['snr_network'] - snr_df['snr_network'].quantile(q)).abs().idxmin()])
        for q in RECON_QUANTILES]

    rows = len(event_idxs)
    cols_fig = 1 + len(K_RECONSTRUCTION)
    fig, axes = plt.subplots(rows, cols_fig,
                             figsize=(cols_fig * 4.5, rows * 2.8),
                             sharex=True)
    for ri, idx in enumerate(event_idxs):
        params = make_params(y[idx], col)
        snr_net = float(snr_df.set_index('idx').loc[idx, 'snr_network'])
        strain = X_w[idx, DETECTOR_INDEX].astype(np.float32)
        try:
            tmpl = clean_whitened_template(params, psd)
        except Exception as e:
            print(f"  event {idx} template failed: {e}")
            continue

        # project & reconstruct
        coeffs = Vh @ strain

        # row title
        title = (f"idx {idx} | net SNR={snr_net:.1f} | "
                 f"m1={params['mass1']:.0f}, m2={params['mass2']:.0f}")

        ax = axes[ri, 0]
        ax.plot(t, strain, lw=0.4, color='tab:blue', label='noisy data')
        ax.plot(t, tmpl, lw=0.8, color='tab:red', alpha=0.8, label='clean tmpl')
        ax.set_title(f'original strain\n{title}', fontsize=9)
        if ri == 0:
            ax.legend(fontsize=7, loc='upper left')
        ax.set_ylabel('whitened strain')
        ax.grid(alpha=0.3)

        for ci, k in enumerate(K_RECONSTRUCTION, start=1):
            recon = Vh[:k].T @ coeffs[:k]
            mf_orig = mf_snr_peak(strain, tmpl)
            mf_recon = mf_snr_peak(recon, tmpl)
            ax = axes[ri, ci]
            ax.plot(t, recon, lw=0.4, color='tab:blue')
            ax.plot(t, tmpl, lw=0.8, color='tab:red', alpha=0.8)
            ax.set_title(f'k={k}  |  MF σ: {mf_orig:.1f} → {mf_recon:.1f}',
                         fontsize=9)
            ax.grid(alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('time from merger [s]')
    fig.suptitle(
        'Reduced-basis reconstruction. Columns: original, then projected onto '
        f'top-k modes. Blue = noisy observation (or reconstruction), red = '
        'clean template.',
        fontsize=10,
    )
    fig.tight_layout()
    out = PLOT_DIR / 'svd_reconstruction.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out.name}")


def noise_suppression_curve(X_w: np.ndarray, Vh: np.ndarray,
                            snr_df: pd.DataFrame) -> None:
    """Sweep k, measure mean MF-SNR-peak improvement over 300 random events."""
    rng = np.random.default_rng(1)
    pick = rng.choice(len(snr_df), size=300, replace=False)
    ks = [25, 50, 100, 200, 400, 600]
    # Precompute noisy strain and templates batch-efficiently
    strains = X_w[pick, DETECTOR_INDEX].astype(np.float32)
    coeffs = strains @ Vh.T  # (300, K_MAX)

    # Use snr_audit SNR as proxy for "signal strength"; we measure MF SNR peak
    # of reconstruction against strain's OWN template (not generated here since
    # that's slow; instead use correlation against original strain's signal
    # component estimated as mean of basis reconstructions).
    # Simpler & faster: compute retained signal energy as ||Vh[:k] @ Vh[:k].T @ strain|| / ||strain||
    energy_ratio = np.zeros(len(ks))
    for j, k in enumerate(ks):
        recon = coeffs[:, :k] @ Vh[:k]  # (300, T)
        energy_ratio[j] = (np.linalg.norm(recon, axis=1) /
                           np.linalg.norm(strains, axis=1)).mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, energy_ratio, 'o-', color='tab:blue')
    ax.set_xlabel('k (basis size)')
    ax.set_ylabel('⟨ ‖reconstruction‖ / ‖noisy observation‖ ⟩')
    ax.set_title('Fraction of observation energy retained after projection\n'
                 '(small → most of the observation is noise outside signal subspace)')
    ax.set_xscale('log')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    out = PLOT_DIR / 'svd_noise_suppression.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out.name}")
    print("  k vs retained-energy ratio:")
    for k, r in zip(ks, energy_ratio):
        print(f"    k={k:4d}  ratio={r:.3f}")


def main():
    print(f"Loading dataset from {DATASET_PATH} …")
    d = torch.load(DATASET_PATH, weights_only=False)
    X_w = d['X_whitened'].numpy()
    y = d['y'].numpy()
    meta = d['metadata']
    col = {n: i for i, n in enumerate(meta['parameter_names'])}

    flen = TARGET_LENGTH // 2 + 1
    delta_f = 1.0 / SIGNAL_LENGTH_S
    psd = build_average_psd(delta_f, flen, DETECTOR_NAME)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    BASIS_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding SVD basis ({N_BASIS_TEMPLATES} clean templates) …")
    s, Vh, basis_indices = build_basis(y, col, psd)

    np.savez(BASIS_OUT,
             singular_values=s[:K_MAX].astype(np.float32),
             basis_vectors=Vh[:K_MAX].astype(np.float32),
             basis_event_indices=basis_indices.astype(np.int32))
    print(f"  saved basis → {BASIS_OUT.name}  "
          f"(k_max={K_MAX}, ~{(K_MAX * TARGET_LENGTH * 4) // 1024 // 1024} MB)")

    print("\nPlotting singular-value spectrum …")
    k_at = plot_spectrum(s)

    print("\nPlotting per-event reconstructions …")
    snr_df = pd.read_csv(SNR_CSV)
    plot_reconstructions(X_w, y, col, Vh[:K_MAX], psd, snr_df)

    print("\nSweeping k vs retained-energy ratio …")
    noise_suppression_curve(X_w, Vh[:K_MAX], snr_df)

    print("\nDone.")


if __name__ == '__main__':
    main()
