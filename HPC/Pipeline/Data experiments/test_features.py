"""
test_features.py — smoke test for gw_datagen's derived-representation API.

Generates one noisy waveform per mode (GR, MG, LV), computes both the
Q-transform and an SVD projection for each, and plots the results to
plots/features_smoke_test.png.

The SVD basis for each mode is built from a small ad-hoc pool of clean
templates (no noise injection) so projection coefficients aren't trivially
degenerate. The pool is kept small so the whole script finishes in < 30 s.

Run:
    python test_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
sys.path.insert(0, str(PIPELINE / 'Data Generation'))

from gw_datagen import (                           # noqa: E402
    _generate_single_waveform,
    _generate_single_modified_waveform,
    _generate_single_lv_waveform,
    whiten_waveform,
    _average_o4_psd,
    _clean_whitened_template,
    build_svd_basis,
    apply_svd_projection,
    compute_qtransform_batch,
    detect_mode,
)


TARGET_LENGTH = 8192
TIME_RESOLUTION = 1.0 / 4096
F_LOWER = 10.0
F_LOWER_WHITEN = 20.0
DETECTOR = 'H1'
N_BASIS_TEMPLATES = 8
K_SVD = 6
PLOT_PATH = HERE / 'plots' / 'features_smoke_test.png'


def _random_params(rng: np.random.Generator, mode: str) -> dict:
    p = {
        'mass1':        float(rng.uniform(20, 60)),
        'mass2':        float(rng.uniform(20, 60)),
        'spin1z':       float(rng.uniform(-0.3, 0.3)),
        'spin2z':       float(rng.uniform(-0.3, 0.3)),
        'distance':     float(rng.uniform(200, 1200)),
        'inclination':  float(np.arccos(rng.uniform(-1, 1))),
        'coa_phase':    float(rng.uniform(0, 2 * np.pi)),
        'ra':           float(rng.uniform(0, 2 * np.pi)),
        'dec':          float(np.arcsin(rng.uniform(-1, 1))),
        'polarization': float(rng.uniform(0, np.pi)),
        'm_g':          0.0,
        'alpha_lv':     0.0,
        'A':            0.0,
    }
    if mode == 'mg':
        p['m_g'] = float(rng.uniform(2.21e-58, 2.21e-56))
    elif mode == 'lv':
        p['m_g'] = float(rng.uniform(2.21e-58, 2.21e-56))
        p['alpha_lv'] = 3.0
        p['A'] = float(rng.uniform(1e20, 1e22))
    return p


def _generate_noisy_whitened(params: dict, mode: str, psd) -> np.ndarray:
    """Run the per-mode worker with noise injection, then whiten."""
    if mode == 'gr':
        r = _generate_single_waveform(
            params=params, time_resolution=TIME_RESOLUTION,
            approximant='IMRPhenomD', f_lower=F_LOWER,
            detectors=[DETECTOR], target_length=TARGET_LENGTH,
            add_noise=True, f_final=2048.0,
        )
    elif mode == 'mg':
        r = _generate_single_modified_waveform(
            params=params, time_resolution=TIME_RESOLUTION,
            approximant='IMRPhenomD', f_lower=F_LOWER,
            detectors=[DETECTOR], target_length=TARGET_LENGTH,
            add_noise=True, m_g=params['m_g'], f_final=2048.0,
        )
    elif mode == 'lv':
        r = _generate_single_lv_waveform(
            params=params, time_resolution=TIME_RESOLUTION,
            approximant='IMRPhenomD', f_lower=F_LOWER,
            detectors=[DETECTOR], target_length=TARGET_LENGTH,
            add_noise=True, m_g=params['m_g'],
            alpha_lv=params['alpha_lv'], A=params['A'],
            f_final=2048.0,
        )
    else:
        raise ValueError(mode)
    if not r['success']:
        raise RuntimeError(r.get('error', 'generation failed'))
    w, _, _ = whiten_waveform(
        r['detectors'][DETECTOR].numpy(),
        delta_t=TIME_RESOLUTION, f_lower=F_LOWER_WHITEN, psd=psd,
    )
    return w.astype(np.float32)


def main() -> int:
    rng = np.random.default_rng(0)
    delta_f = 1.0 / (TARGET_LENGTH * TIME_RESOLUTION)
    flen = TARGET_LENGTH // 2 + 1
    psd = _average_o4_psd(DETECTOR, delta_f, flen)

    modes = ['gr', 'mg', 'lv']

    # Sanity: detect_mode on synthetic label rows
    fake_col = {'m_g': 10, 'alpha_lv': 11, 'A': 12}
    assert detect_mode(np.zeros((3, 13)), fake_col) == 'gr'
    y_mg = np.zeros((3, 13)); y_mg[:, 10] = 1e-57
    assert detect_mode(y_mg, fake_col) == 'mg'
    y_lv = y_mg.copy(); y_lv[:, 11] = 3.0
    assert detect_mode(y_lv, fake_col) == 'lv'
    print("[PASS] detect_mode returns gr/mg/lv for synthetic label rows")

    # Per-mode: build a small basis + generate 1 noisy waveform + features
    results = {}
    for mode in modes:
        print(f"\n--- {mode.upper()} ---")
        # Basis from N_BASIS_TEMPLATES clean templates of the same mode.
        templates = np.zeros((N_BASIS_TEMPLATES, TARGET_LENGTH), dtype=np.float32)
        for i in range(N_BASIS_TEMPLATES):
            templates[i] = _clean_whitened_template(
                _random_params(rng, mode), mode=mode, detector=DETECTOR, psd=psd,
            )
        basis, svals = build_svd_basis(templates, k_max=K_SVD)
        assert basis.shape == (K_SVD, TARGET_LENGTH)
        assert np.all(svals[:-1] >= svals[1:])
        print(f"  [PASS] build_svd_basis → basis={basis.shape}, "
              f"singular values {svals.round(3).tolist()}")

        # One fresh noisy waveform drawn from the same prior.
        params = _random_params(rng, mode)
        x = _generate_noisy_whitened(params, mode, psd)
        X = x[None, None, :]  # (N=1, D=1, T)

        # Q-transform on the batch wrapper
        Q = compute_qtransform_batch(X)
        assert Q.shape == (1, 1, 50, 1000)
        assert Q.dtype == np.float32
        assert np.all(Q >= 0)
        print(f"  [PASS] compute_qtransform_batch → shape {Q.shape}, "
              f"max |q|={Q.max():.2f}")

        # SVD projection
        coeffs = apply_svd_projection(X, basis)
        assert coeffs.shape == (1, 1, K_SVD)
        assert np.isfinite(coeffs).all()
        print(f"  [PASS] apply_svd_projection → coeffs shape {coeffs.shape}, "
              f"values {coeffs[0, 0].round(2).tolist()}")

        results[mode] = {
            'strain': x, 'params': params, 'Q': Q[0, 0], 'coeffs': coeffs[0, 0],
            'svals': svals,
        }

    # ── Plot 3×3 grid: (mode row) × (strain | Q-transform | SVD coeffs) ─────
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    t = (np.arange(TARGET_LENGTH) - TARGET_LENGTH // 2) * TIME_RESOLUTION

    for row, mode in enumerate(modes):
        r = results[mode]
        p = r['params']
        label = (f"{mode.upper()} | m1={p['mass1']:.0f}, m2={p['mass2']:.0f}, "
                 f"d={p['distance']:.0f} Mpc")

        ax = axes[row, 0]
        ax.plot(t, r['strain'], lw=0.5, color='tab:blue')
        ax.set_title(f'whitened strain — {label}', fontsize=9)
        ax.set_ylabel('strain')
        ax.axvline(0, color='k', lw=0.4, alpha=0.4)
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        im = ax.imshow(r['Q'], origin='lower', aspect='auto',
                       extent=(-1.0, 1.0, 20, 300), cmap='viridis')
        ax.set_yscale('log')
        ax.set_title(f'Q-transform magnitude ({r["Q"].shape[0]}×{r["Q"].shape[1]})',
                     fontsize=9)
        ax.set_ylabel('frequency [Hz]')
        fig.colorbar(im, ax=ax, label='|q|')

        ax = axes[row, 2]
        ax.bar(np.arange(K_SVD), r['coeffs'], color='tab:green')
        ax.set_title(f'SVD coefficients (k={K_SVD})', fontsize=9)
        ax.set_xlabel('mode index')
        ax.set_ylabel('coefficient')
        ax.grid(alpha=0.3)

    axes[-1, 0].set_xlabel('time from merger [s]')
    axes[-1, 1].set_xlabel('time from merger [s]')

    fig.suptitle('Q-transform & SVD smoke test — one noisy waveform per mode',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"\nsaved {PLOT_PATH}")
    print("All smoke-test checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
