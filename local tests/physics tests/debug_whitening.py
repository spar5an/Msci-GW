"""
debug_whitening.py — Show processed waveforms: window → whiten → bandpass.

Run:
    python debug_whitening.py
"""

import sys
import os
from pathlib import Path

import numpy as np
from scipy.signal.windows import tukey as scipy_tukey
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DATA_GEN_DIR = str(
    Path(__file__).resolve().parent.parent.parent
    / "HPC" / "Pipeline" / "Data Generation"
)
sys.path.insert(0, _DATA_GEN_DIR)

from gw_datagen import (
    pycbc_data_generator,
    pycbc_massive_gravity_data_generator,
    pycbc_lorentz_violation_data_generator,
    whiten_waveform,
    _DEFAULT_CACHE,
    _cache_path,
)

_M_G         = 2.21e-58
_ALPHA_LV    = 3.0
_A_LV        = 1e15
_SAMPLE_RATE = 4096
_DELTA_T     = 1.0 / _SAMPLE_RATE
_F_LOWER     = 40.0
_TUKEY_ALPHA = 0.1
_OUT_DIR     = Path(__file__).parent

_SMALL_CONFIG = {
    "mass1":        lambda size: np.random.uniform(20, 40, size=size),
    "mass2":        lambda size: np.random.uniform(20, 40, size=size),
    "spin1z":       lambda size: np.zeros(size),
    "spin2z":       lambda size: np.zeros(size),
    "distance":     lambda size: np.random.uniform(200, 500, size=size),
    "inclination":  lambda size: np.zeros(size),
    "coa_phase":    lambda size: np.zeros(size),
    "ra":           lambda size: np.zeros(size),
    "dec":          lambda size: np.zeros(size),
    "polarization": lambda size: np.zeros(size),
    "redshift":     lambda size: np.random.uniform(0.05, 0.15, size=size),
}

_COMMON_KWARGS = dict(
    num_samples=6,
    time_resolution=_DELTA_T,
    approximant="IMRPhenomD",
    f_lower=_F_LOWER,
    f_final=2048.0,
    signal_length=2.0,
    batch_size=6,
    train_split=0.7,
    val_split=0.15,
    add_noise=True,
    noise_backend="o4_psd",
    psd_cache_dir=_DEFAULT_CACHE,
    num_workers=1,
    show_progress=False,
)


def _process(waveform):
    """Window → whiten → bandpass."""
    window = scipy_tukey(len(waveform), alpha=_TUKEY_ALPHA)
    whitened, _, _ = whiten_waveform(
        waveform * window,
        delta_t=_DELTA_T,
        f_lower=_F_LOWER,
        apply_bandpass=True,
        apply_tukey=False,
    )
    return whitened


def _get_waveforms(result, n=3):
    X = result["train_loader"].dataset.dataset.tensors[0]
    return [X[i, 0].numpy() for i in range(min(n, X.shape[0]))]


if __name__ == "__main__":
    if not os.path.isfile(_cache_path("H1", _SAMPLE_RATE, _DEFAULT_CACHE)):
        print("ERROR: O4 PSD cache not found.")
        sys.exit(1)

    print("Generating datasets…")
    gr = pycbc_data_generator(config=_SMALL_CONFIG, detectors=["H1"], **_COMMON_KWARGS)
    mg = pycbc_massive_gravity_data_generator(
        config=_SMALL_CONFIG, m_g=_M_G, detectors=["H1"], **_COMMON_KWARGS)
    lv = pycbc_lorentz_violation_data_generator(
        config=_SMALL_CONFIG, alpha_lv=_ALPHA_LV, A=_A_LV,
        detectors=["H1"], **_COMMON_KWARGS)

    datasets = [
        ("GR",                        "steelblue",      _get_waveforms(gr)),
        (f"MG  m_g={_M_G:.2e} kg",   "tomato",         _get_waveforms(mg)),
        (f"LV  α={_ALPHA_LV}",        "mediumseagreen", _get_waveforms(lv)),
    ]

    n_waves = len(datasets[0][2])
    fig, axes = plt.subplots(
        len(datasets), n_waves,
        figsize=(4.5 * n_waves, 3 * len(datasets)),
        sharex=True,
    )
    fig.suptitle(
        f"Processed waveforms: window → whiten → bandpass  (H1, Tukey α={_TUKEY_ALPHA})",
        fontsize=11,
    )

    t = np.arange(len(datasets[0][2][0])) * _DELTA_T

    for row, (label, color, waveforms) in enumerate(datasets):
        for col, wf in enumerate(waveforms):
            ax = axes[row, col]
            processed = _process(wf)
            ax.plot(t, processed, lw=0.5, color=color)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == len(datasets) - 1:
                ax.set_xlabel("Time (s)")
            ax.set_xlim(t[0], t[-1])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _OUT_DIR / "debug_processed_waveforms.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
