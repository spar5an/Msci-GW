"""
test_signal_processing.py — Physics test: signal processing pipeline on O4-noised GR/MG/LV data.

Generates GR, MG, and LV waveforms injected into real O4 noise, applies the
whitening pipeline (whiten_dataloaders), and verifies the processed signals
satisfy physical expectations. Diagnostic plots show raw vs. whitened strain
for all three physics types.

Skipped automatically if the O4 PSD cache is missing.
To populate it, run once from the Data Generation directory:
    python download_o4_psds.py --detectors H1 L1 --n-segments 100

Run with:
    pytest test_signal_processing.py -v

Generate diagnostic plots (no pytest required):
    python test_signal_processing.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal.windows import tukey as scipy_tukey
from torch.utils.data import DataLoader, Subset, TensorDataset

# ── Path setup ────────────────────────────────────────────────────────────────
_DATA_GEN_DIR = str(
    Path(__file__).resolve().parent.parent.parent
    / "HPC" / "Pipeline" / "Data Generation"
)
sys.path.insert(0, _DATA_GEN_DIR)

from gw_datagen import (
    pycbc_data_generator,
    pycbc_massive_gravity_data_generator,
    pycbc_lorentz_violation_data_generator,
    whiten_dataloaders,
    _DEFAULT_CACHE,
    _cache_path,
)

# ── Physics / generation constants ────────────────────────────────────────────
_M_G      = 2.21e-58  # kg  (graviton mass; λ_g ≈ 1e16 m)
_ALPHA_LV = 3.0       # LV dispersion exponent (doubly special relativity)
_A_LV     = 1e15      # metres  (LV Compton wavelength)

_SAMPLE_RATE = 4096
_DELTA_T     = 1.0 / _SAMPLE_RATE
_SIGNAL_SECS = 2.0
_F_LOWER     = 40.0
_NUM_SAMPLES = 16
_NUM_WORKERS = 1   # avoids WSL2/Linux multiprocessing deadlocks inside pytest
_TUKEY_ALPHA = 0.1

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
    num_samples=_NUM_SAMPLES,
    time_resolution=_DELTA_T,
    approximant="IMRPhenomD",
    f_lower=_F_LOWER,
    f_final=2048.0,
    signal_length=_SIGNAL_SECS,
    batch_size=8,
    train_split=0.7,
    val_split=0.15,
    add_noise=True,
    noise_backend="o4_psd",
    psd_cache_dir=_DEFAULT_CACHE,
    num_workers=_NUM_WORKERS,
    show_progress=False,
)

_PLOTS_DIR = Path(__file__).parent


# ── Cache guard ───────────────────────────────────────────────────────────────
def _cache_available():
    return os.path.isfile(_cache_path("H1", _SAMPLE_RATE, _DEFAULT_CACHE))


pytestmark = pytest.mark.skipif(
    not _cache_available(),
    reason="O4 PSD cache not found. Run download_o4_psds.py to populate it.",
)


# ── Pipeline helper ──────────────────────────────────────────────────────────
def _prewindow_result(result, alpha=_TUKEY_ALPHA):
    """Pre-apply a Tukey window to every waveform in a result dict.

    Returns a new result dict with the same split structure so that the
    subsequent whiten_dataloaders call (with apply_tukey=False) implements
    the correct order: window → whiten → bandpass.
    """
    train_ds = result["train_loader"].dataset
    base_ds  = train_ds.dataset
    X = base_ds.tensors[0].clone().float()  # (N, D, T)
    y = base_ds.tensors[1]

    win = torch.from_numpy(scipy_tukey(X.shape[2], alpha=alpha).astype(np.float32))
    X   = X * win  # broadcast over (N, D)

    new_ds     = TensorDataset(X, y)
    batch_size = result["metadata"].get("batch_size", 8)
    train_idx  = train_ds.indices
    val_idx    = result["val_loader"].dataset.indices
    test_idx   = result["test_loader"].dataset.indices

    return {
        "train_loader": DataLoader(Subset(new_ds, train_idx), batch_size=batch_size, shuffle=True),
        "val_loader":   DataLoader(Subset(new_ds, val_idx),   batch_size=batch_size, shuffle=False),
        "test_loader":  DataLoader(Subset(new_ds, test_idx),  batch_size=batch_size, shuffle=False),
        "metadata":     result["metadata"].copy(),
    }


# ── Module-scoped fixtures ────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def gr_result():
    return pycbc_data_generator(
        config=_SMALL_CONFIG,
        detectors=["H1"],
        **_COMMON_KWARGS,
    )


@pytest.fixture(scope="module")
def mg_result():
    return pycbc_massive_gravity_data_generator(
        config=_SMALL_CONFIG,
        m_g=_M_G,
        detectors=["H1"],
        **_COMMON_KWARGS,
    )


@pytest.fixture(scope="module")
def lv_result():
    return pycbc_lorentz_violation_data_generator(
        config=_SMALL_CONFIG,
        alpha_lv=_ALPHA_LV,
        A=_A_LV,
        detectors=["H1"],
        **_COMMON_KWARGS,
    )


@pytest.fixture(scope="module")
def gr_whitened(gr_result):
    return whiten_dataloaders(
        _prewindow_result(gr_result), f_lower=_F_LOWER,
        apply_tukey=False, num_workers=_NUM_WORKERS, show_progress=False,
    )


@pytest.fixture(scope="module")
def mg_whitened(mg_result):
    return whiten_dataloaders(
        _prewindow_result(mg_result), f_lower=_F_LOWER,
        apply_tukey=False, num_workers=_NUM_WORKERS, show_progress=False,
    )


@pytest.fixture(scope="module")
def lv_whitened(lv_result):
    return whiten_dataloaders(
        _prewindow_result(lv_result), f_lower=_F_LOWER,
        apply_tukey=False, num_workers=_NUM_WORKERS, show_progress=False,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _first_waveform(result):
    """Return the first H1 waveform from the train loader as a numpy array."""
    X, _ = next(iter(result["train_loader"]))
    return X[0, 0].numpy()


def _process_waveform(waveform):
    """Apply the correct pipeline: window → whiten → bandpass."""
    from gw_datagen import whiten_waveform
    window = scipy_tukey(len(waveform), alpha=_TUKEY_ALPHA)
    whitened, _, _ = whiten_waveform(
        waveform * window,
        delta_t=_DELTA_T,
        f_lower=_F_LOWER,
        apply_bandpass=True,
        apply_tukey=False,
    )
    return whitened


def _raw_waveform(result, idx=0):
    """Return waveform idx directly from the base tensor (no shuffle)."""
    X = result["train_loader"].dataset.dataset.tensors[0]
    return X[idx, 0].numpy()


# ── Tests: output contract ────────────────────────────────────────────────────
class TestWhiteningOutputContract:
    """Whitened output must be finite, non-zero, and shape-preserving."""

    def test_gr_whitened_finite(self, gr_whitened):
        for X, _ in gr_whitened["train_loader"]:
            assert torch.isfinite(X).all(), "NaN/Inf in whitened GR train batch"

    def test_mg_whitened_finite(self, mg_whitened):
        for X, _ in mg_whitened["train_loader"]:
            assert torch.isfinite(X).all(), "NaN/Inf in whitened MG train batch"

    def test_lv_whitened_finite(self, lv_whitened):
        for X, _ in lv_whitened["train_loader"]:
            assert torch.isfinite(X).all(), "NaN/Inf in whitened LV train batch"

    def test_shape_preserved(self, gr_result, gr_whitened):
        X_raw, _ = next(iter(gr_result["train_loader"]))
        X_whi, _ = next(iter(gr_whitened["train_loader"]))
        assert X_raw.shape == X_whi.shape, (
            f"Shape changed after whitening: {X_raw.shape} -> {X_whi.shape}"
        )

    def test_whitening_changes_values(self, gr_result, gr_whitened):
        X_raw, _ = next(iter(gr_result["train_loader"]))
        X_whi, _ = next(iter(gr_whitened["train_loader"]))
        assert not torch.equal(X_raw, X_whi), (
            "Whitened output is identical to raw input — whitening had no effect"
        )

    def test_metadata_keys_preserved(self, gr_result, gr_whitened):
        for key in ("num_samples", "waveform_shape", "channels",
                    "train_size", "val_size", "test_size", "time_resolution"):
            assert gr_whitened["metadata"][key] == gr_result["metadata"][key], (
                f"Metadata key '{key}' changed after whitening"
            )


# ── Tests: physical expectations ──────────────────────────────────────────────
class TestWhiteningPhysics:
    """
    Whitening should rescale strain from ~1e-22 to O(1) amplitude.
    A flat (white) noise floor in units of sqrt(Hz)^{-1} maps to
    unit-variance time-domain noise, so the RMS of a whitened
    noise-dominated segment should be O(1).
    """

    def test_gr_amplitude_order_unity(self, gr_whitened):
        w   = _first_waveform(gr_whitened)
        rms = float(np.sqrt(np.mean(w ** 2)))
        assert 1e-3 < rms < 1e3, (
            f"GR whitened RMS {rms:.3e} is not O(1); "
            "whitening may not have applied correctly"
        )

    def test_mg_amplitude_order_unity(self, mg_whitened):
        w   = _first_waveform(mg_whitened)
        rms = float(np.sqrt(np.mean(w ** 2)))
        assert 1e-3 < rms < 1e3, (
            f"MG whitened RMS {rms:.3e} is not O(1)"
        )

    def test_lv_amplitude_order_unity(self, lv_whitened):
        w   = _first_waveform(lv_whitened)
        rms = float(np.sqrt(np.mean(w ** 2)))
        assert 1e-3 < rms < 1e3, (
            f"LV whitened RMS {rms:.3e} is not O(1)"
        )

    def test_whitening_upscales_strain(self, gr_result, gr_whitened):
        """
        Raw strain is ~1e-22; whitened should be orders of magnitude larger.
        This confirms the PSD division actually rescaled the data.
        """
        raw = _first_waveform(gr_result)
        whi = _first_waveform(gr_whitened)
        raw_rms = float(np.sqrt(np.mean(raw ** 2)))
        whi_rms = float(np.sqrt(np.mean(whi ** 2)))
        assert whi_rms > raw_rms * 1e10, (
            f"Whitened RMS ({whi_rms:.2e}) should be >> raw RMS ({raw_rms:.2e}); "
            "expected ~1e22x upscaling from PSD division"
        )

    def test_all_three_have_comparable_whitened_rms(self, gr_whitened, mg_whitened, lv_whitened):
        """
        After whitening with the same O4 PSD, GR, MG, and LV waveforms should
        have RMS values within a factor of 100 of each other — they all live in
        the same noise environment and the physics modification is small.
        """
        rms_gr = float(np.sqrt(np.mean(_first_waveform(gr_whitened) ** 2)))
        rms_mg = float(np.sqrt(np.mean(_first_waveform(mg_whitened) ** 2)))
        rms_lv = float(np.sqrt(np.mean(_first_waveform(lv_whitened) ** 2)))

        ratio_mg = max(rms_gr, rms_mg) / (min(rms_gr, rms_mg) + 1e-300)
        ratio_lv = max(rms_gr, rms_lv) / (min(rms_gr, rms_lv) + 1e-300)

        assert ratio_mg < 100, (
            f"GR vs MG whitened RMS ratio {ratio_mg:.1f} is unexpectedly large"
        )
        assert ratio_lv < 100, (
            f"GR vs LV whitened RMS ratio {ratio_lv:.1f} is unexpectedly large"
        )


# ── Tests: plots ──────────────────────────────────────────────────────────────
class TestPlots:
    """Diagnostic plots — no assertions on plot content."""

    def test_plot_pipeline(self, gr_result, mg_result, lv_result):
        _plot_all(gr_result, mg_result, lv_result,
                  None, None, None, out_dir=_PLOTS_DIR)


# ── Plot helper ───────────────────────────────────────────────────────────────
def _plot_all(gr_raw, mg_raw, lv_raw, gr_whi, mg_whi, lv_whi, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)

    datasets = [
        ("GR",                          "steelblue",      gr_raw),
        (f"MG  (m_g = {_M_G:.2e} kg)", "tomato",         mg_raw),
        (f"LV  (α = {_ALPHA_LV})",      "mediumseagreen", lv_raw),
    ]

    meta = gr_raw["metadata"]
    t    = np.arange(meta["waveform_shape"][1]) * meta["time_resolution"]

    # ── Plot 1: 2×3 grid — raw (top) vs processed (bottom) ───────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Signal processing: raw O4-noised vs window→whiten→bandpass  (H1, sample 0)",
        fontsize=12,
    )

    for col, (label, color, raw_res) in enumerate(datasets):
        raw = _raw_waveform(raw_res, idx=0)
        whi = _process_waveform(raw)

        ax_top = axes[0, col]
        ax_top.plot(t, raw, lw=0.4, color=color, alpha=0.85)
        ax_top.set_title(f"{label}\nraw + O4 noise", fontsize=10)
        ax_top.set_ylabel("Strain  H1")
        ax_top.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_top.set_xlim(t[0], t[-1])

        ax_bot = axes[1, col]
        ax_bot.plot(t, whi, lw=0.4, color=color, alpha=0.85)
        ax_bot.set_title(f"{label}\nprocessed", fontsize=10)
        ax_bot.set_xlabel("Time (s)")
        ax_bot.set_ylabel("Processed strain  H1")
        ax_bot.set_xlim(t[0], t[-1])

    fig.tight_layout()
    p1 = out_dir / "signal_processing_pipeline.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # ── Plot 2: overlay all three processed waveforms ─────────────────────────
    fig2, ax = plt.subplots(figsize=(12, 4))
    for label, color, raw_res in datasets:
        raw = _raw_waveform(raw_res, idx=0)
        ax.plot(t, _process_waveform(raw), lw=0.6, color=color, alpha=0.85, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Processed strain  H1")
    ax.set_title("Processed waveforms: GR vs MG vs LV  (H1, sample 0)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(t[0], t[-1])
    fig2.tight_layout()
    p2 = out_dir / "whitened_comparison.png"
    fig2.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {p2}")


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _cache_available():
        print(
            "ERROR: O4 PSD cache not found at\n"
            f"  {_DEFAULT_CACHE}\n"
            "Run download_o4_psds.py --detectors H1 L1 --n-segments 100 to populate it."
        )
        sys.exit(1)

    print("Generating GR dataset  (O4 noise)…")
    gr = pycbc_data_generator(config=_SMALL_CONFIG, detectors=["H1"], **_COMMON_KWARGS)

    print("Generating MG dataset  (O4 noise)…")
    mg = pycbc_massive_gravity_data_generator(
        config=_SMALL_CONFIG, m_g=_M_G, detectors=["H1"], **_COMMON_KWARGS
    )

    print("Generating LV dataset  (O4 noise)…")
    lv = pycbc_lorentz_violation_data_generator(
        config=_SMALL_CONFIG, alpha_lv=_ALPHA_LV, A=_A_LV,
        detectors=["H1"], **_COMMON_KWARGS,
    )

    print("Whitening GR…")
    gr_whi = whiten_dataloaders(_prewindow_result(gr), f_lower=_F_LOWER,
                                apply_tukey=False, num_workers=_NUM_WORKERS, show_progress=True)

    print("Whitening MG…")
    mg_whi = whiten_dataloaders(_prewindow_result(mg), f_lower=_F_LOWER,
                                apply_tukey=False, num_workers=_NUM_WORKERS, show_progress=True)

    print("Whitening LV…")
    lv_whi = whiten_dataloaders(_prewindow_result(lv), f_lower=_F_LOWER,
                                apply_tukey=False, num_workers=_NUM_WORKERS, show_progress=True)

    _plot_all(gr, mg, lv, None, None, None, out_dir=Path(__file__).parent)
