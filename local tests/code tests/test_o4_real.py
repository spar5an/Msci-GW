"""
test_o4_real.py — Batch 2: real O4 GWOSC PSDs.

Covers:
  - Loading and interpolating a pre-downloaded cached PSD
  - Generating noise from a cached PSD
  - End-to-end: GR, MG, and LV generators with noise_backend='o4_psd'
  - Save / load round-trip for all three generators

Requires a pre-populated O4 PSD cache (no internet access during tests).
To populate the cache, run once from the Data Generation directory:
    python download_o4_psds.py --detectors H1 L1 --n-segments 100

Tests are skipped automatically if the cache is missing.

Run with:
    pytest test_o4_real.py -v
"""

import os
import numpy as np
import pytest
import torch

from gw_datagen import (
    load_random_o4_psd,
    pycbc_data_generator,
    pycbc_massive_gravity_data_generator,
    pycbc_lorentz_violation_data_generator,
    save_dataloaders,
    load_dataloaders,
    _cache_path,
)
from pycbc.noise import noise_from_psd
from conftest import PLOTS_DIR

LAMBDA_G    = 1e22
ALPHA_LV    = 3.0
A_LV        = 1e15

DETECTOR    = "H1"
SAMPLE_RATE = 4096
SIGNAL_SECS = 2.0
DELTA_T     = 1.0 / SAMPLE_RATE
F_LOWER     = 30.0
TARGET_LEN  = int(SIGNAL_SECS * SAMPLE_RATE)   # 8192
DELTA_F     = 1.0 / SIGNAL_SECS                # 0.5 Hz
FLEN        = TARGET_LEN // 2 + 1              # 4097


# ---------------------------------------------------------------------------
# Session fixtures — one dataset per generator with o4_psd noise
# Uses only H1 to match the single-detector psd_cache_dir fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def gr_o4(small_config, aligo_kwargs, psd_cache_dir):
    return pycbc_data_generator(
        config=small_config,
        detectors=["H1"],
        noise_backend="o4_psd",
        psd_cache_dir=psd_cache_dir,
        **aligo_kwargs,
    )

@pytest.fixture(scope="session")
def mg_o4(small_config, aligo_kwargs, psd_cache_dir):
    return pycbc_massive_gravity_data_generator(
        config=small_config,
        lambda_g=LAMBDA_G,
        detectors=["H1"],
        noise_backend="o4_psd",
        psd_cache_dir=psd_cache_dir,
        **aligo_kwargs,
    )

@pytest.fixture(scope="session")
def lv_o4(small_config, aligo_kwargs, psd_cache_dir):
    return pycbc_lorentz_violation_data_generator(
        config=small_config,
        alpha_lv=ALPHA_LV,
        A_lv=A_LV,
        lambda_g=LAMBDA_G,
        detectors=["H1"],
        noise_backend="o4_psd",
        psd_cache_dir=psd_cache_dir,
        **aligo_kwargs,
    )

@pytest.fixture(scope="session")
def saved_path_gr(gr_o4, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("saveload") / "gr_o4.pt")
    save_dataloaders(gr_o4, path)
    return path

@pytest.fixture(scope="session")
def saved_path_mg(mg_o4, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("saveload") / "mg_o4.pt")
    save_dataloaders(mg_o4, path)
    return path

@pytest.fixture(scope="session")
def saved_path_lv(lv_o4, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("saveload") / "lv_o4.pt")
    save_dataloaders(lv_o4, path)
    return path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _first_batch(loader):
    return next(iter(loader))


# ---------------------------------------------------------------------------
# TestCache — verifies the downloaded .npz file
# ---------------------------------------------------------------------------
class TestCache:
    def test_file_exists(self, psd_cache_dir):
        assert os.path.exists(_cache_path(DETECTOR, SAMPLE_RATE, psd_cache_dir))

    def test_keys_and_shape(self, psd_cache_dir):
        data = np.load(_cache_path(DETECTOR, SAMPLE_RATE, psd_cache_dir))
        assert "freqs" in data and "psds" in data
        assert data["psds"].shape[1] == data["freqs"].shape[0]

    def test_freqs_positive_sorted(self, psd_cache_dir):
        freqs = np.load(_cache_path(DETECTOR, SAMPLE_RATE, psd_cache_dir))["freqs"]
        assert np.all(freqs > 0) and np.all(np.diff(freqs) > 0)

    def test_psds_positive(self, psd_cache_dir):
        assert np.all(
            np.load(_cache_path(DETECTOR, SAMPLE_RATE, psd_cache_dir))["psds"] > 0
        )

    def test_psds_distinct(self, psd_cache_dir):
        # np.array_equal (exact) not np.allclose — PSD values ~1e-47 are all within
        # allclose's default atol=1e-8, which would falsely report them as equal
        psds = np.load(_cache_path(DETECTOR, SAMPLE_RATE, psd_cache_dir))["psds"]
        if psds.shape[0] < 2:
            pytest.skip("Only 1 PSD saved")
        assert any(
            not np.array_equal(psds[i], psds[j])
            for i in range(psds.shape[0])
            for j in range(i + 1, psds.shape[0])
        )


# ---------------------------------------------------------------------------
# TestPsdLoading — verifies interpolation onto the generation frequency grid
# ---------------------------------------------------------------------------
class TestPsdLoading:
    def test_returns_frequency_series(self, psd_cache_dir):
        from pycbc.types import FrequencySeries
        psd = load_random_o4_psd(FLEN, DELTA_F, F_LOWER, DETECTOR,
                                  sample_rate=SAMPLE_RATE, cache_dir=psd_cache_dir)
        assert isinstance(psd, FrequencySeries)

    def test_length_and_delta_f(self, psd_cache_dir):
        psd = load_random_o4_psd(FLEN, DELTA_F, F_LOWER, DETECTOR,
                                  sample_rate=SAMPLE_RATE, cache_dir=psd_cache_dir)
        assert len(psd) == FLEN
        assert abs(psd.delta_f - DELTA_F) < 1e-9

    def test_positive_and_finite(self, psd_cache_dir):
        psd = load_random_o4_psd(FLEN, DELTA_F, F_LOWER, DETECTOR,
                                  sample_rate=SAMPLE_RATE, cache_dir=psd_cache_dir)
        arr = psd.numpy()
        assert np.all(arr > 0) and np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# TestNoise — verifies noise generation from cached PSDs
# ---------------------------------------------------------------------------
class TestNoise:
    def test_shape_finite_nonzero(self, psd_cache_dir):
        psd   = load_random_o4_psd(FLEN, DELTA_F, F_LOWER, DETECTOR,
                                    sample_rate=SAMPLE_RATE, cache_dir=psd_cache_dir)
        noise = noise_from_psd(TARGET_LEN, DELTA_T, psd)
        arr   = noise.numpy()
        assert len(noise) == TARGET_LEN
        assert np.all(np.isfinite(arr))
        assert np.any(arr != 0)

    def test_different_seeds_differ(self, psd_cache_dir):
        # np.array_equal (exact) not np.allclose — strain ~1e-22 is within
        # allclose's default atol=1e-8, so allclose would falsely say equal
        psd    = load_random_o4_psd(FLEN, DELTA_F, F_LOWER, DETECTOR,
                                     sample_rate=SAMPLE_RATE, cache_dir=psd_cache_dir)
        noise1 = noise_from_psd(TARGET_LEN, DELTA_T, psd, seed=0).numpy()
        noise2 = noise_from_psd(TARGET_LEN, DELTA_T, psd, seed=1).numpy()
        assert not np.array_equal(noise1, noise2)


# ---------------------------------------------------------------------------
# TestSaveLoad — round-trip for all three o4_psd generators
# ---------------------------------------------------------------------------
class TestSaveLoad:
    def test_gr_file_exists(self, saved_path_gr):
        assert os.path.isfile(saved_path_gr)

    def test_gr_keys_survive(self, saved_path_gr):
        loaded = load_dataloaders(saved_path_gr)
        assert set(loaded.keys()) == {"train_loader", "val_loader", "test_loader", "metadata"}

    def test_gr_metadata_survives(self, saved_path_gr, gr_o4):
        loaded = load_dataloaders(saved_path_gr)
        for key in ("num_samples", "waveform_shape", "channels",
                    "train_size", "val_size", "test_size", "time_resolution"):
            assert loaded["metadata"][key] == gr_o4["metadata"][key]

    def test_gr_tensor_exact_roundtrip(self, saved_path_gr, gr_o4):
        loaded = load_dataloaders(saved_path_gr)
        orig     = gr_o4["train_loader"].dataset.dataset.tensors[0]
        reloaded = loaded["train_loader"].dataset.dataset.tensors[0]
        assert torch.equal(orig, reloaded)

    def test_mg_file_exists(self, saved_path_mg):
        assert os.path.isfile(saved_path_mg)

    def test_mg_keys_survive(self, saved_path_mg):
        loaded = load_dataloaders(saved_path_mg)
        assert set(loaded.keys()) == {"train_loader", "val_loader", "test_loader", "metadata"}

    def test_mg_metadata_survives(self, saved_path_mg, mg_o4):
        loaded = load_dataloaders(saved_path_mg)
        for key in ("num_samples", "waveform_shape", "channels",
                    "train_size", "val_size", "test_size", "time_resolution"):
            assert loaded["metadata"][key] == mg_o4["metadata"][key]

    def test_mg_tensor_exact_roundtrip(self, saved_path_mg, mg_o4):
        loaded = load_dataloaders(saved_path_mg)
        orig     = mg_o4["train_loader"].dataset.dataset.tensors[0]
        reloaded = loaded["train_loader"].dataset.dataset.tensors[0]
        assert torch.equal(orig, reloaded)

    def test_lv_file_exists(self, saved_path_lv):
        assert os.path.isfile(saved_path_lv)

    def test_lv_keys_survive(self, saved_path_lv):
        loaded = load_dataloaders(saved_path_lv)
        assert set(loaded.keys()) == {"train_loader", "val_loader", "test_loader", "metadata"}

    def test_lv_metadata_survives(self, saved_path_lv, lv_o4):
        loaded = load_dataloaders(saved_path_lv)
        for key in ("num_samples", "waveform_shape", "channels",
                    "train_size", "val_size", "test_size", "time_resolution"):
            assert loaded["metadata"][key] == lv_o4["metadata"][key]

    def test_lv_tensor_exact_roundtrip(self, saved_path_lv, lv_o4):
        loaded = load_dataloaders(saved_path_lv)
        orig     = lv_o4["train_loader"].dataset.dataset.tensors[0]
        reloaded = loaded["train_loader"].dataset.dataset.tensors[0]
        assert torch.equal(orig, reloaded)


# ---------------------------------------------------------------------------
# TestGeneratorEndToEnd — all three generators with o4_psd backend
# ---------------------------------------------------------------------------
class TestGeneratorEndToEnd:
    def test_gr_o4_finite_and_nonzero(self, gr_o4):
        for X, _ in gr_o4["train_loader"]:
            assert torch.isfinite(X).all()
        X, _ = _first_batch(gr_o4["train_loader"])
        assert X.abs().sum().item() > 0

    def test_mg_o4_finite_and_nonzero(self, mg_o4):
        for X, _ in mg_o4["train_loader"]:
            assert torch.isfinite(X).all()
        X, _ = _first_batch(mg_o4["train_loader"])
        assert X.abs().sum().item() > 0

    def test_lv_o4_finite_and_nonzero(self, lv_o4):
        for X, _ in lv_o4["train_loader"]:
            assert torch.isfinite(X).all()
        X, _ = _first_batch(lv_o4["train_loader"])
        assert X.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# TestPlots — diagnostic plots, no content assertions
# ---------------------------------------------------------------------------
class TestPlots:
    def test_plot_o4_waveforms(self, gr_o4, mg_o4, lv_o4):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        datasets = [
            (gr_o4, "GR — O4 noise",              "steelblue"),
            (mg_o4, f"MG (λ_g={LAMBDA_G:.0e} m)", "tomato"),
            (lv_o4, f"LV (α={ALPHA_LV})",         "mediumseagreen"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("O4 real PSD noise — generator comparison (H1 strain, first sample)")

        for ax, (result, title, color) in zip(axes, datasets):
            meta = result["metadata"]
            t    = np.arange(meta["waveform_shape"][1]) * meta["time_resolution"]
            X, _ = _first_batch(result["train_loader"])
            ax.plot(t, X[0, 0].numpy(), lw=0.5, color=color)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Strain (H1)")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        fig.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOTS_DIR, "o4_real_waveforms.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
