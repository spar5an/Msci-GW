"""
test_aligo.py — Batch 1: aLIGOZeroDetHighPower PSD backend.

Tests the three standard generators:
  - pycbc_data_generator             (GR)
  - pycbc_massive_gravity_data_generator (MG)
  - pycbc_lorentz_violation_data_generator (LV)

In a fresh test environment the GWOSC O4 cache is absent, so load_random_o4_psd
automatically falls back to aLIGOZeroDetHighPower — these tests therefore
exercise the analytical aLIGO PSD path.

Save/load round-trip tests for the GR result are included at the bottom.
Plot tests produce visual diagnostics in tests/plots/ (no assertions on content).
"""

import os
import pytest
import numpy as np
import torch

pytestmark = pytest.mark.aligo

from gw_datagen import (
    pycbc_data_generator,
    pycbc_massive_gravity_data_generator,
    pycbc_lorentz_violation_data_generator,
    save_dataloaders,
    load_dataloaders,
)
from conftest import PLOTS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SAMPLES = 16
LAMBDA_G = 1e22
ALPHA_LV = 3.0
A_LV = 1e15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _first_batch(loader):
    """Return (X, y) from the first batch of a DataLoader."""
    return next(iter(loader))


def _all_finite(loader):
    """Return True if no NaN or Inf appears in any X batch of the loader."""
    for X, _ in loader:
        if not torch.isfinite(X).all():
            return False
    return True


def _collect_indices(loader):
    """Return the underlying Subset indices of a DataLoader's dataset."""
    return set(loader.dataset.indices.tolist())


# ---------------------------------------------------------------------------
# Session-scoped generation fixtures — each regime is generated exactly once
# and shared across all test classes in this file.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def gr_result_aligo(small_config, aligo_kwargs):
    return pycbc_data_generator(config=small_config, **aligo_kwargs)


@pytest.fixture(scope="session")
def mg_result_aligo(small_config, aligo_kwargs):
    return pycbc_massive_gravity_data_generator(
        config=small_config, lambda_g=LAMBDA_G, **aligo_kwargs
    )


@pytest.fixture(scope="session")
def lv_result_aligo(small_config, aligo_kwargs):
    return pycbc_lorentz_violation_data_generator(
        config=small_config,
        alpha_lv=ALPHA_LV,
        A_lv=A_LV,
        lambda_g=LAMBDA_G,
        **aligo_kwargs,
    )


@pytest.fixture(scope="session")
def aligo_saved_path(gr_result_aligo, tmp_path_factory):
    """Save the GR dataset once; all save/load tests share the same file."""
    path = tmp_path_factory.mktemp("aligo_saveload") / "gr_aligo.pt"
    save_dataloaders(gr_result_aligo, str(path))
    return str(path)


# ---------------------------------------------------------------------------
# GR tests
# ---------------------------------------------------------------------------
class TestGR:
    def test_result_structure(self, gr_result_aligo):
        assert set(gr_result_aligo.keys()) == {
            "train_loader", "val_loader", "test_loader", "metadata"
        }

    def test_metadata_required_keys(self, gr_result_aligo):
        meta = gr_result_aligo["metadata"]
        required = {
            "num_samples", "waveform_shape", "channels",
            "train_size", "val_size", "test_size",
            "parameter_names", "time_resolution", "approximant",
        }
        assert required.issubset(set(meta.keys()))

    def test_metadata_values(self, gr_result_aligo):
        meta = gr_result_aligo["metadata"]
        assert meta["num_samples"] == NUM_SAMPLES
        assert meta["channels"] == ["H1", "L1"]
        assert meta["waveform_shape"][0] == 2  # two detectors
        assert meta["train_size"] + meta["val_size"] + meta["test_size"] == NUM_SAMPLES
        assert meta["train_size"] > 0
        assert meta["val_size"] > 0
        assert meta["test_size"] > 0

    def test_batch_shape(self, gr_result_aligo):
        meta = gr_result_aligo["metadata"]
        waveform_len = meta["waveform_shape"][1]
        n_params = len(meta["parameter_names"])
        X, y = _first_batch(gr_result_aligo["train_loader"])
        assert X.ndim == 3
        assert X.shape[1] == 2           # H1 + L1
        assert X.shape[2] == waveform_len
        assert y.ndim == 2
        assert y.shape[1] == n_params

    def test_no_nan_in_train(self, gr_result_aligo):
        assert _all_finite(gr_result_aligo["train_loader"])

    def test_strain_nonzero(self, gr_result_aligo):
        X, _ = _first_batch(gr_result_aligo["train_loader"])
        assert X.abs().sum().item() > 0.0

    def test_splits_disjoint_and_complete(self, gr_result_aligo):
        train_idx = _collect_indices(gr_result_aligo["train_loader"])
        val_idx   = _collect_indices(gr_result_aligo["val_loader"])
        test_idx  = _collect_indices(gr_result_aligo["test_loader"])
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx  & test_idx) == 0
        assert len(train_idx | val_idx | test_idx) == NUM_SAMPLES

    def test_no_noise_variant(self, small_config, aligo_kwargs):
        """add_noise=False — must generate its own dataset (different parameters)."""
        kwargs = {**aligo_kwargs, "add_noise": False}
        result = pycbc_data_generator(config=small_config, **kwargs)
        assert "train_loader" in result
        X, _ = _first_batch(result["train_loader"])
        assert X.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# MG tests
# ---------------------------------------------------------------------------
class TestMG:
    def test_metadata_waveform_type(self, mg_result_aligo):
        assert mg_result_aligo["metadata"]["waveform_type"] == "massive_gravity"

    def test_metadata_lambda_g(self, mg_result_aligo):
        assert mg_result_aligo["metadata"]["lambda_g"] == pytest.approx(LAMBDA_G)

    def test_batch_shape(self, mg_result_aligo):
        meta = mg_result_aligo["metadata"]
        waveform_len = meta["waveform_shape"][1]
        X, _ = _first_batch(mg_result_aligo["train_loader"])
        assert X.shape[1:] == (2, waveform_len)

    def test_no_nan_in_train(self, mg_result_aligo):
        assert _all_finite(mg_result_aligo["train_loader"])

    def test_strain_nonzero(self, mg_result_aligo):
        X, _ = _first_batch(mg_result_aligo["train_loader"])
        assert X.abs().sum().item() > 0.0

    def test_splits_sum_to_num_samples(self, mg_result_aligo):
        meta = mg_result_aligo["metadata"]
        assert meta["train_size"] + meta["val_size"] + meta["test_size"] == NUM_SAMPLES


# ---------------------------------------------------------------------------
# LV tests
# ---------------------------------------------------------------------------
class TestLV:
    def test_metadata_waveform_type(self, lv_result_aligo):
        assert lv_result_aligo["metadata"]["waveform_type"] == "lorentz_violation"

    def test_metadata_alpha_lv(self, lv_result_aligo):
        assert lv_result_aligo["metadata"]["alpha_lv"] == pytest.approx(ALPHA_LV)

    def test_batch_shape(self, lv_result_aligo):
        meta = lv_result_aligo["metadata"]
        waveform_len = meta["waveform_shape"][1]
        X, _ = _first_batch(lv_result_aligo["train_loader"])
        assert X.shape[1:] == (2, waveform_len)

    def test_no_nan_in_train(self, lv_result_aligo):
        assert _all_finite(lv_result_aligo["train_loader"])

    def test_strain_nonzero(self, lv_result_aligo):
        X, _ = _first_batch(lv_result_aligo["train_loader"])
        assert X.abs().sum().item() > 0.0

    def test_splits_sum_to_num_samples(self, lv_result_aligo):
        meta = lv_result_aligo["metadata"]
        assert meta["train_size"] + meta["val_size"] + meta["test_size"] == NUM_SAMPLES


# ---------------------------------------------------------------------------
# Save / Load round-trip (reuses the GR session fixture)
# ---------------------------------------------------------------------------
class TestSaveLoad:
    def test_file_exists(self, aligo_saved_path):
        assert os.path.isfile(aligo_saved_path)

    def test_load_returns_expected_keys(self, aligo_saved_path):
        loaded = load_dataloaders(aligo_saved_path)
        assert set(loaded.keys()) == {
            "train_loader", "val_loader", "test_loader", "metadata"
        }

    def test_metadata_roundtrip(self, aligo_saved_path, gr_result_aligo):
        loaded = load_dataloaders(aligo_saved_path)
        orig_meta = gr_result_aligo["metadata"]
        load_meta = loaded["metadata"]
        for key in ("num_samples", "waveform_shape", "channels",
                    "train_size", "val_size", "test_size",
                    "time_resolution", "approximant"):
            assert load_meta[key] == orig_meta[key], (
                f"metadata['{key}'] mismatch: {load_meta[key]} != {orig_meta[key]}"
            )

    def test_loaded_batch_shape(self, aligo_saved_path, gr_result_aligo):
        loaded = load_dataloaders(aligo_saved_path)
        waveform_len = gr_result_aligo["metadata"]["waveform_shape"][1]
        X, _ = _first_batch(loaded["train_loader"])
        assert X.shape[1:] == (2, waveform_len)

    def test_x_values_match(self, aligo_saved_path, gr_result_aligo):
        loaded = load_dataloaders(aligo_saved_path)
        orig_full   = gr_result_aligo["train_loader"].dataset.dataset.tensors[0]
        loaded_full = loaded["train_loader"].dataset.dataset.tensors[0]
        assert torch.allclose(orig_full, loaded_full)


# ---------------------------------------------------------------------------
# Plots — visual inspection only, no assertions on content
# ---------------------------------------------------------------------------
class TestPlotsAligo:
    """Generate diagnostic plots for the aLIGO test batch."""

    def _time_axis(self, meta):
        waveform_len = meta["waveform_shape"][1]
        return np.arange(waveform_len) * meta["time_resolution"]

    def test_plot_gr_waveform(self, gr_result_aligo):
        """Two-panel plot: H1 (top) and L1 (bottom) for the first GR sample."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, _ = _first_batch(gr_result_aligo["train_loader"])
        t = self._time_axis(gr_result_aligo["metadata"])

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, X[0, 0].numpy(), lw=0.6, color="steelblue")
        axes[0].set_ylabel("Strain (H1)")
        axes[0].set_title("GR waveform sample — aLIGOZeroDetHighPower noise")
        axes[1].plot(t, X[0, 1].numpy(), lw=0.6, color="darkorange")
        axes[1].set_ylabel("Strain (L1)")
        axes[1].set_xlabel("Time (s)")
        fig.tight_layout()

        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOTS_DIR, "aligo_gr_waveform.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    def test_plot_gr_vs_mg(self, gr_result_aligo, mg_result_aligo):
        """Overlay first H1 sample from GR and MG to show the phase modification."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = self._time_axis(gr_result_aligo["metadata"])
        X_gr, _ = _first_batch(gr_result_aligo["train_loader"])
        X_mg, _ = _first_batch(mg_result_aligo["train_loader"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, X_gr[0, 0].numpy(), lw=0.7, label="GR (IMRPhenomD)",
                color="steelblue")
        ax.plot(t, X_mg[0, 0].numpy(), lw=0.7, label=f"MG (λ_g={LAMBDA_G:.0e} m)",
                color="tomato", alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain (H1)")
        ax.set_title("GR vs Massive Gravity — H1 strain comparison")
        ax.legend()
        fig.tight_layout()

        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOTS_DIR, "aligo_gr_vs_mg.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    def test_plot_lv_waveform(self, lv_result_aligo):
        """First H1 sample from the LV dataset."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, _ = _first_batch(lv_result_aligo["train_loader"])
        t = self._time_axis(lv_result_aligo["metadata"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, X[0, 0].numpy(), lw=0.7, color="mediumseagreen")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain (H1)")
        ax.set_title(
            f"Lorentz-Violation waveform (α={ALPHA_LV}, A_LV={A_LV:.0e} m, "
            f"λ_g={LAMBDA_G:.0e} m)"
        )
        fig.tight_layout()

        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOTS_DIR, "aligo_lv_waveform.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
