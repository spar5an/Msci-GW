"""
test_analytic.py — Batch 1: all generators using synthetic/analytic PSDs.

Covers:
  - pycbc_data_generator            (GR,  aLIGO noise)
  - pycbc_massive_gravity_data_generator  (MG,  aLIGO noise)
  - pycbc_lorentz_violation_data_generator (LV, aLIGO noise)
  - save / load round-trip
  - diagnostic plots → plots/

No internet access required.

Run with:
    pytest test_analytic.py -v
"""

import os
import numpy as np
import pytest
import torch

from gw_datagen import (
    pycbc_data_generator,
    pycbc_massive_gravity_data_generator,
    pycbc_lorentz_violation_data_generator,
    save_dataloaders,
    load_dataloaders,
)
from conftest import PLOTS_DIR

NUM_SAMPLES = 16
LAMBDA_G    = 1e22
ALPHA_LV    = 3.0
A_LV        = 1e15


# ---------------------------------------------------------------------------
# Session fixtures — each dataset generated exactly once
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def gr(small_config, aligo_kwargs):
    return pycbc_data_generator(config=small_config, **aligo_kwargs)

@pytest.fixture(scope="session")
def mg(small_config, aligo_kwargs):
    return pycbc_massive_gravity_data_generator(
        config=small_config, lambda_g=LAMBDA_G, **aligo_kwargs)

@pytest.fixture(scope="session")
def lv(small_config, aligo_kwargs):
    return pycbc_lorentz_violation_data_generator(
        config=small_config, alpha_lv=ALPHA_LV, A_lv=A_LV, lambda_g=LAMBDA_G,
        **aligo_kwargs)

@pytest.fixture(scope="session")
def saved_path(gr, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("saveload") / "gr.pt")
    save_dataloaders(gr, path)
    return path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _first_batch(loader):
    return next(iter(loader))

def _all_finite(loader):
    return all(torch.isfinite(X).all() for X, _ in loader)

def _time_axis(meta):
    return np.arange(meta["waveform_shape"][1]) * meta["time_resolution"]


# ---------------------------------------------------------------------------
# TestGenerators — structure, splits, shape, NaN, and physics metadata
# ---------------------------------------------------------------------------
class TestGenerators:

    # GR covers the shared output contract for all generators
    def test_gr_keys(self, gr):
        assert set(gr.keys()) == {"train_loader", "val_loader", "test_loader", "metadata"}

    def test_gr_splits(self, gr):
        m = gr["metadata"]
        assert m["train_size"] + m["val_size"] + m["test_size"] == NUM_SAMPLES
        assert all(m[k] > 0 for k in ("train_size", "val_size", "test_size"))

    def test_gr_batch_shape(self, gr):
        m = gr["metadata"]
        X, y = _first_batch(gr["train_loader"])
        assert X.shape == (8, 2, m["waveform_shape"][1])  # batch × detectors × samples
        assert y.shape[1] == len(m["parameter_names"])

    def test_gr_finite_and_nonzero(self, gr):
        assert _all_finite(gr["train_loader"])
        X, _ = _first_batch(gr["train_loader"])
        assert X.abs().sum().item() > 0

    # MG — just check the physics-specific parts; shared structure already tested above
    def test_mg_lambda_g(self, mg):
        assert mg["metadata"]["lambda_g"] == pytest.approx(LAMBDA_G)

    def test_mg_finite_and_nonzero(self, mg):
        assert _all_finite(mg["train_loader"])
        X, _ = _first_batch(mg["train_loader"])
        assert X.abs().sum().item() > 0

    # LV — unique physics metadata
    def test_lv_alpha_lv(self, lv):
        assert lv["metadata"]["alpha_lv"] == pytest.approx(ALPHA_LV)

    def test_lv_finite_and_nonzero(self, lv):
        assert _all_finite(lv["train_loader"])
        X, _ = _first_batch(lv["train_loader"])
        assert X.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# TestSaveLoad — round-trip using torch.equal (exact, not allclose)
# Note: torch.allclose(atol=1e-8) gives false positives on strain ~1e-22
# ---------------------------------------------------------------------------
class TestSaveLoad:
    def test_file_exists(self, saved_path):
        assert os.path.isfile(saved_path)

    def test_keys_survive(self, saved_path):
        loaded = load_dataloaders(saved_path)
        assert set(loaded.keys()) == {"train_loader", "val_loader", "test_loader", "metadata"}

    def test_metadata_survives(self, saved_path, gr):
        loaded = load_dataloaders(saved_path)
        for key in ("num_samples", "waveform_shape", "channels",
                    "train_size", "val_size", "test_size", "time_resolution"):
            assert loaded["metadata"][key] == gr["metadata"][key]

    def test_tensor_exact_roundtrip(self, saved_path, gr):
        loaded = load_dataloaders(saved_path)
        orig   = gr["train_loader"].dataset.dataset.tensors[0]
        reloaded = loaded["train_loader"].dataset.dataset.tensors[0]
        assert torch.equal(orig, reloaded)


# ---------------------------------------------------------------------------
# TestPlots — diagnostic plots, no content assertions
# ---------------------------------------------------------------------------
class TestPlots:
    def test_plot_all_generators(self, gr, mg, lv):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Analytic PSD — generator comparison (H1 strain, first sample)")

        datasets = [
            (gr, "GR — aLIGO noise",          "steelblue"),
            (mg, f"MG (λ_g={LAMBDA_G:.0e} m)", "tomato"),
            (lv, f"LV (α={ALPHA_LV})",         "mediumseagreen"),
        ]

        for ax, (result, title, color) in zip(axes.flat, datasets):
            X, _ = _first_batch(result["train_loader"])
            t = _time_axis(result["metadata"])
            ax.plot(t, X[0, 0].numpy(), lw=0.6, color=color)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Strain (H1)")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        fig.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOTS_DIR, "analytic_generators.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
