"""
test_o4psd.py — Batch 2: real PSD CSV backend.

Tests the two real-PSD generators:
  - pycbc_data_generator_real_psd            (GR with real PSD)
  - pycbc_modified_data_generator_real_psd   (MG with real PSD)

The psd_csv_path fixture (conftest.py) supplies a synthetic aLIGO PSD
stored in the required CSV format (detector, event_name, frequency, psd).

Note: pycbc_lorentz_violation_data_generator has no real-PSD variant,
so LV is tested only in test_aligo.py.

Save/load round-trip tests for the GR real-PSD result are included.
Plot tests produce visual diagnostics in tests/plots/.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch

pytestmark = pytest.mark.o4psd

from gw_datagen import (
    pycbc_data_generator_real_psd,
    pycbc_modified_data_generator_real_psd,
    save_dataloaders,
    load_dataloaders,
)
from conftest import PLOTS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SAMPLES = 16
LAMBDA_G = 1e22


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _first_batch(loader):
    return next(iter(loader))


def _all_finite(loader):
    for X, _ in loader:
        if not torch.isfinite(X).all():
            return False
    return True


# ---------------------------------------------------------------------------
# Session-scoped generation fixtures — each regime is generated exactly once
# and shared across all test classes in this file.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def gr_result_o4psd(small_config, base_kwargs, psd_csv_path):
    # base_kwargs deliberately omits f_final (not accepted by this generator)
    return pycbc_data_generator_real_psd(
        config=small_config, psd_csv=psd_csv_path, **base_kwargs
    )


@pytest.fixture(scope="session")
def mg_result_o4psd(small_config, base_kwargs, psd_csv_path):
    # pycbc_modified_data_generator_real_psd does accept f_final
    return pycbc_modified_data_generator_real_psd(
        config=small_config,
        psd_csv=psd_csv_path,
        lambda_g=LAMBDA_G,
        f_final=2048.0,
        **base_kwargs,
    )


@pytest.fixture(scope="session")
def o4psd_saved_path(gr_result_o4psd, tmp_path_factory):
    """Save the GR real-PSD dataset once; all save/load tests share the file."""
    path = tmp_path_factory.mktemp("o4psd_saveload") / "gr_real_psd.pt"
    save_dataloaders(gr_result_o4psd, str(path))
    return str(path)


# ---------------------------------------------------------------------------
# CSV fixture validation — run before anything touches the generators
# ---------------------------------------------------------------------------
class TestPsdCsvFixture:
    def test_has_required_columns(self, psd_csv_path):
        df = pd.read_csv(psd_csv_path)
        required = {"detector", "event_name", "frequency", "psd"}
        assert required.issubset(set(df.columns))

    def test_has_both_detectors(self, psd_csv_path):
        df = pd.read_csv(psd_csv_path)
        assert "H1" in df["detector"].values
        assert "L1" in df["detector"].values

    def test_no_nan_in_psd_column(self, psd_csv_path):
        df = pd.read_csv(psd_csv_path)
        assert not df["psd"].isna().any()

    def test_psd_values_positive(self, psd_csv_path):
        df = pd.read_csv(psd_csv_path)
        assert (df["psd"] > 0).all()

    def test_frequency_monotonic_per_detector(self, psd_csv_path):
        df = pd.read_csv(psd_csv_path)
        for (det, evt), grp in df.groupby(["detector", "event_name"]):
            freqs = grp["frequency"].values
            assert np.all(np.diff(freqs) >= 0), (
                f"Frequencies not monotonically non-decreasing for {det}/{evt}"
            )

    def test_invalid_csv_raises(self, tmp_path, base_kwargs):
        """A CSV missing the 'psd' column should raise ValueError."""
        bad_csv = tmp_path / "bad_psd.csv"
        pd.DataFrame(
            {"detector": ["H1"], "event_name": ["x"], "frequency": [40.0]}
        ).to_csv(str(bad_csv), index=False)
        with pytest.raises((ValueError, KeyError)):
            pycbc_data_generator_real_psd(
                config={
                    "mass1": lambda size: np.full(size, 30.0),
                    "mass2": lambda size: np.full(size, 30.0),
                    "redshift": lambda size: np.full(size, 0.1),
                },
                psd_csv=str(bad_csv),
                **base_kwargs,
            )


# ---------------------------------------------------------------------------
# GR real-PSD tests
# ---------------------------------------------------------------------------
class TestRealPsdGR:
    def test_result_structure(self, gr_result_o4psd):
        assert set(gr_result_o4psd.keys()) == {
            "train_loader", "val_loader", "test_loader", "metadata"
        }

    def test_metadata_channels_and_splits(self, gr_result_o4psd):
        meta = gr_result_o4psd["metadata"]
        assert meta["channels"] == ["H1", "L1"]
        assert meta["train_size"] + meta["val_size"] + meta["test_size"] == NUM_SAMPLES
        assert meta["train_size"] > 0
        assert meta["val_size"] > 0
        assert meta["test_size"] > 0

    def test_batch_shape(self, gr_result_o4psd):
        meta = gr_result_o4psd["metadata"]
        waveform_len = meta["waveform_shape"][1]
        X, y = _first_batch(gr_result_o4psd["train_loader"])
        assert X.shape[1:] == (2, waveform_len)
        assert y.ndim == 2

    def test_no_nan(self, gr_result_o4psd):
        assert _all_finite(gr_result_o4psd["train_loader"])

    def test_strain_nonzero(self, gr_result_o4psd):
        X, _ = _first_batch(gr_result_o4psd["train_loader"])
        assert X.abs().sum().item() > 0.0

    def test_no_f_final_parameter(self, small_config, base_kwargs, psd_csv_path):
        """pycbc_data_generator_real_psd must not accept f_final."""
        with pytest.raises(TypeError):
            pycbc_data_generator_real_psd(
                config=small_config,
                psd_csv=psd_csv_path,
                f_final=2048.0,
                **base_kwargs,
            )


# ---------------------------------------------------------------------------
# MG real-PSD tests
# ---------------------------------------------------------------------------
class TestRealPsdMG:
    def test_lambda_g_in_metadata(self, mg_result_o4psd):
        assert mg_result_o4psd["metadata"]["lambda_g"] == pytest.approx(LAMBDA_G)

    def test_batch_shape(self, mg_result_o4psd):
        meta = mg_result_o4psd["metadata"]
        waveform_len = meta["waveform_shape"][1]
        X, _ = _first_batch(mg_result_o4psd["train_loader"])
        assert X.shape[1:] == (2, waveform_len)

    def test_no_nan(self, mg_result_o4psd):
        assert _all_finite(mg_result_o4psd["train_loader"])

    def test_strain_nonzero(self, mg_result_o4psd):
        X, _ = _first_batch(mg_result_o4psd["train_loader"])
        assert X.abs().sum().item() > 0.0

    def test_splits_sum_to_num_samples(self, mg_result_o4psd):
        meta = mg_result_o4psd["metadata"]
        assert meta["train_size"] + meta["val_size"] + meta["test_size"] == NUM_SAMPLES


# ---------------------------------------------------------------------------
# Save / Load round-trip (reuses the GR session fixture)
# ---------------------------------------------------------------------------
class TestRealPsdSaveLoad:
    def test_file_exists(self, o4psd_saved_path):
        assert os.path.isfile(o4psd_saved_path)

    def test_load_returns_expected_keys(self, o4psd_saved_path):
        loaded = load_dataloaders(o4psd_saved_path)
        assert set(loaded.keys()) == {
            "train_loader", "val_loader", "test_loader", "metadata"
        }

    def test_metadata_roundtrip(self, o4psd_saved_path, gr_result_o4psd):
        loaded = load_dataloaders(o4psd_saved_path)
        for key in ("num_samples", "waveform_shape", "channels",
                    "train_size", "val_size", "test_size", "time_resolution"):
            assert loaded["metadata"][key] == gr_result_o4psd["metadata"][key], (
                f"metadata['{key}'] changed after save/load"
            )

    def test_loaded_batch_shape(self, o4psd_saved_path, gr_result_o4psd):
        loaded = load_dataloaders(o4psd_saved_path)
        waveform_len = gr_result_o4psd["metadata"]["waveform_shape"][1]
        X, _ = _first_batch(loaded["train_loader"])
        assert X.shape[1:] == (2, waveform_len)

    def test_x_values_match(self, o4psd_saved_path, gr_result_o4psd):
        loaded = load_dataloaders(o4psd_saved_path)
        orig_full   = gr_result_o4psd["train_loader"].dataset.dataset.tensors[0]
        loaded_full = loaded["train_loader"].dataset.dataset.tensors[0]
        assert torch.allclose(orig_full, loaded_full)


# ---------------------------------------------------------------------------
# Plots — visual inspection only, no assertions on content
# ---------------------------------------------------------------------------
class TestPlotsO4psd:
    """Diagnostic plots for the real-PSD test batch."""

    def _time_axis(self, meta):
        return np.arange(meta["waveform_shape"][1]) * meta["time_resolution"]

    def test_plot_gr_waveform(self, gr_result_o4psd):
        """Two-panel H1 + L1 for the first GR real-PSD sample."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, _ = _first_batch(gr_result_o4psd["train_loader"])
        t = self._time_axis(gr_result_o4psd["metadata"])

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, X[0, 0].numpy(), lw=0.6, color="steelblue")
        axes[0].set_ylabel("Strain (H1)")
        axes[0].set_title("GR waveform sample — real PSD (synthetic aLIGO CSV)")
        axes[1].plot(t, X[0, 1].numpy(), lw=0.6, color="darkorange")
        axes[1].set_ylabel("Strain (L1)")
        axes[1].set_xlabel("Time (s)")
        fig.tight_layout()

        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOTS_DIR, "o4psd_gr_waveform.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    def test_plot_mg_waveform(self, mg_result_o4psd):
        """Two-panel H1 + L1 for the first MG real-PSD sample."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, _ = _first_batch(mg_result_o4psd["train_loader"])
        t = self._time_axis(mg_result_o4psd["metadata"])

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, X[0, 0].numpy(), lw=0.6, color="mediumseagreen")
        axes[0].set_ylabel("Strain (H1)")
        axes[0].set_title(
            f"Massive Gravity waveform (λ_g={LAMBDA_G:.0e} m) — real PSD CSV"
        )
        axes[1].plot(t, X[0, 1].numpy(), lw=0.6, color="darkorchid")
        axes[1].set_ylabel("Strain (L1)")
        axes[1].set_xlabel("Time (s)")
        fig.tight_layout()

        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOTS_DIR, "o4psd_mg_waveform.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    def test_plot_fixture_psd(self, psd_csv_path):
        """Log-log plot of the H1 PSD from the CSV fixture."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.read_csv(psd_csv_path)
        h1 = df[df["detector"] == "H1"].copy()
        # Use the first event only (they are identical in the fixture)
        h1 = h1[h1["event_name"] == h1["event_name"].iloc[0]]
        freqs = h1["frequency"].values
        psd = h1["psd"].values

        # Only plot the sensitive band (>= 10 Hz)
        mask = freqs >= 10.0
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(freqs[mask], psd[mask], lw=1.2, color="navy")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"PSD (strain$^2$/Hz)")
        ax.set_title("Synthetic aLIGO PSD used in o4psd tests (H1)")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()

        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOTS_DIR, "o4psd_fixture_psd.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
