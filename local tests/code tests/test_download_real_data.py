"""
test_download_real_data.py — offline smoke test for the real-data pipeline.

Runs the processing half of ``download_real_data`` against the HDF5 files
already cached under ``HPC/Pipeline/Real Data/hdf5/``. No network access.
Confirms:

  * the combined .pt is load_dataloaders-compatible (gw_datagen can consume it
    and iterate the DataLoaders);
  * every per-sample tensor is finite, shaped as expected, and whitened tensors
    actually differ from the raw ones;
  * the raw strain plot is written for a cached event.

Skipped if no cached HDF5 pairs (H1+L1) are present.

Run with:
    pytest test_download_real_data.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import torch
from gwpy.timeseries import TimeSeries

# Make the Real Data script importable.
_REAL_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "HPC" / "Pipeline" / "Real Data"
)
sys.path.insert(0, str(_REAL_DATA_DIR))

from download_real_data import (  # noqa: E402
    COMBINED_NAME,
    DURATION,
    N_2S,
    SAMPLE_RATE,
    build_combined,
    layout,
    plot_raw_strain,
    process_to_pt,
)
from gw_datagen import load_dataloaders  # noqa: E402

CACHED_HDF5_DIR = _REAL_DATA_DIR / "hdf5"
N_EVENTS = 2       # keep the test short
BATCH_SIZE = 2


def _discover_cached_events(hdf5_dir: Path) -> dict[str, float]:
    """Scan ``hdf5_dir`` for <EVENT>_H1/L1_strain.hdf5 pairs. Return
    {event: gps_merger} using the HDF5 t0 attribute (gps = t0 + DURATION/2)."""
    events: dict[str, float] = {}
    for h1 in hdf5_dir.glob("*_H1_strain.hdf5"):
        event = h1.name[: -len("_H1_strain.hdf5")]
        l1 = hdf5_dir / f"{event}_L1_strain.hdf5"
        if not l1.exists():
            continue
        ts = TimeSeries.read(str(h1))
        events[event] = float(ts.t0.value) + DURATION / 2
    return events


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """Process cached HDF5 files end-to-end — zero network."""
    if not CACHED_HDF5_DIR.is_dir():
        pytest.skip(f"No cached hdf5 directory at {CACHED_HDF5_DIR}")

    available = _discover_cached_events(CACHED_HDF5_DIR)
    if not available:
        pytest.skip(f"No cached H1+L1 pairs under {CACHED_HDF5_DIR}")

    chosen = list(available.items())[:N_EVENTS]

    out_dir = str(tmp_path_factory.mktemp("real_data_cached"))
    dirs = layout(out_dir)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Symlink cached HDF5 into the temp layout so downstream code sees the
    # canonical hdf5/ folder.
    for event, _ in chosen:
        for det in ("H1", "L1"):
            src = CACHED_HDF5_DIR / f"{event}_{det}_strain.hdf5"
            dst = Path(dirs["hdf5"]) / f"{event}_{det}_strain.hdf5"
            if not dst.exists():
                os.symlink(src, dst)

    per_event = []
    for event, gps in chosen:
        hdf5_paths = {
            det: os.path.join(dirs["hdf5"], f"{event}_{det}_strain.hdf5")
            for det in ("H1", "L1")
        }
        per_event.append((event, process_to_pt(event, gps, hdf5_paths)))

    combined = build_combined(per_event, batch_size=BATCH_SIZE)
    combined_path = os.path.join(dirs["pt"], COMBINED_NAME)
    torch.save(combined, combined_path)

    return {
        "out_dir":       out_dir,
        "dirs":          dirs,
        "combined_path": combined_path,
        "events":        [e for e, _ in chosen],
        "gps_lookup":    dict(chosen),
    }


class TestLayout:
    def test_subfolders_exist(self, ran):
        for key in ("hdf5", "pt"):
            assert os.path.isdir(ran["dirs"][key])

    def test_hdf5_files_exist(self, ran):
        expected = DURATION * SAMPLE_RATE
        for event in ran["events"]:
            for det in ("H1", "L1"):
                path = os.path.join(ran["dirs"]["hdf5"], f"{event}_{det}_strain.hdf5")
                assert os.path.isfile(path)
                assert len(TimeSeries.read(path)) == expected

    def test_combined_pt_exists(self, ran):
        path = os.path.join(ran["dirs"]["pt"], COMBINED_NAME)
        assert path == ran["combined_path"]
        assert os.path.isfile(path)


class TestCombinedSchema:
    def test_top_level_keys(self, ran):
        blob = torch.load(ran["combined_path"], weights_only=False)
        assert set(blob.keys()) >= {
            "X", "X_whitened", "y",
            "train_indices", "val_indices", "test_indices",
            "metadata",
        }

    def test_shapes(self, ran):
        blob = torch.load(ran["combined_path"], weights_only=False)
        n = len(ran["events"])
        assert blob["X"].shape          == (n, 2, N_2S)
        assert blob["X_whitened"].shape == (n, 2, N_2S)
        assert blob["y"].shape          == (n, 13)

    def test_whitened_differs_from_raw(self, ran):
        blob = torch.load(ran["combined_path"], weights_only=False)
        assert not torch.equal(blob["X_whitened"], blob["X"])
        # Raw strain is O(1e-21); whitened is O(1). A ratio collapse would mean
        # the whitening step silently did nothing.
        assert blob["X_whitened"].std().item() > blob["X"].std().item() * 1e10

    def test_finite(self, ran):
        blob = torch.load(ran["combined_path"], weights_only=False)
        for key in ("X", "X_whitened", "y"):
            assert torch.isfinite(blob[key]).all(), f"{key} has non-finite entries"

    def test_default_splits_put_everything_in_test(self, ran):
        blob = torch.load(ran["combined_path"], weights_only=False)
        n = blob["X"].shape[0]
        assert blob["train_indices"] == []
        assert blob["val_indices"]   == []
        assert list(blob["test_indices"]) == list(range(n))

    def test_metadata_has_dataloader_keys(self, ran):
        meta = torch.load(ran["combined_path"], weights_only=False)["metadata"]
        for key in ("num_samples", "waveform_shape", "channels",
                    "train_size", "val_size", "test_size",
                    "batch_size", "time_resolution"):
            assert key in meta


class TestLoadDataloaders:
    """The combined .pt should be consumable by gw_datagen.load_dataloaders."""

    def test_load_returns_dataloaders(self, ran):
        loaded = load_dataloaders(ran["combined_path"])
        assert set(loaded.keys()) == {"train_loader", "val_loader", "test_loader", "metadata"}

    def test_test_loader_yields_expected_batches(self, ran):
        loaded = load_dataloaders(ran["combined_path"])
        batches = list(loaded["test_loader"])
        assert len(batches) >= 1
        X, y = batches[0]
        assert X.dim() == 3 and X.shape[1:] == (2, N_2S)
        assert y.shape[1] == 13

    def test_train_and_val_loaders_are_empty(self, ran):
        loaded = load_dataloaders(ran["combined_path"])
        assert len(list(loaded["train_loader"])) == 0
        assert len(list(loaded["val_loader"]))   == 0


class TestPlot:
    def test_plot_raw_strain_runs(self, ran, tmp_path):
        hdf5_dir = ran["dirs"]["hdf5"]
        event    = ran["events"][0]
        gps      = ran["gps_lookup"][event]
        paths    = {det: os.path.join(hdf5_dir, f"{event}_{det}_strain.hdf5")
                    for det in ("H1", "L1")}
        out = tmp_path / f"raw_{event}.png"
        plot_raw_strain(event, gps, paths, str(out))
        assert out.exists() and out.stat().st_size > 0
