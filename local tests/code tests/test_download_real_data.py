"""
test_download_real_data.py — short smoke test for the real-data downloader.

Downloads raw 32 s HDF5 strain for a small number of O4 events (where both H1
and L1 are online), trims to 2 s around merger, whitens, saves .pt files, and
plots both the raw 32 s strain and the processed (raw 2 s vs whitened 2 s)
waveforms.

Requires network access to GWOSC; the module is skipped if GWOSC is unreachable.

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
    DURATION,
    N_2S,
    SAMPLE_RATE,
    both_detectors_available,
    download_strain_hdf5,
    list_o4_events,
    plot_processed,
    plot_raw_strain,
    process_to_pt,
)
from conftest import PLOTS_DIR  # noqa: E402

N_EVENTS = 2  # keep the test short


@pytest.fixture(scope="module")
def downloaded(tmp_path_factory):
    """Download HDF5, trim to 2 s, whiten, and save .pt for a few O4 events."""
    out_dir = tmp_path_factory.mktemp("real_data_download")

    try:
        events = list_o4_events()
    except Exception as exc:
        pytest.skip(f"Could not reach GWOSC to list events: {exc}")

    collected = []
    for event, gps in events:
        if len(collected) >= N_EVENTS:
            break
        try:
            if not both_detectors_available(gps):
                continue
            paths = download_strain_hdf5(event, gps, str(out_dir))
            data  = process_to_pt(event, gps, paths)
        except Exception:
            continue
        pt_path = os.path.join(str(out_dir), f"{event}_2s_real.pt")
        torch.save(data, pt_path)
        collected.append((event, gps, paths, data, pt_path))

    if not collected:
        pytest.skip("No O4 events could be downloaded (network or GWOSC issue)")
    return collected


class TestDownload:
    def test_hdf5_files_exist(self, downloaded):
        for _, _, paths, _, _ in downloaded:
            assert set(paths.keys()) == {"H1", "L1"}
            for path in paths.values():
                assert os.path.isfile(path)

    def test_hdf5_length_matches_duration(self, downloaded):
        expected = DURATION * SAMPLE_RATE
        for _, _, paths, _, _ in downloaded:
            for path in paths.values():
                ts = TimeSeries.read(path)
                assert len(ts) == expected

    def test_hdf5_start_covers_merger(self, downloaded):
        for _, gps, paths, _, _ in downloaded:
            for path in paths.values():
                ts = TimeSeries.read(path)
                t0 = float(ts.t0.value)
                t1 = t0 + len(ts) / SAMPLE_RATE
                assert t0 <= gps <= t1


class TestProcess:
    def test_shapes(self, downloaded):
        for _, _, _, data, _ in downloaded:
            assert data["X"].shape     == (1, 2, N_2S)
            assert data["X_raw"].shape == (1, 2, N_2S)
            assert data["y"].shape     == (1, 13)

    def test_finite_and_nonzero(self, downloaded):
        for _, _, _, data, _ in downloaded:
            for key in ("X", "X_raw"):
                X = data[key]
                assert torch.isfinite(X).all(), f"{key} has non-finite entries"
                assert X.abs().sum().item() > 0

    def test_whitened_differs_from_raw(self, downloaded):
        # Whitening should materially change amplitude distribution; a whitened
        # strain is O(1) while a raw interferometer strain is O(1e-21).
        for _, _, _, data, _ in downloaded:
            assert not torch.equal(data["X"], data["X_raw"])
            raw_std = data["X_raw"].std().item()
            whi_std = data["X"].std().item()
            assert whi_std > raw_std * 1e10

    def test_pt_roundtrip(self, downloaded):
        for _, _, _, data, pt_path in downloaded:
            loaded = torch.load(pt_path, weights_only=False)
            assert set(loaded.keys()) >= {"X", "X_raw", "y", "metadata"}
            assert torch.equal(loaded["X"],     data["X"])
            assert torch.equal(loaded["X_raw"], data["X_raw"])


class TestPlot:
    def test_plot_raw_strain(self, downloaded):
        os.makedirs(PLOTS_DIR, exist_ok=True)
        for event, gps, paths, _, _ in downloaded:
            out = os.path.join(PLOTS_DIR, f"raw_{event}.png")
            plot_raw_strain(event, gps, paths, out)
            assert os.path.isfile(out) and os.path.getsize(out) > 0

    def test_plot_processed(self, downloaded):
        os.makedirs(PLOTS_DIR, exist_ok=True)
        for event, _, _, data, _ in downloaded:
            out = os.path.join(PLOTS_DIR, f"processed_{event}.png")
            plot_processed(event, data, out)
            assert os.path.isfile(out) and os.path.getsize(out) > 0
