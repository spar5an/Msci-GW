"""
conftest.py — Shared fixtures for gw_datagen test suite.

Fixtures
--------
small_config    : minimal BBH parameter config (all lambdas, session-scoped)
base_kwargs     : fast generation kwargs shared by both test files
aligo_kwargs    : base_kwargs + f_final=2048.0 (for standard generators only)
psd_csv_path    : path to a synthetic aLIGO PSD CSV (session-scoped)
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Add the Data Generation directory to sys.path so both test files can do:
#   from gw_datagen import ...
# This file lives at:
#   Msci-GW/local tests/code tests/conftest.py
# Data Generation is at:
#   Msci-GW/HPC/Pipeline/Data Generation/
# ---------------------------------------------------------------------------
_DATA_GEN_DIR = str(
    Path(__file__).resolve().parent.parent.parent
    / "HPC" / "Pipeline" / "Data Generation"
)
sys.path.insert(0, _DATA_GEN_DIR)

# Directory where plots are saved (code tests/plots/)
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


# ---------------------------------------------------------------------------
# small_config — minimal, physically valid BBH parameter space.
# All values are lambdas so _validate_config accepts them.
# Zero spins and fixed sky location keep waveform generation fast.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def small_config():
    return {
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
        # redshift is required by MG/LV generators for phase-shift calculations
        "redshift":     lambda size: np.random.uniform(0.05, 0.15, size=size),
    }


# ---------------------------------------------------------------------------
# base_kwargs — shared generation settings optimised for test speed.
# num_workers=0 avoids multiprocessing deadlocks inside pytest on WSL2/Linux.
# f_final is NOT included here because pycbc_data_generator_real_psd
# does not accept it; aligo_kwargs below adds it for the standard generators.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def base_kwargs():
    return dict(
        num_samples=16,
        time_resolution=1 / 4096,
        approximant="IMRPhenomD",
        f_lower=40.0,
        signal_length=2.0,
        batch_size=8,
        train_split=0.7,
        val_split=0.15,
        add_noise=True,
        num_workers=0,
        show_progress=False,
    )


@pytest.fixture(scope="session")
def aligo_kwargs(base_kwargs):
    """base_kwargs extended with f_final for the standard (aLIGO) generators."""
    return {**base_kwargs, "f_final": 2048.0}


# ---------------------------------------------------------------------------
# psd_csv_path — build a synthetic PSD CSV once per test session.
# Uses aLIGOZeroDetHighPower from PyCBC so values are physically realistic.
# delta_f=0.125 Hz gives 8× oversampling relative to the generator's 0.5 Hz
# grid (signal_length=2.0 → delta_f_gen = 1/2 = 0.5 Hz), which means the
# interpolation step inside the generators has plenty of anchor points.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def psd_csv_path(tmp_path_factory):
    from pycbc.psd import aLIGOZeroDetHighPower

    tmp_dir = tmp_path_factory.mktemp("psd_data")
    csv_path = tmp_dir / "test_psds.csv"

    # Frequency grid: 0 .. 2048 Hz at 0.125 Hz resolution
    delta_f = 0.125
    f_max = 2048.0
    flow = 20.0  # below f_lower=40 used in generators (safer for interpolation)
    flen = int(f_max / delta_f) + 1

    psd_series = aLIGOZeroDetHighPower(flen, delta_f, flow)
    psd_vals = np.array(psd_series)
    freqs = np.arange(flen) * delta_f

    # Replace any zero / inf entries outside the sensitive band with a small
    # but finite floor so the interpolator doesn't produce degenerate values.
    floor = np.nanmin(psd_vals[psd_vals > 0]) if np.any(psd_vals > 0) else 1e-46
    psd_vals = np.where(np.isfinite(psd_vals) & (psd_vals > 0), psd_vals, floor)

    rows = []
    for event_idx, event_name in enumerate(["mock_event_1", "mock_event_2"]):
        for det in ("H1", "L1"):
            for f, p in zip(freqs, psd_vals):
                rows.append(
                    {
                        "event_rank":  event_idx + 1,
                        "event_name":  event_name,
                        "gps":         1369166418 + event_idx * 1000,
                        "detector":    det,
                        "frequency":   f,
                        "psd":         p,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(str(csv_path), index=False)
    return str(csv_path)
