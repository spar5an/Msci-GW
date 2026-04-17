"""
conftest.py — Shared fixtures for gw_datagen test suite.

Fixtures
--------
small_config    : minimal BBH parameter config (all lambdas, session-scoped)
base_kwargs     : fast generation kwargs shared by both test files
aligo_kwargs    : base_kwargs + f_final=2048.0 (for standard generators only)
"""

import os
import sys
import numpy as np
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
# num_workers=1 avoids multiprocessing deadlocks inside pytest on WSL2/Linux.
# f_final is NOT included here because pycbc_data_generator_real_psd
# does not accept it; aligo_kwargs below adds it for the standard generators.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def base_kwargs():
    return dict(
        num_samples=16,
        time_resolution=1 / 4096,
        approximant="IMRPhenomD",
        f_lower=10.0,
        signal_length=2.0,
        batch_size=8,
        train_split=0.7,
        val_split=0.15,
        add_noise=True,
        num_workers=1,
        show_progress=False,
    )


@pytest.fixture(scope="session")
def aligo_kwargs(base_kwargs):
    """base_kwargs extended with f_final for the standard (aLIGO) generators."""
    return {**base_kwargs, "f_final": 2048.0}


# ---------------------------------------------------------------------------
# psd_cache_dir — points at the fixed o4_psd_cache/ folder next to gw_datagen.py.
# Tests that use this fixture are skipped if the cache has not been populated.
# To populate it, run:
#   python download_o4_psds.py --detectors H1 L1 --n-segments 100
# from the Data Generation directory (requires internet access).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def psd_cache_dir():
    """Return the fixed O4 PSD cache directory; skip if it doesn't contain H1 data."""
    from gw_datagen import _DEFAULT_CACHE, _cache_path

    cache_dir = _DEFAULT_CACHE
    h1_cache = _cache_path("H1", 4096, cache_dir)
    if not os.path.isfile(h1_cache):
        pytest.skip(
            f"O4 PSD cache not found at {h1_cache}. "
            "Run download_o4_psds.py to populate it."
        )
    return cache_dir
