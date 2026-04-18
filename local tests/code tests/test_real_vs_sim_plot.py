"""
test_real_vs_sim_plot.py — sanity plot that proves the real and simulated
pipelines produce compatible .pt files.

Loads a simulated dataset (generated fresh) and a real dataset (built from
cached HDF5 under Real Data/hdf5/) through ``gw_datagen.load_dataloaders``,
then plots three H1 samples from each side-by-side (raw + whitened) to
``plots/real_vs_sim_h1.png``.

Skipped if fewer than N_EVENTS cached HDF5 pairs are present.

Run with:
    pytest test_real_vs_sim_plot.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gwpy.timeseries import TimeSeries

from conftest import PLOTS_DIR
from gw_datagen import (
    load_dataloaders,
    pycbc_data_generator,
    pycbc_lorentz_violation_data_generator,
    pycbc_massive_gravity_data_generator,
    save_dataloaders,
)

# Fixed MG/LV physics values for the schema check (chosen to be within the
# generate_dataset.py bounds; matches the values used in test_analytic.py).
_M_G     = 2.21e-64  # kg
_ALPHA_LV = 3.0
_A_LV    = 5.07e21   # eV^-1 at alpha=3

_REAL_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "HPC" / "Pipeline" / "Real Data"
)
sys.path.insert(0, str(_REAL_DATA_DIR))

from download_real_data import (  # noqa: E402
    COMBINED_NAME,
    DURATION,
    N_2S,
    build_combined,
    layout,
    process_to_pt,
)

CACHED_HDF5_DIR = _REAL_DATA_DIR / "hdf5"
N_EVENTS = 3


@pytest.fixture(scope="module")
def real_pt(tmp_path_factory):
    """Build a real-data .pt from the first N_EVENTS cached H1+L1 pairs."""
    if not CACHED_HDF5_DIR.is_dir():
        pytest.skip(f"No cached hdf5 directory at {CACHED_HDF5_DIR}")

    events: dict[str, float] = {}
    for h1 in CACHED_HDF5_DIR.glob("*_H1_strain.hdf5"):
        event = h1.name[: -len("_H1_strain.hdf5")]
        l1 = CACHED_HDF5_DIR / f"{event}_L1_strain.hdf5"
        if l1.exists():
            ts = TimeSeries.read(str(h1))
            events[event] = float(ts.t0.value) + DURATION / 2
    if len(events) < N_EVENTS:
        pytest.skip(
            f"Need ≥{N_EVENTS} cached H1+L1 pairs under {CACHED_HDF5_DIR}, "
            f"found {len(events)}"
        )

    chosen = list(events.items())[:N_EVENTS]

    out_dir = str(tmp_path_factory.mktemp("real_pt_for_plot"))
    dirs = layout(out_dir)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    per_event = []
    for event, gps in chosen:
        for det in ("H1", "L1"):
            src = CACHED_HDF5_DIR / f"{event}_{det}_strain.hdf5"
            dst = Path(dirs["hdf5"]) / f"{event}_{det}_strain.hdf5"
            if not dst.exists():
                os.symlink(src, dst)
        hdf5_paths = {
            det: os.path.join(dirs["hdf5"], f"{event}_{det}_strain.hdf5")
            for det in ("H1", "L1")
        }
        per_event.append((event, process_to_pt(event, gps, hdf5_paths)))

    combined = build_combined(per_event, batch_size=1)
    combined_path = os.path.join(dirs["pt"], COMBINED_NAME)
    torch.save(combined, combined_path)
    return combined_path


# Realistic BBH config — mirrors the shared astrophysical CONFIG in
# generate_dataset.py (GWTC-3-informed BBH ranges). GR/MG/LV all use this
# same base; MG/LV add their physics labels (m_g, alpha_lv, A) inside the
# generator itself. The MG/LV generators derive cosmological z from
# ``distance`` via ``_redshift_from_distance``, so redshift does not need to
# appear in the config for the phase shifts to stay consistent with d_L.
def _realistic_bbh_config():
    return {
        "mass1":        lambda size: np.random.uniform(5, 90, size=size),
        "mass2":        lambda size: np.random.uniform(5, 90, size=size),
        "spin1z":       lambda size: np.random.uniform(-0.8, 0.8, size=size),
        "spin2z":       lambda size: np.random.uniform(-0.8, 0.8, size=size),
        "distance":     lambda size: np.random.uniform(100, 5000, size=size),
        "inclination":  lambda size: np.arccos(np.random.uniform(-1, 1, size=size)),
        "coa_phase":    lambda size: np.random.uniform(0, 2 * np.pi, size=size),
        "ra":           lambda size: np.random.uniform(0, 2 * np.pi, size=size),
        "dec":          lambda size: np.arcsin(np.random.uniform(-1, 1, size=size)),
        "polarization": lambda size: np.random.uniform(0, np.pi, size=size),
    }


@pytest.fixture(scope="module")
def sim_pt(tmp_path_factory, aligo_kwargs):
    """Generate a GR dataset with realistic BBH parameter ranges and save."""
    result = pycbc_data_generator(config=_realistic_bbh_config(), **aligo_kwargs)
    out = tmp_path_factory.mktemp("sim_pt_for_plot") / "sim.pt"
    save_dataloaders(result, str(out))
    return str(out)


@pytest.fixture(scope="module")
def sim_pt_mg(tmp_path_factory, aligo_kwargs):
    """Generate an MG dataset using the SAME realistic BBH config as GR."""
    result = pycbc_massive_gravity_data_generator(
        config=_realistic_bbh_config(), m_g=_M_G, **aligo_kwargs,
    )
    out = tmp_path_factory.mktemp("sim_pt_mg") / "mg.pt"
    save_dataloaders(result, str(out))
    return str(out)


@pytest.fixture(scope="module")
def sim_pt_lv(tmp_path_factory, aligo_kwargs):
    """Generate an LV dataset using the SAME realistic BBH config as GR."""
    result = pycbc_lorentz_violation_data_generator(
        config=_realistic_bbh_config(), alpha_lv=_ALPHA_LV, A=_A_LV, m_g=_M_G,
        **aligo_kwargs,
    )
    out = tmp_path_factory.mktemp("sim_pt_lv") / "lv.pt"
    save_dataloaders(result, str(out))
    return str(out)


class TestRealVsSimPlot:
    def test_both_load_through_load_dataloaders(self, real_pt, sim_pt):
        # Smoke: the same loader must consume both file types.
        real = load_dataloaders(real_pt)
        sim  = load_dataloaders(sim_pt)
        for loaded in (real, sim):
            assert set(loaded.keys()) == {
                "train_loader", "val_loader", "test_loader", "metadata",
            }

    def test_plot_side_by_side(self, real_pt, sim_pt):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        real_blob = torch.load(real_pt, weights_only=False)
        sim_blob  = torch.load(sim_pt,  weights_only=False)

        assert real_blob["X"].shape[1:] == sim_blob["X"].shape[1:], (
            f"Shape mismatch: real {real_blob['X'].shape} vs sim {sim_blob['X'].shape}"
        )
        assert real_blob["X"].shape[0] >= N_EVENTS
        assert sim_blob["X"].shape[0] >= N_EVENTS

        dt = real_blob["metadata"]["time_resolution"]
        T  = real_blob["X"].shape[-1]
        t  = np.arange(T) * dt
        events = real_blob["metadata"]["events"][:N_EVENTS]

        # 4 rows × N_EVENTS cols:
        #   0 simulated raw, 1 simulated whitened, 2 real raw, 3 real whitened
        fig, axes = plt.subplots(4, N_EVENTS, figsize=(5 * N_EVENTS, 11),
                                 sharex=True)
        fig.suptitle(
            f"Real vs Simulated — H1 channel, {N_EVENTS} samples each",
            fontweight="bold",
        )

        for col in range(N_EVENTS):
            sim_raw = sim_blob["X"][col, 0].numpy()
            sim_wht = sim_blob["X_whitened"][col, 0].numpy()
            rel_raw = real_blob["X"][col, 0].numpy()
            rel_wht = real_blob["X_whitened"][col, 0].numpy()

            axes[0, col].plot(t, sim_raw, lw=0.6, color="steelblue")
            axes[0, col].set_title(f"Simulated #{col} — raw", fontsize=10)
            axes[0, col].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            axes[1, col].plot(t, sim_wht, lw=0.6, color="steelblue")
            axes[1, col].set_title(f"Simulated #{col} — whitened", fontsize=10)

            axes[2, col].plot(t, rel_raw, lw=0.6, color="firebrick")
            axes[2, col].set_title(f"{events[col]} — raw", fontsize=10)
            axes[2, col].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            axes[3, col].plot(t, rel_wht, lw=0.6, color="firebrick")
            axes[3, col].set_title(f"{events[col]} — whitened", fontsize=10)
            axes[3, col].set_xlabel("time (s)")

        axes[0, 0].set_ylabel("sim strain")
        axes[1, 0].set_ylabel("sim whitened")
        axes[2, 0].set_ylabel("real strain")
        axes[3, 0].set_ylabel("real whitened")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        out = os.path.join(PLOTS_DIR, "real_vs_sim_h1.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)

        assert os.path.isfile(out) and os.path.getsize(out) > 0


class TestModesShareRealisticConfig:
    """GR, MG and LV must all accept the same realistic BBH CONFIG and produce
    identically-shaped tensors. This mirrors generate_dataset.py, which builds
    one shared CONFIG and only swaps the generator + adds physics labels."""

    def _load(self, path):
        blob = torch.load(path, weights_only=False)
        return blob["X"], blob["X_whitened"], blob["y"], blob["metadata"]

    def test_shapes_match_across_modes(self, sim_pt, sim_pt_mg, sim_pt_lv):
        X_gr, Xw_gr, _, _ = self._load(sim_pt)
        X_mg, Xw_mg, _, _ = self._load(sim_pt_mg)
        X_lv, Xw_lv, _, _ = self._load(sim_pt_lv)
        assert X_mg.shape == X_gr.shape
        assert X_lv.shape == X_gr.shape
        assert Xw_mg.shape == Xw_gr.shape
        assert Xw_lv.shape == Xw_gr.shape

    def test_parameter_names_cover_bbh_astro(self, sim_pt, sim_pt_mg, sim_pt_lv):
        # All three must expose the same astrophysical BBH columns in y.
        base = {"mass1", "mass2", "spin1z", "spin2z", "distance",
                "inclination", "coa_phase", "ra", "dec", "polarization"}
        for path in (sim_pt, sim_pt_mg, sim_pt_lv):
            names = set(self._load(path)[3]["parameter_names"])
            missing = base - names
            assert not missing, f"{path} missing params: {missing}"

    def test_mg_and_lv_physics_labels_present(self, sim_pt_mg, sim_pt_lv):
        # MG advertises m_g; LV advertises alpha_lv + A.
        assert self._load(sim_pt_mg)[3]["m_g"] == pytest.approx(_M_G)
        meta_lv = self._load(sim_pt_lv)[3]
        assert meta_lv["alpha_lv"] == pytest.approx(_ALPHA_LV)
        assert meta_lv["A"] == pytest.approx(_A_LV)

    def test_whitened_is_finite(self, sim_pt, sim_pt_mg, sim_pt_lv):
        for path in (sim_pt, sim_pt_mg, sim_pt_lv):
            Xw = self._load(path)[1]
            assert torch.isfinite(Xw).all()
            assert Xw.abs().sum().item() > 0

    def test_load_dataloaders_consumes_all_modes(self, sim_pt, sim_pt_mg, sim_pt_lv):
        for path in (sim_pt, sim_pt_mg, sim_pt_lv):
            loaded = load_dataloaders(path)
            assert set(loaded.keys()) == {
                "train_loader", "val_loader", "test_loader", "metadata",
            }
