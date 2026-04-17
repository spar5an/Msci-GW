"""
test_mg_gr_limit.py — Physics test: MG waveform converges to GR as m_g -> 0.

Demonstrates that the massive graviton phase correction (Will 1997,
arXiv:gr-qc/9709011) vanishes as m_g -> 0 (equivalently lambda_g -> inf),
recovering the GR waveform.

All physics imported directly from gw_datagen — no logic is duplicated here.

Run with:
    pytest test_mg_gr_limit.py -v

Generate a convergence plot (no pytest required):
    python test_mg_gr_limit.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from pycbc.waveform import get_fd_waveform

# ── Import everything from gw_datagen ────────────────────────────────────────
_DATA_GEN_DIR = str(
    Path(__file__).resolve().parent.parent.parent
    / "HPC" / "Pipeline" / "Data Generation"
)
sys.path.insert(0, _DATA_GEN_DIR)

from gw_datagen import (
    _C, _M_SUN_SEC, _H_PLANCK,
    _additional_phase,
    _m_g_to_lambda_g,
    _M_OMEGA_PN,
)


def _chirp_mass(m1, m2):
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


# ── Reference binary parameters ───────────────────────────────────────────────
_M1      = 30.0    # solar masses
_M2      = 30.0
_Z       = 0.1
_DIST    = 400.0   # Mpc
_F_LOWER = 30.0    # Hz
_F_FINAL = 512.0   # Hz
_DELTA_F = 0.25    # Hz

# Test m_g values (kg), ordered largest -> smallest (GR deviation -> GR).
# All chosen so delta_Psi(f_lower) is well below 1 rad (linear regime).
_M_G_SEQUENCE = [1e-58, 1e-59, 1e-60, 1e-61, 1e-63]   # kg


def _f_pn():
    return _M_OMEGA_PN / (np.pi * (_M1 + _M2) * _M_SUN_SEC)


# ── Shared waveform fixture ───────────────────────────────────────────────────
@pytest.fixture(scope="module")
def gr_fd():
    """GR frequency-domain waveform (hp polarisation, no phase modification)."""
    hp, _ = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=_M1, mass2=_M2,
        spin1z=0.0, spin2z=0.0,
        distance=_DIST, inclination=0.0, coa_phase=0.0,
        delta_f=_DELTA_F, f_lower=_F_LOWER, f_final=_F_FINAL,
    )
    return hp


# ── Helper ────────────────────────────────────────────────────────────────────
def _apply_phase_shift(hp_arr, pos_mask, freqs_pos, m_g):
    """Return a copy of hp_arr with the MG phase shift for m_g applied."""
    Mc   = _chirp_mass(_M1, _M2)
    dpsi = _additional_phase(freqs_pos, Mc, _Z, _m_g_to_lambda_g(m_g),
                              f_pn_cutoff=_f_pn())
    hp_mg = hp_arr.copy()
    hp_mg[pos_mask] *= np.exp(1j * dpsi)
    return hp_mg


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestPhaseShiftVanishes:
    """The MG phase correction delta_Psi must vanish as m_g -> 0."""

    def test_infinite_lambda_g_gives_exactly_zero(self):
        """lambda_g = inf (exact GR) must produce identically zero phase shift."""
        freqs = np.linspace(_F_LOWER, _F_FINAL, 500)
        dpsi  = _additional_phase(freqs, _chirp_mass(_M1, _M2), _Z, np.inf)
        assert np.all(dpsi == 0.0), (
            "Phase shift is not identically zero for lambda_g = inf"
        )

    def test_phase_shift_decreases_monotonically(self):
        """RMS |delta_Psi| must decrease as m_g decreases along _M_G_SEQUENCE."""
        Mc    = _chirp_mass(_M1, _M2)
        freqs = np.linspace(_F_LOWER + 1.0, _F_FINAL, 1000)

        rms_values = [
            float(np.sqrt(np.mean(
                _additional_phase(freqs, Mc, _Z, _m_g_to_lambda_g(mg),
                                  f_pn_cutoff=_f_pn()) ** 2
            )))
            for mg in _M_G_SEQUENCE
        ]

        for i in range(len(rms_values) - 1):
            assert rms_values[i] > rms_values[i + 1], (
                f"RMS phase shift did not decrease: "
                f"m_g {_M_G_SEQUENCE[i]:.0e} -> {_M_G_SEQUENCE[i+1]:.0e} kg  "
                f"({rms_values[i]:.3e} -> {rms_values[i+1]:.3e} rad)"
            )

    def test_phase_shift_proportional_to_mg_squared(self):
        """
        In the linear regime delta_Psi ~ m_g^2 (because lambda_g = h/m_g c,
        so beta ~ lambda_g^{-2} ~ m_g^2).  Verify the ratio between consecutive
        test values matches the expected m_g^2 scaling to within 0.1 %.
        """
        Mc    = _chirp_mass(_M1, _M2)
        freqs = np.linspace(_F_LOWER + 1.0, _F_FINAL, 1000)

        rms_values = [
            float(np.sqrt(np.mean(
                _additional_phase(freqs, Mc, _Z, _m_g_to_lambda_g(mg),
                                  f_pn_cutoff=_f_pn()) ** 2
            )))
            for mg in _M_G_SEQUENCE
        ]

        for i in range(len(_M_G_SEQUENCE) - 1):
            mg_ratio       = _M_G_SEQUENCE[i] / _M_G_SEQUENCE[i + 1]
            expected_ratio = mg_ratio ** 2
            actual_ratio   = rms_values[i] / rms_values[i + 1]
            assert abs(actual_ratio / expected_ratio - 1.0) < 1e-3, (
                f"Phase shift ratio {actual_ratio:.4f} deviates from expected "
                f"m_g^2 scaling {expected_ratio:.4f} at "
                f"m_g = {_M_G_SEQUENCE[i]:.0e} -> {_M_G_SEQUENCE[i+1]:.0e} kg"
            )

    def test_phase_shift_negligible_at_tiny_mass(self):
        """For m_g = 1e-63 kg, max |delta_Psi| < 1e-8 rad across the signal band."""
        Mc     = _chirp_mass(_M1, _M2)
        freqs  = np.linspace(_F_LOWER + 1.0, _F_FINAL, 1000)
        m_tiny = 1e-63
        lg     = _m_g_to_lambda_g(m_tiny)
        dpsi   = _additional_phase(freqs, Mc, _Z, lg, f_pn_cutoff=_f_pn())
        assert float(np.max(np.abs(dpsi))) < 1e-8


class TestWaveformConvergence:
    """The MG waveform must converge to the GR waveform as m_g -> 0."""

    def test_relative_error_decreases_monotonically(self, gr_fd):
        hp_arr    = gr_fd.numpy().copy()
        all_freqs = gr_fd.sample_frequencies.numpy()
        pos       = all_freqs > 0
        freqs_pos = all_freqs[pos]
        norm_gr   = np.linalg.norm(hp_arr)

        rel_errors = [
            np.linalg.norm(_apply_phase_shift(hp_arr, pos, freqs_pos, mg) - hp_arr)
            / norm_gr
            for mg in _M_G_SEQUENCE
        ]

        for i in range(len(rel_errors) - 1):
            assert rel_errors[i] > rel_errors[i + 1], (
                f"Relative waveform error did not decrease: "
                f"m_g {_M_G_SEQUENCE[i]:.0e} -> {_M_G_SEQUENCE[i+1]:.0e} kg  "
                f"({rel_errors[i]:.3e} -> {rel_errors[i+1]:.3e})"
            )

    def test_gr_recovered_at_tiny_mass(self, gr_fd):
        hp_arr    = gr_fd.numpy().copy()
        all_freqs = gr_fd.sample_frequencies.numpy()
        pos       = all_freqs > 0
        freqs_pos = all_freqs[pos]

        hp_mg     = _apply_phase_shift(hp_arr, pos, freqs_pos, 1e-63)
        rel_error = np.linalg.norm(hp_mg - hp_arr) / np.linalg.norm(hp_arr)
        assert rel_error < 1e-9, (
            f"Relative waveform error {rel_error:.2e} is too large; GR limit not recovered"
        )

    def test_exact_gr_at_infinite_lambda_g(self, gr_fd):
        hp_arr    = gr_fd.numpy().copy()
        all_freqs = gr_fd.sample_frequencies.numpy()
        pos       = all_freqs > 0
        freqs_pos = all_freqs[pos]
        Mc        = _chirp_mass(_M1, _M2)

        dpsi  = _additional_phase(freqs_pos, Mc, _Z, np.inf)
        hp_mg = hp_arr.copy()
        hp_mg[pos] *= np.exp(1j * dpsi)
        assert np.array_equal(hp_mg, hp_arr), (
            "Waveform changed after applying zero phase shift (lambda_g = inf)"
        )


# ── Standalone convergence plot ───────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Generating GR waveform…")
    hp, _ = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=_M1, mass2=_M2,
        spin1z=0.0, spin2z=0.0,
        distance=_DIST, inclination=0.0, coa_phase=0.0,
        delta_f=_DELTA_F, f_lower=_F_LOWER, f_final=_F_FINAL,
    )

    hp_arr    = hp.numpy().copy()
    all_freqs = hp.sample_frequencies.numpy()
    pos       = all_freqs > 0
    freqs_pos = all_freqs[pos]
    Mc        = _chirp_mass(_M1, _M2)
    norm_gr   = np.linalg.norm(hp_arr)

    print("Computing convergence curve…")
    m_g_range  = np.logspace(-63, -57, 50)
    rel_errors = [
        np.linalg.norm(_apply_phase_shift(hp_arr, pos, freqs_pos, mg) - hp_arr) / norm_gr
        for mg in m_g_range
    ]

    freqs_plot = np.linspace(_F_LOWER + 1.0, _F_FINAL, 500)
    m_g_lines  = [1e-58, 1e-59, 1e-60, 1e-61, 1e-63]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"MG -> GR limit:  m1=m2={_M1:.0f} Msun,  z={_Z},  "
        f"f=[{_F_LOWER:.0f},{_F_FINAL:.0f}] Hz",
        fontsize=12,
    )

    ax = axes[0]
    ax.loglog(m_g_range, rel_errors, "o-", ms=4, lw=1.5, color="steelblue")
    ax.set_xlabel(r"Graviton mass  $m_g$  (kg)")
    ax.set_ylabel(r"$\|h_\mathrm{MG} - h_\mathrm{GR}\|_2 / \|h_\mathrm{GR}\|_2$")
    ax.set_title("Relative waveform error")
    ax.grid(True, which="both", ls=":", alpha=0.5)

    ax2 = axes[1]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(m_g_lines)))
    for mg, col in zip(m_g_lines, colors):
        lg   = _m_g_to_lambda_g(mg)
        dpsi = _additional_phase(freqs_plot, Mc, _Z, lg, f_pn_cutoff=_f_pn())
        ax2.semilogy(freqs_plot, np.abs(dpsi), color=col, lw=1.5,
                     label=f"$m_g$ = {mg:.0e} kg")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel(r"$|\delta\Psi(f)|$  (rad)")
    ax2.set_title("Phase shift magnitude vs. frequency")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, which="both", ls=":", alpha=0.5)

    fig.tight_layout()
    out_path = Path(__file__).parent / "mg_gr_limit.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")
