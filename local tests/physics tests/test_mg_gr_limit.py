"""
test_mg_gr_limit.py — Physics test: MG waveform converges to GR as m_g -> 0.

Demonstrates that the massive graviton phase correction (Will 1997,
arXiv:gr-qc/9709011)

    delta_Psi(f) = -beta * u^{-1}

    beta  = pi^2 * c * D_0(z) * M_det / (lambda_g^2 * (1+z))
    u     = pi * M_det * f
    M_det = M_chirp * M_sun_sec * (1+z)      [detector-frame chirp mass, seconds]
    D_0(z) = cosmological distance integral  [metres]
    lambda_g = h / (m_g * c)                 [Compton wavelength, metres]

vanishes as m_g -> 0 (equivalently lambda_g -> inf), recovering the GR waveform.

All physics is inlined here; gw_datagen is not imported.  This makes the test
an independent check of the mathematical limit rather than a round-trip through
the generation code.

Run with:
    pytest test_mg_gr_limit.py -v

Generate a convergence plot (no pytest required):
    python test_mg_gr_limit.py
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from pycbc.waveform import get_fd_waveform


# ── Physical constants (inlined, matching gw_datagen.py) ──────────────────────
_C         = 2.998e8            # speed of light, m/s
_H_PLANCK  = 6.626e-34          # Planck's constant, J*s
_G         = 6.674e-11          # gravitational constant, m^3 kg^-1 s^-2
_M_SUN     = 1.989e30           # solar mass, kg
_MPC       = 3.086e22           # megaparsec, m
_M_SUN_SEC = _G * _M_SUN / _C**3  # solar mass in seconds (~4.926e-6 s)
_H0        = 67.4e3 / _MPC      # Hubble constant, s^-1
_OMEGA_M   = 0.315
_OMEGA_LAMBDA = 0.685


# ── Inlined cosmological distance integral ────────────────────────────────────
def _D0(z):
    """
    Cosmological distance D_0(z) in metres (flat LCDM, alpha=0).

    Matches _D_alpha(0, z) in gw_datagen.py exactly.
    """
    result, _ = quad(
        lambda zp: (1 + zp) ** (-2) / np.sqrt(_OMEGA_M * (1 + zp) ** 3 + _OMEGA_LAMBDA),
        0,
        z,
    )
    return _C * (1 + z) / _H0 * result


# ── Inlined MG phase-shift formula ────────────────────────────────────────────
def _mg_phase_shift(freqs, chirp_mass_msun, z, lambda_g):
    """
    Massive graviton phase correction delta_Psi(f).

    Parameters
    ----------
    freqs : ndarray
        Positive frequency bins in Hz.
    chirp_mass_msun : float
        Source-frame chirp mass in solar masses.
    z : float
        Source redshift.
    lambda_g : float
        Graviton Compton wavelength in metres.  np.inf -> GR (zero shift).

    Returns
    -------
    delta_psi : ndarray  (radians)
    """
    if not np.isfinite(lambda_g) or lambda_g <= 0:
        return np.zeros_like(freqs, dtype=float)

    M_det = chirp_mass_msun * _M_SUN_SEC * (1 + z)
    u     = np.pi * M_det * freqs
    beta  = np.pi ** 2 * _C * _D0(z) * M_det / (lambda_g ** 2 * (1 + z))
    return -beta * u ** (-1)


# ── Conversion helpers ────────────────────────────────────────────────────────
def _m_g_to_lambda_g(m_g_kg):
    """lambda_g = h / (m_g * c)  [metres]."""
    return _H_PLANCK / (m_g_kg * _C)


def _chirp_mass(m1, m2):
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


# ── Reference binary parameters ───────────────────────────────────────────────
_M1      = 30.0    # solar masses
_M2      = 30.0
_Z       = 0.1
_DIST    = 400.0   # Mpc
_F_LOWER = 30.0    # Hz
_F_FINAL = 512.0   # Hz  (modest upper limit keeps the fixture fast)
_DELTA_F = 0.25    # Hz

# Test m_g values (kg), ordered largest -> smallest (i.e. GR deviation -> GR).
# All chosen so that the peak phase shift delta_Psi(f_lower) is well below
# 1 rad, keeping every point in the linear regime where the relative waveform
# error is proportional to m_g^2 and the monotonic decrease is exact.
#
#   delta_Psi(f_lower) ~ pi * c * D0(z) / (lambda_g^2 * (1+z) * f_lower)
#                       = pi * c * D0(z) * m_g^2 / (_H_PLANCK/c)^2 / ((1+z)*f_lower)
#
# For this binary at z=0.1, f_lower=30 Hz: threshold m_g ~ 2e-58 kg.
# All five values below sit safely within the linear regime.
_M_G_SEQUENCE = [1e-58, 1e-59, 1e-60, 1e-61, 1e-63]   # kg


# ── Shared waveform fixture ───────────────────────────────────────────────────
@pytest.fixture(scope="module")
def gr_fd():
    """GR frequency-domain waveform (hp polarisation, no phase modification)."""
    hp, _ = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=_M1,
        mass2=_M2,
        spin1z=0.0,
        spin2z=0.0,
        distance=_DIST,
        inclination=0.0,
        coa_phase=0.0,
        delta_f=_DELTA_F,
        f_lower=_F_LOWER,
        f_final=_F_FINAL,
    )
    return hp


# ── Helper ────────────────────────────────────────────────────────────────────
def _apply_phase_shift(hp_arr, pos_mask, freqs_pos, m_g):
    """Return a copy of hp_arr with the MG phase shift for m_g applied."""
    Mc   = _chirp_mass(_M1, _M2)
    dpsi = _mg_phase_shift(freqs_pos, Mc, _Z, _m_g_to_lambda_g(m_g))
    hp_mg = hp_arr.copy()
    hp_mg[pos_mask] *= np.exp(1j * dpsi)
    return hp_mg


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestPhaseShiftVanishes:
    """The MG phase correction delta_Psi must vanish as m_g -> 0."""

    def test_infinite_lambda_g_gives_exactly_zero(self):
        """lambda_g = inf (exact GR) must produce identically zero phase shift."""
        freqs = np.linspace(_F_LOWER, _F_FINAL, 500)
        dpsi  = _mg_phase_shift(freqs, _chirp_mass(_M1, _M2), _Z, np.inf)
        assert np.all(dpsi == 0.0), (
            "Phase shift is not identically zero for lambda_g = inf"
        )

    def test_phase_shift_decreases_monotonically(self):
        """RMS |delta_Psi| must decrease as m_g decreases along _M_G_SEQUENCE."""
        Mc    = _chirp_mass(_M1, _M2)
        freqs = np.linspace(_F_LOWER + 1.0, _F_FINAL, 1000)

        rms_values = [
            float(np.sqrt(np.mean(_mg_phase_shift(freqs, Mc, _Z,
                                                   _m_g_to_lambda_g(mg)) ** 2)))
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
            float(np.sqrt(np.mean(_mg_phase_shift(freqs, Mc, _Z,
                                                   _m_g_to_lambda_g(mg)) ** 2)))
            for mg in _M_G_SEQUENCE
        ]

        for i in range(len(_M_G_SEQUENCE) - 1):
            mg_ratio      = _M_G_SEQUENCE[i] / _M_G_SEQUENCE[i + 1]
            expected_ratio = mg_ratio ** 2
            actual_ratio   = rms_values[i] / rms_values[i + 1]
            assert abs(actual_ratio / expected_ratio - 1.0) < 1e-3, (
                f"Phase shift ratio {actual_ratio:.4f} deviates from expected "
                f"m_g^2 scaling {expected_ratio:.4f} at "
                f"m_g = {_M_G_SEQUENCE[i]:.0e} -> {_M_G_SEQUENCE[i+1]:.0e} kg"
            )

    def test_phase_shift_negligible_at_tiny_mass(self):
        """
        For m_g = 1e-63 kg, max |delta_Psi| < 1e-8 rad across the signal band.
        (Actual value is O(1e-10) for this binary; threshold has 2 decades margin.)
        """
        Mc     = _chirp_mass(_M1, _M2)
        freqs  = np.linspace(_F_LOWER + 1.0, _F_FINAL, 1000)
        m_tiny = 1e-63   # kg
        lg     = _m_g_to_lambda_g(m_tiny)
        dpsi   = _mg_phase_shift(freqs, Mc, _Z, lg)
        max_shift = float(np.max(np.abs(dpsi)))
        assert max_shift < 1e-8, (
            f"Max phase shift {max_shift:.2e} rad should be negligible "
            f"for m_g = {m_tiny:.0e} kg  (lambda_g = {lg:.2e} m)"
        )


class TestWaveformConvergence:
    """The MG waveform must converge to the GR waveform as m_g -> 0."""

    def test_relative_error_decreases_monotonically(self, gr_fd):
        """
        ||h_MG - h_GR||_2 / ||h_GR||_2 must decrease along _M_G_SEQUENCE.
        """
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
        """
        At m_g = 1e-63 kg the relative waveform error must be < 1e-9.
        The GR waveform is recovered to essentially double-precision accuracy.
        """
        hp_arr    = gr_fd.numpy().copy()
        all_freqs = gr_fd.sample_frequencies.numpy()
        pos       = all_freqs > 0
        freqs_pos = all_freqs[pos]

        m_g_tiny  = 1e-63
        hp_mg     = _apply_phase_shift(hp_arr, pos, freqs_pos, m_g_tiny)
        rel_error = np.linalg.norm(hp_mg - hp_arr) / np.linalg.norm(hp_arr)

        assert rel_error < 1e-9, (
            f"Relative waveform error {rel_error:.2e} is too large for "
            f"m_g = {m_g_tiny:.0e} kg; GR limit not recovered"
        )

    def test_exact_gr_at_infinite_lambda_g(self, gr_fd):
        """
        Applying a zero phase shift (lambda_g = inf) must leave the waveform
        bit-for-bit identical.
        """
        hp_arr    = gr_fd.numpy().copy()
        all_freqs = gr_fd.sample_frequencies.numpy()
        pos       = all_freqs > 0
        freqs_pos = all_freqs[pos]
        Mc        = _chirp_mass(_M1, _M2)

        dpsi  = _mg_phase_shift(freqs_pos, Mc, _Z, np.inf)
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

    # Convergence curve: relative waveform error vs. m_g
    print("Computing convergence curve…")
    m_g_range  = np.logspace(-63, -57, 50)   # kg
    rel_errors = []
    for mg in m_g_range:
        hp_mg = _apply_phase_shift(hp_arr, pos, freqs_pos, mg)
        rel_errors.append(np.linalg.norm(hp_mg - hp_arr) / norm_gr)

    # Phase shift vs. frequency for selected m_g values
    freqs_plot = np.linspace(_F_LOWER + 1.0, _F_FINAL, 500)
    m_g_lines  = [1e-58, 1e-59, 1e-60, 1e-61, 1e-63]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"MG -> GR limit:  m1=m2={_M1:.0f} Msun,  z={_Z},  "
        f"f=[{_F_LOWER:.0f},{_F_FINAL:.0f}] Hz",
        fontsize=12,
    )

    # Left: relative waveform error vs. m_g
    ax = axes[0]
    ax.loglog(m_g_range, rel_errors, "o-", ms=4, lw=1.5, color="steelblue")
    ax.set_xlabel(r"Graviton mass  $m_g$  (kg)")
    ax.set_ylabel(
        r"$\|h_\mathrm{MG} - h_\mathrm{GR}\|_2 \;/\; \|h_\mathrm{GR}\|_2$"
    )
    ax.set_title("Relative waveform error")
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # Right: |delta_Psi(f)| for each test m_g
    ax2 = axes[1]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(m_g_lines)))
    for mg, col in zip(m_g_lines, colors):
        lg   = _m_g_to_lambda_g(mg)
        dpsi = _mg_phase_shift(freqs_plot, Mc, _Z, lg)
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
