"""
plot_mg_waveforms.py — Visualise MG and LV(A=0) waveforms vs. GR for varying m_g.

Two side-by-side comparisons:
  Left  — Massive Graviton (MG):   phase correction using _additional_phase only.
  Right — Lorentz Violation (LV) with A_lv = inf  (LV term suppressed, MG term
          only):  identical physics to MG, plotted independently as a cross-check.

Expected result: MG and LV(A=0) waveforms are bit-for-bit identical for the same
m_g, demonstrating that the LV generator reduces correctly to the MG limit.

Saves:  mg_waveforms.png  in the same directory as this script.

Run with:
    python plot_mg_waveforms.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
from pycbc.waveform import get_fd_waveform
from pycbc.types import TimeSeries
from pycbc.filter import highpass_fir


# ── Physical constants (matching gw_datagen.py) ───────────────────────────────
_C         = 2.998e8
_H_PLANCK  = 6.626e-34
_G         = 6.674e-11
_M_SUN     = 1.989e30
_MPC       = 3.086e22
_M_SUN_SEC = _G * _M_SUN / _C**3
_H0        = 67.4e3 / _MPC
_OMEGA_M   = 0.315
_OMEGA_LAMBDA = 0.685


def _D_alpha(alpha, z):
    result, _ = quad(
        lambda zp: (1 + zp) ** (alpha - 2)
                   / np.sqrt(_OMEGA_M * (1 + zp) ** 3 + _OMEGA_LAMBDA),
        0, z,
    )
    return _C * (1 + z) / _H0 * result


def _mg_phase(freqs, Mc, z, lambda_g):
    """Pure MG phase shift delta_Psi(f) = -beta * u^{-1}."""
    if not np.isfinite(lambda_g) or lambda_g <= 0:
        return np.zeros_like(freqs, dtype=float)
    M_det = Mc * _M_SUN_SEC * (1 + z)
    u     = np.pi * M_det * freqs
    beta  = np.pi**2 * _C * _D_alpha(0, z) * M_det / (lambda_g**2 * (1 + z))
    return -beta * u**(-1)


def _lv_phase(freqs, Mc, z, lambda_g, alpha_lv, A_lv, f_c):
    """
    Full LV phase shift (Mirshekari et al. 2011).

    With A_lv = np.inf the LV term vanishes and only the MG mass term remains,
    so this reduces identically to _mg_phase().
    """
    M_det = Mc * _M_SUN_SEC * (1 + z)
    u     = np.pi * M_det * freqs
    u_c   = np.pi * M_det * f_c

    # MG mass term
    if np.isfinite(lambda_g) and lambda_g > 0:
        beta     = np.pi**2 * _C * _D_alpha(0, z) * M_det / (lambda_g**2 * (1 + z))
        dpsi_mg  = -beta * u**(-1)
    else:
        dpsi_mg = np.zeros_like(freqs)

    # LV term (suppressed when A_lv = inf)
    if not np.isfinite(A_lv) or A_lv <= 0 or alpha_lv == 2.0:
        dpsi_lv = np.zeros_like(freqs)
    elif alpha_lv == 1.0:
        zeta    = _D_alpha(1, z) / A_lv
        dpsi_lv = zeta * (np.log(u) - np.log(u_c))
    else:
        zeta = (
            np.pi**(2 - alpha_lv) / (1 - alpha_lv)
            * _C**(1 - alpha_lv)
            * _D_alpha(alpha_lv, z)
            * M_det**(1 - alpha_lv)
            / (A_lv**(2 - alpha_lv) * (1 + z)**(1 - alpha_lv))
        )
        dpsi_lv = -zeta * (u**(alpha_lv - 1) - u_c**(alpha_lv - 1))

    return dpsi_mg + dpsi_lv


_HIGHPASS_FC        = 35     # high-pass filter cutoff (Hz) — matches gw_datagen
_RINGDOWN_TAPER_LEN = 128    # samples cosine-tapered to zero at array end


def _apply_end_taper(arr):
    """Cosine-taper last _RINGDOWN_TAPER_LEN samples to zero (mirrors gw_datagen)."""
    arr = arr.copy()
    xi = np.linspace(0, 1, _RINGDOWN_TAPER_LEN)
    arr[-_RINGDOWN_TAPER_LEN:] *= 0.5 * (1.0 + np.cos(np.pi * xi))
    return arr


def _m_g_to_lambda_g(m_g_kg):
    return _H_PLANCK / (m_g_kg * _C)


def _chirp_mass(m1, m2):
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


# ── Waveform parameters ───────────────────────────────────────────────────────
M1, M2   = 30.0, 30.0   # solar masses
Z        = 0.1
DIST     = 400.0         # Mpc
F_LOWER  = 30.0          # Hz
F_FINAL  = 512.0         # Hz
DELTA_F  = 0.25          # Hz
ALPHA_LV = 3.0           # DSR exponent (only matters if A_lv is finite)

MC = _chirp_mass(M1, M2)

# m_g values (kg) to display — chosen so delta_Psi(f_lower) is O(0.1)–O(1) rad,
# giving clearly visible but not scrambled phase differences.
M_G_VALUES = [2e-58, 8e-59, 3e-59]
COLORS     = ["#e63946", "#f4a261", "#2a9d8f"]
GR_COLOR   = "#1d3557"


# ── Generate GR reference waveform ───────────────────────────────────────────
print("Generating GR reference waveform…")
hp_gr_fd, _ = get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=M1, mass2=M2,
    spin1z=0.0, spin2z=0.0,
    distance=DIST, inclination=0.0, coa_phase=0.0,
    delta_f=DELTA_F, f_lower=F_LOWER, f_final=F_FINAL,
)

all_freqs = hp_gr_fd.sample_frequencies.numpy()
pos       = all_freqs > 0
freqs_pos = all_freqs[pos]

# f_c for LV normalisation: highest non-zero frequency bin
hp_gr_arr = hp_gr_fd.numpy()
hp_gr_amp = np.abs(hp_gr_arr[1:])
f_c       = float(np.max(freqs_pos[np.nonzero(hp_gr_amp)]))

# TD waveform via IRFFT — keep the full array and trim to the merger window
dt = 1.0 / (2.0 * F_FINAL)   # sample interval (s)


def _to_td(hp_fd_arr):
    """IRFFT + normalise + end-taper + highpass — mirrors gw_datagen pipeline."""
    td  = np.fft.irfft(hp_fd_arr)
    N   = len(td)
    td *= DELTA_F * N
    td  = _apply_end_taper(td)
    ts  = TimeSeries(td.astype(np.float64), delta_t=dt)
    ts  = highpass_fir(ts, _HIGHPASS_FC, 128)
    return ts.numpy()


gr_td = _to_td(hp_gr_arr)
N_td  = len(gr_td)

# Merger is near the peak amplitude — keep ~0.5 s centred on it
n_show  = int(0.5 / dt)
peak    = int(np.argmax(np.abs(gr_td)))
i_start = max(0, peak - n_show)
i_end   = min(N_td, peak + int(0.1 / dt))   # 0.1 s of ringdown after peak
t_axis  = (np.arange(i_start, i_end) - peak) * dt   # seconds relative to merger


def _modified_td(phase_shift_arr):
    """Apply a phase shift to hp_gr_arr and return the trimmed TD waveform."""
    hp_mod = hp_gr_arr.copy()
    hp_mod[pos] *= np.exp(1j * phase_shift_arr)
    td = _to_td(hp_mod)
    return td[i_start:i_end]


gr_td_trim = gr_td[i_start:i_end]


# ── Build modified waveforms ──────────────────────────────────────────────────
print("Applying MG / LV phase shifts…")
mg_td_list = []
lv_td_list = []

for m_g in M_G_VALUES:
    lg = _m_g_to_lambda_g(m_g)

    dpsi_mg = _mg_phase(freqs_pos, MC, Z, lg)
    dpsi_lv = _lv_phase(freqs_pos, MC, Z, lg, ALPHA_LV, np.inf, f_c)

    mg_td_list.append(_modified_td(dpsi_mg))
    lv_td_list.append(_modified_td(dpsi_lv))


# ── Plotting ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 2, figsize=(14, 8),
    gridspec_kw={"height_ratios": [2, 1]},
    sharex=True,
)
fig.suptitle(
    rf"Massive graviton waveforms:  $m_1 = m_2 = {M1:.0f}\,M_\odot$, "
    rf"$z = {Z}$,  $d = {DIST:.0f}$ Mpc",
    fontsize=13,
)

# Normalise to GR peak amplitude for comparison
norm = float(np.max(np.abs(gr_td_trim)))

for col, (td_list, title) in enumerate([
    (mg_td_list, "Massive Graviton (MG)"),
    (lv_td_list, rf"LV with $A = 0$ ($A_{{lv}} = \infty$, $\alpha = {ALPHA_LV}$)"),
]):
    ax_wave = axes[0, col]
    ax_res  = axes[1, col]

    # GR reference
    ax_wave.plot(t_axis * 1e3, gr_td_trim / norm,
                 color=GR_COLOR, lw=1.8, label="GR", zorder=4)

    for m_g, td, color in zip(M_G_VALUES, td_list, COLORS):
        lg_m = _m_g_to_lambda_g(m_g)
        label = rf"$m_g = {m_g:.0e}$ kg  ($\lambda_g = {lg_m:.1e}$ m)"
        ax_wave.plot(t_axis * 1e3, td / norm,
                     color=color, lw=1.2, alpha=0.85, label=label)

        residual = (td - gr_td_trim) / norm
        ax_res.plot(t_axis * 1e3, residual,
                    color=color, lw=1.0, alpha=0.85)

    # Formatting — waveform panel
    ax_wave.axhline(0, color="k", lw=0.4, ls=":")
    ax_wave.set_title(title, fontsize=11)
    ax_wave.set_ylabel("Strain  (normalised)", fontsize=9)
    ax_wave.legend(fontsize=7.5, loc="upper left", framealpha=0.8)
    ax_wave.set_xlim(t_axis[0] * 1e3, t_axis[-1] * 1e3)
    ax_wave.grid(True, ls=":", alpha=0.4)

    # Formatting — residual panel
    ax_res.axhline(0, color="k", lw=0.8)
    ax_res.set_ylabel(r"$h_\mathrm{mod} - h_\mathrm{GR}$  (norm.)", fontsize=9)
    ax_res.set_xlabel("Time relative to merger (ms)", fontsize=9)
    ax_res.grid(True, ls=":", alpha=0.4)

# Annotation confirming the two columns should be identical
axes[0, 1].text(
    0.98, 0.05,
    "Should match MG column exactly\n(LV term = 0 when A = 0)",
    transform=axes[0, 1].transAxes,
    ha="right", va="bottom", fontsize=7.5,
    color="grey", style="italic",
)

fig.tight_layout()
out_path = Path(__file__).parent / "mg_waveforms.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved:  {out_path}")
