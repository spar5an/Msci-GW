"""
plot_mg_waveforms.py — Visualise MG and LV(A=0) waveforms vs. GR for varying m_g.

Two side-by-side comparisons:
  Left  — Massive Graviton (MG):   phase correction using _additional_phase only.
  Right — Lorentz Violation (LV) with A = inf  (LV term suppressed, MG term
          only):  identical physics to MG, plotted independently as a cross-check.

Expected result: MG and LV(A=inf) waveforms are bit-for-bit identical for the same
m_g, demonstrating that the LV generator reduces correctly to the MG limit.

Saves:  mg_waveforms.png  in the same directory as this script.

Run with:
    python plot_mg_waveforms.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycbc.waveform import get_fd_waveform

# ── Import all physics and pipeline helpers from gw_datagen ──────────────────
_DATA_GEN_DIR = str(
    Path(__file__).resolve().parent.parent.parent
    / "HPC" / "Pipeline" / "Data Generation"
)
sys.path.insert(0, _DATA_GEN_DIR)

from gw_datagen import (
    _C, _M_SUN_SEC,
    _D_alpha,
    _additional_phase,
    _additional_phase_lv,
    _m_g_to_lambda_g,
    _fd_to_td_polarisations,
    _N_RINGDOWN,
    _HIGHPASS_FC,
)


def _chirp_mass(m1, m2):
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


# ── Waveform parameters ───────────────────────────────────────────────────────
M1, M2   = 30.0, 30.0   # solar masses
Z        = 0.1
DIST     = 400.0         # Mpc
F_LOWER  = 30.0          # Hz
F_FINAL  = 512.0         # Hz
DELTA_F  = 1.0 / 256     # Hz — matches gw_datagen internal delta_f
ALPHA_LV = 3.0           # DSR exponent

DT            = 1.0 / (2.0 * F_FINAL)   # sample interval (s)
TARGET_LENGTH = int(round(1.0 / (DELTA_F * DT)))  # N = 1/(delta_f * dt)
MERGER_IDX    = TARGET_LENGTH - _N_RINGDOWN        # sample index of coalescence

MC = _chirp_mass(M1, M2)

# m_g values (kg) chosen so delta_Psi(f_lower) is O(0.1)–O(1) rad
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

hp_gr_arr = hp_gr_fd.numpy()
hp_gr_amp = np.abs(hp_gr_arr[1:])
f_c       = float(np.max(freqs_pos[np.nonzero(hp_gr_amp)]))

# Dummy hc (zero) — only hp needed for scalar phase-shift plots
hc_gr_arr = np.zeros_like(hp_gr_arr)

gr_hp_ts, _ = _fd_to_td_polarisations(
    hp_gr_arr, hc_gr_arr, DELTA_F, TARGET_LENGTH, DT, _HIGHPASS_FC)
gr_td = gr_hp_ts.numpy()

# Time axis relative to merger (seconds)
t_axis_full = (np.arange(TARGET_LENGTH) - MERGER_IDX) * DT

# Display window: 0.5 s before merger, 0.15 s after
i_start = max(0, MERGER_IDX - int(0.5 / DT))
i_end   = min(TARGET_LENGTH, MERGER_IDX + int(0.15 / DT))
t_axis  = t_axis_full[i_start:i_end]
gr_td_trim = gr_td[i_start:i_end]


def _modified_td(phase_shift_arr):
    """Apply a phase shift to hp_gr_arr and return the trimmed TD waveform."""
    hp_mod = hp_gr_arr.copy()
    hp_mod[pos] *= np.exp(1j * phase_shift_arr)
    hp_ts, _ = _fd_to_td_polarisations(
        hp_mod, hc_gr_arr, DELTA_F, TARGET_LENGTH, DT, _HIGHPASS_FC)
    return hp_ts.numpy()[i_start:i_end]


# ── Build modified waveforms ──────────────────────────────────────────────────
print("Applying MG / LV phase shifts…")
mg_td_list = []
lv_td_list = []

for m_g in M_G_VALUES:
    lg = _m_g_to_lambda_g(m_g)
    f_pn = 0.1 / (np.pi * (M1 + M2) * _M_SUN_SEC)

    dpsi_mg = _additional_phase(freqs_pos, MC, Z, lg, f_pn_cutoff=f_pn)
    dpsi_lv = _additional_phase_lv(freqs_pos, MC, Z, lg, ALPHA_LV, np.inf, f_c,
                                    f_pn_cutoff=f_pn)

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

norm = float(np.max(np.abs(gr_td_trim)))

for col, (td_list, title) in enumerate([
    (mg_td_list, "Massive Graviton (MG)"),
    (lv_td_list, rf"LV with $A = \infty$ (mass term only, $\alpha = {ALPHA_LV}$)"),
]):
    ax_wave = axes[0, col]
    ax_res  = axes[1, col]

    ax_wave.plot(t_axis * 1e3, gr_td_trim / norm,
                 color=GR_COLOR, lw=1.8, label="GR", zorder=4)

    for m_g, td, color in zip(M_G_VALUES, td_list, COLORS):
        lg_m  = _m_g_to_lambda_g(m_g)
        label = rf"$m_g = {m_g:.0e}$ kg  ($\lambda_g = {lg_m:.1e}$ m)"
        ax_wave.plot(t_axis * 1e3, td / norm,
                     color=color, lw=1.2, alpha=0.85, label=label)
        ax_res.plot(t_axis * 1e3, (td - gr_td_trim) / norm,
                    color=color, lw=1.0, alpha=0.85)

    ax_wave.axhline(0, color="k", lw=0.4, ls=":")
    ax_wave.set_title(title, fontsize=11)
    ax_wave.set_ylabel("Strain  (normalised)", fontsize=9)
    ax_wave.legend(fontsize=7.5, loc="upper left", framealpha=0.8)
    ax_wave.set_xlim(t_axis[0] * 1e3, t_axis[-1] * 1e3)
    ax_wave.grid(True, ls=":", alpha=0.4)

    ax_res.axhline(0, color="k", lw=0.8)
    ax_res.set_ylabel(r"$h_\mathrm{mod} - h_\mathrm{GR}$  (norm.)", fontsize=9)
    ax_res.set_xlabel("Time relative to merger (ms)", fontsize=9)
    ax_res.grid(True, ls=":", alpha=0.4)

axes[0, 1].text(
    0.98, 0.05,
    "Should match MG column exactly\n(LV term = 0 when A = inf)",
    transform=axes[0, 1].transAxes,
    ha="right", va="bottom", fontsize=7.5,
    color="grey", style="italic",
)

fig.tight_layout()
out_path = Path(__file__).parent / "mg_waveforms.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved:  {out_path}")
