"""
test_cutoff_comparison.py

Compare the massive-graviton and LV phase modifications computed with two
different reference (normalisation) frequencies f_c:

  (A) POST-NEWTONIAN CUTOFF  — f_c from the condition (m1+m2)·omega = 0.1
      (computed via inspiral_cutoff_frequency).  This is the frequency at
      which the PN inspiral approximation breaks down.  The phase is
      pinned to zero here.

  (B) MERGER FREQUENCY       — f_c taken as the highest frequency at which
      the FD waveform amplitude is non-zero (waveform amplitude peak).
      This is what the rest of the pipeline currently uses.

Produces 4 subplots (2 × 2):

  [0,0]  Massive-graviton δΨ(f)  –  f_c = PN cutoff   (M·ω=0.1)
  [0,1]  Massive-graviton δΨ(f)  –  f_c = merger freq
  [1,0]  LV α=3 δΨ(f)           –  f_c = PN cutoff   (M·ω=0.1)
  [1,1]  LV α=3 δΨ(f)           –  f_c = merger freq

In each panel a scan over several parameter values is shown so the
shape change introduced by the choice of f_c is visible.

Physics note
------------
f_c is the reference frequency at which the phase correction is defined
to be zero.  Both normalisation conventions are equally valid (they differ
by an unobservable global phase), but the choice affects the *shape* of
δΨ(f) over the observable band — particularly at frequencies well below f_c.
Using the PN cutoff (~108 Hz for 30+30 M☉) keeps the correction small
throughout the band; using the merger frequency (~677 Hz) allows the
correction to grow more at low frequencies.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from JHPY import _additional_phase, _additional_phase_lv, _D_alpha
from pycbc.waveform import get_fd_waveform
from inspiral_freq import inspiral_cutoff_frequency

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'test_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Waveform parameters ───────────────────────────────────────────────────────
M1, M2 = 30.0, 30.0
Z      = 0.1
CHIRP_MASS = (M1 * M2)**(3/5) / (M1 + M2)**(1/5)
F_LOWER    = 30.0
F_FINAL    = 2048.0

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if not passed:
        FAIL += 1
    else:
        PASS += 1
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


# ── Reference frequencies ─────────────────────────────────────────────────────
def get_reference_frequencies():
    """Return (f_c_pn, f_c_merger, freqs, band).

    f_c_pn     : PN breakdown frequency from M·omega = 0.1
    f_c_merger : highest frequency with non-zero FD waveform amplitude
    freqs      : full frequency array from the FD waveform
    band       : boolean mask for [F_LOWER, F_FINAL]
    """
    # PN cutoff from inspiral_freq.py
    f_c_pn = inspiral_cutoff_frequency(M1, M2, M_omega=0.1)

    # Merger frequency from waveform amplitude
    hp_fd, _ = get_fd_waveform(
        approximant='IMRPhenomD', mass1=M1, mass2=M2,
        delta_f=1.0/256, f_lower=F_LOWER, f_final=F_FINAL, distance=410.0
    )
    freqs = hp_fd.sample_frequencies.numpy()[1:]
    amp   = np.abs(hp_fd.numpy()[1:])
    f_c_merger = float(np.max(freqs[np.nonzero(amp)]))

    band = (freqs >= F_LOWER) & (freqs <= F_FINAL)
    return f_c_pn, f_c_merger, freqs, band


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_pn_cutoff_within_band(f_c_pn, f_c_merger):
    """PN cutoff should be between F_LOWER and the merger frequency."""
    print("\n=== Test 1: PN cutoff lies within the observable band ===")
    in_band = F_LOWER < f_c_pn < f_c_merger
    report(
        f"F_LOWER={F_LOWER} < f_c_pn={f_c_pn:.1f} < f_c_merger={f_c_merger:.1f}",
        in_band,
        f"f_c_pn={f_c_pn:.2f} Hz, f_c_merger={f_c_merger:.2f} Hz"
    )


def test_lv_phase_at_fc_is_zero(freqs, band, f_c_pn, f_c_merger):
    """The LV term is normalised to zero at f_c by construction (u^{α-1} − u_c^{α-1}).
    The massive-graviton term uses paper-derived constant terms and is NOT zero at f_c.
    """
    print("\n=== Test 2: LV phase is zero at f_c; mg phase is NOT (different conventions) ===")
    f = freqs[band]
    A_lv = 3.022e-16   # α=3 reference

    # LV: must be zero at f_c for both choices.
    # Use tolerance 1e-6: the nearest frequency bin may not land exactly on f_c
    # (Δf = 1/256 Hz), introducing small residuals from the u^{α-1} subtraction.
    for label, f_c in [("PN cutoff", f_c_pn), ("merger", f_c_merger)]:
        ps   = _additional_phase_lv(f, CHIRP_MASS, Z, np.inf, 3.0, A_lv, f_c)
        i_fc = np.argmin(np.abs(f - f_c))
        val  = abs(ps[i_fc])
        report(
            f"LV δΨ(f_c) ≈ 0  [f_c = {label}, {f_c:.1f} Hz]",
            val < 1e-6,
            f"δΨ(f_c)={val:.2e}"
        )

    # mg: should NOT be zero at f_c (the formula uses an independent constant)
    lambda_g = 1e16
    for label, f_c in [("PN cutoff", f_c_pn), ("merger", f_c_merger)]:
        ps   = _additional_phase_lv(f, CHIRP_MASS, Z, lambda_g, 3.0, np.inf, f_c)
        i_fc = np.argmin(np.abs(f - f_c))
        val  = abs(ps[i_fc])
        report(
            f"mg δΨ(f_c) ≠ 0  [f_c = {label}, {f_c:.1f} Hz]  (expected non-zero)",
            val > 1e-3,
            f"δΨ(f_c)={val:.4e}"
        )


def test_phase_difference_at_low_freq(freqs, band, f_c_pn, f_c_merger):
    """The two conventions give different absolute phases but the same
    frequency *difference* (since they differ by a global constant)."""
    print("\n=== Test 3: Frequency-difference observables are convention-independent ===")
    f = freqs[band]
    lambda_g = 1e16

    i1 = np.argmin(np.abs(f - 50.0))
    i2 = np.argmin(np.abs(f - 200.0))

    for label, (mod, kwargs) in [
        ("mg",   (_additional_phase_lv,
                  dict(chirp_mass=CHIRP_MASS, z=Z, lambda_g=lambda_g,
                       alpha_lv=3.0, A_lv=np.inf))),
        ("LV α=3", (_additional_phase_lv,
                    dict(chirp_mass=CHIRP_MASS, z=Z, lambda_g=np.inf,
                         alpha_lv=3.0, A_lv=3.022e-16))),
    ]:
        ps_pn     = mod(f, **kwargs, f_c=f_c_pn)
        ps_merger = mod(f, **kwargs, f_c=f_c_merger)

        diff_pn     = ps_pn[i1]     - ps_pn[i2]
        diff_merger = ps_merger[i1] - ps_merger[i2]
        residual    = abs(diff_pn - diff_merger)

        # The two conventions should give the same Δφ(50→200 Hz)
        report(
            f"{label}: Δφ(50→200 Hz) same for both f_c conventions",
            residual < 1e-12,
            f"Δφ_PN={diff_pn:.6e}, Δφ_merger={diff_merger:.6e}, diff={residual:.2e}"
        )

    # LV absolute values SHOULD differ between the two f_c choices
    # (different global phase offset, even though Δφ is the same)
    A_lv = 3.022e-16
    ps_pn     = _additional_phase_lv(f, CHIRP_MASS, Z, np.inf, 3.0, A_lv, f_c_pn)
    ps_merger = _additional_phase_lv(f, CHIRP_MASS, Z, np.inf, 3.0, A_lv, f_c_merger)
    abs_diff  = abs(ps_pn[i1] - ps_merger[i1])
    report(
        "LV: absolute δΨ(50 Hz) DIFFERS between f_c conventions (expected)",
        abs_diff > 1e-6,
        f"δΨ_PN={ps_pn[i1]:.4e}, δΨ_merger={ps_merger[i1]:.4e}, diff={abs_diff:.2e}"
    )


# ── 4-panel plot ──────────────────────────────────────────────────────────────
def plot_cutoff_comparison(freqs, band, f_c_pn, f_c_merger):
    """4-panel comparison: mg and LV phases with PN-cutoff vs merger-freq f_c.

    Layout (2×2):
      [0,0] mg,   f_c = PN cutoff (M·ω=0.1)
      [0,1] mg,   f_c = merger frequency
      [1,0] LV α=3, f_c = PN cutoff
      [1,1] LV α=3, f_c = merger frequency
    """
    print("\n=== Generating 4-panel cutoff comparison plot ===")
    f = freqs[band]

    # Parameter scan values
    lambda_g_values = [5e15, 1e16, 5e16, 1e17]
    lg_colors       = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple']

    # A_lv values chosen so |ζ| ≈ [2, 5, 10, 20] at mid-band (α=3, ζ∝A)
    A_BASE = 3.022e-16   # |ζ|≈0.5
    A_lv_values  = [A_BASE * f for f in [4, 11, 22, 44]]
    zeta_approx  = [0.5 * f for f in [4, 11, 22, 44]]
    alv_colors   = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

    fc_configs = [
        (f_c_pn,     fr'$f_c$ = PN cutoff  ($M\omega=0.1$,  $f_c={f_c_pn:.1f}$ Hz)'),
        (f_c_merger, fr'$f_c$ = merger  ($f_c={f_c_merger:.1f}$ Hz)'),
    ]

    # ── Row 0: massive graviton ───────────────────────────────────────────────
    for col, (f_c, fc_label) in enumerate(fc_configs):
        ax = axes[0, col]
        for lg, col_c in zip(lambda_g_values, lg_colors):
            ps = _additional_phase_lv(f, CHIRP_MASS, Z, lg, 3.0, np.inf, f_c)
            ax.plot(f, ps, color=col_c, linewidth=1.6,
                    label=fr'$\lambda_g = {lg:.0e}$ m')

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--',
                   label='GR (baseline)')
        ax.axvline(f_c, color='grey', linewidth=1.0, linestyle=':',
                   label=f'$f_c={f_c:.1f}$ Hz')
        ax.set_xscale('log')
        ax.set_ylabel(r'$\delta\Psi(f)$ (rad)', fontsize=11)
        ax.set_title(f'Massive graviton — {fc_label}', fontsize=11)
        ax.legend(fontsize=8.5, loc='lower right')
        ax.grid(True, alpha=0.3)

    # ── Row 1: LV α=3 ────────────────────────────────────────────────────────
    for col, (f_c, fc_label) in enumerate(fc_configs):
        ax = axes[1, col]
        for A, col_c, zeta_est in zip(A_lv_values, alv_colors, zeta_approx):
            ps = _additional_phase_lv(f, CHIRP_MASS, Z, np.inf, 3.0, A, f_c)
            ax.plot(f, ps, color=col_c, linewidth=1.6,
                    label=fr'$A={A:.2e}$ m  ($|\zeta|\approx{zeta_est:.0f}$)')

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--',
                   label='GR (baseline)')
        ax.axvline(f_c, color='grey', linewidth=1.0, linestyle=':',
                   label=f'$f_c={f_c:.1f}$ Hz')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel(r'$\delta\Psi(f)$ (rad)', fontsize=11)
        ax.set_title(fr'LV $\alpha=3$ — {fc_label}', fontsize=11)
        ax.legend(fontsize=8.5, loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Effect of Normalisation Frequency $f_c$ on Phase Corrections\n'
        r'Left: $f_c = (m_1+m_2)\omega = 0.1$ (PN cutoff) '
        r'— Right: $f_c$ = merger amplitude peak',
        fontsize=13
    )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'cutoff_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── TD waveform helper ───────────────────────────────────────────────────────
_TARGET   = 8192   # output samples  (matches JHPY default)
_N_RING   = 500    # ringdown samples taken from start of IRFFT output

def _make_td_waveform(f_c, lambda_g, alpha_lv, A_lv, delta_f=1.0/256):
    """Return (time_array, h_plus) by applying a phase shift to the FD waveform
    and IFFTing, replicating the pipeline in JHPY._generate_single_lv_waveform
    but with an explicit f_c rather than the internal amplitude-derived value.

    The IRFFT of an FD waveform wraps cyclically: the ringdown appears in the
    first ~500 samples of h_raw, not after the amplitude peak.  This function
    reassembles the array the same way JHPY does:

        h_arr = [h_raw[N - n_pre : N],   ← inspiral + merger (end of IRFFT)
                 h_raw[0 : n_ringdown]]   ← ringdown          (start of IRFFT)

    The time axis is then set so t = 0 at the amplitude peak (merger).

    Parameters
    ----------
    f_c      : float  — normalisation frequency [Hz]
    lambda_g : float  — graviton Compton wavelength [m] (np.inf for GR/LV-only)
    alpha_lv : float  — LV dispersion exponent
    A_lv     : float  — LV Compton wavelength [m] (np.inf for mg-only/GR)
    delta_f  : float  — frequency resolution [Hz]

    Returns
    -------
    t  : ndarray  — time relative to merger peak [s]
    h  : ndarray  — h_plus strain (length _TARGET)
    """
    hp_fd, _ = get_fd_waveform(
        approximant='IMRPhenomD', mass1=M1, mass2=M2,
        delta_f=delta_f, f_lower=F_LOWER, f_final=F_FINAL, distance=410.0
    )
    freqs = hp_fd.sample_frequencies.numpy()[1:]

    chirp_mass = (M1 * M2)**(3/5) / (M1 + M2)**(1/5)
    phase_shift = _additional_phase_lv(freqs, chirp_mass, Z,
                                        lambda_g, alpha_lv, A_lv, f_c)

    hp_array = hp_fd.numpy().copy()
    hp_array[1:] *= np.exp(1j * phase_shift)

    h_raw = np.fft.irfft(hp_array)
    N     = len(h_raw)
    h_raw *= delta_f * N          # normalise (matches JHPY convention)

    # Reassemble: inspiral/merger from tail of h_raw, ringdown from head
    n_pre = _TARGET - _N_RING
    h_arr = np.concatenate([h_raw[N - n_pre:], h_raw[:_N_RING]])

    dt   = 1.0 / (2 * freqs[-1])  # Nyquist time step
    time = np.arange(_TARGET) * dt

    # Shift so t=0 at amplitude peak (merger)
    peak = np.argmax(np.abs(h_arr))
    time = time - time[peak]
    return time, h_arr


# ── TD waveform plot (4 panels) ───────────────────────────────────────────────
def plot_td_waveforms(f_c_pn, f_c_merger):
    """4-panel time-domain comparison: mg and LV with PN-cutoff vs merger f_c.

    Layout (2 × 2):
      [0,0] Massive graviton, f_c = PN cutoff  — GR + modified overlaid
      [0,1] Massive graviton, f_c = merger freq — GR + modified overlaid
      [1,0] LV α=3,           f_c = PN cutoff
      [1,1] LV α=3,           f_c = merger freq

    A large LV value (|ζ|≈11) is used so the phase difference is visible
    in the time domain.  The inspiral window (last ~1 s before merger) is
    shown so the phase accumulation is visible; near the merger the LV
    correction → 0 by construction when f_c = merger.
    """
    print("\n=== Generating 4-panel TD waveform plot ===")

    # Use a physically significant λ_g and a strong LV signal
    LAMBDA_G = 1e16           # m
    A_LV     = 3.022e-16 * 22  # |ζ|≈11, gives ~1 rad phase shift — TD-visible

    # GR baseline (f_c choice irrelevant — phase shift is zero)
    t_gr, h_gr = _make_td_waveform(f_c_merger, np.inf, 3.0, np.inf)

    # Window: inspiral + merger + ringdown.
    # _N_RING=500 samples at dt≈0.244 ms gives ~0.122 s of post-merger data.
    WIN_PRE  = 1.5                         # seconds before merger
    WIN_POST = _N_RING * (1.0 / (2 * F_FINAL))  # all available ringdown samples
    mask = (t_gr >= -WIN_PRE) & (t_gr <= WIN_POST)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey='row')

    fc_configs = [
        (f_c_pn,     fr'$f_c$ = PN cutoff  ({f_c_pn:.1f} Hz)',     'tab:blue'),
        (f_c_merger, fr'$f_c$ = merger  ({f_c_merger:.1f} Hz)',     'tab:red'),
    ]

    # ── Row 0: massive graviton ───────────────────────────────────────────────
    for col, (f_c, fc_label, col_c) in enumerate(fc_configs):
        ax = axes[0, col]
        t_mg, h_mg = _make_td_waveform(f_c, LAMBDA_G, 3.0, np.inf)

        ax.plot(t_gr[mask], h_gr[mask],
                color='black', linewidth=1.2, alpha=0.7, label='GR')
        ax.plot(t_mg[mask], h_mg[mask],
                color=col_c, linewidth=1.4, linestyle='--',
                label=fr'mg: $\lambda_g=10^{{16}}$ m')

        ax.axvline(0, color='grey', linewidth=0.8, linestyle=':', alpha=0.6)
        ax.set_ylabel('Strain', fontsize=11)
        ax.set_title(f'Massive graviton — {fc_label}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── Row 1: LV α=3 ────────────────────────────────────────────────────────
    for col, (f_c, fc_label, col_c) in enumerate(fc_configs):
        ax = axes[1, col]
        t_lv, h_lv = _make_td_waveform(f_c, np.inf, 3.0, A_LV)

        ax.plot(t_gr[mask], h_gr[mask],
                color='black', linewidth=1.2, alpha=0.7, label='GR')
        ax.plot(t_lv[mask], h_lv[mask],
                color=col_c, linewidth=1.4, linestyle='--',
                label=fr'LV $\alpha=3$: $A={A_LV:.2e}$ m  ($|\zeta|\approx11$)')

        ax.axvline(0, color='grey', linewidth=0.8, linestyle=':', alpha=0.6,
                   label='merger (t=0)')
        ax.set_xlabel('Time relative to merger (s)', fontsize=11)
        ax.set_ylabel('Strain', fontsize=11)
        ax.set_title(fr'LV $\alpha=3$ — {fc_label}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Time-Domain Waveforms: Effect of Normalisation Frequency $f_c$\n'
        r'Left: $f_c = (m_1+m_2)\omega=0.1$ (PN cutoff, 107.7 Hz) — '
        r'Right: $f_c$ = merger (676.7 Hz)  |  30+30 $M_\odot$, $z=0.1$',
        fontsize=13
    )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'cutoff_td_waveforms.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("Frequency Cutoff Comparison Tests (PN cutoff vs merger freq)")
    print("=" * 65)

    f_c_pn, f_c_merger, freqs, band = get_reference_frequencies()

    print(f"\n  PN cutoff frequency    : {f_c_pn:.2f} Hz  (M·omega=0.1, M={M1+M2:.0f} Msun)")
    print(f"  Merger frequency       : {f_c_merger:.2f} Hz  (waveform amplitude peak)")
    print(f"  Observable band        : {F_LOWER}–{F_FINAL} Hz")

    test_pn_cutoff_within_band(f_c_pn, f_c_merger)
    test_lv_phase_at_fc_is_zero(freqs, band, f_c_pn, f_c_merger)
    test_phase_difference_at_low_freq(freqs, band, f_c_pn, f_c_merger)

    print("\n=== Generating 4-panel FD phase plot ===")
    plot_cutoff_comparison(freqs, band, f_c_pn, f_c_merger)

    print("\n=== Generating 4-panel TD waveform plot ===")
    plot_td_waveforms(f_c_pn, f_c_merger)

    print("\n" + "=" * 65)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 65)

    if FAIL > 0:
        sys.exit(1)
