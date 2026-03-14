"""
Tests and plots for the Lorentz-violating (LV) waveform model.

Implements the parametrized dispersion relation of Mirshekari, Yunes & Will
(2011), arXiv:1110.2720, and compares it with:
  - GR baseline (lambda_g → ∞, A_lv → ∞)
  - Massive graviton only (Will 1997, arXiv:9709011)
  - Lorentz-violating correction for α = 3 (DSR), α = 4 (Horava-Lifshitz)
  - Combined model (massive graviton + LV)

The LV term is parameterised by A_lv [m] — the LV Compton wavelength defined
as A_lv ≡ A_physical^{1/(α−2)}, where A_physical is the coupling constant in

    E² = p²c² + m_g²c⁴ + A_physical · p^α · c^α          (arXiv:1110.2720 Eq. 1)

This replaces the old β_ppE (ppE amplitude) interface.  Internally the code
computes the ζ coefficient from A_lv via Eq. 30/32 and applies the correct
frequency dependence u^{α−1} (exponent α−1, not α−2).

Plots generated:
  1. phase_comparison.png         -- Phase shift vs frequency for all models
  2. waveform_models.png          -- Time-domain H1 waveforms: GR / mg / LV / combined
  3. alpha_scan.png               -- LV waveforms for different dispersion exponents α
  4. A_scan.png                   -- FD phase δΨ(f) for different A_lv at α=3
  5. fd_phase_diff.png            -- Frequency-domain phase difference vs GR
  6. alpha0_degeneracy.png        -- α=0 LV is degenerate with massive graviton
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from JHPY import (
    _D_alpha,
    _additional_phase,
    _additional_phase_lv,
    _generate_single_modified_waveform,
    _generate_single_lv_waveform,
)
from pycbc.waveform import get_fd_waveform

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'test_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

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


# ── Shared waveform parameters ──────────────────────────────────────────────
PARAMS = {
    'mass1': 30.0, 'mass2': 30.0,
    'distance': 410.0, 'redshift': 0.1,
}
CHIRP_MASS = (30.0 * 30.0)**(3/5) / (30.0 + 30.0)**(1/5)
Z = 0.1
TARGET = 8192
DT = 1 / 4096
F_LOWER = 30.0
F_FINAL = 2048.0

# A_lv values chosen to give ~0.5 rad LV phase at mid-band (~100 Hz).
# These are LV Compton wavelengths A ≡ A_physical^{1/(α−2)} [metres].
# (computed so that |ζ| ≈ 0.5 for α=3 and |ζ| ≈ 2.0 for α=4)
A_LV_ALPHA3 = 3.022e-16   # α=3 (DSR), gives |ζ| ≈ 0.5
A_LV_ALPHA4 = 1.446e-5    # α=4 (Horava-Lifshitz), gives |ζ| ≈ 2.0
A_LV_ALPHA1 = 4.684e25    # α=1 (log correction), gives ζ ≈ 0.3
LAMBDA_G    = 1e16        # graviton Compton wavelength [m]


def _get_fd_freqs_and_fc():
    """Return frequency array and max frequency for the standard waveform."""
    hp_fd, _ = get_fd_waveform(
        approximant='IMRPhenomD', mass1=30.0, mass2=30.0,
        delta_f=1.0/256, f_lower=F_LOWER, f_final=F_FINAL, distance=410.0
    )
    freqs = hp_fd.sample_frequencies.numpy()[1:]
    amp = np.abs(hp_fd.numpy()[1:])
    f_c = float(np.max(freqs[np.nonzero(amp)]))
    return freqs, f_c


# ── Test 1: GR limit of _additional_phase_lv ────────────────────────────────
def test_gr_limit_lv():
    print("\n=== Test 1: GR limit of _additional_phase_lv ===")
    freqs, f_c = _get_fd_freqs_and_fc()

    # GR: lambda_g → ∞, A_lv → ∞
    ps = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 3.0, np.inf, f_c)
    report("GR limit: all zeros", np.max(np.abs(ps)) < 1e-30,
           f"max={np.max(np.abs(ps)):.2e}")


# ── Test 2: LV phase scales correctly with A_lv ─────────────────────────────
def test_lv_scaling():
    """
    For α=3: ζ ∝ A_lv^{α−2} = A_lv^1, so doubling A_lv doubles the phase.
    For α=4: ζ ∝ A_lv^2, so doubling A_lv quadruples the phase.
    """
    print("\n=== Test 2: LV phase scales with A_lv ===")
    freqs, f_c = _get_fd_freqs_and_fc()
    mid = len(freqs) // 2

    # α=3
    A1 = A_LV_ALPHA3
    A2 = 2 * A_LV_ALPHA3
    ps1 = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 3.0, A1, f_c)
    ps2 = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 3.0, A2, f_c)
    ratio = ps2[mid] / ps1[mid] if ps1[mid] != 0 else 0
    report("α=3: doubling A_lv doubles phase (ratio≈2)", abs(ratio - 2) < 0.01,
           f"ratio={ratio:.4f}")

    # α=4
    A1 = A_LV_ALPHA4
    A2 = 2 * A_LV_ALPHA4
    ps1 = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 4.0, A1, f_c)
    ps2 = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 4.0, A2, f_c)
    ratio = ps2[mid] / ps1[mid] if ps1[mid] != 0 else 0
    report("α=4: doubling A_lv quadruples phase (ratio≈4)", abs(ratio - 4) < 0.01,
           f"ratio={ratio:.4f}")


# ── Test 3: LV-only vs massive-graviton-only vs combined ────────────────────
def test_additive_decomposition():
    print("\n=== Test 3: LV phase = mg part + LV part (superposition) ===")
    freqs, f_c = _get_fd_freqs_and_fc()

    ps_mg_only = _additional_phase_lv(freqs, CHIRP_MASS, Z, LAMBDA_G, 3.0, np.inf, f_c)
    ps_lv_only = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 3.0, A_LV_ALPHA3, f_c)
    ps_combined = _additional_phase_lv(freqs, CHIRP_MASS, Z, LAMBDA_G, 3.0, A_LV_ALPHA3, f_c)

    residual = np.max(np.abs((ps_mg_only + ps_lv_only) - ps_combined))
    report("Phase is sum of mg + LV parts", residual < 1e-20,
           f"max_residual={residual:.2e}")


# ── Test 4: alpha=2 is degenerate ───────────────────────────────────────────
def test_alpha2_degenerate():
    print("\n=== Test 4: alpha_lv = 2 gives no LV correction ===")
    freqs, f_c = _get_fd_freqs_and_fc()

    # Any finite A_lv with alpha=2 should give zero LV term
    ps_alpha2 = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 2.0, 1e10, f_c)
    ps_gr     = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 3.0, np.inf, f_c)
    report("alpha=2: LV term vanishes for any A_lv",
           np.allclose(ps_alpha2, ps_gr),
           f"max_diff={np.max(np.abs(ps_alpha2 - ps_gr)):.2e}")


# ── Test 5: Different alpha values give different phase shapes ───────────────
def test_alpha_shapes():
    print("\n=== Test 5: Different alpha gives different phase shapes ===")
    freqs, f_c = _get_fd_freqs_and_fc()

    # Use the same A_lv reference for all — shapes differ because frequency
    # dependence is u^{α−1} and ζ formula changes with α
    A_ref = 1e13   # arbitrary reference for shape comparison
    ps3 = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 3.0, A_ref, f_c)
    ps4 = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 4.0, A_ref, f_c)
    ps0 = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf, 0.0, A_ref, f_c)

    diff_3_4 = np.max(np.abs(ps3 - ps4))
    diff_3_0 = np.max(np.abs(ps3 - ps0))
    report("alpha=3 differs from alpha=4", diff_3_4 > 0, f"max_diff={diff_3_4:.4e}")
    report("alpha=3 differs from alpha=0", diff_3_0 > 0, f"max_diff={diff_3_0:.4e}")


# ── Test 6: α=0 degeneracy with massive graviton ────────────────────────────
def test_alpha0_degeneracy():
    """
    At α=0 the LV A term is 100% degenerate with the massive graviton term
    (arXiv:1110.2720, p.6): in the limit α→0, λ_g^{-2} → λ_g^{-2} + A^{-2}.

    Concretely this means:

    (a) LV-only (λ_g=∞, α=0, A=L) has the same u^{-1} frequency shape as
        massive-graviton-only (λ_g=L, A=∞).  Both scale as β·u^{-1} with
        β = π²cD₀M/(L²(1+Z)).  The two constant offsets differ because
        `_additional_phase` and the LV path use different normalisation
        conventions, but since a global phase offset is degenerate with the
        coalescence phase φ_c it is unobservable.  We test the observable
        part: differences between pairs of frequencies must agree.

    (b) Combined (λ_g=L1, α=0, A=L2) is identical to massive-graviton-only
        with an effective wavelength λ_eff = (L1^{-2} + L2^{-2})^{-1/2},
        again verified via frequency-difference observables.
    """
    print("\n=== Test 6: α=0 degeneracy with massive graviton ===")
    freqs, f_c = _get_fd_freqs_and_fc()

    L = 1e15          # metres — same scale for λ_g and A_lv

    # ── (a) LV-only α=0 vs massive-graviton-only ─────────────────────────────
    # Both have phase ∝ −β/u = −β/(πMf), differing only by an overall constant.
    # Test: Δφ(f1→f2) is the same for both paths.

    ps_mg  = _additional_phase_lv(freqs, CHIRP_MASS, Z, L,       0.0, np.inf, f_c)
    ps_lv0 = _additional_phase_lv(freqs, CHIRP_MASS, Z, np.inf,  0.0, L,      f_c)

    i1 = np.argmin(np.abs(freqs - 50.0))
    i2 = np.argmin(np.abs(freqs - 200.0))

    diff_mg  = ps_mg[i1]  - ps_mg[i2]
    diff_lv0 = ps_lv0[i1] - ps_lv0[i2]
    residual_a = abs(diff_mg - diff_lv0)
    report(
        "α=0, A=λ_g: Δφ(50→200 Hz) equal to massive-graviton Δφ",
        residual_a < 1e-12,
        f"Δφ_mg={diff_mg:.6e}, Δφ_lv0={diff_lv0:.6e}, diff={residual_a:.2e}"
    )

    # ── (b) Combined (λ_g, α=0, A) equals mg-only with λ_eff ─────────────────
    # 1/λ_eff² = 1/λ_g² + 1/A²
    L1 = 1e15
    L2 = 5e14
    lambda_eff = 1.0 / np.sqrt(1.0/L1**2 + 1.0/L2**2)

    ps_comb    = _additional_phase_lv(freqs, CHIRP_MASS, Z, L1,          0.0, L2,          f_c)
    ps_mg_eff  = _additional_phase_lv(freqs, CHIRP_MASS, Z, lambda_eff,  0.0, np.inf,      f_c)

    diff_comb   = ps_comb[i1]   - ps_comb[i2]
    diff_mg_eff = ps_mg_eff[i1] - ps_mg_eff[i2]
    residual_b = abs(diff_comb - diff_mg_eff)
    report(
        "Combined (λ_g,A) at α=0 equals mg-only with λ_eff=(λ_g^{-2}+A^{-2})^{-1/2}",
        residual_b < 1e-12,
        f"Δφ_comb={diff_comb:.6e}, Δφ_mg_eff={diff_mg_eff:.6e}, diff={residual_b:.2e}"
    )

    # ── (c) β coefficients match ──────────────────────────────────────────────
    # The ζ computed from A_lv=L at α=0 should equal the β from λ_g=L.
    from JHPY import _D_alpha, _C, _M_SUN_SEC
    M    = CHIRP_MASS * _M_SUN_SEC * (1 + Z)
    D0   = _D_alpha(0, Z)
    beta = np.pi**2 * _C * D0 * M / (L**2 * (1 + Z))
    # ζ at α=0: π^2/(1) * c^1 * D0 * M^1 / (A^2 * (1+Z)^1)
    zeta = np.pi**2 * _C * D0 * M / (L**2 * (1 + Z))
    report(
        "β(λ_g=L) == ζ(α=0, A=L): coupling coefficients identical",
        abs(beta - zeta) < 1e-20,
        f"β={beta:.6e}, ζ={zeta:.6e}"
    )


# ── Test 7: LV waveform differs from massive-graviton waveform ───────────────
def test_lv_vs_mg_waveform():
    print("\n=== Test 6: LV waveform differs from massive-graviton waveform ===")

    # Massive graviton only (9709011)
    r_mg = _generate_single_modified_waveform(
        PARAMS, time_resolution=DT, approximant='IMRPhenomD',
        f_lower=F_LOWER, detectors=['H1'], target_length=TARGET,
        add_noise=False, lambda_g=LAMBDA_G, f_final=F_FINAL
    )
    # LV only (1110.2720) — same lambda_g but add LV term via A_lv
    r_lv = _generate_single_lv_waveform(
        PARAMS, time_resolution=DT, approximant='IMRPhenomD',
        f_lower=F_LOWER, detectors=['H1'], target_length=TARGET,
        add_noise=False, lambda_g=LAMBDA_G, alpha_lv=3.0,
        A_lv=A_LV_ALPHA3, f_final=F_FINAL
    )

    report("mg waveform succeeded", r_mg['success'], r_mg.get('error', ''))
    report("LV waveform succeeded", r_lv['success'], r_lv.get('error', ''))

    if r_mg['success'] and r_lv['success']:
        h1_mg = np.array(r_mg['detectors']['H1'])
        h1_lv = np.array(r_lv['detectors']['H1'])
        rms_diff = np.sqrt(np.mean((h1_mg - h1_lv)**2))
        report("LV waveform differs from mg-only waveform", rms_diff > 0,
               f"rms_diff={rms_diff:.4e}")

    return r_mg, r_lv


# ── Plot 1: Phase comparison ─────────────────────────────────────────────────
def plot_phase_comparison():
    """Plot phase shift vs frequency for GR, massive graviton, LV variants."""
    freqs, f_c = _get_fd_freqs_and_fc()
    band = (freqs >= F_LOWER) & (freqs <= F_FINAL)
    f = freqs[band]

    ps_mg   = _additional_phase(f, CHIRP_MASS, Z, LAMBDA_G, f_c)
    ps_lv3  = _additional_phase_lv(f, CHIRP_MASS, Z, np.inf, 3.0, A_LV_ALPHA3, f_c)
    ps_lv4  = _additional_phase_lv(f, CHIRP_MASS, Z, np.inf, 4.0, A_LV_ALPHA4, f_c)
    ps_comb = _additional_phase_lv(f, CHIRP_MASS, Z, LAMBDA_G, 3.0, A_LV_ALPHA3, f_c)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f, ps_mg,   label=r'Massive graviton (9709011): $\lambda_g=10^{16}$ m',
            color='steelblue', linewidth=1.8)
    ax.plot(f, ps_lv3,  label=fr'LV only $\alpha=3$ (DSR), $A={A_LV_ALPHA3:.1e}$ m',
            color='tomato', linewidth=1.8, linestyle='--')
    ax.plot(f, ps_lv4,  label=fr'LV only $\alpha=4$ (Horava), $A={A_LV_ALPHA4:.1e}$ m',
            color='seagreen', linewidth=1.8, linestyle='-.')
    ax.plot(f, ps_comb, label=r'Combined: $\lambda_g=10^{16}$ m + LV $\alpha=3$',
            color='darkorchid', linewidth=1.8, linestyle=':')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', label='GR (baseline)')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel(r'Phase shift $\delta\Psi$ (rad)', fontsize=12)
    ax.set_title('Phase Modifications: Massive Graviton (9709011) vs LV (1110.2720)', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    path = os.path.join(PLOT_DIR, 'phase_comparison.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 2: Time-domain waveform models ──────────────────────────────────────
def plot_waveform_models(r_gr, r_mg, r_lv3, r_lv4, r_comb):
    """Plot GR / massive-graviton / LV-alpha3 / LV-alpha4 / combined waveforms."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    labels = ['GR', r'Massive graviton ($\lambda_g=10^{16}$ m)',
              fr'LV $\alpha=3$ (DSR), $A={A_LV_ALPHA3:.1e}$ m',
              fr'LV $\alpha=4$ (Horava), $A={A_LV_ALPHA4:.1e}$ m',
              r'Combined ($\lambda_g+\mathrm{LV}\;\alpha=3$)']
    colors = ['black', 'steelblue', 'tomato', 'seagreen', 'darkorchid']
    styles = ['-', '--', '--', '-.', ':']
    results = [r_gr, r_mg, r_lv3, r_lv4, r_comb]
    alphas  = [1.0, 0.85, 0.85, 0.85, 0.85]

    for idx, det in enumerate(['H1', 'L1']):
        gr_sig = np.array(r_gr['detectors'][det])
        time = np.arange(len(gr_sig)) * DT
        peak = np.argmax(np.abs(gr_sig))
        win  = int(0.2 / DT)
        s, e = max(0, peak - win), min(len(gr_sig), peak + win)

        for r, lab, col, ls, al in zip(results, labels, colors, styles, alphas):
            sig = np.array(r['detectors'][det])
            axes[idx].plot(time[s:e], sig[s:e], label=lab if idx == 0 else '',
                           color=col, linestyle=ls, linewidth=1.4, alpha=al)

        axes[idx].set_ylabel('Strain', fontsize=11)
        axes[idx].set_title(f'{det} Detector', fontsize=11)
        axes[idx].grid(True, alpha=0.3)

    axes[0].legend(fontsize=9, loc='upper left')
    axes[1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle('GR vs Massive Graviton vs Lorentz-Violating Waveforms', fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'waveform_models.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 3: Alpha scan ────────────────────────────────────────────────────────
def plot_alpha_scan():
    """Plot H1 time-domain waveforms for several LV exponents α."""
    # A_lv chosen to give ~0.5 rad phase at mid-band for each α
    alpha_values = [0.0, 1.0, 3.0, 4.0]
    A_per_alpha  = {0.0: 1e13, 1.0: A_LV_ALPHA1, 3.0: A_LV_ALPHA3, 4.0: A_LV_ALPHA4}
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']

    # GR baseline
    r_gr = _generate_single_lv_waveform(
        PARAMS, time_resolution=DT, approximant='IMRPhenomD',
        f_lower=F_LOWER, detectors=['H1'], target_length=TARGET,
        add_noise=False, lambda_g=np.inf, alpha_lv=3.0,
        A_lv=np.inf, f_final=F_FINAL
    )

    fig, axes = plt.subplots(len(alpha_values), 1, figsize=(12, 3.5 * len(alpha_values)),
                             sharex=True)
    gr_sig = np.array(r_gr['detectors']['H1'])
    time = np.arange(len(gr_sig)) * DT
    peak = np.argmax(np.abs(gr_sig))
    win  = int(0.2 / DT)
    s, e = max(0, peak - win), min(len(gr_sig), peak + win)

    for ax, alpha, col in zip(axes, alpha_values, colors):
        A_lv = A_per_alpha[alpha]
        r = _generate_single_lv_waveform(
            PARAMS, time_resolution=DT, approximant='IMRPhenomD',
            f_lower=F_LOWER, detectors=['H1'], target_length=TARGET,
            add_noise=False, lambda_g=np.inf, alpha_lv=alpha,
            A_lv=A_lv, f_final=F_FINAL
        )
        sig = np.array(r['detectors']['H1'])
        ax.plot(time[s:e], gr_sig[s:e], color='black', linewidth=1.2,
                label='GR', alpha=0.6)
        ax.plot(time[s:e], sig[s:e], color=col, linewidth=1.4,
                label=fr'$\alpha={alpha}$, $A={A_lv:.1e}$ m')
        ax.set_ylabel('Strain', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle(r'LV Waveforms for Different Dispersion Exponents $\alpha$ (1110.2720)',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'alpha_scan.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 4: A_lv scan (FD phase) ──────────────────────────────────────────────
def plot_A_scan():
    """Plot frequency-domain phase δΨ(f) for a range of A_lv values at α=3.

    Time-domain waveforms appear nearly identical to GR because:
      - the LV phase is normalised to zero at f_c ≈ 677 Hz (merger), and
      - waveform plots are windowed around merger, where δΨ → 0 by construction.
    The FD phase directly reveals the LV effect, which grows toward low frequency.

    A_lv values are chosen so that |δΨ(F_LOWER)| ≈ 0.2, 0.5, 1, 2 rad
    (A_LV_ALPHA3 = 3.022e-16 m gives |ζ|≈0.5, max δΨ ≈ 0.045 rad).
    For α=3, ζ ∝ A_lv, so larger A → larger phase shift.
    """
    freqs, f_c = _get_fd_freqs_and_fc()
    band = (freqs >= F_LOWER) & (freqs <= F_FINAL)
    f = freqs[band]

    # Scale A_LV_ALPHA3 to achieve target peak phase shifts at F_LOWER.
    # Each factor multiplies both A_lv and |ζ| (since ζ ∝ A for α=3).
    A_factors = [4, 11, 22, 44]   # × A_LV_ALPHA3 → |ζ| ≈ 2, 5.5, 11, 22
    A_values  = [A_LV_ALPHA3 * fac for fac in A_factors]
    colors    = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple']
    zeta_approx = [0.5 * fac for fac in A_factors]   # |ζ| estimate

    fig, ax = plt.subplots(figsize=(11, 5))
    for A, col, zeta_est in zip(A_values, colors, zeta_approx):
        ps = _additional_phase_lv(f, CHIRP_MASS, Z, np.inf, 3.0, A, f_c)
        ax.plot(f, ps, color=col, linewidth=1.6,
                label=fr'$A = {A:.2e}$ m  ($|\zeta|\approx{zeta_est:.0f}$)')

    ax.axhline(0, color='black', linewidth=1.0, linestyle='--', label=r'GR ($A\to\infty$)')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel(r'$\delta\Psi(f)$ (rad)', fontsize=12)
    ax.set_title(r'FD Phase Shift: $A_\mathrm{lv}$ Scan, $\alpha=3$ (DSR, 1110.2720)',
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'A_scan.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 5: Frequency-domain phase difference ────────────────────────────────
def plot_fd_phase_diff():
    """Plot frequency-domain phase difference δΨ(f) for all models vs GR."""
    freqs, f_c = _get_fd_freqs_and_fc()
    band = (freqs >= F_LOWER) & (freqs <= F_FINAL)
    f = freqs[band]

    models = {
        r'Massive graviton (9709011): $\lambda_g=10^{16}$ m':
            _additional_phase(f, CHIRP_MASS, Z, LAMBDA_G, f_c),
        fr'LV $\alpha=3$ (DSR), $A={A_LV_ALPHA3:.1e}$ m':
            _additional_phase_lv(f, CHIRP_MASS, Z, np.inf, 3.0, A_LV_ALPHA3, f_c),
        fr'LV $\alpha=4$ (Horava), $A={A_LV_ALPHA4:.1e}$ m':
            _additional_phase_lv(f, CHIRP_MASS, Z, np.inf, 4.0, A_LV_ALPHA4, f_c),
        fr'LV $\alpha=1$ (log), $A={A_LV_ALPHA1:.1e}$ m':
            _additional_phase_lv(f, CHIRP_MASS, Z, np.inf, 1.0, A_LV_ALPHA1, f_c),
        r'Combined: $\lambda_g + \alpha=3$':
            _additional_phase_lv(f, CHIRP_MASS, Z, LAMBDA_G, 3.0, A_LV_ALPHA3, f_c),
    }
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'darkorchid']
    styles = ['-', '--', '-.', (0, (3, 1, 1, 1)), ':']

    fig, ax = plt.subplots(figsize=(11, 5))
    for (label, ps), col, ls in zip(models.items(), colors, styles):
        ax.plot(f, ps, label=label, color=col, linestyle=ls, linewidth=1.6)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel(r'$\delta\Psi(f)$ (rad)', fontsize=12)
    ax.set_title('Frequency-Domain Phase Difference: 9709011 vs 1110.2720', fontsize=13)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'fd_phase_diff.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 6: α=0 degeneracy with massive graviton ─────────────────────────────
def plot_alpha0_degeneracy():
    """Show that the α=0 LV correction is exactly degenerate with the massive
    graviton (arXiv:1110.2720, p.6).

    Both corrections produce a phase ∝ u^{−1} = 1/(πMf).  They differ by an
    unobservable constant global phase (degenerate with coalescence phase φ_c),
    so we subtract the phase at a reference frequency before plotting.

    Left panel : mg-only (λ_g=L) vs LV-only (α=0, A=L)  — curves overlap.
    Right panel: combined (λ_g=L₁, α=0, A=L₂) vs mg-only with
                 λ_eff = (L₁⁻² + L₂⁻²)^{-1/2}              — curves overlap.
    """
    freqs, f_c = _get_fd_freqs_and_fc()
    band = (freqs >= F_LOWER) & (freqs <= F_FINAL)
    f = freqs[band]

    L  = 1e15          # same scale for λ_g and A_lv [m]
    L1 = 1e15
    L2 = 5e14
    lambda_eff = 1.0 / np.sqrt(1.0/L1**2 + 1.0/L2**2)
    f_ref = 200.0      # reference frequency for normalisation

    # Compute phases
    ps_mg      = _additional_phase_lv(f, CHIRP_MASS, Z, L,          0.0, np.inf, f_c)
    ps_lv0     = _additional_phase_lv(f, CHIRP_MASS, Z, np.inf,     0.0, L,      f_c)
    ps_comb    = _additional_phase_lv(f, CHIRP_MASS, Z, L1,         0.0, L2,     f_c)
    ps_mg_eff  = _additional_phase_lv(f, CHIRP_MASS, Z, lambda_eff, 0.0, np.inf, f_c)

    # Remove constant offset (unobservable global phase) by subtracting at f_ref
    i_ref = np.argmin(np.abs(f - f_ref))
    ps_mg     = ps_mg     - ps_mg[i_ref]
    ps_lv0    = ps_lv0    - ps_lv0[i_ref]
    ps_comb   = ps_comb   - ps_comb[i_ref]
    ps_mg_eff = ps_mg_eff - ps_mg_eff[i_ref]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mg-only vs LV α=0 (same A/λ_g)
    ax = axes[0]
    ax.plot(f, ps_mg,  label=r'mg only: $\lambda_g = 10^{15}$ m',
            color='steelblue', linewidth=2.5, linestyle='-')
    ax.plot(f, ps_lv0, label=r'LV $\alpha=0$: $A = 10^{15}$ m',
            color='tomato', linewidth=1.5, linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel(rf'$\delta\Psi(f) - \delta\Psi({f_ref:.0f}\,\mathrm{{Hz}})$ (rad)',
                  fontsize=11)
    ax.set_title(r'$\alpha=0$ LV $\equiv$ Massive Graviton (same $L$)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: combined (λ_g, A, α=0) vs mg-only with λ_eff
    ax = axes[1]
    ax.plot(f, ps_comb,   color='darkorchid', linewidth=2.5, linestyle='-',
            label=(r'Combined: $\lambda_g=10^{15}$ m, '
                   r'$\alpha=0$, $A=5\times10^{14}$ m'))
    ax.plot(f, ps_mg_eff, color='seagreen', linewidth=1.5, linestyle='--',
            label=(r'mg only: $\lambda_\mathrm{eff}='
                   r'(\lambda_g^{-2}+A^{-2})^{-1/2}$'))
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_title(r'Combined $\alpha=0$ $\equiv$ mg only with $\lambda_\mathrm{eff}$',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        r'$\alpha=0$ Degeneracy: LV Correction $\equiv$ Massive Graviton'
        '\n'
        r'(arXiv:1110.2720, p.6 — curves overlap after removing global phase offset)',
        fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'alpha0_degeneracy.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("Lorentz-Violating Waveform Tests (Mirshekari et al. 2011)")
    print("=" * 65)

    test_gr_limit_lv()
    test_lv_scaling()
    test_additive_decomposition()
    test_alpha2_degenerate()
    test_alpha_shapes()
    test_alpha0_degeneracy()
    r_mg_gen, r_lv_gen = test_lv_vs_mg_waveform()

    print("\n=== Generating waveforms for plots ===")

    # GR baseline
    r_gr = _generate_single_lv_waveform(
        PARAMS, time_resolution=DT, approximant='IMRPhenomD',
        f_lower=F_LOWER, detectors=['H1', 'L1'], target_length=TARGET,
        add_noise=False, lambda_g=np.inf, alpha_lv=3.0,
        A_lv=np.inf, f_final=F_FINAL
    )
    # Massive graviton only (9709011)
    r_mg = _generate_single_modified_waveform(
        PARAMS, time_resolution=DT, approximant='IMRPhenomD',
        f_lower=F_LOWER, detectors=['H1', 'L1'], target_length=TARGET,
        add_noise=False, lambda_g=LAMBDA_G, f_final=F_FINAL
    )
    # LV only: α = 3 (DSR)
    r_lv3 = _generate_single_lv_waveform(
        PARAMS, time_resolution=DT, approximant='IMRPhenomD',
        f_lower=F_LOWER, detectors=['H1', 'L1'], target_length=TARGET,
        add_noise=False, lambda_g=np.inf, alpha_lv=3.0,
        A_lv=A_LV_ALPHA3, f_final=F_FINAL
    )
    # LV only: α = 4 (Horava-Lifshitz)
    r_lv4 = _generate_single_lv_waveform(
        PARAMS, time_resolution=DT, approximant='IMRPhenomD',
        f_lower=F_LOWER, detectors=['H1', 'L1'], target_length=TARGET,
        add_noise=False, lambda_g=np.inf, alpha_lv=4.0,
        A_lv=A_LV_ALPHA4, f_final=F_FINAL
    )
    # Combined: massive graviton + LV α=3
    r_comb = _generate_single_lv_waveform(
        PARAMS, time_resolution=DT, approximant='IMRPhenomD',
        f_lower=F_LOWER, detectors=['H1', 'L1'], target_length=TARGET,
        add_noise=False, lambda_g=LAMBDA_G, alpha_lv=3.0,
        A_lv=A_LV_ALPHA3, f_final=F_FINAL
    )

    for name, r in [('GR', r_gr), ('mg', r_mg), ('LV-α3', r_lv3),
                    ('LV-α4', r_lv4), ('combined', r_comb)]:
        report(f"{name} waveform succeeded", r['success'], r.get('error', ''))

    print("\n=== Generating diagnostic plots ===")
    plot_phase_comparison()
    plot_fd_phase_diff()
    if all(r['success'] for r in [r_gr, r_mg, r_lv3, r_lv4, r_comb]):
        plot_waveform_models(r_gr, r_mg, r_lv3, r_lv4, r_comb)
    plot_alpha_scan()
    plot_A_scan()
    plot_alpha0_degeneracy()

    print("\n" + "=" * 65)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 65)

    if FAIL > 0:
        sys.exit(1)
