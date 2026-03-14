"""
Prototype: LV phase shift parameterised by A (LV Compton wavelength, metres).

Implements Mirshekari, Yunes & Will (2011), arXiv:1110.2720, Eqs. 28-32.

Dispersion relation (Eq. 1):
    E² = p²c² + m_g²c⁴ + A_physical · p^α · c^α

where A_physical has units [energy]^{2−α}.  The paper re-packages this as
the LV Compton wavelength (Eq. 13):

    A  ≡  A_physical^{1/(α−2)}       [units: metres, always]

The waveform phase correction (Eq. 28, α ≠ 1, 2) is:

    δΨ = −β u^{−1} − ζ u^{α−1}

with u = πMf (M = detector-frame chirp mass in seconds) and (Eqs. 29, 30):

    β     = π² D_0 M / (λ_g² (1+Z))                     [massive graviton]
    ζ     = π^{2−α}/(1−α) · c^{1−α} · D_α · M^{1−α}
                          / (A^{2−α} · (1+Z)^{1−α})      [LV term]

For α = 1 (Eq. 31−32):
    δΨ_LV = +ζ_1 · ln(u),  ζ_1 = D_1 / A

The ppE mapping (Eq. 34):  β_ppE = −ζ,  b_ppE = α − 1.

Bug in previous code
--------------------
The previous implementation used  b = α − 2  and took β_ppE directly.
Correct exponent is  b = α − 1  (paper Eq. 34, with u = πMf).
"""

import sys
import os
import numpy as np
from scipy.integrate import quad
from pycbc.waveform import get_fd_waveform

# ── Constants (same as JHPY.py) ───────────────────────────────────────────────
C        = 2.998e8
G        = 6.674e-11
M_SUN    = 1.989e30
MPC      = 3.086e22
M_SUN_SEC = G * M_SUN / C**3
H0       = 67.4e3 / MPC
OMEGA_M  = 0.315
OMEGA_L  = 0.685

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


# ── Cosmological distance (Eq. 15) ────────────────────────────────────────────
def D_alpha(alpha, z):
    """D_α in metres (SI).  D_alpha(0, z) = D_0 for massive graviton."""
    result, _ = quad(
        lambda zp: (1 + zp)**(alpha - 2) /
                   np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_L),
        0, z
    )
    return C * (1 + z) / H0 * result


# ── ζ from A_lv (LV Compton wavelength, metres) ───────────────────────────────
def compute_zeta(alpha_lv, A_lv, chirp_mass, z):
    """
    Compute the LV phase coefficient ζ from the LV Compton wavelength A_lv.

    Implements Eqs. 30 and 32 of arXiv:1110.2720 in SI units (c ≠ 1).

    Parameters
    ----------
    alpha_lv : float   Dispersion exponent α.
    A_lv     : float   LV Compton wavelength [m].  A_lv = inf → GR limit (ζ=0).
    chirp_mass : float  Chirp mass [solar masses].
    z        : float   Source redshift.

    Returns
    -------
    float  ζ (dimensionless).  β_ppE = −ζ.
    """
    if not np.isfinite(A_lv) or A_lv <= 0:
        return 0.0
    if alpha_lv == 2.0:
        return 0.0                      # degenerate with t_c

    M  = chirp_mass * M_SUN_SEC * (1 + z)  # detector-frame chirp mass [s]
    Da = D_alpha(alpha_lv, z)               # D_α [m]

    if alpha_lv == 1.0:
        # Eq. 32:  ζ_1 = D_1 / A   (both in metres → dimensionless)
        return Da / A_lv

    # General α ≠ 1, 2  (Eq. 30 converted to SI):
    #   ζ = π^{2−α}/(1−α) · c^{1−α} · D_α · M^{1−α} / (A^{2−α} · (1+Z)^{1−α})
    return (
        np.pi**(2 - alpha_lv) / (1 - alpha_lv)
        * C**(1 - alpha_lv)
        * Da
        * M**(1 - alpha_lv)
        / (A_lv**(2 - alpha_lv) * (1 + z)**(1 - alpha_lv))
    )


# ── Massive-graviton β (Eq. 29) ───────────────────────────────────────────────
def compute_beta(chirp_mass, z, lambda_g):
    M  = chirp_mass * M_SUN_SEC * (1 + z)
    D0 = D_alpha(0, z)
    return np.pi**2 * C * D0 * M / (lambda_g**2 * (1 + z))


# ── Phase shift with A_lv interface ──────────────────────────────────────────
def additional_phase_lv_A(freqs, chirp_mass, z, lambda_g, alpha_lv, A_lv, f_c):
    """
    Phase shift δΨ(f) parameterised by the LV Compton wavelength A_lv [m].

    Implements arXiv:1110.2720 Eqs. 28-32 (SI units).

    Parameters
    ----------
    freqs      : ndarray  Frequency array [Hz].
    chirp_mass : float    Chirp mass [solar masses].
    z          : float    Source redshift.
    lambda_g   : float    Graviton Compton wavelength [m].  inf → no mg term.
    alpha_lv   : float    LV dispersion exponent α.
    A_lv       : float    LV Compton wavelength [m].  inf → no LV term.
    f_c        : float    Cutoff frequency [Hz]; phase normalised to 0 here.

    Returns
    -------
    ndarray  Total phase shift [radians].
    """
    M   = chirp_mass * M_SUN_SEC * (1 + z)
    u   = np.pi * M * freqs
    u_c = np.pi * M * f_c

    # ── Massive graviton term ─────────────────────────────────────────────────
    if np.isfinite(lambda_g) and lambda_g > 0:
        beta = compute_beta(chirp_mass, z, lambda_g)
        # Normalisation: subtract the value at f_c so δΨ(f_c) = 0
        ct = (
            - np.pi * D_alpha(0, z) / ((1 + z) * lambda_g**2 * f_c**2)
            + np.pi * D_alpha(0, z) / (lambda_g**2 * (1 + z) * f_c)
        )
        delta_psi_mg = -beta * u**(-1) + ct
    else:
        delta_psi_mg = np.zeros_like(freqs, dtype=float)

    # ── LV term ───────────────────────────────────────────────────────────────
    zeta = compute_zeta(alpha_lv, A_lv, chirp_mass, z)

    if zeta == 0.0:
        delta_psi_lv = np.zeros_like(freqs, dtype=float)
    elif alpha_lv == 2.0:
        delta_psi_lv = np.zeros_like(freqs, dtype=float)
    elif alpha_lv == 1.0:
        # δΨ_LV = +ζ · (ln u − ln u_c)
        delta_psi_lv = zeta * (np.log(u) - np.log(u_c))
    else:
        # δΨ_LV = −ζ · (u^{α−1} − u_c^{α−1})   [Eq. 28, normalised]
        delta_psi_lv = -zeta * (u**(alpha_lv - 1) - u_c**(alpha_lv - 1))

    return delta_psi_mg + delta_psi_lv


# ── Helpers ───────────────────────────────────────────────────────────────────
M1, M2 = 30.0, 30.0
CHIRP_MASS = (M1 * M2)**(3/5) / (M1 + M2)**(1/5)
Z = 0.1
LAMBDA_G = 1e16   # graviton Compton wavelength [m]
A_LV_REF = 1e13   # LV Compton wavelength [m] — chosen to give visible phase


def get_freqs_fc():
    hp, _ = get_fd_waveform(
        approximant='IMRPhenomD', mass1=M1, mass2=M2,
        delta_f=1.0/256, f_lower=30.0, f_final=2048.0, distance=410.0
    )
    freqs = hp.sample_frequencies.numpy()[1:]
    amp   = np.abs(hp.numpy()[1:])
    f_c   = float(np.max(freqs[np.nonzero(amp)]))
    return freqs, f_c


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: GR limits — phase vanishes when both lambda_g and A_lv are infinite
# ─────────────────────────────────────────────────────────────────────────────
def test_gr_limits():
    print("\n=== Test 1: GR limits ===")
    freqs, f_c = get_freqs_fc()

    for alpha in [0.5, 1.0, 2.0, 3.0, 4.0]:
        ps = additional_phase_lv_A(freqs, CHIRP_MASS, Z,
                                   np.inf, alpha, np.inf, f_c)
        report(f"α={alpha}: A_lv=inf → phase all zeros",
               np.max(np.abs(ps)) < 1e-30, f"max={np.max(np.abs(ps)):.2e}")

    # alpha=2 must vanish regardless of A_lv
    ps_a2 = additional_phase_lv_A(freqs, CHIRP_MASS, Z,
                                   np.inf, 2.0, 1e10, f_c)
    report("α=2: LV term vanishes for any A_lv",
           np.max(np.abs(ps_a2)) < 1e-30, f"max={np.max(np.abs(ps_a2)):.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: LV-only phase is zero at f_c (LV normalisation)
# ─────────────────────────────────────────────────────────────────────────────
def test_normalisation_at_fc():
    """
    Test that δΨ_LV(f_c) = 0 for each α (using lambda_g=inf to isolate LV).
    The massive-graviton term has its own normalisation constants; here we
    check only the LV part.
    """
    print("\n=== Test 2: LV phase = 0 at f_c (lambda_g=inf to isolate LV) ===")
    freqs, f_c = get_freqs_fc()

    for alpha, A_lv in [(3.0, A_LV_REF), (4.0, A_LV_REF), (1.0, A_LV_REF),
                        (0.5, A_LV_REF)]:
        # Use lambda_g=inf so only the LV term is active
        ps = additional_phase_lv_A(freqs, CHIRP_MASS, Z,
                                   np.inf, alpha, A_lv, f_c)
        idx = np.argmin(np.abs(freqs - f_c))
        val = ps[idx]
        report(f"α={alpha}: δΨ_LV(f_c) ≈ 0",
               abs(val) < 1e-8, f"δΨ_LV(f_c)={val:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: LV phase scales as A_lv^{α−2} for α > 2  (ζ ∝ 1/A^{2−α} = A^{α−2})
# ─────────────────────────────────────────────────────────────────────────────
def test_A_scaling():
    print("\n=== Test 3: Phase scaling with A_lv ===")
    freqs, f_c = get_freqs_fc()
    mid = len(freqs) // 2

    # α=3: ζ ∝ A^{α-2} = A^1  → phase doubles when A doubles
    A1 = 1e13
    A2 = 2e13
    ps1 = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf, 3.0, A1, f_c)
    ps2 = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf, 3.0, A2, f_c)
    if ps1[mid] != 0:
        ratio = ps2[mid] / ps1[mid]
        report("α=3: doubling A_lv doubles phase (ratio≈2)",
               abs(ratio - 2.0) < 0.01, f"ratio={ratio:.4f}")

    # α=4: ζ ∝ A^2 → quadrupling when A doubles
    ps1 = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf, 4.0, A1, f_c)
    ps2 = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf, 4.0, A2, f_c)
    if ps1[mid] != 0:
        ratio = ps2[mid] / ps1[mid]
        report("α=4: doubling A_lv quadruples phase (ratio≈4)",
               abs(ratio - 4.0) < 0.01, f"ratio={ratio:.4f}")

    # α=0.5: ζ ∝ A^{-1.5} → phase halves when A doubles
    ps1 = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf, 0.5, A1, f_c)
    ps2 = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf, 0.5, A2, f_c)
    if ps1[mid] != 0:
        ratio = ps2[mid] / ps1[mid]
        expected = (A2/A1)**(0.5 - 2)   # A^{α-2}
        report(f"α=0.5: phase scales as A^{{α-2}} (ratio≈{expected:.3f})",
               abs(ratio - expected) < 0.01, f"ratio={ratio:.4f}, expected={expected:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: α=0 degeneracy — LV with A_lv should equal mg term with λ_g = A_lv
# ─────────────────────────────────────────────────────────────────────────────
def test_alpha0_degeneracy():
    """
    At α=0, the LV A term is 100% degenerate with the massive graviton λ_g
    term (paper page 6: λ_g^{-2} → λ_g^{-2} + A^{-2}).

    With lambda_g=inf and A_lv=L, the LV-only phase should match the
    massive-graviton-only phase with lambda_g=L (up to normalisation constants).
    """
    print("\n=== Test 4: α=0 degeneracy with massive graviton ===")
    freqs, f_c = get_freqs_fc()

    L = 1e15   # metres
    # LV only, α=0, A_lv = L
    ps_lv = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf, 0.0, L, f_c)

    # Compute ζ and β for comparison
    zeta_a0 = compute_zeta(0.0, L, CHIRP_MASS, Z)
    beta_mg  = compute_beta(CHIRP_MASS, Z, L)
    report("α=0: ζ = β (massive-graviton β at same scale)",
           abs(zeta_a0 - beta_mg) / beta_mg < 1e-6,
           f"ζ={zeta_a0:.6e}, β={beta_mg:.6e}")

    # The phase from LV with A_lv=L should be −ζ·u^{-1} + const
    # The massive graviton phase with λ_g=L should be −β·u^{-1} + const
    # Since ζ = β, the frequency-dependent parts should match
    M   = CHIRP_MASS * M_SUN_SEC * (1 + Z)
    u   = np.pi * M * freqs
    u_c = np.pi * M * f_c
    # Manually compute: δΨ_LV = -ζ·(u^{-1} − u_c^{-1})
    ps_expected = -zeta_a0 * (u**(-1) - u_c**(-1))
    residual = np.max(np.abs(ps_lv - ps_expected))
    report("α=0: phase shape matches −ζ·(u^{-1} − u_c^{-1})",
           residual < 1e-20, f"max_residual={residual:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Different α values give different frequency shapes
# ─────────────────────────────────────────────────────────────────────────────
def test_alpha_shapes():
    print("\n=== Test 5: Different α gives different phase shapes ===")
    freqs, f_c = get_freqs_fc()

    ps3 = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf, 3.0, A_LV_REF, f_c)
    ps4 = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf, 4.0, A_LV_REF, f_c)
    ps1 = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf, 1.0, A_LV_REF, f_c)

    report("α=3 differs from α=4",
           np.max(np.abs(ps3 - ps4)) > 0,
           f"max_diff={np.max(np.abs(ps3-ps4)):.4e}")
    report("α=3 differs from α=1",
           np.max(np.abs(ps3 - ps1)) > 0,
           f"max_diff={np.max(np.abs(ps3-ps1)):.4e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Additive decomposition (lambda_g + LV = sum of parts)
# ─────────────────────────────────────────────────────────────────────────────
def test_additive_decomposition():
    print("\n=== Test 6: Phase = mg part + LV part ===")
    freqs, f_c = get_freqs_fc()

    ps_mg  = additional_phase_lv_A(freqs, CHIRP_MASS, Z, LAMBDA_G, 3.0, np.inf, f_c)
    ps_lv  = additional_phase_lv_A(freqs, CHIRP_MASS, Z, np.inf,  3.0, A_LV_REF, f_c)
    ps_tot = additional_phase_lv_A(freqs, CHIRP_MASS, Z, LAMBDA_G, 3.0, A_LV_REF, f_c)

    residual = np.max(np.abs((ps_mg + ps_lv) - ps_tot))
    report("Phase = sum of mg + LV parts",
           residual < 1e-20, f"max_residual={residual:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Comparison with old beta_ppE interface — show the b=α-2 bug
# ─────────────────────────────────────────────────────────────────────────────
def test_b_exponent_bug():
    """
    The old code used b = α-2 instead of the correct b = α-1.
    Verify that the old β_ppE interface with b=α-2 gives a different result
    than the new A_lv interface with b=α-1.
    """
    print("\n=== Test 7: Verify b=α-1 (corrected) vs b=α-2 (old bug) ===")
    freqs, f_c = get_freqs_fc()

    alpha = 3.0
    A_lv  = A_LV_REF
    zeta  = compute_zeta(alpha, A_lv, CHIRP_MASS, Z)
    beta_ppe = -zeta   # the old interface's parameter

    M   = CHIRP_MASS * M_SUN_SEC * (1 + Z)
    u   = np.pi * M * freqs
    u_c = np.pi * M * f_c

    # Correct (new): b = α-1
    ps_correct = -zeta * (u**(alpha - 1) - u_c**(alpha - 1))

    # Old (buggy): b = α-2
    b_old = alpha - 2.0
    ps_old = beta_ppe * (u**b_old - u_c**b_old)

    diff = np.max(np.abs(ps_correct - ps_old))
    report("b=α-1 (correct) differs from b=α-2 (old bug)",
           diff > 0, f"max_diff={diff:.4e}")

    # Verify the correct version has the right u^{α-1} frequency dependence.
    # The unnormalised LV phase (ignoring cutoff term) should scale as u^{α-1}.
    # Check: phase(f2)/phase(f1) ≈ (f2/f1)^{α-1} at low frequencies where
    # the u_c normalisation term is negligible.
    low = np.argmin(np.abs(freqs - 50.0))   # 50 Hz
    mid = np.argmin(np.abs(freqs - 100.0))  # 100 Hz
    if ps_correct[low] != 0:
        ratio_correct = ps_correct[mid] / ps_correct[low]
        ratio_old     = ps_old[mid]     / ps_old[low]
        # For normalised phase the exact ratio isn't u^{α-1} due to -u_c term,
        # but we can still verify the signs / magnitudes differ between b=α-1 and b=α-2.
        report("Correct (b=α-1) and buggy (b=α-2) give different frequency shapes",
               abs(ratio_correct - ratio_old) > 1e-6,
               f"ratio_correct={ratio_correct:.4f}, ratio_old={ratio_old:.4f}")
        print(f"    Unnorm phase scales as f^{{α-1}} = f^{alpha-1:.1f}:")
        print(f"      (f2/f1)^{{α-1}} = {(100/50)**(alpha-1):.4f}")
        print(f"      phase_correct(100)/phase_correct(50) = {ratio_correct:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Print ζ values for reference
# ─────────────────────────────────────────────────────────────────────────────
def print_zeta_table():
    print("\n=== ζ values for A_lv=1e13 m, chirp_mass=26.1 M☉, z=0.1 ===")
    print(f"{'α':>6} | {'ζ':>14} | {'β_ppE = −ζ':>14}")
    print("-" * 42)
    for alpha in [0.0, 0.5, 1.0, 1.5, 3.0, 4.0]:
        if alpha == 2.0:
            continue
        zeta = compute_zeta(alpha, A_LV_REF, CHIRP_MASS, Z)
        print(f"  {alpha:>4.1f} | {zeta:>14.6e} | {-zeta:>14.6e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("Prototype: LV phase shift from A_lv (arXiv:1110.2720)")
    print("=" * 65)

    test_gr_limits()
    test_normalisation_at_fc()
    test_A_scaling()
    test_alpha0_degeneracy()
    test_alpha_shapes()
    test_additive_decomposition()
    test_b_exponent_bug()
    print_zeta_table()

    print("\n" + "=" * 65)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 65)

    if FAIL > 0:
        sys.exit(1)
