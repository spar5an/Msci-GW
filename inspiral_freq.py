"""
inspiral_freq.py

Compute the gravitational-wave frequency at which the dimensionless
orbital frequency parameter M*omega reaches a given threshold.

Background
----------
In geometric units (G = c = 1) the orbital angular frequency of a
circular binary is written as the dimensionless product

    x  =  (M * omega)^{2/3}          (PN expansion parameter)
or equivalently as the product
    M * omega

where M = m1 + m2 is the total mass.  The inspiral approximation breaks
down near M*omega ~ 0.1 (slightly above the Schwarzschild ISCO value
of M*omega_ISCO ~ 0.0675).  This file exposes a function that converts
that threshold directly into a GW frequency in Hz for any total mass.

Relation used
-------------
    f_GW  =  omega / pi          (quadrupole emission: f_GW = 2 * f_orbital)
           =  (M_omega_threshold) / (pi * M_total_sec)

where M_total_sec = (m1 + m2) * G / c^3  [seconds].
"""

import numpy as np

# Physical constants (SI)
_C       = 2.998e8          # speed of light, m/s
_G       = 6.674e-11        # gravitational constant, m^3 kg^-1 s^-2
_M_SUN   = 1.989e30         # solar mass, kg
_M_SUN_SEC = _G * _M_SUN / _C**3   # solar mass in seconds (~4.926e-6 s)


def inspiral_cutoff_frequency(m1, m2, M_omega=0.1):
    """Return the GW frequency (Hz) at which (m1+m2)*omega = M_omega.

    Parameters
    ----------
    m1, m2 : float or array-like
        Component masses in solar masses.
    M_omega : float, optional
        Dimensionless threshold for the total-mass–orbital-frequency
        product M*omega (geometric units G=c=1).  Default is 0.1.

    Returns
    -------
    f_gw : float or ndarray
        Gravitational-wave frequency in Hz.

    Notes
    -----
    The orbital angular frequency is related to the GW frequency by
        omega = 2*pi*f_orbital  and  f_GW = 2*f_orbital
    so  f_GW = omega / pi.

    In geometric units M*omega = M_omega, so
        omega = M_omega / M_total_sec
        f_GW  = M_omega / (pi * M_total_sec)

    Examples
    --------
    >>> inspiral_cutoff_frequency(30, 30)         # equal-mass 30+30 Msun
    >>> inspiral_cutoff_frequency(1.4, 1.4)       # binary neutron star
    >>> inspiral_cutoff_frequency(10, 5, M_omega=0.0675)  # Schwarzschild ISCO
    """
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    M_total_sec = (m1 + m2) * _M_SUN_SEC   # total mass in seconds
    f_gw = M_omega / (np.pi * M_total_sec)
    return f_gw


# ---------------------------------------------------------------------------
# Demo / self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("f_GW at (M*omega = 0.1)\n" + "-" * 40)

    cases = [
        (10,  10,  "10+10  Msun (stellar BH)"),
        (30,  30,  "30+30  Msun (LIGO-band BH)"),
        (1.4, 1.4, " 1.4+1.4 Msun (BNS)"),
        (36,  29,  "36+29  Msun (GW150914-like)"),
    ]

    for m1, m2, label in cases:
        f = inspiral_cutoff_frequency(m1, m2)
        print(f"  {label:30s}  f_GW = {f:.1f} Hz")

    print()
    print("Schwarzschild ISCO (M*omega = 0.0675)\n" + "-" * 40)
    for m1, m2, label in cases:
        f = inspiral_cutoff_frequency(m1, m2, M_omega=0.0675)
        print(f"  {label:30s}  f_GW = {f:.1f} Hz")

    print()
    print("Mass scan at M*omega = 0.1\n" + "-" * 40)
    masses = np.array([5, 10, 20, 30, 50, 100])
    freqs  = inspiral_cutoff_frequency(masses, masses)
    for m, f in zip(masses, freqs):
        print(f"  {m:3d}+{m:3d} Msun  ->  {f:.1f} Hz")
