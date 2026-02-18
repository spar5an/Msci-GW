"""Diagnostic: scan lambda_g values and check if the phase shift changes."""
import numpy as np
from scipy.integrate import quad
from pycbc.waveform import get_fd_waveform

C = 2.998e8                          # speed of light, m/s
G = 6.674e-11                        # gravitational constant, m^3 kg^-1 s^-2
M_SUN = 1.989e30                     # solar mass, kg
MPC = 3.086e22                       # megaparsec, m
M_SUN_SEC = G * M_SUN / C**3         # solar mass in seconds (~4.926e-6 s)
H0 = 67.4e3 / MPC                   # Hubble constant in 1/s
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
M1 = 30
M2 = 30

def D_alpha(alpha, z):
    integrand = quad(lambda z_prime: (1 + z_prime)**(alpha - 2) / np.sqrt(OMEGA_M * (1 + z_prime)**3 + OMEGA_LAMBDA), 0, z)
    return C * (1 + z) / H0 * integrand[0]  # metres

def additional_phase(freqs, chirp_mass, z, lambda_g, f_c):
    M = chirp_mass * M_SUN_SEC * (1 + z)  # seconds
    u = np.pi * M * freqs                 # dimensionless
    beta = np.pi**2 * C * D_alpha(0, z) * M / (lambda_g**2 * (1 + z))  # dimensionless
    constant_terms = (-1 * np.pi * D_alpha(0, z) / ((1 + z) * lambda_g**2 * f_c**2)
                      + np.pi * D_alpha(0, z) / (lambda_g**2 * (1 + z) * f_c))
    delta_psi = -beta * u**(-1) + constant_terms
    return delta_psi

chirp_mass = (M1 * M2)**(3/5) / (M1 + M2)**(1/5)
z = 0.1
test_freq = 100.0
freqs = np.array([test_freq])

# Generate waveform to derive f_c from non-zero amplitude range
hp_fd, _ = get_fd_waveform(
    approximant='IMRPhenomD', mass1=M1, mass2=M2,
    delta_f=1.0/256, f_lower=30.0, f_final=2048.0, distance=410.0
)
fd_freqs = hp_fd.sample_frequencies.numpy()[1:]
f_c = float(np.max(fd_freqs[np.nonzero(np.abs(hp_fd.numpy()[1:]))]))

# Break down the three terms separately (with correct units)
M = chirp_mass * M_SUN_SEC * (1 + z)  # seconds
u = np.pi * M * test_freq             # dimensionless
nu = M1 * M2 / (M1 + M2)**2
D = D_alpha(0, z)

print("=== Unit checks ===")
print(f"M_SUN_SEC:           {M_SUN_SEC:.6e} s")
print(f"H0 (SI):             {H0:.6e} 1/s")
print(f"Chirp mass:          {chirp_mass:.4f} M_sun = {M:.6e} s")
print(f"D_alpha(0, z=0.1):   {D:.6e} m = {D/MPC:.2f} Mpc")
print(f"u at 100 Hz:         {u:.6e}")
print()

pn_term1 = 5/96 * (743/336 + 11/4 * nu) * nu**(-2/5) * u**(-1)
pn_term2 = -3/8 * np.pi * nu**(-3/5) * u**(-2/3)

print("=== Term breakdown at f=100 Hz ===")
print(f"PN term 1 (u^-1, no lambda_g):   {pn_term1:.6e}")
print(f"PN term 2 (u^-2/3, no lambda_g): {pn_term2:.6e}")
print()

print("=== Scan over lambda_g ===")
print(f"{'lambda_g (m)':>12} | {'beta term':>14} | {'total phase':>14} | {'% from beta':>12}")
print("-" * 62)

for lambda_g in [1e10, 1e12, 1e14, 1e16, 1e18, 1e20, np.inf]:
    if np.isinf(lambda_g):
        beta_term = 0.0
    else:
        beta = np.pi**2 * C * D * M / (lambda_g**2 * (1 + z))
        constant_terms = (-1 * np.pi * D / ((1 + z) * lambda_g**2 * f_c**2)
                          + np.pi * D / (lambda_g**2 * (1 + z) * f_c))
        beta_term = -beta * u**(-1) + constant_terms
    total = beta_term + pn_term1 + pn_term2
    if total != 0:
        pct = abs(beta_term / total) * 100
    else:
        pct = 0
    print(f"  {lambda_g:>10.2e} | {beta_term:>14.6e} | {total:>14.6e} | {pct:>10.1f}%")

print()
print("=== Key insight ===")
gr_phase = pn_term1 + pn_term2
print(f"Phase with lambda_g=inf (GR limit): {gr_phase:.6e}")
print(f"Phase with lambda_g=1e16:           {additional_phase(freqs, chirp_mass, z, 1e16, f_c)[0]:.6e}")
print(f"Phase with lambda_g=1e12:           {additional_phase(freqs, chirp_mass, z, 1e12, f_c)[0]:.6e}")
print()

if abs(pn_term1 + pn_term2) > 0:
    print("The PN terms (which don't depend on lambda_g) dominate the phase shift.")
    print("These terms are ALREADY in the IMRPhenomD waveform — they're being double-counted.")
    print()
    print("FIX: additional_phase should only return the massive graviton term:")
    print("  delta_psi = -beta * u**(-1)")
