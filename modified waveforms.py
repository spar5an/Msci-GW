import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_fd_waveform
from pycbc.detector import Detector
from scipy.integrate import quad

C = 2.998e8                          # speed of light, m/s
G = 6.674e-11                        # gravitational constant, m^3 kg^-1 s^-2
M_SUN = 1.989e30                     # solar mass, kg
MPC = 3.086e22                       # megaparsec, m
M_SUN_SEC = G * M_SUN / C**3         # solar mass in seconds (~4.926e-6 s)
H0 = 67.4e3 / MPC                   # Hubble constant in 1/s
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
M1 = 30                             # solar masses
M2 = 30                             # solar masses
lambda_g = 1e16                        # graviton Compton wavelength in metres (example value)



def normalise(signal):
    """Normalise a frequency-domain signal so its peak amplitude is 1."""
    return signal / abs(signal).max()

def D_alpha(alpha, z):
    integrand = quad(lambda z_prime: (1 + z_prime)**(alpha - 2) / np.sqrt(OMEGA_M * (1 + z_prime)**3 + OMEGA_LAMBDA), 0, z)
    return C * (1 + z) / H0 * integrand[0]  # returns distance in metres

def additional_phase(freqs, chirp_mass, z, lambda_g, mass1, mass2):
    M = chirp_mass * M_SUN_SEC * (1 + z)  # chirp mass in seconds (geometrized)
    u = np.pi * M * freqs                 

    beta = np.pi**2 * C * D_alpha(0, z) * M / (lambda_g**2 * (1 + z))# + 5/96 * (743/336 + 11/4 * nu) * nu**(-2/5) * u**(-1) - 3/8 * np.pi * nu**(-3/5) * u**(-2/3)
    # PN correction terms removed — already included in IMRPhenomD:

    delta_psi = -beta * u**(-1)
    return delta_psi

# Generate a compact binary merger waveform in the frequency domain
hp, hc = get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=M1,
    mass2=M2,
    delta_f=1.0 / 256,
    f_lower=30,
    f_final=2048,
    distance=1000
)

# Project the waveform onto the LIGO Hanford detector
det = Detector("H1")
# Sky location and polarization (arbitrary)
ra, dec, pol = 1.7, 0.4, 0.3
# GPS time for the signal
gps_time = 1187008882  # GW170817-ish epoch

# Plot the frequency-domain waveform (h+)
freqs = hp.sample_frequencies.numpy()[1:]
#plan
#remove amplitude info
phase_info = normalise(hp.numpy()[1:])

chirp_mass = (M1 * M2) **(3/5) / (M1 + M2)**(1/5)

phase_shift = additional_phase(freqs, chirp_mass=chirp_mass, z=0.1, lambda_g=lambda_g, mass1=M1, mass2=M2)

modified_phase_info = phase_info * np.exp(1j * phase_shift)



plt.figure(figsize=(10, 5))
plt.loglog(freqs, normalise(np.abs(hp.numpy()[1:])), label="hp")
plt.loglog(freqs, normalise(np.abs(modified_phase_info)), label="modified")
plt.xlabel("Frequency (Hz)")
plt.xlim(30, 2048)
plt.legend()
plt.tight_layout()
plt.savefig("modified waveform.png")
# IFFT back to time domain
hp_td = np.fft.irfft(normalise(hp.numpy()))
modified_td = np.fft.irfft(np.concatenate(([0], modified_phase_info)))

dt = 1.0 / (2 * hp.sample_frequencies.numpy()[-1])
time = np.arange(len(hp_td)) * dt

peak_idx = np.argmax(np.abs(hp_td))
window = int(0.1 / dt)  # 0.1s either side of coalescence
t_start = max(0, peak_idx - window)
t_end = min(len(hp_td), peak_idx + window)

plt.figure(figsize=(10, 5))
plt.plot(time[t_start:t_end], hp_td[t_start:t_end], label="Original")
plt.plot(time[t_start:t_end], modified_td[t_start:t_end], label="Modified", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Time-Domain Waveforms (around coalescence)")
plt.legend()
plt.tight_layout()
plt.savefig("time_domain_waveform.png")

# Scan over different lambda_g values
lambda_g_values = [1e13, 1e14, 1e15, 1e16, 1e17]

plt.figure(figsize=(10, 5))
plt.plot(time[t_start:t_end], hp_td[t_start:t_end], label="GR", color="black", linewidth=1.5)

for lg in lambda_g_values:
    ps = additional_phase(freqs, chirp_mass=chirp_mass, z=0.1, lambda_g=lg, mass1=M1, mass2=M2)
    mod = phase_info * np.exp(1j * ps)
    mod_td = np.fft.irfft(np.concatenate(([0], mod)))
    plt.plot(time[t_start:t_end], mod_td[t_start:t_end], label=f"$\\lambda_g = 10^{{{int(np.log10(lg))}}}$ m", alpha=0.7)

plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Time-Domain Waveforms for Different Graviton Wavelengths")
plt.legend(loc="upper left", fontsize="small")
plt.tight_layout()
plt.savefig("lambda_g_scan.png")

# This is for general theories, doing massive graviton first
# def D_alpha(alpha, z):
#     intergrand = quad(lambda z_prime: (1 + z_prime)**(alpha - 2) / np.sqrt(OMEGA_M * (1 + z_prime)**3 + OMEGA_LAMBDA), 0, z)
#     return (1 + z)/HUBBLE_CONSTANT * intergrand[0]

# def additional_phase(freq, alpha, chirp_mass, z, luminosity_distance, lambda_g):
#     M = chirp_mass * (1 + z)
#     u = np.pi * M * freq
    
#     beta = np.pi**2 * D_alpha(0, z) * M/ (lambda_g**2 * (1 + z))
    
    
    
#     if alpha == 1:
#         ValueError("Not implimented for alpha=1")
#     else:
#         c = 


