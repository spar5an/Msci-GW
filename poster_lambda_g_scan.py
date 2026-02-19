"""
Poster-quality lambda_g scan plot.
Generates a high-resolution figure with a pastel blue/green palette.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pycbc.waveform import get_fd_waveform
from scipy.integrate import quad

# ─── Constants ───────────────────────────────────────────────────────────────
C = 2.998e8
G = 6.674e-11
M_SUN = 1.989e30
MPC = 3.086e22
M_SUN_SEC = G * M_SUN / C**3
H0 = 67.4e3 / MPC
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
M1 = 30
M2 = 30

# ─── Physics ─────────────────────────────────────────────────────────────────
def normalise(signal):
    return signal / abs(signal).max()

def D_alpha(alpha, z):
    integrand = quad(
        lambda z_prime: (1 + z_prime)**(alpha - 2) /
        np.sqrt(OMEGA_M * (1 + z_prime)**3 + OMEGA_LAMBDA), 0, z)
    return C * (1 + z) / H0 * integrand[0]

def additional_phase(freqs, chirp_mass, z, lambda_g, f_c):
    M = chirp_mass * M_SUN_SEC * (1 + z)
    u = np.pi * M * freqs
    beta = np.pi**2 * C * D_alpha(0, z) * M / (lambda_g**2 * (1 + z))
    constant_terms = (-1 * np.pi * D_alpha(0, z) / ((1 + z) * lambda_g**2 * f_c**2)
                      + np.pi * D_alpha(0, z) / (lambda_g**2 * (1 + z) * f_c))
    delta_psi = -beta * u**(-1) + constant_terms
    return delta_psi

# ─── Generate waveform ───────────────────────────────────────────────────────
hp, hc = get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=M1, mass2=M2,
    delta_f=1.0 / 256,
    f_lower=30, f_final=2048,
    distance=1000
)

freqs = hp.sample_frequencies.numpy()[1:]
phase_info = normalise(hp.numpy()[1:])
chirp_mass = (M1 * M2)**(3/5) / (M1 + M2)**(1/5)
f_c = float(np.max(freqs[np.nonzero(np.abs(hp.numpy()[1:]))]))

dt = 1.0 / (2 * hp.sample_frequencies.numpy()[-1])

def stitch_irfft(raw):
    """Stitch circular IRFFT output: [inspiral+merger | ringdown]."""
    n_ringdown = 500  # ~0.12s at 4096 Hz, matches JHPY.py
    return np.concatenate([raw[n_ringdown:], raw[:n_ringdown]])

hp_td = stitch_irfft(np.fft.irfft(normalise(hp.numpy())))

# Build time axis centred on merger (t=0)
peak_idx = np.argmax(np.abs(hp_td))
time = (np.arange(len(hp_td)) - peak_idx) * dt

window_before = int(0.1 / dt)   # 0.1s before merger
window_after = int(0.1 / dt)    # 0.1s after merger — shows ringdown
t_start = max(0, peak_idx - window_before)
t_end = min(len(hp_td), peak_idx + window_after)

# ─── Poster style ────────────────────────────────────────────────────────────
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# Saturated blue/green palette with strong contrast
LG_16   = 1e16
LG_16_5 = 10**16.2

COLORS = {
    'GR':     '#0D1B2A',   # near-black navy
    LG_16:    '#1D3557',   # dark blue
    LG_16_5:  '#2A9D8F',   # teal
}

LABELS = {
    LG_16:    r'$\lambda_g = 10^{16}$ m',
    LG_16_5:  r'$\lambda_g = 10^{16.2}$ m',
}

lambda_g_values = [LG_16, LG_16_5]

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

hp_td_norm = hp_td / np.max(np.abs(hp_td))
ax.plot(time[t_start:t_end], hp_td_norm[t_start:t_end],
        label="GR", color=COLORS['GR'], linewidth=2.2, zorder=10)

for lg in lambda_g_values:
    ps = additional_phase(freqs, chirp_mass=chirp_mass, z=0.1, lambda_g=lg, f_c=f_c)
    mod = phase_info * np.exp(1j * ps)
    mod_td = stitch_irfft(np.fft.irfft(np.concatenate(([0], mod))))
    mod_td_norm = mod_td / np.max(np.abs(mod_td))
    ax.plot(time[t_start:t_end], mod_td_norm[t_start:t_end],
            label=LABELS[lg],
            color=COLORS[lg], linewidth=1.6, alpha=0.9)

ax.set_xlabel("Time (s)", fontsize=16)
ax.set_ylabel("Strain (normalised)", fontsize=16)
ax.set_title("Modified Gravitational Waveforms for Different Graviton Compton Wavelengths",
             fontsize=18, pad=12)
ax.legend(loc="upper left", fontsize=12, framealpha=0.9,
          edgecolor='#cccccc', fancybox=True)

ax.set_facecolor('#FAFCFF')
fig.patch.set_facecolor('white')
ax.grid(True, alpha=0.5, linewidth=0.8)

plt.tight_layout()
fig.savefig("poster_lambda_g_scan.png", dpi=300, bbox_inches='tight')
fig.savefig("poster_lambda_g_scan.pdf", bbox_inches='tight')
plt.close()
print("Saved: poster_lambda_g_scan.png (300 dpi) and .pdf")
