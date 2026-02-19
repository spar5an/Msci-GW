"""
Poster-quality normalising flows diagram with GW context.
Shows posterior p(θ|h), GW waveform h(t) as conditioning context, and latent space p(z).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from pycbc.waveform import get_td_waveform

# ─── Style ────────────────────────────────────────────────────────────────────
DARK_BLUE       = "#1a2e5a"
TEAL            = "#2a9d8f"
ARROW_COLOR     = "#4dc3e0"   # teal/cyan  – forward transform
INV_ARROW_COLOR = "#4682b4"   # steel blue – inverse transform
GW_COLOR        = "#3a8c60"   # green for waveform & conditioning arrow
BG_COLOR        = "#eef3f9"
FILL_BLUE       = "#a8c4d8"
FILL_TEAL       = "#7dc8be"
POINT_COLOR     = "#e76f51"

plt.rcParams.update({
    "font.family": "serif",
    "axes.facecolor": BG_COLOR,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.labelsize": 14,
})

# ─── GW Waveform ──────────────────────────────────────────────────────────────
hp, hc = get_td_waveform(
    approximant="IMRPhenomD",
    mass1=30, mass2=30,
    delta_t=1.0 / 4096,
    f_lower=20,
    distance=1000,
)

times  = np.array(hp.sample_times)
strain = np.array(hp)

# Trim: 0.45 s before merger → 0.08 s after (shows clear chirp + merger)
peak_idx = np.argmax(np.abs(strain))
i0 = max(0, peak_idx - int(0.45 * 4096))
i1 = min(len(strain), peak_idx + int(0.08 * 4096))
t_gw = times[i0:i1] - times[peak_idx]
h_gw = strain[i0:i1] / np.max(np.abs(strain[i0:i1]))

# ─── Distributions ────────────────────────────────────────────────────────────
def posterior_pdf(theta):
    """Narrow Gaussian – representative posterior given a GW observation."""
    return norm.pdf(theta, loc=1.0, scale=0.4)

def latent_pdf(z):
    return norm.pdf(z, loc=0, scale=1)

# Sample points (mapped via probability integral transform)
rng = np.random.default_rng(42)
n_points = 8
theta_samples = rng.normal(1.0, 0.4, n_points)
u = np.clip(norm.cdf(theta_samples, loc=1.0, scale=0.4), 1e-6, 1 - 1e-6)
z_samples = norm.ppf(u)

# ─── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))

ax_theta = fig.add_axes([0.04, 0.12, 0.37, 0.78])   # posterior
ax_z     = fig.add_axes([0.60, 0.12, 0.37, 0.78])   # latent
ax_gw    = fig.add_axes([0.435, 0.70, 0.13, 0.20])  # waveform inset

theta_range = np.linspace(-5, 5, 400)
z_range     = np.linspace(-4, 4, 400)

# ── Posterior panel ───────────────────────────────────────────────────────────
ax_theta.fill_between(theta_range, posterior_pdf(theta_range), alpha=0.45, color=FILL_BLUE)
ax_theta.plot(theta_range, posterior_pdf(theta_range), color=DARK_BLUE, lw=2.5)
ax_theta.set_xlabel(r"$\theta$")
ax_theta.set_ylabel("Probability density")
ax_theta.set_title(r"Posterior   $p(\theta \mid h)$", color="black", fontsize=15, pad=8)

y_theta = posterior_pdf(theta_samples)
ax_theta.scatter(theta_samples, y_theta, color=POINT_COLOR, s=100, zorder=5,
                 marker="x", linewidths=2.0)
ax_theta.set_xlim(-5, 5)
ax_theta.set_ylim(bottom=0)

# ── Latent panel ──────────────────────────────────────────────────────────────
ax_z.fill_between(z_range, latent_pdf(z_range), alpha=0.45, color=FILL_TEAL)
ax_z.plot(z_range, latent_pdf(z_range), color=TEAL, lw=2.5)
ax_z.set_xlabel("$z$")
ax_z.set_title(r"Latent Space   $p_Z(z)$", color="black", fontsize=15, pad=8)
ax_z.yaxis.set_label_position("right")
ax_z.yaxis.tick_right()

y_z = latent_pdf(z_samples)
ax_z.scatter(z_samples, y_z, color=POINT_COLOR, s=100, zorder=5,
             marker="x", linewidths=2.0)
ax_z.set_xlim(-4, 4)
ax_z.set_ylim(bottom=0)

# ── GW waveform inset ─────────────────────────────────────────────────────────
ax_gw.plot(t_gw, h_gw, color=GW_COLOR, lw=1.4)
ax_gw.set_facecolor(BG_COLOR)
ax_gw.set_xlim(t_gw[0], t_gw[-1])
ax_gw.set_ylim(-1.15, 1.15)
ax_gw.set_xticks([])
ax_gw.set_yticks([])
ax_gw.grid(False)
# Draw box border in GW_COLOR
for spine in ax_gw.spines.values():
    spine.set_visible(True)
    spine.set_color(GW_COLOR)
    spine.set_linewidth(1.8)
ax_gw.text(0.05, 0.90, r"$h(t)$", transform=ax_gw.transAxes,
           ha="left", va="top", fontsize=13, color=GW_COLOR, style="italic")

# ── Arrows (full-figure overlay) ──────────────────────────────────────────────
ax_mid = fig.add_axes([0, 0, 1, 1], facecolor="none")
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 1)
ax_mid.axis("off")

# Downward arrow: from bottom of waveform box → above forward arrow
ax_mid.annotate(
    "", xy=(0.500, 0.610), xytext=(0.500, 0.697),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", color=GW_COLOR, lw=2.0, mutation_scale=18),
    annotation_clip=False,
)
fig.text(0.530, 0.652, r"$f_\theta(\cdot\,;\, h)$",
         ha="left", va="center", fontsize=13, color=GW_COLOR)

# Forward arrow  (posterior → latent)
ax_mid.annotate(
    "", xy=(0.595, 0.58), xytext=(0.420, 0.58),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.5, mutation_scale=22),
    annotation_clip=False,
)

# Inverse arrow  (latent → posterior)
ax_mid.annotate(
    "", xy=(0.420, 0.42), xytext=(0.595, 0.42),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", color=INV_ARROW_COLOR, lw=2.5, mutation_scale=22),
    annotation_clip=False,
)
fig.text(0.508, 0.355, r"$f_\theta^{-1}(\cdot\,;\, h)$",
         ha="center", va="top", fontsize=13, color=INV_ARROW_COLOR)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_pdf = "poster_normalising_flows_gw.pdf"
out_png = "poster_normalising_flows_gw.png"
fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
print(f"Saved → {out_pdf}")
fig.savefig(out_png, dpi=600, bbox_inches="tight")
print(f"Saved → {out_png}")
