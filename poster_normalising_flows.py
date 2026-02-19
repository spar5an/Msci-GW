"""
Poster-quality normalising flows diagram.
Shows data space (bimodal) and latent space (Gaussian) with sample points
mapping between them via f_θ and f_θ^{-1}.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm
from scipy.special import expit  # sigmoid, used for logistic mixture CDF approximation

# ─── Style ────────────────────────────────────────────────────────────────────
DARK_BLUE   = "#1a2e5a"
TEAL        = "#2a9d8f"
ARROW_COLOR = "#4dc3e0"
BG_COLOR    = "#eef3f9"
FILL_BLUE   = "#a8c4d8"
FILL_TEAL   = "#7dc8be"
POINT_COLOR  = "#e76f51"   # warm orange – stands out against both palettes
INV_ARROW_COLOR = "#4682b4"  # steel blue for inverse transform

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

# ─── Distributions ────────────────────────────────────────────────────────────
UNIF_LO, UNIF_HI = -4.0, 4.0
UNIF_H = 1.0 / (UNIF_HI - UNIF_LO)   # = 0.125

def uniform_pdf(x):
    return np.where((x >= UNIF_LO) & (x <= UNIF_HI), UNIF_H, 0.0)

def latent_pdf(z):
    return norm.pdf(z, loc=0, scale=1)

# Map x → z via probability integral transform for Uniform[a,b]
def forward_map(x_vals):
    u = np.clip((x_vals - UNIF_LO) / (UNIF_HI - UNIF_LO), 1e-6, 1 - 1e-6)
    return norm.ppf(u)

# ─── Sample points ────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
n_points = 8
# Stratified: one random draw per equal-width bin → spread out but not regular
bins = np.linspace(UNIF_LO, UNIF_HI, n_points + 1)
x_samples = np.array([rng.uniform(bins[i], bins[i+1]) for i in range(n_points)])
z_samples = forward_map(x_samples)

# ─── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))

# Leave a wide central gap for the arrows
ax_x = fig.add_axes([0.04, 0.12, 0.38, 0.78])   # data space
ax_z = fig.add_axes([0.58, 0.12, 0.38, 0.78])   # latent space

x_range = np.linspace(-5, 5, 400)
z_range = np.linspace(-4, 4, 400)

# ── Parameter-space panel (Uniform) ───────────────────────────────────────────
# Fill and outline the rectangular uniform PDF
ax_x.fill_between(x_range, uniform_pdf(x_range), alpha=0.45, color=FILL_BLUE, step="mid")
# Flat top
ax_x.hlines(UNIF_H, UNIF_LO, UNIF_HI, color=DARK_BLUE, lw=2.5)
# Vertical edges
ax_x.vlines([UNIF_LO, UNIF_HI], 0, UNIF_H, color=DARK_BLUE, lw=2.5)

ax_x.set_xlabel(r"$\theta$")
ax_x.set_ylabel("Probability density")
ax_x.set_title(r"Parameter Space   $p(\theta)$", color="black", fontsize=15, pad=8)

# Sample crosses sit on the top line of the uniform box
y_x = np.full_like(x_samples, UNIF_H)
ax_x.scatter(x_samples, y_x, color=POINT_COLOR, s=100, zorder=5,
             marker="x", linewidths=2.0, label="samples")

ax_x.set_xlim(-5, 5)
# Set ylim so the box occupies ~1/3 of the plot height
ax_x.set_ylim(0, UNIF_H * 3.0)

# ── Latent-space panel ────────────────────────────────────────────────────────
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

# ─── Central arrows ───────────────────────────────────────────────────────────
# Arrows are drawn in figure coordinates via fig.add_artist / annotate
arrow_kw = dict(
    xycoords="figure fraction",
    textcoords="figure fraction",
    arrowprops=dict(
        arrowstyle="-|>",
        color=ARROW_COLOR,
        lw=2.5,
        mutation_scale=22,
    ),
    fontsize=16,
    color=ARROW_COLOR,
    ha="center",
    annotation_clip=False,
)

# Use a dummy axes that spans the whole figure for annotations
ax_mid = fig.add_axes([0, 0, 1, 1], facecolor="none")
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 1)
ax_mid.axis("off")

# Forward arrow  (data → latent)  f_θ
ax_mid.annotate(
    "", xy=(0.57, 0.67), xytext=(0.44, 0.67),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.5, mutation_scale=22),
    annotation_clip=False,
)
fig.text(0.505, 0.72, r"$f_\theta$", ha="center", va="bottom",
         fontsize=17, color=ARROW_COLOR)

# Inverse arrow  (latent → data)  f_θ^{-1}
ax_mid.annotate(
    "", xy=(0.44, 0.45), xytext=(0.57, 0.45),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", color=INV_ARROW_COLOR, lw=2.5, mutation_scale=22),
    annotation_clip=False,
)
fig.text(0.505, 0.37, r"$f_\theta^{-1}$", ha="center", va="top",
         fontsize=17, color=INV_ARROW_COLOR)


# ─── Save ─────────────────────────────────────────────────────────────────────
out = "poster_normalising_flows.pdf"
fig.savefig(out, dpi=600, bbox_inches="tight")
print(f"Saved → {out}")

out_png = "poster_normalising_flows.png"
fig.savefig(out_png, dpi=600, bbox_inches="tight")
print(f"Saved → {out_png}")
