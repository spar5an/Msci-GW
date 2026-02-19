"""
Standalone export: GW waveform inset + green conditioning arrow.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform

GW_COLOR = "#3a8c60"
BG_COLOR = "#eef3f9"

plt.rcParams.update({"font.family": "serif", "figure.facecolor": "none"})

# ─── Waveform ─────────────────────────────────────────────────────────────────
hp, hc = get_td_waveform(
    approximant="IMRPhenomD",
    mass1=30, mass2=30,
    delta_t=1.0 / 4096,
    f_lower=20,
    distance=1000,
)
times  = np.array(hp.sample_times)
strain = np.array(hp)

peak_idx = np.argmax(np.abs(strain))
i0 = max(0, peak_idx - int(0.45 * 4096))
i1 = min(len(strain), peak_idx + int(0.08 * 4096))
t_gw = times[i0:i1] - times[peak_idx]
h_gw = strain[i0:i1] / np.max(np.abs(strain[i0:i1]))

# ─── Figure: arrow above + waveform box ───────────────────────────────────────
fig = plt.figure(figsize=(3.2, 4.0), facecolor="none")

# Waveform axes (lower portion)
ax_gw = fig.add_axes([0.08, 0.06, 0.84, 0.52])
ax_gw.plot(t_gw, h_gw, color=GW_COLOR, lw=1.6)
ax_gw.set_facecolor(BG_COLOR)
ax_gw.set_xlim(t_gw[0], t_gw[-1])
ax_gw.set_ylim(-1.15, 1.15)
ax_gw.set_xticks([])
ax_gw.set_yticks([])
ax_gw.grid(False)
for spine in ax_gw.spines.values():
    spine.set_visible(True)
    spine.set_color(GW_COLOR)
    spine.set_linewidth(2.0)
ax_gw.text(0.05, 0.90, r"$h(t)$", transform=ax_gw.transAxes,
           ha="left", va="top", fontsize=14, color=GW_COLOR, style="italic")

# Overlay axes for the arrow + label
ax_ov = fig.add_axes([0, 0, 1, 1], facecolor="none")
ax_ov.set_xlim(0, 1)
ax_ov.set_ylim(0, 1)
ax_ov.axis("off")

# Upward arrow (arrowhead at top)
ax_ov.annotate(
    "", xy=(0.50, 0.90), xytext=(0.50, 0.67),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", color=GW_COLOR, lw=2.2, mutation_scale=22),
    annotation_clip=False,
)

# ─── Save ─────────────────────────────────────────────────────────────────────
fig.savefig("poster_gw_context.pdf", dpi=600, bbox_inches="tight", transparent=True)
fig.savefig("poster_gw_context.png", dpi=600, bbox_inches="tight", transparent=True)
print("Saved → poster_gw_context.pdf / .png")
