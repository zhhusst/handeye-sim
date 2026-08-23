import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable

# ============================================================
# Pose-shared Legendre morphology visualization
# 2x4 layout:
#   panels 1-7 : individual beta_k * Phi_k
#   panel 8   : mixed morphology sum(beta_k * Phi_k)
#
# Compatibility:
#   - no LaTeX/mathtext commands
#   - fixed axes/colorbar; dragging sliders will NOT shrink plots
# ============================================================

# ---------------- Model ----------------
Lu_mm = 200.0
Lv_mm = 150.0

terms = [
    (2,0), (1,1), (0,2),
    (3,0), (2,1), (1,2), (0,3)
]

term_text = [
    "β₂₀", "β₁₁", "β₀₂",
    "β₃₀", "β₂₁", "β₁₂", "β₀₃"
]

phi_text = [
    "Φ₂₀", "Φ₁₁", "Φ₀₂",
    "Φ₃₀", "Φ₂₁", "Φ₁₂", "Φ₀₃"
]

# Softer publication-style diverging color map:
# negative -> blue, zero -> warm white, positive -> coral
soft_diverging = LinearSegmentedColormap.from_list(
    "soft_height",
    ["#0400FFFF", "#87C3F1", "#F7F5F0", "#CF7554", "#DF2B0B"],
    N=256
)

def P(n, x):
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    if n == 2:
        return 0.5 * (3.0*x**2 - 1.0)
    if n == 3:
        return 0.5 * (5.0*x**3 - 3.0*x)
    raise ValueError("Only P0...P3 are needed.")

def phi_rs(r, s, XI, ETA):
    x = 2.0*XI - 1.0
    y = 2.0*ETA - 1.0

    # Unit-RMS normalization on [0,1]^2
    c_rs = 1.0 / np.sqrt((2*r + 1)*(2*s + 1))

    return P(r, x) * P(s, y) / c_rs


# ---------------- Grid ----------------
# Moderate resolution keeps slider interaction responsive.
xi = np.linspace(0.0, 1.0, 61)
eta = np.linspace(0.0, 1.0, 46)

XI, ETA = np.meshgrid(xi, eta)

U = XI * Lu_mm
V = ETA * Lv_mm

PHI = np.stack(
    [phi_rs(r, s, XI, ETA) for r, s in terms],
    axis=0
)


# ---------------- Slider settings ----------------
# Initial value reproduces the previous single-mode appearance.
beta_min = -0.30
beta_max = +0.30
beta_init = +0.15

beta = np.full(7, beta_init, dtype=float)

# A fixed common z/color range is essential:
# no rescaling or shrinking during interaction.
#
# Compute the theoretical maximum mixed height over all slider extremes:
# max_{beta_k in [-B,B]} |sum beta_k Phi_k|
# = B * sum |Phi_k| at each grid point.
mixed_bound = beta_max * np.sum(np.abs(PHI), axis=0)
zmax = 1.05 * float(np.max(mixed_bound))

norm = TwoSlopeNorm(
    vmin=-zmax,
    vcenter=0.0,
    vmax=zmax
)


# ============================================================
# Figure layout
# ============================================================
fig = plt.figure(figsize=(14.0, 10.0))

# Leave a large fixed area at the bottom for seven sliders.
gs = fig.add_gridspec(
    2, 4,
    left=0.035,
    right=0.985,
    top=0.92,
    bottom=0.29,
    wspace=0.03,
    hspace=0.12
)

axes = [
    fig.add_subplot(gs[i // 4, i % 4], projection="3d")
    for i in range(8)
]

# Dedicated fixed colorbar axis.
cax = fig.add_axes([0.24, 0.245, 0.52, 0.020])

# Seven fixed slider axes.
slider_axes = []
sliders = []

slider_left = 0.18
slider_width = 0.62
slider_height = 0.017
slider_y0 = 0.205
slider_dy = 0.025

for k in range(7):
    sax = fig.add_axes([
        slider_left,
        slider_y0 - k*slider_dy,
        slider_width,
        slider_height
    ])

    slider = Slider(
        ax=sax,
        label=term_text[k] + " / mm",
        valmin=beta_min,
        valmax=beta_max,
        valinit=beta_init,
        valstep=0.005
    )

    slider_axes.append(sax)
    sliders.append(slider)


# ---------------- Axis style ----------------
def style_axis(ax, idx):
    ax.set_xlim(0.0, Lu_mm)
    ax.set_ylim(0.0, Lv_mm)
    ax.set_zlim(-zmax, zmax)

    ax.view_init(elev=27, azim=-58)

    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    pane = (0.988, 0.988, 0.988, 1.0)

    ax.xaxis.set_pane_color(pane)
    ax.yaxis.set_pane_color(pane)
    ax.zaxis.set_pane_color(pane)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = (0.83, 0.83, 0.83, 0.35)
        axis._axinfo["grid"]["linewidth"] = 0.45

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(4))

    ax.tick_params(labelsize=7, pad=0)

    ax.set_xlabel("u / mm", fontsize=8, labelpad=3)
    ax.set_ylabel("v / mm", fontsize=8, labelpad=3)

    # Only the first panel in each row gets a z label.
    if idx in (0, 4):
        ax.set_zlabel("h / mm", fontsize=8, labelpad=3)
    else:
        ax.set_zlabel("")


for idx, ax in enumerate(axes):
    style_axis(ax, idx)


# ============================================================
# Initial surfaces
# ============================================================
surface_artists = [None] * 8

def single_surface(k):
    return beta[k] * PHI[k]

def mixed_surface():
    return np.tensordot(beta, PHI, axes=(0, 0))

def draw_surface(ax, H):
    return ax.plot_surface(
        U,
        V,
        H,
        cmap=soft_diverging,
        norm=norm,
        linewidth=0,
        antialiased=True,
        rcount=46,
        ccount=61,
        shade=False
    )

# First seven panels
for k in range(7):
    surface_artists[k] = draw_surface(
        axes[k],
        single_surface(k)
    )

    axes[k].set_title(
        phi_text[k] + "   (" + term_text[k] +
        " = {:+.3f} mm)".format(beta[k]),
        fontsize=10,
        pad=4
    )

# Eighth panel: mixed morphology
surface_artists[7] = draw_surface(
    axes[7],
    mixed_surface()
)

axes[7].set_title(
    "Mixed morphology:  hβ(ξ,η) = Φ(ξ,η)ᵀβ",
    fontsize=10,
    pad=4
)


# Panel labels
panel_labels = [
    "(a)", "(b)", "(c)", "(d)",
    "(e)", "(f)", "(g)", "(h)"
]

for label, ax in zip(panel_labels, axes):
    ax.text2D(
        0.03,
        0.93,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold"
    )


# ---------------- Shared colorbar ----------------
sm = ScalarMappable(
    norm=norm,
    cmap=soft_diverging
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    cax=cax,
    orientation="horizontal"
)

cbar.set_label(
    "Surface height h / mm",
    fontsize=10
)

cbar.ax.tick_params(
    labelsize=8
)

cbar.outline.set_linewidth(0.55)


# ---------------- Main title ----------------
fig.suptitle(
    "Legendre product modes and their mixed pose-shared surface morphology",
    fontsize=13.5,
    y=0.975
)


# ============================================================
# Interactive update
# ============================================================
# IMPORTANT:
# - do NOT ax.clear()
# - do NOT recreate colorbar
# - do NOT change subplot positions
# - only remove/replot the changed surface objects
#
# This prevents the shrinking problem.
def update_beta(k, value):
    beta[k] = float(value)

    # 1) Update only the corresponding individual basis panel.
    if surface_artists[k] is not None:
        surface_artists[k].remove()

    surface_artists[k] = draw_surface(
        axes[k],
        single_surface(k)
    )

    axes[k].set_title(
        phi_text[k] + "   (" + term_text[k] +
        " = {:+.3f} mm)".format(beta[k]),
        fontsize=10,
        pad=4
    )

    # 2) Update the mixed morphology panel.
    if surface_artists[7] is not None:
        surface_artists[7].remove()

    surface_artists[7] = draw_surface(
        axes[7],
        mixed_surface()
    )

    axes[7].set_title(
        "Mixed morphology:  hβ(ξ,η) = Φ(ξ,η)ᵀβ",
        fontsize=10,
        pad=4
    )

    fig.canvas.draw_idle()


# Bind each slider to its own beta.
for k, slider in enumerate(sliders):
    slider.on_changed(
        lambda value, kk=k: update_beta(kk, value)
    )


# ============================================================
# Buttons
# ============================================================
reset_ax = fig.add_axes([0.83, 0.055, 0.10, 0.040])
reset_button = Button(
    reset_ax,
    "Reset"
)

zero_ax = fig.add_axes([0.83, 0.105, 0.10, 0.040])
zero_button = Button(
    zero_ax,
    "Set all zero"
)

save_ax = fig.add_axes([0.83, 0.155, 0.10, 0.040])
save_button = Button(
    save_ax,
    "Save figure"
)

def reset_all(event):
    for slider in sliders:
        slider.reset()

def set_all_zero(event):
    # Slider callbacks automatically update the plots.
    for slider in sliders:
        slider.set_val(0.0)

def save_current(event):
    # Save the CURRENT slider state.
    fig.savefig(
        "legendre_2x4_current_state.png",
        dpi=500,
        bbox_inches="tight"
    )
    fig.savefig(
        "legendre_2x4_current_state.pdf",
        bbox_inches="tight"
    )
    print("Saved:")
    print("  legendre_2x4_current_state.png")
    print("  legendre_2x4_current_state.pdf")

reset_button.on_clicked(reset_all)
zero_button.on_clicked(set_all_zero)
save_button.on_clicked(save_current)


# ============================================================
# Show
# ============================================================
plt.show()
