import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable

Lu_mm, Lv_mm = 200.0, 150.0
terms = [(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3)]

soft_diverging = LinearSegmentedColormap.from_list(
    "soft_height",
    ["#6F93B6", "#D6E1EA", "#F7F5F0", "#E8C7BC", "#C77C6E"],
    N=256
)

def P(n, x):
    if n == 0: return np.ones_like(x)
    if n == 1: return x
    if n == 2: return 0.5*(3*x**2 - 1)
    if n == 3: return 0.5*(5*x**3 - 3*x)
    raise ValueError("Only P0..P3 are required.")

def phi(r, s, XI, ETA):
    x, y = 2*XI-1, 2*ETA-1
    c_rs = 1.0 / np.sqrt((2*r+1)*(2*s+1))
    return P(r,x)*P(s,y)/c_rs

xi = np.linspace(0,1,101)
eta = np.linspace(0,1,81)
XI, ETA = np.meshgrid(xi,eta)
U, V = XI*Lu_mm, ETA*Lv_mm
PHI = np.stack([phi(r,s,XI,ETA) for r,s in terms])

fig = plt.figure(figsize=(10.8,7.8))

# IMPORTANT:
# Fixed axes positions. The colorbar has its own dedicated axis and is created once.
# It will no longer steal space from the 3D plot every time a slider moves.
ax = fig.add_axes([0.06, 0.34, 0.84, 0.60], projection="3d")
cax = fig.add_axes([0.24, 0.292, 0.48, 0.020])

sliders = []
for i,(r,s) in enumerate(terms):
    y = 0.245 - i*0.030
    sax = fig.add_axes([0.19, y, 0.61, 0.018])
    sl = Slider(
        sax,
        rf"$\beta_{{{r}{s}}}$ / mm",
        -0.50, 0.50,
        valinit=0.0,
        valstep=0.01
    )
    sliders.append(sl)

def beautify_axes():
    ax.set_xlabel(r"$u$ / mm", labelpad=7)
    ax.set_ylabel(r"$v$ / mm", labelpad=7)
    ax.set_zlabel(r"$h(\xi,\eta)$ / mm", labelpad=8)
    ax.set_xlim(0,Lu_mm)
    ax.set_ylim(0,Lv_mm)
    ax.set_zlim(-3.0,3.0)
    ax.view_init(elev=27,azim=-58)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    pane = (0.985,0.985,0.985,1.0)
    ax.xaxis.set_pane_color(pane)
    ax.yaxis.set_pane_color(pane)
    ax.zaxis.set_pane_color(pane)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = (0.82,0.82,0.82,0.40)
        axis._axinfo["grid"]["linewidth"] = 0.5

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(labelsize=9, pad=1)

# Fixed normalization for interaction so the meaning of color is stable while dragging.
# ±3 mm is more than enough for beta in [-0.5, 0.5] mm under the present 7-mode example.
norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)
sm = ScalarMappable(norm=norm, cmap=soft_diverging)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label(r"Surface height $h$ / mm", fontsize=10)
cbar.ax.tick_params(labelsize=9)

surface_holder = {"surface": None}

def draw(_=None):
    beta = np.array([sl.val for sl in sliders])
    H = np.tensordot(beta, PHI, axes=(0,0))

    # Remove only the old surface. Do NOT clear the axis and do NOT recreate colorbar.
    if surface_holder["surface"] is not None:
        surface_holder["surface"].remove()

    surface_holder["surface"] = ax.plot_surface(
        U, V, H,
        cmap=soft_diverging,
        norm=norm,
        linewidth=0,
        antialiased=True,
        rcount=81,
        ccount=101,
        shade=False
    )

    beautify_axes()
    ax.set_title(
        r"$h_{\beta}(\xi,\eta)=\Phi(\xi,\eta)^\mathsf{T}\beta$",
        pad=12
    )
    fig.canvas.draw_idle()

for sl in sliders:
    sl.on_changed(draw)

reset_ax = fig.add_axes([0.83,0.055,0.10,0.045])
reset_btn = Button(reset_ax, "Reset")

def reset(_):
    for sl in sliders:
        sl.reset()

reset_btn.on_clicked(reset)

draw()
plt.show()
