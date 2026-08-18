#!/usr/bin/env python3
"""美化版 Fig15 排版：复用 sphere_validation_rcim_v2 的算法，重新设计布局。

布局改进：
  (a) 7 个 pose 的 Y-Z 重建视图：统一坐标范围、参考半径圆、拟合中心标记
  (b) pose-wise |Δr| + σ 误差条：均值带、参考线、数值标注
  (c) 合并 3D 点云：径向误差着色 + 拟合球面网格 + 右侧 metrics 卡片
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

sys.path.insert(0, "/workspace/tmp_diag")
from sphere_validation_rcim_v2 import (
    fit_sphere,
    load_dataset,
    load_xyz,
    find_pose_files,
)

# ---------- 全局样式 ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#4a4a4a",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.alpha": 0.45,
    "grid.linewidth": 0.6,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3"]
ACCENT = "#C44E52"
TRUE_COLOR = "#2ca02c"
REF_CIRCLE_COLOR = "#999999"


def _sphere_mesh(center, radius, n=16):
    """生成 3D 球面网格顶点/面（用于半透明拟合球显示）。"""
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)
    x = center[0] + radius * np.outer(np.sin(u), np.cos(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.cos(u), np.ones_like(v))
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            quad = [
                [x[i, j], y[i, j], z[i, j]],
                [x[i, j + 1], y[i, j + 1], z[i, j + 1]],
                [x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1]],
                [x[i + 1, j], y[i + 1, j], z[i + 1, j]],
            ]
            faces.append(quad)
    return faces


def _set_3d_equal(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ctr = (mins + maxs) / 2
    span = float(np.max(maxs - mins))
    half = span / 2
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)


def plot_fig15_beautiful(pose_points, r_ref, out_path):
    n = len(pose_points)
    if n != 7:
        raise ValueError(f"Expected exactly 7 validation poses, got {n}.")
    fits = [fit_sphere(p) for p in pose_points]
    combined = np.vstack(pose_points)
    fit_all = fit_sphere(combined)

    dist_all = np.linalg.norm(combined - fit_all.center, axis=1)
    radial_error_ref = dist_all - r_ref
    sigma_e = float(np.std(radial_error_ref, ddof=1))
    rmse_e = float(np.sqrt(np.mean(radial_error_ref ** 2)))
    delta_r = float(fit_all.radius - r_ref)

    # 统一 Y-Z 范围（(a) 面板）
    all_yz = combined[:, 1:]
    yz_min = all_yz.min(axis=0)
    yz_max = all_yz.max(axis=0)
    yz_ctr = (yz_min + yz_max) / 2
    yz_span = float(np.max(yz_max - yz_min))
    yz_half = yz_span * 0.55

    fig = plt.figure(figsize=(16, 11.5))
    # 8 列网格：让 (c) 3D 图在底部行居中，metrics 卡片占右侧 2 列
    gs = GridSpec(
        3, 8,
        height_ratios=[1.0, 0.62, 1.35],
        hspace=0.62, wspace=0.35,
        left=0.05, right=0.97, top=0.93, bottom=0.09,
    )

    # ================= (a) 七个独立位姿 =================
    for i, (pts, fit) in enumerate(zip(pose_points, fits)):
        ax = fig.add_subplot(gs[0, i])
        stride = max(1, len(pts) // 1400)
        q = pts[::stride]
        ax.scatter(q[:, 1], q[:, 2], s=2.2, alpha=0.55, color=PALETTE[i],
                   edgecolors="none", rasterized=True)
        # 参考半径圆（以拟合中心为圆心）
        circ = Circle((fit.center[1], fit.center[2]), r_ref,
                      fill=False, linestyle="--", linewidth=1.0,
                      edgecolor=REF_CIRCLE_COLOR, alpha=0.9)
        ax.add_patch(circ)
        ax.scatter([fit.center[1]], [fit.center[2]], s=42, marker="x",
                   color=ACCENT, linewidths=1.8, zorder=5)
        dr = fit.radius - r_ref
        ax.set_xlim(yz_ctr[0] - yz_half, yz_ctr[0] + yz_half)
        ax.set_ylim(yz_ctr[1] - yz_half, yz_ctr[1] + yz_half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        if i == 0:
            ax.set_yticks([])
        else:
            ax.set_yticks([])
        ax.set_title(
            f"Pose {i + 1}\n"
            f"$r$ = {fit.radius:.4f} mm\n"
            f"$\\Delta r$ = {dr:+.4f} mm",
            fontsize=8.5, pad=6,
        )

    # (a) 面板标签
    fig.text(0.012, 0.965, "(a)  Reconstructed sphere under seven independent validation poses",
             fontsize=12, fontweight="bold", va="top", color="#222222")

    # ================= (b) pose-wise 半径误差 =================
    axb = fig.add_subplot(gs[1, :])
    x = np.arange(1, 8)
    radius_error_abs = np.array([abs(f.radius - r_ref) for f in fits])
    sigma_pose = np.array([f.sigma for f in fits])

    axb.errorbar(x, radius_error_abs, yerr=sigma_pose, fmt="o-",
                 capsize=4, linewidth=1.6, color=PALETTE[0],
                 markeredgecolor="white", markeredgewidth=0.8,
                 markersize=7, ecolor="#888888", elinewidth=1.2,
                 zorder=3)

    mean_d = float(np.mean(radius_error_abs))
    std_d = float(np.std(radius_error_abs))
    axb.fill_between([0.6, 7.4], mean_d - std_d, mean_d + std_d,
                     color=PALETTE[0], alpha=0.10, zorder=1,
                     label=f"Mean ± 1σ  ({mean_d:.4f} ± {std_d:.4f} mm)")
    axb.axhline(mean_d, color=PALETTE[0], linestyle="--", linewidth=1.2,
                alpha=0.7, zorder=2)

    for xi, er, sg in zip(x, radius_error_abs, sigma_pose):
        axb.annotate(f"{er:.4f}", (xi, er + sg),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8.5, color="#333333")

    axb.set_xticks(x)
    axb.set_xlim(0.5, 7.5)
    axb.set_xlabel("Validation pose", fontsize=11)
    axb.set_ylabel("Absolute radius error |Δr| (mm)", fontsize=11)
    axb.set_title("(b)  Pose-wise radius error; error bars indicate radial dispersion σ",
                  fontsize=11.5, fontweight="bold", pad=8, loc="left")
    axb.legend(loc="upper right", fontsize=9, framealpha=0.9)
    axb.set_axisbelow(True)

    # ================= (c) 合并 3D 重建（居中） =================
    axc = fig.add_subplot(gs[2, 1:6], projection="3d")
    stride = max(1, len(combined) // 12000)
    q = combined[::stride]
    q_err = radial_error_ref[::stride]
    cmax = float(np.percentile(np.abs(q_err), 98))
    if not np.isfinite(cmax) or cmax <= 0:
        cmax = 1e-6

    sc = axc.scatter(q[:, 0], q[:, 1], q[:, 2], c=q_err,
                     cmap="coolwarm", vmin=-cmax, vmax=cmax,
                     s=2.5, alpha=0.85, edgecolors="none", rasterized=True)

    # 半透明拟合球面
    faces = _sphere_mesh(fit_all.center, fit_all.radius, n=20)
    mesh = Poly3DCollection(faces, alpha=0.05, facecolor="#4C72B0",
                            edgecolor="#4C72B0", linewidth=0.15)
    axc.add_collection3d(mesh)
    axc.scatter([fit_all.center[0]], [fit_all.center[1]], [fit_all.center[2]],
                s=55, marker="x", color=ACCENT, linewidths=2.0, zorder=6)

    _set_3d_equal(axc, q)
    axc.set_xlabel("X (mm)", fontsize=10)
    axc.set_ylabel("Y (mm)", fontsize=10)
    axc.set_zlabel("Z (mm)", fontsize=10)
    axc.set_title("(c)  Combined reconstruction from all seven poses\n"
                  "Point color = radial error relative to the reference sphere",
                  fontsize=11.5, fontweight="bold", pad=10, loc="left")
    axc.grid(True, alpha=0.2)

    cbar = fig.colorbar(sc, ax=axc, fraction=0.032, pad=0.06)
    cbar.set_label("Radial error to reference sphere (mm)", fontsize=10)
    cbar.outline.set_linewidth(0.6)

    # ================= metrics 卡片（右侧 2 列，不重叠） =================
    axm = fig.add_subplot(gs[2, 6:])
    axm.axis("off")
    axm.set_xlim(0, 1)
    axm.set_ylim(0, 1)

    axm.text(0.03, 0.955, "Combined metrics", fontsize=13,
             fontweight="bold", color="#222222", va="top")

    rows = [
        ("Reference radius", f"{r_ref:.4f} mm"),
        ("Fitted radius", f"{fit_all.radius:.4f} mm"),
        ("Signed radius error Δr", f"{delta_r:+.4f} mm"),
        ("Radial dispersion σₑ", f"{sigma_e:.4f} mm"),
        ("Radial RMSE", f"{rmse_e:.4f} mm"),
    ]
    row_h = 0.088
    start_y = 0.86
    for i, (label, value) in enumerate(rows):
        yy = start_y - i * row_h
        axm.text(0.03, yy, label, fontsize=10.5, color="#555555",
                 va="center")
        axm.text(0.97, yy, value, fontsize=11, color="#111111",
                 va="center", ha="right", fontweight="bold")

    # 球心单独放：三行数字逐行写，避免多行 text 与相邻行重叠
    cy = start_y - len(rows) * row_h - 0.05
    axm.text(0.03, cy + 0.055, "Fitted center (X, Y, Z)", fontsize=10.5,
             color="#555555", va="center")
    center_lines = [
        f"X   {fit_all.center[0]:.4f} mm",
        f"Y   {fit_all.center[1]:.4f} mm",
        f"Z   {fit_all.center[2]:.4f} mm",
    ]
    for j, line in enumerate(center_lines):
        axm.text(0.03, cy - j * 0.075, line, fontsize=11,
                 color="#111111", va="center", fontweight="bold")

    fig.suptitle("Sphere-based validation of hand–eye calibration",
                 fontsize=15, fontweight="bold", y=0.985, color="#111111")

    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    return fits, fit_all, {
        "center": fit_all.center,
        "radius": fit_all.radius,
        "delta_r_signed": delta_r,
        "delta_r_abs": abs(delta_r),
        "sigma_e": sigma_e,
        "rmse_e": rmse_e,
        "radial_error_ref": radial_error_ref,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("Fig15_beautiful.png"))
    ap.add_argument("--radius", type=float, default=17.4605)
    args = ap.parse_args()

    data = load_dataset(args.root)
    first_name = next(iter(data))
    pose_points = data[first_name]
    print(f"Scanner: {first_name}, {len(pose_points)} poses")

    fits, fit_all, metrics = plot_fig15_beautiful(
        pose_points, args.radius, args.out
    )
    print(f"Saved: {args.out.resolve()}")
    print(f"Combined: r_fit={fit_all.radius:.4f} mm, "
          f"Δr={metrics['delta_r_signed']:+.4f} mm, "
          f"σₑ={metrics['sigma_e']:.4f} mm")


if __name__ == "__main__":
    main()
