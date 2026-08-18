#!/usr/bin/env python3
"""图14/15/16 论文风格可视化（Gautam et al. RCIM 2025 §8）。

复现论文三张图：
  Fig.14: 球扫描点分布（3D 散点，按组着色）+ size error / form error 标注
  Fig.15: 单传感器验证（Top: 每组半径+球心；Middle: 平均误差+std；Bottom: 组合数据）
  Fig.16: 多传感器验证（Top: 每个传感器/候选结果；Bottom: 组合数据对比）

输入：评价方法2结果 json（method2_result.json）或直接 npz+手眼。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.sphere_validation import (
    SphereArtifact,
    _free_sphere,
    transform_profile_to_base,
)

TRUE_RADIUS_MM = 10.001
TRUE_DIAMETER_MM = 20.002
ARTIFACT = SphereArtifact(
    artifact_id="sphere_20mm", model="D20GZ",
    diameter_m=0.020002, roundness_m=0.0007,
)

# 论文图配色（接近原图风格）
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def load_base_groups(npz_path: Path, R_fs, t_fs):
    npz = np.load(str(npz_path), allow_pickle=True)
    offsets = npz["selected_frame_offsets"]
    points = npz["selected_points_sensor_m"]
    fpi = npz["frame_pose_indices"]
    rots = npz["flange_rotations"]
    ts = npz["flange_translations_m"]
    groups = []
    for pose_index in np.unique(fpi):
        frames = np.where(fpi == pose_index)[0]
        pose_pts = []
        for f in frames:
            pts_s = points[offsets[f]:offsets[f + 1]]
            pts_s = pts_s[np.all(np.isfinite(pts_s), axis=1)]
            if len(pts_s) == 0:
                continue
            pose_pts.append(transform_profile_to_base(
                pts_s, rots[f], ts[f], R_fs, t_fs
            ))
        if pose_pts:
            groups.append(np.vstack(pose_pts))
    return groups


def fit_group(points, robust_scale_m=0.00010):
    center, radius = _free_sphere(points, robust_scale_m=robust_scale_m)
    residual = np.linalg.norm(points - center, axis=1) - radius
    return {
        "center": center,
        "radius_mm": 1000.0 * float(radius),
        "delta_r_mm": 1000.0 * float(radius) - TRUE_RADIUS_MM,
        "form_mm": 1000.0 * float(np.max(residual) - np.min(residual)),
        "rmse_mm": 1000.0 * float(np.sqrt(np.mean(residual ** 2))),
    }


def fig14(groups, combined, out_path: Path, label: str = ""):
    """Fig.14: 球扫描点分布 + size/dispersion 标注。"""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    all_pts = np.vstack(groups)
    center = all_pts.mean(axis=0)
    # 去均值便于观察
    pts = all_pts - center
    for gi, group in enumerate(groups):
        g = group - center
        ax.scatter(g[::3, 0] * 1000, g[::3, 1] * 1000, g[::3, 2] * 1000,
                   s=0.5, c=COLORS[gi % len(COLORS)], alpha=0.6,
                   label=f"Set {gi + 1}")
    c = combined
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Fig.14  Distribution of points from scanning the precision sphere\n"
                 f"size error: {c['size_error_mm']:+.4f} mm, "
                 f"form error: {c['form_mm']:.4f} mm")
    ax.legend(markerscale=4, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Fig.14 saved: {out_path}")


def fig15(per_group, combined, summary, out_path: Path, label: str = ""):
    """Fig.15: 单传感器。Top=每组半径+球心；Middle=平均误差+std；Bottom=组合。"""
    fig, axes = plt.subplots(3, 1, figsize=(8, 11))

    # Top: 每组拟合半径（误差条显示 form error 范围）和球心偏移
    ax = axes[0]
    radii = np.array([g["radius_mm"] for g in per_group])
    forms = np.array([g["form_mm"] for g in per_group])
    xs = np.arange(1, len(per_group) + 1)
    ax.errorbar(xs, radii, yerr=forms / 2.0, fmt="o-", capsize=3,
                color="#1f77b4", ecolor="#999999", label="Fitted radius")
    ax.axhline(TRUE_RADIUS_MM, color="red", linestyle="--", linewidth=1.2,
               label=f"True radius ({TRUE_RADIUS_MM} mm)")
    ax.set_ylabel("Radius (mm)")
    ax.set_xlabel("Scan set")
    ax.set_title("Estimated radius and center for each set")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: 半径偏差（平均误差 + std 误差条）
    ax = axes[1]
    deltas = np.array([g["delta_r_mm"] for g in per_group])
    ax.bar(xs, deltas, color="#ff7f0e", alpha=0.85, width=0.6)
    ax.axhline(0.0, color="black", linewidth=1.0)
    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas))
    ax.axhline(mean_d, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean Δr = {mean_d:+.4f} mm")
    ax.fill_between([0.4, len(per_group) + 0.6],
                    mean_d - std_d, mean_d + std_d,
                    color="red", alpha=0.15, label=f"±1σ = {std_d:.4f} mm")
    ax.set_ylabel("Δr (mm)")
    ax.set_xlabel("Scan set")
    ax.set_title("Mean error and std")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: 组合数据拟合结果表格
    ax = axes[2]
    ax.axis("off")
    c = combined
    rows = [
        ["Combined data", ""],
        ["Points", f"{c['point_count']}"],
        ["Fitted radius (mm)", f"{c['radius_mm']:.4f}"],
        ["Size error (mm)", f"{c['size_error_mm']:+.4f}"],
        ["Form error (mm)", f"{c['form_mm']:.4f}"],
        ["Dispersion RMSE (mm)", f"{c['rmse_mm']:.4f}"],
        ["Center (base, m)",
         f"({c['center'][0]:.4f}, {c['center'][1]:.4f}, {c['center'][2]:.4f})"],
    ]
    table = ax.table(cellText=rows, colWidths=[0.35, 0.5],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    ax.set_title("Results for combined data", fontsize=12)

    fig.suptitle(f"Fig.15  Single-sensor sphere validation{label}", y=0.995,
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Fig.15 saved: {out_path}")


def fig16(candidates: dict, out_path: Path):
    """Fig.16: 多传感器/多候选。Top=各候选独立结果；Bottom=组合对比。"""
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))

    # Top: 每个候选的合并拟合半径（误差条=form）
    ax = axes[0]
    names = list(candidates.keys())
    radii = [candidates[n]["combined"]["radius_mm"] for n in names]
    forms = [candidates[n]["combined"]["form_mm"] for n in names]
    xs = np.arange(len(names))
    ax.errorbar(xs, radii, yerr=[f / 2.0 for f in forms], fmt="o",
                capsize=5, markersize=8, color="#1f77b4", ecolor="#999999")
    ax.axhline(TRUE_RADIUS_MM, color="red", linestyle="--", linewidth=1.2,
               label=f"True radius ({TRUE_RADIUS_MM} mm)")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Fitted radius (mm)")
    ax.set_title("Results for each candidate (combined fit)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: 各候选 Δr 对比
    ax = axes[1]
    deltas = [candidates[n]["combined"]["delta_r_mm"] for n in names]
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
    ax.bar(xs, deltas, color=colors, alpha=0.85, width=0.55)
    ax.axhline(0.0, color="black", linewidth=1.0)
    for i, d in enumerate(deltas):
        ax.text(i, d + (0.005 if d >= 0 else -0.02), f"{d:+.4f}",
                ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Δr (mm)")
    ax.set_title("Combined data: radius deviation from true value")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fig.16  Multi-sensor validation (candidate comparison)",
                 y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Fig.16 saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="图14/15/16 论文风格可视化")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--outdir", type=Path,
                        default=Path("/workspace/tmp_diag/paper_figs"))
    args = parser.parse_args()

    result = json.load(open(args.result))
    R_fs = np.array(result["handeye"]["rotation"])
    t_fs = np.array(result["handeye"]["translation"])
    groups = load_base_groups(args.npz, R_fs, t_fs)
    print(f"loaded {len(groups)} groups")

    per_group = [fit_group(g) for g in groups]
    all_pts = np.vstack(groups)
    combined = fit_group(all_pts)
    combined["size_error_mm"] = 2.0 * combined["delta_r_mm"]
    combined["point_count"] = len(all_pts)
    summary = {
        "mean_delta_r_mm": float(np.mean([g["delta_r_mm"] for g in per_group])),
        "std_delta_r_mm": float(np.std([g["delta_r_mm"] for g in per_group])),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.label}" if args.label else ""
    fig14(groups, combined, args.outdir / f"fig14{suffix}.png", args.label)
    fig15(per_group, combined, summary,
          args.outdir / f"fig15{suffix}.png", f" ({args.label})" if args.label else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
