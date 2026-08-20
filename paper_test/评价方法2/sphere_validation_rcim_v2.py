
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce the real-robot sphere-validation workflow used in:
Gautam et al., RCIM 2025, "Streamlined robotic hand–eye calibration of multiple 2D-profilers..."

Outputs:
  Fig15_sphere_validation.png  - one comprehensive validation figure for a single scanner
  metrics.txt                  - numerical validation metrics


Recommended input directory:
validation/
  scanner1/
    pose1.csv
    pose2.csv
    ...
    pose7.csv
  scanner2/
    pose1.csv
    ...
  scanner3/
    pose1.csv
    ...

Each CSV/TXT/NPY file must contain 3D reconstructed points in one common coordinate
frame, with columns x,y,z in millimetres.

For an eye-in-hand laser profiler, reconstruct each raw profile first:
    p_B = T_BF_i @ T_FS @ p_S
where p_S = [x, 0, z, 1]^T.

Paper reference sphere radius:
    r_ref = 17.4605 mm

Metrics:
    radius error Δr = r_fit - r_ref
    form error       = std(||p-c_fit|| - r_fit)
    dispersion shell = 2 * form error
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


@dataclass
class SphereFit:
    center: np.ndarray
    radius: float
    residuals: np.ndarray

    @property
    def sigma(self) -> float:
        if len(self.residuals) < 2:
            return float("nan")
        return float(np.std(self.residuals, ddof=1))

    @property
    def dispersion_shell_2sigma(self) -> float:
        return 2.0 * self.sigma


def load_xyz(path: Path) -> np.ndarray:
    """Load Nx3 points from .csv, .txt, or .npy."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        pts = np.asarray(np.load(path), dtype=float)
    else:
        # Try comma-separated data first. np.genfromtxt tolerates a header such as x,y,z.
        try:
            pts = np.genfromtxt(path, delimiter=",", dtype=float)
            pts = np.atleast_2d(pts)
            # Drop rows that are entirely NaN (e.g. a text header).
            pts = pts[~np.all(~np.isfinite(pts), axis=1)]
            if pts.size == 0 or pts.shape[1] < 3:
                raise ValueError("not a valid comma-separated XYZ file")
        except Exception:
            pts = np.genfromtxt(path, dtype=float)
            pts = np.atleast_2d(pts)
            pts = pts[~np.all(~np.isfinite(pts), axis=1)]

    pts = np.atleast_2d(pts)
    if pts.shape[1] < 3:
        raise ValueError(f"{path}: expected at least 3 columns [x,y,z], got {pts.shape}")
    pts = pts[:, :3]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 4:
        raise ValueError(f"{path}: not enough valid points for sphere fitting.")
    return pts


def transform_profile_eye_in_hand(points_s: np.ndarray,
                                  T_BF: np.ndarray,
                                  T_FS: np.ndarray) -> np.ndarray:
    """
    Convert one 2D-profiler frame from sensor frame S to robot base B.

    points_s:
      Nx2 -> interpreted as [x,z] and converted to [x,0,z]
      Nx3 -> interpreted as [x,y,z]
    T_BF: 4x4, base <- flange
    T_FS: 4x4, flange <- sensor
    """
    points_s = np.asarray(points_s, dtype=float)
    T_BF = np.asarray(T_BF, dtype=float).reshape(4, 4)
    T_FS = np.asarray(T_FS, dtype=float).reshape(4, 4)

    if points_s.ndim != 2 or points_s.shape[1] not in (2, 3):
        raise ValueError("points_s must be Nx2 [x,z] or Nx3 [x,y,z].")

    if points_s.shape[1] == 2:
        pts3 = np.column_stack([points_s[:, 0],
                                np.zeros(len(points_s)),
                                points_s[:, 1]])
    else:
        pts3 = points_s

    ph = np.column_stack([pts3, np.ones(len(pts3))])
    T_BS = T_BF @ T_FS
    pb = (T_BS @ ph.T).T[:, :3]
    return pb


def algebraic_sphere_init(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Algebraic least-squares initialization:
      x^2+y^2+z^2 - 2cx*x - 2cy*y - 2cz*z + d = 0
    """
    p = np.asarray(points, dtype=float)
    A = np.column_stack([2*p[:, 0], 2*p[:, 1], 2*p[:, 2], np.ones(len(p))])
    b = np.sum(p*p, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    k = sol[3]
    r2 = float(np.dot(c, c) + k)
    r = np.sqrt(max(r2, 0.0))
    return c, float(r)


def fit_sphere(points: np.ndarray, geometric_refine: bool = True) -> SphereFit:
    """
    Least-squares sphere regression.

    Paper-compatible default:
      - algebraic least-squares initialization
      - geometric least-squares refinement if SciPy is available
    """
    p = np.asarray(points, dtype=float)
    c0, r0 = algebraic_sphere_init(p)

    if geometric_refine and least_squares is not None:
        x0 = np.r_[c0, r0]

        def fun(x):
            c = x[:3]
            r = x[3]
            return np.linalg.norm(p - c, axis=1) - r

        # Linear loss to stay close to the paper's "least-squares sphere regression".
        res = least_squares(fun, x0, loss="linear")
        c = res.x[:3]
        r = abs(float(res.x[3]))
    else:
        c, r = c0, r0

    residuals = np.linalg.norm(p - c, axis=1) - r
    return SphereFit(center=np.asarray(c), radius=float(r), residuals=residuals)


def sphere_metrics(points: np.ndarray, r_ref: float) -> Dict[str, float]:
    fit = fit_sphere(points)
    dr = fit.radius - r_ref
    return {
        "cx": float(fit.center[0]),
        "cy": float(fit.center[1]),
        "cz": float(fit.center[2]),
        "r_fit": float(fit.radius),
        "radius_error_signed": float(dr),
        "radius_error_abs": float(abs(dr)),
        "form_error_sigma": float(fit.sigma),
        "dispersion_shell_2sigma": float(fit.dispersion_shell_2sigma),
    }


def _set_equal_3d(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ctr = (mins + maxs) / 2
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    half = span / 2
    ax.set_xlim(ctr[0]-half, ctr[0]+half)
    ax.set_ylim(ctr[1]-half, ctr[1]+half)
    ax.set_zlim(ctr[2]-half, ctr[2]+half)


def plot_fig14(points: np.ndarray, r_ref: float, out_path: Path):
    """
    Fig.14-like plot:
      radial-distance distribution
      reference radius / fitted radius
      size error and 2σ dispersion shell
      small 3D point-cloud inset
    """
    fit = fit_sphere(points)
    radial = np.linalg.norm(points - fit.center, axis=1)

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_axes([0.08, 0.15, 0.62, 0.75])

    ax.hist(radial, bins=35, density=True, alpha=0.75)
    ax.axvline(r_ref, linestyle="--", linewidth=1.5,
               label=f"Reference radius = {r_ref:.4f} mm")
    ax.axvline(fit.radius, linestyle="-.", linewidth=1.5,
               label=f"Fitted radius = {fit.radius:.4f} mm")

    sigma = fit.sigma
    ax.axvspan(fit.radius - sigma, fit.radius + sigma, alpha=0.15,
               label=f"2σ shell = {2*sigma:.4f} mm")

    ax.set_xlabel("Radial distance to fitted center (mm)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Fig.14-like: size error and sphere-form dispersion\n"
        f"|Δr| = {abs(fit.radius-r_ref):.4f} mm, "
        f"form error σ = {sigma:.4f} mm"
    )
    ax.legend()

    ax3 = fig.add_axes([0.74, 0.18, 0.23, 0.68], projection="3d")
    stride = max(1, len(points)//2500)
    q = points[::stride]
    ax3.scatter(q[:, 0], q[:, 1], q[:, 2], s=1.0, alpha=0.6)
    ax3.scatter([fit.center[0]], [fit.center[1]], [fit.center[2]], s=25)
    _set_equal_3d(ax3, q)
    ax3.set_title("Reconstructed sphere")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fit



def plot_fig15(pose_points: Sequence[np.ndarray], r_ref: float, out_path: Path):
    """
    Comprehensive single-scanner validation figure.

    (a) Independent validation poses:
        reconstructed sphere scans + fitted radius/center.

    (b) Pose-wise quantitative comparison:
        absolute radius error |Δr_k| with ±σ_k radial-dispersion error bars.

    (c) All poses merged:
        each reconstructed point is colored by its radial error relative to
        the REFERENCE sphere radius,

            e_j = ||p_j - c_all|| - r_ref

        where c_all is the fitted center from all merged points.

    Reported combined metrics:
        r_fit
        Δr = r_fit - r_ref
        sigma_e = std(e_j)
        RMSE_e = sqrt(mean(e_j^2))
    """
    n = len(pose_points)
    if n < 4:
        raise ValueError(f"Expected at least 4 validation poses, got {n}.")

    fits = [fit_sphere(p) for p in pose_points]

    combined = np.vstack(pose_points)
    fit_all = fit_sphere(combined)

    # Radial error to the REFERENCE sphere, using the center estimated
    # from all merged points.
    dist_all = np.linalg.norm(combined - fit_all.center, axis=1)
    radial_error_ref = dist_all - r_ref

    sigma_e = float(np.std(radial_error_ref, ddof=1))
    rmse_e = float(np.sqrt(np.mean(radial_error_ref**2)))
    delta_r = float(fit_all.radius - r_ref)

    # ---- Figure layout ----
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        3, 7,
        height_ratios=[1.0, 0.85, 1.55],
        hspace=0.42, wspace=0.35
    )

    # ------------------------------------------------------------------
    # (a) Independent validation poses
    # ------------------------------------------------------------------
    for i, (pts, fit) in enumerate(zip(pose_points, fits)):
        ax = fig.add_subplot(gs[0, i])

        stride = max(1, len(pts)//1400)
        q = pts[::stride]

        # Y-Z view is close to the visual presentation in the paper.
        ax.scatter(q[:, 1], q[:, 2], s=1.2, alpha=0.65)
        ax.scatter([fit.center[1]], [fit.center[2]], s=18, marker="x")

        dr = fit.radius - r_ref
        ax.set_title(
            f"Pose {i+1}\n"
            f"r={fit.radius:.4f} mm\n"
            f"Δr={dr:+.4f} mm",
            fontsize=8
        )
        ax.set_xlabel("Y (mm)")
        if i == 0:
            ax.set_ylabel("Z (mm)")
        ax.set_aspect("equal", adjustable="box")

    # Label for panel (a)
    fig.text(
        0.015, 0.965,
        f"(a) Reconstructed sphere under {n} independent validation poses",
        fontsize=11, weight="bold", va="top"
    )

    # ------------------------------------------------------------------
    # (b) Pose-wise radius error and radial dispersion
    # ------------------------------------------------------------------
    axb = fig.add_subplot(gs[1, :])

    x = np.arange(1, n + 1)
    radius_error_abs = np.array([abs(f.radius - r_ref) for f in fits])
    sigma_pose = np.array([f.sigma for f in fits])

    axb.errorbar(
        x,
        radius_error_abs,
        yerr=sigma_pose,
        fmt="o-",
        capsize=4,
        linewidth=1.5
    )

    axb.set_xticks(x)
    axb.set_xlabel("Validation pose")
    axb.set_ylabel("Absolute radius error |Δr| (mm)")
    axb.set_title(
        "(b) Pose-wise radius error; error bars indicate radial dispersion σ",
        fontsize=11
    )
    axb.grid(True, alpha=0.25)

    for xi, er, sg in zip(x, radius_error_abs, sigma_pose):
        axb.annotate(
            f"{er:.3f}",
            (xi, er),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8
        )

    # ------------------------------------------------------------------
    # (c) Combined reconstruction, colored by radial error to reference
    # ------------------------------------------------------------------
    axc = fig.add_subplot(gs[2, :6], projection="3d")

    # Downsample only for rendering; fitting and metrics use all points.
    stride = max(1, len(combined)//12000)
    q = combined[::stride]
    q_err = radial_error_ref[::stride]

    # Symmetric color limits around zero make inward/outward errors intuitive.
    cmax = float(np.percentile(np.abs(q_err), 98))
    if not np.isfinite(cmax) or cmax <= 0:
        cmax = 1e-6

    sc = axc.scatter(
        q[:, 0], q[:, 1], q[:, 2],
        c=q_err,
        cmap="coolwarm",
        vmin=-cmax,
        vmax=cmax,
        s=3.0,
        alpha=0.85
    )

    axc.scatter(
        [fit_all.center[0]],
        [fit_all.center[1]],
        [fit_all.center[2]],
        s=35,
        marker="x"
    )

    _set_equal_3d(axc, q)
    axc.set_xlabel("X (mm)")
    axc.set_ylabel("Y (mm)")
    axc.set_zlabel("Z (mm)")
    axc.set_title(
        "(c) Combined reconstruction from all seven poses\n"
        "Point color = radial error relative to the reference sphere",
        fontsize=11
    )

    cbar = fig.colorbar(sc, ax=axc, fraction=0.028, pad=0.08)
    cbar.set_label("Radial error to reference sphere (mm)")

    # Metrics panel on the right of combined point cloud.
    axm = fig.add_subplot(gs[2, 6])
    axm.axis("off")
    metrics_text = (
        "Combined metrics\n\n"
        f"Reference radius\n"
        f"{r_ref:.4f} mm\n\n"
        f"Fitted radius\n"
        f"{fit_all.radius:.4f} mm\n\n"
        f"Signed radius error Δr\n"
        f"{delta_r:+.4f} mm\n\n"
        f"Radial dispersion σₑ\n"
        f"{sigma_e:.4f} mm\n\n"
        f"Radial RMSE\n"
        f"{rmse_e:.4f} mm\n\n"
        f"Fitted center\n"
        f"({fit_all.center[0]:.3f},\n"
        f" {fit_all.center[1]:.3f},\n"
        f" {fit_all.center[2]:.3f}) mm"
    )
    axm.text(
        0.0, 1.0, metrics_text,
        va="top", ha="left",
        fontsize=10,
        linespacing=1.25
    )

    fig.suptitle(
        "Sphere-based validation of hand-eye calibration",
        fontsize=14,
        y=0.995
    )

    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    combined_metrics = {
        "center": fit_all.center,
        "radius": fit_all.radius,
        "delta_r_signed": delta_r,
        "delta_r_abs": abs(delta_r),
        "sigma_e": sigma_e,
        "rmse_e": rmse_e,
        "radial_error_ref": radial_error_ref,
    }

    return fits, fit_all, combined_metrics

def plot_fig16(scanner_pose_points: Dict[str, Sequence[np.ndarray]],
               r_ref: float,
               out_path: Path):
    """
    Fig.16-like plot:
      each scanner: merged 7-pose sphere fit + residual histogram
      all scanners combined: merged sphere fit + residual histogram
    """
    names = list(scanner_pose_points.keys())
    if len(names) < 2:
        raise ValueError("Fig.16 requires at least two scanners.")

    scanner_clouds = {name: np.vstack(scanner_pose_points[name]) for name in names}
    scanner_fits = {name: fit_sphere(scanner_clouds[name]) for name in names}

    all_points = np.vstack([scanner_clouds[name] for name in names])
    fit_all = fit_sphere(all_points)

    n = len(names)
    fig = plt.figure(figsize=(5.2*n, 9))
    gs = fig.add_gridspec(3, n)

    for i, name in enumerate(names):
        pts = scanner_clouds[name]
        fit = scanner_fits[name]

        ax = fig.add_subplot(gs[0, i])
        stride = max(1, len(pts)//2500)
        q = pts[::stride]
        ax.scatter(q[:, 1], q[:, 2], s=1.0, alpha=0.55)
        ax.scatter([fit.center[1]], [fit.center[2]], s=20)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.set_title(
            f"{name}\nr={fit.radius:.4f} mm\n"
            f"c=({fit.center[0]:.2f}, {fit.center[1]:.2f}, {fit.center[2]:.2f})"
        )

        axh = fig.add_subplot(gs[1, i])
        axh.hist(fit.residuals, bins=24, density=True, alpha=0.75)
        axh.axvline(0.0, linestyle="--", linewidth=1)
        axh.set_xlabel("Radial residual (mm)")
        axh.set_ylabel("Density")
        axh.set_title(
            f"Δr={fit.radius-r_ref:+.4f} mm, σ={fit.sigma:.4f} mm"
        )

    # Bottom row: combine all scanner data
    axc = fig.add_subplot(gs[2, :max(1, n//2 + n%2)])
    stride = max(1, len(all_points)//6000)
    q = all_points[::stride]
    axc.scatter(q[:, 1], q[:, 2], s=1.0, alpha=0.4)
    axc.scatter([fit_all.center[1]], [fit_all.center[2]], s=25)
    # scanner-specific fitted centers
    for name, fit in scanner_fits.items():
        axc.scatter([fit.center[1]], [fit.center[2]], s=20, marker="x")
        axc.text(fit.center[1], fit.center[2], name, fontsize=8)
    axc.set_aspect("equal", adjustable="box")
    axc.set_xlabel("Y")
    axc.set_ylabel("Z")
    axc.set_title(
        f"All scanners combined: r={fit_all.radius:.4f} mm"
    )

    axh = fig.add_subplot(gs[2, max(1, n//2 + n%2):])
    axh.hist(fit_all.residuals, bins=30, density=True, alpha=0.75)
    axh.axvline(0.0, linestyle="--", linewidth=1)
    axh.set_xlabel("Radial residual (mm)")
    axh.set_ylabel("Density")
    axh.set_title(
        f"Combined: Δr={fit_all.radius-r_ref:+.4f} mm, "
        f"σ={fit_all.sigma:.4f} mm"
    )

    fig.suptitle("Fig.16-like: multi-scanner calibration validation", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return scanner_fits, fit_all


def find_pose_files(scanner_dir: Path) -> List[Path]:
    files = []
    for i in range(1, 1000):
        candidates = []
        for ext in (".csv", ".txt", ".npy"):
            candidates += [
                scanner_dir / f"pose{i}{ext}",
                scanner_dir / f"Pose{i}{ext}",
                scanner_dir / f"pose_{i}{ext}",
                scanner_dir / f"Pose_{i}{ext}",
            ]
        found = [p for p in candidates if p.exists()]
        if not found:
            break
        files.append(found[0])
    if len(files) < 4:
        raise FileNotFoundError(
            f"Found only {len(files)} contiguous pose files under {scanner_dir}; "
            "at least pose1 ... pose4 are required."
        )
    return files


def load_dataset(root: Path) -> Dict[str, List[np.ndarray]]:
    root = Path(root)
    scanner_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not scanner_dirs:
        # Allow root itself to be a single scanner folder.
        scanner_dirs = [root]

    data = {}
    for sd in scanner_dirs:
        try:
            files = find_pose_files(sd)
        except FileNotFoundError:
            if sd == root:
                raise
            continue
        data[sd.name] = [load_xyz(p) for p in files]

    if not data:
        raise RuntimeError(
            f"No scanner datasets found under {root}. "
            "Expected at least scanner1/pose1.csv ... pose4.csv."
        )
    return data


def make_demo_data(root: Path, r_ref: float = 17.4605, seed: int = 7):
    """Generate synthetic test data only to verify the script and figure layout."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)

    pose_offsets = [
        (0, 0, 0),
        (10, 0, 0),
        (-10, 0, 0),
        (0, 10, 0),
        (0, -10, 0),
        (0, 0, 10),
        (0, 0, -10),
    ]

    for sidx in range(1, 4):
        sd = root / f"scanner{sidx}"
        sd.mkdir(exist_ok=True)
        sensor_bias = (sidx - 2) * 0.06

        for pidx, _angles in enumerate(pose_offsets, start=1):
            n = 2200
            # Partial sphere "visible cap" to mimic line-scan reconstruction.
            phi = rng.uniform(0.15, 1.45, n)
            th = rng.uniform(-1.25, 1.25, n)
            rr = r_ref + sensor_bias + rng.normal(0, 0.055 + 0.01*sidx, n)

            x = rr * np.sin(phi) * np.cos(th)
            y = rr * np.sin(phi) * np.sin(th)
            z = rr * np.cos(phi)
            pts = np.column_stack([x, y, z])

            # Pose-dependent small bias, as seen in real validation data.
            pts += np.array([100.0, 100.0, 42.0])
            pts += rng.normal(0, 0.015*pidx, size=pts.shape)

            np.savetxt(sd / f"pose{pidx}.csv", pts, delimiter=",",
                       header="x,y,z", comments="")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path, default=Path("validation"),
        help="Validation data root. Provide at least four contiguous pose files under this directory or one scanner subfolder."
    )
    ap.add_argument(
        "--out", type=Path, default=Path("validation_figures"),
        help="Output directory."
    )
    ap.add_argument(
        "--radius", type=float, default=17.4605,
        help="Reference sphere radius in mm."
    )
    ap.add_argument(
        "--demo", action="store_true",
        help="Generate synthetic demo data under --root first."
    )
    args = ap.parse_args()

    if args.demo:
        make_demo_data(args.root, args.radius)

    data = load_dataset(args.root)
    args.out.mkdir(parents=True, exist_ok=True)

    # This version intentionally evaluates ONLY ONE scanner.
    # If multiple scanner folders are present, the first one is used.
    first_name = next(iter(data))
    pose_points = data[first_name]

    fits, fit_all, combined_metrics = plot_fig15(
        pose_points,
        args.radius,
        args.out / "Fig15_sphere_validation.png"
    )

    # Save numeric report.
    report_lines = []
    report_lines.append(f"Scanner: {first_name}\n")
    report_lines.append(f"Reference radius: {args.radius:.6f} mm\n\n")

    report_lines.append("[Per-pose validation]\n")
    for i, fit in enumerate(fits, 1):
        dr = fit.radius - args.radius
        report_lines.append(
            f"Pose{i}: "
            f"center=({fit.center[0]:.6f}, {fit.center[1]:.6f}, {fit.center[2]:.6f}) mm, "
            f"r_fit={fit.radius:.6f} mm, "
            f"delta_r={dr:+.6f} mm, "
            f"abs_delta_r={abs(dr):.6f} mm, "
            f"sigma_fit={fit.sigma:.6f} mm\n"
        )

    report_lines.append(f"\n[All {len(pose_points)} poses combined]\n")
    report_lines.append(
        f"center=({fit_all.center[0]:.6f}, {fit_all.center[1]:.6f}, {fit_all.center[2]:.6f}) mm\n"
    )
    report_lines.append(f"r_fit={fit_all.radius:.6f} mm\n")
    report_lines.append(
        f"delta_r_signed={combined_metrics['delta_r_signed']:+.6f} mm\n"
    )
    report_lines.append(
        f"delta_r_abs={combined_metrics['delta_r_abs']:.6f} mm\n"
    )
    report_lines.append(
        f"sigma_e={combined_metrics['sigma_e']:.6f} mm\n"
    )
    report_lines.append(
        f"rmse_e={combined_metrics['rmse_e']:.6f} mm\n"
    )
    report_lines.append(
        "\nDefinitions:\n"
        "  radial error e_j = ||p_j - c_all|| - r_ref\n"
        "  sigma_e = std(e_j)\n"
        "  RMSE_e = sqrt(mean(e_j^2))\n"
    )

    (args.out / "metrics.txt").write_text(
        "".join(report_lines),
        encoding="utf-8"
    )

    print(f"Saved figure: {(args.out / 'Fig15_sphere_validation.png').resolve()}")
    print(f"Saved metrics: {(args.out / 'metrics.txt').resolve()}")


if __name__ == "__main__":
    main()
