#!/usr/bin/env python3
"""Reproducible progress-meeting analysis for the nine selected real runs.

The script never modifies source experiments.  It parses seed logs, seed JSON,
the string-valued seed-state topic in each MCAP bag, and the frozen A/B/C
morphology-ablation outputs, then writes derived tables and figures under
``group_meeting_paper_progress``.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


WORKSPACE = Path(__file__).resolve().parents[2]
OUT = WORKSPACE / "group_meeting_paper_progress"
FIGURES = OUT / "figures"
RESULTS = OUT / "results"
RUN_ROOT = WORKSPACE / "data/calibration_runs"
ABLATION = WORKSPACE / "data/ablation_runs/shared_morphology_real9_r5"

RUNS = [
    ("D1", "20260820_115707_圆点标定板背面_位置1_真机_2"),
    ("D2", "20260820_132928_圆点标定板背面_位置1_真机_3"),
    ("D3", "20260820_133622_圆点标定板背面_位置1_真机_5"),
    ("D4", "20260820_135111_圆点标定板背面_位置2_真机_1"),
    ("D5", "20260820_140109_圆点标定板背面_位置2_真机_2"),
    ("D6", "20260820_141546_原点标定板背面_位置3_真机_1"),
    ("D7", "20260820_142718_知象光电宣传册_位置1_真机_1_nbv初始化失败"),
    ("D8", "20260820_144753_焊接书_位置1_真机_1"),
    ("D9", "20260820_150032_生锈的刚板_位置1_真机_1"),
]

LABELS = [
    "reference",
    "rx_positive",
    "rx_negative",
    "ry_positive",
    "ry_negative",
    "rx_ry_positive",
]
SHORT_LABELS = {
    "reference": "Ref",
    "rx_positive": "Rx+",
    "rx_negative": "Rx-",
    "ry_positive": "Ry+",
    "ry_negative": "Ry-",
    "rx_ry_positive": "Rx+Ry+",
}

LOG_RE = re.compile(r"^\[(?:INFO|WARN|ERROR)\] \[([0-9.]+)\].*?: (.*)$")
SERVO_RE = re.compile(
    r"dual-feature servo: x_mid=([-0-9.]+) mm, length=([-0-9.]+) mm, "
    r"error=\[([-0-9.]+),([-0-9.]+)\] mm.*iteration=(\d+)/(\d+)"
)
ACCEPT_RE = re.compile(r"accepted physical seed (\d+): ([^;]+); endpoint inliers=(\d+)/(\d+)")
ROTATION_RE = re.compile(r"rotation command: step=([-0-9.]+) deg")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_seed_log(run_id: str, run_name: str) -> tuple[dict, list[dict], list[dict]]:
    path = RUN_ROOT / run_name / "seed_collection.log"
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = LOG_RE.match(raw)
        if match:
            lines.append((float(match.group(1)), match.group(2)))
    current_target = "reference"
    start_time = None
    completion_time = None
    target_starts: dict[str, float] = {}
    target_accepts: dict[str, float] = {}
    target_servo = defaultdict(int)
    target_rotation = defaultdict(int)
    target_rotation_deg = defaultdict(float)
    servo_rows: list[dict] = []
    accepted_rows: list[dict] = []
    rejected_updates = 0
    accepted_updates = 0
    rollback_count = 0
    recovery_count = 0
    unsafe_count = 0
    for timestamp, text in lines:
        if "stationary seed capture started: reference" in text and start_time is None:
            start_time = timestamp
            target_starts.setdefault("reference", timestamp)
        if text.startswith("target "):
            current_target = text.split("target ", 1)[1].strip()
            target_starts.setdefault(current_target, timestamp)
        match = SERVO_RE.search(text)
        if match:
            target_servo[current_target] += 1
            servo_rows.append(
                {
                    "run_id": run_id,
                    "dataset": run_name,
                    "target": current_target,
                    "time_from_seed_start_s": "" if start_time is None else timestamp - start_time,
                    "x_mid_mm": float(match.group(1)),
                    "length_mm": float(match.group(2)),
                    "x_error_mm": float(match.group(3)),
                    "length_error_mm": float(match.group(4)),
                    "iteration": int(match.group(5)),
                }
            )
        match = ROTATION_RE.search(text)
        if match:
            step = abs(float(match.group(1)))
            target_rotation[current_target] += 1
            target_rotation_deg[current_target] += step
        if "dual-feature model update: accepted=True" in text:
            accepted_updates += 1
        if "dual-feature model update: accepted=False" in text:
            rejected_updates += 1
        lower = text.lower()
        if "rollback, next rotation step" in lower:
            rollback_count += 1
        if "rollback restored" in lower or "recover" in lower or "reacqui" in lower:
            recovery_count += 1
        if "became unsafe" in lower or "unsafe bilateral" in lower:
            unsafe_count += 1
        match = ACCEPT_RE.search(text)
        if match:
            label = match.group(2)
            target_accepts[label] = timestamp
            accepted_rows.append(
                {
                    "run_id": run_id,
                    "dataset": run_name,
                    "seed_index": int(match.group(1)),
                    "target": label,
                    "target_duration_s": timestamp - target_starts.get(label, timestamp),
                    "servo_corrections": target_servo[label],
                    "rotation_commands": target_rotation[label],
                    "commanded_rotation_deg": target_rotation_deg[label],
                    "inlier_frames": int(match.group(3)),
                    "raw_frames": int(match.group(4)),
                }
            )
        if text.startswith("seed collection complete:"):
            completion_time = timestamp

    dataset = json.loads((RUN_ROOT / run_name / "seeds.json").read_text(encoding="utf-8"))
    diversity = dataset["rotation_diversity"]
    result_exists = (RUN_ROOT / run_name / "calibration_result.json").exists()
    active_log = RUN_ROOT / run_name / "active_calibration.log"
    active_text = active_log.read_text(encoding="utf-8") if active_log.exists() else ""
    initialization_failed = "initialization failed" in active_text
    active_lines = []
    for raw in active_text.splitlines():
        match = LOG_RE.match(raw)
        if match:
            active_lines.append((float(match.group(1)), match.group(2)))
    loaded_time = next((time for time, text in active_lines if text.startswith("loaded 6 physical seeds")), None)
    initialized_time = next((time for time, text in active_lines if "12-DOF-V2 initialized" in text), None)
    failed_time = next((time for time, text in active_lines if "initialization failed" in text), None)
    complete_time = next((time for time, text in active_lines if "real calibration complete" in text), None)
    nbv_commits = sum("NBV " in text and " committed:" in text for _, text in active_lines)
    nbv_rejections = sum("real observation rejected" in text for _, text in active_lines)
    early_stop = any("stopped early" in text for _, text in active_lines)
    initialization_duration = math.nan
    if loaded_time is not None and (initialized_time is not None or failed_time is not None):
        initialization_duration = (initialized_time if initialized_time is not None else failed_time) - loaded_time
    active_duration = math.nan
    if loaded_time is not None and (complete_time is not None or failed_time is not None):
        active_duration = (complete_time if complete_time is not None else failed_time) - loaded_time
    summary = {
        "run_id": run_id,
        "dataset": run_name,
        "artifact": (
            "brochure" if "宣传册" in run_name else
            "welding_book" if "焊接书" in run_name else
            "rusted_steel" if "刚板" in run_name else
            "calibration_board_back"
        ),
        "seed_collection_success": len(accepted_rows) == 6 and completion_time is not None,
        "physical_seed_count": dataset["physical_seed_count"],
        "synchronized_observations": dataset["observation_count"],
        "seed_duration_s": (
            math.nan if start_time is None or completion_time is None else completion_time - start_time
        ),
        "servo_corrections": sum(target_servo.values()),
        "rotation_commands": sum(target_rotation.values()),
        "commanded_rotation_deg": sum(target_rotation_deg.values()),
        "accepted_broyden_updates": accepted_updates,
        "rejected_broyden_updates": rejected_updates,
        "rollback_log_events": rollback_count,
        "recovery_log_events": recovery_count,
        "unsafe_log_events": unsafe_count,
        "minimum_pairwise_rotation_deg": diversity["minimum_pairwise_deg"],
        "minimum_rotation_gram_eigenvalue": diversity["minimum_gram_eigenvalue"],
        "flat_initialization_success": not initialization_failed,
        "initialization_duration_s": initialization_duration,
        "committed_nbv_count": nbv_commits,
        "rejected_nbv_batches": nbv_rejections,
        "active_stage_duration_s": active_duration,
        "active_stage_early_stop": early_stop,
        "final_result_available": result_exists,
    }
    return summary, accepted_rows, servo_rows


def seed_rotation_rows(run_id: str, run_name: str) -> list[dict]:
    data = json.loads((RUN_ROOT / run_name / "seeds.json").read_text(encoding="utf-8"))
    seeds = data["seeds"]
    reference = np.asarray(seeds[0]["R_BF"], dtype=float)
    rows = []
    for seed in seeds:
        rotation = np.asarray(seed["R_BF"], dtype=float)
        relative = Rotation.from_matrix(reference.T @ rotation)
        vector = np.rad2deg(relative.as_rotvec())
        rows.append(
            {
                "run_id": run_id,
                "dataset": run_name,
                "target": seed["label"],
                "rotvec_x_deg": vector[0],
                "rotvec_y_deg": vector[1],
                "rotvec_z_deg": vector[2],
                "relative_angle_deg": np.linalg.norm(vector),
            }
        )
    return rows


def read_seed_states(run_id: str, run_name: str) -> list[dict]:
    """Read only String seed states; all high-volume profile data stay untouched."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from std_msgs.msg import String
    except ImportError as exc:
        raise RuntimeError("source ROS 2 and use /usr/bin/python3") from exc
    bag = RUN_ROOT / run_name / "full_run_bag"
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    raw_rows = []
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic != "/calibration/seed_motion_state":
            continue
        try:
            payload = json.loads(deserialize_message(serialized, String).data)
        except (ValueError, json.JSONDecodeError):
            continue
        if not payload.get("started"):
            continue
        endpoints = payload.get("endpoints_S")
        if endpoints is None or len(endpoints) != 2:
            x_mid = z_mid = separation = math.nan
        else:
            endpoints_array = np.asarray(endpoints, dtype=float)
            x_mid = 500.0 * float(endpoints_array[0, 0] + endpoints_array[1, 0])
            z_mid = 500.0 * float(endpoints_array[0, 2] + endpoints_array[1, 2])
            separation = 1000.0 * float(np.linalg.norm(endpoints_array[1] - endpoints_array[0]))
        raw_rows.append(
            {
                "timestamp_s": timestamp_ns * 1e-9,
                "run_id": run_id,
                "dataset": run_name,
                "state": payload.get("state", ""),
                "phase": payload.get("phase", ""),
                "motion_stage": payload.get("motion_stage", ""),
                "target": payload.get("target", ""),
                "seed_count": payload.get("seed_count", 0),
                "accumulated_rotation_deg": payload.get("accumulated_rotation_deg", 0.0),
                "x_mid_mm": x_mid,
                "z_mid_mm": z_mid,
                "endpoint_separation_mm": separation,
                "servo_iteration": payload.get("servo_iteration", 0),
            }
        )
    if not raw_rows:
        return []
    origin = raw_rows[0]["timestamp_s"]
    # The state is published at high rate.  Retain transitions and at most 10 Hz.
    rows = []
    last_time = -math.inf
    last_signature = None
    for row in raw_rows:
        signature = (row["state"], row["target"], row["seed_count"], row["motion_stage"])
        if row["timestamp_s"] - last_time < 0.1 and signature == last_signature:
            continue
        row["time_from_seed_start_s"] = row["timestamp_s"] - origin
        rows.append(row)
        last_time = row["timestamp_s"]
        last_signature = signature
    return rows


def make_rq1_figures(
    run_rows: list[dict],
    target_rows: list[dict],
    servo_rows: list[dict],
    rotation_rows: list[dict],
    state_rows: list[dict],
) -> None:
    ids = [item[0] for item in RUNS]
    by_run = {row["run_id"]: row for row in run_rows}
    durations = [by_run[key]["seed_duration_s"] for key in ids]
    servos = [by_run[key]["servo_corrections"] for key in ids]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    colors = ["#4472C4" if key != "D7" else "#ED7D31" for key in ids]
    axes[0].bar(ids, durations, color=colors)
    axes[0].axhline(np.median(durations), color="#333333", linestyle="--", linewidth=1, label=f"median {np.median(durations):.1f} s")
    axes[0].set_ylabel("Seed acquisition time (s)")
    axes[0].legend(frameon=False)
    axes[0].set_title("All nine runs acquired six diverse seed poses")
    axes[1].bar(ids, servos, color="#70AD47")
    axes[1].set_ylabel("Reactive servo corrections")
    axes[1].set_xlabel("Real-data run")
    axes[1].text(0.01, 0.96, "D7: seeds succeeded; flat initialization failed later", transform=axes[1].transAxes, va="top", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "rq1_seed_success_time_operations")

    matrix = np.full((len(ids), len(LABELS)), np.nan)
    servo_matrix = np.full_like(matrix, np.nan)
    for row in target_rows:
        if row["target"] in LABELS:
            i = ids.index(row["run_id"])
            j = LABELS.index(row["target"])
            matrix[i, j] = float(row["target_duration_s"])
            servo_matrix[i, j] = float(row["servo_corrections"])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for ax, values, title, cmap in [
        (axes[0], matrix, "Per-seed elapsed time (s)", "Blues"),
        (axes[1], servo_matrix, "Reactive servo corrections", "Greens"),
    ]:
        image = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(LABELS)), [SHORT_LABELS[label] for label in LABELS], rotation=35, ha="right")
        ax.set_yticks(range(len(ids)), ids)
        ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i,j]:.0f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.78)
    fig.tight_layout()
    save_figure(fig, "rq1_per_target_time_and_servo")

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    colors_by_target = dict(zip(LABELS, ["#555555", "#4472C4", "#ED7D31", "#70AD47", "#A64D79", "#8064A2"]))
    for label in LABELS:
        points = np.asarray([[row["rotvec_x_deg"], row["rotvec_y_deg"]] for row in rotation_rows if row["target"] == label])
        ax.scatter(points[:, 0], points[:, 1], s=28, alpha=0.78, label=SHORT_LABELS[label], color=colors_by_target[label])
    ax.axhline(0, color="#BBBBBB", linewidth=0.8)
    ax.axvline(0, color="#BBBBBB", linewidth=0.8)
    ax.set_xlabel("Relative rotation-vector x (deg)")
    ax.set_ylabel("Relative rotation-vector y (deg)")
    ax.set_title("Measured six-pose excitation relative to the reference")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    save_figure(fig, "rq1_measured_pose_diversity")

    representative = [row for row in state_rows if row["run_id"] == "D1" and row["phase"] in {"REFERENCE", "COLLECT"}]
    if representative:
        time = np.asarray([row["time_from_seed_start_s"] for row in representative])
        xmid = np.asarray([row["x_mid_mm"] for row in representative])
        length = np.asarray([row["endpoint_separation_mm"] for row in representative])
        zmid = np.asarray([row["z_mid_mm"] for row in representative])
        fig, axes = plt.subplots(3, 1, figsize=(10.0, 6.6), sharex=True)
        axes[0].plot(time, xmid, color="#4472C4", linewidth=1)
        axes[0].axhspan(-10, 10, color="#4472C4", alpha=0.10, label="centering band")
        axes[0].set_ylabel("x_mid (mm)")
        axes[0].legend(frameon=False, loc="upper right")
        axes[1].plot(time, length, color="#70AD47", linewidth=1)
        axes[1].axhspan(70, 90, color="#70AD47", alpha=0.12, label="operating band")
        axes[1].axhline(50, color="#C00000", linestyle=":", linewidth=0.9)
        axes[1].axhline(120, color="#C00000", linestyle=":", linewidth=0.9, label="hard identity guard")
        axes[1].set_ylabel("Breakpoint span (mm)")
        axes[1].legend(frameon=False, loc="upper right")
        axes[2].plot(time, zmid, color="#ED7D31", linewidth=1)
        axes[2].axhspan(80, 450, color="#ED7D31", alpha=0.10, label="real depth envelope")
        axes[2].set_ylabel("z_mid (mm)")
        axes[2].set_xlabel("Time from seed start (s)")
        axes[2].legend(frameon=False, loc="upper right")
        previous = None
        for row in representative:
            if row["target"] != previous:
                for ax in axes:
                    ax.axvline(row["time_from_seed_start_s"], color="#999999", linewidth=0.5, alpha=0.6)
                axes[0].text(row["time_from_seed_start_s"], axes[0].get_ylim()[1], SHORT_LABELS.get(row["target"], row["target"]), rotation=90, va="top", ha="right", fontsize=7)
                previous = row["target"]
        fig.suptitle("Representative measured feedback timeline (D1)")
        fig.tight_layout()
        save_figure(fig, "rq1_feedback_timeline_D1")

    stages = [
        ("Seed acquisition", sum(bool(row["seed_collection_success"]) for row in run_rows)),
        ("Flat initialization", sum(bool(row["flat_initialization_success"]) for row in run_rows)),
        ("Final result file", sum(bool(row["final_result_available"]) for row in run_rows)),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    names = [item[0] for item in stages]
    values = [item[1] for item in stages]
    ax.barh(names[::-1], values[::-1], color=["#70AD47", "#ED7D31", "#4472C4"][::-1])
    ax.set_xlim(0, 9.5)
    ax.set_xlabel("Successful runs out of 9")
    for index, value in enumerate(values[::-1]):
        ax.text(value + 0.08, index, f"{value}/9", va="center")
    ax.set_title("Do not conflate acquisition success with solver success")
    fig.tight_layout()
    save_figure(fig, "rq1_pipeline_success_funnel")

    d1_servo = [row for row in servo_rows if row["run_id"] == "D1"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for target in LABELS[1:]:
        selected = [row for row in d1_servo if row["target"] == target]
        if not selected:
            continue
        index = np.arange(1, len(selected) + 1)
        axes[0].plot(index, [row["x_mid_mm"] for row in selected], marker="o", markersize=3, label=SHORT_LABELS[target])
        axes[1].plot(index, [row["length_mm"] for row in selected], marker="o", markersize=3, label=SHORT_LABELS[target])
    axes[0].axhspan(-10, 10, color="#4472C4", alpha=0.10)
    axes[0].set_ylabel("x_mid before correction (mm)")
    axes[1].axhspan(70, 90, color="#70AD47", alpha=0.12)
    axes[1].axhline(50, color="#C00000", linestyle=":", linewidth=0.8)
    axes[1].axhline(120, color="#C00000", linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("Breakpoint span before correction (mm)")
    for ax in axes:
        ax.set_xlabel("Reactive servo correction index within target")
        ax.legend(frameon=False, ncol=2)
    fig.suptitle("D1: active rotation creates feature error; feedback corrections restore the operating region")
    fig.tight_layout()
    save_figure(fig, "rq1_servo_convergence_D1")

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8))
    for ax, key, ylabel, bands in [
        (axes[0], "x_mid_mm", "x_mid (mm)", [(-10, 10, "#4472C4")]),
        (axes[1], "endpoint_separation_mm", "Breakpoint span (mm)", [(70, 90, "#70AD47")]),
        (axes[2], "z_mid_mm", "z_mid (mm)", [(80, 450, "#ED7D31")]),
    ]:
        values_by_run = []
        for run_id in ids:
            values_by_run.append(
                [float(row[key]) for row in state_rows if row["run_id"] == run_id and np.isfinite(float(row[key]))]
            )
        ax.boxplot(values_by_run, labels=ids, showfliers=False, widths=0.6)
        for lower, upper, color in bands:
            ax.axhspan(lower, upper, color=color, alpha=0.10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Run")
    axes[1].axhline(50, color="#C00000", linestyle=":", linewidth=0.8)
    axes[1].axhline(120, color="#C00000", linestyle=":", linewidth=0.8)
    fig.suptitle("Measured feedback distributions during all nine seed acquisitions")
    fig.tight_layout()
    save_figure(fig, "rq1_feedback_safety_all_runs")


def make_rq2_figures() -> list[dict]:
    with (ABLATION / "dataset_summary.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    usable = [row for row in rows if float(row["usable_solution_rate"]) > 0.0]
    write_csv(RESULTS / "rq2_abc_dataset_summary.csv", usable)
    ids = [item[0] for item in RUNS]
    dataset_to_id = {name: run_id for run_id, name in RUNS}
    group_order = ["A_ideal_plane", "B_shared_morphology", "C_pose_specific_morphology"]
    group_label = {"A_ideal_plane": "A Ideal", "B_shared_morphology": "B Shared", "C_pose_specific_morphology": "C Pose-specific"}
    colors = {"A_ideal_plane": "#4472C4", "B_shared_morphology": "#70AD47", "C_pose_specific_morphology": "#ED7D31"}
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    width = 0.24
    x = np.arange(len(ids))
    for offset, group in zip([-width, 0, width], group_order):
        values = []
        for run_id in ids:
            selected = [row for row in usable if dataset_to_id[row["dataset"]] == run_id and row["group"] == group]
            values.append(float(selected[0]["sphere_fixed_rmse_mm"]) if selected else np.nan)
        ax.bar(x + offset, values, width=width, label=group_label[group], color=colors[group])
    ax.set_xticks(x, ids)
    ax.set_ylabel("Independent sphere RMSE (mm)")
    ax.set_title("External validation: shared morphology is not consistently better than ideal plane")
    ax.legend(frameon=False, ncol=3)
    ax.text(6, ax.get_ylim()[1] * 0.92, "D7 initialization unavailable", ha="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "rq2_abc_external_sphere_rmse")

    with (ABLATION / "paired_a_b_differences.csv").open(encoding="utf-8-sig") as stream:
        paired = list(csv.DictReader(stream))
    paired = [
        row for row in paired
        if row.get("B_minus_A_sphere_fixed_rmse_mm") not in {None, "", "nan"}
    ]
    write_csv(RESULTS / "rq2_ab_paired_differences.csv", paired)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    run_ids = [dataset_to_id[row["dataset"]] for row in paired]
    sphere_delta = np.asarray(
        [float(row["B_minus_A_sphere_fixed_rmse_mm"]) for row in paired]
    )
    surface_delta = np.asarray(
        [float(row["B_minus_A_surface_residual_rms_mm"]) for row in paired]
    )
    for ax, values, ylabel, title in [
        (axes[0], sphere_delta, "B - A sphere RMSE (mm)", "External precision"),
        (axes[1], surface_delta, "B - A surface RMS (mm)", "Internal fit"),
    ]:
        ax.bar(run_ids, values, color=np.where(values < 0, "#70AD47", "#C0504D"))
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    axes[0].text(0.02, 0.96, "negative = B better", transform=axes[0].transAxes, va="top", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "rq2_ab_paired_internal_external")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    markers = {"A_ideal_plane": "o", "B_shared_morphology": "s", "C_pose_specific_morphology": "^"}
    for group in group_order:
        selected = [row for row in usable if row["group"] == group]
        ax.scatter(
            [float(row["surface_residual_rms_mm"]) for row in selected],
            [float(row["sphere_fixed_rmse_mm"]) for row in selected],
            label=group_label[group], marker=markers[group], color=colors[group], s=48, alpha=0.85,
        )
    ax.set_xlabel("Calibration-data surface residual RMS (mm)")
    ax.set_ylabel("Independent sphere RMSE (mm)")
    ax.set_title("Lower internal residual does not imply better hand-eye accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "rq2_internal_residual_vs_external_error")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    width = 0.24
    x = np.arange(len(ids))
    for offset, group in zip([-width, 0, width], group_order):
        rot_values = []
        trans_values = []
        for run_id in ids:
            selected = [row for row in usable if dataset_to_id[row["dataset"]] == run_id and row["group"] == group]
            rot_values.append(float(selected[0]["rotation_dispersion_deg"]) if selected and selected[0]["rotation_dispersion_deg"] else np.nan)
            trans_values.append(float(selected[0]["translation_dispersion_mm"]) if selected and selected[0]["translation_dispersion_mm"] else np.nan)
        axes[0].bar(x + offset, rot_values, width=width, label=group_label[group], color=colors[group])
        axes[1].bar(x + offset, trans_values, width=width, label=group_label[group], color=colors[group])
    axes[0].set_ylabel("Rotation dispersion (deg)")
    axes[1].set_ylabel("Translation dispersion (mm)")
    for ax in axes:
        ax.set_xticks(x, ids)
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Bootstrap hand-eye repeatability: pose-specific morphology is often less stable")
    fig.tight_layout()
    save_figure(fig, "rq2_handeye_repeatability_abc")

    trials = json.loads((ABLATION / "trial_results.json").read_text(encoding="utf-8"))
    shared = [row for row in trials if row["group"] == "B_shared_morphology" and row["repeat"] == 0 and row["usable_solution"]]
    # Reproduce the normalized degree-three basis without importing project code.
    grid_norm = np.linspace(0.0, 1.0, 81)
    gx, gy = np.meshgrid(grid_norm, grid_norm, indexing="ij")
    terms = [(xo, total - xo) for total in range(2, 4) for xo in range(total + 1)]
    def raw_basis(xi, eta):
        x = 2.0 * xi - 1.0
        y = 2.0 * eta - 1.0
        columns = []
        for xo, yo in terms:
            xc = np.zeros(xo + 1); xc[-1] = 1.0
            yc = np.zeros(yo + 1); yc[-1] = 1.0
            columns.append(np.polynomial.legendre.legval(x, xc) * np.polynomial.legendre.legval(y, yc))
        return np.stack(columns, axis=-1)
    scales = np.sqrt(np.mean(raw_basis(gx, gy) ** 2, axis=(0, 1)))
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.4), constrained_layout=True)
    surfaces = []
    for row in shared:
        coeff = np.asarray(row["shape_coefficients_m"], dtype=float)
        surface = 1000.0 * np.sum(raw_basis(gx, gy) / scales * coeff, axis=-1)
        surfaces.append((dataset_to_id[row["dataset"]], surface))
    vmax = max(np.max(np.abs(surface)) for _, surface in surfaces)
    for ax, (run_id, surface) in zip(axes.flat, surfaces):
        image = ax.imshow(surface.T, origin="lower", extent=[0, 200, 0, 150], cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(run_id)
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")
    for ax in axes.flat[len(surfaces):]:
        ax.axis("off")
    fig.colorbar(image, ax=axes, label="Estimated out-of-plane height (mm)", shrink=0.82)
    fig.suptitle("Shared morphology estimated independently from each usable run")
    save_figure(fig, "rq2_estimated_shared_surfaces")
    return usable


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_style()
    run_rows: list[dict] = []
    target_rows: list[dict] = []
    servo_rows: list[dict] = []
    rotation_rows: list[dict] = []
    state_rows: list[dict] = []
    for run_id, run_name in RUNS:
        print(f"Analyzing {run_id}: {run_name}", flush=True)
        summary, targets, servos = parse_seed_log(run_id, run_name)
        run_rows.append(summary)
        target_rows.extend(targets)
        servo_rows.extend(servos)
        rotation_rows.extend(seed_rotation_rows(run_id, run_name))
        state_rows.extend(read_seed_states(run_id, run_name))
    write_csv(RESULTS / "rq1_run_summary.csv", run_rows)
    write_csv(RESULTS / "rq1_target_summary.csv", target_rows)
    write_csv(RESULTS / "rq1_servo_events.csv", servo_rows)
    write_csv(RESULTS / "rq1_rotation_vectors.csv", rotation_rows)
    write_csv(RESULTS / "rq1_seed_state_timeseries.csv", state_rows)
    finite_x = np.asarray([float(row["x_mid_mm"]) for row in state_rows if np.isfinite(float(row["x_mid_mm"]))])
    finite_z = np.asarray([float(row["z_mid_mm"]) for row in state_rows if np.isfinite(float(row["z_mid_mm"]))])
    finite_length = np.asarray([float(row["endpoint_separation_mm"]) for row in state_rows if np.isfinite(float(row["endpoint_separation_mm"]))])
    (RESULTS / "rq1_summary.json").write_text(
        json.dumps(
            {
                "runs": run_rows,
                "aggregate": {
                    "seed_success_count": sum(bool(row["seed_collection_success"]) for row in run_rows),
                    "flat_initialization_success_count": sum(bool(row["flat_initialization_success"]) for row in run_rows),
                    "median_seed_duration_s": float(np.median([row["seed_duration_s"] for row in run_rows])),
                    "seed_duration_range_s": [float(np.min([row["seed_duration_s"] for row in run_rows])), float(np.max([row["seed_duration_s"] for row in run_rows]))],
                    "median_minimum_pairwise_rotation_deg": float(np.median([row["minimum_pairwise_rotation_deg"] for row in run_rows])),
                    "total_reactive_servo_corrections": int(sum(row["servo_corrections"] for row in run_rows)),
                    "total_actual_seed_rollbacks": int(sum(row["rollback_log_events"] for row in run_rows)),
                    "z_mid_recorded_range_mm": [float(np.min(finite_z)), float(np.max(finite_z))],
                    "z_mid_fraction_inside_80_450": float(np.mean((finite_z >= 80.0) & (finite_z <= 450.0))),
                    "x_mid_fraction_inside_abs_25": float(np.mean(np.abs(finite_x) <= 25.0)),
                    "length_fraction_inside_hard_50_120": float(np.mean((finite_length >= 50.0) & (finite_length <= 120.0))),
                    "committed_nbv_total": int(sum(row["committed_nbv_count"] for row in run_rows)),
                    "rejected_nbv_batch_total": int(sum(row["rejected_nbv_batches"] for row in run_rows)),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    make_rq1_figures(run_rows, target_rows, servo_rows, rotation_rows, state_rows)
    make_rq2_figures()
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
