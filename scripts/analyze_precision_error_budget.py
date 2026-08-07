#!/usr/bin/env python3
"""Controlled error-budget experiment for the bilateral hand-eye solver.

The experiment deliberately reuses the production profile renderer, endpoint
detector, stationary-frame aggregation and 12-DOF-V2 solver.  Every condition
uses the same physical pose set.  Simulation truth is used only for reporting
errors and for assigning the two physical edge labels in this offline audit.
It is never passed to the solver.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = WORKSPACE / "ros2_ws/src/handeye_sim_bridge"
sys.path.insert(0, str(PACKAGE_ROOT))

from calibration_pipeline.dataset_io import (  # noqa: E402
    SeedObservationGroup,
    aggregate_seed_group,
)
from calibration_pipeline.geometry import (  # noqa: E402
    make_transform,
    rotation_distance_deg,
    so3_exp,
)
from calibration_pipeline.models import FlangePose, Measurement  # noqa: E402
from calibration_pipeline.perception import (  # noqa: E402
    EndpointDetectionConfig,
    ProfileEndpointDetector,
)
from calibration_pipeline.seed_collection.multiframe import (  # noqa: E402
    robust_endpoint_inliers,
)
from calibration_pipeline.simulation import (  # noqa: E402
    SimulationNoiseConfig,
    SimulationNoiseModel,
    compute_fov_plate_scanline,
)
from calibration_pipeline.simulation.synthetic import (  # noqa: E402
    default_scene,
    generate_seed_dataset,
)
from calibration_pipeline.solvers import TwelveDofV2Solver  # noqa: E402
from calibration_pipeline.v2_backend.information import (  # noqa: E402
    effective_handeye_information,
    information_gain,
)
from calibration_pipeline.v2_backend.residual import (  # noqa: E402
    numerical_jacobian,
    variable_projection_residual,
)


CONFIG_PATH = PACKAGE_ROOT / "config/calibration.yaml"
DEFAULT_JSON = WORKSPACE / "data/precision_error_budget.json"
DEFAULT_REPORT = WORKSPACE / "docs/精度误差预算与归因报告.md"


def _parameters() -> dict:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return payload["/**"]["ros__parameters"]


def _detector_config(parameters: dict) -> EndpointDetectionConfig:
    values = parameters["endpoint_detection"]
    fields = EndpointDetectionConfig.__dataclass_fields__
    return EndpointDetectionConfig(
        **{name: values[name] for name in fields if name in values}
    )


def _noise_config(parameters: dict) -> SimulationNoiseConfig:
    values = parameters["simulation_noise"]
    fields = SimulationNoiseConfig.__dataclass_fields__
    return SimulationNoiseConfig(
        **{name: values[name] for name in fields if name in values}
    )


def _solver(parameters: dict) -> TwelveDofV2Solver:
    values = parameters["solver"]
    return TwelveDofV2Solver(
        plane_weight=float(values["plane_weight"]),
        edge_weight=float(values["edge_weight"]),
        endpoint_plane_weight=float(values["endpoint_plane_weight"]),
        max_evaluations=int(values["max_evaluations"]),
        tolerance=float(values["tolerance"]),
        state_scale=np.array(
            [
                np.deg2rad(float(values["handeye_rotation_scale_deg"])),
            ]
            * 3
            + [float(values["handeye_translation_scale_m"])] * 3
            + [np.deg2rad(float(values["plane_rotation_scale_deg"]))] * 3
        ),
        maximum_condition_number=float(values["maximum_condition_number"]),
    )


def _nominal_handeye(parameters: dict, scene) -> tuple[np.ndarray, np.ndarray]:
    rotation_axis = np.array([1.0, -2.0, 1.5], dtype=float)
    rotation_axis /= np.linalg.norm(rotation_axis)
    translation_axis = np.array([1.0, -0.6, 0.8], dtype=float)
    translation_axis /= np.linalg.norm(translation_axis)
    rotation_error = np.deg2rad(
        float(parameters["handeye_init_rotation_error_deg"])
    )
    translation_error = (
        float(parameters["handeye_init_translation_error_mm"]) / 1000.0
    )
    return (
        scene.handeye_rotation @ so3_exp(rotation_axis * rotation_error),
        scene.handeye_translation + translation_axis * translation_error,
    )


def _zero_noise(base: SimulationNoiseConfig, *, random_seed: int) -> SimulationNoiseConfig:
    return replace(
        base,
        random_seed=random_seed,
        profile_gaussian_std_m=0.0,
        endpoint_gaussian_std_m=0.0,
        robot_translation_std_m=0.0,
        robot_rotation_std_deg=0.0,
        board_flatness_rms_m=0.0,
        sync_delay_mean_s=0.0,
        sync_jitter_std_s=0.0,
        point_outlier_probability=0.0,
        point_outlier_std_m=0.0,
        endpoint_outlier_probability=0.0,
        endpoint_outlier_std_m=0.0,
        point_dropout_probability=0.0,
        frame_dropout_probability=0.0,
        endpoint_dropout_probability=0.0,
    )


def _condition_configs(
    base: SimulationNoiseConfig, *, random_seed: int
) -> dict[str, SimulationNoiseConfig | None]:
    zero = _zero_noise(base, random_seed=random_seed)
    return {
        # None bypasses rendering/detection and proves the analytic model.
        "analytic_ideal": None,
        # This includes only renderer sampling and endpoint extraction.
        "raw_discretization": zero,
        "profile_only": replace(
            zero,
            profile_gaussian_std_m=base.profile_gaussian_std_m,
            point_outlier_probability=base.point_outlier_probability,
            point_outlier_std_m=base.point_outlier_std_m,
            point_dropout_probability=base.point_dropout_probability,
            frame_dropout_probability=base.frame_dropout_probability,
        ),
        "robot_pose_only": replace(
            zero,
            robot_translation_std_m=base.robot_translation_std_m,
            robot_rotation_std_deg=base.robot_rotation_std_deg,
        ),
        "flatness_only": replace(
            zero,
            board_flatness_rms_m=base.board_flatness_rms_m,
        ),
        "flatness_stress_0p5mm": replace(
            zero,
            board_flatness_rms_m=0.0005,
        ),
        # A metrology-grade-artifact comparison; all other current disturbances
        # remain unchanged.  0.03 mm is an explicit experiment setting, not a
        # hidden change to the runtime calibration configuration.
        "combined_precision_artifact": replace(
            base,
            random_seed=random_seed,
            board_flatness_rms_m=0.00003,
            endpoint_gaussian_std_m=0.0,
            endpoint_outlier_probability=0.0,
            endpoint_outlier_std_m=0.0,
            endpoint_dropout_probability=0.0,
            # Stationary capture makes time offset locally unobservable; the
            # dynamic Gazebo check is reported separately.
            sync_delay_mean_s=0.0,
            sync_jitter_std_s=0.0,
        ),
        "combined_ultraprecision_artifact": replace(
            base,
            random_seed=random_seed,
            board_flatness_rms_m=0.00001,
            endpoint_gaussian_std_m=0.0,
            endpoint_outlier_probability=0.0,
            endpoint_outlier_std_m=0.0,
            endpoint_dropout_probability=0.0,
            sync_delay_mean_s=0.0,
            sync_jitter_std_s=0.0,
        ),
        "combined_no_flatness": replace(
            base,
            random_seed=random_seed,
            board_flatness_rms_m=0.0,
            endpoint_gaussian_std_m=0.0,
            endpoint_outlier_probability=0.0,
            endpoint_outlier_std_m=0.0,
            endpoint_dropout_probability=0.0,
            sync_delay_mean_s=0.0,
            sync_jitter_std_s=0.0,
        ),
        "combined_current_stationary": replace(
            base,
            random_seed=random_seed,
            endpoint_gaussian_std_m=0.0,
            endpoint_outlier_probability=0.0,
            endpoint_outlier_std_m=0.0,
            endpoint_dropout_probability=0.0,
            sync_delay_mean_s=0.0,
            sync_jitter_std_s=0.0,
        ),
        "combined_stress_0p5mm": replace(
            base,
            random_seed=random_seed,
            board_flatness_rms_m=0.0005,
            endpoint_gaussian_std_m=0.0,
            endpoint_outlier_probability=0.0,
            endpoint_outlier_std_m=0.0,
            endpoint_dropout_probability=0.0,
            sync_delay_mean_s=0.0,
            sync_jitter_std_s=0.0,
        ),
    }


def _ordered_endpoints(
    detection,
    rendered_endpoints: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    truth = {label: np.asarray(point, dtype=float) for label, point in rendered_endpoints}
    if "e1" not in truth or "e2" not in truth:
        return None
    direct = np.linalg.norm(detection.first - truth["e1"]) + np.linalg.norm(
        detection.second - truth["e2"]
    )
    swapped = np.linalg.norm(detection.first - truth["e2"]) + np.linalg.norm(
        detection.second - truth["e1"]
    )
    if direct <= swapped:
        return detection.first, detection.second
    return detection.second, detection.first


def _render_frame(
    nominal_pose: FlangePose,
    scene,
    model: SimulationNoiseModel,
    detector: ProfileEndpointDetector,
) -> tuple[FlangePose, Measurement, float] | None:
    physical_flange = model.perturb_flange(nominal_pose.transform)
    sensor = physical_flange @ scene.handeye_transform
    rotation = sensor[:3, :3]
    translation = sensor[:3, 3]
    rendered = compute_fov_plate_scanline(
        rotation_sensor_base=rotation,
        translation_sensor_base=translation,
        corner=scene.board.corner,
        normal=scene.board.normal,
        u=scene.board.u,
        v=scene.board.v,
        width=scene.board.length_u,
        height=scene.board.length_v,
        half_fov_deg=15.0,
        min_range=scene.roi.hard_domain.z_near,
        max_range=scene.roi.hard_domain.z_far,
    )
    if not rendered["has_intersection"] or len(rendered["endpoints_B"]) != 2:
        return None

    laser_normal = rotation[:, 1]
    labels = [label for label, _ in rendered["endpoints_B"]]
    endpoint_points_ideal = np.asarray(
        [point for _, point in rendered["endpoints_B"]]
    )
    points_base, endpoint_points_base = model.deform_bounded_scanline(
        rendered["scan_pts_B"],
        endpoint_points_ideal,
        boundary_labels=labels,
        laser_normal=laser_normal,
        board_normal=scene.board.normal,
        corner=scene.board.corner,
        board_u=scene.board.u,
        board_v=scene.board.v,
        width=scene.board.length_u,
        height=scene.board.length_v,
    )
    points_sensor = (rotation.T @ (points_base - translation).T).T

    endpoint_points_sensor = (
        rotation.T @ (endpoint_points_base - translation).T
    ).T
    rendered_endpoints = list(zip(labels, endpoint_points_sensor))

    dropped = model.sample_frame_dropout()
    profile = model.corrupt_profile(points_sensor, frame_dropped=dropped)
    if len(profile) < detector.config.minimum_points:
        return None
    detection = detector.detect(profile)
    if detection is None:
        return None
    endpoints = _ordered_endpoints(detection, rendered_endpoints)
    if endpoints is None:
        return None
    endpoint_u, endpoint_v = endpoints
    truth = {label: point for label, point in rendered_endpoints}
    endpoint_error = 500.0 * (
        np.linalg.norm(endpoint_u - truth["e1"])
        + np.linalg.norm(endpoint_v - truth["e2"])
    )
    # The reported pose is the nominal encoder pose.  The physical perturbation
    # remains hidden, exactly as in the Gazebo publisher.
    return (
        nominal_pose,
        Measurement(profile, endpoint_u, endpoint_v),
        float(endpoint_error),
    )


def _aggregate_physical_pose(
    label: str,
    frames: list[tuple[FlangePose, Measurement, float]],
    *,
    mad_multiplier: float,
) -> tuple[FlangePose, Measurement, list[float]] | None:
    endpoints_u = np.asarray([item[1].endpoint_u for item in frames])
    endpoints_v = np.asarray([item[1].endpoint_v for item in frames])
    inliers, _ = robust_endpoint_inliers(
        endpoints_u,
        endpoints_v,
        mad_multiplier=mad_multiplier,
    )
    accepted = [item for item, keep in zip(frames, inliers) if keep]
    if len(accepted) < 4:
        return None
    group = SeedObservationGroup(
        label,
        tuple(item[0] for item in accepted),
        tuple(item[1] for item in accepted),
    )
    pose, measurement = aggregate_seed_group(group)
    return pose, measurement, [item[2] for item in accepted]


def _solve_trial(
    condition: str,
    noise_config: SimulationNoiseConfig | None,
    *,
    trial: int,
    physical_poses: list[FlangePose],
    ideal_measurements: list[Measurement],
    frames_per_pose: int,
    parameters: dict,
    detector_config: EndpointDetectionConfig,
) -> dict:
    scene = default_scene()
    nominal_rotation, nominal_translation = _nominal_handeye(parameters, scene)
    poses: list[FlangePose] = []
    measurements: list[Measurement] = []
    endpoint_errors: list[float] = []
    accepted_frames = 0
    requested_frames = len(physical_poses) * frames_per_pose

    if noise_config is None:
        poses = list(physical_poses)
        measurements = list(ideal_measurements)
        accepted_frames = requested_frames
    else:
        config = replace(
            noise_config,
            random_seed=int(noise_config.random_seed + 104729 * trial),
        )
        model = SimulationNoiseModel(config)
        detector = ProfileEndpointDetector(detector_config)
        mad_multiplier = float(parameters["seed"]["endpoint_mad_multiplier"])
        for pose_index, nominal_pose in enumerate(physical_poses):
            frames = []
            for _ in range(frames_per_pose):
                rendered = _render_frame(nominal_pose, scene, model, detector)
                if rendered is not None:
                    frames.append(rendered)
            aggregated = _aggregate_physical_pose(
                f"pose_{pose_index:02d}",
                frames,
                mad_multiplier=mad_multiplier,
            )
            if aggregated is None:
                continue
            pose, measurement, errors = aggregated
            poses.append(pose)
            measurements.append(measurement)
            endpoint_errors.extend(errors)
            accepted_frames += len(errors)

    started = time.perf_counter()
    try:
        result = _solver(parameters).solve(
            poses,
            measurements,
            nominal_rotation,
            nominal_translation,
            board_dimensions=(scene.board.length_u, scene.board.length_v),
        )
    except Exception as error:
        return {
            "condition": condition,
            "trial": trial,
            "converged": False,
            "failure": str(error),
            "pose_count": len(poses),
            "accepted_frames": accepted_frames,
            "requested_frames": requested_frames,
            "elapsed_s": time.perf_counter() - started,
        }
    translation_delta = (
        result.estimate.handeye_translation - scene.handeye_translation
    )
    covariance = result.estimate.covariance_x9
    return {
        "condition": condition,
        "trial": trial,
        "converged": bool(result.converged),
        "failure": "" if result.converged else result.message,
        "pose_count": len(poses),
        "accepted_frames": accepted_frames,
        "requested_frames": requested_frames,
        "acceptance_rate": accepted_frames / max(requested_frames, 1),
        "endpoint_error_mm": {
            "median": None if not endpoint_errors else float(np.median(endpoint_errors)),
            "p95": None if not endpoint_errors else float(np.percentile(endpoint_errors, 95.0)),
        },
        "rotation_error_deg": rotation_distance_deg(
            result.estimate.handeye_rotation, scene.handeye_rotation
        ),
        "translation_error_mm": 1000.0 * float(np.linalg.norm(translation_delta)),
        "translation_error_vector_mm": (1000.0 * translation_delta).tolist(),
        "maximum_rotation_std_deg": float(
            np.rad2deg(np.sqrt(np.max(np.diag(covariance)[:3])))
        ),
        "maximum_translation_std_mm": float(
            1000.0 * np.sqrt(np.max(np.diag(covariance)[3:6]))
        ),
        "cost": float(result.cost),
        "rank": int(result.diagnostics.rank),
        "condition_number": float(result.diagnostics.condition_number),
        "elapsed_s": time.perf_counter() - started,
    }


def _information_selected_indices(
    poses: list[FlangePose],
    measurements: list[Measurement],
    *,
    selected_count: int,
    seed_count: int,
    parameters: dict,
) -> list[int]:
    """Select an idealized fixed-size set with the production information model.

    This is an offline pose-set diagnostic.  The first ``seed_count`` poses are
    the exploration set.  Thereafter each ideal candidate is evaluated at the
    current estimate using the same reduced Jacobian and hand-eye Schur
    complement as the runtime NBV implementation.
    """
    if selected_count > len(poses):
        raise ValueError("selected_count exceeds candidate pool")
    if not 4 <= seed_count <= selected_count:
        raise ValueError("seed_count must be between four and selected_count")
    scene = default_scene()
    nominal_rotation, nominal_translation = _nominal_handeye(parameters, scene)
    selected = list(range(seed_count))
    remaining = list(range(seed_count, len(poses)))
    solver = _solver(parameters)
    while len(selected) < selected_count:
        current_poses = [poses[index] for index in selected]
        current_measurements = [measurements[index] for index in selected]
        result = solver.solve(
            current_poses,
            current_measurements,
            nominal_rotation,
            nominal_translation,
            board_dimensions=(scene.board.length_u, scene.board.length_v),
        )
        x9 = result.estimate.x9
        weights = solver.weights
        current_residual = lambda state: variable_projection_residual(
            state, current_poses, current_measurements, **weights
        )
        current_information = effective_handeye_information(
            numerical_jacobian(current_residual, x9),
            state_scale=solver.state_scale,
        )
        best_index = None
        best_score = (float("-inf"), float("-inf"))
        for index in remaining:
            augmented_poses = current_poses + [poses[index]]
            augmented_measurements = current_measurements + [measurements[index]]
            augmented_residual = lambda state: variable_projection_residual(
                state, augmented_poses, augmented_measurements, **weights
            )
            augmented_information = effective_handeye_information(
                numerical_jacobian(augmented_residual, x9),
                state_scale=solver.state_scale,
            )
            score = (
                information_gain(current_information, augmented_information),
                float(np.min(np.linalg.eigvalsh(augmented_information))),
            )
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            raise RuntimeError("information selection could not find a candidate")
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def _statistics(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
    }


def _summarize(rows: list[dict]) -> dict:
    converged = [row for row in rows if row.get("converged")]
    translation_vectors = np.asarray(
        [row["translation_error_vector_mm"] for row in converged], dtype=float
    )
    return {
        "trials": len(rows),
        "convergence_rate": len(converged) / max(len(rows), 1),
        "rotation_error_deg": _statistics(
            [row["rotation_error_deg"] for row in converged]
        ),
        "translation_error_mm": _statistics(
            [row["translation_error_mm"] for row in converged]
        ),
        "translation_bias_vector_mm": (
            [None, None, None]
            if not len(translation_vectors)
            else np.mean(translation_vectors, axis=0).tolist()
        ),
        "endpoint_error_mm": {
            "median": _statistics(
                [
                    row["endpoint_error_mm"]["median"]
                    for row in converged
                    if row["endpoint_error_mm"]["median"] is not None
                ]
            ),
            "p95": _statistics(
                [
                    row["endpoint_error_mm"]["p95"]
                    for row in converged
                    if row["endpoint_error_mm"]["p95"] is not None
                ]
            ),
        },
        "maximum_translation_std_mm": _statistics(
            [row["maximum_translation_std_mm"] for row in converged]
        ),
        "acceptance_rate": _statistics(
            [row.get("acceptance_rate", 1.0) for row in converged]
        ),
        "elapsed_s": _statistics([row["elapsed_s"] for row in rows]),
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _markdown(payload: dict) -> str:
    lines = [
        "# 0.1 mm / 0.1° 精度误差预算与归因报告",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 实验约束",
        "",
        f"- 固定物理位姿数：{payload['experiment']['pose_count']}",
        f"- 候选池位姿数：{payload['experiment']['candidate_pool']}",
        f"- 位姿选择：{payload['experiment']['pose_selection']}",
        f"- 每个位姿帧数：{payload['experiment']['frames_per_pose']}",
        f"- 每个条件重复次数：{payload['experiment']['trials']}",
        "- 所有条件使用同一物理位姿集合；仿真真值只用于离线评价和物理边标签。",
        "- `analytic_ideal` 绕过轮廓渲染和断点检测；其余条件均通过生产检测链路。",
        "- 同步项未在离线静止采集模型中注入；动态同步必须通过 Gazebo 时间戳对照另测。",
        "",
        "## 汇总",
        "",
        "| 条件 | 收敛率 | 旋转中位数/° | 旋转P95/° | 平移中位数/mm | 平移P95/mm | 断点P95中位数/mm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in payload["summary"].items():
        rotation = summary["rotation_error_deg"]
        translation = summary["translation_error_mm"]
        endpoint = summary["endpoint_error_mm"]["p95"]
        lines.append(
            f"| `{name}` | {summary['convergence_rate']:.1%} | "
            f"{_fmt(rotation['median'])} | {_fmt(rotation['p95'])} | "
            f"{_fmt(translation['median'])} | {_fmt(translation['p95'])} | "
            f"{_fmt(endpoint['median'])} |"
        )
    lines += [
        "",
        "## 条件定义",
        "",
        "- `raw_discretization`：无随机噪声，仅保留轮廓采样和断点算法。",
        "- `profile_only`：轮廓高斯噪声、点离群与点漏检。",
        "- `robot_pose_only`：隐藏物理法兰扰动，求解器仍接收名义编码器位姿。",
        "- `flatness_only`：固定空间平板形变场。",
        "- `flatness_stress_0p5mm`：仅注入 0.5 mm RMS 固定空间平板形变。",
        "- `combined_precision_artifact`：当前组合噪声，但平板 RMS 为 0.03 mm。",
        "- `combined_ultraprecision_artifact`：当前组合噪声，但平板 RMS 为 0.01 mm。",
        "- `combined_no_flatness`：当前组合噪声，但关闭固定平板形变。",
        "- `combined_current_stationary`：当前组合噪声，采集阶段视为机器人已静止。",
        "- `combined_stress_0p5mm`：当前随机噪声叠加 0.5 mm RMS 平板压力测试。",
        "",
        "## 解释边界",
        "",
        "- 本报告用于分离求解器、断点、机器人位姿和平板形变的因果影响。",
        "- 离线位姿集合不替代完整 Gazebo/MoveIt 可达性测试。",
        "- 只有冻结配置、保存完整元数据后的结果才可作为论文消融数据。",
        "",
        f"原始结果：`{payload['output_json']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--poses", type=int, default=12)
    parser.add_argument("--candidate-pool", type=int)
    parser.add_argument(
        "--pose-selection",
        choices=("fixed", "information"),
        default="fixed",
    )
    parser.add_argument("--exploration-poses", type=int, default=6)
    parser.add_argument("--frames-per-pose", type=int, default=18)
    parser.add_argument("--pose-seed", type=int, default=17)
    parser.add_argument("--random-seed", type=int, default=20260807)
    parser.add_argument("--plane-weight", type=float)
    parser.add_argument("--edge-weight", type=float)
    parser.add_argument("--endpoint-plane-weight", type=float)
    parser.add_argument(
        "--condition",
        action="append",
        dest="conditions",
        help="run only the named condition; may be supplied repeatedly",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.trials < 1 or args.poses < 4 or args.frames_per_pose < 4:
        parser.error("trials >= 1, poses >= 4 and frames-per-pose >= 4 are required")

    parameters = _parameters()
    weight_overrides = {
        "plane_weight": args.plane_weight,
        "edge_weight": args.edge_weight,
        "endpoint_plane_weight": args.endpoint_plane_weight,
    }
    for name, value in weight_overrides.items():
        if value is not None:
            if value < 0.0:
                parser.error(f"{name} must be non-negative")
            parameters["solver"][name] = float(value)
    scene = default_scene()
    candidate_pool = args.poses if args.candidate_pool is None else args.candidate_pool
    if candidate_pool < args.poses:
        parser.error("candidate-pool must be at least poses")
    all_poses, all_measurements = generate_seed_dataset(
        scene, count=candidate_pool, seed=args.pose_seed
    )
    if args.pose_selection == "information":
        selected_indices = _information_selected_indices(
            all_poses,
            all_measurements,
            selected_count=args.poses,
            seed_count=args.exploration_poses,
            parameters=parameters,
        )
    else:
        selected_indices = list(range(args.poses))
    physical_poses = [all_poses[index] for index in selected_indices]
    ideal_measurements = [all_measurements[index] for index in selected_indices]
    detector_config = _detector_config(parameters)
    base_noise = _noise_config(parameters)
    condition_templates = _condition_configs(
        base_noise, random_seed=args.random_seed
    )
    if args.conditions:
        unknown = sorted(set(args.conditions) - set(condition_templates))
        if unknown:
            parser.error("unknown conditions: " + ", ".join(unknown))
        selected = set(args.conditions)
        condition_templates = {
            name: config
            for name, config in condition_templates.items()
            if name in selected
        }

    rows: dict[str, list[dict]] = {name: [] for name in condition_templates}
    condition_count = len(condition_templates)
    for condition_index, (name, config) in enumerate(
        condition_templates.items(), start=1
    ):
        print(
            f"[{condition_index}/{condition_count}] running {name}...",
            flush=True,
        )
        trial_count = 1 if name == "analytic_ideal" else args.trials
        for trial in range(trial_count):
            rows[name].append(
                _solve_trial(
                    name,
                    config,
                    trial=trial,
                    physical_poses=physical_poses,
                    ideal_measurements=ideal_measurements,
                    frames_per_pose=args.frames_per_pose,
                    parameters=parameters,
                    detector_config=detector_config,
                )
            )
        summary = _summarize(rows[name])
        print(
            f"{name:30s} "
            f"R50={_fmt(summary['rotation_error_deg']['median'])} deg, "
            f"t50={_fmt(summary['translation_error_mm']['median'])} mm, "
            f"t95={_fmt(summary['translation_error_mm']['p95'])} mm",
            flush=True,
        )

    try:
        git_revision = subprocess.run(
            ["git", "-C", str(WORKSPACE), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision,
        "configuration_file": str(CONFIG_PATH),
        "configuration_snapshot": parameters,
        "experiment": {
            "trials": args.trials,
            "pose_count": args.poses,
            "frames_per_pose": args.frames_per_pose,
            "pose_seed": args.pose_seed,
            "candidate_pool": candidate_pool,
            "pose_selection": args.pose_selection,
            "exploration_poses": args.exploration_poses,
            "selected_pose_indices": selected_indices,
            "random_seed": args.random_seed,
            "stationary_sync_assumption": True,
        },
        "condition_noise": {
            name: None if config is None else asdict(config)
            for name, config in condition_templates.items()
        },
        "summary": {name: _summarize(values) for name, values in rows.items()},
        "trials": rows,
        "output_json": str(args.output_json),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_markdown(payload), encoding="utf-8")
    print(f"JSON: {args.output_json}")
    print(f"报告: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
