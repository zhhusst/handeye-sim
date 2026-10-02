#!/usr/bin/env python3
"""Plan a simulation-only reference pose from the YAML board and hand-eye truth.

This is scene preparation, not a calibration algorithm.  The output is a list
of IK-reachable candidates for a later MoveIt collision check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(WORKSPACE / "ros2_ws/src/handeye_calibration_core"),
    str(WORKSPACE / "ros2_ws/src/fanuc_m20id25_support"),
]
from calibration_pipeline.geometry import invert_transform, make_transform, so3_exp  # noqa: E402
from calibration_pipeline.models import SensorROI, TrapezoidDomain  # noqa: E402
from calibration_pipeline.seed_collection.features import evaluate_bilateral_feature  # noqa: E402
from calibration_pipeline.seed_collection.initial_pose import (  # noqa: E402
    InitialPoseCriteria, assess_initial_pose,
)
from calibration_pipeline.simulation.scanline import compute_fov_plate_scanline  # noqa: E402
from calibration_pipeline.simulation.scene_truth import load_handeye_truth  # noqa: E402
from fanuc_m20id25_support.fanuc_kinematic import (  # noqa: E402
    JOINT_LIMITS_DEG, inverse_kinematics_numeric,
)

NOMINAL_JOINTS = np.array([-0.236623, -0.049108, -0.636688, -0.402198, -1.071110, 0.878050])


def plan_candidates(config: Path, fov_path: Path) -> dict:
    params = yaml.safe_load(config.read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    enabled = params.get("simulation_auto_reference", {}).get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("simulation_auto_reference.enabled must be a boolean")
    if not enabled:
        return {"enabled": False, "candidates": [], "reason": "disabled; manual initial positioning"}

    board, guide, seed, sensor = (params[key] for key in ("board", "endpoint_detection", "seed", "sensor"))
    preference = params.get("simulation_auto_reference", {})
    preferred_joints = np.deg2rad(np.asarray(
        preference.get("preferred_joints_deg", np.rad2deg(NOMINAL_JOINTS)), dtype=float
    ))
    posture_weights = np.asarray(
        preference.get("posture_weights", [1.0] * 6), dtype=float
    )
    if (preferred_joints.shape != (6,) or posture_weights.shape != (6,)
            or not np.isfinite(preferred_joints).all()
            or not np.isfinite(posture_weights).all()
            or np.any(posture_weights < 0.0)
            or np.all(posture_weights == 0.0)):
        raise ValueError("simulation_auto_reference posture preference needs six finite angles and nonnegative weights")
    minimum_outward_cosine = float(preference.get("minimum_outward_sensor_y_cosine", -1.0))
    if not np.isfinite(minimum_outward_cosine) or not -1.0 <= minimum_outward_cosine <= 1.0:
        raise ValueError("minimum_outward_sensor_y_cosine must be between -1 and 1")
    C = np.asarray(board["corner"], dtype=float)
    board_R = np.asarray(board["rotation"], dtype=float).reshape(3, 3)
    u, v, n = board_R.T
    length = float(guide["alignment_template_length_m"])
    if not (0.0 < length < np.hypot(board["length_u_m"], board["length_v_m"])):
        raise ValueError("guide length cannot intersect both adjacent board edges")
    theta = np.deg2rad(float(guide["alignment_template_angle_deg"]))
    center = np.array([guide["alignment_template_center_x_m"], 0.0, guide["alignment_template_center_z_m"]], dtype=float)
    half_chord = 0.5 * length * np.array([np.cos(theta), 0.0, np.sin(theta)])
    guide_left, guide_right = center - half_chord, center + half_chord
    roi = SensorROI(
        hard_domain=TrapezoidDomain(*sensor["hard_trapezoid"]),
        safe_domain=TrapezoidDomain(*sensor["safe_trapezoid"]),
    )
    fov = np.asarray(json.loads(fov_path.read_text(encoding="utf-8"))["fov_corners_S"], dtype=float)
    R_fs, t_fs = load_handeye_truth(config)
    T_sf = invert_transform(make_transform(R_fs, t_fs))
    initial = seed["initial"]
    criteria = InitialPoseCriteria(**{
        "maximum_abs_x_mid_m": initial["maximum_abs_x_mid_m"],
        "minimum_z_mid_m": initial["minimum_z_mid_m"],
        "maximum_z_mid_m": initial["maximum_z_mid_m"],
        "minimum_domain_margin_m": initial["minimum_domain_margin_m"],
        "minimum_profile_length_m": initial["minimum_profile_length_m"],
        "maximum_profile_length_m": initial["maximum_profile_length_m"],
        "minimum_absolute_endpoint_depth_delta_m": initial["minimum_absolute_endpoint_depth_delta_m"],
        "minimum_normalized_joint_margin": initial["minimum_normalized_joint_margin"],
        "minimum_local_ik_directions": initial["minimum_local_ik_directions"],
    })
    sensor_chord = (guide_right - guide_left) / length
    sensor_y = np.array([0.0, 1.0, 0.0])
    sensor_basis = np.column_stack((sensor_chord, sensor_y, np.cross(sensor_chord, sensor_y)))
    results = []
    rejected = {"wrong_plate_side": 0, "sensor_y_toward_base": 0, "edge_or_fov": 0,
                "envelope": 0, "ik": 0, "local_ik": 0}
    # Vary both edge intercepts while preserving exactly the guide chord length.
    for ratio in (0.65, 0.8, 1.0, 1.25, 1.55):
        a = length / np.sqrt(1.0 + ratio * ratio)
        b = ratio * a
        if a >= board["length_u_m"] - 0.01 or b >= board["length_v_m"] - 0.01:
            continue
        physical = {"e1": C + a * u, "e2": C + b * v}
        first_label = str(guide["initial_first_label"])
        if first_label not in physical:
            raise ValueError("initial_first_label must be e1 or e2")
        second_label = "e1" if first_label == "e2" else "e2"
        A, B = physical[first_label], physical[second_label]
        chord = (B - A) / length
        plane_cross = n - np.dot(n, chord) * chord
        plane_cross /= np.linalg.norm(plane_cross)
        board_basis = np.column_stack((chord, plane_cross, np.cross(chord, plane_cross)))
        base_rotation = board_basis @ sensor_basis.T
        for roll_deg in range(0, 360, 15):
            R_bs = Rotation.from_rotvec(np.deg2rad(roll_deg) * chord).as_matrix() @ base_rotation
            t_bs = A - R_bs @ guide_left
            # The profile geometry is two-sided, but the physical sensor must
            # stay above the board and look toward its front face.  Without
            # this test an upside-down pose can pass IK/MoveIt yet collide in
            # Gazebo and drift away from the commanded joints.
            if np.dot(t_bs - C, n) < 0.05 or np.dot(R_bs[:, 2], n) > -0.15:
                rejected["wrong_plate_side"] += 1
                continue
            sensor_radius_xy = np.linalg.norm(t_bs[:2])
            if sensor_radius_xy < 1e-6:
                rejected["sensor_y_toward_base"] += 1
                continue
            outward_cosine = float(np.dot(R_bs[:2, 1], t_bs[:2] / sensor_radius_xy))
            if outward_cosine < minimum_outward_cosine:
                rejected["sensor_y_toward_base"] += 1
                continue
            T_bf = make_transform(R_bs, t_bs) @ T_sf
            scan = compute_fov_plate_scanline(
                R_bs, t_bs, C, n, u, v,
                board["length_u_m"], board["length_v_m"],
                n_sample=2401, fov_corners_S=fov,
            )
            endpoints = scan["endpoints_S"]
            if (
                len(endpoints) != 2
                or {label for label, _ in endpoints} != {"e1", "e2"}
                or len(scan["scan_pts_S"]) < seed["minimum_profile_points"]
            ):
                rejected["edge_or_fov"] += 1
                continue
            endpoints = dict(endpoints)
            feature = evaluate_bilateral_feature(endpoints["e1"], endpoints["e2"], roi)
            # The seed batch checks this *stricter* 70-90 mm working band.
            if not (
                seed["servo"]["length_lower_m"] <= feature.profile_length <= seed["servo"]["length_upper_m"]
                and abs(feature.x_mid) <= seed["x_mid_tolerance_m"]
                and all(roi.contains(p) for p in scan["scan_pts_S"])
            ):
                rejected["envelope"] += 1
                continue
            solutions = inverse_kinematics_numeric(T_bf, q_init=preferred_joints, max_iter=70)
            if not len(solutions):
                solutions = inverse_kinematics_numeric(T_bf, q_init=NOMINAL_JOINTS, max_iter=70)
            if not len(solutions):
                rejected["ik"] += 1
                continue
            joints = solutions[0].copy()
            # J6 has a wide multi-turn range.  Equivalent +/-360-degree
            # solutions produce the same sensor pose; prefer the shorter,
            # less twisted approach from the nominal resting posture.
            joint_limits = np.deg2rad(JOINT_LIMITS_DEG)
            equivalent_j6 = [joints[5] + 2.0 * np.pi * turn for turn in (-1, 0, 1)]
            equivalent_j6 = [value for value in equivalent_j6
                             if joint_limits[5, 0] <= value <= joint_limits[5, 1]]
            joints[5] = min(equivalent_j6, key=lambda value: abs(value - preferred_joints[5]))
            local_ik = 0
            for axis in (0, 1):
                for sign in (-1, 1):
                    delta = np.zeros(3)
                    delta[axis] = sign * np.deg2rad(initial["local_ik_test_step_deg"])
                    target = T_bf.copy()
                    target[:3, :3] = T_bf[:3, :3] @ so3_exp(delta)
                    nearby = inverse_kinematics_numeric(target, q_init=joints, max_iter=40)
                    if len(nearby) and np.max(np.abs(nearby[0] - joints)) <= np.deg2rad(initial["maximum_local_joint_step_deg"]):
                        local_ik += 1
            assessment = assess_initial_pose(
                feature, joints, np.deg2rad(JOINT_LIMITS_DEG),
                local_ik_directions=local_ik, criteria=criteria,
            )
            if not assessment.accepted:
                rejected["local_ik"] += 1
                continue
            results.append({
                "joints_rad": joints.tolist(),
                "roll_deg": roll_deg,
                "edge_intercepts_m": {"e1": a, "e2": b},
                "x_mid_m": feature.x_mid,
                "z_mid_m": feature.z_mid,
                "endpoint_spacing_m": feature.profile_length,
                "domain_margin_m": feature.domain_margin,
                "local_ik_directions": local_ik,
                "outward_sensor_y_cosine": outward_cosine,
                "score": float(
                    np.linalg.norm(posture_weights * (joints - preferred_joints))
                    + 0.5 * (4 - local_ik)
                ),
            })
    results.sort(key=lambda candidate: candidate["score"])
    if not results:
        raise RuntimeError(f"simulation reference pose infeasible before collision checking: {rejected}")
    return {"enabled": True, "candidates": results[:24], "rejected": rejected,
            "note": "Collision validity must be confirmed by MoveIt before motion."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=WORKSPACE / "ros2_ws/src/handeye_sim_bridge/config/calibration.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fov_path = WORKSPACE / "data/fov_calib.json"
    if not fov_path.is_file():
        fov_path = WORKSPACE / "ros2_ws/src/handeye_sim_backend/config/fov_factory_calib.json"
    result = plan_candidates(args.config, fov_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Simulation reference pose: {len(result['candidates'])} candidates; {result.get('reason', result.get('rejected'))}")


if __name__ == "__main__":
    main()
