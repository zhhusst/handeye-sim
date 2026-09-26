#!/usr/bin/env python3
"""Visualize one real micro-rotation followed by one translation-servo step.

This is an independent paper-figure sidecar.  It does not modify the
six-final-seed visualization or any calibration/robot code.

The default A/B/C states are taken from the first formal micro-step of the
``ry_positive`` branch in the real run recorded on 2026-08-20:

* A: valid observation immediately before a pure +1 degree local-y rotation;
* B: the settled pose after that rotation and before translation servo;
* C: the settled pose after the corresponding translation correction.

Run with ``--refresh-from-bag`` after sourcing ROS 2 to reproduce the cached
state file directly from ``/calibration/seed_motion_state``.  Ordinary figure
viewing only needs NumPy and Matplotlib because the extracted states are cached.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE))

# Reuse only mesh/URDF rendering utilities.  The earlier figure itself and its
# source data are never changed by this script.
from star_seed_pose_figure.visualize_star_seed_poses import (  # noqa: E402
    PoseRecord,
    draw_triad,
    draw_urdf_visual,
    load_gocator_mesh,
    load_weld_gun_mesh,
    set_equal_limits,
    set_view,
)


DEFAULT_RUN = Path(
    "/workspace/data/calibration_runs/"
    "20260820_144753_焊接书_位置1_真机_1"
)
DEFAULT_URDF = Path("/workspace/urdf/calib_robot.urdf")
DEFAULT_STATES = SCRIPT_DIR / "microstep_states.json"
DEFAULT_OUTPUTS = {
    "rotation": SCRIPT_DIR / "process1_rotation_microstep.png",
    "servo": SCRIPT_DIR / "process2_translation_servo.png",
    "all": SCRIPT_DIR / "rotation_servo_microstep_current_view.png",
    "target": SCRIPT_DIR / "servo_target_region_only.png",
}

# These timestamps correspond to one continuous A->B->C sequence verified
# against seed_collection.log.  Nearest messages are selected from the bag.
DEFAULT_TARGET_TIMES_S = {
    "A": 1787237332.624657,
    "B": 1787237333.6748514,
    "C": 1787237334.675370,
}

STATE_COLORS = {
    "A": "#4d4d4d",
    "B": "#d55e00",
    "C": "#0072b2",
}
STATE_NAMES = {
    "A": "A: valid before rotation",
    "B": r"B: after $+1^\circ$ local-$y_F$ rotation",
    "C": "C: after translation servo",
}
PROCESS_STATE_KEYS = {
    "rotation": ("A", "B"),
    "servo": ("B", "C"),
    "all": ("A", "B", "C"),
    "target": (),
}
PROCESS_TITLES = {
    "rotation": r"Process 1: local-$y_F$ rotation micro-step (A$\rightarrow$B)",
    "servo": r"Process 2: translation-servo correction (B$\rightarrow$C)",
    "all": r"Complete micro-step (A$\rightarrow$B$\rightarrow$C)",
    "target": "Servo target region on the board",
}
DEFAULT_VIEWS = {
    "rotation": (60.0, -55.0, 0.0),
    "servo": (60.0, -55.0, 0.0),
    "all": (60.0, -55.0, 0.0),
    # Near board-normal view makes the family of acceptable chords legible.
    "target": (86.0, -90.0, 0.0),
}


@dataclass(frozen=True)
class MicrostepState:
    key: str
    stamp_s: float
    R_BF: np.ndarray
    t_BF: np.ndarray
    joints_rad: np.ndarray
    endpoints_S: np.ndarray
    endpoints_B: np.ndarray
    endpoints_B_projected: np.ndarray
    profile_points_S: np.ndarray
    profile_bag_stamp_s: float
    profile_sensor_stamp_s: float
    endpoint_separation_mm: float
    x_mid_mm: float
    motion_stage: str
    accumulated_rotation_deg: float
    pending_rotation_deg: float
    servo_command_local_mm: np.ndarray


@dataclass(frozen=True)
class MicrostepScene:
    run_dir: Path
    states: tuple[MicrostepState, ...]
    R_FS: np.ndarray
    t_FS: np.ndarray
    board_corner_B: np.ndarray
    board_rotation_B: np.ndarray
    metrics: dict
    control_target: dict


def _array(value, shape, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _nearest_messages_from_bag(run_dir: Path) -> dict[str, tuple[float, dict]]:
    """Read the three nearest seed-motion-state messages from the real MCAP."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python modules are required for --refresh-from-bag. Run: "
            "source /opt/ros/jazzy/setup.bash && "
            "source /workspace/ros2_ws/install/setup.bash"
        ) from exc

    bag_dir = run_dir / "full_run_bag"
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    topic = "/calibration/seed_motion_state"
    if topic not in topic_types:
        raise RuntimeError(f"{topic} is absent from {bag_dir}")
    message_class = get_message(topic_types[topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    nearest = {key: (float("inf"), 0.0, None) for key in DEFAULT_TARGET_TIMES_S}
    lower = min(DEFAULT_TARGET_TIMES_S.values()) - 1.0
    upper = max(DEFAULT_TARGET_TIMES_S.values()) + 1.0
    while reader.has_next():
        _, data, stamp_ns = reader.read_next()
        stamp_s = stamp_ns * 1.0e-9
        if stamp_s < lower:
            continue
        if stamp_s > upper:
            break
        payload = json.loads(deserialize_message(data, message_class).data)
        for key, target in DEFAULT_TARGET_TIMES_S.items():
            distance = abs(stamp_s - target)
            if distance < nearest[key][0]:
                nearest[key] = (distance, stamp_s, payload)

    result = {}
    for key, (distance, stamp_s, payload) in nearest.items():
        if payload is None or distance > 0.08:
            raise RuntimeError(
                f"could not recover state {key} near {DEFAULT_TARGET_TIMES_S[key]:.6f}"
            )
        result[key] = (stamp_s, payload)
    return result


def _nearest_profiles_from_bag(
    run_dir: Path, state_stamps_s: dict[str, float]
) -> dict[str, dict]:
    """Recover real filtered plate-profile points nearest each A/B/C state."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs_py import point_cloud2
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 sensor_msgs_py is required to extract target surface points"
        ) from exc

    bag_dir = run_dir / "full_run_bag"
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    topic = "/calibration/target_surface_points"
    if topic not in topic_types:
        raise RuntimeError(f"{topic} is absent from {bag_dir}")
    message_class = get_message(topic_types[topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    nearest = {key: (float("inf"), 0.0, None) for key in state_stamps_s}
    lower = min(state_stamps_s.values()) - 0.5
    upper = max(state_stamps_s.values()) + 0.5
    while reader.has_next():
        _, data, stamp_ns = reader.read_next()
        stamp_s = stamp_ns * 1.0e-9
        if stamp_s < lower:
            continue
        if stamp_s > upper:
            break
        for key, target in state_stamps_s.items():
            distance = abs(stamp_s - target)
            if distance < nearest[key][0]:
                nearest[key] = (
                    distance,
                    stamp_s,
                    deserialize_message(data, message_class),
                )

    result = {}
    for key, (distance, bag_stamp_s, message) in nearest.items():
        if message is None or distance > 0.08:
            raise RuntimeError(
                f"could not recover target surface profile for state {key}"
            )
        points = np.asarray(
            point_cloud2.read_points_numpy(
                message,
                field_names=["x", "y", "z"],
                skip_nans=True,
            ),
            dtype=float,
        )
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise RuntimeError(f"invalid target surface profile for state {key}")
        sensor_stamp_s = (
            float(message.header.stamp.sec)
            + 1.0e-9 * float(message.header.stamp.nanosec)
        )
        result[key] = {
            "bag_stamp_s": bag_stamp_s,
            "sensor_stamp_s": sensor_stamp_s,
            "frame_id": str(message.header.frame_id),
            "points_S_m": points.tolist(),
        }
    return result


def extract_states(run_dir: Path, output: Path) -> dict:
    """Extract, transform, validate, and cache one actual A->B->C micro-step."""
    try:
        from fanuc_m20id25_support.fanuc_kinematic import forward_kinematics_urdf
    except ImportError as exc:
        raise RuntimeError(
            "fanuc_m20id25_support is required for --refresh-from-bag; source the "
            "workspace install/setup.bash first"
        ) from exc

    result_path = run_dir / "calibration_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    R_FS = _array(result["handeye"]["rotation"], (3, 3), "handeye.rotation")
    t_FS = _array(result["handeye"]["translation"], (3,), "handeye.translation")
    corner = _array(result["board"]["corner"], (3,), "board.corner")
    R_BP = _array(result["board"]["rotation"], (3, 3), "board.rotation")
    normal = R_BP[:, 2]

    messages = _nearest_messages_from_bag(run_dir)
    profiles = _nearest_profiles_from_bag(
        run_dir, {key: value[0] for key, value in messages.items()}
    )
    # x_mid tolerance is not repeated in every state message. Read it from
    # the exact versioned parameter file stored with this run. The length
    # target/band below are also retained in the state topic itself.
    try:
        import yaml

        parameter_payload = yaml.safe_load(
            (run_dir / "calibration_parameters.yaml").read_text(encoding="utf-8")
        )
        seed_parameters = parameter_payload["/**"]["ros__parameters"]["seed"]
        x_mid_tolerance_mm = 1000.0 * float(seed_parameters["x_mid_tolerance_m"])
    except (ImportError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise RuntimeError(
            "cannot recover seed.x_mid_tolerance_m from the run configuration"
        ) from exc
    records = []
    transforms = {}
    for key in ("A", "B", "C"):
        stamp_s, status = messages[key]
        joints = _array(status["joints_rad"], (6,), f"{key}.joints_rad")
        T_BF = np.asarray(forward_kinematics_urdf(joints), dtype=float)
        transforms[key] = T_BF
        endpoints_S = _array(status["endpoints_S"], (2, 3), f"{key}.endpoints_S")
        endpoints_B = np.asarray(
            [T_BF[:3, :3] @ (R_FS @ p + t_FS) + T_BF[:3, 3] for p in endpoints_S]
        )
        signed_normal_offsets = (endpoints_B - corner) @ normal
        endpoints_projected = endpoints_B - signed_normal_offsets[:, None] * normal
        records.append(
            {
                "key": key,
                "name": STATE_NAMES[key],
                "bag_stamp_s": stamp_s,
                "state": status.get("state"),
                "motion_stage": status.get("motion_stage"),
                "R_BF": T_BF[:3, :3].tolist(),
                "t_BF_m": T_BF[:3, 3].tolist(),
                "joints_rad": joints.tolist(),
                "joints_deg": np.rad2deg(joints).tolist(),
                "endpoints_S_m": endpoints_S.tolist(),
                "endpoints_B_m": endpoints_B.tolist(),
                "endpoints_B_projected_to_board_m": endpoints_projected.tolist(),
                "target_surface_profile": profiles[key],
                "endpoint_board_normal_offsets_mm": (1000.0 * signed_normal_offsets).tolist(),
                "endpoint_separation_mm": float(status["endpoint_separation_mm"]),
                "x_mid_mm": float(
                    500.0 * (endpoints_S[0, 0] + endpoints_S[1, 0])
                ),
                "dual_feature_error_mm": status.get("dual_feature_error_mm"),
                "servo_command_local_mm": status.get(
                    "dual_servo_command_mm", [0.0, 0.0, 0.0]
                ),
                "accumulated_rotation_deg": float(
                    status.get("accumulated_rotation_deg", 0.0)
                ),
                "pending_rotation_deg": float(status.get("pending_rotation_deg", 0.0)),
            }
        )

    T_A, T_B, T_C = (transforms[key] for key in ("A", "B", "C"))
    R_AB = T_A[:3, :3].T @ T_B[:3, :3]
    R_BC = T_B[:3, :3].T @ T_C[:3, :3]
    t_AB_A_mm = 1000.0 * T_A[:3, :3].T @ (T_B[:3, 3] - T_A[:3, 3])
    t_BC_B_mm = 1000.0 * T_B[:3, :3].T @ (T_C[:3, 3] - T_B[:3, 3])
    metrics = {
        "A_to_B_rotation_deg": _rotation_angle_deg(R_AB),
        "A_to_B_translation_in_A_flange_mm": t_AB_A_mm.tolist(),
        "A_to_B_translation_norm_mm": float(np.linalg.norm(t_AB_A_mm)),
        "B_to_C_rotation_deg": _rotation_angle_deg(R_BC),
        "B_to_C_translation_in_B_flange_mm": t_BC_B_mm.tolist(),
        "B_to_C_translation_norm_mm": float(np.linalg.norm(t_BC_B_mm)),
        "expected_servo_command_local_mm": records[2]["servo_command_local_mm"],
        "endpoint_separation_A_mm": records[0]["endpoint_separation_mm"],
        "endpoint_separation_B_mm": records[1]["endpoint_separation_mm"],
        "endpoint_separation_C_mm": records[2]["endpoint_separation_mm"],
    }

    checks = {
        "A_to_B_is_one_degree_rotation": 0.9 < metrics["A_to_B_rotation_deg"] < 1.1,
        "A_to_B_is_nearly_pure_rotation": metrics["A_to_B_translation_norm_mm"] < 0.1,
        "B_to_C_preserves_rotation": metrics["B_to_C_rotation_deg"] < 0.01,
        "B_has_clear_spacing_drift": (
            metrics["endpoint_separation_B_mm"]
            - metrics["endpoint_separation_A_mm"]
            > 20.0
        ),
        "C_recovers_spacing": (
            metrics["endpoint_separation_C_mm"]
            < metrics["endpoint_separation_B_mm"] - 15.0
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"selected A/B/C sequence failed validation: {checks}")

    payload = {
        "source_run": str(run_dir),
        "source_bag": str(run_dir / "full_run_bag"),
        "source_topic": "/calibration/seed_motion_state",
        "branch": "ry_positive",
        "selection_reason": (
            "first formal ry_positive micro-step: A->B is a pure +1 deg local-y "
            "rotation with zero feedforward; B->C is its immediately following "
            "dual-feature translation-servo correction"
        ),
        "event_log_times_s": {
            "rotation_command": 1787237332.677197845,
            "rotation_disturbance_update": 1787237333.675512004,
            "servo_command": 1787237333.676484615,
            "servo_model_update": 1787237334.676646846,
        },
        "handeye_F_T_S": {"rotation": R_FS.tolist(), "translation_m": t_FS.tolist()},
        "board_B_T_P": {"corner_m": corner.tolist(), "rotation": R_BP.tolist()},
        "states": records,
        "metrics": metrics,
        "control_target": {
            "x_mid_tolerance_mm": x_mid_tolerance_mm,
            "endpoint_separation_lower_mm": float(
                messages["C"][1]["servo_length_band_mm"][0]
            ),
            "endpoint_separation_upper_mm": float(
                messages["C"][1]["servo_length_band_mm"][1]
            ),
            "endpoint_separation_target_mm": float(
                messages["C"][1]["servo_length_target_mm"]
            ),
            "board_mapping_reference_state": "C",
            "meaning": (
                "soft servo acceptance set; C may stop anywhere inside this band, "
                "while the 80 mm chord at x_mid=0 is the central regulation target"
            ),
        },
        "validation_checks": checks,
        "display_note": (
            "Measured endpoints are projected by only their small normal residual "
            "onto the estimated board for drawing the colored scan chords."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_scene(states_path: Path) -> MicrostepScene:
    payload = json.loads(states_path.read_text(encoding="utf-8"))
    R_FS = _array(
        payload["handeye_F_T_S"]["rotation"], (3, 3), "handeye.rotation"
    )
    t_FS = _array(
        payload["handeye_F_T_S"]["translation_m"], (3,), "handeye.translation"
    )
    states = []
    for item in payload["states"]:
        states.append(
            MicrostepState(
                key=item["key"],
                stamp_s=float(item["bag_stamp_s"]),
                R_BF=_array(item["R_BF"], (3, 3), f"{item['key']}.R_BF"),
                t_BF=_array(item["t_BF_m"], (3,), f"{item['key']}.t_BF"),
                joints_rad=_array(
                    item["joints_rad"], (6,), f"{item['key']}.joints"
                ),
                endpoints_S=_array(
                    item["endpoints_S_m"], (2, 3), f"{item['key']}.endpoints_S"
                ),
                endpoints_B=_array(
                    item["endpoints_B_m"], (2, 3), f"{item['key']}.endpoints_B"
                ),
                endpoints_B_projected=_array(
                    item["endpoints_B_projected_to_board_m"],
                    (2, 3),
                    f"{item['key']}.endpoints_B_projected",
                ),
                profile_points_S=_array(
                    item["target_surface_profile"]["points_S_m"],
                    (len(item["target_surface_profile"]["points_S_m"]), 3),
                    f"{item['key']}.profile_points_S",
                ),
                profile_bag_stamp_s=float(
                    item["target_surface_profile"]["bag_stamp_s"]
                ),
                profile_sensor_stamp_s=float(
                    item["target_surface_profile"]["sensor_stamp_s"]
                ),
                endpoint_separation_mm=float(item["endpoint_separation_mm"]),
                x_mid_mm=float(item["x_mid_mm"]),
                motion_stage=str(item["motion_stage"]),
                accumulated_rotation_deg=float(item["accumulated_rotation_deg"]),
                pending_rotation_deg=float(item["pending_rotation_deg"]),
                servo_command_local_mm=_array(
                    item["servo_command_local_mm"],
                    (3,),
                    f"{item['key']}.servo_command",
                ),
            )
        )
    return MicrostepScene(
        run_dir=Path(payload["source_run"]),
        states=tuple(states),
        R_FS=R_FS,
        t_FS=t_FS,
        board_corner_B=_array(
            payload["board_B_T_P"]["corner_m"], (3,), "board.corner"
        ),
        board_rotation_B=_array(
            payload["board_B_T_P"]["rotation"], (3, 3), "board.rotation"
        ),
        metrics=payload["metrics"],
        control_target=payload["control_target"],
    )


def _pose_record(scene: MicrostepScene, state: MicrostepState, index: int) -> PoseRecord:
    R_BS = state.R_BF @ scene.R_FS
    t_BS = state.t_BF + state.R_BF @ scene.t_FS
    return PoseRecord(
        index=index,
        label=state.key,
        display_label=STATE_NAMES[state.key],
        R_BF=state.R_BF,
        t_BF=state.t_BF,
        R_BS=R_BS,
        t_BS=t_BS,
        joints_rad=state.joints_rad,
        endpoint_u_S=state.endpoints_S[0],
        endpoint_v_S=state.endpoints_S[1],
    )


def _draw_arrow(ax, start: np.ndarray, end: np.ndarray, color: str, linewidth=2.0):
    delta = end - start
    ax.quiver(
        *start,
        *delta,
        color=color,
        linewidth=linewidth,
        arrow_length_ratio=0.17,
        normalize=False,
    )


def _target_chord_envelope(scene: MicrostepScene, extent_u: float, extent_v: float):
    """Map the code's feature-space servo target to board-local scan chords.

    A scan chord is represented by endpoints ``C+a*u`` and ``C+b*v`` on the
    two directed edges.  Its physical length is ``sqrt(a^2+b^2)``.  For this
    real B->C correction we evaluate sensor-frame x_mid using state C, because
    C is the accepted post-servo pose.  The result is an equivalent board-space
    envelope for this particular correction, not a universal fixed ROI.
    """
    reference_key = scene.control_target["board_mapping_reference_state"]
    reference = next(state for state in scene.states if state.key == reference_key)
    C = scene.board_corner_B
    u, v = scene.board_rotation_B[:, 0], scene.board_rotation_B[:, 1]
    x_tol = 0.001 * float(scene.control_target["x_mid_tolerance_mm"])
    length_lower = 0.001 * float(
        scene.control_target["endpoint_separation_lower_mm"]
    )
    length_upper = 0.001 * float(
        scene.control_target["endpoint_separation_upper_mm"]
    )
    length_target = 0.001 * float(
        scene.control_target["endpoint_separation_target_mm"]
    )

    # Dense deterministic grid: enough to display the acceptance set without
    # claiming that it is a rectangular spatial ROI.
    a_values = np.linspace(0.001, extent_u, 150)
    b_values = np.linspace(0.001, extent_v, 150)
    aa, bb = np.meshgrid(a_values, b_values, indexing="ij")
    a = aa.ravel()
    b = bb.ravel()
    endpoint_u_B = C[None, :] + a[:, None] * u[None, :]
    endpoint_v_B = C[None, :] + b[:, None] * v[None, :]

    def to_sensor(points_B):
        points_F = (reference.R_BF.T @ (points_B - reference.t_BF).T).T
        return (scene.R_FS.T @ (points_F - scene.t_FS).T).T

    endpoint_u_S = to_sensor(endpoint_u_B)
    endpoint_v_S = to_sensor(endpoint_v_B)
    x_mid = 0.5 * (endpoint_u_S[:, 0] + endpoint_v_S[:, 0])
    chord_length = np.hypot(a, b)
    valid = (
        (np.abs(x_mid) <= x_tol)
        & (chord_length >= length_lower)
        & (chord_length <= length_upper)
    )
    if not np.any(valid):
        raise RuntimeError("servo target set has no board-local chord samples")

    a_valid = a[valid]
    b_valid = b[valid]
    x_valid = x_mid[valid]
    length_valid = chord_length[valid]
    endpoint_u_valid = endpoint_u_B[valid]
    endpoint_v_valid = endpoint_v_B[valid]
    score = (x_valid / max(x_tol, 1.0e-12)) ** 2 + (
        (length_valid - length_target)
        / max(0.5 * (length_upper - length_lower), 1.0e-12)
    ) ** 2
    center_index = int(np.argmin(score))

    # Downsample only for rendering. Extrema and the central target use all
    # valid samples.
    render_indices = np.linspace(
        0, len(endpoint_u_valid) - 1, min(650, len(endpoint_u_valid)), dtype=int
    )
    segments = np.stack(
        (endpoint_u_valid[render_indices], endpoint_v_valid[render_indices]), axis=1
    )
    return {
        "segments_B": segments,
        "central_chord_B": np.vstack(
            (endpoint_u_valid[center_index], endpoint_v_valid[center_index])
        ),
        "edge_u_interval_B": np.vstack(
            (C + np.min(a_valid) * u, C + np.max(a_valid) * u)
        ),
        "edge_v_interval_B": np.vstack(
            (C + np.min(b_valid) * v, C + np.max(b_valid) * v)
        ),
        "central_x_mid_mm": 1000.0 * float(x_valid[center_index]),
        "central_length_mm": 1000.0 * float(length_valid[center_index]),
    }


def _draw_sensor_profile_inset(ax, scene: MicrostepScene, selected_states):
    """Draw real filtered x_S-z_S profiles in the upper-right paper inset."""
    # Target-only mode has no 3-D A/B/C body. Use C because the board-space
    # target envelope is also evaluated in the accepted C sensor frame.
    if not selected_states:
        selected_states = tuple(state for state in scene.states if state.key == "C")

    inset = ax.figure.add_axes([0.68, 0.655, 0.285, 0.285])
    for state in selected_states:
        points_mm = 1000.0 * state.profile_points_S
        color = STATE_COLORS[state.key]
        inset.plot(
            points_mm[:, 0],
            points_mm[:, 2],
            color=color,
            linewidth=1.15,
            alpha=0.92,
            label=state.key,
        )
        endpoints_mm = 1000.0 * state.endpoints_S
        inset.scatter(
            endpoints_mm[:, 0],
            endpoints_mm[:, 2],
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
        )

    inset.set_title(r"Measured plate profile in $\{S\}$", fontsize=8.5, pad=4)
    inset.set_xlabel(r"$x_S$ (mm)", fontsize=8, labelpad=1)
    inset.set_ylabel(r"$z_S$ (mm)", fontsize=8, labelpad=1)
    inset.tick_params(axis="both", labelsize=7, length=2.5)
    inset.grid(False)
    inset.set_aspect("equal", adjustable="datalim")
    inset.legend(loc="best", fontsize=7, framealpha=0.82, handlelength=1.4)
    for spine in inset.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#5a5a5a")
    inset.set_facecolor((1.0, 1.0, 1.0, 0.92))
    return inset


def render_scene(
    ax,
    scene: MicrostepScene,
    gocator_mesh,
    weld_gun_mesh,
    *,
    axis_length: float,
    show_labels: bool,
    show_process_arrows: bool,
    show_laser_planes: bool,
    show_target_region: bool,
    show_profile_inset: bool,
    show_grid: bool,
    show_base_axes: bool,
    process: str,
    pose_keys: tuple[str, ...] | None,
    clean: bool,
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.set_proj_type("ortho")

    selected_keys = pose_keys if pose_keys is not None else PROCESS_STATE_KEYS[process]
    selected_states = tuple(
        state for state in scene.states if state.key in selected_keys
    )
    # The board extent is a display crop. In target-only mode there are no
    # displayed A/B/C states, so retain the same extent derived from all three
    # real observations instead of inventing a different plate patch.
    extent_states = selected_states if selected_states else scene.states
    if show_profile_inset:
        _draw_sensor_profile_inset(ax, scene, selected_states)

    C = scene.board_corner_B
    R_BP = scene.board_rotation_B
    u, v, n = R_BP[:, 0], R_BP[:, 1], R_BP[:, 2]
    all_local = np.vstack(
        [
            (R_BP.T @ (state.endpoints_B_projected - C).T).T
            for state in extent_states
        ]
    )
    extent_u = max(0.13, 1.28 * float(np.max(all_local[:, 0])))
    extent_v = max(0.13, 1.28 * float(np.max(all_local[:, 1])))
    board_corners = np.array(
        [C, C + extent_u * u, C + extent_u * u + extent_v * v, C + extent_v * v]
    )
    board = Poly3DCollection(
        [board_corners],
        facecolors="#e8d5a7",
        edgecolors="#6f5425",
        linewidths=1.2,
        alpha=0.34,
    )
    ax.add_collection3d(board)
    # Highlight the two physical edges sharing corner C.
    ax.plot(*np.vstack((C, C + extent_u * u)).T, color="#5b401b", linewidth=3.0)
    ax.plot(*np.vstack((C, C + extent_v * v)).T, color="#5b401b", linewidth=3.0)
    ax.scatter(*C, marker="D", s=34, color="#5b401b", depthshade=False, zorder=10)
    if show_labels:
        ax.text(*(C + 0.008 * n), "C", fontsize=10, color="#4b3518")

    target_envelope = None
    if show_target_region:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        target_envelope = _target_chord_envelope(scene, extent_u, extent_v)
        target_lines = Line3DCollection(
            target_envelope["segments_B"],
            colors=(0.10, 0.62, 0.26, 0.035),
            linewidths=0.75,
            zorder=3,
        )
        ax.add_collection3d(target_lines)
        central = target_envelope["central_chord_B"] + 0.0012 * n
        ax.plot(
            central[:, 0],
            central[:, 1],
            central[:, 2],
            color="#11823b",
            linewidth=2.2,
            linestyle="--",
            zorder=8,
        )
        for interval in (
            target_envelope["edge_u_interval_B"],
            target_envelope["edge_v_interval_B"],
        ):
            interval = interval + 0.0010 * n
            ax.plot(
                interval[:, 0],
                interval[:, 1],
                interval[:, 2],
                color="#2ca25f",
                linewidth=5.0,
                alpha=0.58,
                solid_capstyle="round",
                zorder=7,
            )

    scene_points = [*board_corners]
    scan_midpoints = {}
    flange_origins = {}
    # Transparency is useful only when multiple sensor poses overlap. A
    # single selected pose should look like the opaque physical URDF model.
    gocator_alpha_scale = 0.98 if len(selected_states) == 1 else 0.42
    for index, state in enumerate(selected_states):
        color = STATE_COLORS[state.key]
        pose = _pose_record(scene, state, index)
        flange_origins[state.key] = pose.t_BF

        # Physical sensor and optional welding gun use the same URDF transforms
        # as RViz; the calibrated measurement frame is used only for the laser.
        R_BG_visual = pose.R_BF @ gocator_mesh.R_FS_rviz
        t_BG_visual = pose.t_BF + pose.R_BF @ gocator_mesh.t_FS_rviz
        sensor_vertices = t_BG_visual + (R_BG_visual @ gocator_mesh.vertices_S.T).T
        scene_points.extend(sensor_vertices)
        draw_urdf_visual(
            ax,
            pose,
            gocator_mesh,
            alpha_scale=gocator_alpha_scale,
            color_override=color,
        )
        if weld_gun_mesh is not None:
            R_BW = pose.R_BF @ weld_gun_mesh.R_FS_rviz
            t_BW = pose.t_BF + pose.R_BF @ weld_gun_mesh.t_FS_rviz
            scene_points.extend(t_BW + (R_BW @ weld_gun_mesh.vertices_S.T).T)
            draw_urdf_visual(ax, pose, weld_gun_mesh, alpha_scale=0.13)

        draw_triad(ax, pose.t_BF, pose.R_BF, axis_length, linewidth=2.1)
        ax.scatter(
            *pose.t_BF,
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            depthshade=False,
            zorder=12,
        )

        endpoints = state.endpoints_B_projected
        scan_midpoints[state.key] = np.mean(endpoints, axis=0)
        ax.plot(
            endpoints[:, 0],
            endpoints[:, 1],
            endpoints[:, 2],
            color=color,
            linewidth=4.2,
            solid_capstyle="round",
            zorder=15,
        )
        ax.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            endpoints[:, 2],
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            depthshade=False,
            zorder=16,
        )

        # Actual calibrated sensor origin and the measured board chord define
        # the displayed laser plane for this state.
        sensor_origin = pose.t_BS
        scene_points.append(sensor_origin)
        if show_laser_planes:
            laser = Poly3DCollection(
                [[sensor_origin, endpoints[0], endpoints[1]]],
                facecolors=color,
                edgecolors=color,
                linewidths=0.7,
                alpha=0.15,
            )
            ax.add_collection3d(laser)

        if show_labels:
            # Put A/B/C at different fractions along their chords.  The three
            # real chords are close, so a shared midpoint label would overlap.
            fraction = {"A": 0.16, "B": 0.50, "C": 0.84}[state.key]
            anchor = (1.0 - fraction) * endpoints[0] + fraction * endpoints[1]
            offset = (0.007 + 0.004 * index) * n
            ax.text(
                *(anchor + offset),
                state.key,
                fontsize=12,
                fontweight="bold",
                color=color,
                zorder=20,
            )

    if show_process_arrows:
        # The arrows connect the real measured scan-chord centers.  They expose
        # the large observation drift even though the physical rotation is 1 deg.
        if "A" in selected_keys and "B" in selected_keys:
            _draw_arrow(
                ax,
                scan_midpoints["A"] + 0.004 * n,
                scan_midpoints["B"] + 0.004 * n,
                STATE_COLORS["B"],
                linewidth=2.2,
            )
        if "B" in selected_keys and "C" in selected_keys:
            _draw_arrow(
                ax,
                scan_midpoints["B"] + 0.009 * n,
                scan_midpoints["C"] + 0.009 * n,
                STATE_COLORS["C"],
                linewidth=2.2,
            )
            # B->C is a real translation with essentially unchanged rotation.
            _draw_arrow(
                ax,
                flange_origins["B"],
                flange_origins["C"],
                STATE_COLORS["C"],
                linewidth=2.8,
            )

    if show_base_axes or show_grid:
        ax.set_xlabel(r"$x_B$ (m)")
        ax.set_ylabel(r"$y_B$ (m)")
        ax.set_zlabel(r"$z_B$ (m)")
    else:
        ax.set_axis_off()
    ax.grid(show_grid)
    if clean:
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
    else:
        title = PROCESS_TITLES[process]
        if pose_keys is not None:
            title += " | selected pose" + ("s " if len(pose_keys) > 1 else " ")
            title += ", ".join(pose_keys)
        ax.set_title(title, fontsize=11, pad=15)

    state_handles = [
        Line2D(
            [0],
            [0],
            color=STATE_COLORS[key],
            marker="o",
            linewidth=3.0,
            markersize=5,
            label=STATE_NAMES[key],
        )
        for key in selected_keys
    ]
    state_handles.append(
        Patch(facecolor="#e8d5a7", edgecolor="#6f5425", alpha=0.5, label="board")
    )
    if selected_states:
        state_handles.extend(
            [
                Line2D([0], [0], color="#d62728", lw=2, label=r"flange $x_F$"),
                Line2D([0], [0], color="#2ca02c", lw=2, label=r"flange $y_F$"),
                Line2D([0], [0], color="#1f77b4", lw=2, label=r"flange $z_F$"),
            ]
        )
    if show_target_region:
        state_handles.insert(
            len(selected_keys),
            Line2D(
                [0],
                [0],
                color="#2ca25f",
                lw=5,
                alpha=0.58,
                label=(
                    r"servo target: $|x_{mid}|\leq10$ mm, "
                    r"$70\leq L\leq90$ mm"
                ),
            ),
        )
    ax.legend(handles=state_handles, loc="upper left", fontsize=8, framealpha=0.90)

    scene_points.extend(state.t_BF for state in selected_states)
    for state in selected_states:
        scene_points.extend(state.endpoints_B_projected)
    set_equal_limits(ax, np.asarray(scene_points), margin_fraction=0.07)


def save_figure(scene, gocator, weld_gun, args, elev, azim, roll):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10.6, 8.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    render_scene(
        axis,
        scene,
        gocator,
        weld_gun,
        axis_length=args.axis_length,
        show_labels=not args.hide_labels,
        show_process_arrows=not args.hide_process_arrows,
        show_laser_planes=not args.hide_laser_planes,
        show_target_region=not args.hide_target_region,
        show_profile_inset=not args.hide_profile_inset,
        show_grid=args.show_grid,
        show_base_axes=args.show_base_axes,
        process=args.process,
        pose_keys=args.pose_keys,
        clean=True,
    )
    set_view(axis, elev, azim, roll)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        transparent=args.transparent,
    )
    plt.close(figure)
    print(
        f"Saved: {args.output}\n"
        f"View: elev={elev:.1f}, azim={azim:.1f}, roll={roll:.1f}"
    )


def interactive(scene, gocator, weld_gun, args):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    figure = plt.figure(figsize=(12.0, 8.6))
    figure.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.18)
    axis = figure.add_subplot(111, projection="3d")
    render_scene(
        axis,
        scene,
        gocator,
        weld_gun,
        axis_length=args.axis_length,
        show_labels=not args.hide_labels,
        show_process_arrows=not args.hide_process_arrows,
        show_laser_planes=not args.hide_laser_planes,
        show_target_region=not args.hide_target_region,
        show_profile_inset=not args.hide_profile_inset,
        show_grid=args.show_grid,
        show_base_axes=args.show_base_axes,
        process=args.process,
        pose_keys=args.pose_keys,
        clean=False,
    )
    set_view(axis, args.elev, args.azim, args.roll)

    elev_axis = figure.add_axes([0.12, 0.105, 0.58, 0.025])
    azim_axis = figure.add_axes([0.12, 0.070, 0.58, 0.025])
    roll_axis = figure.add_axes([0.12, 0.035, 0.58, 0.025])
    elev_slider = Slider(elev_axis, "Elevation", -90, 90, valinit=args.elev)
    azim_slider = Slider(azim_axis, "Azimuth", -180, 180, valinit=args.azim)
    roll_slider = Slider(roll_axis, "Roll", -180, 180, valinit=args.roll)
    save_axis = figure.add_axes([0.75, 0.075, 0.10, 0.045])
    reset_axis = figure.add_axes([0.87, 0.075, 0.10, 0.045])
    save_button = Button(save_axis, "Save PNG")
    reset_button = Button(reset_axis, "Reset view")

    def update(_event=None):
        set_view(axis, elev_slider.val, azim_slider.val, roll_slider.val)
        figure.canvas.draw_idle()

    def current_view():
        return (
            float(getattr(axis, "elev", elev_slider.val)),
            float(getattr(axis, "azim", azim_slider.val)),
            float(getattr(axis, "roll", roll_slider.val)),
        )

    def save(_event=None):
        save_figure(scene, gocator, weld_gun, args, *current_view())

    def reset(_event=None):
        elev_slider.reset()
        azim_slider.reset()
        roll_slider.reset()
        update()

    def key(event):
        if event.key == "s":
            save()
        elif event.key == "p":
            e, a, r = current_view()
            print(f"Current view: --elev {e:.1f} --azim {a:.1f} --roll {r:.1f}")
        elif event.key == "r":
            reset()

    elev_slider.on_changed(update)
    azim_slider.on_changed(update)
    roll_slider.on_changed(update)
    save_button.on_clicked(save)
    reset_button.on_clicked(reset)
    figure.canvas.mpl_connect("key_press_event", key)
    print(
        "Interactive controls:\n"
        "  mouse drag: rotate; wheel/right drag: zoom (backend dependent)\n"
        "  sliders: exact elevation / azimuth / roll\n"
        "  s: save clean PNG; p: print view; r: reset"
    )
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Visualize one real Ry(+) micro-rotation and servo recovery."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument(
        "--process",
        choices=("rotation", "servo", "all", "target"),
        default="all",
        help=(
            "rotation: only A->B; servo: only B->C; all: complete A->B->C; "
            "target: only the board and servo target region"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG; defaults to a process-specific filename",
    )
    parser.add_argument(
        "--poses",
        nargs="+",
        choices=("A", "B", "C"),
        default=None,
        help=(
            "display only selected poses inside the chosen process, for example "
            "--process servo --poses B or --process all --poses A C"
        ),
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--refresh-from-bag",
        action="store_true",
        help="re-extract A/B/C from the real MCAP before plotting",
    )
    parser.add_argument("--save-only", action="store_true")
    parser.add_argument("--elev", type=float, default=None)
    parser.add_argument("--azim", type=float, default=None)
    parser.add_argument("--roll", type=float, default=None)
    parser.add_argument("--axis-length", type=float, default=0.052)
    parser.add_argument("--mesh-voxel-mm", type=float, default=2.0)
    parser.add_argument("--weld-gun-voxel-mm", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show-weld-gun",
        action="store_true",
        help="also draw the translucent URDF weld-gun mesh; hidden by default",
    )
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-process-arrows", action="store_true")
    parser.add_argument("--hide-laser-planes", action="store_true")
    parser.add_argument(
        "--hide-target-region",
        action="store_true",
        help="hide the green board-space envelope of acceptable servo chords",
    )
    parser.add_argument(
        "--hide-profile-inset",
        action="store_true",
        help="hide the upper-right inset of real sensor-frame plate profiles",
    )
    parser.add_argument("--show-grid", action="store_true")
    parser.add_argument("--show-base-axes", action="store_true")
    parser.add_argument("--transparent", action="store_true")
    args = parser.parse_args()
    if args.poses is None:
        args.pose_keys = None
        return args

    # Canonical ordering also removes accidental repetitions such as A A B.
    requested = set(args.poses)
    args.pose_keys = tuple(key for key in ("A", "B", "C") if key in requested)
    allowed = set(PROCESS_STATE_KEYS[args.process])
    invalid = requested - allowed
    if invalid:
        parser.error(
            f"pose(s) {sorted(invalid)} do not belong to process {args.process}; "
            f"allowed poses are {list(PROCESS_STATE_KEYS[args.process])}"
        )
    return args


def main():
    args = parse_arguments()
    args.run_dir = args.run_dir.resolve()
    args.states = args.states.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    elif args.pose_keys is not None:
        pose_suffix = "_".join(args.pose_keys)
        args.output = (SCRIPT_DIR / f"selected_pose_{pose_suffix}.png").resolve()
    else:
        args.output = DEFAULT_OUTPUTS[args.process].resolve()
    args.urdf = args.urdf.resolve()
    default_elev, default_azim, default_roll = DEFAULT_VIEWS[args.process]
    args.elev = default_elev if args.elev is None else args.elev
    args.azim = default_azim if args.azim is None else args.azim
    args.roll = default_roll if args.roll is None else args.roll

    if args.refresh_from_bag or not args.states.is_file():
        payload = extract_states(args.run_dir, args.states)
        print(f"Extracted real A/B/C states: {args.states}")
        print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))

    if args.save_only or not os.environ.get("DISPLAY"):
        import matplotlib

        matplotlib.use("Agg")

    scene = load_scene(args.states)
    gocator = load_gocator_mesh(args.urdf, voxel_size_mm=args.mesh_voxel_mm)
    weld_gun = None
    if args.show_weld_gun:
        weld_gun = load_weld_gun_mesh(
            args.urdf, voxel_size_mm=args.weld_gun_voxel_mm
        )

    print(
        f"Display mode: {args.process} | {PROCESS_TITLES[args.process]}\n"
        f"Displayed poses: "
        f"{list(args.pose_keys if args.pose_keys is not None else PROCESS_STATE_KEYS[args.process])}\n"
        "Selected real sequence:\n"
        f"  A->B rotation: {scene.metrics['A_to_B_rotation_deg']:.6f} deg\n"
        f"  A->B translation: {scene.metrics['A_to_B_translation_norm_mm']:.4f} mm\n"
        f"  B->C rotation: {scene.metrics['B_to_C_rotation_deg']:.6f} deg\n"
        f"  B->C local translation: "
        f"{np.round(scene.metrics['B_to_C_translation_in_B_flange_mm'], 3).tolist()} mm\n"
        f"  endpoint spacing A/B/C: "
        f"{scene.metrics['endpoint_separation_A_mm']:.2f} / "
        f"{scene.metrics['endpoint_separation_B_mm']:.2f} / "
        f"{scene.metrics['endpoint_separation_C_mm']:.2f} mm"
    )

    if args.save_only or not os.environ.get("DISPLAY"):
        if not args.save_only and not os.environ.get("DISPLAY"):
            print("DISPLAY unavailable; generating a static preview.")
        save_figure(scene, gocator, weld_gun, args, args.elev, args.azim, args.roll)
    else:
        interactive(scene, gocator, weld_gun, args)


if __name__ == "__main__":
    main()
