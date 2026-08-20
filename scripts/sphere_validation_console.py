#!/usr/bin/env python3
"""Keyboard-driven, observe-only precision-sphere validation on real hardware."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import yaml

from calibration_pipeline.geometry import quaternion_to_matrix, rotation_distance_deg
from calibration_pipeline.sphere_validation import (
    SphereArtifact,
    SphereSegmentParameters,
    SphereValidationThresholds,
    select_sphere_profile_segment,
    transform_profile_to_base,
    validate_sphere_views,
)


WORKSPACE = Path("/workspace")
REAL_CONFIG = (
    WORKSPACE
    / "ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml"
)
CALIBRATION_RUNS = WORKSPACE / "data/calibration_runs"
VALIDATION_RUNS = WORKSPACE / "data/sphere_validation_runs"


@dataclass(frozen=True)
class SynchronizedFrame:
    stamp_ns: int
    points_sensor_m: np.ndarray
    sample_indices: np.ndarray
    flange_rotation: np.ndarray
    flange_translation_m: np.ndarray


@dataclass(frozen=True)
class AcceptedFrame:
    synchronized: SynchronizedFrame
    selected_points_sensor_m: np.ndarray
    selected_sample_indices: np.ndarray
    circle_radius_m: float
    circle_rms_m: float
    chord_m: float


def _stamp_ns(message) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def _cloud_xyz_index(message: PointCloud2) -> tuple[np.ndarray, np.ndarray]:
    field_names = {field.name for field in message.fields}
    requested = ("x", "y", "z", "index") if "index" in field_names else (
        "x",
        "y",
        "z",
    )
    values = point_cloud2.read_points(
        message, field_names=requested, skip_nans=False
    )
    if getattr(values.dtype, "names", None):
        points = np.column_stack(
            tuple(np.asarray(values[name], dtype=float) for name in ("x", "y", "z"))
        )
        indices = (
            np.asarray(values["index"], dtype=np.int64)
            if "index" in requested
            else np.arange(len(points), dtype=np.int64)
        )
    else:
        array = np.asarray(values)
        points = np.asarray(array[:, :3], dtype=float)
        indices = (
            np.asarray(array[:, 3], dtype=np.int64)
            if len(requested) == 4
            else np.arange(len(points), dtype=np.int64)
        )
    finite = np.all(np.isfinite(points), axis=1)
    return points[finite], indices[finite]


class ExactStampFrameCollector(Node):
    """Pair the normalized profile and stationary flange pose by exact stamp."""

    def __init__(self, profile_topic: str, flange_pose_topic: str) -> None:
        super().__init__("sphere_validation_collector")
        self._condition = threading.Condition()
        self._profiles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._ready: deque[SynchronizedFrame] = deque(maxlen=200)
        self._seen: set[int] = set()
        self.profile_messages = 0
        self.pose_messages = 0
        self.paired_messages = 0
        self.last_error = "waiting_for_profile_and_flange_pose"
        self.create_subscription(PointCloud2, profile_topic, self._profile, 50)
        self.create_subscription(PoseStamped, flange_pose_topic, self._pose, 50)

    def _profile(self, message: PointCloud2) -> None:
        try:
            points, indices = _cloud_xyz_index(message)
            if len(points) < 6:
                raise ValueError("profile contains fewer than six finite points")
        except (ValueError, TypeError, IndexError) as error:
            self.last_error = str(error)
            return
        stamp = _stamp_ns(message)
        with self._condition:
            self.profile_messages += 1
            self._profiles[stamp] = (points, indices)
            self._pair(stamp)
            self._trim()

    def _pose(self, message: PoseStamped) -> None:
        orientation = message.pose.orientation
        position = message.pose.position
        try:
            rotation = quaternion_to_matrix(
                np.array(
                    [
                        orientation.x,
                        orientation.y,
                        orientation.z,
                        orientation.w,
                    ]
                )
            )
        except ValueError as error:
            self.last_error = str(error)
            return
        translation = np.array([position.x, position.y, position.z], dtype=float)
        stamp = _stamp_ns(message)
        with self._condition:
            self.pose_messages += 1
            self._poses[stamp] = (rotation, translation)
            self._pair(stamp)
            self._trim()

    def _pair(self, stamp: int) -> None:
        if stamp in self._seen or stamp not in self._profiles or stamp not in self._poses:
            return
        points, indices = self._profiles.pop(stamp)
        rotation, translation = self._poses.pop(stamp)
        self._seen.add(stamp)
        self._ready.append(
            SynchronizedFrame(
                stamp,
                points,
                indices,
                rotation,
                translation,
            )
        )
        self.paired_messages += 1
        self.last_error = ""
        self._condition.notify_all()

    def _trim(self) -> None:
        for pending in (self._profiles, self._poses):
            while len(pending) > 200:
                pending.pop(next(iter(pending)))
        if len(self._seen) > 1000:
            self._seen = set(sorted(self._seen)[-500:])

    def begin_fresh_capture(self) -> int:
        with self._condition:
            self._ready.clear()
            return max(self._seen, default=-1)

    def next_frame(self, after_stamp_ns: int, timeout_s: float) -> SynchronizedFrame | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                while self._ready:
                    frame = self._ready.popleft()
                    if frame.stamp_ns > after_stamp_ns:
                        return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=min(remaining, 0.25))


def _load_parameters(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return payload["/**"]["ros__parameters"]["sphere_validation"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing sphere_validation configuration in {path}") from error


def _artifacts(parameters: dict) -> dict[str, SphereArtifact]:
    result = {}
    for artifact_id, values in parameters["artifacts"].items():
        result[artifact_id] = SphereArtifact(
            artifact_id=artifact_id,
            diameter_m=0.001 * float(values["engraved_diameter_mm"]),
            roundness_m=0.001 * float(values["engraved_roundness_mm"]),
            model=str(values.get("model", "")) or None,
        )
    return result


def _segment_parameters(parameters: dict) -> SphereSegmentParameters:
    values = parameters["segmentation"]
    return SphereSegmentParameters(
        minimum_points=int(values["minimum_points"]),
        minimum_chord_m=float(values["minimum_chord_m"]),
        maximum_arc_length_m=float(values["maximum_arc_length_m"]),
        absolute_neighbor_gap_m=float(values["absolute_neighbor_gap_m"]),
        neighbor_gap_multiplier=float(values["neighbor_gap_multiplier"]),
        minimum_slice_radius_fraction=float(values["minimum_slice_radius_fraction"]),
        maximum_slice_radius_overrun_m=float(values["maximum_slice_radius_overrun_m"]),
        maximum_circle_rms_m=float(values["maximum_circle_rms_m"]),
        robust_scale_m=float(values["circle_robust_scale_m"]),
    )


def _thresholds(parameters: dict) -> SphereValidationThresholds:
    values = parameters["thresholds"]
    return SphereValidationThresholds(
        minimum_poses=int(values["minimum_poses"]),
        maximum_fixed_radius_rmse_m=float(values["maximum_fixed_radius_rmse_m"]),
        maximum_fixed_radius_p95_m=float(values["maximum_fixed_radius_p95_m"]),
        maximum_free_diameter_error_m=float(values["maximum_free_diameter_error_m"]),
    )


def _result_candidates() -> list[Path]:
    candidates = sorted(
        CALIBRATION_RUNS.glob("*/calibration_result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    real = []
    other = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("simulation", {}).get("mode") == "real":
                real.append(path)
            else:
                other.append(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return real + other


def _choose_result(argument: str | None) -> Path:
    candidates = _result_candidates()
    default = Path(argument).expanduser() if argument else (candidates[0] if candidates else None)
    while True:
        prompt = "最终标定结果文件"
        if default is not None:
            prompt += f" [{default}]"
        value = input(prompt + "：").strip()
        path = Path(value).expanduser() if value else default
        if path is None:
            print("尚未找到标定结果，请输入 calibration_result.json 的路径。")
            continue
        if not path.is_absolute():
            path = WORKSPACE / path
        if path.exists():
            return path.resolve()
        print(f"文件不存在：{path}")


def _load_handeye(path: Path) -> tuple[np.ndarray, np.ndarray, dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    try:
        rotation = np.asarray(payload["handeye"]["rotation"], dtype=float)
        translation = np.asarray(payload["handeye"]["translation"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid hand-eye result: {error}") from error
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("hand-eye result has an invalid transform shape")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("hand-eye result rotation is not orthonormal")
    return rotation, translation, payload, hashlib.sha256(raw).hexdigest()


def _choose_artifact(artifacts: dict[str, SphereArtifact], requested: str | None) -> SphereArtifact:
    if requested:
        if requested not in artifacts:
            raise ValueError(f"unknown sphere {requested}; choose from {sorted(artifacts)}")
        return artifacts[requested]
    ordered = list(artifacts.values())
    print("\n请选择本次安装的标定球：")
    for index, artifact in enumerate(ordered, 1):
        print(
            f"  {index}. {artifact.artifact_id}："
            f"Ø{1000.0 * artifact.diameter_m:.4f} mm，"
            f"圆度 {1000.0 * (artifact.roundness_m or 0.0):.4f} mm"
        )
    while True:
        value = input("选择 [1]：").strip() or "1"
        try:
            return ordered[int(value) - 1]
        except (ValueError, IndexError):
            print(f"请输入 1～{len(ordered)}。")


def _yes_no(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        value = input(f"{prompt} {suffix}：").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "是", "确认"}:
            return True
        if value in {"n", "no", "否", "取消"}:
            return False
        print("请输入 y 或 n。")


def _choose_collection_mode(requested: str | None) -> str:
    if requested is not None:
        return requested
    print("\n请选择精密球采集方式：")
    print("  1. moving：7个方向，每个方向人工平移扫描7个位置（评价方法2，推荐）")
    print("  2. stationary：每个独立位姿只采一个静止截面（评价方法1）")
    while True:
        value = input("选择 [1]：").strip() or "1"
        if value in {"1", "moving"}:
            return "moving"
        if value in {"2", "stationary"}:
            return "stationary"
        print("请输入 1 或 2。")


def _collect_pose(
    collector: ExactStampFrameCollector,
    artifact: SphereArtifact,
    segmentation: SphereSegmentParameters,
    frame_count: int,
    timeout_s: float,
) -> list[AcceptedFrame]:
    accepted: list[AcceptedFrame] = []
    rejected = 0
    after_stamp = collector.begin_fresh_capture()
    deadline = time.monotonic() + timeout_s
    last_reason = ""
    while len(accepted) < frame_count:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        frame = collector.next_frame(after_stamp, min(remaining, 1.0))
        if frame is None:
            print(
                f"\r  等待同步球面轮廓……有效 {len(accepted)}/{frame_count}",
                end="",
                flush=True,
            )
            continue
        after_stamp = frame.stamp_ns
        try:
            selected = select_sphere_profile_segment(
                frame.points_sensor_m,
                artifact,
                sample_indices=frame.sample_indices,
                parameters=segmentation,
            )
        except ValueError as error:
            rejected += 1
            last_reason = str(error)
            continue
        accepted.append(
            AcceptedFrame(
                frame,
                selected.points_sensor_m,
                selected.sample_indices,
                selected.circle.radius_m,
                selected.circle.rms_m,
                selected.chord_m,
            )
        )
        print(
            f"\r  采集球面帧 {len(accepted):2d}/{frame_count}；"
            f"当前弦长 {1000.0 * selected.chord_m:6.2f} mm；"
            f"截面半径 {1000.0 * selected.circle.radius_m:6.3f} mm；"
            f"圆拟合RMS {1000.0 * selected.circle.rms_m:6.3f} mm",
            end="",
            flush=True,
        )
    print()
    if len(accepted) < frame_count:
        detail = last_reason or collector.last_error or "没有收到完全同步的数据"
        raise RuntimeError(
            f"只得到 {len(accepted)}/{frame_count} 个有效球面帧；"
            f"拒绝 {rejected} 帧；最后原因：{detail}"
        )
    if rejected:
        print(f"  同时丢弃了 {rejected} 个不满足球面截面几何的原始帧。")
    return accepted


def _representative_flange(frames: list[AcceptedFrame]) -> tuple[np.ndarray, np.ndarray]:
    reference = frames[0].synchronized.flange_rotation
    # Stationary frames differ far below the pose-diversity warning threshold;
    # use the first valid rotation and the mean translation only for that warning.
    translation = np.mean(
        [frame.synchronized.flange_translation_m for frame in frames], axis=0
    )
    return reference, translation


def _position_summary(frames: list[AcceptedFrame]) -> dict[str, float]:
    """Summarize one stationary stop of a manually stepped sphere scan."""
    rotations = [frame.synchronized.flange_rotation for frame in frames]
    translations = np.asarray(
        [frame.synchronized.flange_translation_m for frame in frames], dtype=float
    )
    reference_rotation = rotations[0]
    return {
        "median_chord_mm": float(
            1000.0 * np.median([frame.chord_m for frame in frames])
        ),
        "minimum_chord_mm": float(
            1000.0 * np.min([frame.chord_m for frame in frames])
        ),
        "median_circle_radius_mm": float(
            1000.0 * np.median([frame.circle_radius_m for frame in frames])
        ),
        "median_circle_rms_mm": float(
            1000.0 * np.median([frame.circle_rms_m for frame in frames])
        ),
        "maximum_circle_rms_mm": float(
            1000.0 * np.max([frame.circle_rms_m for frame in frames])
        ),
        "stationary_translation_span_mm": float(
            1000.0
            * np.max(np.linalg.norm(translations - np.mean(translations, axis=0), axis=1))
        ),
        "stationary_rotation_span_deg": float(
            max(rotation_distance_deg(reference_rotation, rotation) for rotation in rotations)
        ),
    }


def _moving_direction_summary(
    position_frames: list[list[AcceptedFrame]],
) -> dict[str, object]:
    """Measure straightness, monotonic progress, orientation hold and sphere coverage."""
    representatives = [_representative_flange(frames) for frames in position_frames]
    rotations = [item[0] for item in representatives]
    translations = np.asarray([item[1] for item in representatives], dtype=float)
    position_summaries = [_position_summary(frames) for frames in position_frames]

    centered = translations - translations[0]
    if len(translations) >= 2 and np.linalg.norm(centered[-1]) > 1e-12:
        _, _, vh = np.linalg.svd(
            translations - np.mean(translations, axis=0), full_matrices=False
        )
        direction = np.asarray(vh[0], dtype=float)
        if float(direction @ centered[-1]) < 0.0:
            direction *= -1.0
        progress_m = centered @ direction
        lateral_m = np.linalg.norm(centered - np.outer(progress_m, direction), axis=1)
    else:
        direction = np.zeros(3, dtype=float)
        progress_m = np.zeros(len(translations), dtype=float)
        lateral_m = np.zeros(len(translations), dtype=float)

    chord_mm = np.asarray(
        [summary["median_chord_mm"] for summary in position_summaries], dtype=float
    )
    peak_index = int(np.argmax(chord_mm)) if len(chord_mm) else 0
    return {
        "positions": len(position_frames),
        "position_summaries": position_summaries,
        "scan_direction_base": direction.tolist(),
        "progress_mm": (1000.0 * progress_m).tolist(),
        "step_mm": (1000.0 * np.diff(progress_m)).tolist(),
        "total_travel_mm": float(1000.0 * (progress_m[-1] - progress_m[0])),
        "maximum_line_deviation_mm": float(1000.0 * np.max(lateral_m)),
        "maximum_orientation_drift_deg": float(
            max(rotation_distance_deg(rotations[0], rotation) for rotation in rotations)
        ),
        "median_chords_mm": chord_mm.tolist(),
        "peak_chord_position": peak_index + 1,
        "start_chord_drop_mm": float(chord_mm[peak_index] - chord_mm[0]),
        "end_chord_drop_mm": float(chord_mm[peak_index] - chord_mm[-1]),
    }


def _moving_position_issues(summary: dict[str, float], parameters: dict) -> list[str]:
    issues = []
    minimum_chord_mm = float(parameters.get("moving_minimum_median_chord_mm", 8.0))
    maximum_rms_mm = float(parameters.get("moving_maximum_median_circle_rms_mm", 0.15))
    if summary["median_chord_mm"] < minimum_chord_mm:
        issues.append(
            f"中位弦长 {summary['median_chord_mm']:.2f} mm < {minimum_chord_mm:.2f} mm"
        )
    if summary["median_circle_rms_mm"] > maximum_rms_mm:
        issues.append(
            f"中位圆拟合RMS {summary['median_circle_rms_mm']:.3f} mm > "
            f"{maximum_rms_mm:.3f} mm"
        )
    return issues


def _moving_direction_issues(summary: dict[str, object], parameters: dict) -> list[str]:
    issues = []
    minimum_step_mm = float(parameters.get("moving_minimum_position_step_mm", 1.0))
    maximum_step_mm = float(parameters.get("moving_maximum_position_step_mm", 7.0))
    minimum_travel_mm = float(parameters.get("moving_minimum_total_travel_mm", 12.0))
    maximum_travel_mm = float(parameters.get("moving_maximum_total_travel_mm", 30.0))
    maximum_line_mm = float(parameters.get("moving_maximum_line_deviation_mm", 2.0))
    maximum_drift_deg = float(
        parameters.get("moving_maximum_orientation_drift_deg", 0.25)
    )
    minimum_drop_mm = float(parameters.get("moving_minimum_endpoint_chord_drop_mm", 2.0))
    steps = np.asarray(summary["step_mm"], dtype=float)
    if np.any(steps < minimum_step_mm):
        issues.append(f"存在平移步长 < {minimum_step_mm:.1f} mm 或扫描发生回头")
    if np.any(steps > maximum_step_mm):
        issues.append(f"存在平移步长 > {maximum_step_mm:.1f} mm，截面覆盖可能不连续")
    travel = float(summary["total_travel_mm"])
    if travel < minimum_travel_mm or travel > maximum_travel_mm:
        issues.append(
            f"总平移行程 {travel:.1f} mm 不在 {minimum_travel_mm:.1f}–"
            f"{maximum_travel_mm:.1f} mm"
        )
    if float(summary["maximum_line_deviation_mm"]) > maximum_line_mm:
        issues.append(
            f"平移轨迹最大偏线 {summary['maximum_line_deviation_mm']:.2f} mm > "
            f"{maximum_line_mm:.2f} mm"
        )
    if float(summary["maximum_orientation_drift_deg"]) > maximum_drift_deg:
        issues.append(
            f"方向内姿态漂移 {summary['maximum_orientation_drift_deg']:.3f}° > "
            f"{maximum_drift_deg:.3f}°"
        )
    if (
        float(summary["start_chord_drop_mm"]) < minimum_drop_mm
        or float(summary["end_chord_drop_mm"]) < minimum_drop_mm
    ):
        issues.append(
            "弦长没有形成清晰的“短→长→短”覆盖；扫描可能未跨过近球心截面"
        )
    return issues


def _moving_increment_issues(
    position_frames: list[list[AcceptedFrame]], parameters: dict
) -> list[str]:
    """Reject an obviously wrong manual step before the operator completes a group."""
    if len(position_frames) < 2:
        return []
    summary = _moving_direction_summary(position_frames)
    minimum_step_mm = float(parameters.get("moving_minimum_position_step_mm", 1.0))
    maximum_step_mm = float(parameters.get("moving_maximum_position_step_mm", 7.0))
    maximum_line_mm = float(parameters.get("moving_maximum_line_deviation_mm", 2.0))
    maximum_drift_deg = float(
        parameters.get("moving_maximum_orientation_drift_deg", 0.25)
    )
    issues = []
    last_step_mm = float(summary["step_mm"][-1])
    if last_step_mm < minimum_step_mm:
        issues.append(
            f"本步沿扫描方向仅前进 {last_step_mm:.2f} mm，可能没有移动或发生回头"
        )
    if last_step_mm > maximum_step_mm:
        issues.append(
            f"本步沿扫描方向前进 {last_step_mm:.2f} mm，超过 {maximum_step_mm:.1f} mm"
        )
    if (
        len(position_frames) >= 3
        and float(summary["maximum_line_deviation_mm"]) > maximum_line_mm
    ):
        issues.append(
            f"累计轨迹偏离直线 {summary['maximum_line_deviation_mm']:.2f} mm，"
            f"超过 {maximum_line_mm:.1f} mm"
        )
    if float(summary["maximum_orientation_drift_deg"]) > maximum_drift_deg:
        issues.append(
            f"姿态相对本方向起点变化 {summary['maximum_orientation_drift_deg']:.3f}°，"
            f"超过 {maximum_drift_deg:.3f}°；方向内只能平移"
        )
    return issues


def _collect_moving_scan(
    collector: ExactStampFrameCollector,
    artifact: SphereArtifact,
    segmentation: SphereSegmentParameters,
    parameters: dict,
    direction_count: int,
    position_count: int,
    frames_per_position: int,
    timeout_s: float,
) -> tuple[
    list[list[AcceptedFrame]],
    list[list[int]],
    list[tuple[np.ndarray, np.ndarray]],
    list[dict[str, object]],
    bool,
]:
    pose_frames: list[list[AcceptedFrame]] = []
    position_indices_by_pose: list[list[int]] = []
    representative_poses: list[tuple[np.ndarray, np.ndarray]] = []
    direction_summaries: list[dict[str, object]] = []
    minimum_direction_separation_deg = float(
        parameters.get(
            "moving_minimum_direction_rotation_separation_deg",
            parameters["minimum_pose_rotation_separation_deg"],
        )
    )

    direction_index = 1
    while direction_index <= direction_count:
        print(
            f"\n{'=' * 68}\n"
            f"[扫描方向 {direction_index}/{direction_count}] 请先设置新的机器人姿态。\n"
            "随后保持该姿态不变，只做近似直线平移，让激光从球的一侧逐步扫到另一侧。\n"
            "第1个位置应为较短但完整的球面弦，随后弦长增大，越过近球心截面后再缩短。"
        )
        positions: list[list[AcceptedFrame]] = []
        cancelled = False

        while len(positions) < position_count:
            position_index = len(positions) + 1
            if position_index == 1:
                hint = "设置本方向的姿态和球面一侧起始截面"
            else:
                hint = "保持姿态不变，仅沿同一方向平移约2–4 mm"
            print(
                f"\n[方向 {direction_index}/{direction_count}，"
                f"位置 {position_index}/{position_count}] {hint}；机器人停稳后回到这里。"
            )
            command = input(
                "Enter=采集，u=撤销上一位置，d=重采本方向，r=重看提示，q=取消："
            ).strip().lower()
            if command == "q":
                cancelled = True
                break
            if command == "r":
                continue
            if command == "d":
                print("  已丢弃本方向已采位置，请回到球面一侧重新开始。")
                positions.clear()
                continue
            if command == "u":
                if positions:
                    positions.pop()
                    print("  已撤销上一位置；请将机器人移回对应截面后重新采集。")
                else:
                    print("  本方向还没有可撤销的位置。")
                continue
            if command:
                print("  未识别的输入；请输入 Enter、u、d、r 或 q。")
                continue

            try:
                frames = _collect_pose(
                    collector,
                    artifact,
                    segmentation,
                    frames_per_position,
                    timeout_s,
                )
            except RuntimeError as error:
                print(f"  本位置采集失败：{error}")
                print("  请调整截面/曝光，保持姿态和扫描方向不变后重试。")
                continue

            position_summary = _position_summary(frames)
            print(
                "  位置质量："
                f"中位弦长 {position_summary['median_chord_mm']:.2f} mm；"
                f"截面半径 {position_summary['median_circle_radius_mm']:.3f} mm；"
                f"圆拟合RMS {position_summary['median_circle_rms_mm']:.3f} mm；"
                f"定点漂移 {position_summary['stationary_translation_span_mm']:.3f} mm/"
                f"{position_summary['stationary_rotation_span_deg']:.4f}°"
            )
            issues = _moving_position_issues(position_summary, parameters)
            trial_positions = positions + [frames]
            issues.extend(_moving_increment_issues(trial_positions, parameters))
            if issues:
                print("  本位置存在以下质量问题：")
                for issue in issues:
                    print(f"    · {issue}")
                if not _yes_no("仍保留这个位置？", False):
                    print("  已丢弃本位置，请调整后重新采集。")
                    continue
            positions.append(frames)

        if cancelled:
            return (
                pose_frames,
                position_indices_by_pose,
                representative_poses,
                direction_summaries,
                True,
            )
        summary = _moving_direction_summary(positions)
        direction_issues = _moving_direction_issues(summary, parameters)
        representative = _representative_flange(
            [frame for position in positions for frame in position]
        )
        if representative_poses:
            separation_deg = min(
                rotation_distance_deg(previous[0], representative[0])
                for previous in representative_poses
            )
            summary["minimum_direction_rotation_separation_deg"] = separation_deg
            if separation_deg < minimum_direction_separation_deg:
                direction_issues.append(
                    f"与已有扫描方向最近仅相差 {separation_deg:.2f}° < "
                    f"{minimum_direction_separation_deg:.2f}°"
                )
        else:
            summary["minimum_direction_rotation_separation_deg"] = None

        chord_text = " → ".join(
            f"{value:.1f}" for value in summary["median_chords_mm"]
        )
        print(
            f"\n  方向 {direction_index} 扫描摘要：\n"
            f"    弦长序列：{chord_text} mm\n"
            f"    总平移：{summary['total_travel_mm']:.2f} mm；"
            f"最大偏线：{summary['maximum_line_deviation_mm']:.2f} mm；"
            f"姿态漂移：{summary['maximum_orientation_drift_deg']:.4f}°"
        )
        if direction_issues:
            print("  本方向存在以下覆盖/运动问题：")
            for issue in direction_issues:
                print(f"    · {issue}")
            keep = _yes_no("仍保留整个方向？", False)
        else:
            keep = _yes_no("本方向检查通过，确认保留？", True)
        if not keep:
            print("  已丢弃整个方向；请回到球面一侧，并重新设置该方向。")
            continue

        flattened = [frame for position in positions for frame in position]
        indices = [
            position_index
            for position_index, frames in enumerate(positions)
            for _ in frames
        ]
        pose_frames.append(flattened)
        position_indices_by_pose.append(indices)
        representative_poses.append(representative)
        direction_summaries.append(summary)
        print(
            f"  ✓ 已保留方向 {direction_index}/{direction_count}："
            f"{position_count} 个截面，{len(flattened)} 帧。"
        )
        direction_index += 1

    return (
        pose_frames,
        position_indices_by_pose,
        representative_poses,
        direction_summaries,
        False,
    )


def _save_dataset(
    destination: Path,
    pose_frames: list[list[AcceptedFrame]],
    position_indices_by_pose: list[list[int]] | None = None,
) -> None:
    frames = [frame for group in pose_frames for frame in group]
    raw_offsets = [0]
    selected_offsets = [0]
    raw_points = []
    raw_indices = []
    selected_points = []
    selected_indices = []
    pose_indices = []
    position_indices = []
    for pose_index, group in enumerate(pose_frames):
        if position_indices_by_pose is None:
            group_position_indices = [0] * len(group)
        else:
            group_position_indices = position_indices_by_pose[pose_index]
            if len(group_position_indices) != len(group):
                raise ValueError("position index count does not match captured frames")
        for frame, position_index in zip(group, group_position_indices):
            raw_points.append(frame.synchronized.points_sensor_m)
            raw_indices.append(frame.synchronized.sample_indices)
            selected_points.append(frame.selected_points_sensor_m)
            selected_indices.append(frame.selected_sample_indices)
            raw_offsets.append(raw_offsets[-1] + len(raw_points[-1]))
            selected_offsets.append(selected_offsets[-1] + len(selected_points[-1]))
            pose_indices.append(pose_index)
            position_indices.append(position_index)
    np.savez_compressed(
        destination,
        raw_points_sensor_m=np.vstack(raw_points),
        raw_sample_indices=np.concatenate(raw_indices),
        raw_frame_offsets=np.asarray(raw_offsets, dtype=np.int64),
        selected_points_sensor_m=np.vstack(selected_points),
        selected_sample_indices=np.concatenate(selected_indices),
        selected_frame_offsets=np.asarray(selected_offsets, dtype=np.int64),
        frame_pose_indices=np.asarray(pose_indices, dtype=np.int64),
        frame_position_indices=np.asarray(position_indices, dtype=np.int64),
        frame_stamps_ns=np.asarray(
            [frame.synchronized.stamp_ns for frame in frames], dtype=np.int64
        ),
        flange_rotations=np.asarray(
            [frame.synchronized.flange_rotation for frame in frames]
        ),
        flange_translations_m=np.asarray(
            [frame.synchronized.flange_translation_m for frame in frames]
        ),
        profile_circle_radii_m=np.asarray(
            [frame.circle_radius_m for frame in frames]
        ),
        profile_circle_rms_m=np.asarray([frame.circle_rms_m for frame in frames]),
        profile_chords_m=np.asarray([frame.chord_m for frame in frames]),
    )


def _write_markdown(path: Path, report: dict, metadata: dict) -> None:
    fixed = report["fixed_radius"]["all_points"]
    free = report["free_radius_diagnostic"]
    bootstrap = report["pose_bootstrap"]
    checks = report["checks"]
    lines = [
        "# 精密球真机手眼标定精度验证报告",
        "",
        f"- 结论：**{'通过' if report['passed'] else '未通过'}**。",
        f"- 标定球：{report['artifact']['artifact_id']}，"
        f"刻字直径 {report['artifact']['engraved_diameter_mm']:.4f} mm，"
        f"刻字圆度 {report['artifact']['engraved_roundness_mm']:.4f} mm。",
        f"- 固定的手眼结果：`{metadata['handeye_result_file']}`。",
        f"- 采集模式：`{metadata.get('collection_mode', 'stationary')}`。",
        f"- 独立验证位姿：{report['pose_count']}；有效球面点：{report['point_count']}。",
        "",
        "## 主要结果",
        "",
        "| 指标 | 结果 | 阈值 | 判定 |",
        "|---|---:|---:|---|",
        f"| 固定刻字半径、全部点 RMSE | {fixed['rmse_mm']:.4f} mm | "
        f"≤ {report['thresholds']['maximum_fixed_radius_rmse_mm']:.4f} mm | "
        f"{'通过' if checks['fixed_radius_rmse'] else '未通过'} |",
        f"| 固定刻字半径、全部点 P95 | {fixed['p95_abs_mm']:.4f} mm | "
        f"≤ {report['thresholds']['maximum_fixed_radius_p95_mm']:.4f} mm | "
        f"{'通过' if checks['fixed_radius_p95'] else '未通过'} |",
        f"| 自由拟合直径误差 | {free['diameter_error_mm']:.4f} mm | "
        f"|e| ≤ {report['thresholds']['maximum_free_diameter_error_mm']:.4f} mm | "
        f"{'通过' if checks['free_diameter_error'] else '未通过'} |",
        f"| 独立位姿数量 | {report['pose_count']} | "
        f"≥ {report['thresholds']['minimum_poses']} | "
        f"{'通过' if checks['minimum_pose_count'] else '未通过'} |",
        "",
        "固定半径球心（机器人基坐标系，m）："
        f"`{report['fixed_radius']['center_base_m']}`。",
        "",
        "## 逐位姿误差",
        "",
        "| 位姿 | 点数 | RMSE/mm | P95/mm | 最大误差/mm |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["per_pose"]:
        lines.append(
            f"| {row['pose_index']} | {row['count']} | {row['rmse_mm']:.4f} | "
            f"{row['p95_abs_mm']:.4f} | {row['maximum_abs_mm']:.4f} |"
        )
    lines += [
        "",
        "## 重复性诊断",
        "",
        f"- 留一位姿球心散布 RMS："
        f"{report['leave_one_pose_out']['center_spread_rms_mm']} mm。",
        f"- 位姿自助法成功次数：{bootstrap['successful_trials']}/"
        f"{bootstrap['requested_trials']}。",
        f"- 自助法拟合直径标准差：{bootstrap['diameter_std_mm']} mm。",
        "",
        "## 解释边界",
        "",
        "本报告冻结标定得到的手眼矩阵，只用标定球进行独立评价，"
        "不使用球面数据重新优化手眼参数。固定半径的全部点误差是主指标；"
        "鲁棒内点结果仅作故障诊断，不参与通过判定。刻字圆度是形状指标，"
        "不是刻字直径的计量不确定度；取得证书后应补充直径溯源信息。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="精密球真机手眼标定精度验证")
    parser.add_argument("--result", help="calibration_result.json path")
    parser.add_argument("--sphere", help="configured sphere artifact id")
    parser.add_argument(
        "--mode", choices=("moving", "stationary"),
        help="moving=人工步进移动扫描（评价方法2）；stationary=单截面静止采集",
    )
    parser.add_argument("--poses", type=int, help="number of directions/views")
    parser.add_argument(
        "--positions", type=int,
        help="manual translation stops per direction in moving mode",
    )
    parser.add_argument(
        "--frames", type=int,
        help="stationary frames per translation stop (moving) or per view (stationary)",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    parameters = _load_parameters(REAL_CONFIG)
    artifacts = _artifacts(parameters)
    artifact = _choose_artifact(artifacts, arguments.sphere)
    collection_mode = _choose_collection_mode(arguments.mode)
    result_file = _choose_result(arguments.result)
    handeye_rotation, handeye_translation, result_payload, result_sha256 = _load_handeye(
        result_file
    )
    result_mode = result_payload.get("simulation", {}).get("mode")
    if result_mode not in {None, "real"}:
        print(
            "警告：所选文件被标记为仿真结果。它可以用于联调，但不能形成真机精度结论。"
        )
        if not _yes_no("仍继续采集？", False):
            return 1

    if collection_mode == "moving":
        pose_count = arguments.poses or int(parameters.get("moving_directions", 7))
        position_count = arguments.positions or int(
            parameters.get("moving_positions_per_direction", 7)
        )
        frame_count = arguments.frames or int(
            parameters.get("moving_frames_per_position", 20)
        )
    else:
        pose_count = arguments.poses or int(parameters["target_poses"])
        position_count = 1
        frame_count = arguments.frames or int(parameters["frames_per_pose"])
    minimum_poses = int(parameters["thresholds"]["minimum_poses"])
    if pose_count < minimum_poses:
        raise ValueError(f"pose count must be at least {minimum_poses}")
    if frame_count < 3:
        raise ValueError("frames per position must be at least three")
    if collection_mode == "moving" and position_count < 3:
        raise ValueError("moving mode requires at least three translation positions")
    segmentation = _segment_parameters(parameters)
    thresholds = _thresholds(parameters)
    profile_topic = str(parameters["profile_topic"])
    flange_topic = str(parameters["flange_pose_topic"])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{artifact.artifact_id}"
    run_directory = VALIDATION_RUNS / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    print("=" * 68)
    print("             精密球真机手眼标定精度验证（只观察）")
    print("=" * 68)
    print(f"标定结果：{result_file}")
    if collection_mode == "moving":
        print(
            f"标定球：{artifact.artifact_id}，Ø{1000.0 * artifact.diameter_m:.4f} mm；"
            f"移动扫描：{pose_count} 个方向 × {position_count} 个平移位置 × "
            f"{frame_count} 个静止同步帧"
        )
        print(
            "同一方向内必须保持姿态不变、只做近似直线的单向平移；"
            "每个方向的所有位置会保存为同一评价组。"
        )
    else:
        print(
            f"标定球：{artifact.artifact_id}，Ø{1000.0 * artifact.diameter_m:.4f} mm；"
            f"静止截面：{pose_count} 个独立位姿 × {frame_count} 个静止同步帧"
        )
    print("本程序不会向机器人发送任何运动指令，也不会用球面数据修改手眼结果。")
    print("若直线断点节点在球面上显示 REJECTED，这是正常现象，与本验证无关。")

    rclpy.init()
    collector = ExactStampFrameCollector(profile_topic, flange_topic)
    executor_thread = threading.Thread(target=rclpy.spin, args=(collector,), daemon=True)
    executor_thread.start()
    pose_frames: list[list[AcceptedFrame]] = []
    position_indices_by_pose: list[list[int]] = []
    representative_poses: list[tuple[np.ndarray, np.ndarray]] = []
    direction_summaries: list[dict[str, object]] = []
    try:
        print("\n等待同步轮廓与法兰位姿……")
        deadline = time.monotonic() + float(parameters["initial_data_timeout_s"])
        while collector.paired_messages == 0 and time.monotonic() < deadline:
            time.sleep(0.1)
        if collector.paired_messages == 0:
            raise RuntimeError(
                "没有收到完全同步的轮廓。确认机器人已经静止、激光已开启，并检查 "
                "/measurement_sync/status。"
            )
        print("同步测量链正常。请固定标定球，全程不要移动球或底座。")
        if not _yes_no("球已牢固安装并确认开始？", True):
            return 0

        if collection_mode == "moving":
            (
                pose_frames,
                position_indices_by_pose,
                representative_poses,
                direction_summaries,
                cancelled,
            ) = _collect_moving_scan(
                collector,
                artifact,
                segmentation,
                parameters,
                pose_count,
                position_count,
                frame_count,
                float(parameters["capture_timeout_s"]),
            )
            if cancelled:
                print("\n已取消移动扫描；未完成的数据不会形成精度结论。")
        else:
            pose_index = 1
            while pose_index <= pose_count:
                print(
                    f"\n[{pose_index}/{pose_count}] 使用示教器低速移动机器人，"
                    "让激光切到球面不同区域；停稳后回到这里。"
                )
                command = input(
                    "按 Enter 采集；输入 q 结束；输入 r 重看提示："
                ).strip().lower()
                if command == "q":
                    break
                if command == "r":
                    continue
                try:
                    frames = _collect_pose(
                        collector,
                        artifact,
                        segmentation,
                        frame_count,
                        float(parameters["capture_timeout_s"]),
                    )
                except RuntimeError as error:
                    print(f"  本位姿采集失败：{error}")
                    print("  请调整曝光/入射角，使RViz中出现连续球面圆弧后重试。")
                    continue
                representative = _representative_flange(frames)
                if representative_poses:
                    rotation_separation = min(
                        rotation_distance_deg(previous[0], representative[0])
                        for previous in representative_poses
                    )
                    translation_separation_mm = 1000.0 * min(
                        np.linalg.norm(previous[1] - representative[1])
                        for previous in representative_poses
                    )
                    if (
                        rotation_separation
                        < float(parameters["minimum_pose_rotation_separation_deg"])
                        and translation_separation_mm
                        < float(parameters["minimum_pose_translation_separation_mm"])
                    ):
                        print(
                            f"  位姿多样性警告：距已有位姿最近仅 "
                            f"{rotation_separation:.2f}° / "
                            f"{translation_separation_mm:.1f} mm。"
                        )
                        if not _yes_no("仍保留这个位姿？", False):
                            print("  已丢弃，请重新移动机器人。")
                            continue
                pose_frames.append(frames)
                position_indices_by_pose.append([0] * len(frames))
                representative_poses.append(representative)
                pose_index += 1

        if len(pose_frames) < minimum_poses:
            raise RuntimeError(
                f"仅采集 {len(pose_frames)} 个位姿，少于最低要求 {minimum_poses}，"
                "不生成精度结论。"
            )

        _save_dataset(
            run_directory / "sphere_acquisition.npz",
            pose_frames,
            position_indices_by_pose,
        )
        pose_points_base = []
        for group in pose_frames:
            transformed = [
                transform_profile_to_base(
                    frame.selected_points_sensor_m,
                    frame.synchronized.flange_rotation,
                    frame.synchronized.flange_translation_m,
                    handeye_rotation,
                    handeye_translation,
                )
                for frame in group
            ]
            pose_points_base.append(np.vstack(transformed))
        report = validate_sphere_views(
            pose_points_base,
            artifact,
            robust_scale_m=float(parameters["fit_robust_scale_m"]),
            thresholds=thresholds,
            bootstrap_trials=int(parameters["bootstrap_trials"]),
            random_seed=int(parameters["random_seed"]),
        )
        metadata = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "handeye_result_file": str(result_file),
            "handeye_result_sha256": result_sha256,
            "handeye_snapshot": {
                "rotation": handeye_rotation.tolist(),
                "translation_m": handeye_translation.tolist(),
            },
            "artifact": artifact.as_dict_mm(),
            "profile_topic": profile_topic,
            "flange_pose_topic": flange_topic,
            "pose_count": len(pose_frames),
            "collection_mode": collection_mode,
            "positions_per_direction": position_count,
            "frames_per_position": frame_count,
            "frames_per_pose": position_count * frame_count,
            "direction_summaries": direction_summaries,
            "raw_dataset": "sphere_acquisition.npz",
            "sphere_data_used_to_update_handeye": False,
        }
        (run_directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_directory / "sphere_validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_markdown(
            run_directory / "sphere_validation_report.md", report, metadata
        )
        fixed = report["fixed_radius"]["all_points"]
        free = report["free_radius_diagnostic"]
        print("\n" + "=" * 68)
        print(f"验证结论：{'通过' if report['passed'] else '未通过'}")
        print(
            f"固定刻字半径：RMSE {fixed['rmse_mm']:.4f} mm，"
            f"P95 {fixed['p95_abs_mm']:.4f} mm"
        )
        print(
            f"自由球拟合：直径 {free['fitted_diameter_mm']:.4f} mm，"
            f"相对刻字误差 {free['diameter_error_mm']:+.4f} mm"
        )
        print(f"详细报告：{run_directory / 'sphere_validation_report.md'}")
        print(f"原始数据：{run_directory / 'sphere_acquisition.npz'}")
        return 0 if report["passed"] else 2
    finally:
        collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        executor_thread.join(timeout=2.0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已取消精密球验证；已采集但未形成完整实验的数据不作为精度结论。")
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"\n精密球验证失败：{error}", file=sys.stderr)
        sys.exit(1)
