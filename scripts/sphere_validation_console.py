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


def _save_dataset(
    destination: Path,
    pose_frames: list[list[AcceptedFrame]],
) -> None:
    frames = [frame for group in pose_frames for frame in group]
    raw_offsets = [0]
    selected_offsets = [0]
    raw_points = []
    raw_indices = []
    selected_points = []
    selected_indices = []
    pose_indices = []
    for pose_index, group in enumerate(pose_frames):
        for frame in group:
            raw_points.append(frame.synchronized.points_sensor_m)
            raw_indices.append(frame.synchronized.sample_indices)
            selected_points.append(frame.selected_points_sensor_m)
            selected_indices.append(frame.selected_sample_indices)
            raw_offsets.append(raw_offsets[-1] + len(raw_points[-1]))
            selected_offsets.append(selected_offsets[-1] + len(selected_points[-1]))
            pose_indices.append(pose_index)
    np.savez_compressed(
        destination,
        raw_points_sensor_m=np.vstack(raw_points),
        raw_sample_indices=np.concatenate(raw_indices),
        raw_frame_offsets=np.asarray(raw_offsets, dtype=np.int64),
        selected_points_sensor_m=np.vstack(selected_points),
        selected_sample_indices=np.concatenate(selected_indices),
        selected_frame_offsets=np.asarray(selected_offsets, dtype=np.int64),
        frame_pose_indices=np.asarray(pose_indices, dtype=np.int64),
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
    parser.add_argument("--poses", type=int, help="number of manually positioned views")
    parser.add_argument("--frames", type=int, help="stationary frames per view")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    parameters = _load_parameters(REAL_CONFIG)
    artifacts = _artifacts(parameters)
    artifact = _choose_artifact(artifacts, arguments.sphere)
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

    pose_count = arguments.poses or int(parameters["target_poses"])
    frame_count = arguments.frames or int(parameters["frames_per_pose"])
    minimum_poses = int(parameters["thresholds"]["minimum_poses"])
    if pose_count < minimum_poses:
        raise ValueError(f"pose count must be at least {minimum_poses}")
    if frame_count < 3:
        raise ValueError("frames per pose must be at least three")
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
    print(
        f"标定球：{artifact.artifact_id}，Ø{1000.0 * artifact.diameter_m:.4f} mm；"
        f"计划 {pose_count} 个独立位姿 × {frame_count} 个静止同步帧"
    )
    print("本程序不会向机器人发送任何运动指令，也不会用球面数据修改手眼结果。")
    print("若直线断点节点在球面上显示 REJECTED，这是正常现象，与本验证无关。")

    rclpy.init()
    collector = ExactStampFrameCollector(profile_topic, flange_topic)
    executor_thread = threading.Thread(target=rclpy.spin, args=(collector,), daemon=True)
    executor_thread.start()
    pose_frames: list[list[AcceptedFrame]] = []
    representative_poses: list[tuple[np.ndarray, np.ndarray]] = []
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

        pose_index = 1
        while pose_index <= pose_count:
            print(
                f"\n[{pose_index}/{pose_count}] 使用示教器低速移动机器人，"
                "让激光切到球面不同区域；停稳后回到这里。"
            )
            command = input("按 Enter 采集；输入 q 结束；输入 r 重看提示：").strip().lower()
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
                        f"{rotation_separation:.2f}° / {translation_separation_mm:.1f} mm。"
                    )
                    if not _yes_no("仍保留这个位姿？", False):
                        print("  已丢弃，请重新移动机器人。")
                        continue
            pose_frames.append(frames)
            representative_poses.append(representative)
            pose_index += 1

        if len(pose_frames) < minimum_poses:
            raise RuntimeError(
                f"仅采集 {len(pose_frames)} 个位姿，少于最低要求 {minimum_poses}，"
                "不生成精度结论。"
            )

        _save_dataset(run_directory / "sphere_acquisition.npz", pose_frames)
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
            "frames_per_pose": frame_count,
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
