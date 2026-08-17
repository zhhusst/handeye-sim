#!/usr/bin/env python3
"""Two-terminal console shared by simulation and real calibration backends."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import yaml


WORKSPACE = Path("/workspace")
PARAM_FILE = (
    WORKSPACE
    / "ros2_ws/src/handeye_sim_bridge/config/calibration.yaml"
)
REAL_PARAM_FILE = (
    WORKSPACE
    / "ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml"
)
RUNS_DIR = WORKSPACE / "data/calibration_runs"
LEGACY_SEED_FILE = WORKSPACE / "data/seed_measurements_v5.json"
SIM_ROTATION_ERROR_TARGET_DEG = 0.05
SIM_TRANSLATION_ERROR_TARGET_MM = 0.1
VALIDATED_NOISE_BASELINE = {
    "endpoint_gaussian_std_m": 0.000080,
    # Frozen 2026-08-07 error-budget baseline.  Direct endpoint injection is
    # disabled; the 0.05 mm flange disturbance is averaged within each
    # stationary synchronized batch.
    "robot_translation_std_m": 0.000050,
    "robot_rotation_std_deg": 0.003,
    # Flat-model ablation qualification allowance.
    "board_flatness_rms_m": 0.000010,
}
SHARED_SHAPE_VALIDATED_FLATNESS_RMS_M = 0.0005

STATE_NAMES = {
    "WAIT_MANUAL_INIT": "等待人工设置初始位姿",
    "MOVING": "机器人运动中",
    "SETTLING": "等待机器人稳定",
    "CAPTURING_SEED": "定点采集同步帧",
    "WAIT_SEEDS": "等待种子数据",
    "WAIT_JOINTS": "等待关节状态",
    "RANKING": "生成候选并计算信息增益",
    "CHECKING_GOAL": "检查目标位姿碰撞",
    "PLANNING": "MoveIt 路径规划",
    "EXECUTING": "执行下一最佳位姿",
    "WAIT_MEASUREMENT": "等待双边轮廓测量",
    "ROLLBACK": "观测无效，回退上一位姿",
    "WAIT_ROLLBACK_MEASUREMENT": "验证回退位姿双边观测",
    "DONE": "完成",
    "FAILED": "失败",
}

TARGET_NAMES = {
    "reference": "参考位姿",
    "rx_positive": "绕局部 X 轴正向",
    "rx_negative": "绕局部 X 轴负向",
    "ry_positive": "绕局部 Y 轴正向",
    "ry_negative": "绕局部 Y 轴负向",
    "rx_positive_half": "绕局部 X 轴正向 2.5°",
    "rx_negative_half": "绕局部 X 轴负向 2.5°",
    "ry_positive_half": "绕局部 Y 轴正向 2.5°",
    "ry_negative_half": "绕局部 Y 轴负向 2.5°",
    "rx_ry_positive": "绕局部 X/Y 轴组合",
    "rx_ry_opposite": "绕局部 X 正向/Y 负向补采",
    "rx_negative_ry_positive": "绕局部 X 负向/Y 正向补采",
    "rx_ry_negative": "绕局部 X/Y 负向补采",
    "ry_rx_positive": "绕局部 Y/X 正向补采",
    "ry_positive_rx_negative": "绕局部 Y 正向/X 负向补采",
    "ry_negative_rx_positive": "绕局部 Y 负向/X 正向补采",
    "ry_rx_negative": "绕局部 Y/X 负向补采",
    "preflight_rx_negative": "预检：局部 X 轴负向 2°",
    "preflight_rx_positive": "预检：局部 X 轴正向 2°",
    "preflight_ry_negative": "预检：局部 Y 轴负向 2°",
    "preflight_ry_positive": "预检：局部 Y 轴正向 2°",
    "complete": "已完成",
}

INITIAL_REASON_NAMES = {
    "measurement_missing": "缺少同步双边测量",
    "joints_missing": "缺少新鲜关节状态",
    "bilateral_not_safe": "双边不在安全域",
    "x_mid": "|x_mid| 超过 30 mm",
    "z_mid": "工作深度应在 300–550 mm",
    "domain_margin": "安全余量不足 20 mm",
    "profile_length": "轮廓长度应在 50–250 mm",
    "absolute_endpoint_depth_delta": "应满足 |z(e2)-z(e1)| ≥ 15 mm",
    "joint_margin": "关节限位余量不足 5%",
    "local_ik": "局部 ±X/±Y 的 2° IK 覆盖不足",
}

ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def clean_terminal_text(value: str) -> str:
    return ANSI_ESCAPE.sub("", value).strip()


def parse_key_values(message: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in message.split(";"):
        if "=" not in item:
            continue
        key, value = item.strip().split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def configured_surface_model(path: Path = PARAM_FILE) -> str:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return str(
            payload["/**"]["ros__parameters"]["solver"]["surface_model"]
        ).strip().lower()
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return "flat"


def noise_regime_exceedances(
    noise: dict, *, surface_model: str = "flat"
) -> list[tuple[str, float]]:
    """Return factors outside the end-to-end validated noise regime."""
    labels = {
        "endpoint_gaussian_std_m": "断点提取",
        "robot_translation_std_m": "机器人平移",
        "robot_rotation_std_deg": "机器人旋转",
        "board_flatness_rms_m": "平板平面度",
    }
    exceedances: list[tuple[str, float]] = []
    for name, baseline in VALIDATED_NOISE_BASELINE.items():
        if name == "board_flatness_rms_m" and surface_model == "shared":
            baseline = SHARED_SHAPE_VALIDATED_FLATNESS_RMS_M
        if (
            name == "endpoint_gaussian_std_m"
            and noise.get("direct_endpoint_injection_active") is False
        ):
            continue
        try:
            ratio = float(noise[name]) / baseline
        except (KeyError, TypeError, ValueError):
            continue
        if ratio > 1.5:
            exceedances.append((labels[name], ratio))
    return exceedances


def extract_trigger_response(output: str) -> tuple[bool, str]:
    text = clean_terminal_text(output)
    match = re.search(
        r"success=(True|False),\s*message='([^']*)'",
        text,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1) == "True", match.group(2)
    success_match = re.search(r"success:\s*(true|false)", text, flags=re.I)
    message_match = re.search(r"message:\s*['\"]?([^\n'\"]*)", text)
    if success_match:
        return (
            success_match.group(1).lower() == "true",
            message_match.group(1).strip() if message_match else text,
        )
    return False, text or "没有收到服务响应"


def run_command(
    arguments: list[str], *, timeout: float = 8.0
) -> subprocess.CompletedProcess:
    return subprocess.run(
        arguments,
        cwd=WORKSPACE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def ros_list(kind: str) -> set[str]:
    result = run_command(["ros2", kind, "list"], timeout=5.0)
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def ros_topic_has_publisher(topic: str) -> bool:
    try:
        result = run_command(["ros2", "topic", "info", topic], timeout=4.0)
    except subprocess.TimeoutExpired:
        return False
    match = re.search(r"Publisher count:\s*(\d+)", result.stdout)
    return (
        result.returncode == 0
        and match is not None
        and int(match.group(1)) > 0
    )


def call_trigger(service: str, *, timeout: float = 6.0) -> tuple[bool, str]:
    try:
        result = run_command(
            ["ros2", "service", "call", service, "std_srvs/srv/Trigger", "{}"],
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "服务暂时忙碌或响应超时"
    return extract_trigger_response(result.stdout)


def wait_for_service(service: str, process: "ManagedNode", timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        if service in ros_list("service"):
            return True
        time.sleep(0.5)
    return False


class ManagedNode:
    """Own one ROS child process and continuously drain its output to a log."""

    def __init__(self, name: str, arguments: list[str], log_file: Path) -> None:
        self.name = name
        self.arguments = arguments
        self.log_file = log_file
        self.process: subprocess.Popen | None = None
        self.recent_lines: deque[str] = deque(maxlen=30)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            self.arguments,
            cwd=WORKSPACE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        self._thread = threading.Thread(target=self._copy_output, daemon=True)
        self._thread.start()

    def _copy_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        with self.log_file.open("w", encoding="utf-8") as destination:
            for line in self.process.stdout:
                cleaned = clean_terminal_text(line)
                if cleaned:
                    self.recent_lines.append(cleaned)
                destination.write(line)
                destination.flush()

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def tail(self, count: int = 12) -> str:
        return "\n".join(list(self.recent_lines)[-count:])

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            self.process.wait(timeout=5.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(timeout=3.0)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if self.process.poll() is None:
                        os.killpg(self.process.pid, signal.SIGKILL)
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class CalibrationConsole:
    def __init__(
        self, *, backend: str = "simulation", poll_interval: float = 1.0
    ) -> None:
        normalized = backend.strip().lower()
        aliases = {"sim": "simulation", "simulation": "simulation", "real": "real"}
        if normalized not in aliases:
            raise ValueError("backend must be simulation or real")
        self.backend = aliases[normalized]
        self.poll_interval = poll_interval
        self.children: list[ManagedNode] = []

    def _parameter_file_arguments(self) -> list[str]:
        arguments = ["--params-file", str(PARAM_FILE)]
        if self.backend == "real":
            arguments += ["--params-file", str(REAL_PARAM_FILE)]
        return arguments

    def cleanup(self) -> None:
        for child in reversed(self.children):
            child.stop()
        self.children.clear()

    def _start_node(
        self, name: str, command: list[str], log_file: Path
    ) -> ManagedNode:
        child = ManagedNode(name, command, log_file)
        child.start()
        self.children.append(child)
        return child

    def _stop_node(self, child: ManagedNode) -> None:
        child.stop()
        if child in self.children:
            self.children.remove(child)

    def verify_simulation(self) -> bool:
        print("\n[环境检查] 正在检查 Gazebo、MoveIt 和轮廓仿真……")
        nodes = ros_list("node")
        topics = ros_list("topic")
        services = ros_list("service")
        required_nodes = {
            "/controller_manager",
            "/move_group",
            "/scene_publisher",
            "/profile_endpoint_detector",
            "/profile_viz_node",
        }
        required_topics = {
            "/joint_states",
            "/gocator/profile",
            "/calibration/target_surface_points",
            "/calibration/endpoints",
            "/calibration/flange_pose",
        }
        required_services = {
            "/controller_manager/list_controllers",
            "/scene_publisher/noise_status",
            "/profile_endpoint_detector/status",
            "/profile_endpoint_detector/lock",
            "/profile_endpoint_detector/reset",
        }
        missing_nodes = sorted(required_nodes - nodes)
        missing_topics = sorted(required_topics - topics)
        missing_services = sorted(required_services - services)
        unpowered_topics = sorted(
            topic
            for topic in required_topics
            if topic in topics and not ros_topic_has_publisher(topic)
        )
        if missing_nodes or missing_topics or missing_services or unpowered_topics:
            print("仿真环境尚未准备好。")
            if missing_nodes:
                print("  缺少节点：" + ", ".join(missing_nodes))
            if missing_topics:
                print("  缺少话题：" + ", ".join(missing_topics))
            if missing_services:
                print("  控制服务不可用：" + ", ".join(missing_services))
            if unpowered_topics:
                print("  话题没有发布者：" + ", ".join(unpowered_topics))
            print("\n请在第一个终端运行：")
            print("  cd /workspace && ./scripts/start_simulation.sh --web")
            return False
        conflicting = {
            "/bilateral_seed_collection",
            "/active_calibration_sim",
        } & nodes
        if conflicting:
            print("检测到遗留的标定节点：" + ", ".join(sorted(conflicting)))
            print("请先在之前的标定终端按 Ctrl+C，再重新运行本控制台。")
            return False
        print(
            "环境正常：MoveIt、原始轮廓、独立断点检测和同步位姿话题均已发现。"
        )
        success, message = call_trigger("/scene_publisher/noise_status")
        if success:
            try:
                noise = json.loads(message)
                print(
                    "仿真噪声："
                    f"轮廓 {1e3 * noise['profile_gaussian_std_m']:.3f} mm；"
                    f"机器人 {1e3 * noise['robot_translation_std_m']:.3f} mm/"
                    f"{noise['robot_rotation_std_deg']:.4f}°；"
                    f"平面度 {1e3 * noise['board_flatness_rms_m']:.3f} mm RMS；"
                    f"同步抖动 {1e3 * noise['sync_jitter_std_s']:.3f} ms"
                )
                print(
                    "           "
                    f"点离群 {100 * noise['point_outlier_probability']:.2f}%；"
                    f"点漏检 {100 * noise['point_dropout_probability']:.2f}%；"
                    f"整帧漏检 {100 * noise['frame_dropout_probability']:.2f}%"
                )
                if not noise.get("direct_endpoint_injection_active", True):
                    print(
                        "           直接断点真值注噪已禁用；断点误差由原始轮廓检测自然产生。"
                    )
                surface_model = configured_surface_model()
                exceedances = noise_regime_exceedances(
                    noise, surface_model=surface_model
                )
                if (
                    surface_model == "shared"
                    and float(noise.get("board_flatness_rms_m", 0.0))
                    <= SHARED_SHAPE_VALIDATED_FLATNESS_RMS_M
                ):
                    print(
                        "共享形貌后端：已启用；当前平面度位于0.5 mm RMS仿真"
                        "回归范围内。"
                    )
                if exceedances:
                    details = "；".join(
                        f"{label}为已验证基线的{ratio:.1f}倍"
                        for label, ratio in exceedances
                    )
                    print(
                        "精度适用性警告：" + details + "。"
                    )
                    print(
                        "  当前属于压力测试区；流程可以继续，但不能预期单次运行"
                        "必然达到0.05°/0.1 mm。建议逐项消融后再组合。"
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                print("仿真噪声状态：" + message)
        else:
            print("警告：无法读取仿真噪声状态：" + message)
        detector_success, detector_message = call_trigger(
            "/profile_endpoint_detector/status"
        )
        try:
            detector = json.loads(detector_message)
            if detector_success:
                print(
                    "断点检测：有效；"
                    "目标表面点 "
                    f"{detector.get('target_surface_points', detector.get('support_points', '?'))}；"
                    f"拟合RMS {float(detector.get('residual_rms_mm', 0.0)):.3f} mm；"
                    f"估计σ {float(detector.get('endpoint_sigma_mm', 0.0)):.3f} mm；"
                    f"置信度 {float(detector.get('confidence', 0.0)):.3f}"
                )
            else:
                print(
                    "断点检测暂未有效："
                    + str(detector.get("reason", detector_message))
                )
        except (TypeError, ValueError, json.JSONDecodeError):
            print("断点检测状态：" + detector_message)
        return True

    def verify_real(self) -> bool:
        print("\n[环境检查] 正在检查FANUC与Gocator真机环境……")
        nodes = ros_list("node")
        topics = ros_list("topic")
        services = ros_list("service")
        required_nodes = {
            "/fanuc_joint_state",
            "/gocator_profile_driver",
            "/gocator_metric_adapter",
            "/measurement_sync",
            "/profile_endpoint_detector",
            "/profile_viz_node",
        }
        required_topics = {
            "/fanuc/joint_states_raw",
            "/gocator/profile_raw_mm",
            "/gocator/profile",
            "/calibration/target_surface_points",
            "/calibration/endpoints",
        }
        required_services = {
            "/fanuc_joint_state/status",
            "/gocator_metric_adapter/status",
            "/measurement_sync/status",
            "/profile_endpoint_detector/status",
            "/profile_endpoint_detector/lock",
            "/profile_endpoint_detector/reset",
        }
        missing_nodes = sorted(required_nodes - nodes)
        missing_topics = sorted(required_topics - topics)
        missing_services = sorted(required_services - services)
        if missing_nodes or missing_topics or missing_services:
            print("真机只观察环境尚未准备好。")
            if missing_nodes:
                print("  缺少节点：" + ", ".join(missing_nodes))
            if missing_topics:
                print("  缺少话题：" + ", ".join(missing_topics))
            if missing_services:
                print("  缺少状态服务：" + ", ".join(missing_services))
            print("\n请在第一个终端运行：")
            print("  cd /workspace && ./scripts/start_environment.sh real")
            return False

        joint_ok, joint_message = call_trigger("/fanuc_joint_state/status")
        try:
            joint_status = json.loads(joint_message)
        except (TypeError, ValueError, json.JSONDecodeError):
            joint_status = {"last_error": joint_message}
        if not joint_ok:
            print("FANUC状态不可用：" + str(joint_status.get("last_error", "未知")))
            return False
        if not joint_status.get("j23_validated", False):
            print(
                "FANUC通信正常，但J23约定尚未验证；为防止错误FK，"
                "系统没有发布标定用/joint_states。"
            )
            print("当前只能查看原始控制器关节，不能开始采集。")
            return False
        required_calibration_topics = {
            "/joint_states",
            "/calibration/flange_pose",
        }
        unavailable = sorted(
            topic
            for topic in required_calibration_topics
            if topic not in topics or not ros_topic_has_publisher(topic)
        )
        if unavailable:
            print("标定同步链尚未输出：" + ", ".join(unavailable))
            return False
        if "/fanuc_motion_bridge/status" in services:
            _, motion_message = call_trigger("/fanuc_motion_bridge/status")
            try:
                motion = json.loads(motion_message)
                print(
                    "真机测量链正常；运动桥 "
                    f"mode={motion.get('mode', '?')}，"
                    f"writes={motion.get('motion_writes_enabled', False)}，"
                    f"state={motion.get('state', '?')}（启动默认未解锁）。"
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                print("真机测量链正常；运动桥状态暂不可解析。")
        else:
            print("真机测量链正常；当前为observe_only，不具备自动运动权限。")
        return True

    def _real_motion_status(self) -> dict | None:
        success, message = call_trigger("/fanuc_motion_bridge/status")
        try:
            payload = json.loads(message)
        except (TypeError, ValueError, json.JSONDecodeError):
            print("无法解析真机运动桥状态：" + message)
            return None
        if not success and payload.get("mode") != "plan_only":
            print("真机运动桥未就绪：" + str(payload.get("last_error", message)))
        return payload

    def _arm_real_motion(self) -> bool:
        status = self._real_motion_status()
        if status is None:
            print("第一终端没有可用的真机运动桥。请重新启动：")
            print(
                "  ./scripts/start_environment.sh real "
                "--motion-mode automatic"
            )
            return False
        mode = status.get("mode")
        if mode == "plan_only" or not status.get("motion_writes_enabled", False):
            print("运动桥当前只规划、不执行。请重新启动第一终端：")
            print(
                "  ./scripts/start_environment.sh real "
                "--motion-mode automatic"
            )
            return False
        success, message = call_trigger("/fanuc_motion_bridge/arm")
        if not success:
            print("运动桥解锁失败：" + message)
            return False
        print("真机运动桥已解锁；标定结束或异常退出时会自动软件撤防。")
        return True

    @staticmethod
    def _print_motion_safety() -> None:
        print("\n[真机运动安全确认]")
        print("  · 示教器运行 PC_TRACK_ALL，且程序指针已进入循环；")
        print("  · 当前 UF/UT 为 1/1，R[100]=0；")
        print("  · 使用 T1/低速，人员已离开运动范围并可随时按 HOLD/急停；")
        print("  · 每个目标还会经过关节步长、局部直线路径 IK 和 CURPOS/FK 一致性检查。")

    @staticmethod
    def _disarm_real_motion() -> None:
        call_trigger("/fanuc_motion_bridge/disarm", timeout=2.0)

    def verify_environment(self) -> bool:
        if self.backend == "real":
            return self.verify_real()
        return self.verify_simulation()

    @staticmethod
    def make_run_directory() -> Path:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = RUNS_DIR / run_id
        suffix = 1
        while destination.exists():
            destination = RUNS_DIR / f"{run_id}_{suffix}"
            suffix += 1
        destination.mkdir(parents=True)
        return destination

    def run_new(self, mode: str) -> None:
        run_directory = self.make_run_directory()
        seed_file = run_directory / "seeds.json"
        result_file = run_directory / "calibration_result.json"
        print(f"\n本次运行目录：{run_directory}")
        preflight_mode = (
            ask_preflight_mode() if mode == "automatic" else "off"
        )
        seed_node = self._start_seed_node(
            mode, seed_file, run_directory, preflight_mode
        )
        real_motion_armed = False
        try:
            if not wait_for_service(
                "/bilateral_seed_collection/status", seed_node
            ):
                self._node_start_failure(seed_node)
                return
            reset_success, reset_message = call_trigger(
                "/profile_endpoint_detector/reset"
            )
            if not reset_success:
                print(f"无法复位目标引导检测器：{reset_message}")
                return
            status = self._confirm_initial_pose(mode, preflight_mode)
            if status is None:
                return
            if mode == "automatic":
                if self.backend == "real":
                    self._print_motion_safety()
                if not ask_yes_no("确认开始自动采集 6 个种子位姿？", default=True):
                    print("已取消种子采集。")
                    return
                if self.backend == "real":
                    real_motion_armed = self._arm_real_motion()
                    if not real_motion_armed:
                        return
                success, message = call_trigger(
                    "/bilateral_seed_collection/start"
                )
                if not success:
                    print(f"启动失败：{message}")
                    return
                if not self._monitor_automatic_seeds(seed_node, seed_file):
                    return
            else:
                success, message = call_trigger(
                    "/bilateral_seed_collection/start"
                )
                if not success:
                    print(f"人工采集启动失败：{message}")
                    return
                print("\n初始位姿已确认，现在把它保存为第 1 个人工种子。")
                if not self._capture_manual_seed(1):
                    return
                if not self._collect_manual_seeds(seed_node, start_index=2):
                    return
        finally:
            self._stop_node(seed_node)
            if self.backend == "real" and real_motion_armed:
                self._disarm_real_motion()

        if not seed_file.exists():
            print("没有生成种子文件，流程结束。")
            return
        self._print_seed_summary(seed_file)
        if self.backend == "real":
            # NBV 安全验收已通过：真机允许进入主动标定，但保留运动安全确认。
            print(
                "真机种子数据采集完成；NBV 执行已通过安全验收，"
                "继续前请确认以下条件。"
            )
            self._print_motion_safety()
            if not ask_yes_no("真机自动运动仍处于解锁状态，确认开始主动标定？", default=False):
                print(f"已保留种子文件：{seed_file}")
                return
            maximum_nbv = ask_integer(
                "主动 NBV 可选预算（真机建议先用 1~3 个小预算验证）",
                default=1,
                minimum=0,
                maximum=20,
            )
            self._run_active(seed_file, result_file, run_directory, maximum_nbv)
            return
        if not ask_yes_no("种子采集完成，确认开始主动标定？", default=True):
            print(f"已保留种子文件：{seed_file}")
            return
        maximum_nbv = ask_integer(
            "主动 NBV 可选预算（0=自适应停止；内部仍保留20个NBV紧急保护）",
            default=0,
            minimum=0,
            maximum=20,
        )
        self._run_active(seed_file, result_file, run_directory, maximum_nbv)

    def _start_seed_node(
        self,
        mode: str,
        seed_file: Path,
        run_directory: Path,
        preflight_mode: str,
    ) -> ManagedNode:
        command = [
            "ros2",
            "run",
            "handeye_sim_bridge",
            "seed_collection",
            "--ros-args",
            *self._parameter_file_arguments(),
            "-p",
            f"collection_mode:={mode}",
            "-p",
            f"output_file:={seed_file}",
            "-p",
            f"seed.preflight.mode:='{preflight_mode}'",
        ]
        print(
            "\n[种子节点] "
            + ("自动采集模式" if mode == "automatic" else "人工采集模式")
            + f"；动态预检={preflight_mode}"
        )
        return self._start_node(
            "种子采集节点", command, run_directory / "seed_collection.log"
        )

    def _status(
        self, service: str, *, timeout: float = 5.0
    ) -> tuple[bool, dict[str, str], str]:
        success, message = call_trigger(service, timeout=timeout)
        return success, parse_key_values(message), message

    def _confirm_initial_pose(
        self, collection_mode: str, preflight_mode: str
    ) -> dict[str, str] | None:
        positioning_tool = (
            "FANUC示教器（T1/手动低速）"
            if self.backend == "real"
            else "RViz"
        )
        print(
            f"\n请使用{positioning_tool}设置初始位姿，把黄色原始轮廓中的平板"
            "表面线段移动到 RViz 紫色期望线及其矩形 ROI 内，并满足下方初始工作包络。"
        )
        print(
            "真实现场不需要看见空气中的激光平面：以 Gocator 轮廓、双边断点和"
            "这里的数值反馈为准；绿色线是实际选中的平板表面，Δz 正负均可。"
        )
        print("设置完成后回到这里按 Enter；输入 r 刷新检测状态，输入 q 取消。")
        while True:
            success, fields, raw = self._status(
                "/bilateral_seed_collection/status"
            )
            if not success and not fields:
                print(f"  暂时无法读取观测状态：{raw}")
            else:
                self._print_observation(fields)
            detector_success, detector_message = call_trigger(
                "/profile_endpoint_detector/status"
            )
            try:
                detector = json.loads(detector_message)
            except (TypeError, ValueError, json.JSONDecodeError):
                detector = {}
            if detector:
                guide_first = detector.get("guide_first_mm", ["?", "?", "?"])
                guide_second = detector.get("guide_second_mm", ["?", "?", "?"])
                print(
                    "  引导检测："
                    f"模式 {detector.get('mode', '?')}；"
                    f"稳定帧 {detector.get('alignment_stable_frames', 0)}/"
                    f"{detector.get('minimum_lock_frames', '?')}；"
                    f"法向ROI ±{detector.get('guide_normal_gate_mm', '?')} mm；"
                    f"期望端点 XZ=({guide_first[0]}, {guide_first[2]})/"
                    f"({guide_second[0]}, {guide_second[2]}) mm；"
                    f"{'已识别' if detector_success else '未稳定识别'}"
                )
            choice = input("初始位姿 [Enter=确认, r=刷新, q=取消]：").strip().lower()
            if choice == "q":
                print("已取消。")
                return None
            if choice == "r":
                continue
            if (
                fields.get("observation") == "SAFE"
                and fields.get("stable") == "true"
                and fields.get("initial_ready") == "true"
            ):
                lock_success, lock_message = call_trigger(
                    "/profile_endpoint_detector/lock"
                )
                if not lock_success:
                    print(
                        "目标线段尚不能锁定："
                        f"{lock_message}。请保持机器人静止后输入 r 刷新。"
                    )
                    continue
                if collection_mode == "manual":
                    print("初始位姿静态检查通过。人工种子模式不执行动态预检。")
                elif fields.get("preflight_required") == "true":
                    print(
                        "初始位姿静态检查通过。当前模式将执行局部"
                        " ±X/±Y 各 2° 的动态预检。"
                    )
                elif preflight_mode == "auto":
                    print(
                        "初始位姿静态余量充分，auto 模式将跳过独立动态预检；"
                        "正式种子运动仍逐步检查双边并支持回退。"
                    )
                else:
                    print(
                        "初始位姿静态检查通过。已选择跳过独立动态预检；"
                        "正式种子运动仍逐步检查双边并支持回退。"
                    )
                return fields
            print("当前位姿还不满足初始工作包络，请按提示调整后再确认。")

    def _print_observation(self, fields: dict[str, str]) -> None:
        observation = fields.get("observation", "UNKNOWN")
        stable = fields.get("stable", "false")
        observation_cn = {
            "SAFE": "双边有效且位于安全域",
            "UNSAFE": "检测到双边，但端点超出安全域",
            "MISSING": "没有同步检测到两条边",
        }.get(observation, observation)
        print(
            "  当前检测："
            f"{observation_cn}；"
            f"机器人{'已稳定' if stable == 'true' else '仍在运动'}；"
            f"轮廓点 {fields.get('profile_points', '?')}；"
            f"x_mid {fields.get('x_mid_mm', '?')} mm；"
            f"z_mid {fields.get('z_mid_mm', '?')} mm；"
            f"安全余量 {fields.get('safe_margin_mm', '?')} mm；"
            f"轮廓长度 {fields.get('profile_length_mm', '?')} mm；"
            f"Δz(e2-e1) {fields.get('endpoint_depth_delta_mm', '?')} mm；"
            f"|Δz| {fields.get('absolute_endpoint_depth_delta_mm', '?')} mm；"
            f"关节余量 {fields.get('joint_margin_percent', '?')}%；"
            f"局部IK {fields.get('local_ik', '?')}"
        )
        if fields.get("initial_ready") == "true":
            print("  初始工作包络：通过")
            return
        reason_codes = fields.get("initial_reasons", "").split(",")
        reason_names = dict(INITIAL_REASON_NAMES)
        if self.backend == "real":
            reason_names["z_mid"] = "工作深度应在真机配置的 80–450 mm 正深度范围"
        reasons = [
            reason_names.get(code, code)
            for code in reason_codes
            if code and code != "none"
        ]
        if reasons:
            print("  初始工作包络：不通过；" + "；".join(reasons))
            depth_advice = (
                "让 z_mid 保持在 80–450 mm 的真机正深度范围"
                if self.backend == "real"
                else "让 z_mid 接近 400–450 mm"
            )
            print(
                "  调整建议：一次只做小幅移动并刷新，优先让双边进入安全域，"
                f"再{depth_advice}、x_mid 接近 0；"
                "|Δz| 太小时只需增大腕部倾斜，方向正负均可。"
            )

    def _monitor_automatic_seeds(
        self, node: ManagedNode, seed_file: Path
    ) -> bool:
        print("\n[自动种子采集] 已开始。机器人会自动运动，请勿在 RViz 中下发目标。")
        previous_count = -1
        previous_target = ""
        started_at = time.monotonic()
        last_status_at = started_at
        approval_seen = False
        while True:
            if node.poll() is not None:
                self._node_start_failure(node, "种子节点意外退出")
                return False
            success, fields, raw = self._status(
                "/bilateral_seed_collection/status"
            )
            if not fields:
                print("\r[自动种子] 状态服务暂时忙碌，继续等待……", end="", flush=True)
                if time.monotonic() - last_status_at > 10.0:
                    print("\n自动种子节点已超过 10 秒无法提供状态。")
                    print("这通常表示 ROS 控制器或 DDS 通信已经中断。")
                    print(f"详细日志：{node.log_file}")
                    print(node.tail())
                    return False
                time.sleep(self.poll_interval)
                continue
            if self.backend == "real":
                motion = self._real_motion_status()
                waiting = motion is not None and motion.get("state") == "WAIT_APPROVAL"
                if waiting and not approval_seen:
                    print("\n\n运动桥正在等待本步确认：")
                    plan = motion.get("last_plan") or {}
                    print(
                        "  最大单轴变化 "
                        f"{float(plan.get('maximum_joint_step_deg', 0.0)):.2f}°；"
                        "关节距离 "
                        f"{float(plan.get('joint_distance_rad', 0.0)):.3f} rad；"
                        "直线位移 "
                        f"{float(plan.get('translation_mm', 0.0)):.2f} mm；"
                        "姿态变化 "
                        f"{float(plan.get('rotation_deg', 0.0)):.2f}°"
                    )
                    if not ask_yes_no("允许执行这一步？", default=False):
                        self._disarm_real_motion()
                        print("已拒绝该步并撤防。")
                        return False
                    approved, approval_message = call_trigger(
                        "/fanuc_motion_bridge/approve"
                    )
                    if not approved:
                        print("批准失败：" + approval_message)
                        return False
                    approval_seen = True
                elif not waiting:
                    approval_seen = False
            last_status_at = time.monotonic()
            count_text = fields.get("seeds", "0/6")
            try:
                count = int(count_text.split("/", 1)[0])
            except ValueError:
                count = 0
            target = fields.get("target", "?")
            target_cn = TARGET_NAMES.get(target.replace("_partial", ""), target)
            if target.endswith("_partial"):
                target_cn += "（安全部分角度）"
            state = fields.get("state", "?")
            state_cn = STATE_NAMES.get(state, state)
            rotation = fields.get("rotation_deg", "?")
            target_failures = fields.get("target_failures", "0/3")
            preflight = fields.get("preflight", "0/4")
            preflight_mode = fields.get("preflight_mode", "?")
            if fields.get("preflight_required") == "false":
                preflight_text = f"已跳过({preflight_mode})"
            elif fields.get("phase") == "PREFLIGHT":
                preflight_text = f"进行中({preflight_mode}) {preflight}"
            else:
                preflight_text = f"已执行({preflight_mode}) {preflight}"
            seed_batch = fields.get("seed_batch", "0/?")
            elapsed = time.monotonic() - started_at
            line = (
                f"[自动种子] {progress_bar(count, 6)} {count_text} | "
                f"{target_cn} | {state_cn} | 旋转 {rotation}° | "
                f"定点帧 {seed_batch} | "
                f"预检 {preflight_text} | 本目标失败 {target_failures} | "
                f"已用 {elapsed:.0f}s"
            )
            print("\r" + line.ljust(150), end="", flush=True)
            if count != previous_count and count > 0:
                print()
                print(f"  ✓ 第 {count}/6 个种子已通过双边与旋转多样性检查")
            elif target != previous_target and previous_target:
                print()
            previous_count = count
            previous_target = target
            if state == "DONE":
                print(f"\n自动种子采集完成，用时 {elapsed:.1f} 秒。")
                return seed_file.exists()
            if state == "FAILED" or not success:
                print("\n自动种子采集失败。")
                failure_reason = fields.get("failure_reason", "none")
                if failure_reason != "none":
                    print(f"失败原因：{failure_reason}")
                print(node.tail())
                return False
            time.sleep(self.poll_interval)

    def _capture_manual_seed(self, index: int) -> bool:
        success, message = call_trigger("/bilateral_seed_collection/capture")
        if not success:
            print(f"  ✗ 当前位姿未开始采集：{message}")
            return False
        print(f"  正在定点采集同步帧：{message}")
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            status_success, fields, raw = self._status(
                "/bilateral_seed_collection/status"
            )
            count_text = fields.get("seeds", "0/6")
            try:
                count = int(count_text.split("/", 1)[0])
            except ValueError:
                count = 0
            batch = fields.get("seed_batch", "?")
            print(
                f"\r  定点多帧进度：{batch}；鲁棒筛选后保存",
                end="",
                flush=True,
            )
            if count >= index:
                print(f"\n  ✓ 第 {index}/6 个物理种子多帧保存成功")
                return True
            if fields.get("state") == "FAILED" or (
                not status_success and not fields
            ):
                print(f"\n  ✗ 多帧采集失败：{raw}")
                return False
            if fields.get("state") == "WAIT_MANUAL_INIT":
                print("\n  ✗ 本批次有效内点不足，请保持位姿稳定后重试。")
                return False
            time.sleep(0.25)
        print("\n  ✗ 多帧采集等待超时。")
        return False

    def _collect_manual_seeds(
        self, node: ManagedNode, *, start_index: int
    ) -> bool:
        for index in range(start_index, 7):
            positioning_tool = (
                "FANUC示教器（T1/手动低速）"
                if self.backend == "real"
                else "RViz"
            )
            print(
                f"\n[人工种子 {index}/6] 请使用{positioning_tool}移动到新的安全位姿。"
            )
            print("建议相对已有位姿改变末端姿态至少 3°，并保持双边可见。")
            while True:
                if node.poll() is not None:
                    self._node_start_failure(node, "种子节点意外退出")
                    return False
                success, fields, raw = self._status(
                    "/bilateral_seed_collection/status"
                )
                if fields:
                    self._print_observation(fields)
                elif not success:
                    print(f"  状态读取失败：{raw}")
                choice = input(
                    f"种子 {index}/6 [Enter=采集, r=刷新, q=取消]："
                ).strip().lower()
                if choice == "q":
                    print("已取消人工采集。")
                    return False
                if choice == "r":
                    continue
                if (
                    fields.get("observation") != "SAFE"
                    or fields.get("stable") != "true"
                ):
                    print("当前双边观测不安全或机器人尚未稳定，未执行采集。")
                    continue
                if self._capture_manual_seed(index):
                    break
                print("请改变末端姿态后重试。")
        return True

    @staticmethod
    def _print_seed_summary(seed_file: Path) -> None:
        try:
            payload = json.loads(seed_file.read_text(encoding="utf-8"))
            records = payload.get("seeds", [])
            diversity = payload.get("rotation_diversity", {})
            print("\n种子数据摘要")
            print(f"  文件：{seed_file}")
            observation_count = int(
                payload.get(
                    "observation_count",
                    sum(
                        len(record.get("frames", [record]))
                        for record in records
                    ),
                )
            )
            print(f"  物理位姿数量：{len(records)}")
            print(f"  同步观测总数：{observation_count}")
            if payload.get("measurement_batch_size"):
                print(
                    "  每物理位姿目标帧数："
                    f"{int(payload['measurement_batch_size'])}"
                )
            print(
                "  标签："
                + ", ".join(str(record.get("label", "?")) for record in records)
            )
            if diversity:
                print(
                    "  最小两两旋转间隔："
                    f"{float(diversity.get('minimum_pairwise_deg', 0.0)):.3f}°"
                )
                eigenvalues = diversity.get("gram_eigenvalues", [])
                if eigenvalues:
                    print(
                        "  旋转激励 Gram 特征值："
                        + ", ".join(f"{float(value):.4e}" for value in eigenvalues)
                    )
        except (OSError, ValueError, TypeError) as error:
            print(f"种子摘要读取失败：{error}")

    def run_existing(self, *, initialization_only: bool = False) -> None:
        default = self._latest_seed_file()
        prompt = f"种子文件 [{default}]："
        value = input(prompt).strip()
        seed_file = Path(value).expanduser() if value else default
        if not seed_file.is_absolute():
            seed_file = WORKSPACE / seed_file
        try:
            payload = json.loads(seed_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"无法读取种子文件：{error}")
            return
        count = len(payload.get("seeds", []))
        if count < 6:
            print(f"种子数量不足：当前 {count}/6。")
            return
        self._print_seed_summary(seed_file)
        maximum_nbv = ask_integer(
            "主动 NBV 可选预算（0=自适应停止；内部仍保留20个NBV紧急保护）",
            default=0,
            minimum=0,
            maximum=20,
        )
        run_directory = self.make_run_directory()
        result_file = run_directory / "calibration_result.json"
        self._run_active(
            seed_file,
            result_file,
            run_directory,
            maximum_nbv,
            initialization_only=initialization_only,
        )

    @staticmethod
    def _latest_seed_file() -> Path:
        candidates = sorted(
            RUNS_DIR.glob("*/seeds.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        return LEGACY_SEED_FILE

    def _run_active(
        self,
        seed_file: Path,
        result_file: Path,
        run_directory: Path,
        maximum_nbv: int,
        *,
        initialization_only: bool = False,
    ) -> None:
        command = [
            "ros2",
            "run",
            "handeye_sim_bridge",
            "active_calibration",
            "--ros-args",
            *self._parameter_file_arguments(),
            "-p",
            "auto_start:=false",
            "-p",
            f"seed_file:={seed_file}",
            "-p",
            f"output_file:={result_file}",
            "-p",
            f"maximum_nbv_poses:={maximum_nbv}",
        ]
        if initialization_only:
            command += ["-p", "initialization_only:=true"]
        print("\n[主动标定] 正在启动节点……")
        node = self._start_node(
            "主动标定节点", command, run_directory / "active_calibration.log"
        )
        try:
            if not wait_for_service("/active_calibration_sim/start", node):
                self._node_start_failure(node)
                return
            success, message = call_trigger("/active_calibration_sim/start")
            if not success:
                print(f"主动标定启动失败：{message}")
                return
            print(
                "主动标定已开始：联合初始化 → 候选评分 → MoveIt 规划 "
                "→ Gazebo 执行 → 双边验证 → 滚动更新。"
            )
            self._monitor_active(node, result_file)
        finally:
            if not initialization_only:
                self._stop_node(node)
            else:
                print(
                    "[只初始化] 节点保持运行以持续发布平板位姿 marker；"
                    "观察完毕后可在控制台按 Ctrl+C 或重启环境停止。"
                )

    def _monitor_active(self, node: ManagedNode, result_file: Path) -> bool:
        displayed_iterations = 0
        previous_state = ""
        unavailable_reported = False
        while True:
            if node.poll() is not None:
                displayed_iterations = self._show_new_iterations(
                    result_file, displayed_iterations
                )
                self._node_start_failure(node, "主动标定节点意外退出")
                return False

            displayed_iterations = self._show_new_iterations(
                result_file, displayed_iterations
            )
            success, fields, raw = self._status(
                "/active_calibration_sim/status", timeout=4.0
            )
            if not fields:
                if not unavailable_reported:
                    print("  · 求解或候选评分正在计算，状态服务暂时忙碌……")
                    unavailable_reported = True
                time.sleep(self.poll_interval)
                continue
            unavailable_reported = False
            state = fields.get("state", "?")
            if state != previous_state:
                state_cn = STATE_NAMES.get(state, state)
                nbv = fields.get("nbv", "?")
                print(f"  → {state_cn}（NBV {nbv}）")
                candidate = fields.get("candidate", "not_selected")
                if candidate != "not_selected":
                    print(f"    当前候选：{candidate}")
                previous_state = state
            if state == "DONE":
                self._show_new_iterations(result_file, displayed_iterations)
                print("\n主动标定完成。")
                self._print_final_result(result_file)
                return True
            if state == "FAILED" or (not success and state):
                print(f"\n主动标定失败：{fields.get('stop_reason', raw)}")
                print(f"详细日志：{node.log_file}")
                print(node.tail())
                return False
            time.sleep(self.poll_interval)

    @staticmethod
    def _show_new_iterations(result_file: Path, displayed: int) -> int:
        if not result_file.exists():
            return displayed
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            iterations = payload.get("simulation", {}).get("iterations", [])
        except (OSError, json.JSONDecodeError):
            return displayed
        for record in iterations[displayed:]:
            phase = record.get("phase")
            if phase == "initial":
                print("\n  ┌─ 六种子联合初始化结果")
            else:
                print(f"\n  ┌─ NBV {record.get('nbv_index')} 位姿更新结果")
            candidate = record.get("candidate")
            if candidate:
                print(
                    "  │ 候选："
                    f"{candidate.get('id')}；"
                    f"有效概率 {candidate.get('valid_probability', 0.0):.3f}；"
                    f"信息增益 {candidate.get('information_gain', 0.0):.6g}"
                )
                if "joint_distance_rad" in candidate:
                    print(
                        "  │ 运动代价："
                        f"关节距离 {candidate.get('joint_distance_rad', 0.0):.3f} rad；"
                        f"最大单轴变化 {candidate.get('maximum_joint_step_deg', 0.0):.1f}°"
                    )
                print(
                    "  │ 位姿参数："
                    f"a={candidate.get('a_m', 0.0):.3f} m，"
                    f"b={candidate.get('b_m', 0.0):.3f} m，"
                    f"α={candidate.get('alpha_deg', 0.0):.1f}°，"
                    f"ψ={candidate.get('psi_deg', 0.0):.1f}°，"
                    f"工作距离={candidate.get('working_distance_m', 0.0):.3f} m"
                )
                print(
                    "  │ 名义安全余量："
                    f"ROI {1000.0 * candidate.get('roi_margin_m', 0.0):.2f} mm；"
                    f"边段 {1000.0 * candidate.get('edge_margin_m', 0.0):.2f} mm"
                )
            translation = record.get("handeye_translation_m", [0.0, 0.0, 0.0])
            print(
                "  │ 手眼平移："
                + "["
                + ", ".join(f"{1000.0 * float(value):.3f}" for value in translation)
                + "] mm"
            )
            print(
                "  │ 求解（仅观测数据）："
                f"cost={record.get('cost', 0.0):.6g}，"
                f"rank={record.get('data_only_rank', record.get('rank'))}，"
                "condition="
                f"{record.get('data_only_condition_number', record.get('condition_number', 0.0)):.3e}"
            )
            if "prior_augmented_rank" in record:
                print(
                    "  │ 求解（观测+形貌先验）："
                    f"rank={record.get('prior_augmented_rank')}，"
                    "condition="
                    f"{record.get('prior_augmented_condition_number', 0.0):.3e}"
                )
            if record.get("surface_model") == "shared":
                print(
                    "  │ 共享形貌："
                    f"RMS {float(record.get('surface_rms_mm', 0.0)):.4f} mm；"
                    f"最大绝对高度 "
                    f"{float(record.get('surface_maximum_mm', 0.0)):.4f} mm"
                )
            if "maximum_rotation_std_deg" in record:
                print(
                    "  │ 预测1σ精度（不使用真值）："
                    f"旋转≤{record.get('maximum_rotation_std_deg', 0.0):.4f}°；"
                    f"平移≤{record.get('maximum_translation_std_mm', 0.0):.4f} mm"
                )
            validation_score = record.get(
                "held_out_validation_score_mm"
            )
            if validation_score is not None:
                print(
                    "  │ 留出帧几何诊断（不参与停止或回滚）："
                    f"{float(validation_score):.4f} mm"
                )
            stability = record.get("initial_stability")
            if phase == "initial" and isinstance(stability, dict):
                if stability.get("available"):
                    print(
                        "  │ 重采样稳定性（不使用真值）："
                        f"{stability.get('trials_converged', 0)}/"
                        f"{stability.get('trials_requested', 0)} 收敛；"
                        f"旋转P95 {stability.get('rotation_p95_deg', 0.0):.4f}°；"
                        f"平移P95 "
                        f"{stability.get('translation_p95_mm', 0.0):.4f} mm；"
                        f"{'通过' if stability.get('accepted') else '不通过'}"
                    )
                else:
                    print("  │ 重采样稳定性：旧版单帧种子，不可评估")
            rotation_error = record.get("rotation_error_deg")
            translation_error = record.get("translation_error_mm")
            if rotation_error is None or translation_error is None:
                print("  └─ 真值误差：真机模式无仿真真值（N/A）")
            else:
                print(
                    "  └─ 仿真真值误差："
                    f"旋转 {float(rotation_error):.4f}°，"
                    f"平移 {float(translation_error):.4f} mm"
                )
                if (
                    float(rotation_error) < SIM_ROTATION_ERROR_TARGET_DEG
                    and float(translation_error)
                    < SIM_TRANSLATION_ERROR_TARGET_MM
                ):
                    print(
                        "     ✓ 已同时达到仿真目标："
                        f"<{SIM_ROTATION_ERROR_TARGET_DEG:.2f}° / "
                        f"<{SIM_TRANSLATION_ERROR_TARGET_MM:.1f} mm"
                    )
        return len(iterations)

    @staticmethod
    def _print_final_result(result_file: Path) -> None:
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            handeye = payload["handeye"]
            diagnostics = payload["diagnostics"]
            simulation = payload.get("simulation", {})
        except (OSError, KeyError, json.JSONDecodeError) as error:
            print(f"最终结果读取失败：{error}")
            return
        print("\n最终标定结果")
        print(f"  文件：{result_file}")
        print("  R_F<-S：")
        for row in handeye["rotation"]:
            print("    [" + ", ".join(f"{float(value): .9f}" for value in row) + "]")
        print(
            "  t_F<-S： ["
            + ", ".join(
                f"{1000.0 * float(value):.4f}" for value in handeye["translation"]
            )
            + "] mm"
        )
        print(
            f"  cost={float(payload.get('cost', 0.0)):.6g}，"
            "data-only rank="
            f"{diagnostics.get('data_only', {}).get('rank', diagnostics.get('rank'))}，"
            "condition="
            f"{float(diagnostics.get('data_only', {}).get('condition_number', diagnostics.get('condition_number', 0.0))):.3e}"
        )
        prior = diagnostics.get("prior_augmented")
        if isinstance(prior, dict):
            print(
                "  观测+形貌先验："
                f"rank={prior.get('rank')}，"
                f"condition={float(prior.get('condition_number', 0.0)):.3e}"
            )
        surface = payload.get("surface", {})
        if surface.get("model") == "shared":
            print(
                "  共享形貌：Legendre"
                f" {surface.get('degree')}阶；"
                f"RMS {1000.0 * float(surface.get('rms_m', 0.0)):.4f} mm；"
                "最大绝对高度 "
                f"{1000.0 * float(surface.get('maximum_abs_m', 0.0)):.4f} mm"
            )
        simulation = simulation if isinstance(simulation, dict) else {}
        if simulation.get("rotation_error_deg") is not None:
            print(
                "  仿真真值误差："
                f"旋转 {float(simulation.get('rotation_error_deg', 0.0)):.4f}°，"
                f"平移 {float(simulation.get('translation_error_mm', 0.0)):.4f} mm"
            )
            if (
                float(simulation.get("rotation_error_deg", float("inf")))
                < SIM_ROTATION_ERROR_TARGET_DEG
                and float(simulation.get("translation_error_mm", float("inf")))
                < SIM_TRANSLATION_ERROR_TARGET_MM
            ):
                print("  目标判定：已同时达到（仅用于当前仿真真值评价）")
        else:
            print("  真值误差：真机模式无仿真真值（N/A）")
        print(
            f"  使用种子 {simulation.get('seed_count', '?')} 个，"
            f"主动 NBV {simulation.get('nbv_count', '?')} 个；"
            f"停止原因：{simulation.get('stop_reason') or '正常完成'}"
        )
        held_out = simulation.get("held_out_validation", {})
        if simulation.get("result_selection") == "latest_committed":
            print(
                "  结果选择：保留最后一次通过双边验证的滚动解；"
                "留出帧仅诊断 "
                f"{float(held_out.get('current_score_mm', float('nan'))):.4f} mm"
            )

    def show_latest_result(self) -> None:
        candidates = sorted(
            RUNS_DIR.glob("*/calibration_result.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            legacy = WORKSPACE / "data/calibration_result_v5.json"
            if legacy.exists():
                candidates = [legacy]
        if not candidates:
            print("尚未找到标定结果。")
            return
        self._print_final_result(candidates[0])

    def run_sphere_validation(self) -> None:
        if self.backend != "real":
            print("精密球验证用于真机独立精度评价，请以 --backend real 启动控制台。")
            return
        print(
            "\n将进入独立精密球验证。该程序只订阅同步轮廓和法兰位姿，"
            "不会向机器人写入运动，也不会用球面数据重新优化手眼结果。"
        )
        completed = subprocess.run(
            [sys.executable, str(WORKSPACE / "scripts/sphere_validation_console.py")],
            cwd=WORKSPACE,
            check=False,
        )
        if completed.returncode == 2:
            print("精密球实验已完成，但结果没有达到配置中的精度阈值。")
        elif completed.returncode not in {0, 130}:
            print(f"精密球验证异常结束，退出码 {completed.returncode}。")

    @staticmethod
    def _node_start_failure(
        node: ManagedNode, title: str = "节点启动失败"
    ) -> None:
        print(f"\n{title}。日志文件：{node.log_file}")
        tail = node.tail()
        if tail:
            print(tail)

    def interactive(self) -> int:
        print("=" * 66)
        environment_name = "真机" if self.backend == "real" else "仿真"
        print(f"        线激光双边角点主动手眼标定 — {environment_name}交互控制台")
        print("=" * 66)
        if not self.verify_environment():
            return 1
        while True:
            print("\n请选择操作：")
            print("  1. 新建标定：自动采集 6 个种子位姿")
            print("  2. 新建标定：人工采集 6 个种子位姿")
            print("  3. 使用已有种子，直接执行主动标定")
            print("  4. 查看最近一次标定结果")
            if self.backend == "real":
                print("  5. 使用精密球独立验证真机标定精度")
            print("  0. 退出")
            choice = input("选择 [1]：").strip() or "1"
            if choice == "1":
                self.run_new("automatic")
            elif choice == "2":
                self.run_new("manual")
            elif choice == "3":
                if self.backend == "real":
                    inspection_only = ask_yes_no(
                        "只初始化并显示平板位姿（不执行任何 NBV 运动）？", default=True
                    )
                    if not inspection_only:
                        self._print_motion_safety()
                        if not ask_yes_no(
                            "将解锁真机自动运动并执行 NBV，确认开始？", default=False
                        ):
                            continue
                        real_motion_armed = self._arm_real_motion()
                        if not real_motion_armed:
                            continue
                        try:
                            self.run_existing()
                        finally:
                            self._disarm_real_motion()
                    else:
                        self.run_existing(initialization_only=True)
                else:
                    self.run_existing()
            elif choice == "4":
                self.show_latest_result()
            elif choice == "5" and self.backend == "real":
                self.run_sphere_validation()
            elif choice == "0":
                return 0
            else:
                maximum = 5 if self.backend == "real" else 4
                print(f"无效选择，请输入 0～{maximum}。")
                continue
            if not ask_yes_no("返回主菜单？", default=False):
                return 0


def progress_bar(current: int, total: int, width: int = 18) -> str:
    current = max(0, min(current, total))
    filled = int(round(width * current / total)) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def ask_yes_no(prompt: str, *, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            value = input(f"{prompt} {suffix}：").strip().lower()
        except EOFError:
            return default
        if not value:
            return default
        if value in {"y", "yes", "是", "好", "确认"}:
            return True
        if value in {"n", "no", "否", "取消"}:
            return False
        print("请输入 y 或 n。")


def ask_preflight_mode() -> str:
    print("\n动态实测预检（可选工程增强）：")
    print("  1. auto：静态余量不足时自动执行（推荐）")
    print("  2. always：每次都执行，较保守但更耗时")
    print("  3. off：跳过独立预检，正式种子仍实时检查和回退")
    while True:
        try:
            value = input("选择 [1]：").strip().lower() or "1"
        except EOFError:
            return "auto"
        aliases = {
            "1": "auto",
            "auto": "auto",
            "2": "always",
            "always": "always",
            "3": "off",
            "off": "off",
        }
        if value in aliases:
            return aliases[value]
        print("请输入 1、2 或 3。")


def ask_integer(
    prompt: str, *, default: int, minimum: int, maximum: int
) -> int:
    while True:
        value = input(f"{prompt} [{default}]：").strip()
        if not value:
            return default
        try:
            parsed = int(value)
        except ValueError:
            print("请输入整数。")
            continue
        if minimum <= parsed <= maximum:
            return parsed
        print(f"请输入 {minimum}～{maximum} 之间的整数。")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gazebo/MoveIt 双边角点主动手眼标定交互控制台"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查仿真环境，不进入交互菜单",
    )
    parser.add_argument(
        "--backend",
        choices=("simulation", "sim", "real"),
        default="simulation",
        help="选择仿真或真机只观察后端",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    console = CalibrationConsole(backend=arguments.backend)

    def request_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGHUP, request_shutdown)
    try:
        if arguments.check:
            return 0 if console.verify_environment() else 1
        return console.interactive()
    except (KeyboardInterrupt, EOFError):
        print("\n收到中断，正在停止本控制台启动的标定节点……")
        return 130
    finally:
        console.cleanup()


if __name__ == "__main__":
    sys.exit(main())
