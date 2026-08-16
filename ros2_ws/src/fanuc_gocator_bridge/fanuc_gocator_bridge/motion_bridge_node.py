"""Safety-gated FollowJointTrajectory bridge for PC_TRACK_ALL STEP mode."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from fanuc_m20id25_support.fanuc_kinematic import forward_kinematics_urdf
from fanuc_m20id25_support.fanuc_transforms import pose_to_matrix_fanuc

from .eip_io_thread import EIPIOThread
from .fanuc_eip import FanucCIPSession
from .motion_safety import plan_small_linear_move, rotation_distance_rad
from .reg_sender import PcTrackAllStepSender


class FanucMotionBridge(Node):
    """Expose the existing TP STEP protocol through the standard ROS action."""

    MODES = {"plan_only", "step_confirm", "automatic"}

    def __init__(self) -> None:
        super().__init__("fanuc_motion_bridge")
        self.declare_parameter("robot_ip", "192.168.0.10")
        self.declare_parameter("mode", "plan_only")
        self.declare_parameter("motion_writes_enabled", False)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter(
            "action_name",
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self.declare_parameter(
            "joint_names",
            [
                "J1_joint",
                "J2_joint",
                "J3_joint",
                "J4_joint",
                "J5_joint",
                "J6_joint",
            ],
        )
        self.declare_parameter("utool", 1)
        self.declare_parameter("uframe", 1)
        self.declare_parameter("pr_utool_wire_value", 0)
        self.declare_parameter("pr_uframe_wire_value", 0)
        self.declare_parameter(
            "controller_origin_in_base_mm", [0.0, 0.0, 425.0]
        )
        self.declare_parameter("speed_mm_s", 5.0)
        self.declare_parameter("minimum_speed_mm_s", 1)
        self.declare_parameter("maximum_speed_mm_s", 20)
        self.declare_parameter("maximum_joint_step_deg", 6.0)
        self.declare_parameter("maximum_joint_distance_rad", 0.20)
        self.declare_parameter("maximum_translation_mm", 80.0)
        self.declare_parameter("maximum_rotation_deg", 10.0)
        self.declare_parameter("minimum_joint_margin_deg", 3.0)
        self.declare_parameter("cartesian_path_samples", 5)
        self.declare_parameter("maximum_joint_state_age_s", 0.5)
        self.declare_parameter("maximum_joint_speed_rad_s", 0.01)
        self.declare_parameter("arrival_tolerance_deg", 0.20)
        self.declare_parameter("approval_timeout_s", 120.0)
        self.declare_parameter("motion_timeout_s", 90.0)
        self.declare_parameter("validation_position_tolerance_mm", 2.0)
        self.declare_parameter("validation_rotation_tolerance_deg", 1.0)

        self.robot_ip = str(self.get_parameter("robot_ip").value)
        self.mode = str(self.get_parameter("mode").value).strip().lower()
        self.motion_writes_enabled = bool(
            self.get_parameter("motion_writes_enabled").value
        )
        self.joint_names = tuple(
            str(value) for value in self.get_parameter("joint_names").value
        )
        self.utool = int(self.get_parameter("utool").value)
        self.uframe = int(self.get_parameter("uframe").value)
        self.pr_utool_wire_value = int(
            self.get_parameter("pr_utool_wire_value").value
        )
        self.pr_uframe_wire_value = int(
            self.get_parameter("pr_uframe_wire_value").value
        )
        self.controller_origin_in_base_mm = np.asarray(
            self.get_parameter("controller_origin_in_base_mm").value,
            dtype=float,
        )
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if self.mode == "plan_only" and self.motion_writes_enabled:
            raise ValueError("plan_only cannot have motion_writes_enabled=true")
        if len(self.joint_names) != 6:
            raise ValueError("joint_names must contain six names")
        if self.controller_origin_in_base_mm.shape != (3,):
            raise ValueError("controller_origin_in_base_mm needs three values")

        self._callback_group = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self._approval = threading.Event()
        self._latest_joints = None
        self._latest_joint_wall_time = 0.0
        self._latest_speed = float("inf")
        self._armed = False
        self._active_goal = False
        self._triggered = False
        self._state = "PLAN_ONLY" if self.mode == "plan_only" else "DISARMED"
        self._last_error = ""
        self._last_plan = None
        self._last_protocol_status = None

        self._session = None
        self._eip_io = None
        self._sender = None
        if self.motion_writes_enabled:
            self._session = FanucCIPSession(self.robot_ip)
            self._eip_io = EIPIOThread(self._session)
            self._eip_io.start()
            self._sender = PcTrackAllStepSender(
                self._eip_io,
                min_speed_mm_s=int(
                    self.get_parameter("minimum_speed_mm_s").value
                ),
                max_speed_mm_s=int(
                    self.get_parameter("maximum_speed_mm_s").value
                ),
            )

        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_callback,
            20,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger, "~/status", self._status_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger, "~/arm", self._arm_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger, "~/disarm", self._disarm_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger, "~/approve", self._approve_callback,
            callback_group=self._callback_group,
        )
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("action_name").value),
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self.get_logger().warning(
            "PC_TRACK_ALL bridge started: "
            f"mode={self.mode}, writes={self.motion_writes_enabled}, "
            f"UF/UT={self.uframe}/{self.utool}; it starts DISARMED"
        )

    def _joint_callback(self, message: JointState) -> None:
        try:
            index = {name: i for i, name in enumerate(message.name)}
            joints = np.asarray(
                [message.position[index[name]] for name in self.joint_names],
                dtype=float,
            )
        except (KeyError, IndexError, ValueError):
            return
        now = time.monotonic()
        with self._lock:
            if self._latest_joints is not None and now > self._latest_joint_wall_time:
                self._latest_speed = float(
                    np.max(np.abs(joints - self._latest_joints))
                    / (now - self._latest_joint_wall_time)
                )
            self._latest_joints = joints
            self._latest_joint_wall_time = now

    def _fresh_stationary_joints(self) -> np.ndarray:
        with self._lock:
            joints = (
                None if self._latest_joints is None else self._latest_joints.copy()
            )
            age = time.monotonic() - self._latest_joint_wall_time
            speed = self._latest_speed
        if joints is None:
            raise RuntimeError("no normalized /joint_states sample")
        maximum_age = float(
            self.get_parameter("maximum_joint_state_age_s").value
        )
        if age > maximum_age:
            raise RuntimeError(
                f"joint state is stale ({age:.3f}s > {maximum_age:.3f}s)"
            )
        maximum_speed = float(
            self.get_parameter("maximum_joint_speed_rad_s").value
        )
        if speed > maximum_speed:
            raise RuntimeError(
                f"robot is moving ({speed:.5f}rad/s > {maximum_speed:.5f}rad/s)"
            )
        return joints

    def _ensure_eip(self) -> None:
        if not self.motion_writes_enabled or self._session is None:
            raise RuntimeError("motion writes are disabled")
        if self._session._driver is None:
            self._session.open()

    def _controller_setup_check(self, joints: np.ndarray) -> dict:
        self._ensure_eip()
        protocol = self._sender.assert_ready()
        current = self._eip_io.call("get_curpos", timeout=2.0)
        if int(current[0]) != self.utool or int(current[1]) != self.uframe:
            raise RuntimeError(
                "controller CURPOS does not use the validated pair: "
                f"expected UF/UT={self.uframe}/{self.utool}, "
                f"got {int(current[1])}/{int(current[0])}"
            )
        controller_from_flange = pose_to_matrix_fanuc(current[2:8])
        controller_from_flange[:3, 3] /= 1000.0
        base_from_controller = np.eye(4)
        base_from_controller[:3, 3] = self.controller_origin_in_base_mm / 1000.0
        measured = base_from_controller @ controller_from_flange
        expected = forward_kinematics_urdf(joints)
        position_error_mm = float(
            1000.0 * np.linalg.norm(measured[:3, 3] - expected[:3, 3])
        )
        rotation_error_deg = float(
            np.rad2deg(rotation_distance_rad(measured, expected))
        )
        if position_error_mm > float(
            self.get_parameter("validation_position_tolerance_mm").value
        ) or rotation_error_deg > float(
            self.get_parameter("validation_rotation_tolerance_deg").value
        ):
            raise RuntimeError(
                "CURPOS/FK consistency check failed: "
                f"{position_error_mm:.3f}mm, {rotation_error_deg:.3f}deg"
            )
        with self._lock:
            self._last_protocol_status = protocol
        return {
            "template": current,
            "protocol": protocol,
            "position_error_mm": position_error_mm,
            "rotation_error_deg": rotation_error_deg,
        }

    def _plan(self, target: np.ndarray):
        current = self._fresh_stationary_joints()
        plan = plan_small_linear_move(
            current,
            target,
            controller_origin_in_base_mm=self.controller_origin_in_base_mm,
            maximum_joint_step_deg=float(
                self.get_parameter("maximum_joint_step_deg").value
            ),
            maximum_joint_distance_rad=float(
                self.get_parameter("maximum_joint_distance_rad").value
            ),
            maximum_translation_mm=float(
                self.get_parameter("maximum_translation_mm").value
            ),
            maximum_rotation_deg=float(
                self.get_parameter("maximum_rotation_deg").value
            ),
            minimum_joint_margin_deg=float(
                self.get_parameter("minimum_joint_margin_deg").value
            ),
            cartesian_path_samples=int(
                self.get_parameter("cartesian_path_samples").value
            ),
            check_cartesian_ik=True,
        )
        with self._lock:
            self._last_plan = plan.as_dict()
        return current, plan

    def _goal_callback(self, goal_request):
        if len(goal_request.trajectory.points) != 1:
            self._last_error = "only one-point trajectories are allowed"
            return GoalResponse.REJECT
        if tuple(goal_request.trajectory.joint_names) != self.joint_names:
            self._last_error = "trajectory joint names/order do not match"
            return GoalResponse.REJECT
        if len(goal_request.trajectory.points[0].positions) != 6:
            self._last_error = "trajectory target must contain six joints"
            return GoalResponse.REJECT
        with self._lock:
            if self._active_goal:
                self._last_error = "another motion goal is active"
                return GoalResponse.REJECT
            self._active_goal = True
            self._triggered = False
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle):
        with self._lock:
            if self._triggered:
                self._last_error = (
                    "TP move is already triggered; cancel it with the FANUC "
                    "HOLD/E-stop controls"
                )
                return CancelResponse.REJECT
        self._approval.set()
        return CancelResponse.ACCEPT

    @staticmethod
    def _result(error_code: int, text: str):
        result = FollowJointTrajectory.Result()
        result.error_code = int(error_code)
        result.error_string = str(text)
        return result

    def _execute_callback(self, goal_handle):
        try:
            target = np.asarray(
                goal_handle.request.trajectory.points[0].positions,
                dtype=float,
            )
            with self._lock:
                self._state = "PLANNING"
                self._last_error = ""
            _, plan = self._plan(target)
            if self.mode == "plan_only":
                goal_handle.abort()
                with self._lock:
                    self._state = "PLAN_ONLY"
                return self._result(
                    FollowJointTrajectory.Result.INVALID_GOAL,
                    "target passed local checks; plan_only forbids execution",
                )
            with self._lock:
                if not self._armed:
                    raise RuntimeError("motion bridge is disarmed")
            if plan.joint_distance_rad <= 1.0e-7:
                goal_handle.succeed()
                with self._lock:
                    self._state = "ARMED"
                return self._result(
                    FollowJointTrajectory.Result.SUCCESSFUL,
                    "zero-distance target accepted without writing PR[10]",
                )

            self._approval.clear()
            if self.mode == "step_confirm":
                with self._lock:
                    self._state = "WAIT_APPROVAL"
                deadline = time.monotonic() + float(
                    self.get_parameter("approval_timeout_s").value
                )
                while not self._approval.wait(timeout=0.05):
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        return self._result(
                            FollowJointTrajectory.Result.SUCCESSFUL,
                            "goal canceled before TP trigger",
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("operator approval timed out")
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._result(
                    FollowJointTrajectory.Result.SUCCESSFUL,
                    "goal canceled before TP trigger",
                )

            # Recheck current state and controller protocol immediately before
            # the only motion-triggering register write.
            current, plan = self._plan(target)
            setup = self._controller_setup_check(current)
            with self._lock:
                if not self._armed:
                    raise RuntimeError("motion bridge was disarmed before trigger")
                self._state = "EXECUTING"
                self._triggered = True
            execution = self._sender.execute(
                plan.target_pose_xyz_wpr,
                template=setup["template"],
                utool=self.pr_utool_wire_value,
                uframe=self.pr_uframe_wire_value,
                speed_mm_s=float(self.get_parameter("speed_mm_s").value),
                timeout_s=float(self.get_parameter("motion_timeout_s").value),
            )

            arrival_deadline = time.monotonic() + 5.0
            arrival_error_deg = float("inf")
            while time.monotonic() < arrival_deadline:
                try:
                    actual = self._fresh_stationary_joints()
                except RuntimeError:
                    time.sleep(0.05)
                    continue
                arrival_error_deg = float(
                    np.max(np.abs(np.rad2deg(actual - target)))
                )
                if arrival_error_deg <= float(
                    self.get_parameter("arrival_tolerance_deg").value
                ):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(
                    f"TP reported done but joint error is {arrival_error_deg:.3f}deg"
                )

            goal_handle.succeed()
            with self._lock:
                self._state = "ARMED"
                self._last_protocol_status = execution
            return self._result(
                FollowJointTrajectory.Result.SUCCESSFUL,
                f"PC_TRACK_ALL STEP completed; joint error={arrival_error_deg:.3f}deg",
            )
        except Exception as error:
            with self._lock:
                self._last_error = str(error)
                self._state = "FAULT"
                self._armed = False
            goal_handle.abort()
            self.get_logger().error(str(error))
            return self._result(
                FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                str(error),
            )
        finally:
            with self._lock:
                self._active_goal = False
                self._triggered = False
            self._approval.clear()

    def _arm_callback(self, _request, response):
        try:
            if self.mode == "plan_only" or not self.motion_writes_enabled:
                raise RuntimeError("bridge is plan_only; restart with an enabled motion mode")
            joints = self._fresh_stationary_joints()
            setup = self._controller_setup_check(joints)
            with self._lock:
                if self._active_goal:
                    raise RuntimeError("cannot arm while a goal is active")
                self._armed = True
                self._state = "ARMED"
                self._last_error = ""
            response.success = True
            response.message = json.dumps(
                {
                    "armed": True,
                    "mode": self.mode,
                    "UF": self.uframe,
                    "UT": self.utool,
                    "protocol": setup["protocol"],
                    "curpos_fk_position_error_mm": setup["position_error_mm"],
                    "curpos_fk_rotation_error_deg": setup["rotation_error_deg"],
                },
                ensure_ascii=False,
            )
        except Exception as error:
            with self._lock:
                self._armed = False
                self._state = "FAULT"
                self._last_error = str(error)
            response.success = False
            response.message = str(error)
        return response

    def _disarm_callback(self, _request, response):
        with self._lock:
            self._armed = False
            if self._state != "EXECUTING":
                self._state = "DISARMED"
            triggered = self._triggered
        self._approval.set()
        response.success = not triggered
        response.message = (
            "software gate disarmed"
            if not triggered
            else "software gate disarmed, but an already-triggered TP move must be stopped on FANUC"
        )
        return response

    def _approve_callback(self, _request, response):
        with self._lock:
            allowed = (
                self.mode == "step_confirm"
                and self._armed
                and self._state == "WAIT_APPROVAL"
                and self._active_goal
            )
        response.success = allowed
        response.message = (
            "pending move approved" if allowed else "no approvable pending move"
        )
        if allowed:
            self._approval.set()
        return response

    def _status_callback(self, _request, response):
        with self._lock:
            age = (
                None
                if self._latest_joints is None
                else time.monotonic() - self._latest_joint_wall_time
            )
            payload = {
                "mode": self.mode,
                "motion_writes_enabled": self.motion_writes_enabled,
                "armed": self._armed,
                "state": self._state,
                "active_goal": self._active_goal,
                "triggered": self._triggered,
                "UF": self.uframe,
                "UT": self.utool,
                "PR_wire_UF": self.pr_uframe_wire_value,
                "PR_wire_UT": self.pr_utool_wire_value,
                "joint_state_age_s": age,
                "joint_speed_rad_s": (
                    None if not np.isfinite(self._latest_speed) else self._latest_speed
                ),
                "last_plan": self._last_plan,
                "protocol": self._last_protocol_status,
                "last_error": self._last_error,
                "warning": (
                    "after R[110]=1, only FANUC HOLD/E-stop can interrupt L motion"
                ),
            }
        response.success = bool(
            self.mode == "plan_only" or (self.motion_writes_enabled and not self._last_error)
        )
        response.message = json.dumps(payload, ensure_ascii=False)
        return response

    def destroy_node(self):
        with self._lock:
            self._armed = False
        self._approval.set()
        self._action_server.destroy()
        if self._eip_io is not None:
            self._eip_io.stop()
            self._eip_io.join(timeout=2.0)
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FanucMotionBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
