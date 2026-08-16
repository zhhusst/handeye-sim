"""Read FANUC joints over EtherNet/IP without exposing motion commands."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from fanuc_m20id25_support.fanuc_kinematic import forward_kinematics_urdf
from fanuc_m20id25_support.fanuc_transforms import pose_to_matrix_fanuc

from .fanuc_eip import (
    FanucCIPSession,
    return_cartesian_current_position,
    return_joint_current_position,
)


class FanucJointStateNode(Node):
    """Publish raw controller joints and optionally validated URDF joints."""

    def __init__(self) -> None:
        super().__init__("fanuc_joint_state")
        self.declare_parameter("robot_ip", "192.168.0.10")
        self.declare_parameter("poll_rate_hz", 10.0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("j23_factor", 0.0)
        self.declare_parameter("j23_validated", False)
        self.declare_parameter("raw_topic", "/fanuc/joint_states_raw")
        self.declare_parameter("normalized_topic", "/joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("validation_utool", 0)
        self.declare_parameter("validation_uframe", 0)
        self.declare_parameter(
            "validation_controller_origin_in_base_mm", [0.0, 0.0, 0.0]
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

        self.robot_ip = str(self.get_parameter("robot_ip").value)
        self.poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        self.j23_factor = float(self.get_parameter("j23_factor").value)
        self.j23_validated = bool(
            self.get_parameter("j23_validated").value
        )
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.validation_utool = int(
            self.get_parameter("validation_utool").value
        )
        self.validation_uframe = int(
            self.get_parameter("validation_uframe").value
        )
        self.validation_controller_origin_in_base_mm = np.asarray(
            self.get_parameter(
                "validation_controller_origin_in_base_mm"
            ).value,
            dtype=float,
        )
        self.joint_names = tuple(
            str(value) for value in self.get_parameter("joint_names").value
        )
        if self.poll_rate_hz <= 0.0:
            raise ValueError("poll_rate_hz must be positive")
        if len(self.joint_names) != 6:
            raise ValueError("joint_names must contain six names")
        if self.validation_controller_origin_in_base_mm.shape != (3,):
            raise ValueError(
                "validation_controller_origin_in_base_mm must have 3 values"
            )
        if self.j23_factor not in {-1.0, 0.0, 1.0}:
            raise ValueError("j23_factor must be -1, 0, or 1")

        self.raw_publisher = self.create_publisher(
            JointState, str(self.get_parameter("raw_topic").value), 10
        )
        self.normalized_publisher = self.create_publisher(
            JointState,
            str(self.get_parameter("normalized_topic").value),
            10,
        )
        self.create_service(Trigger, "~/status", self._status_callback)
        self.create_service(
            Trigger, "~/kinematic_check", self._kinematic_check_callback
        )

        publish_rate = float(self.get_parameter("publish_rate_hz").value)
        if publish_rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        self.create_timer(1.0 / publish_rate, self._publish_latest)

        self._lock = threading.Lock()
        self._latest_joints_deg: np.ndarray | None = None
        self._latest_stamp = None
        self._sample_sequence = 0
        self._published_sequence = -1
        self._successful_reads = 0
        self._failed_reads = 0
        self._last_error = "not_connected"
        self._running = True
        self._session = FanucCIPSession(self.robot_ip)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        validation_state = "validated" if self.j23_validated else "blocked"
        self.get_logger().warning(
            "FANUC backend is observe-only; normalized /joint_states is "
            f"{validation_state} by j23_validated={self.j23_validated}"
        )

    def _poll_loop(self) -> None:
        period = 1.0 / self.poll_rate_hz
        while self._running and rclpy.ok():
            started = time.monotonic()
            try:
                if self._session._driver is None:
                    self._session.open()
                joints = np.asarray(
                    return_joint_current_position(self._session), dtype=float
                )
                if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                    raise RuntimeError(f"invalid joint sample: {joints}")
                stamp = self.get_clock().now().to_msg()
                with self._lock:
                    self._latest_joints_deg = joints
                    self._latest_stamp = stamp
                    self._sample_sequence += 1
                    self._successful_reads += 1
                    self._last_error = ""
            except Exception as error:  # hardware retry loop
                with self._lock:
                    self._failed_reads += 1
                    self._last_error = str(error)
                try:
                    self._session.close()
                except Exception:
                    pass
            elapsed = time.monotonic() - started
            time.sleep(max(period - elapsed, 0.001))

    @staticmethod
    def _joint_message(names, positions_rad, stamp, frame_id) -> JointState:
        message = JointState()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.name = list(names)
        message.position = [float(value) for value in positions_rad]
        return message

    def _publish_latest(self) -> None:
        with self._lock:
            if (
                self._latest_joints_deg is None
                or self._latest_stamp is None
                or self._sample_sequence == self._published_sequence
            ):
                return
            joints_deg = self._latest_joints_deg.copy()
            stamp = self._latest_stamp
            sequence = self._sample_sequence
        raw_names = [f"J{index}_controller" for index in range(1, 7)]
        self.raw_publisher.publish(
            self._joint_message(
                raw_names, np.deg2rad(joints_deg), stamp, self.base_frame
            )
        )
        if self.j23_validated:
            normalized_deg = joints_deg.copy()
            normalized_deg[2] += self.j23_factor * normalized_deg[1]
            self.normalized_publisher.publish(
                self._joint_message(
                    self.joint_names,
                    np.deg2rad(normalized_deg),
                    stamp,
                    self.base_frame,
                )
            )
        self._published_sequence = sequence

    def _status_callback(self, _request, response):
        with self._lock:
            payload = {
                "connected": self._latest_joints_deg is not None,
                "successful_reads": self._successful_reads,
                "failed_reads": self._failed_reads,
                "last_error": self._last_error,
                "j23_validated": self.j23_validated,
                "j23_factor": self.j23_factor,
                "normalized_output_enabled": self.j23_validated,
                "motion_writes_enabled": False,
            }
        response.success = bool(payload["connected"])
        response.message = json.dumps(payload, ensure_ascii=False)
        return response

    def _kinematic_check_callback(self, _request, response):
        """Compare the three J2/J3 conventions against read-only CURPOS XYZ."""
        with self._lock:
            joints_deg = (
                None
                if self._latest_joints_deg is None
                else self._latest_joints_deg.copy()
            )
        if joints_deg is None:
            response.success = False
            response.message = json.dumps(
                {"error": "no joint sample is available"}, ensure_ascii=False
            )
            return response
        try:
            controller_pose = return_cartesian_current_position(self._session)
        except Exception as error:
            response.success = False
            response.message = json.dumps(
                {"error": f"CURPOS read failed: {error}"}, ensure_ascii=False
            )
            return response

        controller_pose_matrix = pose_to_matrix_fanuc(
            [
                controller_pose["x_mm"],
                controller_pose["y_mm"],
                controller_pose["z_mm"],
                controller_pose["w_deg"],
                controller_pose["p_deg"],
                controller_pose["r_deg"],
            ],
        )
        base_from_controller = np.eye(4)
        base_from_controller[:3, 3] = (
            self.validation_controller_origin_in_base_mm
        )
        controller_flange = base_from_controller @ controller_pose_matrix
        candidates = {}
        for factor in (0.0, 1.0, -1.0):
            urdf_joints_deg = joints_deg.copy()
            urdf_joints_deg[2] += factor * urdf_joints_deg[1]
            transform = forward_kinematics_urdf(np.deg2rad(urdf_joints_deg))
            flange_xyz = 1000.0 * transform[:3, 3]
            rotation_delta = (
                transform[:3, :3].T @ controller_flange[:3, :3]
            )
            rotation_error_deg = float(
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            (np.trace(rotation_delta) - 1.0) / 2.0,
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            candidates[str(int(factor))] = {
                "urdf_j3_deg": float(urdf_joints_deg[2]),
                "flange_xyz_mm": [float(value) for value in flange_xyz],
                "position_error_mm": float(
                    np.linalg.norm(
                        flange_xyz - controller_flange[:3, 3]
                    )
                ),
                "rotation_error_deg": rotation_error_deg,
            }

        comparable = (
            controller_pose["utool"] == self.validation_utool
            and controller_pose["uframe"] == self.validation_uframe
        )
        recommended_factor = min(
            candidates,
            key=lambda key: (
                candidates[key]["position_error_mm"],
                candidates[key]["rotation_error_deg"],
            ),
        )
        payload = {
            "raw_joints_deg": [float(value) for value in joints_deg],
            "controller_curpos": controller_pose,
            "candidates_by_j23_factor": candidates,
            "validation_setup": {
                "expected_utool": self.validation_utool,
                "expected_uframe": self.validation_uframe,
                "controller_origin_in_base_mm": [
                    float(value)
                    for value in self.validation_controller_origin_in_base_mm
                ],
            },
            "position_errors_comparable": comparable,
            "recommended_j23_factor": (
                int(recommended_factor) if comparable else None
            ),
            "note": (
                "smallest position error identifies the convention"
                if comparable
                else "select the configured validation UTOOL/UFRAME before "
                "comparing; no robot motion is required"
            ),
            "motion_writes_enabled": False,
        }
        response.success = comparable
        response.message = json.dumps(payload, ensure_ascii=False)
        return response

    def destroy_node(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self._session.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FanucJointStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
