"""Pair each stationary metric profile with a validated FANUC flange pose."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from std_srvs.srv import Trigger

from fanuc_m20id25_support.fanuc_kinematic import forward_kinematics_urdf
from fanuc_m20id25_support.fanuc_transforms import matrix_to_quat


def stamp_nanoseconds(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass(frozen=True)
class JointSample:
    stamp_ns: int
    joints: np.ndarray
    maximum_speed_rad_s: float
    stable_count: int


class MeasurementSyncNode(Node):
    """Publish an exact-profile-stamp base->flange pose only while stationary."""

    def __init__(self) -> None:
        super().__init__("measurement_sync")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("profile_topic", "/gocator/profile")
        self.declare_parameter(
            "flange_pose_topic", "/calibration/flange_pose"
        )
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("flange_frame", "fanuc_flange")
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
        self.declare_parameter("maximum_stamp_skew_s", 0.20)
        self.declare_parameter("maximum_joint_speed_rad_s", 0.01)
        self.declare_parameter("minimum_stable_samples", 3)
        self.declare_parameter("joint_buffer_size", 200)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.flange_frame = str(self.get_parameter("flange_frame").value)
        self.joint_names = tuple(
            str(value) for value in self.get_parameter("joint_names").value
        )
        if len(self.joint_names) != 6:
            raise ValueError("joint_names must contain six names")
        self.maximum_skew_ns = int(
            float(self.get_parameter("maximum_stamp_skew_s").value) * 1e9
        )
        self.maximum_joint_speed = float(
            self.get_parameter("maximum_joint_speed_rad_s").value
        )
        self.minimum_stable_samples = int(
            self.get_parameter("minimum_stable_samples").value
        )
        if self.maximum_skew_ns <= 0:
            raise ValueError("maximum_stamp_skew_s must be positive")
        if self.minimum_stable_samples < 2:
            raise ValueError("minimum_stable_samples must be at least two")

        buffer_size = int(self.get_parameter("joint_buffer_size").value)
        self.samples: deque[JointSample] = deque(maxlen=max(buffer_size, 10))
        self.publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("flange_pose_topic").value),
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_callback,
            50,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("profile_topic").value),
            self._profile_callback,
            10,
        )
        self.create_service(Trigger, "~/status", self._status_callback)

        self.joint_messages = 0
        self.profile_messages = 0
        self.synchronized_profiles = 0
        self.rejected_no_joint = 0
        self.rejected_skew = 0
        self.rejected_motion = 0
        self.last_skew_ms = None
        self.last_speed_rad_s = None
        self.last_error = "waiting_for_joint_state_and_profile"

    def _joint_callback(self, message: JointState) -> None:
        try:
            joints = np.asarray(
                [
                    message.position[message.name.index(name)]
                    for name in self.joint_names
                ],
                dtype=float,
            )
        except (ValueError, IndexError):
            self.last_error = "normalized joint state is missing expected names"
            return
        if not np.all(np.isfinite(joints)):
            self.last_error = "normalized joint state contains non-finite values"
            return
        stamp_ns = stamp_nanoseconds(message.header.stamp)
        maximum_speed = float("inf")
        stable_count = 0
        if self.samples:
            previous = self.samples[-1]
            dt = (stamp_ns - previous.stamp_ns) / 1e9
            if dt > 1e-6:
                maximum_speed = float(
                    np.max(np.abs(joints - previous.joints)) / dt
                )
            if maximum_speed <= self.maximum_joint_speed:
                stable_count = previous.stable_count + 1
        self.samples.append(
            JointSample(stamp_ns, joints, maximum_speed, stable_count)
        )
        self.joint_messages += 1

    def _profile_callback(self, message: PointCloud2) -> None:
        self.profile_messages += 1
        if not self.samples:
            self.rejected_no_joint += 1
            self.last_error = "profile rejected: no normalized joint sample"
            return
        profile_stamp_ns = stamp_nanoseconds(message.header.stamp)
        sample = min(
            self.samples,
            key=lambda item: abs(item.stamp_ns - profile_stamp_ns),
        )
        skew_ns = abs(sample.stamp_ns - profile_stamp_ns)
        self.last_skew_ms = skew_ns / 1e6
        self.last_speed_rad_s = sample.maximum_speed_rad_s
        if skew_ns > self.maximum_skew_ns:
            self.rejected_skew += 1
            self.last_error = (
                f"profile rejected: joint/profile skew {skew_ns / 1e6:.1f} ms"
            )
            return
        if (
            sample.stable_count < self.minimum_stable_samples
            or sample.maximum_speed_rad_s > self.maximum_joint_speed
        ):
            self.rejected_motion += 1
            self.last_error = "profile rejected: robot is not stably stationary"
            return

        transform = forward_kinematics_urdf(sample.joints)
        quaternion_wxyz, translation = matrix_to_quat(transform)
        output = PoseStamped()
        output.header.stamp = message.header.stamp
        output.header.frame_id = self.base_frame
        output.pose.position.x = float(translation[0])
        output.pose.position.y = float(translation[1])
        output.pose.position.z = float(translation[2])
        output.pose.orientation.x = float(quaternion_wxyz[1])
        output.pose.orientation.y = float(quaternion_wxyz[2])
        output.pose.orientation.z = float(quaternion_wxyz[3])
        output.pose.orientation.w = float(quaternion_wxyz[0])
        self.publisher.publish(output)
        self.synchronized_profiles += 1
        self.last_error = ""

    def _status_callback(self, _request, response):
        payload = {
            "joint_messages": self.joint_messages,
            "profile_messages": self.profile_messages,
            "synchronized_profiles": self.synchronized_profiles,
            "rejected_no_joint": self.rejected_no_joint,
            "rejected_skew": self.rejected_skew,
            "rejected_motion": self.rejected_motion,
            "last_skew_ms": self.last_skew_ms,
            "last_speed_rad_s": self.last_speed_rad_s,
            "last_error": self.last_error,
            "stationary_only": True,
        }
        response.success = self.synchronized_profiles > 0 and not self.last_error
        response.message = json.dumps(payload, ensure_ascii=False)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MeasurementSyncNode()
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
