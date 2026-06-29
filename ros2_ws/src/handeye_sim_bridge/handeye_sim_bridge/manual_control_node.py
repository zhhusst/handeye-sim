#!/usr/bin/env python3
"""
manual_control_node.py — 键盘手动控制机器人 + 实时扫描线显示

按键说明:
  1-6    选择关节 (J1~J6)
  +/-    增大/减小选中关节角度 (步长 1°)
  Shift+ +/-  步长 0.1° (精调)
  Space   切换: 传感器位姿模式 / 点动模式
  R       归零所有关节
  Q/ESC   退出

显示:
  - FANUC 机器人 3D 模型 (通过 joint_state_publisher)
  - 平板 Marker
  - FOV 三角锥
  - 扫描线 + 断点
  - 激光平面与平板的交线
"""

import rclpy
from rclpy.node import Node
import numpy as np
import sys
import os
import select
import threading

sys.path.insert(0, '/workspace/common')

from fov_geometry import (
    compute_fov_plate_scanline, compute_fov_triangle,
    generate_plane, generate_hand_eye_gt,
    build_R_edge, make_transform,
)
from handeye_sim_bridge.bridge_publisher import CalibPublisher
from handeye_sim_bridge.fanuc_kinematic import forward_kinematics, inverse_kinematics
from handeye_sim_bridge.fanuc_types import RobotParam
from sensor_msgs.msg import JointState

JOINT_NAMES = ['J1_joint', 'J2_joint', 'J3_joint',
               'J4_joint', 'J5_joint', 'J6_joint']

JOINT_LIMITS_DEG = [(-185, 185), (-100, 160), (-90, 220),
                    (-200, 200), (-180, 180), (-450, 450)]


class ManualControlNode(Node):
    def __init__(self):
        super().__init__('manual_control_node')

        # 发布器
        self.publisher = CalibPublisher()
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # 场景 (固定, 每次随机生成)
        rng = np.random.default_rng(42)
        C, n_B, u_B, v_B, w, h = generate_plane(rng)
        X_gt = generate_hand_eye_gt(rng)
        R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

        self.scene = {
            'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
            'w': w, 'h': h,
            'R_he': R_he, 't_he': t_he,
        }
        self.publisher.set_scene(C, n_B, u_B, v_B, w, h)

        # 关节状态 — 初始位姿 (让传感器大致指向平板)
        self.joints = np.deg2rad(np.array([0.0, 30.0, -60.0, 0.0, -30.0, 0.0]))
        self.selected_joint = 0  # 当前选中关节索引

        self.get_logger().info("=" * 60)
        self.get_logger().info("手动控制 — 按键:")
        self.get_logger().info("  1-6: 选中关节 J1~J6")
        self.get_logger().info("  +/-: 增大/减小关节角 (步长 1°)")
        self.get_logger().info("  Shift+ +/-: 精调 0.1°")
        self.get_logger().info("  R: 归零     Q: 退出")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"场景: C={C}")
        self.get_logger().info(f"手眼 GT: t_he={t_he}")

        # 先发布一次场景
        self.publish_frame()

        # 键盘监听线程
        self.running = True
        self.key_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.key_thread.start()

    def publish_frame(self):
        """计算当前关节角的扫描线并发布"""
        joints = self.joints.copy()
        T_B_H = forward_kinematics(joints)
        R_i = T_B_H[:3, :3]
        t_i = T_B_H[:3, 3]

        R_he = self.scene['R_he']
        t_he = self.scene['t_he']
        C = self.scene['C']
        n_B = self.scene['n_B']
        u_B = self.scene['u_B']
        v_B = self.scene['v_B']
        w = self.scene['w']
        h = self.scene['h']

        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he

        stamp = self.get_clock().now().to_msg()

        # 场景 (每帧都发)
        self.publisher.publish_scene_markers(stamp)

        # TF
        self.publisher.publish_plate_tf(stamp, C, n_B, u_B, v_B)
        self.publisher.publish_sensor_tf(stamp, R_BS, t_BS)

        # 扫描线
        sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, w, h)
        scan_pts_B = sl['scan_pts_B']
        endpoints_B_base = [(e, R_BS @ p + t_BS) for e, p in sl['endpoints_S']]
        P0 = sl.get('line_origin_B')
        line_dir = sl.get('line_dir')

        self.publisher.publish_frame_markers(stamp, R_BS, t_BS, scan_pts_B,
                                              endpoints_B_base, P0, line_dir)

        # 关节角
        js = JointState()
        js.header.stamp = stamp
        js.name = JOINT_NAMES
        js.position = self.joints.tolist()
        self.joint_pub.publish(js)

        # 状态栏
        n_ep = len(sl['endpoints_S'])
        eps = [e for e, _ in sl['endpoints_S']]
        dist_str = ""
        if n_ep >= 2:
            pts = [pt for _, pt in sl['endpoints_S']]
            dist = np.linalg.norm(pts[0] - pts[1])
            dist_str = f" 间距={dist*1000:.1f}mm" if dist > 0 else ""

        joint_str = ' '.join(
            f"[{np.rad2deg(self.joints[i]):6.1f}°]" if i == self.selected_joint
            else f" {np.rad2deg(self.joints[i]):6.1f}° "
            for i in range(6)
        )
        ep_str = f"断点={'+'.join(eps)}{dist_str}" if n_ep >= 2 else f"断点={'/'.join(eps)}" if n_ep > 0 else "无断点"
        print(f"\r  J1~J6: {joint_str}  |  {ep_str}   ", end='', flush=True)

    def keyboard_listener(self):
        """键盘监听 — 直接用 /dev/tty 读终端 (避免 ROS2 launch 的 stdin 限制)"""
        try:
            fd = os.open('/dev/tty', os.O_RDONLY)
        except OSError:
            fd = sys.stdin.fileno()
        while self.running:
            if select.select([fd], [], [], 0.1)[0]:
                key = os.read(fd, 1).decode('utf-8', errors='replace')
                self.handle_key(key)
            else:
                self.publish_frame()
        if fd != sys.stdin.fileno():
            os.close(fd)

    def handle_key(self, key):
        if key in '123456':
            self.selected_joint = int(key) - 1
            self.get_logger().info(f"\n  选中 J{self.selected_joint + 1}")
        elif key == '+':
            step = np.deg2rad(1.0)
            self.joints[self.selected_joint] = np.clip(
                self.joints[self.selected_joint] + step,
                np.deg2rad(JOINT_LIMITS_DEG[self.selected_joint][0]),
                np.deg2rad(JOINT_LIMITS_DEG[self.selected_joint][1]))
            self.publish_frame()
        elif key == '-':
            step = np.deg2rad(1.0)
            self.joints[self.selected_joint] = np.clip(
                self.joints[self.selected_joint] - step,
                np.deg2rad(JOINT_LIMITS_DEG[self.selected_joint][0]),
                np.deg2rad(JOINT_LIMITS_DEG[self.selected_joint][1]))
            self.publish_frame()
        elif key == 'r' or key == 'R':
            self.joints = np.zeros(6)
            self.get_logger().info("\n  归零")
            self.publish_frame()
        elif key == 'q' or key == 'Q' or key == '\x1b':
            self.get_logger().info("\n  退出")
            self.running = False
            rclpy.shutdown()

    def __del__(self):
        self.running = False


def main(args=None):
    rclpy.init(args=args)
    node = ManualControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
