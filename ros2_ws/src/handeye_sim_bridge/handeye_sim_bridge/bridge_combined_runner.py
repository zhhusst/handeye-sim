#!/usr/bin/env python3
"""
bridge_combined_runner.py — 手眼标定仿真主节点

流程:
  1. 生成场景 (手眼真值 + 平板参数)
  2. 生成轨迹 (从角点向板中心扫描)
  3. 对每帧计算 IK → 关节角
  4. 逐帧发布 TF + Marker + JointState → RViz 实时显示
"""

import rclpy
from rclpy.node import Node
import numpy as np
import sys
import os

# common 目录
_COMMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
    '..', '..', '..', '..', '..', 'common'))
if not os.path.isdir(_COMMON_DIR):
    _COMMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
        '..', '..', '..', '..', 'common'))
if not os.path.isdir(_COMMON_DIR):
    _COMMON_DIR = '/workspace/common'
sys.path.insert(0, _COMMON_DIR)

from fov_geometry import (
    compute_fov_plate_scanline, compute_fov_triangle,
    generate_hand_eye_gt, generate_plane,
    generate_smooth_trajectory, collect_frames,
    build_R_edge, make_transform,
)
from handeye_sim_bridge.bridge_publisher import CalibPublisher
from handeye_sim_bridge.fanuc_kinematic import inverse_kinematics, forward_kinematics
from sensor_msgs.msg import JointState


# URDF 关节名 (FANUC M-20iD/25)
JOINT_NAMES = [
    'J1_joint', 'J2_joint', 'J3_joint',
    'J4_joint', 'J5_joint', 'J6_joint',
]


class CalibratioSimRunner(Node):
    """标定仿真主控制器"""

    def __init__(self):
        super().__init__('calib_sim_runner')

        # 参数
        self.declare_parameter('n_frames', 40)
        self.declare_parameter('frame_rate', 5.0)
        self.declare_parameter('half_fov_deg', 15.0)

        n_frames = self.get_parameter('n_frames').value
        frame_rate = self.get_parameter('frame_rate').value

        # 创建发布器
        self.publisher = CalibPublisher()
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # 生成场景
        rng = np.random.default_rng(42)
        C, n_B, u_B, v_B, w, h = generate_plane(rng)
        X_gt = generate_hand_eye_gt(rng)
        R_he = X_gt[:3, :3]
        t_he = X_gt[:3, 3]

        scene = {
            'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
            'w': w, 'h': h,
            'R_he': R_he, 't_he': t_he,
        }
        self.scene = scene
        self.publisher.set_scene(C, n_B, u_B, v_B, w, h)

        self.get_logger().info(f"场景: C={C}, n_B={n_B}")
        self.get_logger().info(f"手眼真值: R_he(ax={np.rad2deg(np.arccos((np.trace(R_he)-1)/2)):.1f}°), t_he={t_he}")
        self.get_logger().info(f"板尺寸: {w*1000:.0f}x{h*1000:.0f}mm")

        # 生成轨迹
        self.get_logger().info("生成采集轨迹...")
        poses = generate_smooth_trajectory(scene, n_frames=n_frames)
        if len(poses) == 0:
            self.get_logger().error("无法生成有效轨迹!")
            sys.exit(1)
        self.get_logger().info(f"生成 {len(poses)} 帧位姿")

        # 采集帧
        frames = collect_frames(scene, poses)
        n_valid = sum(1 for f in frames if f['valid'])
        self.get_logger().info(f"有效帧 (两边可见): {n_valid}/{len(frames)}")

        self.poses = poses
        self.frames = frames

        # IK: 将法兰位姿转为关节角
        self.get_logger().info("计算 IK...")
        joint_trajs = []
        ik_success = 0
        for k, (R_i, t_i) in enumerate(poses):
            T_B_H = make_transform(R_i, t_i)
            sols = inverse_kinematics(T_B_H)
            if len(sols) == 0:
                # IK 失败, 用前一帧
                if joint_trajs:
                    joint_trajs.append(joint_trajs[-1].copy())
                else:
                    joint_trajs.append(np.zeros(6))
                continue
            # 选最近邻解 (与前一帧最接近的)
            best = sols[0]
            if joint_trajs:
                prev = joint_trajs[-1]
                diffs = [np.sum(np.abs(s - prev)) for s in sols]
                best = sols[np.argmin(diffs)]
            joint_trajs.append(best)
            ik_success += 1

        self.joint_trajs = joint_trajs
        self.get_logger().info(f"IK 成功: {ik_success}/{len(poses)}")

        # 状态
        self.current_frame = 0
        self.paused = False
        self.last_log_frame = -1

        # 定时器
        self.timer = self.create_timer(1.0 / frame_rate, self.timer_callback)

        self.get_logger().info("=" * 60)
        self.get_logger().info("标定仿真已启动! (含机器人模型)")
        self.get_logger().info("=" * 60)

    def timer_callback(self):
        if self.paused or len(self.poses) == 0:
            return

        idx = self.current_frame % len(self.poses)
        R_i, t_i = self.poses[idx]
        frame_data = self.frames[idx]

        R_he = self.scene['R_he']
        t_he = self.scene['t_he']
        C = self.scene['C']
        n_B = self.scene['n_B']
        u_B = self.scene['u_B']
        v_B = self.scene['v_B']

        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he

        # 端点
        endpoints_B = frame_data.get('endpoints_S', [])
        scan_pts_S = frame_data.get('scan_pts_S', np.zeros((0, 3)))
        endpoints_B_base = [(e, R_BS @ p + t_BS) for e, p in endpoints_B]
        scan_pts_B = np.array([R_BS @ p + t_BS for p in scan_pts_S]) if len(scan_pts_S) > 0 else np.zeros((0, 3))

        sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B,
                                         self.scene['w'], self.scene['h'])
        P0 = sl.get('line_origin_B', None)
        line_dir = sl.get('line_dir', None)

        stamp = self.get_clock().now().to_msg()

        # --- 发布 ---
        # 场景 Marker 每帧都发 (避免 RViz 晚启动看不到)
        self.publisher.publish_scene_markers(stamp)

        self.publisher.publish_plate_tf(stamp, C, n_B, u_B, v_B)
        self.publisher.publish_sensor_tf(stamp, R_BS, t_BS)
        self.publisher.publish_frame_markers(stamp, R_BS, t_BS, scan_pts_B,
                                              endpoints_B_base, P0, line_dir)

        # 发布关节角 → robot_state_publisher 更新 TF
        js = JointState()
        js.header.stamp = stamp
        js.name = JOINT_NAMES
        js.position = self.joint_trajs[idx].tolist() if idx < len(self.joint_trajs) else [0.0]*6
        self.joint_pub.publish(js)

        # 日志
        if idx // 5 != self.last_log_frame // 5:
            n_ep = len(endpoints_B)
            dist_str = ""
            if 'physical_dist' in frame_data and frame_data['physical_dist'] > 0:
                dist_str = f"，间距={frame_data['physical_dist']*1000:.1f}mm"
            self.get_logger().info(
                f"  帧 {idx}/{len(self.poses)} 断点数={n_ep}{dist_str}")
            self.last_log_frame = idx

        self.current_frame = (idx + 1)
        if self.current_frame >= len(self.poses):
            self.get_logger().info("采集完成! 循环播放中...")
            self.current_frame = 0


def main(args=None):
    rclpy.init(args=args)
    node = CalibratioSimRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
