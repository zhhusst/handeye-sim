#!/usr/bin/env python3
"""
auto_servo_collect.py — v4 带断点反馈的沿边伺服

核心:
  - 读取扫描线断点 x_e 位置，做 P 控制保持 FOV 在边上
  - 沿边推进用基系方向（非法兰方向）
  - 每位姿采集后检查旋转多样性
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
import numpy as np
import sys, os, time, select

sys.path.insert(0, '/workspace/common')
from calib_solver import combined_solve_lm, combined_residuals, compute_errors
from fov_geometry import so3_exp, so3_log, rodrigues, rpy_to_matrix


def ros_transform_to_matrix(t):
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=[t.transform.translation.x,t.transform.translation.y,t.transform.translation.z]
    return T

def matrix_to_pose(R, t):
    q = np.zeros(4); tr = np.trace(R)
    if tr>0: S=np.sqrt(tr+1)*2; q[3]=0.25*S; q[0]=(R[2,1]-R[1,2])/S; q[1]=(R[0,2]-R[2,0])/S; q[2]=(R[1,0]-R[0,1])/S
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]: S=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; q[3]=(R[2,1]-R[1,2])/S; q[0]=0.25*S; q[1]=(R[0,1]+R[1,0])/S; q[2]=(R[0,2]+R[2,0])/S
    elif R[1,1]>R[2,2]: S=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; q[3]=(R[0,2]-R[2,0])/S; q[0]=(R[0,1]+R[1,0])/S; q[1]=0.25*S; q[2]=(R[1,2]+R[2,1])/S
    else: S=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; q[3]=(R[1,0]-R[0,1])/S; q[0]=(R[0,2]+R[2,0])/S; q[1]=(R[1,2]+R[2,1])/S; q[2]=0.25*S
    return Pose(position=Point(x=float(t[0]),y=float(t[1]),z=float(t[2])), orientation=Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3])))


class AutoServoCollect(Node):
    def __init__(self):
        super().__init__('auto_servo_collect')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PointCloud2, '/gocator/profile', self.profile_cb, 1)
        self.create_subscription(Float64MultiArray, '/gocator/endpoints', self.endpoint_cb, 1)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 1)

        self.latest_profile = None
        self.latest_endpoints = None
        self.latest_joints = None
        self.records = []
        self.moveit2 = None
        self.moveit_ready = False
        
        # 目标断点位置（传感器系 x 坐标，目标 = 居左一点）
        self.TARGET_X_E = -0.01  # -1cm (略偏左，留空间让扫描线覆盖)

        # 真值
        from fov_geometry import rpy_to_matrix
        rpy = np.array([0.485145, 0.160648, -1.509479])
        self.R_he_gt = rpy_to_matrix(np.rad2deg(rpy[0]), np.rad2deg(rpy[1]), np.rad2deg(rpy[2]))
        self.t_he_gt = np.array([-0.011579, -0.004621, 0.359284])

    def profile_cb(self, msg):
        from sensor_msgs_py.point_cloud2 import read_points
        try:
            gen = read_points(msg, field_names=('x','y','z'), skip_nans=True)
            pts = [list(p) for p in gen]
            if pts: self.latest_profile = np.array(pts, dtype=np.float64)
        except: pass

    def endpoint_cb(self, msg):
        if msg.data and len(msg.data) >= 9:
            self.latest_endpoints = {
                'n_endpoints': int(msg.data[0]),
                'e1_valid': bool(msg.data[4]),
                'e2_valid': bool(msg.data[8]),
                'p_S_e1': np.array(msg.data[1:4]) if msg.data[4] else None,
                'p_S_e2': np.array(msg.data[5:8]) if msg.data[8] else None,
            }

    def joint_cb(self, msg):
        JOINT_NAMES = ['J1_joint','J2_joint','J3_joint','J4_joint','J5_joint','J6_joint']
        try:
            self.latest_joints = [msg.position[msg.name.index(j)] for j in JOINT_NAMES]
        except: pass

    def get_hand_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('world', 'fanuc_flange', rclpy.time.Time())
            return ros_transform_to_matrix(t)
        except: return None

    def get_current_edge(self):
        ep = self.latest_endpoints
        if ep is None: return None
        if ep['e1_valid']: return 'e1'
        if ep['e2_valid']: return 'e2'
        return None

    def check_quality(self):
        return (self.latest_profile is not None and len(self.latest_profile) >= 5
                and self.latest_endpoints is not None
                and (self.latest_endpoints['e1_valid'] or self.latest_endpoints['e2_valid']))

    def get_breakpoint_x(self):
        """获取当前断点的传感器系 x 坐标"""
        ep = self.latest_endpoints
        if ep is None: return None
        if ep['e1_valid'] and ep['p_S_e1'] is not None:
            return 'e1', ep['p_S_e1'][0]
        if ep['e2_valid'] and ep['p_S_e2'] is not None:
            return 'e2', ep['p_S_e2'][0]
        return None

    def record_current(self):
        if not self.check_quality(): return False
        T = self.get_hand_pose()
        if T is None: return False
        ep = self.latest_endpoints
        # 检查旋转多样性
        if len(self.records) >= 1:
            min_diff = 999
            for r in self.records:
                dR = T[:3,:3].T @ r['T_B_H'][:3,:3]
                ang = np.rad2deg(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
                min_diff = min(min_diff, ang)
            if min_diff < 5.0 and len(self.records) >= 3:
                print(f"  旋转变化过小 ({min_diff:.1f}°<5°), 跳过")
                return False
        self.records.append({
            'T_B_H': T.copy(),
            'pts_S': self.latest_profile.copy(),
            'e1_S': ep['p_S_e1'].copy() if ep['e1_valid'] else None,
            'e2_S': ep['p_S_e2'].copy() if ep['e2_valid'] else None,
            'has_e1': ep['e1_valid'], 'has_e2': ep['e2_valid'],
        })
        pos = T[:3,3]; w = so3_log(T[:3,:3])
        print(f"  ✅ #{len(self.records)}: pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
              f"rot={np.rad2deg(np.linalg.norm(w)):.1f}° "
              f"pts={len(self.latest_profile)} e1={ep['e1_valid']} e2={ep['e2_valid']}")
        return True

    def init_moveit(self):
        if self.moveit_ready: return True
        try:
            from pymoveit2 import MoveIt2
            self.moveit2 = MoveIt2(
                node=self,
                joint_names=['J1_joint','J2_joint','J3_joint','J4_joint','J5_joint','J6_joint'],
                base_link_name='base_link', end_effector_name='flange',
                group_name='arm', use_move_group_action=True,
            )
            self.moveit_ready = True
            print("MoveIt2 ready")
            return True
        except Exception as e:
            print(f"MoveIt2 init error: {e}")
            return False

    def move_to_pose(self, R, t):
        """MoveIt2 移动到目标位姿"""
        if not self.moveit_ready: return False
        try:
            ps = PoseStamped(header=Header(frame_id='world'))
            ps.pose = matrix_to_pose(R, t)
            traj = self.moveit2.plan(pose=ps)
            if traj and hasattr(traj, 'points') and len(traj.points) > 0:
                self.moveit2.execute(traj)
                return True
            return False
        except Exception as e:
            print(f"  Move error: {e}")
            return False

    def _build_servo_pose(self, base_t, base_R, pitch_deg, yaw_deg, step_vec, lateral_corr):
        """构建伺服目标位姿
        
        Args:
            base_t, base_R: 当前法兰位姿
            pitch_deg, yaw_deg: 在传感器系中做的旋转
            step_vec: 基系推进方向 + 大小
            lateral_corr: 横向校正（基系）
        Returns:
            R_target, t_target
        """
        # 在传感器系做 pitch/yaw
        R_s = base_R @ self.R_he_gt
        R_p = rodrigues(R_s[:,0], np.deg2rad(pitch_deg))
        R_ps = R_p @ R_s
        R_y = rodrigues(R_ps[:,2], np.deg2rad(yaw_deg))
        R_ns = R_y @ R_p @ R_s
        R_target = R_ns @ self.R_he_gt.T
        t_target = base_t + step_vec + lateral_corr
        return R_target, t_target

    def run_calibration(self):
        if len(self.records) < 4:
            print(f"需要 >=4 记录，当前 {len(self.records)}")
            return
        poses = [(r['T_B_H'][:3,:3], r['T_B_H'][:3,3]) for r in self.records]
        meas = [{'p_S_plane': r['pts_S'], 'valid_e1': r['has_e1'], 'valid_e2': r['has_e2'],
                 'p_S_e1': r['e1_S'], 'p_S_e2': r['e2_S']} for r in self.records]
        best_th, best_c = None, np.inf
        for _ in range(10):
            th0 = np.concatenate([np.random.uniform(-0.8,0.8,3), np.random.uniform(-0.05,0.05,3), np.random.uniform(-0.35,0.35,3)])
            th = combined_solve_lm(th0, poses, meas, max_iter=200)
            rv, mask, _ = combined_residuals(th, poses, meas, 0.1, 1.0)
            c = 0.5 * np.dot(rv[mask], rv[mask])
            if c < best_c: best_c, best_th = c, th
        if best_th is None: return
        R_he = so3_exp(best_th[:3])
        dR = R_he.T @ self.R_he_gt
        R_err = np.rad2deg(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
        t_err = np.linalg.norm(best_th[3:6] - self.t_he_gt) * 1000
        n1 = sum(1 for r in self.records if r['has_e1']); n2 = sum(1 for r in self.records if r['has_e2'])
        print(f"\n{'='*40}\n标定结果: {len(self.records)} rec (e1={n1} e2={n2})\n"
              f"  R error: {R_err:.6f}°\n  t error: {t_err:.6f} mm\n{'='*40}")


def main():
    rclpy.init()
    node = AutoServoCollect()

    print("等待话题数据...")
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_endpoints is not None:
            break

    print("\n=== Auto Servo Collect ===")
    print("  'g' — 开始自动采集")
    print("  'r' — 手动记录单帧")
    print("  'c' — 运行标定")
    print("  'q' — 退出")

    # 预定义旋转序列 (pitch, yaw) — 覆盖大范围
    ANGLE_E1 = [(15,-25),(-8,20),(12,-10),(-15,25),(5,-20)]
    ANGLE_E2 = [(-15,25),(10,-20),(-12,15),(18,-25),(-10,-15)]
    N_PER_EDGE = 4  # 每边4个

    running = True
    while running and rclpy.ok():
        if select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            if key == 'q': running = False; break
            elif key == 'r': node.record_current()
            elif key == 'c': node.run_calibration()
            elif key == 'g':
                if not node.check_quality():
                    print("无效位姿！请先摇到有断点位姿")
                    continue
                if not node.init_moveit():
                    print("MoveIt2 init failed")
                    continue

                # 识别初始边
                edge = node.get_current_edge()
                print(f"\n初始边: {edge}")
                print("开始伺服采集...")

                # 确定采集顺序
                phases = ['e1'] if edge == 'e1' else ['e2']
                if edge == 'e1':
                    phases.append('e2')
                # phases = ['e1', 'e2'] 或 ['e2']

                for phase in phases:
                    print(f"\n--- 沿 {phase} 采集 ---")
                    angles = ANGLE_E1 if phase == 'e1' else ANGLE_E2

                    for idx in range(N_PER_EDGE):
                        T = node.get_hand_pose()
                        if T is None: continue
                        R_cur, t_cur = T[:3,:3], T[:3,3]
                        pa, ya = angles[idx]

                        # 读取断点位置做反馈
                        ep_data = node.get_breakpoint_x()
                        if ep_data is None:
                            print(f"  [{phase}] #{idx+1}: 无断点，跳过")
                            continue
                        e_type, x_e = ep_data
                        if e_type != phase:
                            print(f"  [{phase}] #{idx+1}: 当前看到{e_type}而非{phase}，跳过")
                            continue

                        # P 控制: 断点横向偏移校正
                        x_err = x_e - node.TARGET_X_E
                        K_p = 0.15  # 比例增益 (m / m_sensor)
                        lateral_corr_world = K_p * x_err * R_cur[:,2]  # 沿传感器 z 方向（垂直于扫描线方向）
                        # 注: 传感器 z 方向大致指向平板，横向校正在传感器 x-z 平面内
                        # 更精确：用传感器 y 方向（扫描平面法向）做横向校正
                        R_s = R_cur @ node.R_he_gt
                        lateral_corr = -K_p * x_err * R_s[:,1]  # 沿传感器 y_S 方向横向调整

                        # 推进方向：沿传感器 x_S 方向（扫描线方向）
                        step_mag = 0.04  # 每步 4cm
                        step_vec = step_mag * (1 if idx % 2 == 0 else -1) * R_s[:,0]

                        # 构建目标位姿
                        R_target, t_target = node._build_servo_pose(
                            t_cur, R_cur, pa, ya, step_vec, lateral_corr)

                        print(f"  [{phase}] #{idx+1}: pitch={pa:+d} yaw={ya:+d} x_e={x_e*1000:.0f}mm corr={np.linalg.norm(lateral_corr)*1000:.0f}mm")

                        # 移动
                        ok = node.move_to_pose(R_target, t_target)
                        if not ok:
                            print(f"  移动失败")
                            continue

                        # 稳定等待
                        for _ in range(10):
                            rclpy.spin_once(node, timeout_sec=0.05)
                            time.sleep(0.03)

                        # 采集
                        if node.check_quality():
                            node.record_current()
                        else:
                            print(f"  移动后无有效断点，尝试小幅度调整")
                            for retry in range(3):
                                rpa = pa + np.random.uniform(-5, 5)
                                rya = ya + np.random.uniform(-8, 8)
                                RT, tT = node._build_servo_pose(
                                    t_target, R_target, rpa, rya, np.zeros(3), np.zeros(3))
                                if node.move_to_pose(RT, tT):
                                    for _ in range(8):
                                        rclpy.spin_once(node, timeout_sec=0.05)
                                        time.sleep(0.03)
                                    if node.check_quality():
                                        node.record_current()
                                        break
                            else:
                                print(f"  调整失败，继续")

                    # 边1完成 → 过渡到边2
                    if phase == 'e1' and 'e2' in phases:
                        print("\n--- 90°旋转过渡到边2 ---")
                        T = node.get_hand_pose()
                        if T is not None:
                            R_s = T[:3,:3] @ node.R_he_gt
                            # 绕传感器 z 轴转 -90°
                            R_z90 = rodrigues(np.array([0,0,1]), np.deg2rad(-90))
                            R_new = (R_s @ R_z90.T) @ node.R_he_gt.T
                            node.move_to_pose(R_new, T[:3,3] + np.array([0, 0, 0.02]))
                            for _ in range(20):
                                rclpy.spin_once(node, timeout_sec=0.1)
                                time.sleep(0.05)

                # 完成
                n1 = sum(1 for r in node.records if r['has_e1'])
                n2 = sum(1 for r in node.records if r['has_e2'])
                print(f"\n采集完成！共 {len(node.records)} 记录 (e1={n1}, e2={n2})")
                print("按 'c' 运行标定")

        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
