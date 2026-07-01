#!/usr/bin/env python3
"""
auto_calib_node.py — v4: 基于视觉特征的自动采集

状态机: Init → EdgeKeep → ActiveExcite → Stabilize → Collect → Loop

特征:
  s = [x_b(断点横向), z_b(断点深度), α(板面倾角)]
控制:
  u = -K·e + u_c  (比例 + 主动激励)
  
无需手眼矩阵参与反馈（仅用URDF名义值做运动学映射）
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
import numpy as np
import sys, os, json, select, time
from enum import Enum, auto

sys.path.insert(0, '/workspace/common')
from calib_solver import combined_solve_lm, combined_residuals, compute_errors
from fov_geometry import so3_exp, so3_log, rodrigues, rpy_to_matrix, make_transform

# ========== 几何工具 ==========
def ros_transform_to_matrix(t):
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                   [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                   [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=[t.transform.translation.x,t.transform.translation.y,t.transform.translation.z]
    return T

def matrix_to_pose(R, t):
    R = np.asarray(R, dtype=np.float64); t = np.asarray(t, dtype=np.float64).flatten()
    q = np.zeros(4); tr = np.trace(R)
    if tr>0: S=np.sqrt(tr+1)*2; q[3]=0.25*S; q[0]=(R[2,1]-R[1,2])/S; q[1]=(R[0,2]-R[2,0])/S; q[2]=(R[1,0]-R[0,1])/S
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]: S=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; q[3]=(R[2,1]-R[1,2])/S; q[0]=0.25*S; q[1]=(R[0,1]+R[1,0])/S; q[2]=(R[0,2]+R[2,0])/S
    elif R[1,1]>R[2,2]: S=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; q[3]=(R[0,2]-R[2,0])/S; q[0]=(R[0,1]+R[1,0])/S; q[1]=0.25*S; q[2]=(R[1,2]+R[2,1])/S
    else: S=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; q[3]=(R[1,0]-R[0,1])/S; q[0]=(R[0,2]+R[2,0])/S; q[1]=(R[1,2]+R[2,1])/S; q[2]=0.25*S
    return Pose(position=Point(x=float(t[0]),y=float(t[1]),z=float(t[2])), orientation=Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3])))

class State(Enum):
    INIT = auto()
    EDGE_KEEP = auto()
    ACTIVE_EXCITE = auto()
    STABILIZE = auto()
    COLLECT = auto()
    DONE = auto()


class AutoCalibNode(Node):
    def __init__(self):
        super().__init__('auto_calib_node')
        from fov_geometry import rpy_to_matrix
        rpy = np.array([0.485145, 0.160648, -1.509479])
        self.R_he_nominal = rpy_to_matrix(np.rad2deg(rpy[0]), np.rad2deg(rpy[1]), np.rad2deg(rpy[2]))
        self.t_he_nominal = np.array([-0.011579, -0.004621, 0.359284])

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PointCloud2, '/gocator/profile', self._profile_cb, 1)
        self.create_subscription(Float64MultiArray, '/gocator/endpoints', self._endpoint_cb, 1)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 1)

        self.latest_profile = None
        self.latest_endpoints = None
        self.latest_joints = None
        self.records = []
        self.moveit2 = None

        # 伺服参数
        self.K_x = 0.08      # 横向增益 (m/m)
        self.K_z = 0.05      # 法向增益 (m/m)
        self.K_a = 0.3       # 旋转增益 (rad/rad)
        self.Z_D = 0.45      # 目标standoff (m)
        self.V_STEP = 0.04   # 沿边步进 (m)
        self.W_EXC = 0.2     # 激励振荡幅度 (rad)
        self.TOL_X = 0.005   # 横向收敛阈值 (m)
        self.TOL_Z = 0.005   # 法向收敛阈值 (m)
        self.TOL_A = 0.02    # 倾角收敛阈值 (rad)
        self.N_TARGET = 10   # 目标位姿数

        self.state = State.INIT
        self.state_count = 0
        self.excite_phase = 0  # 当前激励阶段的索引
        self.servo_running = False

        self.create_timer(0.1, self._keyboard_check)
        self.get_logger().info(
            "\n=== Auto Calib v4 (Visual Servo) ===\n"
            "  'g' — 开始自动采集\n  'r' — 手动记录  'c' — 标定\n"
            "  's' — 汇总  'q' — 退出\n"
            "======================================")

    def _profile_cb(self, msg):
        from sensor_msgs_py.point_cloud2 import read_points
        try:
            gen = read_points(msg, field_names=('x','y','z'), skip_nans=True)
            pts = [list(p) for p in gen]
            if pts: self.latest_profile = np.array(pts, dtype=np.float64)
        except: pass

    def _endpoint_cb(self, msg):
        if msg.data and len(msg.data) >= 9:
            self.latest_endpoints = {
                'n_endpoints': int(msg.data[0]),
                'e1_valid': bool(msg.data[4]), 'e2_valid': bool(msg.data[8]),
                'p_S_e1': np.array(msg.data[1:4]) if msg.data[4] else None,
                'p_S_e2': np.array(msg.data[5:8]) if msg.data[8] else None,
            }

    def _joint_cb(self, msg):
        JOINT_NAMES = ['J1_joint','J2_joint','J3_joint','J4_joint','J5_joint','J6_joint']
        try:
            self.latest_joints = np.array([msg.position[msg.name.index(j)] for j in JOINT_NAMES])
        except: pass

    def _get_hand_pose(self):
        try:
            return ros_transform_to_matrix(
                self.tf_buffer.lookup_transform('world', 'fanuc_flange', rclpy.time.Time()))
        except: return None

    def _get_current_edge(self):
        ep = self.latest_endpoints
        if ep is None: return None
        if ep['e1_valid']: return 'e1'
        if ep['e2_valid']: return 'e2'
        return None

    def _check_quality(self):
        return (self.latest_profile is not None and len(self.latest_profile) >= 5
                and self.latest_endpoints is not None
                and (self.latest_endpoints['e1_valid'] or self.latest_endpoints['e2_valid']))

    # ===== 视觉特征提取 =====
    def _extract_features(self):
        """从传感器数据提取视觉特征 s = [x_b, z_b, α]"""
        ep = self.latest_endpoints
        if ep is None: return None
        pt = ep['p_S_e1'] if ep['e1_valid'] else (ep['p_S_e2'] if ep['e2_valid'] else None)
        if pt is None: return None
        x_b = float(pt[0])  # 断点x (m)
        z_b = float(pt[2])  # 断点z (m)

        # 拟合板面点求倾角 α = arctan(k)   z = k*x + b
        prof = self.latest_profile
        alpha = 0.0
        if prof is not None and len(prof) >= 5:
            xs = prof[:, 0]; zs = prof[:, 2]
            A = np.column_stack([xs, np.ones_like(xs)])
            try:
                k, b = np.linalg.lstsq(A, zs, rcond=None)[0]
                alpha = float(np.arctan(k))
            except: pass

        return {
            'x_b': x_b, 'z_b': z_b, 'alpha': alpha,
            'on_edge': self._get_current_edge(),
        }

    # ===== 伺服核心 =====
    def _compute_servo_move(self, features, excite=False, step_dir=1):
        """计算基于视觉误差的相对运动"""
        e_x = features['x_b']           # 边缘横向偏差
        e_z = features['z_b'] - self.Z_D  # 距离偏差
        e_a = features['alpha']          # 倾角偏差

        # 比例控制
        v_x = -self.K_x * e_x           # 横向校正 (传感器 x 方向)
        v_z = -self.K_z * e_z           # 法向校正 (传感器 z 方向)
        w_y = -self.K_a * e_a           # 旋转校正 (绕传感器 y 轴)

        # 主动激励
        v_step = 0.0
        w_exc = 0.0
        if excite:
            v_step = self.V_STEP * step_dir  # 沿边推进
            w_exc = self.W_EXC * (1 if self.excite_phase % 2 == 0 else -1)  # 振荡

        return {
            'v_x': v_x + v_step,
            'v_z': v_z,
            'w_y': w_y + w_exc,
            'errors': (e_x, e_z, e_a),
        }

    def _move_to_servo_target(self, move_cmd):
        """将伺服指令转为机器人目标位姿并移动"""
        T = self._get_hand_pose()
        if T is None: return False

        R_hand = T[:3, :3]
        t_hand = T[:3, 3]

        # 当前传感器位姿 (使用名义手眼)
        R_sensor = R_hand @ self.R_he_nominal
        t_sensor = t_hand + R_hand @ self.t_he_nominal

        # 在传感器系做相对运动
        # 平移: [v_x, 0, v_z] 在传感器系
        delta_t_s = np.array([move_cmd['v_x'], 0.0, move_cmd['v_z']])
        # 旋转: 绕传感器 y 轴转 w_y
        R_delta = rodrigues(np.array([0, 1, 0]), move_cmd['w_y'])

        # 目标传感器位姿
        R_target_s = R_delta @ R_sensor
        t_target_s = t_sensor + R_sensor @ delta_t_s

        # 转回法兰位姿
        R_target_h = R_target_s @ self.R_he_nominal.T
        t_target_h = t_target_s - R_target_h @ self.t_he_nominal

        return self._move_to_pose(R_target_h, t_target_h)

    def _move_to_pose(self, R, t):
        if self.moveit2 is None: return False
        try:
            ps = PoseStamped(header=Header(frame_id='world'))
            ps.pose = matrix_to_pose(R, t)
            traj = self.moveit2.plan(pose=ps)
            if traj and hasattr(traj, 'points') and len(traj.points) > 0:
                self.moveit2.execute(traj)
                return True
            # 失败则尝试关节空间规划
            if self.latest_joints is not None:
                # 加微小关节扰动（防止原地不动）
                j_perturbed = self.latest_joints + np.deg2rad(np.array([0, 0, 2, 0, 3, 0]))
                traj2 = self.moveit2.plan(joint_positions=list(j_perturbed))
                if traj2 and hasattr(traj2, 'points') and len(traj2.points) > 0:
                    self.moveit2.execute(traj2)
                    return True
            return False
        except Exception as e:
            self.get_logger().error(f"Move: {e}")
            return False

    def _init_moveit(self):
        if self.moveit2 is not None: return True
        try:
            from pymoveit2 import MoveIt2
            self.moveit2 = MoveIt2(
                node=self,
                joint_names=['J1_joint','J2_joint','J3_joint','J4_joint','J5_joint','J6_joint'],
                base_link_name='base_link', end_effector_name='flange',
                group_name='arm', use_move_group_action=True,
            )
            return True
        except Exception as e:
            self.get_logger().error(f"MoveIt2: {e}")
            return False

    def _record_current(self):
        if not self._check_quality(): return False
        T = self._get_hand_pose()
        if T is None: return False
        ep = self.latest_endpoints
        self.records.append({
            'T_B_H': T.copy(),
            'pts_S': self.latest_profile.copy(),
            'e1_S': ep['p_S_e1'].copy() if ep['e1_valid'] else None,
            'e2_S': ep['p_S_e2'].copy() if ep['e2_valid'] else None,
            'has_e1': ep['e1_valid'], 'has_e2': ep['e2_valid'],
        })
        pos = T[:3, 3]
        w = so3_log(T[:3, :3])
        self.get_logger().info(
            f"  COLLECT #{len(self.records)}: pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
            f"rot={np.rad2deg(np.linalg.norm(w)):.1f}° "
            f"e1={ep['e1_valid']} e2={ep['e2_valid']}")
        return True

    # ===== 自动采集主循环 =====
    def _servo_loop(self):
        """伺服采集主循环——在同步循环内运行"""
        self.get_logger().info("\n开始自动采集...")
        self.state = State.EDGE_KEEP
        self.state_count = 0
        self.records = []
        self.excite_phase = 0
        self.servo_running = True

        step_dir = 1  # 沿边推进方向
        collected_count = 0
        max_attempts = 50  # 防止无限循环
        attempt = 0

        while self.servo_running and collected_count < self.N_TARGET and attempt < max_attempts:
            attempt += 1
            rclpy.spin_once(self, timeout_sec=0.05)

            # 读特征
            features = self._extract_features()
            if features is None or features['on_edge'] is None:
                self.get_logger().warn(f"  位姿{attempt}: 无有效特征，跳过")
                continue

            # ---- 状态机 ----
            # 1. Edge Keep: 小幅校正
            move = self._compute_servo_move(features, excite=False)
            ex, ez, ea = move['errors']
            if abs(ex) > self.TOL_X or abs(ez) > self.TOL_Z or abs(ea) > self.TOL_A:
                # 误差还大，继续校正
                self.get_logger().info(
                    f"  EDGE_KEEP: x_err={ex*1000:.0f}z_err={ez*1000:.0f}α_err={np.rad2deg(ea):.1f}")
                ok = self._move_to_servo_target(move)
                if ok:
                    for _ in range(10):
                        rclpy.spin_once(self, timeout_sec=0.05)
                        time.sleep(0.02)
                continue

            # 2. Edge 已保持住 → 稳定 → 采集
            # 先采集当前位姿
            if self._check_quality():
                self._record_current()
                collected_count += 1
                self.excite_phase += 1

            # 3. 如果采够了，退出
            if collected_count >= self.N_TARGET:
                break

            # 4. 主动激励: 沿边推进 + 振荡
            if self.excite_phase % 2 == 0:
                step_dir = 1
            else:
                step_dir = -1

            move_exc = self._compute_servo_move(features, excite=True, step_dir=step_dir)
            self.get_logger().info(
                f"  ACTIVE_EXCITE #{self.excite_phase}: "
                f"v_x={move_exc['v_x']*1000:.0f} w_y={np.rad2deg(move_exc['w_y']):.0f}°")
            ok = self._move_to_servo_target(move_exc)
            if not ok:
                self.get_logger().warn("  激励移动失败，尝试更小步长")
                move_exc['v_x'] *= 0.3
                move_exc['w_y'] *= 0.3
                self._move_to_servo_target(move_exc)

            # 等待稳定
            for _ in range(15):
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.02)

            if self.excite_phase > 20:
                break

        # 完成
        self.servo_running = False
        self.state = State.DONE
        n1 = sum(1 for r in self.records if r['has_e1'])
        n2 = sum(1 for r in self.records if r['has_e2'])
        self.get_logger().info(
            f"\n采集完成! {len(self.records)} 记录 (e1={n1}, e2={n2})\n"
            f"按 'c' 运行标定")

    def start_servo(self):
        if self.servo_running:
            self.get_logger().warn("正在采集中")
            return
        if not self._check_quality():
            self.get_logger().warn("初始位姿无效，请先摇到有断点位姿")
            return
        if not self._init_moveit():
            return

        import threading
        t = threading.Thread(target=self._servo_loop, daemon=True)
        t.start()

    def run_calibration(self):
        if len(self.records) < 4:
            self.get_logger().warn(f"需 ≥4 记录，当前 {len(self.records)}")
            return
        poses = [(r['T_B_H'][:3,:3], r['T_B_H'][:3,3]) for r in self.records]
        meas = [{'p_S_plane': r['pts_S'], 'valid_e1': r['has_e1'], 'valid_e2': r['has_e2'],
                 'p_S_e1': r['e1_S'], 'p_S_e2': r['e2_S']} for r in self.records]

        from fov_geometry import rpy_to_matrix
        rpy = np.array([0.485145, 0.160648, -1.509479])
        R_gt = rpy_to_matrix(np.rad2deg(rpy[0]), np.rad2deg(rpy[1]), np.rad2deg(rpy[2]))
        t_gt = np.array([-0.011579, -0.004621, 0.359284])

        best_th, best_c = None, np.inf
        for _ in range(10):
            th0 = np.concatenate([np.random.uniform(-0.8,0.8,3), np.random.uniform(-0.05,0.05,3), np.random.uniform(-0.35,0.35,3)])
            th = combined_solve_lm(th0, poses, meas, max_iter=200)
            rv, mask, _ = combined_residuals(th, poses, meas, 0.1, 1.0)
            c = 0.5 * np.dot(rv[mask], rv[mask])
            if c < best_c: best_c, best_th = c, th
        if best_th is None: return
        R_he = so3_exp(best_th[:3])
        dR = R_he.T @ R_gt
        R_err = np.rad2deg(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
        t_err = np.linalg.norm(best_th[3:6] - t_gt) * 1000
        self.get_logger().info(f"\nR: {R_err:.6f}°  t: {t_err:.6f} mm")

    def show_summary(self):
        self.get_logger().info(f"Records: {len(self.records)}")
        for i, r in enumerate(self.records):
            pos = r['T_B_H'][:3,3]
            e1="e1" if r['has_e1'] else "  "; e2="e2" if r['has_e2'] else "  "
            self.get_logger().info(f"  [{i}] ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) {e1} {e2}")

    def _keyboard_check(self):
        if select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            if key == 'g': self.start_servo()
            elif key == 'r': self._record_current()
            elif key == 'c': self.run_calibration()
            elif key == 's': self.show_summary()
            elif key == 'q': raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = AutoCalibNode()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
