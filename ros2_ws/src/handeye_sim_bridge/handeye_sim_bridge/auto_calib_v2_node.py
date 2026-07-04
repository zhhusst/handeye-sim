#!/usr/bin/env python3
"""
auto_calib_v2_node.py — METHOD.md 附录 B v2 自动标定管线

Phase 0: 锚点初始化 (N_pts 验证)
Phase 1: 探索运动 — 6 扰动 + PCA → (n_B, d, C)
Phase 2: NBV 循环 — Z预检查 + FIM + Z位置修正 + 增量PCA + 弹球
Phase 3: 平面约束 LM 求解

参考: METHOD.md 附录 B.4 v2
隔离: 不依赖 calib_solver.py / calibrate.py / verify_12dof.py
      仅共享 plane_calib.py (求解器), fov_geometry.py, fanuc_kinematic.py

用法:
  ros2 run handeye_sim_bridge auto_calib_v2
  按键: 'a' — 启动自动标定  'c' — 仅求解  'q' — 退出
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
from tf2_ros import Buffer, TransformListener
import numpy as np
import sys, os, json, select

sys.path.insert(0, '/workspace/common')
from plane_calib import (
    solve_plane_he, init_plane_from_scans,
    _unpack_plane_theta, _n_from_angles,
    compute_fim, parameter_covariance, predict_info_gain,
    compute_laser_plate_intersection,
)
from fov_geometry import so3_exp, so3_log, rodrigues, rpy_to_matrix
from handeye_sim_bridge.fanuc_kinematic import inverse_kinematics

# ─── 工具函数 ──────────────────────────────────────────────

def ros_tf_to_matrix(t):
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return R, np.array([t.transform.translation.x,
                         t.transform.translation.y,
                         t.transform.translation.z])


def weighted_pca(points, weights):
    """加权 PCA：返回 (n_B, d, centroid)。"""
    if len(points) < 10:
        return None, None, None
    w = np.asarray(weights) / np.sum(weights)
    centroid = np.average(points, axis=0, weights=w)
    centered = points - centroid
    cov = (centered.T * w) @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    n_B = eigvecs[:, 0]   # 最小特征值 → 法向量
    d = float(np.dot(n_B, centroid))
    return n_B, d, centroid


# ─── 主节点 ────────────────────────────────────────────────

class AutoCalibV2Node(Node):
    def __init__(self):
        super().__init__('auto_calib_v2_node')

        # ─── 名义手眼（URDF真值）─────────────────────
        rpy = np.array([0.485145, 0.160648, -1.509479])
        self.R_he_nom = rpy_to_matrix(np.rad2deg(rpy[0]),
                                       np.rad2deg(rpy[1]),
                                       np.rad2deg(rpy[2]))
        self.t_he_nom = np.array([-0.011579, -0.004621, 0.359284])

        # ─── 真值（用于评估）────────────────────────
        self.R_he_gt = rpy_to_matrix(27.8, 9.2, -86.5)
        self.t_he_gt = np.array([-0.011579, -0.004621, 0.359284])

        # ─── TF / 传感器 / 关节 ──────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PointCloud2, '/gocator/profile',
                                  self._profile_cb, 1)
        self.create_subscription(JointState, '/joint_states',
                                  self._joint_cb, 1)
        self.latest_profile = None
        self.latest_joints = None
        self.records = []

        # ─── 动作客户端 ──────────────────────────────
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        # ─── Phase 0 锚点 ────────────────────────────
        self.Z_0 = None           # 锚点激光中心深度
        self.N_anchor = None      # 锚点点数
        self.R_BS_0 = None        # 锚点传感器朝向
        self.t_BS_0 = None        # 锚点传感器位置
        self.vTCP = None          # 虚拟 TCP (基座标系)

        # ─── Phase 1 板估计 ──────────────────────────
        self.n_B = None
        self.d_plate = None
        self.C = None             # 板中心 (基座标系)

        # ─── 状态机 ──────────────────────────────────
        self._state = 'IDLE'
        self._auto_phase = None    # 'PHASE0' | 'PHASE1' | 'PHASE2' | None
        self._phase1_step = 0
        self._phase2_count = 0
        self._phase2_target = 25
        self._auto_queue = []
        self._rng = np.random.RandomState(42)

        # ─── Phase 2 状态 ────────────────────────────
        self._weighted_pts = []     # [(pt, weight), ...] 增量 PCA
        self._cand_angles = [15, 25]  # 候选范围 (随置信度扩展)
        self._last_valid_T = None    # 上一有效位姿 (回退用)
        self._last_valid_scan = None
        self._total_collected = 0

        # ─── 定时器 ──────────────────────────────────
        self.create_timer(0.2, self._state_machine)
        self.create_timer(0.1, self._keyboard_check)

        self.get_logger().info(
            "\n╔══════════════════════════════════════╗\n"
            "║  Auto Calib v2  (METHOD.md Appendix B) ║\n"
            "║  'a' — 启动  'c' — 求解  'q' — 退出   ║\n"
            "╚══════════════════════════════════════╝")

    # ─── 回调 ───────────────────────────────────────────

    def _profile_cb(self, msg):
        from sensor_msgs_py.point_cloud2 import read_points
        try:
            pts = [list(p) for p in read_points(
                msg, field_names=('x','y','z'), skip_nans=True)]
            self.latest_profile = np.array(pts, dtype=np.float64) if pts else None
        except Exception:
            pass

    def _joint_cb(self, msg):
        JOINT_NAMES = ['J1_joint','J2_joint','J3_joint',
                        'J4_joint','J5_joint','J6_joint']
        try:
            self.latest_joints = np.array(
                [msg.position[msg.name.index(j)] for j in JOINT_NAMES])
        except Exception:
            pass

    def _get_hand_pose(self):
        try:
            return self.tf_buffer.lookup_transform(
                'world', 'fanuc_flange', rclpy.time.Time())
        except Exception:
            return None

    def _get_sensor_pose(self):
        """传感器→基座标系 (通过名义手眼)"""
        tf = self._get_hand_pose()
        if tf is None:
            return None, None
        R_BH, t_BH = ros_tf_to_matrix(tf)
        R_BS = R_BH @ self.R_he_nom
        t_BS = t_BH + R_BH @ self.t_he_nom
        return R_BS, t_BS

    def _extract_features(self):
        """提取 2D 轮廓特征。返回 dict 或 None。"""
        if self.latest_profile is None or len(self.latest_profile) < 5:
            return None
        pts = self.latest_profile
        xs, zs = pts[:, 0], pts[:, 2]
        return {
            'Z_center': float(np.mean(zs)),
            'Z_min': float(np.min(zs)),
            'Z_max': float(np.max(zs)),
            'n_pts': len(pts),
        }

    def _valid(self, min_pts=10):
        return (self.latest_profile is not None
                and len(self.latest_profile) >= min_pts)

    # ─── 关节移动 ───────────────────────────────────────

    def _send_joint_target(self, joints, label=''):
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'J1_joint','J2_joint','J3_joint',
            'J4_joint','J5_joint','J6_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in joints]
        pt.time_from_start = Duration(sec=2, nanosec=0)
        goal.trajectory.points = [pt]
        self._state = 'MOVING'
        self._traj_client.wait_for_server(timeout_sec=1.0)
        self._traj_client.send_goal_async(goal).add_done_callback(
            self._goal_response_cb)

    def _goal_response_cb(self, f):
        h = f.result()
        if not h.accepted:
            self._state = 'IDLE'
            return
        h.get_result_async().add_done_callback(
            lambda _: setattr(self, '_state', 'WAIT'))

    def _move_to_pose(self, R_BH, t_BH, label=''):
        """法兰位姿 → IK → 关节移动。返回 True 如果成功入队。"""
        Tt = np.eye(4)
        Tt[:3, :3] = R_BH
        Tt[:3, 3] = t_BH
        qs = inverse_kinematics(Tt)
        if len(qs) == 0:
            self.get_logger().warn(f'  IK 失败: {label}')
            return False
        qs = np.atleast_2d(qs)
        # 选最近解
        if self.latest_joints is not None:
            d = np.linalg.norm(qs - self.latest_joints, axis=1)
            q = qs[np.argmin(d)]
        else:
            q = qs[0]
        self._auto_queue.append((label, q))
        return True

    # ─── 状态机 ──────────────────────────────────────────

    def _state_machine(self):
        if not hasattr(self, '_auto_phase'):
            return
        if self._state == 'IDLE' and self._auto_queue:
            label, joints = self._auto_queue.pop(0)
            self.get_logger().info(f'  → {label}')
            self._send_joint_target(joints, label)
        if self._state == 'WAIT':
            self._scan_wait = getattr(self, '_scan_wait', 0) + 1
            if self._scan_wait >= 6:
                self._state = 'IDLE'
                self._scan_wait = 0
                self._on_move_done()
        if (not self._auto_queue and self._state == 'IDLE'
                and self._auto_phase):
            self._next_step()

    def _on_move_done(self):
        """每次移动后：Z位置修正 → 特征检查 → 记录 → 增量更新"""
        f = self._extract_features()
        if f is None or f['n_pts'] < 5:
            self.get_logger().warn('  → 无效位姿')
            self._on_invalid_move()
            return

        # ─── 步骤 3b: Z 位置修正 (Xiao Eq.12 线激光版) ───
        if (self._auto_phase == 'PHASE2' and self.Z_0 is not None
                and f['n_pts'] >= 10):
            dz = f['Z_center'] - self.Z_0
            if abs(dz) > 0.005:  # 5mm
                self.get_logger().info(
                    f'  Z修正: {dz*1000:.1f}mm')
                self._correct_z(dz)
                return  # 等待修正后重新回调

        # ─── 步骤 4: 特征检查 ───────────────────────
        self._check_and_record(f)

    def _correct_z(self, dz):
        """沿传感器 Z 轴修正位置"""
        if self._servo_count >= 3:
            self._servo_count = 0
            self.get_logger().warn('  Z修正已达3次，跳过')
            self._check_and_record(self._extract_features())
            return
        self._servo_count = getattr(self, '_servo_count', 0) + 1
        T = self._get_hand_pose()
        if T is None:
            return
        R_BH, t_BH = ros_tf_to_matrix(T)
        zB = R_BH @ self.R_he_nom[:, 2]  # 传感器 Z 轴在 base 系
        t_new = t_BH + zB * (-dz)         # 修正方向：让 Z_center 回到 Z_0
        self._move_to_pose(R_BH, t_new,
                           f'Z修正 {-dz*1000:.0f}mm')
        self._state = 'IDLE'
        self._scan_wait = 0

    def _check_and_record(self, f):
        """特征检查 + 记录 + 增量更新"""
        if self._auto_phase == 'PHASE1':
            # Phase 1 简化检查
            if f['n_pts'] < self.N_anchor * 0.3:
                self.get_logger().warn(f'  Phase1 点数骤降 ({f["n_pts"]})')
                self._next_step()
                return
            self._record_pose()
            self._next_step()
            return

        if self._auto_phase == 'PHASE2':
            n = f['n_pts']
            # 正常
            if n > self.N_anchor * 0.5 and abs(f['Z_center'] - self.Z_0) < 0.02:
                self._record_pose()
                self._incremental_update()
                self._next_step()
                return
            # 弹球反弹
            if 10 <= n <= self.N_anchor * 0.5:
                self.get_logger().warn(
                    f'  弹球: N_pts={n} → 向C收缩')
                self._bounce_toward_C()
                return
            # 紧急回退
            if n < 10:
                self.get_logger().error('  紧急回退!')
                self._retreat()
                return
            # Z 漂移
            if abs(f['Z_center'] - self.Z_0) > 0.02:
                self.get_logger().warn(
                    f'  Z漂移: {abs(f["Z_center"]-self.Z_0)*1000:.0f}mm')
                self._correct_z(f['Z_center'] - self.Z_0)
                return

    def _on_invalid_move(self):
        """移动后无法获取有效数据"""
        if self._auto_phase == 'PHASE1':
            self.get_logger().warn('  Phase1 扰动无效，跳过')
            self._next_step()
            return
        if self._auto_phase == 'PHASE2':
            self.get_logger().error('  Phase2 无效 → 回退')
            self._retreat()

    def _record_pose(self):
        """记录当前位姿和扫描数据"""
        T = self._get_hand_pose()
        if T is None:
            return
        R_BH, t_BH = ros_tf_to_matrix(T)
        scan = (self.latest_profile.copy()
                if self.latest_profile is not None
                else np.zeros((0, 3)))
        self.records.append({
            'T_B_H': np.eye(4),
            'R_BH': R_BH, 't_BH': t_BH,
            'pts_S': scan,
        })
        self._last_valid_T = (R_BH.copy(), t_BH.copy())
        self._last_valid_scan = scan.copy()
        self._total_collected += 1
        self.get_logger().info(
            f'  REC #{self._total_collected}'
            f' ({t_BH[0]:.3f},{t_BH[1]:.3f},{t_BH[2]:.3f})'
            f' pts={len(scan)}')

    def _bounce_toward_C(self):
        """弹球：激光线中心向板中心 C 移动 20mm"""
        if self.C is None:
            self._retreat()
            return
        R_BS, t_BS = self._get_sensor_pose()
        if R_BS is None:
            return
        # 当前激光线中心在 base 系
        f = self._extract_features()
        Z_now = f['Z_center'] if f else self.Z_0
        p_center = t_BS + R_BS @ np.array([0, 0, Z_now])
        # 向 C 方向
        direction = self.C - p_center
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return
        direction /= dist
        step = min(0.02, dist * 0.5)  # 最多 20mm
        t_BS_new = t_BS + direction * step
        # 保持朝向不变
        R_BH = R_BS @ self.R_he_nom.T
        t_BH = t_BS_new - R_BH @ self.t_he_nom
        self._move_to_pose(R_BH, t_BH, f'弹球→C {step*1000:.0f}mm')

    def _retreat(self):
        """紧急回退到上一有效位姿"""
        if self._last_valid_T is None:
            self.get_logger().error('  无可回退位姿!')
            self._next_step()
            return
        R_BH, t_BH = self._last_valid_T
        self._move_to_pose(R_BH, t_BH, '回退')

    # ─── 增量更新 (Phase 2) ────────────────────────────

    def _incremental_update(self):
        """每个位姿后增量更新板平面 (Xiao EKF 等价)"""
        R_BS, t_BS = self._get_sensor_pose()
        if R_BS is None or self.latest_profile is None:
            return
        scan = self.latest_profile
        # 转点到 base 系
        pts_B = (R_BS @ scan.T).T + t_BS
        # 加权衰减 (新帧权重 1.0, 旧帧指数衰减)
        decay = 0.7  # λ ≈ 0.36 (1 - 0.7 ≈ 0.3)
        for i in range(len(self._weighted_pts)):
            self._weighted_pts[i] = (
                self._weighted_pts[i][0],
                self._weighted_pts[i][1] * decay)
        # 加入新点
        for pt in pts_B:
            self._weighted_pts.append((pt, 1.0))
        # 保持总点数上限
        if len(self._weighted_pts) > 5000:
            self._weighted_pts = self._weighted_pts[-5000:]
        # PCA
        pts_arr = np.array([p for p, _ in self._weighted_pts])
        w_arr = np.array([w for _, w in self._weighted_pts])
        n_B, d_plate, C = weighted_pca(pts_arr, w_arr)
        if n_B is not None:
            self.n_B = n_B
            self.d_plate = d_plate
            self.C = C
            # 协方差检查 → 是否扩展候选范围
            centered = pts_arr - C
            cov = (centered.T * w_arr) @ centered
            cov /= np.sum(w_arr)
            det = np.linalg.det(cov[:2, :2]) if len(cov) > 2 else 1e10
            if det < 0.0001 and self._cand_angles != [15, 25, 35, 45]:
                self._cand_angles = [15, 25, 35, 45]
                self.get_logger().info('  → 候选范围扩展至 ±45°')

    def _init_incremental_pca(self):
        """用 Phase 1 全部数据初始化加权点集"""
        self._weighted_pts = []
        for r in self.records:
            R_BH = r.get('R_BH')
            t_BH = r.get('t_BH')
            scan = r.get('pts_S')
            if R_BH is None or scan is None:
                continue
            R_BS = R_BH @ self.R_he_nom
            t_BS = t_BH + R_BH @ self.t_he_nom
            pts_B = (R_BS @ scan.T).T + t_BS
            for pt in pts_B:
                self._weighted_pts.append((pt, 1.0))

    # ─── Phase 0: 锚点 ──────────────────────────────────

    def _start(self):
        """按 'a' 触发"""
        if not self._valid(5):
            self.get_logger().warn('初始位姿无效 (N_pts < 5)')
            return
        self.get_logger().info('\n╔══ Phase 0: 锚点 ══╗')
        self.records = []
        self._weighted_pts = []
        self._total_collected = 0
        self._servo_count = 0
        self._cand_angles = [15, 25]

        # 传感器位姿
        R_BS, t_BS = self._get_sensor_pose()
        f = self._extract_features()
        if f is None:
            self.get_logger().error('无法提取特征')
            return

        # N_pts 验证 (METHOD.md B.4 Phase 0 步骤 3)
        if f['n_pts'] < 20:
            self.get_logger().error(
                f'N_pts={f["n_pts"]} < 20，请调整位姿')
            return

        self.R_BS_0 = R_BS
        self.t_BS_0 = t_BS
        self.Z_0 = f['Z_center']
        self.N_anchor = f['n_pts']

        # vTCP
        self.vTCP = t_BS + R_BS @ np.array([0, 0, self.Z_0])

        self.get_logger().info(
            f'  Z0={self.Z_0*1000:.0f}mm'
            f'  N_anchor={self.N_anchor}'
            f'  vTCP=({self.vTCP[0]:.3f},{self.vTCP[1]:.3f},{self.vTCP[2]:.3f})')

        self._record_pose()   # pose_0
        self._last_valid_T = (R_BS @ self.R_he_nom.T,
                              t_BS - (R_BS @ self.R_he_nom.T) @ self.t_he_nom)
        self._last_valid_scan = self.latest_profile.copy()

        # → Phase 1
        self._build_phase1()
        self._auto_phase = 'PHASE1'
        self._phase1_step = 0
        self._next_step()

    # ─── Phase 1: 探索运动 ─────────────────────────────

    def _build_phase1(self):
        """6 扰动: +X, -X, +Z, RY+, RY-, RZ+ (METHOD.md B.4 Phase 1)"""
        self.get_logger().info('\n╔══ Phase 1: 探索运动 (6 扰动) ══╗')

        self._phase1_tasks = []
        deg = np.deg2rad

        # pose_1: +X 8mm (沿激光线)
        self._phase1_tasks.append(('+X 8mm', 'translate',
                                    np.array([0.008, 0, 0])))

        # pose_2: -X 8mm
        self._phase1_tasks.append(('-X 8mm', 'translate',
                                    np.array([-0.008, 0, 0])))

        # pose_3: +Z 5mm (靠近板)
        self._phase1_tasks.append(('+Z 5mm', 'translate',
                                    np.array([0, 0, 0.005])))

        # pose_4: RY +3°
        self._phase1_tasks.append(('RY +3°', 'rotate',
                                    rodrigues(np.array([0, 1, 0]), deg(3))))

        # pose_5: RY -3°
        self._phase1_tasks.append(('RY -3°', 'rotate',
                                    rodrigues(np.array([0, 1, 0]), deg(-3))))

        # pose_6: RZ +5° (平面内旋转)
        self._phase1_tasks.append(('RZ +5°', 'rotate',
                                    rodrigues(np.array([0, 0, 1]), deg(5))))

    def _next_step(self):
        """Phase 1/2 的步骤调度"""
        if self._auto_phase == 'PHASE1':
            self._next_phase1()
        elif self._auto_phase == 'PHASE2':
            self._next_phase2()

    def _next_phase1(self):
        """Phase 1 步骤调度"""
        if self._phase1_step >= len(self._phase1_tasks):
            self._finish_phase1()
            return

        label, mtype, param = self._phase1_tasks[self._phase1_step]
        self._phase1_step += 1
        self.get_logger().info(f'\n  ── Phase1.{self._phase1_step}: {label}')

        if mtype == 'translate':
            # 传感器系平移 → vTCP 方程 → 法兰位姿
            t_BS_new = self.t_BS_0 + self.R_BS_0 @ param
            R_BS_new = self.R_BS_0
            R_BH = R_BS_new @ self.R_he_nom.T
            t_BH = t_BS_new - R_BH @ self.t_he_nom
            self._move_to_pose(R_BH, t_BH, label)

        elif mtype == 'rotate':
            R_p = param  # 旋转矩阵 (绕传感器轴)
            R_BS_new = R_p @ self.R_BS_0
            # vTCP 约束：保持激光中心打在同一点
            t_BS_new = self.vTCP - R_BS_new @ np.array([0, 0, self.Z_0])
            R_BH = R_BS_new @ self.R_he_nom.T
            t_BH = t_BS_new - R_BH @ self.t_he_nom
            self._move_to_pose(R_BH, t_BH, label)

    def _finish_phase1(self):
        """Phase 1 完成：PCA → (n_B, d, C)"""
        n = len(self.records)
        self.get_logger().info(f'\n╚══ Phase 1 完成: {n} 位姿 ══╝')

        if n >= 4:
            poses = [(r['R_BH'], r['t_BH']) for r in self.records
                     if 'R_BH' in r]
            scans = [r['pts_S'] for r in self.records]
            th = init_plane_from_scans(
                poses, scans, self.R_he_nom, self.t_he_nom)
            if th is not None:
                w_he, t_he, tn, pn, d = _unpack_plane_theta(th)
                self.n_B = _n_from_angles(tn, pn)
                self.d_plate = d
                # 估算板中心 C (所有转点均值)
                all_pts = []
                for (R_BH, t_BH), scan in zip(poses, scans):
                    R_BS = R_BH @ self.R_he_nom
                    t_BS = t_BH + R_BH @ self.t_he_nom
                    pts_B = (R_BS @ scan.T).T + t_BS
                    all_pts.append(pts_B)
                if all_pts:
                    self.C = np.mean(np.vstack(all_pts), axis=0)
                self.get_logger().info(
                    f'  n_B=({self.n_B[0]:.3f},{self.n_B[1]:.3f},{self.n_B[2]:.3f})'
                    f'  d={self.d_plate:.3f}'
                    f'  C=({self.C[0]:.3f},{self.C[1]:.3f},{self.C[2]:.3f})')

        # → Phase 2
        self._init_incremental_pca()
        self._auto_phase = 'PHASE2'
        self._phase2_count = 0
        self._build_phase2()

    # ─── Phase 2: NBV 循环 ──────────────────────────────

    def _build_phase2(self):
        """生成首批 NBV 候选 (METHOD.md B.4 Phase 2)"""
        self.get_logger().info('\n╔══ Phase 2: NBV 循环 ══╗')

        if self.C is None or self.n_B is None:
            self.get_logger().error('无板平面估计，无法 Phase 2')
            self._auto_phase = None
            return

        self._phase2_candidates = self._generate_candidates()
        if not self._phase2_candidates:
            self.get_logger().error('无有效候选')
            self._auto_phase = None
            return

        self.get_logger().info(
            f'  候选: {len(self._phase2_candidates)} 个')

    def _generate_candidates(self):
        """生成 + 过滤 + FIM 排序候选位姿"""
        deg = np.deg2rad
        raw = []

        # 生成候选朝向
        for ang in self._cand_angles:
            for ax in range(3):
                a = np.zeros(3)
                a[ax] = 1
                for sgn in [1, -1]:
                    R_p = rodrigues(a, deg(sgn * ang))
                    R_BS = R_p @ self.R_BS_0
                    raw.append(R_BS)
            # 组合旋转
            if ang <= 25:
                for (a1, a2) in [(0, 1), (0, 2), (1, 2)]:
                    v1 = np.zeros(3); v1[a1] = 1
                    v2 = np.zeros(3); v2[a2] = 1
                    a = deg(ang * 0.7)
                    R_p = rodrigues(v1, a) @ rodrigues(v2, a)
                    R_BS = R_p @ self.R_BS_0
                    raw.append(R_BS)

        # 板平面约束 → 传感器位置 (METHOD.md B.4 Phase 2 步骤 2a)
        d_offset = 0.50  # 500mm 工作距离
        candidates = []
        for R_BS in raw:
            # 传感器位置: t_BS = C + d_offset * n_B
            # 但还需要保证激光面朝向合理
            # 简化：t_BS = C + d_offset * n_B
            t_BS = self.C + d_offset * self.n_B

            # Z 预检查 (步骤 2b): 预测交线 Z 是否接近 Z_0
            pts_pred, seg_len = compute_laser_plate_intersection(
                self.n_B, self.d_plate, self.C,
                0.400, 0.500,  # 板尺寸
                R_BS, t_BS, n_pts=20)
            if pts_pred is None or seg_len < 0.03:
                continue

            # 预测 Z 值
            Z_pred = np.mean(pts_pred[:, 2])
            if abs(Z_pred - self.Z_0) > 0.050:
                continue

            # 预测扫描点用于 FIM
            # (交线在传感器系)
            R_BH = R_BS @ self.R_he_nom.T
            t_BH = t_BS - R_BH @ self.t_he_nom
            candidates.append((R_BH, t_BH, R_BS, pts_pred))

        # FIM 排序 (步骤 2d-e)
        if len(self.records) >= 8 and self.n_B is not None:
            poses_cur = [(r['R_BH'], r['t_BH']) for r in self.records
                         if 'R_BH' in r]
            scans_cur = [r['pts_S'] for r in self.records]
            # 用板参数构造 theta
            tn = np.arccos(np.clip(self.n_B[2], -1, 1))
            pn = np.arctan2(self.n_B[1], self.n_B[0])
            from plane_calib import _pack_plane_theta
            theta_cur = _pack_plane_theta(
                np.zeros(3), self.t_he_nom, tn, pn, self.d_plate)

            def score(c):
                R_BH, t_BH, R_BS, pts_pred = c
                gain = predict_info_gain(
                    theta_cur, scans_cur, poses_cur,
                    R_BH, t_BH, pts_pred)
                return gain

            candidates.sort(key=score, reverse=True)
        else:
            # 无足够数据 → 随机
            self._rng.shuffle(candidates)

        return candidates

    def _next_phase2(self):
        """Phase 2 下一步"""
        if self._phase2_count >= self._phase2_target:
            self._finish_phase2()
            return

        # 定期重新生成候选
        if self._phase2_count % 5 == 0 or not self._phase2_candidates:
            self._phase2_candidates = self._generate_candidates()

        if not self._phase2_candidates:
            self._finish_phase2()
            return

        R_BH, t_BH, R_BS, pts_pred = self._phase2_candidates.pop(0)
        self._phase2_count += 1
        self.get_logger().info(
            f'\n  ── Phase2.{self._phase2_count}/{self._phase2_target}')
        self._move_to_pose(R_BH, t_BH, f'NBV #{self._phase2_count}')

    def _finish_phase2(self):
        """Phase 2 完成 → Phase 3"""
        n = len(self.records)
        self.get_logger().info(f'\n╚══ Phase 2 完成: {n} 位姿 ══╝')
        self._auto_phase = None
        self._run_solve()

    # ─── Phase 3: 求解 ──────────────────────────────────

    def _run_solve(self):
        """平面约束 LM 求解 (基于 plane_calib.py)"""
        self.get_logger().info('\n╔══ Phase 3: 求解 ══╗')

        poses = [(r['R_BH'], r['t_BH']) for r in self.records
                 if 'R_BH' in r]
        scans = [r['pts_S'] for r in self.records]
        if len(poses) < 10:
            self.get_logger().warn(f'位姿不足 ({len(poses)} < 10)')
            return

        R_he, t_he, info = solve_plane_he(
            poses, scans,
            self.R_he_nom, self.t_he_nom,
            n_restarts=20, verbose=True)

        if R_he is not None:
            dR = R_he.T @ self.R_he_gt
            tr = np.clip((np.trace(dR) - 1) / 2, -1, 1)
            R_err = np.rad2deg(np.arccos(tr))
            t_err = np.linalg.norm(t_he - self.t_he_gt) * 1000
            self.get_logger().info(
                f'\n{"="*50}\n'
                f'  R 误差: {R_err:.4f}°\n'
                f'  t 误差: {t_err:.4f} mm\n'
                f'  cost:   {info["best_cost"]:.2e}\n'
                f'{"="*50}')
            if R_err < 0.5:
                self.get_logger().info('  🎉 R < 0.5° — 达标!')
            elif R_err < 1.0:
                self.get_logger().info('  ✅ R < 1° — 接近')
            else:
                self.get_logger().warn(f'  ⚠ R > 1°')

    # ─── 键盘 ──────────────────────────────────────────

    def _keyboard_check(self):
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return
        k = sys.stdin.read(1)
        if k == 'a':
            self._start()
        elif k == 'c' and len(self.records) >= 6:
            self._run_solve()
        elif k == 's':
            for i, r in enumerate(self.records):
                t = r.get('t_BH', r['T_B_H'][:3, 3])
                self.get_logger().info(
                    f'  [{i}] ({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})')
        elif k == 'w':
            data = {
                'poses': [{
                    'R_i': r['R_BH'].tolist(),
                    't_i': r['t_BH'].tolist(),
                    'scan_pts_S': r['pts_S'].tolist(),
                } for r in self.records],
                'scene': {
                    'R_he_gt': self.R_he_gt.tolist(),
                    't_he_gt': self.t_he_gt.tolist(),
                },
            }
            path = os.path.expanduser('~/recorded_poses_v2.json')
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self.get_logger().info(f'已保存: {path}')
        elif k == 'j' and self.latest_joints is not None:
            jd = np.rad2deg(self.latest_joints)
            self.get_logger().info(
                f'  J={jd[0]:.1f} {jd[1]:.1f} {jd[2]:.1f}'
                f' {jd[3]:.1f} {jd[4]:.1f} {jd[5]:.1f}')
        elif k == 'q':
            raise SystemExit


def main():
    rclpy.init()
    n = AutoCalibV2Node()
    try:
        rclpy.spin(n)
    except Exception:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
