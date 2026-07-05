#!/usr/bin/env python3
"""
auto_calib_v2_node.py — 自动手眼标定采集节点

模式:
  'a' — Phase 0 锚点 → Phase 2 角点伺服 → Phase 3 求解
  'g' — Phase 0 锚点 → Phase 2 伺服 → Phase 2b Auto-Grid 倾斜 → Phase 3 求解

Phase 2 伺服: 1D X 轴比例控制, 锁定 ẽ→0 后做朝向探索 + 平移多样性。
Phase 2b Auto-Grid: 从伺服数据粗估板位姿 → pose_generator 生成倾斜候选 → 执行。
Phase 3: 三种求解器对比 + Gauge 自动诊断。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
from tf2_ros import Buffer, TransformListener
import numpy as np
import sys, os, json, select

sys.path.insert(0, '/workspace/common')
from plane_calib import (
    solve_plane_he, _unpack_plane_theta, _n_from_angles,
)
from fov_geometry import so3_exp, so3_log, rodrigues, rpy_to_matrix
from handeye_sim_bridge.fanuc_kinematic import inverse_kinematics, inverse_kinematics_numeric


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
    if len(points) < 10:
        return None, None, None
    w = np.asarray(weights) / np.sum(weights)
    centroid = np.average(points, axis=0, weights=w)
    centered = points - centroid
    cov = (centered.T * w) @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    n_B = eigvecs[:, 0]
    d = float(np.dot(n_B, centroid))
    return n_B, d, centroid


def detect_corners_from_profile(profile_pts, gap_threshold=0.02):
    """
    从传感器系 profile 点检测角点断点 e1, e2.

    线激光 profile 在传感器 XZ 平面内 (y≈0).
    激光打在板上的部分产生连续测量点; 板边以外无数据.
    按 X 排序后, 最左和最右的点即为 e1, e2.

    Args:
        profile_pts: (N,3) 传感器系点
        gap_threshold: 相邻点间 X 方向最大间距 [m]. 超过此值视为断点.

    Returns:
        e1, e2: 各为 (x,z) 或 None 如果未检测到.
        同时返回 has_two_edges: bool, 是否检测到 ≥2 段(即有角点特征).
    """
    if profile_pts is None or len(profile_pts) < 5:
        return None, None, False

    # 按 X 排序
    sorted_idx = np.argsort(profile_pts[:, 0])
    xs = profile_pts[sorted_idx, 0]
    zs = profile_pts[sorted_idx, 2]

    # 检测 X 方向的跳变 (gap)
    gaps = np.diff(xs)
    break_idx = np.where(gaps > gap_threshold)[0]

    if len(break_idx) == 0:
        # 连续段, 取两端点
        e1 = (float(xs[0]), float(zs[0]))
        e2 = (float(xs[-1]), float(zs[-1]))
        has_two = False  # 只有一段, 不一定是角点
        return e1, e2, has_two

    # 找到最长连续段
    segments = []
    start = 0
    for bi in break_idx:
        segments.append((start, bi))
        start = bi + 1
    segments.append((start, len(xs) - 1))

    # 最长段 (板上的点)
    seg_lens = [end - start + 1 for start, end in segments]
    main_idx = np.argmax(seg_lens)
    main_start, main_end = segments[main_idx]

    e1 = (float(xs[main_start]), float(zs[main_start]))
    e2 = (float(xs[main_end]), float(zs[main_end]))

    # 有两个及以上断点 → 至少 3 段 → 角点特征
    has_two = len(break_idx) >= 2

    return e1, e2, has_two


def compute_servo_signal(e1, e2):
    """计算角点偏移 ẽ = (e1.x + e2.x) / 2"""
    if e1 is None or e2 is None:
        return None
    return (e1[0] + e2[0]) / 2.0


# ─── 主节点 ────────────────────────────────────────────────

class AutoCalibV2Node(Node):
    def __init__(self):
        super().__init__('auto_calib_v2_node')

        self.R_he_gt = rpy_to_matrix(27.8, 9.2, -86.5)
        self.t_he_gt = np.array([-0.011579, -0.004621, 0.359284])

        rng = np.random.RandomState(12345)
        axis = rng.randn(3); axis /= np.linalg.norm(axis)
        angle = rng.normal(0, np.deg2rad(5))
        R_perturb = so3_exp(axis * angle)
        self.R_he_nom = R_perturb @ self.R_he_gt
        self.t_he_nom = self.t_he_gt + rng.randn(3) * 0.005

        self.get_logger().info(
            f'名义手眼 (真值+扰动): '
            f'R_err={np.rad2deg(np.linalg.norm(so3_log(self.R_he_nom.T @ self.R_he_gt))):.1f}° '
            f't_err={np.linalg.norm(self.t_he_nom - self.t_he_gt)*1000:.1f}mm')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PointCloud2, '/gocator/profile',
                                  self._profile_cb, 1)
        self.create_subscription(JointState, '/joint_states',
                                  self._joint_cb, 1)
        self.latest_profile = None
        self.latest_joints = None
        self.records = []

        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        # ─── Phase 0 锚点 ───
        self.Z_0 = None
        self.N_anchor = None
        self.R_BS_0 = None
        self.t_BS_0 = None

        # ─── 伺服数据 板估计 ───
        self.n_B = None
        self.d_plate = None
        self.C = None
        self.u_B = None
        self.v_B = None

        # ─── 状态机 ───
        self._state = 'IDLE'
        self._auto_phase = None
        pass  # 伺服数据 removed (v6 servo replaces it)
        self._auto_queue = []          # [(label, joints), ...]

        # ─── Phase 2 伺服 ───
        self._servo_count = 0           # 伺服迭代计数 (防死循环)
        self._servo_max = 12            # 最大伺服迭代 (含朝向重伺服)
        self._phase2_sub = None         # 'SERVO' | 'LOCKED' | 'ORIENT' | 'DIVERSITY'
        self._orient_queue = []         # [(R_tilt, label), ...] 待探索朝向
        self._locked_R_BH = None        # 锁定时的法兰姿态
        self._locked_t_BH = None        # 锁定时的法兰位置

        # ─── 伺服参数 ───
        self._servo_gain = 0.5          # 比例控制器增益 k
        self._lock_threshold = 0.002    # 锁定阈值 [m] (2mm)

        self.create_timer(0.2, self._state_machine)
        self.create_timer(0.1, self._keyboard_check)

        self.get_logger().info(
            "\n╔══════════════════════════════════════╗\n"
            "║  Auto Calib v6  角点伺服              ║\n"
            "║  'a' — 纯伺服  'g' — 伺服+AG  'q' — 退出   ║\n"
            "╚══════════════════════════════════════╝")

    # ═══════════════════════════════════════════════════════
    # 回调
    # ═══════════════════════════════════════════════════════

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
        tf = self._get_hand_pose()
        if tf is None:
            return None, None
        R_BH, t_BH = ros_tf_to_matrix(tf)
        return R_BH @ self.R_he_nom, t_BH + R_BH @ self.t_he_nom

    # ═══════════════════════════════════════════════════════
    # 特征提取 (含角点)
    # ═══════════════════════════════════════════════════════

    def _extract_features(self):
        """返回 dict: n_pts, Z_center, e1, e2, e_tilde, has_corner"""
        if self.latest_profile is None or len(self.latest_profile) < 3:
            return {'n_pts': 0, 'Z_center': None,
                    'e1': None, 'e2': None, 'e_tilde': None,
                    'has_corner': False}

        pts = self.latest_profile
        zs = pts[:, 2]
        e1, e2, has_two = detect_corners_from_profile(pts)

        return {
            'n_pts': len(pts),
            'Z_center': float(np.mean(zs)),
            'e1': e1,
            'e2': e2,
            'e_tilde': compute_servo_signal(e1, e2),
            'has_corner': has_two,
        }

    # ═══════════════════════════════════════════════════════
    # 关节移动
    # ═══════════════════════════════════════════════════════

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
        Tt = np.eye(4)
        Tt[:3, :3] = R_BH
        Tt[:3, 3] = t_BH
        # 使用数值 IK (基于正确 URDF FK), 从当前关节角出发
        q_init = self.latest_joints if self.latest_joints is not None else None
        qs = inverse_kinematics_numeric(Tt, q_init=q_init)
        if len(qs) == 0:
            self.get_logger().warn(f'  IK 失败: {label}')
            return False
        q = qs[0]  # 数值 IK 返回单一解
        self._auto_queue.append((label, q))
        return True

    # ═══════════════════════════════════════════════════════
    # 状态机
    # ═══════════════════════════════════════════════════════

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
                and self._auto_phase == 'PHASE2'):
            self._next_step()

    def _on_move_done(self):
        """运动完成后回调: 检查特征 → 按子状态处理"""
        f = self._extract_features()

        if self._auto_phase == 'PHASE2':
            self._on_phase2_move_done(f)
        elif self._auto_phase == 'AUTOGRID':
            # 记录当前扫描数据
            self._record_pose(f)
            self._queue_next_ag_pose()

    def _check_z_and_correct(self, f):
        """Z 深度修正: 仅当 Z 超出 Gocator 合理范围 (300~700mm) 时触发"""
        Z_MIN, Z_MAX, Z_TARGET = 0.300, 0.700, 0.500
        if f['Z_center'] is None or f['n_pts'] < 10:
            return False

        zc = f['Z_center']
        if Z_MIN <= zc <= Z_MAX:
            return False  # 在安全范围内, 不修正

        self._z_correction_count = getattr(self, '_z_correction_count', 0) + 1
        if self._z_correction_count > 3:
            self.get_logger().warn(f'  Z 修正迭代超限 (Z={zc*1000:.0f}mm), 放弃')
            return False

        dz = zc - Z_TARGET
        T = self._get_hand_pose()
        if T is None:
            return False
        R_BH, t_BH = ros_tf_to_matrix(T)
        zB = R_BH @ self.R_he_nom[:, 2]
        delta = zB * dz * 0.5  # 半增益, 避免过冲
        d_mag = np.linalg.norm(delta)
        if d_mag > 0.030:
            delta *= 0.030 / d_mag
        t_new = t_BH + delta
        self.get_logger().info(
            f'  Z修正 Z={zc*1000:.0f}→{Z_TARGET*1000:.0f}mm δ={d_mag*1000:.0f}mm')
        self._move_to_pose(R_BH, t_new, f'Z修正 {dz*1000:+.0f}mm')
        self._state = 'IDLE'
        self._scan_wait = 0
        return True

    # ═══════════════════════════════════════════════════════
    # Phase 2 伺服
    # ═══════════════════════════════════════════════════════

    def _on_phase2_move_done(self, f):
        """Phase 2 运动完成后的处理"""
        sub = self._phase2_sub

        # 诊断: 记录移动后的关节角和传感器位姿
        if self.latest_joints is not None:
            jd = np.rad2deg(self.latest_joints)
            self.get_logger().info(
                f'  [诊断] 关节={jd[0]:.1f}° {jd[1]:.1f}° {jd[2]:.1f}° '
                f'{jd[3]:.1f}° {jd[4]:.1f}° {jd[5]:.1f}°')
        T = self._get_hand_pose()
        if T:
            _, t_BH = ros_tf_to_matrix(T)
            self.get_logger().info(
                '  [诊断] 法兰位=(t:{:.3f},{:.3f},{:.3f}) n_pts={}'.format(
                    t_BH[0], t_BH[1], t_BH[2], f['n_pts']))

        if sub == 'SERVO':
            self._do_servo(f)
        elif sub == 'LOCKED':
            self._do_locked(f)
        elif sub == 'DIVERSITY':
            # 平移多样性位姿: 直接记录, 继续下一个偏移
            self._record_pose(f)
            self._try_next_diversity()
        elif sub == 'ORIENT':
            self._on_orient_done(f)
        else:
            # 默认进入伺服
            self._phase2_sub = 'SERVO'
            self._servo_count = 0
            self._z_correction_count = 0
            self._do_servo(f)

    def _do_servo(self, f):
        """角点 E 伺服: ẽ → δt = +k·ẽ·ŝ_x, 闭环至 |ẽ| ≤ 2mm"""
        # 先做 Z 修正
        if self._check_z_and_correct(f):
            return

        e_tilde = f['e_tilde']
        if e_tilde is None:
            # 丢失角点: 回到锚点关节角 (不走 IK, 避免臂构型变化)
            self._retreat_count = getattr(self, '_retreat_count', 0) + 1
            self.get_logger().warn(
                f'  伺服: 未检测到角点 (n_pts={f["n_pts"]}) '
                f'[回退 {self._retreat_count}/{self._servo_max}]')

            if self._retreat_count > self._servo_max:
                self.get_logger().error(
                    f'  回退超限, 放弃当前角点')
                self._retreat_count = 0
                self._finish_phase2()
                return

            if self._anchor_joints is not None:
                self.get_logger().info(
                    '  回到锚点关节角 (直接关节命令, 不走IK)')
                self._auto_queue.append(
                    ('回锚点关节', self._anchor_joints.copy()))
            elif self._locked_t_BH is not None:
                self.get_logger().info('  回到锁定位置 (IK)…')
                self._move_to_pose(
                    self._locked_R_BH, self._locked_t_BH,
                    '回到锁定位置')
            else:
                self.get_logger().error('  无参考位置可回退, 标定中止')
                self._auto_phase = None
            return

        # 角点恢复, 重置回退计数
        self._retreat_count = 0

        abs_e = abs(e_tilde)
        self.get_logger().info(
            f'  伺服: ẽ={e_tilde*1000:+.1f}mm (阈值±{self._lock_threshold*1000:.0f}mm)')

        if abs_e <= self._lock_threshold:
            # 锁定!
            self.get_logger().info(f'  ★ 锁定! (ẽ={e_tilde*1000:+.1f}mm)')
            self._servo_count = 0
            self._z_correction_count = 0
            self._phase2_sub = 'LOCKED'
            T = self._get_hand_pose()
            if T:
                self._locked_R_BH, self._locked_t_BH = ros_tf_to_matrix(T)
                # ── 倾斜自觉诊断 ──
                z_S = self._locked_R_BH @ self.R_he_nom[:, 2]  # 传感器 Z 轴在基座标系
                cos_theta = abs(np.dot(z_S, [0., 0., 1.]))  # vs 平板法向量 (板朝上 n_B=[0,0,1])
                tilt_deg = np.rad2deg(np.arccos(np.clip(cos_theta, 0, 1)))
                warn_str = "⚠ 接近垂直!" if tilt_deg < 5 else "✓ 安全"
                self.get_logger().info(
                    f'    倾斜: {tilt_deg:.1f}° (vs 平板法向), {warn_str}')
            self._do_locked(f)
            return

        # 未锁定: 计算修正
        self._servo_count += 1
        if self._servo_count > self._servo_max:
            self.get_logger().error(
                f'  伺服迭代超限 ({self._servo_count}次), 放弃当前角点')
            self._finish_phase2()
            return

        T = self._get_hand_pose()
        if T is None:
            return
        R_BH, t_BH = ros_tf_to_matrix(T)

        # ŝ_x = 传感器 X 轴在世界系 (用名义手眼)
        s_x = R_BH @ self.R_he_nom[:, 0]
        s_x = s_x / np.linalg.norm(s_x)

        # δt = +k · ẽ · ŝ_x
        # (传感器右移 → 角点相对左移 → ẽ 减小 → 闭环收敛)
        delta = self._servo_gain * e_tilde * s_x
        d_mag = np.linalg.norm(delta)
        if d_mag > 0.030:
            delta *= 0.030 / d_mag  # 单步上限 30mm

        t_new = t_BH + delta
        self._move_to_pose(R_BH, t_new,
                           f'伺服 ẽ={e_tilde*1000:+.1f}mm → δt={d_mag*1000:.1f}mm')

    def _do_locked(self, f):
        """锁定状态下: 记录 ẽ≈0 位姿, 然后采集平移多样性"""
        self._record_pose(f)  # ẽ≈0
        # 有意偏置: 在传感器 X 方向平移, 记录不同 ẽ 的位姿
        self._diversity_offsets = [0.015, -0.015, 0.030]  # m
        self._diversity_idx = 0
        self._phase2_sub = 'DIVERSITY'
        self._try_next_diversity()

    def _try_next_diversity(self):
        """采集下一个平移多样性位姿, 或进入朝向探索"""
        if self._diversity_idx >= len(self._diversity_offsets):
            self._phase2_sub = 'ORIENT'
            self._next_phase2_step()
            return
        offset = self._diversity_offsets[self._diversity_idx]
        self._diversity_idx += 1
        T = self._get_hand_pose()
        if T is None:
            self._try_next_diversity()
            return
        R_BH, t_BH = ros_tf_to_matrix(T)
        s_x = R_BH @ self.R_he_nom[:, 0]
        s_x = s_x / np.linalg.norm(s_x)
        delta = offset * s_x
        t_new = t_BH + delta
        self._move_to_pose(R_BH, t_new,
                           f'平移多样性 {offset*1000:+.0f}mm')

    def _next_phase2_step(self):
        """Phase 2 状态机: 伺服 → 锁定 → 朝向探索 → 移动角点"""
        if self._phase2_sub == 'SERVO':
            # 提取特征并开始伺服
            f = self._extract_features()
            self._do_servo(f)
        elif self._phase2_sub == 'ORIENT':
            self._try_next_orientation()

    def _try_next_orientation(self):
        """尝试下一个朝向探索"""
        if not self._orient_queue:
            # 朝向探索完毕 → 直接结束 Phase 2
            self._finish_phase2()
            return

        R_tilt, label = self._orient_queue.pop(0)
        self.get_logger().info(f'\n  ── 朝向: {label}')

        if self._locked_R_BH is None:
            self.get_logger().warn('  无锁定姿态, 跳过朝向')
            self._next_phase2_step()
            return

        # 应用朝向变化: R_new = R_tilt @ R_locked
        R_new = R_tilt @ self._locked_R_BH
        self._move_to_pose(R_new, self._locked_t_BH, f'朝向 {label}')

    def _on_orient_done(self, f):
        """朝向变化完成后: 重伺服 → 记录"""
        # 检查角点是否仍在 FOV
        if f['n_pts'] < 10:
            self.get_logger().warn(
                f'  朝向后激光丢失 (n_pts={f["n_pts"]}), 回到锁定姿态')
            if self._locked_R_BH is not None:
                self._move_to_pose(
                    self._locked_R_BH, self._locked_t_BH,
                    '回到锁定姿态')
                self._phase2_sub = 'ORIENT'  # 恢复后继续尝试
            return

        # 有数据: 先 Z 修正
        if self._check_z_and_correct(f):
            return

        # 检查是否需要重新伺服
        e_tilde = f['e_tilde']
        if e_tilde is not None and abs(e_tilde) > self._lock_threshold:
            # 朝向变化导致 ẽ 偏移, 重伺服
            self.get_logger().info(
                f'  朝向后 ẽ={e_tilde*1000:+.1f}mm, 重新伺服')
            self._phase2_sub = 'SERVO'
            self._servo_count = 0
            self._z_correction_count = 0
            self._do_servo(f)
        else:
            # 角点仍在锁定范围, 直接记录
            self._record_pose(f)
            # 更新锁定位置为当前位置
            T = self._get_hand_pose()
            if T:
                self._locked_R_BH, self._locked_t_BH = ros_tf_to_matrix(T)
            self._phase2_sub = 'ORIENT'
            self._next_phase2_step()

    # ═══════════════════════════════════════════════════════
    # 记录
    # ═══════════════════════════════════════════════════════

    def _record_pose(self, f=None):
        T = self._get_hand_pose()
        if T is None:
            return
        R_BH, t_BH = ros_tf_to_matrix(T)
        scan = (self.latest_profile.copy()
                if self.latest_profile is not None
                else np.zeros((0, 3)))
        rec = {
            'T_B_H': np.eye(4),
            'R_BH': R_BH, 't_BH': t_BH,
            'pts_S': scan,
        }
        # 保存 e1/e2 端点 (用于 12-DOF 求解器)
        if f and f.get('e1'):
            rec['p_S_e1'] = np.array([f['e1'][0], 0.0, f['e1'][1]])
            rec['valid_e1'] = True
        else:
            rec['valid_e1'] = False
        if f and f.get('e2'):
            rec['p_S_e2'] = np.array([f['e2'][0], 0.0, f['e2'][1]])
            rec['valid_e2'] = True
        else:
            rec['valid_e2'] = False
        self.records.append(rec)
        e1_str = ''
        e2_str = ''
        if rec['valid_e1']:
            e1_str = f' e1=({f["e1"][0]*1000:.0f},{f["e1"][1]*1000:.0f})'
        if rec['valid_e2']:
            e2_str = f' e2=({f["e2"][0]*1000:.0f},{f["e2"][1]*1000:.0f})'
        # 保存锁定时的传感器朝向（用于 Phase 2b 板位姿估计）
        if getattr(self, '_phase2_sub', None) in ('LOCKED', 'DIVERSITY'):
            rec['_z_S_locked'] = R_BH @ self.R_he_nom[:, 2]
        self.get_logger().info(
            f'  REC #{len(self.records)} pts={len(scan)}{e1_str}{e2_str}')

    # ═══════════════════════════════════════════════════════
    # Phase 0: 锚点
    # ═══════════════════════════════════════════════════════

    def _start(self):
        if self.latest_profile is None or len(self.latest_profile) < 5:
            self.get_logger().warn('初始位姿无效 (无 profile 数据)')
            return

        f = self._extract_features()
        if f['n_pts'] < 10:
            self.get_logger().error(f'初始位姿点数不足 (N={f["n_pts"]})')
            return

        # 验证角点: e1 和 e2 需同时存在
        if f['e1'] is None or f['e2'] is None:
            self.get_logger().warn(
                '未检测到 e1+e2 两个断点. '
                '请将激光线放到板角处, 确保同时切到两条边.')
            self.get_logger().info(
                f'  当前: e1={f["e1"]}, e2={f["e2"]}, '
                f'n_pts={f["n_pts"]}')
            return

        self.get_logger().info('\n╔══ Phase 0: 锚点 + 角点验证 ══╗')
        self.records = []
        self._servo_count = 0

        R_BS, t_BS = self._get_sensor_pose()
        self.R_BS_0 = R_BS
        self.t_BS_0 = t_BS
        self.Z_0 = f['Z_center']
        self.N_anchor = f['n_pts']
        self._anchor_joints = self.latest_joints.copy() if self.latest_joints is not None else None
        # 保存锚点法兰位姿 (用于角点移动基准)
        T = self._get_hand_pose()
        if T:
            self._anchor_R_BH, self._anchor_t_BH = ros_tf_to_matrix(T)
        else:
            self._anchor_R_BH, self._anchor_t_BH = None, None

        e_tilde_str = f'{f["e_tilde"]*1000:+.1f}mm' if f['e_tilde'] else '?'
        self.get_logger().info(
            f'  Z0={self.Z_0*1000:.0f}mm  N_anchor={self.N_anchor}'
            f'  ẽ={e_tilde_str}')

        self._record_pose(f)
        self.get_logger().info('\n╔══ Phase 2: 角点伺服采集 ══╗')
        self._build_phase2()
        self._auto_phase = 'PHASE2'

    def _next_step(self):
        if self._auto_phase == 'PHASE2':
            self._next_phase2_step()

    # ═══════════════════════════════════════════════════════
    # Phase 2: 角点伺服采集
    # ═══════════════════════════════════════════════════════

    def _build_phase2(self):
        """构建朝向探索队列和角点移动队列"""
        self.get_logger().info('\n╔══ Phase 2: 角点伺服采集 ══╗')

        # ── 朝向变体 (传感器坐标系) ──
        deg = np.deg2rad
        self._orient_queue = []

        # 绕 Z 轴 (roll): 激光面绕发射方向旋转, 角点基本不丢
        for ang in [5, -5, 8, -8, 12, -12]:
            self._orient_queue.append(
                (rodrigues(np.array([0., 0., 1.]), deg(ang)),
                 f'Z{ang:+d}°'))

        # 绕 X 轴 (pitch): 小角度, 需重伺服
        for ang in [5, -5, 8, -8]:
            self._orient_queue.append(
                (rodrigues(np.array([1., 0., 0.]), deg(ang)),
                 f'X{ang:+d}°'))

        # 绕 Y 轴 (yaw): 小角度, 需重伺服
        for ang in [5, -5, 8, -8]:
            self._orient_queue.append(
                (rodrigues(np.array([0., 1., 0.]), deg(ang)),
                 f'Y{ang:+d}°'))

        self.get_logger().info(
            f'  朝向变体: {len(self._orient_queue)} 个')

        # 初始化: 锚点角点直接进入伺服
        self._phase2_sub = 'SERVO'
        self._servo_count = 0
        T = self._get_hand_pose()
        if T:
            self._locked_R_BH, self._locked_t_BH = ros_tf_to_matrix(T)

    def _finish_phase2(self):
        n = len(self.records)
        self.get_logger().info(f'\\n╚══ Phase 2 完成: {n} 位姿 ══╝')

        if getattr(self, '_use_autogrid', False) and len(self.records) >= 3:
            self._start_autogrid()
        else:
            self._auto_phase = None
            self._run_solve()

    def _start_autogrid(self):
        """从伺服数据粗估板位姿 → auto_grid 生成倾斜候选 → 执行"""
        self.get_logger().info('\\n╔══ Phase 2b: Auto-Grid 倾斜采集 ══╗')

        # 1. 从伺服数据估算板方向
        meas = self._assign_edges_for_12dof()
        n_B, u_B, v_B, C_est = self._estimate_plate_from_data(meas)
        if n_B is None:
            self.get_logger().warn('  板估计失败, 回退到纯伺服')
            self._auto_phase = None
            self._run_solve()
            return

        self.get_logger().info(
            f'  估计: C=({C_est[0]:.2f},{C_est[1]:.2f},{C_est[2]:.2f}) '
            f'n_B=({n_B[0]:.2f},{n_B[1]:.2f},{n_B[2]:.2f})')

        # 2. 生成 auto_grid 候选（用名义手眼 + 估计板位姿）
        try:
            sys.path.insert(0, '/workspace/common')
            from pose_generator import generate_tilted_poses
        except ImportError:
            self.get_logger().warn('  pose_generator 不可用')
            self._auto_phase = None
            self._run_solve()
            return

        poses_ag, cands = generate_tilted_poses(
            C_est, n_B, u_B, v_B,
            self.R_he_nom, self.t_he_nom,
            plate_w=400, plate_h=500,
            n_poses=8, seed=None, return_candidates=True)

        if len(poses_ag) == 0:
            self.get_logger().warn('  auto_grid 未找到有效候选')
            self._auto_phase = None
            self._run_solve()
            return

        z_dev_str = ', '.join(f'{c["z_dev"]:.0f}°' for c in cands[:5])
        self.get_logger().info(
            f'  生成 {len(poses_ag)} 个候选 (z_dev: {z_dev_str}...)')

        # 3. 入队执行
        self._ag_poses = list(poses_ag)
        self._ag_idx = 0
        self._auto_phase = 'AUTOGRID'
        self._queue_next_ag_pose()

    def _estimate_plate_from_data(self, meas):
        """从伺服采集数据估算板位姿 (C, n_B, u_B, v_B)
        
        混合策略：n_B 从锁定位置的传感器朝向估计（稳定），
        u_B/v_B 从端点 PCA 估计，C 从边线交点。
        """
        # ── n_B: 从锁定位置的传感器 Z 轴推算 ──
        # 锁定后 ẽ≈0 → 传感器大致朝向平板 → n_B ≈ -z_S
        z_S_samples = []
        for m in meas:
            if m.get('_z_S_locked') is not None:
                z_S_samples.append(m['_z_S_locked'])
        if len(z_S_samples) >= 2:
            z_S_avg = np.mean(z_S_samples, axis=0)
            z_S_avg /= np.linalg.norm(z_S_avg) + 1e-12
            n_B = -z_S_avg  # 平板法向量 ≈ 传感器朝向的反方向
        else:
            # 回退：用平面点 PCA
            plane_pts = []
            for m in meas:
                if '_p_B_plane' in m:
                    plane_pts.extend(m['_p_B_plane'])
            if len(plane_pts) >= 10:
                plane_pts = np.array(plane_pts)
                c = plane_pts.mean(0)
                _, ev = np.linalg.eigh((plane_pts-c).T @ (plane_pts-c)/len(plane_pts))
                n_B = ev[:, 0]; n_B /= np.linalg.norm(n_B)
            else:
                return None, None, None, None

        # ── u_B, v_B: 从端点 PCA ──
        p_be1, p_be2 = [], []
        for m in meas:
            p1 = m.get('_p_B_e1'); p2 = m.get('_p_B_e2')
            if p1 is not None: p_be1.append(p1)
            if p2 is not None: p_be2.append(p2)
        if len(p_be1) < 3 or len(p_be2) < 3:
            return None, None, None, None
        p_be1 = np.array(p_be1); p_be2 = np.array(p_be2)
        u_B = np.linalg.svd(p_be1 - p_be1.mean(0))[2][0]
        v_B = np.linalg.svd(p_be2 - p_be2.mean(0))[2][0]
        # 投影到 n_B 的法平面 + 正交化
        u_B -= np.dot(u_B, n_B) * n_B
        u_B /= np.linalg.norm(u_B) + 1e-12
        v_B = np.cross(n_B, u_B)
        v_B /= np.linalg.norm(v_B) + 1e-12

        # ── C: 两条边线的交点 ──
        def skew(v):
            return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        A = np.vstack([skew(u_B), skew(v_B), n_B.reshape(1,3)])
        b = np.hstack([skew(u_B)@p_be1.mean(0), skew(v_B)@p_be2.mean(0),
                       [np.dot(n_B, (p_be1.mean(0)+p_be2.mean(0))/2)]])
        C_est = np.linalg.lstsq(A, b, rcond=None)[0]

        return n_B, u_B, v_B, C_est

    def _queue_next_ag_pose(self):
        while self._ag_idx < len(self._ag_poses):
            R_i, t_i = self._ag_poses[self._ag_idx]
            self.get_logger().info(
                f'\n  ── AG #{self._ag_idx+1}/{len(self._ag_poses)} ──')
            ok = self._move_to_pose(R_i, t_i, f'AG#{self._ag_idx+1}')
            self._ag_idx += 1
            if ok:
                return  # 成功入队, 等运动完成
            self.get_logger().warn(f'  IK 失败, 跳过 AG#{self._ag_idx}')
        # 全部跳过或已完成
        self.get_logger().info(f'\n╚══ Auto-Grid 完成 ══╝')
        self._auto_phase = None
        self._run_solve()

    # ═══════════════════════════════════════════════════════
    # Phase 3: 求解
    # ═══════════════════════════════════════════════════════

    def _run_solve(self):
        self.get_logger().info('\n╔══ Phase 3: 三种求解器对比 ══╗')
        poses = [(r['R_BH'], r['t_BH']) for r in self.records
                 if 'R_BH' in r]
        n = len(poses)
        if n < 4:
            self.get_logger().warn(f'位姿不足 ({n} < 4)')
            return

        meas = self._assign_edges_for_12dof()
        n_e1 = sum(1 for m in meas if m.get('valid_e1'))
        n_e2 = sum(1 for m in meas if m.get('valid_e2'))
        self.get_logger().info(f'  位姿={n}  有e1={n_e1}  有e2={n_e2}')

        from calib_solver import (
            solve_12dof_with_restarts,
            solve_principle_with_restarts,
            solve_principle_12dof_with_restarts,
        )

        results = []

        # --- M1: cross-product ---
        self.get_logger().info('\n  -- M1: cross-product 12-DOF --')
        t1, i1 = solve_12dof_with_restarts(poses, meas, n_restarts=20, verbose=True)
        if t1 is not None:
            R1 = so3_exp(t1[0:3]); dR = R1.T @ self.R_he_gt
            re = np.rad2deg(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
            te = np.linalg.norm(t1[3:6] - self.t_he_gt)*1000
            results.append(('M1 cross-product', re, te, i1['best_cost']))

        # --- M2: scalar ---
        self.get_logger().info('\n  -- M2: 标量边约束 --')
        t2, i2 = solve_principle_with_restarts(poses, meas, n_restarts=20, verbose=True)
        if t2 is not None:
            R2 = so3_exp(t2[0:3]); dR = R2.T @ self.R_he_gt
            re = np.rad2deg(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
            te = np.linalg.norm(t2[3:6] - self.t_he_gt)*1000
            results.append(('M2 scalar', re, te, i2['best_cost']))

        # --- M3: PRINCIPLE.md sensor-frame + gauge ---
        self.get_logger().info('\n  -- M3: 传感器帧预测 + gauge固定 --')
        t3, i3 = solve_principle_12dof_with_restarts(
            poses, meas, self.R_he_nom, self.t_he_nom,
            n_restarts=20, verbose=True)
        if t3 is not None:
            R3 = so3_exp(t3[0:3]); dR = R3.T @ self.R_he_gt
            re = np.rad2deg(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
            te = np.linalg.norm(t3[3:6] - self.t_he_gt)*1000
            results.append(('M3 PRINCIPLE.md', re, te, i3['best_cost']))

        # --- 对比 ---
        self.get_logger().info(
            '\n' + '='*60 + '\n'
            f'  {"求解器":<27s} {"R误差":>8s} {"t误差":>10s} {"cost":>10s}\n'
            f'  {"-"*55}')
        for name, re, te, co in results:
            flag = '🎉' if re < 0.5 else '✅' if re < 1.0 else '⚠'
            self.get_logger().info(
                f'  {flag} {name:<24s} {re:>7.4f}° {te:>9.4f}mm {co:>10.2e}')
        self.get_logger().info('='*60)

        # ── Gauge 自动检测 ──
        self._diagnose_gauge(poses, meas, results)

    def _diagnose_gauge(self, poses, meas, results):
        """对最优解的 Jacobian 做 SVD，诊断 gauge 状态"""
        if not results: return
        # 用 M1 (cross-product) 的解做 Jacobian 分析
        from calib_solver import residuals_12dof, so3_exp
        # M1 uses 12 params: w_he(3), t_he(3), w_pl(3), C(3)
        # We need the optimal theta from M1 — re-extract from solve result
        t1_opt, _ = solve_12dof_with_restarts(poses, meas, n_restarts=1, verbose=False)
        if t1_opt is None:
            self.get_logger().info('  [Gauge] 无法获取 M1 最优解, 跳过')
            return

        # 数值 Jacobian
        r0, mask = residuals_12dof(t1_opt, poses, meas)
        rv = r0[mask]
        eps = 1e-6; J = np.zeros((len(r0), 12))
        for j in range(12):
            tp = t1_opt.copy(); tp[j] += eps
            tm = t1_opt.copy(); tm[j] -= eps
            rp, _ = residuals_12dof(tp, poses, meas)
            rm, _ = residuals_12dof(tm, poses, meas)
            J[:, j] = (rp - rm) / (2 * eps)
        Jv = J[mask, :]
        try:
            S = np.linalg.svd(Jv, compute_uv=False)
            cond = S[0] / S[-1] if S[-1] > 1e-15 else float('inf')
            n_zero = int(np.sum(S < 1e-8))
            gauge_status = '✗ 存在' if n_zero > 0 else '✓ 消失'
            self.get_logger().info(
                f'\n  ── Gauge 诊断 ──\n'
                f'  cond(J)={cond:.2e}  σ_min={S[-1]:.2e}  σ_min/σ_max={S[-1]/S[0]:.2e}\n'
                f'  零奇异值={n_zero} → Gauge: {gauge_status}')
            if n_zero > 0:
                self.get_logger().warn(
                    f'  ⚠ 存在 {n_zero} 维 gauge! 数据可能退化 —— '
                    f'检查传感器是否垂直照射 (z_S ∥ -n_B)')
            # 位姿多样性诊断
            z_devs = []
            for R_i, _ in poses:
                z_S = R_i @ self.R_he_nom[:, 2]  # sensor Z in base
                ang = np.rad2deg(np.arccos(np.clip(np.dot(z_S, [0., 0., -1.]), -1, 1)))
                z_devs.append(ang)
            z_devs = np.array(z_devs)
            if len(z_devs) > 1:
                self.get_logger().info(
                    f'  z_S 偏离 -n_B: min={z_devs.min():.1f}°  max={z_devs.max():.1f}°  '
                    f'mean={z_devs.mean():.1f}°  std={z_devs.std():.1f}°')
                if z_devs.std() < 3.0:
                    self.get_logger().warn(
                        f'  ⚠ z_S 多样性不足 (std={z_devs.std():.1f}° < 3°) '
                        f'→ 增加 tilt 可改善 gauge')
        except np.linalg.LinAlgError:
            self.get_logger().warn('  [Gauge] SVD 失败')

    def _compute_plate_directions_from_endpoints(self, meas):
        """从所有端点数据的基座标系运动 PCA 计算 u_B, v_B
        
        从端点差分运动方向计算。
        端点沿边的运动在各 bit 位姿间形成可检测的方向模式。
        """
        all_ep = []
        for m in meas:
            if not (m.get('valid_e1') and m.get('valid_e2')): continue
            if m.get('p_S_e1') is None or m.get('p_S_e2') is None: continue
            all_ep.append(m.get('_p_B_e1')); all_ep.append(m.get('_p_B_e2'))
        all_ep = [p for p in all_ep if p is not None]
        if len(all_ep) < 3:
            # 回退: 用任意正交基
            n_ref = np.array([0., 0., 1.])
            u_B = np.array([1., 0., 0.]) if abs(n_ref[2]) < 0.9 else np.array([0., 1., 0.])
            v_B = np.cross(n_ref, u_B)
            return u_B, v_B
        ep = np.array(all_ep)
        c = np.mean(ep, axis=0)
        _, _, Vt = np.linalg.svd(ep - c, full_matrices=False)
        # 前两个主成分是面内方向 (端点主要在板面内移动)
        d1 = Vt[0] / np.linalg.norm(Vt[0])
        d2 = Vt[1] / np.linalg.norm(Vt[1])
        # d1, d2 形成面内基, n = d1 x d2
        n = np.cross(d1, d2)
        if np.linalg.norm(n) < 1e-6:
            return d1, np.cross(np.array([0.,0.,1.]), d1)
        n /= np.linalg.norm(n)
        self.n_B = n
        return d1, d2

    def _assign_edges_for_12dof(self):
        """用端点 PCA 方向分配边归属, 不依赖 伺服数据"""
        # 第一步: 将所有端点投影到基座标系
        meas = []
        for r in self.records:
            m = {
                'p_S_plane': r['pts_S'],
                'valid_e1': r.get('valid_e1', False),
                'valid_e2': r.get('valid_e2', False),
                'p_S_e1': r.get('p_S_e1'),
                'p_S_e2': r.get('p_S_e2'),
                '_p_B_e1': None, '_p_B_e2': None,
                '_z_S_locked': r.get('_z_S_locked'),
            }
            if m['valid_e1'] and m['p_S_e1'] is not None:
                R_BS = r['R_BH'] @ self.R_he_nom
                t_BS = r['t_BH'] + r['R_BH'] @ self.t_he_nom
                m['_p_B_e1'] = R_BS @ m['p_S_e1'] + t_BS
            if m['valid_e2'] and m['p_S_e2'] is not None:
                R_BS = r['R_BH'] @ self.R_he_nom
                t_BS = r['t_BH'] + r['R_BH'] @ self.t_he_nom
                m['_p_B_e2'] = R_BS @ m['p_S_e2'] + t_BS
            meas.append(m)

        # 第二步: 从端点 PCA 计算 u_B, v_B
        u_B, v_B = self._compute_plate_directions_from_endpoints(meas)
        self.u_B = u_B; self.v_B = v_B

        # 第三步: 用 u_B/v_B 分配边归属
        for m in meas:
            if m['valid_e1'] and m['valid_e2'] and m['_p_B_e1'] is not None:
                pA, pB = m['_p_B_e1'], m['_p_B_e2']
                dAu = abs(np.dot(pA, u_B)); dAv = abs(np.dot(pA, v_B))
                dBu = abs(np.dot(pB, u_B)); dBv = abs(np.dot(pB, v_B))
                if (dAu - dAv) <= (dBu - dBv):
                    m['p_S_e1'], m['p_S_e2'] = m['p_S_e2'], m['p_S_e1']
        return meas

    # ═══════════════════════════════════════════════════════
    # 键盘
    # ═══════════════════════════════════════════════════════

    def _keyboard_check(self):
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return
        k = sys.stdin.read(1)
        if k == 'a':
            self._use_autogrid = False
            self._start()
        elif k == 'g':
            self._use_autogrid = True
            self._start()
        elif k == 'c' and len(self.records) >= 4:
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
