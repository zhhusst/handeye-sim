#!/usr/bin/env python3
"""auto_calib_v2_node.py v29 — single-edge peristaltic for e1 + e2.
User provides two starting joint poses → 
  Phase 1: e1 peristaltic (rotate → check single breakpoint → servo → record)
  Phase 2: e2 peristaltic (same)
  Phase 3: combined_solve_lm + iterative_refine_he + tilted_corner"""

import rclpy, numpy as np, sys, json
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, '/workspace/common')
from fov_geometry import so3_exp, so3_log, rodrigues, rpy_to_matrix
from handeye_sim_bridge.fanuc_kinematic import inverse_kinematics_numeric

# ═══════════════════════════════════════════════════════════════
# USER: set these two starting joint poses
#   E1: laser line crosses u_B edge (single breakpoint = e1)
#   E2: laser line crosses v_B edge (single breakpoint = e2)
# ═══════════════════════════════════════════════════════════════
E1_START_JOINTS = np.array([-0.061, -0.353, -0.530, -0.038, -0.889, 0.078])  # TODO: fill in
E2_START_JOINTS = np.array([-0.019, -0.760, -1.059, -0.679, -0.940, 2.117])

N_B_GT = np.array([0., 0., 1.])
U_B_GT = np.array([1., 0., 0.])
V_B_GT = np.array([0., 1., 0.])


def ros_tf_to_matrix(t):
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])
    t_vec = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
    return R, t_vec


def detect_corners(pts, gap=0.02):
    if pts is None or len(pts) < 5:
        return None, None, False
    si = np.argsort(pts[:, 0])
    xs, zs = pts[si, 0], pts[si, 2]
    bi = np.where(np.diff(xs) > gap)[0]
    if len(bi) == 0:
        return (float(xs[0]), float(zs[0])), (float(xs[-1]), float(zs[-1])), False
    segs = []
    s = 0
    for b in bi:
        segs.append((s, b))
        s = b + 1
    segs.append((s, len(xs)-1))
    mi = np.argmax([e-s+1 for s, e in segs])
    ms, me = segs[mi]
    return (float(xs[ms]), float(zs[ms])), (float(xs[me]), float(zs[me])), len(bi) >= 2


def fit_line_3d(pts):
    if len(pts) < 2:
        return None, None
    p = np.array(pts)
    c = np.mean(p, axis=0)
    _, _, vh = np.linalg.svd(p - c, full_matrices=False)
    return vh[0], c


def angle_between(a, b):
    a = np.asarray(a) / np.linalg.norm(a)
    b = np.asarray(b) / np.linalg.norm(b)
    return np.rad2deg(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1)))


class AutoCalibV2Node(Node):
    def __init__(self):
        super().__init__('auto_calib_v2_node')
        self.R_he_gt = rpy_to_matrix(27.8, 9.2, -86.5)
        self.t_he_gt = np.array([-0.011579, -0.004621, 0.359284])
        rng = np.random.RandomState(12345)
        ax = rng.randn(3); ax /= np.linalg.norm(ax)
        R_perturb = so3_exp(ax * rng.normal(0, np.deg2rad(5)))
        self.R_he_nom = R_perturb @ self.R_he_gt
        self.t_he_nom = self.t_he_gt + rng.randn(3) * 0.005
        self.get_logger().info(
            f'R_he_nom err={np.rad2deg(np.linalg.norm(so3_log(self.R_he_nom.T @ self.R_he_gt))):.1f}deg '
            f't_err={np.linalg.norm(self.t_he_nom - self.t_he_gt)*1000:.1f}mm')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(PointCloud2, '/gocator/profile', self._pc_cb, 1)
        self.create_subscription(JointState, '/joint_states', self._js_cb, 1)
        self.latest_profile = None
        self.latest_joints = None

        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        self.records = []
        self._auto_mode = False
        self._auto_queue = []
        self._state = 'IDLE'
        self._scan_wait = 0

        self._pl_state = 'IDLE'
        self._pl_step = 0
        self._pl_steps = []

        self.R_he_true = None
        self.t_he_true = None

        # Peristaltic state (single-edge)
        self._peri_phase = None          # 'e1' or 'e2'
        self._peri_current_joints = None
        self._peri_last_valid_joints = None
        self._peri_last_R_BH = None
        self._peri_frame_count = 0
        self._peri_target_count = 12      # frames per edge
        self._peri_dir_idx = 0
        self._peri_consecutive_fail = 0
        self._peri_bases = []
        self._peri_base_idx = 0
        self._peri_start_joints = None
        self._peri_in_recovery = False  # skip recording when arriving at recovery pose
        self._peri_max_fail = 8
        self._peri_servo_count = 0

        self._peri_max_servo = 3

        self.create_timer(0.2, self._state_machine)
        self.get_logger().info("v29 single-edge peristaltic — e1 + e2 separate")

    def _pc_cb(self, msg):
        from sensor_msgs_py.point_cloud2 import read_points
        try:
            pts = [list(p) for p in read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)]
            self.latest_profile = np.array(pts, dtype=np.float64) if pts else None
        except Exception:
            pass

    def _js_cb(self, msg):
        JN = ['J1_joint', 'J2_joint', 'J3_joint', 'J4_joint', 'J5_joint', 'J6_joint']
        try:
            self.latest_joints = np.array(
                [msg.position[msg.name.index(j)] for j in JN])
        except Exception:
            pass

    def _get_hand_pose(self):
        """Get flange pose. TF first, FK from joint_states as fallback."""
        try:
            return self.tf_buffer.lookup_transform('world', 'fanuc_flange', rclpy.time.Time())
        except Exception:
            pass
        # FK fallback: compute flange pose from joint_states
        if self.latest_joints is not None:
            try:
                from handeye_sim_bridge.fanuc_kinematic import forward_kinematics
                T = forward_kinematics(self.latest_joints)
                # Convert 4x4 numpy array to TransformStamped-like object
                from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion
                ts = TransformStamped()
                ts.transform.translation.x = float(T[0, 3])
                ts.transform.translation.y = float(T[1, 3])
                ts.transform.translation.z = float(T[2, 3])
                # Rotation matrix to quaternion
                R = T[:3, :3]
                w = np.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2
                x = np.sqrt(max(0, 1 + R[0,0] - R[1,1] - R[2,2])) / 2
                y = np.sqrt(max(0, 1 - R[0,0] + R[1,1] - R[2,2])) / 2
                z = np.sqrt(max(0, 1 - R[0,0] - R[1,1] + R[2,2])) / 2
                x = np.copysign(x, R[2,1] - R[1,2])
                y = np.copysign(y, R[0,2] - R[2,0])
                z = np.copysign(z, R[1,0] - R[0,1])
                ts.transform.rotation.x = float(x)
                ts.transform.rotation.y = float(y)
                ts.transform.rotation.z = float(z)
                ts.transform.rotation.w = float(w)
                return ts
            except Exception:
                pass
        return None

    def _get_sensor_pose_tf(self):
        try:
            tf = self.tf_buffer.lookup_transform('world', 'gocator_sensor', rclpy.time.Time())
            return ros_tf_to_matrix(tf)
        except Exception:
            tf = self._get_hand_pose()
            if tf is None:
                return None, None
            R_BH, t_BH = ros_tf_to_matrix(tf)
            return R_BH @ self.R_he_nom, t_BH + R_BH @ self.t_he_nom

    def _extract_features(self):
        if self.latest_profile is None or len(self.latest_profile) < 3:
            return {'n_pts': 0, 'e1': None, 'e2': None, 'has_corner': False, 'pts_S': np.array([])}
        pts = self.latest_profile
        e1, e2, h2 = detect_corners(pts)
        return {'n_pts': len(pts), 'e1': e1, 'e2': e2, 'has_corner': h2, 'pts_S': pts}

    def _send_joint_target(self, joints, label=''):
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'J1_joint', 'J2_joint', 'J3_joint', 'J4_joint', 'J5_joint', 'J6_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in joints]
        pt.time_from_start = Duration(sec=2, nanosec=0)
        goal.trajectory.points = [pt]
        self._state = 'MOVING'
        if not self._traj_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('FollowJointTrajectory action server not available!')
            self._state = 'IDLE'
            return
        self._traj_client.send_goal_async(goal).add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, f):
        h = f.result()
        if not h.accepted:
            self.get_logger().warn('goal rejected')
            self._state = 'IDLE'
            return
        h.get_result_async().add_done_callback(lambda _: setattr(self, '_state', 'WAIT'))

    def _move_to_pose(self, R_BH, t_BH, label=''):
        Tt = np.eye(4)
        Tt[:3, :3] = R_BH
        Tt[:3, 3] = t_BH
        qs = inverse_kinematics_numeric(Tt, q_init=self.latest_joints)
        if len(qs) == 0:
            self.get_logger().warn(f'  IK FAIL: {label}')
            return False
        self._auto_queue.append((label, qs[0]))
        return True

    def _record_pose(self, f):
        T = self._get_hand_pose()
        if T is None:
            return
        R_BH, t_BH = ros_tf_to_matrix(T)
        R_BS, t_BS = self._get_sensor_pose_tf()
        rec = {'R_BH': R_BH, 't_BH': t_BH, 'R_BS': R_BS, 't_BS': t_BS,
               'n_pts': f['n_pts']}
        if self.latest_joints is not None:
            rec['J_i'] = self.latest_joints.copy().tolist()
        if f.get('e1'):
            rec['p_S_e1'] = np.array([f['e1'][0], 0.0, f['e1'][1]])
            rec['valid_e1'] = True
        else:
            rec['valid_e1'] = False
        if f.get('e2'):
            rec['p_S_e2'] = np.array([f['e2'][0], 0.0, f['e2'][1]])
            rec['valid_e2'] = True
        else:
            rec['valid_e2'] = False
        rec['pts_S'] = np.array(f.get('pts_S', []))
        self.records.append(rec)

    def _state_machine(self):
        if not hasattr(self, '_auto_phase'):
            return
        if self._state == 'IDLE' and self._auto_queue:
            label, joints = self._auto_queue.pop(0)
            self.get_logger().info(f'  -> {label}')
            self._send_joint_target(joints, label)
        if self._state == 'WAIT':
            self._scan_wait += 1
            if self._scan_wait >= 6:
                self._state = 'IDLE'
                self._scan_wait = 0
                self._on_move_done()
        if self._state == 'IDLE' and not self._auto_queue and self._pl_state not in ('IDLE', 'DONE'):
            self._pl_next()

    def _on_move_done(self):
        if self._pl_state == 'PHASE1_PERI':
            self._peri_on_arrival()
            return
        f = self._extract_features()
        if f and f['n_pts'] >= 5:
            self._record_pose(f)
        else:
            self.get_logger().warn(f'  no pts: n_pts={f["n_pts"] if f else 0}')
        self._pl_step += 1
        self._pl_next()

    def _auto_start_once(self):
        if self._auto_mode:
            return
        self._auto_mode = True
        # v41: Start pipeline directly — it first moves to E1_START_JOINTS
        # Profile check happens naturally once the robot is in position
        self.get_logger().info('--auto: starting pipeline...')
        self._delayed_timer = self.create_timer(1.0, self._pipeline_start)

    # ========================================================================
    # PIPELINE
    # ========================================================================

    def _pipeline_start(self):
        self.destroy_timer(self._delayed_timer)
        self.get_logger().info('\n' + '=' * 50 + '\n  v29 Single-Edge Peristaltic\n' + '=' * 50)
        self._auto_phase = 'PIPELINE'

        # Get R_he_true from TF (fallback to R_he_nom if TF not ready yet)
        R_BS_a, t_BS_a = self._get_sensor_pose_tf()
        if R_BS_a is None:
            self.get_logger().warn('no sensor TF yet, using R_he_nom for R_he_true')
            R_BS_a = self.R_he_nom
            t_BS_a = np.zeros(3)
        ht = self._get_hand_pose()
        if ht is None:
            # TF not ready yet — retry in 1s
            self.get_logger().warn('no hand TF yet, retrying...')
            self._delayed_timer = self.create_timer(1.0, self._pipeline_start)
            return
        R_BH_a, t_BH_a = ros_tf_to_matrix(ht)
        self.R_he_true = R_BH_a.T @ R_BS_a
        self.t_he_true = R_BH_a.T @ (t_BS_a - t_BH_a)

        err_R = np.rad2deg(np.linalg.norm(so3_log(self.R_he_true.T @ self.R_he_gt)))
        self.get_logger().info(f'  R_he_true vs R_he_gt: {err_R:.2f}deg')

        # Start e1 peristaltic
        self._start_peristaltic('e1')

    def _pl_next(self):
        pass  # unused in v29: peristaltic drives its own state machine

    # ========================================================================
    # SINGLE-EDGE PERISTALTIC
    # ========================================================================

    def _start_peristaltic(self, edge):
        """edge: 'e1' or 'e2'"""
        self._peri_phase = edge
        start_joints = E1_START_JOINTS if edge == 'e1' else E2_START_JOINTS
        label = f'{edge} start'

        self.get_logger().info(f'\n{"="*50}\n  Phase: {edge} peristaltic\n{"="*50}')
        self.get_logger().info(f'  Start joints: [{", ".join(f"{j:.3f}" for j in start_joints)}]')

        self._peri_last_valid_joints = start_joints.copy()
        self._peri_current_joints = start_joints.copy()
        self._peri_start_joints = start_joints.copy()
        self._peri_last_R_BH = None
        self._peri_first_R_BH = None   # anchor for cumulative angular distance
        self._peri_frame_count = 0
        self._peri_dir_idx = 0
        self._peri_base_idx = 0
        self._peri_consecutive_fail = 0
        self._peri_bases = [start_joints.copy()]  # base 0 = initial start pose

        # v44: Continuous rotation directions — keep rotating in axis until boundary or target
        # (axis_idx, label) — rotation continues adaptively, no fixed angle
        self._peri_directions = [
            (0, 'SX'), (1, 'SY'), (2, 'SZ'),
        ]
        self._peri_rot_step = 0.017   # ~1° per step
        self._peri_rot_target = np.deg2rad(15)  # max 15° per direction
        self._peri_rot_accum = 0.0
        self._peri_mode = 'arrive'    # 'arrive' | 'rotate' | 'servo'
        self._peri_last_npts = 999
        self._peri_last_recorded_R = None

        self._pl_state = 'PHASE1_PERI'
        self._send_joint_target(start_joints, f'{edge}_start')

    def _peri_send_next(self):
        """v44: continuous rotation + adaptive servo loop"""
        if self._peri_frame_count >= self._peri_target_count:
            self.get_logger().info(f'  {self._peri_phase} done: {self._peri_frame_count} frames')
            self._peri_switch_phase()
            return

        # Check for exhausted directions → advance base or finish
        if self._peri_dir_idx >= len(self._peri_directions):
            next_base = self._peri_base_idx + 1
            if hasattr(self, '_peri_bases') and next_base < len(self._peri_bases):
                self.get_logger().info(
                    f'  >>> advancing to base {next_base} of {len(self._peri_bases)-1} '
                    f'({self._peri_frame_count} frames so far)')
                self._peri_base_idx = next_base
                self._peri_dir_idx = 0
                self._peri_mode = 'arrive'
                self._peri_rot_accum = 0.0
                self._peri_consecutive_fail = 0
                self._peri_first_R_BH = None
                self._send_joint_target(self._peri_bases[next_base], f'base_{next_base}')
                return
            self.get_logger().info(f'  all bases done: {self._peri_frame_count} frames')
            self._peri_switch_phase()
            return

        if self._peri_consecutive_fail >= self._peri_max_fail:
            self.get_logger().warn(f'  too many failures ({self._peri_consecutive_fail})')
            self._peri_switch_phase()
            return

        # ── ROTATE mode: apply next rotation step ──
        if self._peri_mode == 'rotate':
            axis_idx, label = self._peri_directions[self._peri_dir_idx]
            # Build rotation in sensor frame
            axis_vec = np.zeros(3); axis_vec[axis_idx] = 1.0
            R_sensor_rot = so3_exp(axis_vec * self._peri_rot_step)
            self._peri_rot_accum += self._peri_rot_step

            T = self._get_hand_pose()
            if T is None:
                self._peri_consecutive_fail += 1; self._peri_switch_phase(); return
            R_BH, t_BH = ros_tf_to_matrix(T)
            R_BH_new = R_BH @ self.R_he_nom @ R_sensor_rot @ self.R_he_nom.T
            T_target = np.vstack([np.hstack([R_BH_new, t_BH.reshape(3,1)]), [[0,0,0,1]]])
            qs = inverse_kinematics_numeric(T_target, q_init=self.latest_joints)
            if len(qs) == 0:
                self.get_logger().warn(f'  [{self._peri_phase}] {label} rotate IK fail')
                self._peri_dir_idx += 1; self._peri_rot_accum = 0.0; self._peri_send_next(); return
            self._send_joint_target(np.array(qs[0]), f'{label}_rot')
            return

        # ── ARRIVE mode: just landed at a base or finished a direction — start rotating ──
        if self._peri_mode == 'arrive':
            self._peri_mode = 'rotate'
            self._peri_rot_accum = 0.0
            self._peri_send_next()
            return

        # ── Default: unexpected mode ──
        self.get_logger().warn(f'  unexpected mode {self._peri_mode} in _peri_send_next, ignoring')

    def _peri_switch_phase(self):
        if self._peri_phase == 'e1':
            self._start_peristaltic('e2')
        else:
            self._peri_finish()

    def _peri_on_arrival(self):
        """v44: arrive/rotate/servo state machine"""
        # ── Base generation (first arrival at any base) ──
        if hasattr(self, "_peri_bases") and self._peri_dir_idx == 0 and len(self._peri_bases) == 1:
            T = self._get_hand_pose()
            if T is not None:
                R_BH, t_BH = ros_tf_to_matrix(T)
                sensor_rots = [
                    (np.deg2rad([20, 0, 0]), 'pitch+20'),
                    (np.deg2rad([-15, 0, 0]), 'pitch-15'),
                    (np.deg2rad([0, 15, 0]), 'yaw+15'),
                ]
                for rot_vec, label in sensor_rots:
                    R_sensor = so3_exp(rot_vec)
                    R_BH_target = R_BH @ self.R_he_nom @ R_sensor @ self.R_he_nom.T
                    T_target = np.vstack([np.hstack([R_BH_target, t_BH.reshape(3,1)]), [[0,0,0,1]]])
                    qs = inverse_kinematics_numeric(T_target, q_init=self._peri_start_joints)
                    if len(qs) > 0:
                        self._peri_bases.append(np.array(qs[0]))
                        self.get_logger().info(f'  base {len(self._peri_bases)-1}: {label} IK OK')
                    else:
                        self.get_logger().warn(f'  base {label}: IK FAIL')

        # ── Recovery handling ──
        if getattr(self, '_peri_in_recovery', False):
            self._peri_in_recovery = False
            f = self._extract_features()
            has_edge = (f.get(self._peri_phase) is not None)
            if has_edge:
                self.get_logger().info('    recovery OK')
                self._peri_current_joints = (
                    self.latest_joints.copy() if self.latest_joints is not None
                    else self._peri_current_joints)
                self._peri_send_next()
            else:
                self.get_logger().warn('    recovery also lost edge, skip direction')
                self._peri_dir_idx += 1; self._peri_rot_accum = 0.0; self._peri_mode = 'arrive'
                self._peri_send_next()
            return

        f = self._extract_features()

        # Check for the edge we care about
        if self._peri_phase == 'e1':
            has_edge = f.get('e1') is not None
            et = f['e1'][0] if has_edge else None
        else:
            has_edge = f.get('e2') is not None
            et = f['e2'][0] if has_edge else None

        if not has_edge:
            self._peri_consecutive_fail += 1
            npts = f.get('n_pts', 0)
            self.get_logger().warn(
                f'    no {self._peri_phase} (n_pts={npts}, fail={self._peri_consecutive_fail})')
            self._peri_current_joints = self._peri_last_valid_joints.copy()
            self._peri_in_recovery = True
            self._send_joint_target(self._peri_last_valid_joints, 'back_to_safe')
            return

        self._peri_consecutive_fail = 0
        self._peri_current_joints = (
            self.latest_joints.copy() if self.latest_joints is not None
            else self._peri_current_joints)

        # ── Mode dispatch ──
        boundary_mm = 15.0  # e_tilde threshold for "near FOV boundary"

        if self._peri_mode == 'rotate':
            # Just completed a rotation step
            accum_deg = np.rad2deg(self._peri_rot_accum)
            axis_idx, label = self._peri_directions[self._peri_dir_idx]

            npts = f.get('n_pts', 0)
            npts_dropped = npts < max(10, self._peri_last_npts * 0.6)  # dropped >40% or below 10
            self._peri_last_npts = npts
            if abs(et) > boundary_mm / 1000.0 or self._peri_rot_accum >= self._peri_rot_target or npts_dropped:
                # Boundary reached, target achieved, or n_pts dropping (laser leaving plate) → servo to center
                self.get_logger().info(
                    f'    {label} accum={accum_deg:.1f}deg '
                    f'e_tilde={et*1000:.1f}mm npts={npts}' + (' DROP' if npts_dropped else '') + ' → switch to servo')
                self._peri_mode = 'servo'
                self._peri_do_servo(et)
                return
            else:
                # Keep rotating
                self.get_logger().info(
                    f'    {label} accum={accum_deg:.1f}deg e_tilde={et*1000:.1f}mm → keep rotating')
                self._peri_send_next()
                return

        # 'arrive' or 'servo' mode: center then check diversity
        if abs(et) > 0.003:  # > 3mm off-center → servo
            self._peri_servo_count = getattr(self, '_peri_servo_count', 0) + 1
            if self._peri_servo_count > self._peri_max_servo:
                self.get_logger().warn(
                    f'    servo retries exhausted (e_tilde={et*1000:.1f}mm), recording anyway')
                self._peri_record_frame(f, label='exhausted_servo')
                return
            self.get_logger().info(
                f'    {self._peri_mode} e_tilde={et*1000:.1f}mm → servo #{self._peri_servo_count}')
            self._peri_do_servo(et)
            return

        # Centered — check diversity and record
        self._peri_servo_count = 0
        if self._peri_mode == 'arrive':
            # First arrival at this base/direction: record initial frame, then start rotating
            self._peri_record_frame(f, label='initial')
            self._peri_mode = 'rotate'
            self._peri_rot_accum = 0.0
            self._peri_send_next()
        elif self._peri_mode == 'servo':
            # Servo done — check if diverse from last recorded
            if self._check_diverse_from_last():
                self._peri_record_frame(f, label='diverse')
            # Advance direction if target rotation reached, else keep rotating
            if self._peri_rot_accum >= self._peri_rot_target:
                self.get_logger().info(
                    f'    direction complete (accum={np.rad2deg(self._peri_rot_accum):.1f}deg)')
                self._peri_dir_idx += 1
                self._peri_rot_accum = 0.0
                self._peri_mode = 'arrive'
            else:
                self._peri_mode = 'rotate'
            self._peri_send_next()
        else:
            self.get_logger().warn(f'  unexpected mode {self._peri_mode}')
            self._peri_mode = 'arrive'
            self._peri_send_next()

    def _check_diverse_from_last(self):
        """Check if current orientation differs enough from last recorded frame."""
        T = self._get_hand_pose()
        if T is None:
            return False
        R_curr, _ = ros_tf_to_matrix(T)
        if self._peri_last_recorded_R is None:
            return True  # first recording always passes
        Rd = self._peri_last_recorded_R.T @ R_curr
        tr = np.clip((np.trace(Rd) - 1) / 2, -1, 1)
        ang_dist = np.rad2deg(np.arccos(tr))
        return ang_dist >= 6.0

    def _peri_record_frame(self, f, label=''):
        """Record current frame with feature data."""
        T = self._get_hand_pose()
        if T is None:
            return
        R_curr, _ = ros_tf_to_matrix(T)
        self._peri_last_recorded_R = R_curr

        edge_label = self._peri_phase
        et = f.get(edge_label, (0,))[0] if f.get(edge_label) else 0
        npts = f.get('n_pts', 0)
        ang_dist = f'{label}' if label else ''
        self.get_logger().info(
            f'    RECORD {edge_label} frame={self._peri_frame_count} '
            f'e_tilde={et*1000:.1f}mm {ang_dist} n_pts={npts}')
        self._record_pose(f)
        self._peri_last_npts = npts
        self._peri_last_valid_joints = self._peri_current_joints.copy()
        self._peri_frame_count += 1

    def _peri_do_servo(self, et):
        """Servo along sensor X to center the breakpoint."""
        T = self._get_hand_pose()
        if T is None:
            self._peri_mode = 'arrive'
            self._peri_send_next()
            return
        R_BH, t_BH = ros_tf_to_matrix(T)
        x_S_world = R_BH @ self.R_he_nom[:, 0]
        gain = 0.8
        delta = x_S_world * (et * gain)
        t_BH_target = t_BH + delta
        qs = inverse_kinematics_numeric(
            np.vstack([np.hstack([R_BH, t_BH_target.reshape(3, 1)]), [[0, 0, 0, 1]]]),
            q_init=self.latest_joints)
        if len(qs) == 0:
            self.get_logger().warn('    servo IK fail')
            self._peri_mode = 'arrive'
            self._peri_send_next()
            return
        self._send_joint_target(np.array(qs[0]), f'servo_{self._peri_servo_count}')

    def _peri_finish(self):
        self.get_logger().info(f'\n  All peristaltic done: {len(self.records)} total frames')
        self._phase3_solve()

    def _phase3_solve(self):
        self.get_logger().info('\n' + '=' * 50 + '\n  Phase 3: Diagnostics\n' + '=' * 50)
        self._pl_state = 'DONE'
        self._auto_phase = None

        valid_recs = [r for r in self.records if 'R_BH' in r and r.get('n_pts', 0) >= 10]
        self.get_logger().info(f'  total frames: {len(valid_recs)}')
        for i, r in enumerate(valid_recs):
            self.get_logger().info(
                f'    [{i}] n_pts={r["n_pts"]:3d} e1={r.get("valid_e1",False)} e2={r.get("valid_e2",False)}')

        # Diagnostic: compute R_he from each frame and check consistency
        self.get_logger().info('\n  --- R_he per frame diagnostic ---')
        R_he_list = []
        for i, r in enumerate(valid_recs):
            R_BH = r['R_BH']
            R_BS = r.get('R_BS')
            if R_BS is None:
                self.get_logger().warn(f'  [{i}] no R_BS, skipping')
                continue
            R_he_i = R_BH.T @ R_BS
            R_he_list.append(R_he_i)
            err_vs_true = np.rad2deg(np.linalg.norm(so3_log(R_he_i.T @ self.R_he_true)))
            self.get_logger().info(f'  [{i}] R_he_i vs R_he_true: {err_vs_true:.2f}deg')

        if len(R_he_list) >= 2:
            # Check spread
            max_diff = 0
            for i in range(len(R_he_list)):
                for j in range(i+1, len(R_he_list)):
                    Rd = R_he_list[i].T @ R_he_list[j]
                    tr = np.clip((np.trace(Rd)-1)/2, -1, 1)
                    d = np.rad2deg(np.arccos(tr))
                    if d > max_diff: max_diff = d
            self.get_logger().info(f'  Max spread between R_he estimates: {max_diff:.2f}deg')
            # Average
            w_avg = np.mean([so3_log(R) for R in R_he_list], axis=0)
            R_he_avg = so3_exp(w_avg)
            err_avg = np.rad2deg(np.linalg.norm(so3_log(R_he_avg.T @ self.R_he_true)))
            self.get_logger().info(f'  R_he_avg vs R_he_true: R_err={err_avg:.2f}deg')

        # Solver with correct edge assignment
        self.get_logger().info('\n  --- Solver (corrected edge assignment) ---')

        # Classify each frame: is sensor X closer to u_B or v_B?
        meas_corrected = []
        for r in valid_recs:
            # Compute sensor X direction in world
            R_BS = r.get('R_BS')
            if R_BS is None:
                R_BS = r['R_BH'] @ self.R_he_true
            x_S = R_BS[:, 0]  # sensor X axis in world

            dot_u = abs(np.dot(x_S, U_B_GT))
            dot_v = abs(np.dot(x_S, V_B_GT))

            m = {'p_S_plane': list(r.get('pts_S', []))}
            if r.get('valid_e1') and r.get('valid_e2') and 'p_S_e1' in r and 'p_S_e2' in r:
                if dot_u > dot_v:
                    # Sensor X closer to u_B → laser line ~u_B → edges on v_B → both are e2
                    m['valid_e1'] = False
                    m['valid_e2'] = True
                    # Use BOTH edge points — they're on the same v_B edge but at different positions
                    # Store them as two separate e2 measurements
                    m['_p_S_e2_second'] = r['p_S_e2']
                    m['p_S_e2'] = r['p_S_e1']
                    self.get_logger().info(f'    sensor X~u_B → both edges→e2 (2pts)')
                else:
                    m['valid_e1'] = True
                    m['valid_e2'] = False
                    m['_p_S_e1_second'] = r['p_S_e2']
                    m['p_S_e1'] = r['p_S_e1']
                    self.get_logger().info(f'    sensor X~v_B → both edges→e1 (2pts)')
            else:
                m['valid_e1'] = r.get('valid_e1', False)
                m['valid_e2'] = r.get('valid_e2', False)
                m['p_S_e1'] = r.get('p_S_e1')
                m['p_S_e2'] = r.get('p_S_e2')
            meas_corrected.append(m)

        poses = [(r['R_BH'], r['t_BH']) for r in valid_recs]
        n_e1 = sum(1 for m in meas_corrected if m['valid_e1'])
        n_e2 = sum(1 for m in meas_corrected if m['valid_e2'])

        # Custom 6-DOF solver: fix R_pl=I, optimize R_he(3)+t_he(3) only
        self.get_logger().info('\n  --- 6-DOF Solver (fixed R_pl=I) ---')
        self.get_logger().info(f'  corrected: e1={n_e1} e2={n_e2}')

        sys.path.insert(0, '/workspace/Num2')
        from nbv_edge_plane import combined_residuals

        # DIAGNOSTIC: cost at R_he_true vs R_he_gt (both with R_pl=I)
        for label, Rh, th in [("R_he_true", self.R_he_true, self.t_he_true),
                               ("R_he_gt", self.R_he_gt, self.t_he_gt)]:
            w = so3_log(Rh)
            t9 = np.concatenate([w, th, np.zeros(3)])
            r, mask, info = combined_residuals(t9, poses, meas_corrected, 0.1, 1.0)
            rv = r[mask]
            c = 0.5*np.dot(rv, rv) if len(rv) > 0 else 0
            self.get_logger().info(
                f'  Cost@{label}: {c:.6f} (n_plane={info.get("n_plane",0)} '
                f'e1_pairs={info.get("e1_pairs",0)} e2_pairs={info.get("e2_pairs",0)})')
            if len(rv) > 0:
                self.get_logger().info(f'    rms={np.sqrt(np.mean(rv**2)):.6f} max_abs={np.max(np.abs(rv)):.6f}')

        def residuals_6dof(theta6):
            """theta6 = [w_he(3), t_he(3)], R_pl fixed to I"""
            theta9 = np.zeros(9)
            theta9[0:6] = theta6
            theta9[6:9] = 0.0  # R_pl = I
            r, mask, info = combined_residuals(theta9, poses, meas_corrected, 1.0, 0.0)
            return r[mask]

        def cost_6dof(theta6):
            r = residuals_6dof(theta6)
            return 0.5 * np.dot(r, r)

        # Init at R_he_true
        w_init = so3_log(self.R_he_true)
        theta6 = np.concatenate([w_init, self.t_he_true])

        # Simple LM
        lam = 1e-4
        for it in range(30):
            r0 = residuals_6dof(theta6)
            cost0 = 0.5 * np.dot(r0, r0)
            # Numerical Jacobian
            eps = 1e-6
            J = np.zeros((len(r0), 6))
            for k in range(6):
                step = np.zeros(6); step[k] = eps
                rp = residuals_6dof(theta6 + step)
                rm = residuals_6dof(theta6 - step)
                J[:, k] = (rp - rm) / (2 * eps)
            H = J.T @ J + lam * np.eye(6)
            g = J.T @ r0
            try:
                delta = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            theta_new = theta6 + delta
            cost_new = cost_6dof(theta_new)
            if cost_new < cost0:
                theta6 = theta_new
                lam = max(lam / 3, 1e-10)
                if abs(cost0 - cost_new) < 1e-12:
                    break
            else:
                lam = min(lam * 3, 1e8)

        Re = so3_exp(theta6[0:3]); te = theta6[3:6]
        Rd = Re.T @ self.R_he_true
        tr = np.clip((np.trace(Rd)-1)/2, -1, 1)
        self.get_logger().info(f'  6DOF vs R_he_true: R_err={np.rad2deg(np.arccos(tr)):.4f}deg t_err={np.linalg.norm(te-self.t_he_true)*1000:.2f}mm')
        Rd2 = Re.T @ self.R_he_gt
        tr2 = np.clip((np.trace(Rd2)-1)/2, -1, 1)
        self.get_logger().info(f'  6DOF vs R_he_gt:   R_err={np.rad2deg(np.arccos(tr2)):.4f}deg t_err={np.linalg.norm(te-self.t_he_gt)*1000:.2f}mm')

        if self._auto_mode:
            diag = [{'R_BH': r['R_BH'].tolist(), 't_BH': r['t_BH'].tolist(),
                     'R_BS': r.get('R_BS', np.eye(3)).tolist() if r.get('R_BS') is not None else None}
                    for r in valid_recs]
            with open('/tmp/auto_calib_diag.json', 'w') as f:
                json.dump(diag, f, indent=2)
            self.get_logger().info('  saved')
            # v38 DIAG: save FULL data for solver diagnosis
            full_diag = {
                'R_he_true': self.R_he_true.tolist(), 't_he_true': self.t_he_true.tolist(),
                'R_he_nom': self.R_he_nom.tolist(), 't_he_nom': self.t_he_nom.tolist(),
                'R_he_gt': self.R_he_gt.tolist(), 't_he_gt': self.t_he_gt.tolist(),
                'U_B_GT': U_B_GT.tolist(), 'V_B_GT': V_B_GT.tolist(),
                'records': [{
                    'R_BH': r['R_BH'].tolist(), 't_BH': r['t_BH'].tolist(),
                    'R_BS': r.get('R_BS', np.eye(3)).tolist() if r.get('R_BS') is not None else None,
                    'valid_e1': r.get('valid_e1', False), 'valid_e2': r.get('valid_e2', False),
                    'p_S_e1': r['p_S_e1'].tolist() if ('p_S_e1' in r and r['p_S_e1'] is not None) else None,
                    'p_S_e2': r['p_S_e2'].tolist() if ('p_S_e2' in r and r['p_S_e2'] is not None) else None,
                    'pts_S': r.get('pts_S', []).tolist() if hasattr(r.get('pts_S', []), 'tolist') else r.get('pts_S', []), 'n_pts': r.get('n_pts', 0),
                } for r in valid_recs]
            }
            with open('/tmp/auto_calib_full_diag.json', 'w') as ff:
                json.dump(full_diag, ff, indent=2)

        # ── Solver 1: combined_solve_lm (9-DOF, plane+edge) ──
        self.get_logger().info('\n  --- Solver: combined_solve_lm (9-DOF) ---')
        try:
            from nbv_edge_plane import combined_solve_lm as cslm
            wn = so3_log(self.R_he_nom)
            theta9_init = np.concatenate([wn, self.t_he_nom, np.zeros(3)])
            t9 = cslm(theta9_init, poses, meas_corrected, w_plane=0.1, w_edge=1.0, max_iter=100)
            R9 = so3_exp(t9[0:3]); t9v = t9[3:6]
            Rd = R9.T @ self.R_he_true
            tr = np.clip((np.trace(Rd)-1)/2, -1, 1)
            self.get_logger().info(f'  combined_solve_lm vs R_he_true: R_err={np.rad2deg(np.arccos(tr)):.4f}deg')
        except Exception as e:
            self.get_logger().warn(f'  combined_solve_lm FAILED: {e}')

        # ── Solver 2: iterative_refine_he (交替 PCA→LM) ──
        self.get_logger().info('\n  --- Solver: iterative_refine_he (alternating) ---')
        try:
            from calib_solver import iterative_refine_he as irh
            Rh_i, th_i, Rpl_i, nB_i, nit_i = irh(poses, meas_corrected, self.R_he_nom, self.t_he_nom, max_iter=5)
            Rd = Rh_i.T @ self.R_he_true
            tr = np.clip((np.trace(Rd)-1)/2, -1, 1)
            self.get_logger().info(
                f'  iterative_refine_he vs R_he_true: R_err={np.rad2deg(np.arccos(tr)):.4f}deg '
                f't_err={np.linalg.norm(th_i-self.t_he_true)*1000:.2f}mm n_iter={nit_i}')
        except Exception as e:
            self.get_logger().warn(f'  iterative_refine_he FAILED: {e}')

        # ── Solver 3: tilted_corner (12-DOF, solve_12dof_with_restarts) ──
        self.get_logger().info('\n  --- Solver: tilted_corner (12-DOF with restarts) ---')
        try:
            from calib_solver import solve_12dof_with_restarts
            # Use raw measurements (not corrected) for 12-DOF
            meas_raw = []
            for r in valid_recs:
                m = {'p_S_plane': list(r.get('pts_S', []))}
                m['valid_e1'] = r.get('valid_e1', False)
                m['valid_e2'] = r.get('valid_e2', False)
                m['p_S_e1'] = r.get('p_S_e1')
                m['p_S_e2'] = r.get('p_S_e2')
                meas_raw.append(m)
            t0 = self.get_clock().now()
            best_theta, info = solve_12dof_with_restarts(
                poses, meas_raw, n_restarts=5, seed=42,
                w_he_init=so3_log(self.R_he_nom))
            R12 = so3_exp(best_theta[0:3]); t12 = best_theta[3:6]
            Rd = R12.T @ self.R_he_true
            tr = np.clip((np.trace(Rd)-1)/2, -1, 1)
            self.get_logger().info(
                f'  tilted_corner vs R_he_true: R_err={np.rad2deg(np.arccos(tr)):.4f}deg '
                f't_err={np.linalg.norm(t12-self.t_he_true)*1000:.2f}mm '
                f'cost={info["best_cost"]:.2e} n_good={info["n_good"]}')
        except Exception as e:
            self.get_logger().warn(f'  tilted_corner FAILED: {e}')


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--auto', action='store_true')
    args, _ = p.parse_known_args()
    rclpy.init()
    node = AutoCalibV2Node()
    if args.auto:
        node.create_timer(1.0, node._auto_start_once)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
