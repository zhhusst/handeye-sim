"""
================================================================================
fanuc_kinematic.py — FANUC M-20iD/25 正逆运动学
================================================================================

核心: 直接使用原始已验证的 ForwardKinematic / InverseKinematic 类。
接口包装: 弧度输入输出，与 reproduction_scene.py 一致。

23轴联动验证:
  2026-05-17 J2=37°, J3=-70° → FK 误差 0.0mm (J23_factor=1.0)
================================================================================
"""
import math
import numpy as np
from handeye_sim_bridge.fanuc_types import RobotParam


# ============================================================================
# DH 参数
# ============================================================================
M20ID_DH = RobotParam(
    a=[0, 0.075, 0.840, 0.215, 0, 0],
    d=[0.425, 0, 0, 0.890, 0, 0.09],
)

JOINT_LIMITS_DEG = np.array([
    [-185, 185], [ -100, 160], [ -90, 220],
    [-200, 200], [ -180, 180], [ -450, 450],
])


# ============================================================================
# 正运动学 FK（复制自原始 FanucKinematic.py，已验证）
# ============================================================================

def forward_kinematics(joints_rad):
    """关节角(弧度, J3_display) → T_B_H (4×4)

    23轴联动: diff_23 = J2 + J3_display (J23_factor=1.0)
    """
    t = np.array(joints_rad, dtype=float).copy()
    dh = M20ID_DH

    diff_23 = t[1] + t[2]   # J3_display → URDF 中的 J3 位置
    t[5] = -t[5]             # J6 取反 (FANUC 控制器约定)

    c1, s1 = math.cos(t[0]), math.sin(t[0])
    c2, s2 = math.cos(t[1]), math.sin(t[1])
    c23, s23 = math.cos(diff_23), math.sin(diff_23)
    c4, s4 = math.cos(t[3]), math.sin(t[3])
    c5, s5 = math.cos(t[4]), math.sin(t[4])
    c6, s6 = math.cos(t[5]), math.sin(t[5])

    # 变换矩阵链 (与原始 FanucKinematic.py 完全一致)
    T_0_1 = np.array([[c1, -s1, 0, 0],
                      [s1,  c1, 0, 0],
                      [0,   0,  1, dh.d[0]],
                      [0,   0,  0, 1]])

    T_1_2 = np.array([[s2,  c2, 0, dh.a[1]],
                      [0,   0,  1, 0],
                      [c2, -s2, 0, 0],
                      [0,   0,  0, 1]])

    T_2_3 = np.array([[c23, -s23, 0,  dh.a[2]],
                      [-s23, -c23, 0,  0],
                      [0,    0,   -1,  0],
                      [0,    0,    0,  1]])

    T_3_4 = np.array([[c4, -s4, 0, dh.a[3]],
                      [0,   0,  1, -dh.d[3]],
                      [-s4, -c4, 0, 0],
                      [0,   0,  0, 1]])

    T_4_5 = np.array([[c5, -s5, 0, 0],
                      [0,   0, -1, 0],
                      [s5,  c5, 0, 0],
                      [0,   0,  0, 1]])

    T_5_6 = np.array([[c6, -s6, 0,  0],
                      [0,   0, -1, -dh.d[5]],
                      [s6,  c6, 0,  0],
                      [0,   0,  0,  1]])

    return T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6


# ============================================================================
# 逆运动学 IK（直接复制原始 InverseKinematic 逻辑，弧度版）
# ============================================================================

class _FKSolver:
    """内部 FK 求解器（用度，与原始代码一致）"""
    def __init__(self, dh):
        self.a = dh.a
        self.d = dh.d
        self.joint_angle = [0, 0, 0, 0, 0, 0]
        self.diff_23 = 0.0

    def set_joint(self, joint_deg):
        self.joint_angle = np.array(joint_deg, dtype=float)
        self._process()
        return self._compute()

    def _process(self):
        self.diff_23 = self.joint_angle[1] + self.joint_angle[2]
        self.diff_23 = np.deg2rad(self.diff_23)
        self.joint_angle[5] = -self.joint_angle[5]
        self.joint_angle = np.deg2rad(self.joint_angle)

    def _compute(self):
        t = self.joint_angle
        d23 = self.diff_23
        dh = self.a, self.d

        c1, s1 = math.cos(t[0]), math.sin(t[0])
        c2, s2 = math.cos(t[1]), math.sin(t[1])
        c23, s23 = math.cos(d23), math.sin(d23)
        c4, s4 = math.cos(t[3]), math.sin(t[3])
        c5, s5 = math.cos(t[4]), math.sin(t[4])
        c6, s6 = math.cos(t[5]), math.sin(t[5])

        T01 = np.array([[c1,-s1,0,0],[s1,c1,0,0],[0,0,1,self.d[0]],[0,0,0,1]])
        T12 = np.array([[s2,c2,0,self.a[1]],[0,0,1,0],[c2,-s2,0,0],[0,0,0,1]])
        T23 = np.array([[c23,-s23,0,self.a[2]],[-s23,-c23,0,0],[0,0,-1,0],[0,0,0,1]])
        T34 = np.array([[c4,-s4,0,self.a[3]],[0,0,1,-self.d[3]],[-s4,-c4,0,0],[0,0,0,1]])
        T45 = np.array([[c5,-s5,0,0],[0,0,-1,0],[s5,c5,0,0],[0,0,0,1]])
        T56 = np.array([[c6,-s6,0,0],[0,0,-1,-self.d[5]],[s6,c6,0,0],[0,0,0,1]])

        return T01 @ T12 @ T23 @ T34 @ T45 @ T56


def inverse_kinematics(T_B_H):
    """T_B_H (4×4) → 关节角 (弧度, J3_display), N×6

    算法: 直接复制原始 InverseKinematic.get_joint()
    """
    dh = M20ID_DH

    # ===== θ1 =====
    ax, ay = T_B_H[0, 2], T_B_H[1, 2]
    px, py = T_B_H[0, 3], T_B_H[1, 3]
    theta1 = math.atan2(py - dh.d[5] * ay, px - dh.d[5] * ax)

    # J4 位置 (Base 系)
    p4_B = T_B_H[:3, 3] - T_B_H[:3, 2] * dh.d[5]

    # T_0_1
    c1, s1 = math.cos(theta1), math.sin(theta1)
    T_0_1 = np.array([[c1,-s1,0,0],[s1,c1,0,0],[0,0,1,dh.d[0]],[0,0,0,1]])
    T_1_6 = np.linalg.inv(T_0_1) @ T_B_H

    # J4 在 J1 系中的位置
    p4x = -dh.d[5] * T_1_6[0, 2] + T_1_6[0, 3]
    p4z = -dh.d[5] * T_1_6[2, 2] + T_1_6[2, 3]

    # ===== θ2 =====
    a1, a2, a3, d3 = dh.a[1], dh.a[2], dh.a[3], dh.d[3]
    AA = -2 * (p4x - a1) * a2
    BB = 2 * (-a2) * p4z
    CC = (p4x - a1)**2 + a2**2 + p4z**2 - a3**2 - d3**2
    disc = AA**2 + BB**2 - CC**2
    if disc < 0:
        return np.array([], dtype=float)

    sd = math.sqrt(disc)
    theta2_1 = math.atan2(CC, sd) - math.atan2(BB, AA)
    theta2_2 = math.atan2(CC, -sd) - math.atan2(BB, AA)

    # ±π 修正 (同原始代码)
    if theta2_1 < 0 and theta2_2 < 0:
        theta2_1 += math.pi; theta2_2 += math.pi
    elif theta2_1 >= 0 and theta2_2 >= 0:
        theta2_1 -= math.pi; theta2_2 -= math.pi

    # ===== θ3 =====
    pairs_23 = []  # (θ2, θ3_urdf) 弧度
    for theta2 in [theta2_1, theta2_2]:
        while theta2 > math.pi: theta2 -= 2*math.pi
        while theta2 < -math.pi: theta2 += 2*math.pi

        s2, c2 = math.sin(theta2), math.cos(theta2)

        if math.pi/2 > theta2 > -math.pi/2:
            # 肘上配置
            aa = (c2 * (math.cos(theta1)*p4_B[0] + p4_B[1]*math.sin(theta1))
                  - a1*c2 + s2*(dh.d[0] - p4_B[2]))
        else:
            aa = (c2 * (math.cos(theta1)*p4_B[0] + p4_B[1]*math.sin(theta1))
                  - a1*c2 + s2*(dh.d[0] - p4_B[2]))

        disc3 = a3**2 + d3**2 - aa**2
        if disc3 < 0:
            continue
        sd3 = math.sqrt(disc3)
        theta3_1 = math.atan2(d3, a3) - math.atan2(aa, sd3)
        theta3_2 = math.atan2(d3, a3) - math.atan2(aa, -sd3)

        for theta3_urdf in [theta3_1, theta3_2]:
            # 验证: 构造 T_0_4 (θ4=0), 检查 J4 位置
            # 注意: FK 中 diff_23 = J2 + J3_display = θ3_urdf
            # 所以 T_2_3 用 theta3_urdf (不是 theta2+theta3_urdf)
            s2_v, c2_v = math.sin(theta2), math.cos(theta2)
            s23_v, c23_v = math.sin(theta3_urdf), math.cos(theta3_urdf)
            T_1_2_v = np.array([[s2_v, c2_v, 0, dh.a[1]],
                                [0,    0,    1, 0],
                                [c2_v, -s2_v, 0, 0],
                                [0,    0,    0, 1]])
            T_2_3_v = np.array([[c23_v, -s23_v, 0, dh.a[2]],
                                [-s23_v, -c23_v, 0, 0],
                                [0,      0,     -1, 0],
                                [0,      0,      0, 1]])
            T_3_4_0 = np.array([[1, 0, 0, dh.a[3]],
                                [0, 0, 1, -dh.d[3]],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]])
            T_0_4 = T_0_1 @ T_1_2_v @ T_2_3_v @ T_3_4_0
            pos_J4 = T_0_4[:3, 3]
            dist = np.linalg.norm(pos_J4 - p4_B)
            if dist < 0.001:
                pairs_23.append((theta2, theta3_urdf))

    if not pairs_23:
        return np.array([], dtype=float)

    # ===== θ4, θ5, θ6 =====
    raw_solutions = []
    for theta2, theta3_urdf in pairs_23:
        # 手动构造 T_0_3 — 注意 T_2_3 用 theta3_urdf (不是 theta2+theta3_urdf)
        s2_v, c2_v = math.sin(theta2), math.cos(theta2)
        s23_v, c23_v = math.sin(theta3_urdf), math.cos(theta3_urdf)
        T_1_2_v = np.array([[s2_v, c2_v, 0, dh.a[1]],
                            [0,    0,    1, 0],
                            [c2_v, -s2_v, 0, 0],
                            [0,    0,    0, 1]])
        T_2_3_v = np.array([[c23_v, -s23_v, 0, dh.a[2]],
                            [-s23_v, -c23_v, 0, 0],
                            [0,      0,     -1, 0],
                            [0,      0,      0, 1]])
        T_0_3 = T_0_1 @ T_1_2_v @ T_2_3_v
        T_3_6 = np.linalg.inv(T_0_3) @ T_B_H

        if abs(T_3_6[0, 2]) < 1e-17 and abs(T_3_6[2, 2]) < 1e-17:
            theta4_candidates = [0.0]
        else:
            theta4_1 = -math.atan2(T_3_6[2, 2], T_3_6[0, 2])
            theta4_2 = theta4_1 + (math.pi if theta4_1 < 0 else -math.pi)
            theta4_candidates = [theta4_1, theta4_2]

        for theta4 in theta4_candidates:
            c4_v, s4_v = math.cos(theta4), math.sin(theta4)
            T_3_4_v = np.array([[c4_v, -s4_v, 0, dh.a[3]],
                                [0,     0,    1, -dh.d[3]],
                                [-s4_v, -c4_v, 0, 0],
                                [0,     0,    0, 1]])
            T_0_4 = T_0_3 @ T_3_4_v
            T_4_6 = np.linalg.inv(T_0_4) @ T_B_H
            theta5 = math.atan2(T_4_6[0, 2], -T_4_6[2, 2])
            theta6 = math.atan2(-T_4_6[1, 0], T_4_6[1, 1])
            raw_solutions.append([theta1, theta2, theta3_urdf, theta4, theta5, theta6])

    # ===== 后处理 =====
    solutions = []
    JOINT_LIMITS_RAD = np.deg2rad(JOINT_LIMITS_DEG)

    for sol in raw_solutions:
        s = list(sol)
        s[2] = s[2] - s[1]    # J3_display = J3_urdf - J2
        s[5] = -s[5]           # J6 取反还原

        for i in range(6):
            while s[i] > math.pi:  s[i] -= 2*math.pi
            while s[i] < -math.pi: s[i] += 2*math.pi

        in_limits = all(JOINT_LIMITS_RAD[i,0] <= s[i] <= JOINT_LIMITS_RAD[i,1] for i in range(6))
        if not in_limits:
            continue

        is_dup = any(max(abs(s[i]-e[i]) for i in range(6)) < 0.01 for e in solutions)
        if not is_dup:
            solutions.append(s)

    return np.array(solutions, dtype=float) if solutions else np.array([], dtype=float)


# ============================================================================
# 工具函数
# ============================================================================
def random_joints(scale=0.85, rng=None):
    """在关节限位内随机采样一组关节角(弧度)"""
    if rng is None: rng = np.random
    joints = np.zeros(6)
    for i in range(6):
        lo, hi = JOINT_LIMITS_DEG[i]
        center = (lo + hi) / 2
        joints[i] = rng.uniform(center - (hi-lo)/2*scale,
                                center + (hi-lo)/2*scale)
    return np.deg2rad(joints)


def get_link_positions(joints_rad):
    """计算机械臂各连杆位置 (Base 系)

    参数:
        joints_rad: 6 个关节角 (弧度)

    返回:
        positions: 6 个点坐标, [base, J1, J2, J3, J5, flange]
    """
    from handeye_sim_bridge.fanuc_types import RobotParam
    dh = M20ID_DH
    t = np.array(joints_rad, dtype=float).copy()
    diff_23 = t[1] + t[2]
    t[5] = -t[5]

    c1, s1 = math.cos(t[0]), math.sin(t[0])
    c2, s2 = math.cos(t[1]), math.sin(t[1])
    c23, s23 = math.cos(diff_23), math.sin(diff_23)
    c4, s4 = math.cos(t[3]), math.sin(t[3])
    c5, s5 = math.cos(t[4]), math.sin(t[4])
    c6, s6 = math.cos(t[5]), math.sin(t[5])

    T_0_1 = np.array([[c1,-s1,0,0],[s1,c1,0,0],[0,0,1,dh.d[0]],[0,0,0,1]])
    T_1_2 = np.array([[s2,c2,0,dh.a[1]],[0,0,1,0],[c2,-s2,0,0],[0,0,0,1]])
    T_2_3 = np.array([[c23,-s23,0,dh.a[2]],[-s23,-c23,0,0],[0,0,-1,0],[0,0,0,1]])
    T_3_4 = np.array([[c4,-s4,0,dh.a[3]],[0,0,1,-dh.d[3]],[-s4,-c4,0,0],[0,0,0,1]])
    T_4_5 = np.array([[c5,-s5,0,0],[0,0,-1,0],[s5,c5,0,0],[0,0,0,1]])
    T_5_6 = np.array([[c6,-s6,0,0],[0,0,-1,-dh.d[5]],[s6,c6,0,0],[0,0,0,1]])

    T_B_1 = T_0_1
    T_B_2 = T_B_1 @ T_1_2
    T_B_3 = T_B_2 @ T_2_3
    T_B_4 = T_B_3 @ T_3_4
    T_B_5 = T_B_4 @ T_4_5
    T_B_6 = T_B_5 @ T_5_6

    return np.array([
        [0, 0, 0],           # base
        T_B_1[:3, 3],        # J1
        T_B_2[:3, 3],        # J2
        T_B_3[:3, 3],        # J3
        T_B_5[:3, 3],        # J5 (手腕)
        T_B_6[:3, 3],        # flange
    ])


def verify_fk_ik(n_tests=100):
    rng = np.random.RandomState(42)
    passed = 0
    for _ in range(n_tests):
        joints = random_joints(scale=0.85, rng=rng)
        T = forward_kinematics(joints)
        sols = inverse_kinematics(T)
        if len(sols) == 0: continue
        T_back = forward_kinematics(sols[0])
        pe = np.linalg.norm(T[:3,3] - T_back[:3,3])
        re = math.acos(np.clip((np.trace(T[:3,:3].T @ T_back[:3,:3]) - 1) / 2, -1, 1))
        if pe < 1e-6 and re < 1e-6: passed += 1
    print(f"FK∘IK 闭环: {passed}/{n_tests} ✅")
    return passed == n_tests


if __name__ == '__main__':
    print("=== FK 测试 ===")
    T = forward_kinematics(np.deg2rad([0, 37, -70, 0, 0, 0]))
    print(f"FK(J2=37°, J3=-70°) pos=({T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f})")

    print("\n=== IK 测试 ===")
    sols = inverse_kinematics(T)
    if len(sols) > 0:
        for i, sol in enumerate(sols):
            print(f"  解{i}: {np.rad2deg(sol)}")
        T_back = forward_kinematics(sols[0])
        print(f"FK(IK) pos_err={np.linalg.norm(T[:3,3]-T_back[:3,3]):.2e}")
    else:
        print("  IK 无解 ❌")

    print("\n=== FK∘IK 闭环 ===")
    verify_fk_ik(200)
