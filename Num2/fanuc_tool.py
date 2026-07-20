"""
================================================================================
fanuc_tool.py — FANUC M-20iD/25 焊枪/传感器工具模型
================================================================================

法兰→焊枪 TCP:   T_TCP_FLANGE   (来自 FrameConfig.py)
法兰→Gocator:    T_GOCATOR_FLANGE (来自手眼标定结果，实际 = 待标定的 X_gt)
焊枪→接触点:     T_CONTACT_TCP   (接触传感触碰点偏移)

坐标系约定:
  T_X_Y: X 坐标系到 Y 坐标系的变换 (与 reproduction_scene.py 一致)
  T_B_H = T_B_H (法兰位姿)
  T_B_TCP = T_B_H @ T_TCP_FLANGE (焊枪TCP在Base系)
  T_B_S = T_B_H @ X_gt (传感器在Base系)
================================================================================
"""
import numpy as np


# ============================================================================
# 已标定的变换参数 (来自 CODEBASE_REFERENCE.md + historical data)
# ============================================================================

# 法兰 → 焊枪 TCP (2026-05-17 标定结果)
# 注: 这是焊接喷嘴尖端的位姿
TCP_TO_FLANGE_XYZ = [-0.046256, -0.000142, 0.375235]   # 米
TCP_TO_FLANGE_RPY = [-3.141540, -0.384130, -0.000070]   # 弧度 (近似 -180°, -22°, 0°)


def rpy_to_matrix(rpy):
    """ZYX 欧拉角 (弧度) → 3×3 旋转矩阵 R = Rz·Ry·Rx"""
    rx, ry, rz = rpy
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # ZYX


def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# 法兰 → 焊枪 TCP (4×4)
T_TCP_FLANGE = make_transform(
    rpy_to_matrix(TCP_TO_FLANGE_RPY),
    np.array(TCP_TO_FLANGE_XYZ)
)

# 焊枪 TCP 偏移 (接触传感触碰点相对于 TCP 的偏移)
# 假设焊丝伸出长度 ~15mm, 沿 TCP Z 轴方向
T_CONTACT_TCP = make_transform(
    np.eye(3),
    np.array([0.0, 0.0, 0.015])  # 15mm 焊丝伸出
)


# ============================================================================
# 工具函数
# ============================================================================

def get_T_B_TCP(T_B_H):
    """计算焊枪 TCP 在基座坐标系下的位姿"""
    return T_B_H @ T_TCP_FLANGE


def get_T_B_contact(T_B_H):
    """计算接触传感触碰点在基座坐标系下的位置 (3D 点)"""
    T_B_contact = T_B_H @ T_TCP_FLANGE @ T_CONTACT_TCP
    return T_B_contact[:3, 3]


def get_T_B_sensor(T_B_H, X_gt):
    """计算传感器在基座坐标系下的位姿 (线激光手眼)"""
    return T_B_H @ X_gt


# ============================================================================
# 接触传感仿真: 模拟触碰平面
# ============================================================================

def simulate_contact_point(T_B_H, n_B, d_plane, noise_std=0.0003):
    """模拟一次接触传感触碰 —— 焊丝尖端沿 TCP Z 轴移动直至接触平板

    参数:
        T_B_H: 法兰位姿 (4×4)
        n_B: 平面法向量 (归一化, Base系)
        d_plane: 平面距离 = n_B · p_B (平面上一点在法向上的投影)
        noise_std: 接触传感噪声标准差 (默认 0.3mm)

    返回:
        p_contact: 触碰点在 Base 系中的位置 (3D)
    """
    # 焊枪 TCP 原点 (Base 系)
    p_tcp = get_T_B_TCP(T_B_H)[:3, 3]

    # 焊丝方向: TCP Z 轴 (Base 系)
    R_B_TCP = get_T_B_TCP(T_B_H)[:3, :3]
    wire_dir = R_B_TCP[:, 2]  # Z 轴

    # 求 TCP 原点沿焊丝方向与平面的交点
    # 平面方程: n_B · p = d_plane
    # 射线方程: p = p_tcp + t * wire_dir
    # n_B · (p_tcp + t * wire_dir) = d_plane
    # t = (d_plane - n_B · p_tcp) / (n_B · wire_dir)
    denom = n_B @ wire_dir
    if abs(denom) < 1e-10:
        return None  # 焊丝与平面平行, 无法触碰

    t = (d_plane - n_B @ p_tcp) / denom
    if t > 0.2:  # 焊丝伸出 < 20cm (超出焊丝长度了)
        return None

    p_contact = p_tcp + t * wire_dir
    # 加高斯噪声
    p_contact += np.random.randn(3) * noise_std
    return p_contact


# ============================================================================
# 自测试
# ============================================================================

if __name__ == '__main__':
    print(f"T_TCP_FLANGE 平移: {T_TCP_FLANGE[:3, 3]}")
    print(f"T_TCP_FLANGE RPY: 约 (-180°, -22°, 0°)")

    # 测试接触传感仿真
    T_B_H = np.eye(4)
    T_B_H[:3, 3] = [0.5, 0.0, 0.3]
    n_B = np.array([0., 0., 1.])
    d_plane = 0.1

    p = simulate_contact_point(T_B_H, n_B, d_plane)
    print(f"\n接触传感仿真:")
    print(f"  TCP 位置: {get_T_B_TCP(T_B_H)[:3, 3]}")
    print(f"  触碰点:   {p}")
    print(f"  平面距离: {n_B @ p:.4f} (期望 0.1)")
