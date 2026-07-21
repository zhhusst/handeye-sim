#!/usr/bin/env python3
"""
core/kinematics.py — FANUC M-20iD/25 正运动学 (FK)

从 fanuc_kinematic.py 提取核心 FK，无 ROS 依赖。
关节角输入: 弧度, J3 使用 display 值 (23轴联动: J2+J3 组合)
"""

import math
import numpy as np

# DH 参数 (FANUC M-20iD/25)
_A = [0, 0.075, 0.840, 0.215, 0, 0]
_D = [0.425, 0, 0, 0.890, 0, 0.09]


def fanuc_fk(joints_rad):
    """FANUC M-20iD/25 正运动学: 6维关节角(rad) → T_B_H (4×4)

    关节角格式: [J1, J2, J3_display, J4, J5, J6] (弧度)
    23轴联动: diff_23 = J2 + J3_display
    """
    t = np.array(joints_rad, dtype=float).copy()

    diff_23 = t[1] + t[2]    # 23轴联动
    t[5] = -t[5]              # J6 取反 (FANUC 控制器约定)

    c1, s1 = math.cos(t[0]), math.sin(t[0])
    c2, s2 = math.cos(t[1]), math.sin(t[1])
    c23, s23 = math.cos(diff_23), math.sin(diff_23)
    c4, s4 = math.cos(t[3]), math.sin(t[3])
    c5, s5 = math.cos(t[4]), math.sin(t[4])
    c6, s6 = math.cos(t[5]), math.sin(t[5])

    T_0_1 = np.array([[c1, -s1, 0, 0],
                      [s1,  c1, 0, 0],
                      [0,   0,  1, _D[0]],
                      [0,   0,  0, 1]])

    T_1_2 = np.array([[s2,  c2, 0, _A[1]],
                      [0,   0,  1, 0],
                      [c2, -s2, 0, 0],
                      [0,   0,  0, 1]])

    T_2_3 = np.array([[c23, -s23, 0,  _A[2]],
                      [-s23, -c23, 0,  0],
                      [0,    0,   -1,  0],
                      [0,    0,    0,  1]])

    T_3_4 = np.array([[c4, -s4, 0, _A[3]],
                      [0,   0,  1, -_D[3]],
                      [-s4, -c4, 0, 0],
                      [0,   0,  0, 1]])

    T_4_5 = np.array([[c5, -s5, 0, 0],
                      [0,   0, -1, 0],
                      [s5,  c5, 0, 0],
                      [0,   0,  0, 1]])

    T_5_6 = np.array([[c6, -s6, 0,  0],
                      [0,   0, -1, -_D[5]],
                      [s6,  c6, 0,  0],
                      [0,   0,  0,  1]])

    T = T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6
    return T
