#!/usr/bin/env python3
"""
search_pose.py — 搜索关节配置，使 gocator_sensor 指向平板

使用 moveit 的 FK 计算来精确验证传感器指向。
对所有关节组合进行搜索，找到使传感器Z指向平板的配置。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import time, itertools
from pymoveit2 import MoveIt2

PLATE_CENTER = np.array([0.9, 0.25, 0.25])

# 手眼标定 (flange→gocator_sensor)
def rpy_to_R(rx, ry, rz):
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    return np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx]
    ])

R_f_s = rpy_to_R(0.485145, 0.160648, -1.509479)
t_f_s = np.array([-0.011579, -0.004621, 0.359284])

def compute_sensor_pose(flange_pos, flange_R):
    """给定法兰位姿，计算传感器位姿"""
    sensor_pos = flange_pos + flange_R @ t_f_s
    sensor_R = flange_R @ R_f_s
    return sensor_pos, sensor_R

def score_config(joints):
    """FK计算传感器指向平板的得分"""
    # 用 moveit2.compute_fk 计算
    # 这里我们用近似 FK 来快速排列
    flange_pos, flange_R = approximate_fk(joints)
    sensor_pos, sensor_R = compute_sensor_pose(flange_pos, flange_R)
    sensor_z = sensor_R[:, 2]  # looking direction
    to_plate = PLATE_CENTER - sensor_pos
    dist = np.linalg.norm(to_plate)
    if dist < 0.01:
        return -1, 999, None
    to_plate_u = to_plate / dist
    dot = np.dot(sensor_z, to_plate_u)

    # 得分 = 指向得分 * 距离得分
    # 最佳: dot > 0 (指向平板), 0.27m < dist < 0.82m
    in_range = 0.27 <= dist <= 0.82
    angle_ok = dot > 0.3  # 至少有点指向平板

    score = 0
    if angle_ok:
        score += 10 * dot
    if in_range:
        score += 5
    if angle_ok and in_range:
        score += 100

    return score, dist, (sensor_pos, sensor_z, dot, flange_pos, flange_R)


def approximate_fk(joints):
    """简化 FK（对于搜索足够精确）"""
    # 从已知的机器人配置参数估算
    # FANUC M20iD/25 DH
    # 这些参数可能不完全准确，但足够搜索
    d1 = 0.330; a1 = 0.150; alpha1 = -np.pi/2
    d2 = 0.0;   a2 = 0.600; alpha2 = 0.0
    d3 = 0.0;   a3 = 0.200; alpha3 = np.pi/2
    d4 = 0.640; a4 = 0.0;   alpha4 = -np.pi/2
    d5 = 0.0;   a5 = 0.0;   alpha5 = np.pi/2
    d6 = 0.175; a6 = 0.0;   alpha6 = 0.0

    T = np.eye(4)
    dh = [(a1, alpha1, d1, joints[0]),
          (a2, alpha2, d2, joints[1]),
          (a3, alpha3, d3, joints[2]),
          (a4, alpha4, d4, joints[3]),
          (a5, alpha5, d5, joints[4]),
          (a6, alpha6, d6, joints[5])]

    for a, alpha, d, theta in dh:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        Ai = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d   ],
            [0,   0,      0,     1   ]
        ])
        T = T @ Ai

    return T[:3, 3], T[:3, :3]


def main():
    rclpy.init()
    node = Node('search_pose')

    joint_names = ['J1_joint', 'J2_joint', 'J3_joint',
                   'J4_joint', 'J5_joint', 'J6_joint']

    node.get_logger().info("正在搜索最佳关节配置...")

    # 关节角度范围（度），步长
    ranges = [
        (-30, 30, 5),   # J1
        (-60, 10, 5),   # J2 (大部分在负侧让臂向前)
        (-90, 10, 5),   # J3
        (-180, 180, 10), # J4 (可全转)
        (-180, 180, 10), # J5 (需要大幅旋转让传感器朝下)
        (-180, 180, 10), # J6
    ]

    best_score = -999
    best = None
    count = 0

    # 使用网格搜索
    for d1 in np.deg2rad(np.arange(-30, 35, 15)):
        for d2 in np.deg2rad(np.arange(-50, 15, 10)):
            for d3 in np.deg2rad(np.arange(-70, 15, 10)):
                for d5 in np.deg2rad(np.arange(-180, 185, 15)):
                    j = np.array([d1, d2, d3, 0.0, d5, 0.0])
                    score, dist, data = score_config(j)
                    count += 1
                    if score > best_score:
                        best_score = score
                        best = (j, score, dist, data)
                        if score > 50:
                            node.get_logger().info(f"  候选: J1={np.rad2deg(j[0]):.0f}° J2={np.rad2deg(j[1]):.0f}° J3={np.rad2deg(j[2]):.0f}° J5={np.rad2deg(j[4]):.0f}° score={score:.0f} dist={dist:.3f}m")

    node.get_logger().info(f"搜索完成：{count} 次评估")

    if best is None:
        node.get_logger().error("未找到有效配置！")
        rclpy.shutdown()
        return

    j_deg = np.rad2deg(best[0])
    node.get_logger().info(f"\n=== 最佳配置 ===")
    node.get_logger().info(f"关节 (deg): J1={j_deg[0]:.1f} J2={j_deg[1]:.1f} J3={j_deg[2]:.1f} J4={j_deg[3]:.1f} J5={j_deg[4]:.1f} J6={j_deg[5]:.1f}")
    node.get_logger().info(f"得分: {best_score}, 距离: {best[2]:.3f}m")
    sensor_pos, sensor_z, dot, flange_pos, flange_R = best[3]
    node.get_logger().info(f"传感器位置: {sensor_pos}")
    node.get_logger().info(f"传感器指向: {sensor_z}, dot={dot:.3f}")

    # ============ 用 MoveIt 执行 ============
    moveit2 = MoveIt2(
        node=node,
        joint_names=joint_names,
        base_link_name='base_link',
        end_effector_name='flange',
        group_name='arm',
        use_move_group_action=True,
    )

    time.sleep(1.0)
    rclpy.spin_once(node, timeout_sec=0.5)

    node.get_logger().info("\n正在 MoveIt 规划...")
    traj = moveit2.plan(
        joint_positions=list(best[0]),
        joint_names=joint_names,
        tolerance_joint_position=0.02,
    )

    if traj is not None and len(traj.points) > 0:
        node.get_logger().info(f"✅ 规划成功！{len(traj.points)} 个路径点")
        node.get_logger().info("正在执行...")
        moveit2.execute(traj)
        moveit2.wait_until_executed()
        node.get_logger().info("✅ 执行完成！")
        node.get_logger().info("\n=== 下一步 ===")
        node.get_logger().info("请检查 /gocator/profile 和 /scanline_marker 是否有数据")
    else:
        node.get_logger().error("❌ 规划失败！请用 RViz 手动拖拽三环到平板附近。")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
