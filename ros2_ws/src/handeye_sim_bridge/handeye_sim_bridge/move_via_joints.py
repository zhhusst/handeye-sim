#!/usr/bin/env python3
"""
move_via_joints.py — 用 joint space 方式把机械臂移动到传感器对准平板

策略: 
1. 先用正运动学算合适的位姿
2. 用 MoveIt 关节空间规划到目标关节角
"""
import rclpy
from rclpy.node import Node
import numpy as np
import time

from pymoveit2 import MoveIt2

# ==================== 正运动学 ====================
def dh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,  sa,     ca,     d   ],
        [0,  0,      0,      1   ]
    ])

def fanuc_fk(joints):
    """FANUC M20iD/25 正运动学，返回 flange 位姿"""
    # FANUC standard DH parameters (approximate)
    d1, a1, alpha1 = 0.330, 0.150, -np.pi/2   # J1
    d2, a2, alpha2 = 0.0, 0.600, 0.0          # J2
    d3, a3, alpha3 = 0.0, 0.200, np.pi/2      # J3
    d4, a4, alpha4 = 0.640, 0.0, -np.pi/2     # J4
    d5, a5, alpha5 = 0.0, 0.0, np.pi/2        # J5
    d6, a6, alpha6 = 0.175, 0.0, 0.0          # J6

    T = np.eye(4)
    params = [(a1, alpha1, d1, joints[0]),
              (a2, alpha2, d2, joints[1]),
              (a3, alpha3, d3, joints[2]),
              (a4, alpha4, d4, joints[3]),
              (a5, alpha5, d5, joints[4]),
              (a6, alpha6, d6, joints[5])]
    for a, alpha, d, theta in params:
        T = T @ dh_transform(a, alpha, d, theta)
    return T[:3, 3], T[:3, :3]  # position, rotation


def matrix_to_quat(R):
    q = np.zeros(4)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q[3] = 0.25 * S
        q[0] = (R[2,1] - R[1,2]) / S
        q[1] = (R[0,2] - R[2,0]) / S
        q[2] = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        q[3] = (R[2,1] - R[1,2]) / S
        q[0] = 0.25 * S
        q[1] = (R[0,1] + R[1,0]) / S
        q[2] = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        q[3] = (R[0,2] - R[2,0]) / S
        q[0] = (R[0,1] + R[1,0]) / S
        q[1] = 0.25 * S
        q[2] = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        q[3] = (R[1,0] - R[0,1]) / S
        q[0] = (R[0,2] + R[2,0]) / S
        q[1] = (R[1,2] + R[2,1]) / S
        q[2] = 0.25 * S
    return q


def main():
    rclpy.init()
    node = Node('move_via_joints')

    joint_names = ['J1_joint', 'J2_joint', 'J3_joint',
                   'J4_joint', 'J5_joint', 'J6_joint']

    # ============ 尝试多种关节配置 ============
    configs = [
        # [J1, J2, J3, J4, J5, J6] — 注释说明
        # 配置1: 弯腰向前看平板 (传感器Z指向下方)
        [0.0, -0.3, -0.5, 0.0, 0.8, 0.0],
        # 配置2: 更弯一点
        [0.0, -0.4, -0.6, 0.0, 1.0, 0.0],
        # 配置3: 向左偏
        [0.3, -0.3, -0.5, 0.0, 0.8, 0.0],
        # 配置4: 类似home但更前倾
        [0.0, -0.5, -0.7, 0.0, 1.2, 0.0],
        # 配置5: 当前位姿微调 (J5减小让传感器朝下)
        [0.454, 0.136, -0.409, -1.281, 1.2, -1.758],
        # 配置6
        [0.0, -0.2, -0.4, 0.0, 0.6, 0.0],
        # 配置7: 大幅度弯腰
        [0.0, -0.6, -0.8, 0.0, 1.4, 0.0],
    ]

    node.get_logger().info("=== 预检各配置的传感器位姿 ===")

    # 手眼标定 (flange→gocator_sensor)
    def rpy_to_matrix(rx, ry, rz):
        cz, sz = np.cos(rz), np.sin(rz)
        cy, sy = np.cos(ry), np.sin(ry)
        cx, sx = np.cos(rx), np.sin(rx)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        return Rz @ Ry @ Rx

    t_f_s = np.array([-0.011579, -0.004621, 0.359284])
    R_f_s = rpy_to_matrix(0.485145, 0.160648, -1.509479)
    plate_center = np.array([0.9, 0.25, 0.25])

    for i, q in enumerate(configs):
        flange_pos, flange_R = fanuc_fk(np.array(q))
        sensor_pos = flange_pos + flange_R @ t_f_s
        sensor_z = flange_R @ R_f_s[:, 2]  # sensor Z in world

        # 传感器到平板的向量
        to_plate = plate_center - sensor_pos
        dist = np.linalg.norm(to_plate)
        to_plate_u = to_plate / dist
        dot = np.dot(sensor_z, to_plate_u)

        if dot > 0:
            orient_str = f"✅ 指向平板 (dot={dot:.3f}, dist={dist:.3f}m)"
        else:
            orient_str = f"❌ 背对平板 (dot={dot:.3f}, dist={dist:.3f}m)"

        node.get_logger().info(f"  配置{i+1}: sensor pos=({sensor_pos[0]:.3f}, {sensor_pos[1]:.3f}, {sensor_pos[2]:.3f}) {orient_str}")

    # ============ 选择最好的配置 ============
    best_idx = -1
    best_dot = -2.0
    for i, q in enumerate(configs):
        flange_pos, flange_R = fanuc_fk(np.array(q))
        sensor_pos = flange_pos + flange_R @ t_f_s
        sensor_z = flange_R @ R_f_s[:, 2]
        to_plate = plate_center - sensor_pos
        dist = np.linalg.norm(to_plate)
        if dist < 0.01:
            continue
        to_plate_u = to_plate / dist
        dot = np.dot(sensor_z, to_plate_u)
        # Score: prioritize pointing at plate AND being in FOV distance range (0.27-0.82m)
        if 0.27 <= dist <= 0.82 and dot > best_dot:
            best_dot = dot
            best_idx = i

    if best_idx < 0:
        # Fallback to closest to pointing at plate
        for i, q in enumerate(configs):
            flange_pos, flange_R = fanuc_fk(np.array(q))
            sensor_pos = flange_pos + flange_R @ t_f_s
            sensor_z = flange_R @ R_f_s[:, 2]
            to_plate = plate_center - sensor_pos
            dist = np.linalg.norm(to_plate)
            if dist < 0.01:
                continue
            to_plate_u = to_plate / dist
            dot = np.dot(sensor_z, to_plate_u)
            if dot > best_dot:
                best_dot = dot
                best_idx = i

    if best_idx < 0:
        node.get_logger().error("没有可行配置！")
        node.get_logger().info("请用 RViz 手动拖拽三环到平板附近位置。")
        rclpy.shutdown()
        return

    best_q = configs[best_idx]
    flange_pos, flange_R = fanuc_fk(np.array(best_q))
    sensor_pos = flange_pos + flange_R @ t_f_s
    sensor_z = flange_R @ R_f_s[:, 2]
    to_plate = plate_center - sensor_pos
    dist = np.linalg.norm(to_plate)
    dot = np.dot(sensor_z, to_plate / dist)

    node.get_logger().info(f"\n=== 选择配置{best_idx+1} ===")
    node.get_logger().info(f"关节角: {best_q}")
    node.get_logger().info(f"传感器位姿: {sensor_pos}")
    node.get_logger().info(f"传感器Z指向: {sensor_z}")
    node.get_logger().info(f"指向平板: dot={dot:.3f}, dist={dist:.3f}m")

    # ============ 执行 MoveIt 规划 ============
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

    node.get_logger().info(f"\n正在规划到配置{best_idx+1}...")
    traj = moveit2.plan(
        joint_positions=best_q,
        joint_names=joint_names,
        tolerance_joint_position=0.02,
    )

    if traj is not None and len(traj.points) > 0:
        node.get_logger().info(f"✅ 规划成功！{len(traj.points)} 个路径点")
        node.get_logger().info("正在执行...")
        moveit2.execute(traj)
        moveit2.wait_until_executed()
        node.get_logger().info("✅ 执行完成！")

        time.sleep(1.0)

        # 验证实际位姿
        node.get_logger().info("\n=== 验证 ===")
        node.get_logger().info("请检查 /gocator/profile 和 /scanline_marker 数据")
    else:
        node.get_logger().error("❌ 规划失败！")
        node.get_logger().info("请用 RViz 手动拖拽三环到平板附近位置。")

    rclpy.shutdown()


if __name__ == '__main__':
    main()
