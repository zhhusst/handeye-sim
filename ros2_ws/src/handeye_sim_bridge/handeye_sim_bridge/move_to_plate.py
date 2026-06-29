#!/usr/bin/env python3
"""
move_to_plate.py — 用 pymoveit2 将机械臂移动到 gocator_sensor 对准标定平板

传感器目标：在平板中心 [0.9, 0.25, 0.25] 上方 0.45m，指向平板
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import Header
import numpy as np
import sys
import time

from pymoveit2 import MoveIt2

# ==================== 运动学工具 ====================
def rpy_to_matrix(rx, ry, rz):
    """ZYX 欧拉角 → 旋转矩阵"""
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx

def rpy_to_quat(rx, ry, rz):
    R = rpy_to_matrix(rx, ry, rz)
    return matrix_to_quat(R)

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
    node = Node('move_to_plate')

    # ============ 手眼标定 (flange → gocator_sensor) ============
    t_f_s = np.array([-0.011579, -0.004621, 0.359284])
    rpy_f_s = np.array([0.485145, 0.160648, -1.509479])
    R_f_s = rpy_to_matrix(rpy_f_s[0], rpy_f_s[1], rpy_f_s[2])

    node.get_logger().info(f"R_f_s:\n{R_f_s}")
    node.get_logger().info(f"t_f_s: {t_f_s}")

    # ============ 计算目标传感器位姿 (world 系) ============
    # 平板中心
    plate_center = np.array([0.9, 0.25, 0.25])
    plate_normal = np.array([0., 0., 1.])

    dist = 0.45  # 传感器离平板 0.45m
    s_pos = plate_center + plate_normal * dist

    # 传感器 Z 轴 = 指向平板中心 (looking direction)
    z_axis = (plate_center - s_pos)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # 传感器 X 轴 = 沿平板 u_B (水平扫描方向) ≈ [1,0,0]
    x_axis = np.array([1., 0., 0.])

    # 正交化: Y = Z × X, 然后 X = Y × Z
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    R_s_w = np.column_stack([x_axis, y_axis, z_axis])

    node.get_logger().info(f"--- 目标传感器位姿 ---")
    node.get_logger().info(f"位置: {s_pos}")
    node.get_logger().info(f"旋转:\n{R_s_w}")

    # ============ 转换到法兰位姿 ============
    # 目标: flange位姿 = 传感器位姿 @ inv(flange→传感器)
    # R_f_w = R_s_w @ R_f_s.T
    # t_f_w = s_pos - R_f_w @ t_f_s
    R_f_w = R_s_w @ R_f_s.T
    t_f_w = s_pos - R_f_w @ t_f_s

    node.get_logger().info(f"--- 目标法兰位姿 ---")
    node.get_logger().info(f"位置: {t_f_w}")
    node.get_logger().info(f"旋转:\n{R_f_w}")

    # ============ 尝试多个距离 (如果主目标失败) ============
    targets = [(dist, s_pos, R_s_w)]

    for alt_dist in [0.50, 0.40, 0.55, 0.60, 0.35, 0.65, 0.30]:
        alt_s_pos = plate_center + plate_normal * alt_dist
        alt_z = (plate_center - alt_s_pos)
        alt_z = alt_z / np.linalg.norm(alt_z)
        alt_y = np.cross(alt_z, x_axis)
        alt_y = alt_y / np.linalg.norm(alt_y)
        alt_x = np.cross(alt_y, alt_z)
        alt_R = np.column_stack([alt_x, alt_y, alt_z])
        targets.append((alt_dist, alt_s_pos, alt_R))

    # ============ 设置 MoveIt2 ============
    joint_names = ['J1_joint', 'J2_joint', 'J3_joint',
                   'J4_joint', 'J5_joint', 'J6_joint']

    moveit2 = MoveIt2(
        node=node,
        joint_names=joint_names,
        base_link_name='base_link',
        end_effector_name='flange',
        group_name='arm',
        use_move_group_action=True,
    )

    # Spin once to let MoveIt2 initialize
    time.sleep(1.0)
    rclpy.spin_once(node, timeout_sec=0.5)

    # ============ 尝试规划到每个目标 ============
    success = False
    for dist, s_pos, R_s in targets:
        # 计算法兰位姿
        R_f = R_s @ R_f_s.T
        t_f = s_pos - R_f @ t_f_s
        q = matrix_to_quat(R_f)

        node.get_logger().info(f"")
        node.get_logger().info(f"尝试 dist={dist:.2f}m, 法兰 pos=({t_f[0]:.3f}, {t_f[1]:.3f}, {t_f[2]:.3f})")

        # 构建目标位姿
        target_pose = Pose()
        target_pose.position.x = float(t_f[0])
        target_pose.position.y = float(t_f[1])
        target_pose.position.z = float(t_f[2])
        target_pose.orientation.x = float(q[0])
        target_pose.orientation.y = float(q[1])
        target_pose.orientation.z = float(q[2])
        target_pose.orientation.w = float(q[3])

        # 先做规划
        traj = moveit2.plan(
            pose=target_pose,
            frame_id='world',
            target_link='flange',
            tolerance_position=0.02,
            tolerance_orientation=0.05,
        )

        if traj is not None and len(traj.points) > 0:
            node.get_logger().info(f"✅ 规划成功于 dist={dist:.2f}m！点数为 {len(traj.points)}")

            # 执行
            node.get_logger().info("正在执行...")
            moveit2.execute(traj)
            moveit2.wait_until_executed()
            node.get_logger().info("✅ 执行完成！")

            # 等待机械臂到达
            time.sleep(1.0)

            success = True
            break
        else:
            node.get_logger().warn(f"❌ 规划失败于 dist={dist:.2f}m")

    if not success:
        node.get_logger().error("所有目标位姿均规划失败！")
        node.get_logger().info("尝试关节空间移动到中间位姿...")

        # 回 home 再试
        home_pos = [0.0, -1.2, -1.0, 0.0, 1.0, 0.0]  # 近似弯腰看下方
        traj = moveit2.plan(
            joint_positions=home_pos,
            joint_names=joint_names,
            tolerance_joint_position=0.02,
        )
        if traj is not None and len(traj.points) > 0:
            node.get_logger().info("关节空间规划成功，正在执行...")
            moveit2.execute(traj)
            moveit2.wait_until_executed()
            time.sleep(1.0)

            # 重试所有目标
            for dist, s_pos, R_s in targets:
                R_f = R_s @ R_f_s.T
                t_f = s_pos - R_f @ t_f_s
                q = matrix_to_quat(R_f)
                target_pose = Pose()
                target_pose.position.x = float(t_f[0])
                target_pose.position.y = float(t_f[1])
                target_pose.position.z = float(t_f[2])
                target_pose.orientation.x = float(q[0])
                target_pose.orientation.y = float(q[1])
                target_pose.orientation.z = float(q[2])
                target_pose.orientation.w = float(q[3])
                traj = moveit2.plan(
                    pose=target_pose,
                    frame_id='world',
                    target_link='flange',
                    tolerance_position=0.02,
                    tolerance_orientation=0.05,
                )
                if traj is not None and len(traj.points) > 0:
                    node.get_logger().info(f"✅ 二次尝试成功于 dist={dist:.2f}m！")
                    moveit2.execute(traj)
                    moveit2.wait_until_executed()
                    success = True
                    break

    if success:
        node.get_logger().info("🎉 机械臂已移动到目标位置！FOV 应对准平板。")
    else:
        node.get_logger().error("❌ 失败！请用 RViz 手动拖拽。")

    rclpy.shutdown()


if __name__ == '__main__':
    main()
