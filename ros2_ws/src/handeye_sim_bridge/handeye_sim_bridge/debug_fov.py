#!/usr/bin/env python3
"""
debug_fov.py — 用当前实际传感器位姿测试 compute_fov_plate_scanline
"""
import rclpy
from rclpy.node import Node
import numpy as np
import sys, os
sys.path.insert(0, '/workspace/common')
from fov_geometry import compute_fov_plate_scanline
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import rclpy.time

def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])

def main():
    rclpy.init()
    node = Node('debug_fov')

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    # 平板参数（与 scene_publisher_node 一致）
    C = np.array([0.7, 0.0, 0.25])    # 角点
    n_B = np.array([0., 0., 1.])       # 法向量
    u_B = np.array([1., 0., 0.])
    v_B = np.array([0., 1., 0.])
    w, h = 0.4, 0.5

    node.get_logger().info("等待 TF...")
    rclpy.spin_once(node, timeout_sec=2.0)
    rclpy.spin_once(node, timeout_sec=1.0)

    try:
        t = tf_buffer.lookup_transform('world', 'gocator_sensor', rclpy.time.Time())
        tw = t.transform.translation
        qw = t.transform.rotation
        q_arr = np.array([qw.x, qw.y, qw.z, qw.w])
        R = quat_to_matrix(q_arr)
        t_vec = np.array([tw.x, tw.y, tw.z])

        node.get_logger().info(f"传感器位置: {t_vec}")
        node.get_logger().info(f"传感器旋转:\n{R}")

        # 传感器 Z 轴 (looking direction)
        sensor_z = R[:, 2]
        node.get_logger().info(f"传感器Z轴(指向): {sensor_z}")

        # 平板中心
        plate_center = C + (w/2)*u_B + (h/2)*v_B
        node.get_logger().info(f"平板中心: {plate_center}")

        # 传感器到平板的向量
        to_plate = plate_center - t_vec
        dist = np.linalg.norm(to_plate)
        to_plate_u = to_plate / dist
        dot = np.dot(sensor_z, to_plate_u)

        node.get_logger().info(f"传感器→平板距离: {dist:.3f}m")
        node.get_logger().info(f"传感器Z·指向平板: {dot:.3f}")

        # FOV 参数
        node.get_logger().info(f"FOV: half_fov=15°, range=[0.27, 0.82]m")
        node.get_logger().info(f"FOV检查: dist={dist:.3f}m, in_range={0.27 <= dist <= 0.82}")

        # ============ 调用 compute_fov_plate_scanline ============
        res = compute_fov_plate_scanline(
            R_BS=R, t_BS=t_vec,
            C=C, n_B=n_B, u_B=u_B, v_B=v_B,
            pw=w, ph=h,
            half_fov_deg=15.0, min_range=0.27, max_range=0.82,
        )

        node.get_logger().info(f"\n=== compute_fov_plate_scanline 结果 ===")
        node.get_logger().info(f"has_intersection: {res['has_intersection']}")

        if res['has_intersection']:
            node.get_logger().info(f"扫描点数量: {len(res['scan_pts_B'])}")
            node.get_logger().info(f"端点: {res['endpoints_B']}")
        else:
            # 检查激光平面与板是否平行
            laser_normal = R[:, 1]  # sensor Y axis
            line_dir = np.cross(laser_normal, n_B)
            dn = np.linalg.norm(line_dir)
            node.get_logger().info(f"激光平面法向量(sensor Y): {laser_normal}")
            node.get_logger().info(f"交线方向norm: {dn:.6f}")
            node.get_logger().info(f"激光平面与板法向量点积: {np.dot(laser_normal, n_B):.3f}")

            if dn < 1e-10:
                node.get_logger().error("❌ 激光平面与平板平行，无交线！")

            # 检查是否有采样点在范围内
            n_sample = 500
            half_span = 0.5
            tan_fov = np.tan(np.deg2rad(15.0))
            P0 = res.get('line_origin_B', np.zeros(3))
            line_dir = res.get('line_dir', np.zeros(3))

            if dn >= 1e-10 and line_dir is not None:
                R_SB = R.T
                t_SB = -R_SB @ t_vec
                t_vals = np.linspace(-half_span, half_span, n_sample)

                in_plate = 0
                in_fov = 0
                in_range = 0

                for t_val in t_vals:
                    p_B = P0 + t_val * line_dir
                    dp = p_B - C
                    u = np.dot(dp, u_B)
                    v = np.dot(dp, v_B)
                    if u < -1e-6 or v < -1e-6 or u > w + 1e-6 or v > h + 1e-6:
                        continue
                    in_plate += 1

                    p_S = R_SB @ p_B + t_SB
                    z = p_S[2]
                    x = p_S[0]

                    if z >= 0.27 and z <= 0.82:
                        in_range += 1
                    if abs(x) <= z * tan_fov:
                        in_fov += 1

                node.get_logger().info(f"采样点在板内: {in_plate}")
                node.get_logger().info(f"其中在测量范围内: {in_range}")
                node.get_logger().info(f"其中在FOV内: {in_fov}")

    except Exception as e:
        node.get_logger().error(f"TF 错误: {e}")
        import traceback
        traceback.print_exc()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
