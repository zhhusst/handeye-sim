#!/usr/bin/env python3
"""
debug_fov2.py — 用当前实际传感器位姿测试 compute_fov_plate_scanline
带重试机制解决 TF buffer 同步问题
"""
import rclpy
from rclpy.node import Node
import numpy as np
import sys, os, time
sys.path.insert(0, '/workspace/common')
from fov_geometry import compute_fov_plate_scanline
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import rclpy.time
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

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
    C = np.array([0.7, 0.0, 0.25])
    n_B = np.array([0., 0., 1.])
    u_B = np.array([1., 0., 0.])
    v_B = np.array([0., 1., 0.])
    w, h = 0.4, 0.5

    node.get_logger().info("等待 TF...")
    
    # 多次 spin 等待 Buffer 填充
    for i in range(10):
        rclpy.spin_once(node, timeout_sec=0.5)
        try:
            # 检查 gocator_sensor frame 是否存在
            frame_exists = tf_buffer.can_transform('world', 'gocator_sensor', rclpy.time.Time())
            if frame_exists:
                node.get_logger().info(f"✅ 在第 {i+1} 次尝试找到 TF 变换")
                break
        except Exception:
            pass
        if i == 2:
            # Fallback: wait longer
            node.get_logger().info(f"等待中... (尝试 {i+1}/10)")

    try:
        t = tf_buffer.lookup_transform('world', 'gocator_sensor', rclpy.time.Time())
        tw = t.transform.translation
        qw = t.transform.rotation
        q_arr = np.array([qw.x, qw.y, qw.z, qw.w])
        R = quat_to_matrix(q_arr)
        t_vec = np.array([tw.x, tw.y, tw.z])

        node.get_logger().info(f"✅ TF 查找成功！")
        node.get_logger().info(f"传感器位置: [{t_vec[0]:.4f}, {t_vec[1]:.4f}, {t_vec[2]:.4f}]")
        node.get_logger().info(f"传感器旋转四元数: [{q_arr[0]:.4f}, {q_arr[1]:.4f}, {q_arr[2]:.4f}, {q_arr[3]:.4f}]")

        # 传感器 Z 轴 (looking direction)
        sensor_z = R[:, 2]
        sensor_y = R[:, 1]  # 激光平面法线

        plate_center = C + (w/2)*u_B + (h/2)*v_B
        to_plate = plate_center - t_vec
        dist = np.linalg.norm(to_plate)

        node.get_logger().info(f"平板中心: {plate_center}")
        node.get_logger().info(f"传感器→平板距离: {dist:.3f}m")
        node.get_logger().info(f"传感器Z(指向): [{sensor_z[0]:.4f}, {sensor_z[1]:.4f}, {sensor_z[2]:.4f}]")
        node.get_logger().info(f"传感器Y(激光法线): [{sensor_y[0]:.4f}, {sensor_y[1]:.4f}, {sensor_y[2]:.4f}]")

        if dist > 0.01:
            to_plate_u = to_plate / dist
            dot = np.dot(sensor_z, to_plate_u)
            node.get_logger().info(f"传感器Z·指向平板: {dot:.4f} {'✅' if dot > 0 else '❌'}")

        # 检查激光平面与板是否相交
        laser_normal = sensor_y
        cross_n = np.cross(laser_normal, n_B)
        dn = np.linalg.norm(cross_n)
        node.get_logger().info(f"激光平面·板法线夹角: {np.dot(laser_normal, n_B):.3f}")
        node.get_logger().info(f"交线方向norm: {dn:.6f} {'✅' if dn > 1e-10 else '❌ 平行无交线'}")

        if dn > 1e-10:
            line_dir = cross_n / dn
            A = np.vstack([laser_normal.reshape(1, 3), n_B.reshape(1, 3)])
            b = np.array([np.dot(laser_normal, t_vec), np.dot(n_B, C)])
            P0 = np.linalg.lstsq(A, b, rcond=None)[0]
            t_proj = np.dot(line_dir, C - P0)
            P0 = P0 + t_proj * line_dir

            R_SB = R.T
            t_SB = -R_SB @ t_vec
            tan_fov = np.tan(np.deg2rad(15.0))
            half_span = 0.5
            n_sample = 500

            in_plate = 0; in_range = 0; in_fov = 0
            for t_val in np.linspace(-half_span, half_span, n_sample):
                p_B = P0 + t_val * line_dir
                dp = p_B - C
                u = np.dot(dp, u_B); v = np.dot(dp, v_B)
                if u < -1e-6 or v < -1e-6 or u > w + 1e-6 or v > h + 1e-6:
                    continue
                in_plate += 1
                p_S = R_SB @ p_B + t_SB
                z, x = p_S[2], p_S[0]
                if 0.27 <= z <= 0.82:
                    in_range += 1
                if abs(x) <= z * tan_fov:
                    in_fov += 1

            node.get_logger().info(f"板内采样点: {in_plate}")
            node.get_logger().info(f"在测量范围 [0.27,0.82]m: {in_range}")
            node.get_logger().info(f"在FOV三角内: {in_fov}")

        # 调用计算函数
        res = compute_fov_plate_scanline(
            R_BS=R, t_BS=t_vec,
            C=C, n_B=n_B, u_B=u_B, v_B=v_B,
            pw=w, ph=h)

        node.get_logger().info(f"\ncompute_fov_plate_scanline 返回:")
        node.get_logger().info(f"  has_intersection: {res['has_intersection']}")
        if res['has_intersection']:
            node.get_logger().info(f"  scan_pts_B: {len(res['scan_pts_B'])} 点")
            node.get_logger().info(f"  endpoints: {res['endpoints_B']}")

    except Exception as e:
        node.get_logger().error(f"错误: {e}")
        import traceback
        traceback.print_exc()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
