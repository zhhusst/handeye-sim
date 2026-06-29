#!/usr/bin/env python3
"""scene_publisher_node.py — 只读 joint_states，发布标定场景 Marker

与 MoveIt 配合使用：MoveIt 控制关节，本节点显示平板/FOV/扫描线。
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from std_srvs.srv import Trigger

import sys, os
sys.path.insert(0, '/workspace/common')

from fov_geometry import (
    compute_fov_plate_scanline, compute_fov_triangle,
    build_R_edge, make_transform, rpy_to_matrix,
)
from handeye_sim_bridge.bridge_publisher import CalibPublisher
from handeye_sim_bridge.fanuc_kinematic import forward_kinematics
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

JOINT_NAMES = ['J1_joint', 'J2_joint', 'J3_joint',
               'J4_joint', 'J5_joint', 'J6_joint']


def matrix_to_quat(R):
    """3x3 rotation matrix → [x,y,z,w] quaternion"""
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


def quat_to_matrix(q):
    """[x,y,z,w] quaternion → 3x3 rotation matrix"""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


class ScenePublisher(Node):
    def __init__(self):
        super().__init__('scene_publisher')

        self.publisher = CalibPublisher()

        # 查询 FOV 角点的服务（用于读取当前拖拽后的值）
        self._fov_query_srv = self.create_service(
            Trigger, '~/query_fov_corners',
            self._query_fov_corners_cb)

        # latest_joints 需要先初始化，FOV IM 引用它
        self.latest_joints = None

        # 场景 (固定, 每次随机生成)
        rng = np.random.default_rng(42)
        # 平板位置 - 放远一些
        C = np.array([rng.uniform(0.6, 1.0),
                      rng.uniform(-0.2, 0.2),
                      rng.uniform(0.1, 0.3)])
        # 平板法向/面内方向
        n_B = np.array([0., 0., 1.])
        ax = rng.uniform(-10, 10); ay = rng.uniform(-10, 10); az = rng.uniform(-10, 10)
        R_plane = rpy_to_matrix(ax, ay, az)
        n_B = R_plane @ np.array([0., 0., 1.]); n_B = n_B / np.linalg.norm(n_B)
        if abs(n_B[2]) < 0.9:
            u_B = np.cross(np.array([0., 0., 1.]), n_B)
        else:
            u_B = np.cross(np.array([0., 1., 0.]), n_B)
        u_B = u_B / np.linalg.norm(u_B)
        v_B = np.cross(n_B, u_B)
        w, h = 0.4, 0.5

        # 手眼真值 — gocator_sensor 坐标系原点（在激光平面上）
        # fanuc_flange → gocator_sensor 关节: xyz=[-0.0116,-0.0046,0.3593] rpy=[0.485,0.161,-1.509]
        # 注意：gocator_sensor 坐标系定义在激光平面上，visual mesh 的 (0,0,-0.27) 偏移只影响渲染
        R_he = np.array([[ 0.99964905,  0.02634223,  0.00280383],
                         [-0.02631765,  0.99961777, -0.00846724],
                         [-0.00302581,  0.00839048,  0.99996022]])
        t_he = np.array([-0.011579, -0.004621, 0.359284])

        X_gt = make_transform(R_he, t_he)
        R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

        R_plate = rpy_to_matrix(ax, ay, az)  # 初始平板旋转
        # FOV 激光平面 = gocator_sensor 的 XZ 平面
        # 4个角点（在传感器系中），默认=GoCator 2450 规格
        fov_corners_S = [
            np.array([ 0.0, 0.0, -0.282]),  # corner 0: 底部顶点
            np.array([ 0.0, 0.0, -0.282]),  # corner 1: 底部顶点（与0重合，三角形）
            np.array([ 0.22, 0.0, 0.820]),  # corner 2: 右上
            np.array([-0.22, 0.0, 0.820]),  # corner 3: 左上
        ]
        self.scene = {
            'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
            'w': w, 'h': h,
            'R_he': R_he, 't_he': t_he,
            'R_plate': R_plate,
            'fov_corners_S': fov_corners_S,
        }
        self.publisher.set_scene(C, n_B, u_B, v_B, w, h)

        # TF 缓存 — 用于将 gocator_sensor 系坐标转 world 系
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 为平板创建 Interactive Marker (在 RViz 中拖拽调整)
        self._setup_plate_interactive_marker()

        # FOV 角点 IM 延迟创建 — 等首次收到 joint_states 后 (TF 就绪)
        self._fov_im_setup = False

        # 订阅 joint_states (由 robot_state_publisher 发布)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # 定时器 — 确保场景 marker 持续发布 (30Hz，配合 robot_state_publisher)
        self.timer = self.create_timer(0.033, self.timer_callback)

        self.get_logger().info("场景发布器已启动 — 等待 joint_states...")
        self.get_logger().info(f"平板中心: {C}")
        self.get_logger().info(f"手眼 GT: t_he={t_he}")

        # 先发一次场景 (无 joint 信息时只发平板)
        stamp = self.get_clock().now().to_msg()
        self.publisher.publish_scene_markers(stamp)

    def joint_callback(self, msg):
        """收到 joint_states 时更新"""
        try:
            q = [msg.position[msg.name.index(j)] for j in JOINT_NAMES]
            self.latest_joints = np.array(q)
            # 首次收到 joint_states → 此时 TF 树应有 gocator_sensor，创建 FOV 角点 IM
            if not self._fov_im_setup:
                if self._setup_fov_corner_markers():
                    self._fov_im_setup = True
            self.publish_all_markers(msg.header.stamp)
        except Exception as e:
            self.get_logger().warn(f"joint_states 回调异常: {e}")

    def _setup_plate_interactive_marker(self):
        """创建 FOV 角点的 IM server（仅初始化 server，供 FOV 角点使用）"""
        self.im_server = InteractiveMarkerServer(self, 'plate_im')
        self.get_logger().info('InteractiveMarkerServer 已初始化（供 FOV 角点使用）')

    def _get_sensor_pose(self):
        """获取当前传感器在世界系中的位姿 (R_BS, t_BS)"""
        if self.latest_joints is None:
            return None, None
        T_B_H = forward_kinematics(self.latest_joints)
        R_i, t_i = T_B_H[:3, :3], T_B_H[:3, 3]
        R_BS = R_i @ self.scene['R_he']
        t_BS = t_i + R_i @ self.scene['t_he']
        return R_BS, t_BS

    def _setup_fov_corner_markers(self):
        """创建 FOV 平面的4个角点 Interactive Marker

        用 TF 将传感器系角点转换到 world 系，IM 创建在 world 系中。
        """
        from geometry_msgs.msg import Quaternion as GeoQuat

        # 用 TF 查 gocator_sensor → world
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'gocator_sensor', rclpy.time.Time())
            t_w = t.transform.translation
            q_w = t.transform.rotation
            q_arr = np.array([q_w.x, q_w.y, q_w.z, q_w.w])
            R_SW = quat_to_matrix(q_arr)
            t_SW = np.array([t_w.x, t_w.y, t_w.z])
            self.get_logger().info(
                f'gocator_sensor→base_link: t={t_SW}')
        except Exception as e:
            self.get_logger().warn(f'TF 查找失败，下次重试: {e}')
            return False

        colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0), (1.0, 1.0, 0.2)]
        labels = ['左下', '右下', '右上', '左上']

        for i, corner_S in enumerate(self.scene['fov_corners_S']):
            # 传感器系 → 世界系
            pos_world = t_SW + R_SW @ corner_S

            im = InteractiveMarker()
            im.header.frame_id = 'world'
            im.header.stamp = self.get_clock().now().to_msg()
            im.name = f'fov_corner_{i}'
            im.description = f'FOV {labels[i]}'
            im.pose.position = Point(x=float(pos_world[0]), y=float(pos_world[1]),
                                     z=float(pos_world[2]))
            im.pose.orientation = GeoQuat(x=0.0, y=0.0, z=0.0, w=1.0)
            im.scale = 0.15

            # 彩色小球 + MOVE_3D 合并在一个 control 里
            mk = Marker()
            mk.type = Marker.SPHERE
            mk.scale.x = 0.04; mk.scale.y = 0.04; mk.scale.z = 0.04
            r, g, b = colors[i]
            mk.color.r = r; mk.color.g = g; mk.color.b = b; mk.color.a = 0.9

            ctrl = InteractiveMarkerControl()
            ctrl.always_visible = True
            ctrl.interaction_mode = InteractiveMarkerControl.MOVE_3D
            ctrl.name = 'move'
            ctrl.markers.append(mk)
            im.controls.append(ctrl)

            self.im_server.insert(im, feedback_callback=\
                lambda fb, idx=i: self._corner_feedback_callback(fb, idx))
        self.im_server.applyChanges()
        self.get_logger().info('FOV 4角点 IM 已创建 (world 系, TF 定位)')
        return True

    def _corner_feedback_callback(self, feedback, idx):
        """FOV 角点被拖拽 — 反馈位姿在 world 系，转存为传感器系"""
        p = feedback.pose.position
        pt_world = np.array([p.x, p.y, p.z])
        # 用 TF 查 gocator_sensor → world
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'gocator_sensor', rclpy.time.Time())
            t_w = t.transform.translation
            q_w = t.transform.rotation
            q_arr = np.array([q_w.x, q_w.y, q_w.z, q_w.w])
            R_SW = quat_to_matrix(q_arr)
            t_SW = np.array([t_w.x, t_w.y, t_w.z])
            # world → gocator_sensor
            pt_sensor = R_SW.T @ (pt_world - t_SW)
        except Exception as e:
            self.get_logger().warn(f'TF 查找失败: {e}')
            return
        # 约束到 XZ 平面 (Y=0)
        pt_sensor[1] = 0.0
        self.scene['fov_corners_S'][idx] = pt_sensor
        self.get_logger().info(
            f'角点{idx} → ({pt_sensor[0]:.3f}, {pt_sensor[1]:.3f}, {pt_sensor[2]:.3f}) [传感器系]')

        # 投影 world 位姿（约束 Y=0）
        pt_world_proj = t_SW + R_SW @ pt_sensor
        feedback.pose.position.x = float(pt_world_proj[0])
        feedback.pose.position.y = float(pt_world_proj[1])
        feedback.pose.position.z = float(pt_world_proj[2])
        self.im_server.setPose(f'fov_corner_{idx}', feedback.pose)
        self.im_server.applyChanges()

        # 刷新场景
        stamp = self.get_clock().now().to_msg()
        self.publish_all_markers(stamp)

    def timer_callback(self):
        """定期刷新场景 marker (即使无新 joint_states)"""
        try:
            stamp = self.get_clock().now().to_msg()
            self.publish_all_markers(stamp)
        except Exception as e:
            self.get_logger().error(f"timer_callback 异常: {e}")

    def publish_all_markers(self, stamp):
        """一次性发布所有场景 Marker（平板 + FOV 平面 + TF）"""
        if self.scene is None:
            return
        C = self.scene['C']
        n_B = self.scene['n_B']
        u_B = self.scene['u_B']
        v_B = self.scene['v_B']
        w = self.scene['w']
        h = self.scene['h']
        corners_S = self.scene['fov_corners_S']
        if self.latest_joints is not None:
            # 用 TF 将角点从传感器系转换到 world 系（与URDF/RSP完全一致）
            try:
                t = self.tf_buffer.lookup_transform(
                    'world', 'gocator_sensor', rclpy.time.Time())
                tw = t.transform.translation
                qw = t.transform.rotation
                q_arr = np.array([qw.x, qw.y, qw.z, qw.w])
                R = quat_to_matrix(q_arr)
                t_vec = np.array([tw.x, tw.y, tw.z])
                corners_world = [t_vec + R @ c for c in corners_S]
                self.publisher.publish_all_markers(
                    stamp, corners_world, C, n_B, u_B, v_B, w, h,
                    frame_id="world", sensor_frame="world")
            except Exception:
                self.publisher.publish_all_markers(
                    stamp, corners_S, C, n_B, u_B, v_B, w, h)
        else:
            # 无 joint 数据时只发平板（FOV 需要传感器位姿）
            self.publisher.publish_scene_markers(stamp)

    def _query_fov_corners_cb(self, req, res):
        """返回当前 FOV 4个角点坐标（传感器系）"""
        corners = self.scene['fov_corners_S']
        msg = '\\n'.join([f'corner_{i}: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]'
                         for i, c in enumerate(corners)])
        self.get_logger().info(f'FOV角点查询:\\n{msg}')
        res.success = True
        res.message = msg
        return res

    # ──────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = ScenePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()