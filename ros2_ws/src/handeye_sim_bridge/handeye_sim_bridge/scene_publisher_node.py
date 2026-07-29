#!/usr/bin/env python3
"""scene_publisher_node.py — 只读 joint_states，发布标定场景 Marker

与 MoveIt 配合使用：MoveIt 控制关节，本节点显示平板/FOV/扫描线。
"""

import json
import os
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose, Quaternion
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from moveit_msgs.msg import CollisionObject
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker

from calibration_pipeline.simulation.scanline import (
    compute_fov_plate_scanline,
    make_transform,
)
from calibration_pipeline.simulation.noise import (
    JointSnapshotBuffer,
    SimulationNoiseConfig,
    SimulationNoiseModel,
)
from calibration_pipeline.simulation.scene_truth import (
    HAND_EYE_ROTATION,
    HAND_EYE_TRANSLATION,
)
from handeye_sim_bridge.bridge_publisher import CalibPublisher
from handeye_sim_bridge.fanuc_kinematic import (
    forward_kinematics,
    forward_kinematics_urdf,
)

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

        self.publisher = CalibPublisher(self)

        # FOV 拖拽校准属于运行数据，不写回源码目录。
        self.declare_parameter(
            'fov_calibration_file', '/workspace/data/fov_calib.json'
        )
        self._fov_calib_path = str(
            self.get_parameter('fov_calibration_file').value
        )
        self._fov_factory_path = os.path.join(
            get_package_share_directory('handeye_sim_bridge'),
            'config',
            'fov_factory_calib.json',
        )

        # 模拟 GoCator 数据发布器（点云，传感器系 XZ 坐标）
        self.gocator_pub = self.create_publisher(
            PointCloud2, '/gocator/profile', 10)
        # 断点信息发布器（传感器系，正确标记 e1/e2）
        self.endpoint_pub = self.create_publisher(
            Float64MultiArray, '/gocator/endpoints', 10)
        self.collision_pub = self.create_publisher(
            CollisionObject, '/collision_object', 10
        )
        # All simulation disturbances are injected before publishing the
        # profile/endpoints.  The calibration nodes receive no truth flag and
        # therefore exercise the same rejection and estimation path as a real
        # sensor.
        noise_defaults = SimulationNoiseConfig()
        noise_values = {}
        for name, default in noise_defaults.as_dict().items():
            parameter = f'simulation_noise.{name}'
            self.declare_parameter(parameter, default)
            noise_values[name] = self.get_parameter(parameter).value
        self.noise_config = SimulationNoiseConfig(**noise_values)
        self.noise_model = SimulationNoiseModel(self.noise_config)
        self.joint_history = JointSnapshotBuffer()
        self._noise_status_srv = self.create_service(
            Trigger, '~/noise_status', self._noise_status_cb
        )

        # 调试日志节流
        self._debug_frame = 0
        self._collision_publish_counter = 0

        # 查询 FOV 角点的服务（用于读取当前拖拽后的值）
        self._fov_query_srv = self.create_service(
            Trigger, '~/query_fov_corners',
            self._query_fov_corners_cb)
        self._fov_save_srv = self.create_service(
            Trigger, '~/save_fov_calib',
            self._save_fov_calib_cb)

        # latest_joints 需要先初始化，FOV IM 引用它
        self.latest_joints = None

        # 场景参数与 Gazebo 中的物理平板保持一致。
        self.declare_parameter('board.corner', [0.7, 0.0, 0.25])
        self.declare_parameter(
            'board.rotation',
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        self.declare_parameter('board.length_u_m', 0.4)
        self.declare_parameter('board.length_v_m', 0.5)
        C = np.asarray(self.get_parameter('board.corner').value, dtype=float)
        R_plate = np.asarray(
            self.get_parameter('board.rotation').value, dtype=float
        ).reshape(3, 3)
        u_B, v_B, n_B = R_plate[:, 0], R_plate[:, 1], R_plate[:, 2]
        w = float(self.get_parameter('board.length_u_m').value)
        h = float(self.get_parameter('board.length_v_m').value)

        # 手眼真值 — gocator_sensor 原点位于激光平面内，但不在激光窗口上。
        # fanuc_flange → gocator_sensor 关节: xyz=[-0.0116,-0.0046,0.3593] rpy=[0.485,0.161,-1.509]
        # 激光窗口约位于 z_S=-0.29 m，见 fov_factory_calib.json。
        R_he = HAND_EYE_ROTATION.copy()
        t_he = HAND_EYE_TRANSLATION.copy()

        X_gt = make_transform(R_he, t_he)
        R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

        # FOV 激光平面 = gocator_sensor 的 XZ 平面。前两个点定义真实激光
        # 窗口，后两个点定义量程远端；传感器原点位于主 FOV 三角区域内部。
        fov_corners_S = self._load_factory_fov_corners()
        self.scene = {
            'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
            'w': w, 'h': h,
            'R_he': R_he, 't_he': t_he,
            'R_plate': R_plate,
            'fov_corners_S': fov_corners_S,
        }
        self.publisher.set_scene(C, n_B, u_B, v_B, w, h)
        self._publish_plate_collision()

        # 加载用户拖拽保存的运行时校准（如存在，则覆盖出厂几何）。
        self._load_fov_calib()

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

        # Joint states may arrive at controller rate (50-100 Hz).  Generating
        # the complete synthetic profile and every RViz marker in that callback
        # floods DDS and can starve ros2_control.  Keep the callback lightweight
        # and publish the complete simulation frame at one bounded rate.
        self.declare_parameter('simulation_publish_rate_hz', 10.0)
        publish_rate = max(
            1.0, float(self.get_parameter('simulation_publish_rate_hz').value)
        )
        self._collision_publish_period = max(1, int(round(publish_rate)))
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)

        self.get_logger().info("场景发布器已启动 — 等待 joint_states...")
        self.get_logger().info(f"平板中心: {C}")
        self.get_logger().info(f"手眼 GT: t_he={t_he}")
        self.get_logger().info(
            "仿真噪声已启用: "
            f"profile={1e3 * self.noise_config.profile_gaussian_std_m:.3f} mm, "
            f"endpoint={1e3 * self.noise_config.endpoint_gaussian_std_m:.3f} mm, "
            f"robot_t={1e3 * self.noise_config.robot_translation_std_m:.3f} mm, "
            f"robot_R={self.noise_config.robot_rotation_std_deg:.4f} deg, "
            f"flatness={1e3 * self.noise_config.board_flatness_rms_m:.3f} mm, "
            f"sync_jitter={1e3 * self.noise_config.sync_jitter_std_s:.3f} ms"
        )

        # 先发一次场景 (无 joint 信息时只发平板)
        stamp = self.get_clock().now().to_msg()
        self.publisher.publish_scene_markers(stamp)

    def joint_callback(self, msg):
        """收到 joint_states 时更新"""
        try:
            q = [msg.position[msg.name.index(j)] for j in JOINT_NAMES]
            self.latest_joints = np.array(q)
            self.joint_history.append(time.monotonic_ns(), self.latest_joints)
            # 首次收到 joint_states → 此时 TF 树应有 gocator_sensor，创建 FOV 角点 IM
            if not self._fov_im_setup:
                if self._setup_fov_corner_markers():
                    self._fov_im_setup = True
        except Exception as e:
            self.get_logger().warn(f"joint_states 回调异常: {e}")

    def _noise_status_cb(self, request, response):
        """Return the immutable noise settings loaded at process startup."""
        del request
        response.success = True
        response.message = json.dumps(
            self.noise_config.as_dict(), ensure_ascii=False, sort_keys=True
        )
        return response

    def _setup_plate_interactive_marker(self):
        """创建 FOV 角点的 IM server（仅初始化 server，供 FOV 角点使用）"""
        self.im_server = InteractiveMarkerServer(self, 'plate_im')
        self.get_logger().info('InteractiveMarkerServer 已初始化（供 FOV 角点使用）')

    def _get_sensor_pose(self, fallback_fk=False):
        """获取当前传感器在世界系中的位姿 (R_BS, t_BS)
        
        参数:
            fallback_fk: True=只用FK回退（不试TF），False=先TF后FK
        """
        # 非fallback模式：优先用TF
        if not fallback_fk:
            try:
                t = self.tf_buffer.lookup_transform(
                    'world', 'gocator_sensor', rclpy.time.Time())
                tw = t.transform.translation
                qw = t.transform.rotation
                q_arr = np.array([qw.x, qw.y, qw.z, qw.w])
                R = quat_to_matrix(q_arr)
                t_vec = np.array([tw.x, tw.y, tw.z])
                return R, t_vec
            except Exception:
                pass

        # TF不可用或fallback → FK（补 flange→fanuc_flange）
        if self.latest_joints is None:
            return None, None
        T_B_H = forward_kinematics(self.latest_joints)
        R_i, t_i = T_B_H[:3, :3], T_B_H[:3, 3]

        # URDF 中额外有 flange → fanuc_flange (rpy=180°, -90°, 0°)
        R_ff = np.array([[0., 0., 1.],
                         [0., -1., 0.],
                         [1., 0., 0.]])
        R_BS = R_i @ R_ff @ self.scene['R_he']
        t_BS = t_i + R_i @ R_ff @ self.scene['t_he']
        return R_BS, t_BS

    def _setup_fov_corner_markers(self):
        """创建 FOV 平面的4个角点 Interactive Marker

        用 TF 将传感器系角点转换到 world 系（与 FOV 平面渲染一致）。
        如果 TF 还没就绪，返回 False 等下一个 joint_callback 重试。
        （不用 sleep 阻塞 — 那样会卡住 spin 导致 TF 永远收不到）
        """
        from geometry_msgs.msg import Quaternion as GeoQuat

        # TF 查询 — 与 publish_all_markers 完全一致
        try:
            t = self.tf_buffer.lookup_transform(
                'world', 'gocator_sensor', rclpy.time.Time())
            q_arr = np.array([t.transform.rotation.x,
                              t.transform.rotation.y,
                              t.transform.rotation.z,
                              t.transform.rotation.w])
            R_BS = quat_to_matrix(q_arr)
            t_BS = np.array([t.transform.translation.x,
                             t.transform.translation.y,
                             t.transform.translation.z])
        except Exception as e:
            self.get_logger().info(
                f'等待 TF world→gocator_sensor 就绪... ({e})',
                once=True)
            return False

        self.get_logger().info(
            f'gocator_sensor→world: t={t_BS}')

        colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0), (1.0, 1.0, 0.2)]
        labels = ['左下', '右下', '右上', '左上']

        for i, corner_S in enumerate(self.scene['fov_corners_S']):
            # 传感器系 → 世界系
            pos_world = t_BS + R_BS @ corner_S

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
        try:
            R_BS, t_BS = self._get_sensor_pose()
            if R_BS is None:
                self.get_logger().warn('传感器位姿获取失败')
                return
            # world → gocator_sensor
            pt_sensor = R_BS.T @ (pt_world - t_BS)
        except Exception as e:
            self.get_logger().warn(f'传感器位姿获取异常: {e}')
            return
        # 约束到 XZ 平面 (Y=0)
        pt_sensor[1] = 0.0
        self.scene['fov_corners_S'][idx] = pt_sensor
        self.get_logger().info(
            f'角点{idx} → ({pt_sensor[0]:.3f}, {pt_sensor[1]:.3f}, {pt_sensor[2]:.3f}) [传感器系]')
        # 自动保存（下次重启自动加载）
        self._save_fov_calib()

        # 投影 world 位姿（约束 Y=0）
        pt_world_proj = t_BS + R_BS @ pt_sensor
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
            self._collision_publish_counter += 1
            if (
                self._collision_publish_counter
                % self._collision_publish_period
                == 0
            ):
                self._publish_plate_collision()
        except Exception as e:
            self.get_logger().error(f"timer_callback 异常: {e}")

    def publish_all_markers(self, stamp):
        """一次性发布所有场景 Marker + 模拟 GoCator 数据"""
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
            try:
                # The current encoder drives Gazebo.  A hidden perturbation
                # creates the physical flange used to render the profile,
                # while a delayed history sample becomes the pose reported to
                # the calibration algorithm.
                current_flange = forward_kinematics_urdf(self.latest_joints)
                physical_flange = self.noise_model.perturb_flange(
                    current_flange
                )
                delay_s = self.noise_model.sample_sync_delay_s()
                delayed_joints = self.joint_history.delayed(
                    time.monotonic_ns(), delay_s
                )
                if delayed_joints is None:
                    delayed_joints = self.latest_joints
                reported_flange = forward_kinematics_urdf(delayed_joints)

                sensor_transform = physical_flange @ make_transform(
                    self.scene['R_he'], self.scene['t_he']
                )
                R = sensor_transform[:3, :3]
                t_vec = sensor_transform[:3, 3]
                corners_world = [t_vec + R @ c for c in corners_S]

                res = compute_fov_plate_scanline(
                    rotation_sensor_base=R,
                    translation_sensor_base=t_vec,
                    corner=C,
                    normal=n_B,
                    u=u_B,
                    v=v_B,
                    width=w,
                    height=h,
                    fov_corners_S=self.scene['fov_corners_S'])

                # A fixed spatial height field approximates plate flatness.
                # Move each ideal intersection point along the board-normal
                # component that remains inside the physical laser plane.
                if res['has_intersection']:
                    laser_normal = R[:, 1]
                    res['scan_pts_B'] = (
                        self.noise_model.deform_points_in_laser_plane(
                            res['scan_pts_B'],
                            laser_normal=laser_normal,
                            board_normal=n_B,
                            corner=C,
                            board_u=u_B,
                            board_v=v_B,
                            width=w,
                            height=h,
                        )
                    )
                    res['scan_pts_S'] = (
                        R.T @ (res['scan_pts_B'] - t_vec).T
                    ).T
                    endpoint_labels = [
                        label for label, _ in res['endpoints_B']
                    ]
                    if endpoint_labels:
                        endpoint_points = np.asarray(
                            [point for _, point in res['endpoints_B']]
                        )
                        endpoint_points = (
                            self.noise_model.deform_points_in_laser_plane(
                                endpoint_points,
                                laser_normal=laser_normal,
                                board_normal=n_B,
                                corner=C,
                                board_u=u_B,
                                board_v=v_B,
                                width=w,
                                height=h,
                            )
                        )
                        res['endpoints_B'] = list(
                            zip(endpoint_labels, endpoint_points)
                        )
                        endpoint_points_sensor = (
                            R.T @ (endpoint_points - t_vec).T
                        ).T
                        res['endpoints_S'] = list(
                            zip(endpoint_labels, endpoint_points_sensor)
                        )

                # 发布场景 markers（平板 + FOV平面 + 扫描线 + 断点）
                if res['has_intersection']:
                    self.publisher.publish_frame_markers(
                        stamp, R, t_vec,
                        res['scan_pts_B'], res['endpoints_B'],
                        P0=res['line_origin_B'], line_dir=res['line_dir'],
                        frame_id='world',
                        corners_B=corners_world)  # 用校准过的角点
                    # publish_frame_markers 不包含平板，额外发一次
                    self.publisher.publish_scene_markers(stamp)
                else:
                    # 无交线时只发平板 + FOV平面（基于角点）
                    self.publisher.publish_fov_plane(stamp, corners_world, 'world')
                    self.publisher.publish_scene_markers(stamp)

                # 发布模拟 GoCator 数据（传感器系 2D 轮廓点）
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                ]
                if res['has_intersection'] and len(res['scan_pts_S']) >= 3:
                    pts_S = res['scan_pts_S']
                    frame_dropped = self.noise_model.sample_frame_dropout()
                    pts_2d = self.noise_model.corrupt_profile(
                        pts_S, frame_dropped=frame_dropped
                    ).astype(np.float32)
                    # 只保留扫描线范围调试（每秒1次，方便确认有无断点）
                    self._debug_frame += 1
                    if len(pts_S) >= 3 and self._debug_frame % 30 == 0:
                        z_min, z_max = float(pts_S[:, 2].min()), float(pts_S[:, 2].max())
                        x_min, x_max = float(pts_S[:, 0].min()), float(pts_S[:, 0].max())
                        eps_S = res.get('endpoints_S', [])
                        if not eps_S:
                            self.get_logger().info(
                                f"scan: pts={len(pts_S)} x=[{x_min:.3f},{x_max:.3f}] z=[{z_min:.3f},{z_max:.3f}] no ep")
                else:
                    # 无交线 → 发布空点云，让 RViz 清掉旧帧
                    frame_dropped = False
                    pts_2d = np.zeros((0, 3), dtype=np.float32)
                cloud = pc2.create_cloud(
                    Header(stamp=stamp, frame_id='gocator_sensor'),
                    fields, pts_2d)
                self.gocator_pub.publish(cloud)

                # 发布断点信息（传感器系，正确标记 e1/e2）
                ep = Float64MultiArray()
                ep.layout.dim = [
                    MultiArrayDimension(
                        label='endpoints_flange_pose_and_stamp',
                        size=23,
                        stride=23,
                    )
                ]
                ep.layout.data_offset = 0
                # First 9 values retain the legacy endpoint layout.  The
                # remaining values are R_BF row-major and t_BF from the
                # delayed encoder snapshot associated with this profile.
                ep_data = [0.0] * 23
                eps_S = res.get('endpoints_S', [])
                e1_valid, e2_valid = 0.0, 0.0
                for et, pt in eps_S:
                    noisy_point, valid = self.noise_model.corrupt_endpoint(
                        pt, frame_dropped=frame_dropped
                    )
                    if not valid:
                        continue
                    if et == 'e1':
                        ep_data[1:4] = noisy_point; e1_valid = 1.0
                    elif et == 'e2':
                        ep_data[5:8] = noisy_point; e2_valid = 1.0
                ep_data[0] = e1_valid + e2_valid
                ep_data[4] = e1_valid
                ep_data[8] = e2_valid
                ep_data[9:18] = reported_flange[:3, :3].reshape(-1).tolist()
                ep_data[18:21] = reported_flange[:3, 3].tolist()
                # Match this endpoint/encoder snapshot to the PointCloud2
                # frame without relying on DDS callback arrival order.
                ep_data[21] = float(stamp.sec)
                ep_data[22] = float(stamp.nanosec)
                ep.data = ep_data
                self.endpoint_pub.publish(ep)

            except Exception as e:
                self.get_logger().warn(
                    f'TF/scene publish error: {e}',
                    throttle_duration_sec=2.0)
                # TF 失败时只发平板（不画 FOV，因为没 TF 算不准位置）
                self.publisher.publish_scene_markers(stamp)
        else:
            # 无 joint 数据时只发平板
            self.publisher.publish_scene_markers(stamp)

    def _publish_plate_collision(self):
        """Keep the Gazebo plate and MoveIt collision world geometrically aligned."""
        scene = self.scene
        collision = CollisionObject()
        collision.header.frame_id = 'world'
        collision.id = 'calibration_plate'
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [scene['w'], scene['h'], 0.01]
        pose = Pose()
        center = (
            scene['C']
            + 0.5 * scene['w'] * scene['u_B']
            + 0.5 * scene['h'] * scene['v_B']
            - 0.005 * scene['n_B']
        )
        pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        quaternion = matrix_to_quat(scene['R_plate'])
        pose.orientation = Quaternion(
            x=float(quaternion[0]),
            y=float(quaternion[1]),
            z=float(quaternion[2]),
            w=float(quaternion[3]),
        )
        collision.primitives = [primitive]
        collision.primitive_poses = [pose]
        collision.operation = CollisionObject.ADD
        self.collision_pub.publish(collision)

    def _load_factory_fov_corners(self):
        """Load the immutable laser-window geometry shipped with the package."""
        fallback = [
            [-0.019891060683164145, 0.0, -0.2892952979686891],
            [-0.020634857260870637, 0.0, -0.2922409434429649],
            [0.22, 0.0, 0.82],
            [-0.22, 0.0, 0.82],
        ]
        try:
            with open(self._fov_factory_path, 'r', encoding='utf-8') as stream:
                data = json.load(stream)
            corners = np.asarray(data['fov_corners_S'], dtype=float)
            if corners.shape != (4, 3):
                raise ValueError('fov_corners_S must have shape (4, 3)')
            return [corner.copy() for corner in corners]
        except Exception as error:
            self.get_logger().error(
                f'加载出厂 FOV 几何失败，使用内置回退值: {error}')
            return [np.asarray(corner, dtype=float) for corner in fallback]

    def _load_fov_calib(self):
        """Load an optional runtime override saved by interactive dragging."""
        try:
            with open(self._fov_calib_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            corners = [np.array(c, dtype=float) for c in data['fov_corners_S']]
            if len(corners) == 4:
                self.scene['fov_corners_S'] = corners
                self.get_logger().info(
                    f'已加载运行时 FOV 校准: '
                    f'{[f"[{c[0]:.3f},{c[1]:.3f},{c[2]:.3f}]" for c in corners]}')
        except FileNotFoundError:
            self.get_logger().info('无运行时 FOV 校准，使用出厂激光窗口几何')
        except Exception as e:
            self.get_logger().warn(f'加载 FOV 校准失败: {e}')

    def _save_fov_calib(self):
        """保存当前 FOV 角点到文件"""
        corners = self.scene['fov_corners_S']
        data = {
            'fov_corners_S': [c.tolist() for c in corners],
        }
        try:
            os.makedirs(os.path.dirname(self._fov_calib_path), exist_ok=True)
            with open(self._fov_calib_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.get_logger().info(f'FOV 校准已保存 -> {self._fov_calib_path}')
        except Exception as e:
            self.get_logger().warn(f'保存 FOV 校准失败: {e}')

    def _query_fov_corners_cb(self, req, res):
        """返回当前 FOV 4个角点坐标（传感器系）"""
        corners = self.scene['fov_corners_S']
        msgs = [f'corner_{i}: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]'
                for i, c in enumerate(corners)]
        msg = '\n'.join(msgs)
        self.get_logger().info(f'FOV角点查询:\n{msg}')
        res.success = True
        res.message = msg
        return res

    def _save_fov_calib_cb(self, req, res):
        """手动保存当前 FOV 校准"""
        self._save_fov_calib()
        res.success = True
        res.message = f'已保存到 {self._fov_calib_path}'
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
