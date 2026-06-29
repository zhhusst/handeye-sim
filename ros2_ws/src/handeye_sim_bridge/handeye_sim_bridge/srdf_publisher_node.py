#!/usr/bin/env python3
"""srdf_publisher_node.py — 发布 robot_description_semantic 话题（transient_local 持久性）

MoveIt MotionPlanning RViz 显示插件需要订阅 /robot_description_semantic 话题，
且需用 TRANSIENT_LOCAL durability（类似 ROS1 latch），确保后来订阅的节点也能收到。

用法:
  ros2 run handeye_sim_bridge srdf_publisher_node
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
import sys


def main():
    rclpy.init()
    node = Node('srdf_publisher')

    # 声明参数（带默认路径）
    node.declare_parameter('srdf_path', '')

    srdf_path = node.get_parameter('srdf_path').value
    if not srdf_path:
        # 默认路径
        srdf_path = '/workspace/ros2_ws/src/handeye_sim_bridge/config/fanuc.srdf'

    # 读取 SRDF 文件
    try:
        with open(srdf_path, 'r') as f:
            srdf_content = f.read()
    except FileNotFoundError:
        node.get_logger().error(f'SRDF file not found: {srdf_path}')
        sys.exit(1)

    node.get_logger().info(f'Loaded SRDF ({len(srdf_content)} chars) from {srdf_path}')

    # TRANSIENT_LOCAL 持久性 + RELIABLE
    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    pub = node.create_publisher(
        String,
        '/robot_description_semantic',
        qos,
    )

    # 发布一次（TRANSIENT_LOCAL 会缓存给后来订阅者）
    msg = String()
    msg.data = srdf_content
    pub.publish(msg)
    node.get_logger().info(f'Published /robot_description_semantic (transient_local)')

    # 持续运行保持话题存活（TRANSIENT_LOCAL 需要 publisher 节点存在）
    rclpy.spin(node)


if __name__ == '__main__':
    main()
