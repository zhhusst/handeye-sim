#!/usr/bin/env python3
"""跨 DDS 测试: 用 FASTDDS udp_only 发布 marker, 无 FASTDDS 的订阅者收。
"""
import rclpy
import time
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

def main():
    rclpy.init()
    node = Node("cross_dds_pub")
    marker_qos = QoSProfile(
        depth=10,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )
    pub = node.create_publisher(MarkerArray, "/calib/cross_dds_test", marker_qos)
    arr = MarkerArray()
    m = Marker()
    m.header.frame_id = "world"
    m.ns = "test"
    m.id = 1
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = 1.0; m.pose.position.y = 0.0; m.pose.position.z = 0.5
    m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1
    m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0
    arr.markers.append(m)
    for i in range(20):
        pub.publish(arr)
        time.sleep(0.2)
    print("published 20 cross-dds markers")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
