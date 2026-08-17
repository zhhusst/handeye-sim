#!/usr/bin/env python3
"""模拟 RViz: 用 Volatile + Reliable QoS 订阅 /calib/markers, 持续监听。"""
import rclpy
import time
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

def main():
    rclpy.init()
    node = Node("rviz_like_sub")
    qos = QoSProfile(
        depth=10,
        durability=QoSDurabilityPolicy.VOLATILE,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )
    got = {"count": 0}
    def cb(msg):
        for m in msg.markers:
            got["count"] += 1
            print(f"[{got['count']}] ns={m.ns} id={m.id} frame={m.header.frame_id} "
                  f"pos=({m.pose.position.x:.3f},{m.pose.position.y:.3f},{m.pose.position.z:.3f}) "
                  f"scale=({m.scale.x:.3f},{m.scale.y:.3f},{m.scale.z:.3f})", flush=True)
    node.create_subscription(MarkerArray, "/calib/markers", cb, qos)
    deadline = time.time() + 60
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
    print(f"60s 内共收到: {got['count']}", flush=True)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
