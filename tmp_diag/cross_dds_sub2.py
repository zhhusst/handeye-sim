#!/usr/bin/env python3
"""无 FASTDDS 订阅者: 收 /calib/cross_dds_test, 持续 8 秒。"""
import rclpy
import time
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

def main():
    rclpy.init()
    node = Node("cross_dds_sub_no_fastdds")
    got = {"count": 0}
    def cb(msg):
        got["count"] += 1
        print(f"收到 marker 数组 {got['count']}, markers={len(msg.markers)}", flush=True)
    node.create_subscription(MarkerArray, "/calib/cross_dds_test", cb, 10)
    deadline = time.time() + 8
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
    print(f"共收到: {got['count']}", flush=True)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
