#!/usr/bin/env python3
"""监听检测节点诊断话题"""
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

rclpy.init()
node = Node("diag_listener")
received = []

def cb(msg):
    try:
        d = json.loads(msg.data)
        received.append(d)
    except Exception as e:
        received.append({"raw_error": str(e), "raw": msg.data[:200]})

node.create_subscription(String, "/profile_endpoint_detector/diagnostics", cb, 10)
import time
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(node, timeout_sec=0.1)
if received:
    d = received[-1]
    for k, v in d.items():
        print(f"{k}: {v}")
else:
    print("no diagnostics received")
node.destroy_node()
rclpy.shutdown()
