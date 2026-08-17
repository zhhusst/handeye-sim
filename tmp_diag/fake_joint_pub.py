#!/usr/bin/env python3
"""发布模拟关节状态（用种子参考位姿），触发 active_calibration 初始化。"""
import json
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
joints = seeds["seeds"][0]["joints"]
names = ["J1_joint", "J2_joint", "J3_joint", "J4_joint", "J5_joint", "J6_joint"]

rclpy.init()
node = Node("fake_joint_pub")
pub = node.create_publisher(JointState, "/joint_states", 10)
msg = JointState()
msg.name = names
msg.position = joints
msg.velocity = [0.0] * 6
msg.effort = [0.0] * 6
for _ in range(8):
    msg.header.stamp = node.get_clock().now().to_msg()
    pub.publish(msg)
    time.sleep(0.3)
print("published 8 joint states")
rclpy.shutdown()
