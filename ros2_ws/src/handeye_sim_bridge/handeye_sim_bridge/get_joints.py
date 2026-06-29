#!/usr/bin/env python3
"""Get current joint states"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

rclpy.init()
node = Node('get_joints')

joints = None
def cb(msg):
    global joints
    joints = np.array(msg.position)
    names = msg.name
    print("Joint positions (rad):")
    for n, j in zip(names, joints):
        print(f"  {n}: {j:.4f} rad = {np.rad2deg(j):.1f} deg")
    rclpy.shutdown()

sub = node.create_subscription(JointState, '/joint_states', cb, 1)
rclpy.spin(node)
rclpy.shutdown()
