#!/usr/bin/env python3
"""Check TF tree"""
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

rclpy.init()
node = Node('tf_checker')

def tf_cb(msg):
    for t in msg.transforms:
        print("TF: {} -> {}: pos=({:.3f}, {:.3f}, {:.3f})".format(
            t.header.frame_id, t.child_frame_id,
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z))

def static_cb(msg):
    for t in msg.transforms:
        print("STATIC: {} -> {}".format(t.header.frame_id, t.child_frame_id))

sub = node.create_subscription(TFMessage, '/tf', tf_cb, 10)
sub_static = node.create_subscription(TFMessage, '/tf_static', static_cb, 10)

print("Waiting for TF messages...")
for _ in range(5):
    rclpy.spin_once(node, timeout_sec=1.0)
    print("---")
rclpy.shutdown()
