#!/usr/bin/env python3
"""Read and display /gocator/profile data"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

rclpy.init()
node = Node('read_profile')

data = None
def cb(msg):
    global data
    data = msg
    rclpy.shutdown()

sub = node.create_subscription(PointCloud2, '/gocator/profile', cb, 1)
rclpy.spin(node)

if data is not None:
    print(f"Frame: {data.header.frame_id}")
    print(f"Width: {data.width}, Height: {data.height}")
    print(f"Point step: {data.point_step}, Row step: {data.row_step}")
    
    points = list(pc2.read_points(data, field_names=('x', 'y', 'z'), skip_nans=True))
    pts = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float32)
    print(f"\nTotal points: {len(pts)}")
    if len(pts) > 0:
        print(f"X range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
        print(f"Y range: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] (should be ~0)")
        print(f"Z range: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
        print(f"\nFirst 5 points (X, Y, Z):")
        for i in range(min(5, len(pts))):
            print(f"  [{pts[i,0]:.4f}, {pts[i,1]:.4f}, {pts[i,2]:.4f}]")
        print(f"\nLast 5 points:")
        for i in range(max(0, len(pts)-5), len(pts)):
            print(f"  [{pts[i,0]:.4f}, {pts[i,1]:.4f}, {pts[i,2]:.4f}]")

rclpy.shutdown()
