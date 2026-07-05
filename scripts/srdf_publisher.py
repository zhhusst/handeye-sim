#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SRDFPublisher(Node):
    def __init__(self):
        super().__init__('srdf_publisher')
        self.pub = self.create_publisher(String, '/robot_description_semantic', 1)
        timer = self.create_timer(0.5, self.publish_srdf)
        self.get_logger().info('SRDF Publisher started')
        # Also declare the parameter
        self.declare_parameter('robot_description_semantic', '')

    def publish_srdf(self):
        srdf = self.get_parameter('robot_description_semantic').value
        if srdf:
            msg = String()
            msg.data = srdf
            self.pub.publish(msg)

def main():
    rclpy.init()
    node = SRDFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
