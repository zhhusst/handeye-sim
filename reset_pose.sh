#!/bin/bash
# Reset robot to initial pose
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 action send_goal /scaled_joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{
  trajectory: {
    joint_names: [J1_joint, J2_joint, J3_joint, J4_joint, J5_joint, J6_joint],
    points: [{
      positions: [-0.2357, -0.0364, -0.6328, -0.4062, -1.0504, 0.8788],
      time_from_start: {sec: 3, nanosec: 0}
    }]
  }
}" --feedback 2>&1
