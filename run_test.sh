#!/bin/bash
# Full test: reset robot → run auto_calib_v2 → save results
set -e

source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash

echo "=== Resetting robot to initial pose ==="
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{
  trajectory: {
    joint_names: [J1_joint, J2_joint, J3_joint, J4_joint, J5_joint, J6_joint],
    points: [{
      positions: [-0.2357, -0.0364, -0.6328, -0.4062, -1.0504, 0.8788],
      time_from_start: {sec: 3, nanosec: 0}
    }]
  }
}" --feedback 2>&1 | tail -3
sleep 4  # wait for robot to settle

echo ""
echo "=== Running auto_calib_v2 ==="
# Pipe ENTER (to start) and 'w' (to save) after a delay
( sleep 1; echo ""; sleep 240; echo "w"; sleep 2; echo "q" ) | \
  timeout 300 auto_calib_v2 2>&1 | tee /tmp/auto_calib_output.txt

echo ""
echo "=== Done ==="
grep -E "R_err|t_err|tilt|z_S tilt" /tmp/auto_calib_output.txt | tail -5
