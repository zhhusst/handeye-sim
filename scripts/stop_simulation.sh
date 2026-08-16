#!/usr/bin/env bash
echo "Stopping all sim processes..."
TARGET_PATTERNS=(
    "^ruby .*/gz sim( |$)"
    "^gz sim( |$)"
    "^/usr/bin/python3 /opt/ros/.*/bin/ros2 run ros_gz_bridge parameter_bridge( |$)"
    "^/usr/bin/python3 /opt/ros/.*/bin/ros2 run ros_gz_sim create( |$)"
    "^/usr/bin/python3 /opt/ros/.*/bin/ros2 run robot_state_publisher robot_state_publisher( |$)"
    "^/usr/bin/python3 /opt/ros/.*/bin/ros2 run tf2_ros static_transform_publisher( |$)"
    "^/usr/bin/python3 /opt/ros/.*/bin/ros2 run moveit_ros_move_group move_group( |$)"
    "^/usr/bin/python3 /opt/ros/.*/bin/ros2 run rviz2 rviz2( |$)"
    "^/opt/ros/.*/lib/(ros_gz_bridge|robot_state_publisher|tf2_ros|moveit_ros_move_group|rviz2)/"
    "^/usr/bin/python3 /workspace/ros2_ws/install/handeye_sim_bridge/lib/handeye_sim_bridge/(scene_publisher_node|profile_endpoint_detector|srdf_publisher_node|profile_viz|seed_collection|active_calibration|active_calibration_sim)( |$)"
    "^/usr/bin/python3 /workspace/ros2_ws/install/handeye_sim_backend/lib/handeye_sim_backend/scene_publisher( |$)"
    "^python3 /workspace/scripts/calibration_console.py( |$)"
    "^Xvfb :99( |$)"
    "^openbox --sm-disable( |$)"
    "^x11vnc .* -rfbport 15900( |$)"
    "websockify .*127\\.0\\.0\\.1:6080 .*127\\.0\\.0\\.1:15900"
)
for pattern in "${TARGET_PATTERNS[@]}"; do
    pkill -TERM -f "$pattern" 2>/dev/null
done
sleep 1
for pattern in "${TARGET_PATTERNS[@]}"; do
    pkill -KILL -f "$pattern" 2>/dev/null
done
tmux kill-session -t handeye_sim 2>/dev/null || true
echo "All stopped"
