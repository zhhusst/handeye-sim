#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash

echo "=== 生成所有参数文件 ==="
python3 /workspace/ros2_ws/gen_all_params.py

echo "=== 启动 robot_state_publisher ==="
/opt/ros/jazzy/lib/robot_state_publisher/robot_state_publisher \
  --ros-args --params-file /tmp/rsp.yaml &
sleep 1

echo "=== 启动 ros2_control (mock_components) ==="
/opt/ros/jazzy/lib/controller_manager/ros2_control_node \
  --ros-args --params-file /tmp/cm.yaml &
sleep 2

echo "=== 加载 joint_state_broadcaster ==="
/opt/ros/jazzy/lib/controller_manager/spawner joint_state_broadcaster \
  --controller-manager /controller_manager \
  -p /tmp/jsb.yaml &
sleep 1

echo "=== 加载 joint_trajectory_controller ==="
/opt/ros/jazzy/lib/controller_manager/spawner joint_trajectory_controller \
  --controller-manager /controller_manager \
  -p /tmp/jtc.yaml &
sleep 1

echo "=== 启动 move_group (双参数文件) ==="
/opt/ros/jazzy/lib/moveit_ros_move_group/move_group \
  --ros-args \
  --params-file /tmp/move_group_main.yaml \
  --params-file /tmp/ompl_pipeline.yaml &
sleep 2

echo "=== 启动 scene_publisher ==="
python3 /workspace/ros2_ws/install/handeye_sim_bridge/lib/handeye_sim_bridge/scene_publisher_node.py &
sleep 0.5

echo "=== 启动 RViz2 ==="
/opt/ros/jazzy/lib/rviz2/rviz2 \
  -d /workspace/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge/rviz/handeye_sim_moveit.rviz \
  --ros-args --params-file /tmp/rviz.yaml &

echo "=== 全部启动完成 ==="
wait
