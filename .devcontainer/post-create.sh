#!/usr/bin/env bash
set -eo pipefail

echo "===== 手眼标定仿真 — 初始化 ====="

# ROS2 环境
source /opt/ros/jazzy/setup.bash
grep -qxF "source /opt/ros/jazzy/setup.bash" ~/.bashrc ||
  echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

# 构建 ROS2 包
cd /workspace/ros2_ws
colcon build --symlink-install
grep -qxF "source /workspace/ros2_ws/install/setup.bash" ~/.bashrc ||
  echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc

echo ""
echo "===== 初始化完成 ====="
echo "快速启动:"
echo "  ros2 launch handeye_sim_bridge handeye_sim.launch.py"
echo ""
