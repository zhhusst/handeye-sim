#!/bin/bash
# post-create.sh — 容器启动后自动执行
set -e

echo "===== 手眼标定仿真 — 初始化 ====="

# ROS2 环境
source /opt/ros/jazzy/setup.bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

# 构建 ROS2 包
cd /workspace/ros2_ws
colcon build --symlink-install
echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc

echo ""
echo "===== 初始化完成 ====="
echo "快速启动:"
echo "  ros2 launch handeye_sim_bridge handeye_sim.launch.py"
echo ""
