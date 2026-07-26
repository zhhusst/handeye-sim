#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
cd /workspace/ros2_ws
colcon build --symlink-install
