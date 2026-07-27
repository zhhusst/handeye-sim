#!/usr/bin/env bash
# Keyboard-driven simulation calibration workflow for the second terminal.
set -eo pipefail

cd /workspace
source /opt/ros/jazzy/setup.bash
export FASTDDS_DEFAULT_PROFILES_FILE=/workspace/ros2_ws/src/handeye_sim_bridge/config/fastdds_udp_only.xml
if [[ ! -f /workspace/ros2_ws/install/setup.bash ]]; then
    echo "ROS 2 工作区尚未构建，请先运行 ./scripts/build.sh。" >&2
    exit 1
fi
source /workspace/ros2_ws/install/setup.bash

exec python3 /workspace/scripts/calibration_console.py "$@"
