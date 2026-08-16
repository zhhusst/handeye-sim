#!/usr/bin/env bash
set -eo pipefail

cd /workspace
source /opt/ros/jazzy/setup.bash
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/workspace/ros2_ws/src/handeye_calibration_core:/workspace/ros2_ws/src/fanuc_m20id25_support:/workspace/ros2_ws/src/handeye_sim_bridge:/workspace/ros2_ws/src/handeye_sim_backend${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest
