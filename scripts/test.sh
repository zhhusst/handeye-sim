#!/usr/bin/env bash
set -eo pipefail

cd /workspace
source /opt/ros/jazzy/setup.bash
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/workspace/ros2_ws/src/handeye_sim_bridge${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest
