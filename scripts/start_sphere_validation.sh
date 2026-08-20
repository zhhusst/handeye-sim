#!/usr/bin/env bash
# Run the independent precision-sphere experiment in the second terminal.
# The default interactive choice is the manually stepped moving scan required
# by evaluation method 2; pass --mode stationary for the legacy single-section
# repeatability experiment.
set -eo pipefail

cd /workspace
source /opt/ros/jazzy/setup.bash
if [[ ! -f /workspace/ros2_ws/install/setup.bash ]]; then
    echo "ROS 2工作区尚未构建，请先运行 ./scripts/build.sh。" >&2
    exit 1
fi
source /workspace/ros2_ws/install/setup.bash

exec python3 /workspace/scripts/sphere_validation_console.py "$@"
