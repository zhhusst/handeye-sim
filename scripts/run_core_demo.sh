#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/ros2_ws/src/handeye_calibration_core
python3 -m calibration_pipeline.cli "$@"
