#!/bin/bash
# =============================================================================
# 手眼标定仿真 — 构建 + 运行
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM_ROOT="$SCRIPT_DIR"
ROS_WS="$SIM_ROOT/ros2_ws"

echo "=========================================="
echo "  手眼标定仿真 — Docker 构建与运行"
echo "=========================================="

# 第一步: 构建 Docker 镜像
echo ""
echo "[1/3] 构建 Docker 镜像..."
cd "$SIM_ROOT/docker"
docker compose build

# 第二步: 在容器内构建 ROS2 包
echo ""
echo "[2/3] 构建 ROS2 包..."
docker compose run --rm handeye-sim bash -c "
    source /opt/ros/jazzy/setup.bash
    cd /workspace/ros2_ws
    colcon build --symlink-install
    echo 'ROS2 包构建完成'
"

echo ""
echo "[3/3] 启动仿真..."
# 启动仿真 (单独运行容器, 保持前台)
echo "运行: docker compose up handeye-sim"
echo ""
echo "在容器内执行:"
echo "  source /workspace/ros2_ws/install/setup.bash"
echo "  ros2 launch handeye_sim_bridge handeye_sim.launch.py"
echo ""
echo "或在宿主机直接:"
echo "  cd $SIM_ROOT/docker && docker compose run --rm handeye-sim"
echo "  然后在新终端执行上述 ros2 launch 命令"
