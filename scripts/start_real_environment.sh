#!/usr/bin/env bash
# Start the real measurement chain, optionally with a safety-gated motion bridge.
set -eo pipefail

LAUNCH_ARGS=()
MOTION_MODE="disabled"
while [[ $# -gt 0 ]]; do
    arg="$1"
    shift
    case "$arg" in
        --no-rviz) LAUNCH_ARGS+=("start_rviz:=false") ;;
        --motion-mode)
            if [[ $# -eq 0 ]]; then
                echo "--motion-mode 需要参数。" >&2
                exit 2
            fi
            MOTION_MODE="$1"
            shift
            ;;
        --motion-mode=*) MOTION_MODE="${arg#*=}" ;;
        -h|--help)
            echo "用法: ./scripts/start_environment.sh real [--no-rviz] [--motion-mode disabled|plan_only|step_confirm|automatic] [robot_ip:=IP] [sensor_ip:=IP]"
            exit 0
            ;;
        *) LAUNCH_ARGS+=("$arg") ;;
    esac
done

case "$MOTION_MODE" in
    disabled|plan_only) MOTION_WRITES="false" ;;
    step_confirm|automatic) MOTION_WRITES="true" ;;
    *)
        echo "无效 motion mode: $MOTION_MODE" >&2
        exit 2
        ;;
esac

cd /workspace
source /opt/ros/jazzy/setup.bash
if [[ ! -f /workspace/ros2_ws/install/setup.bash ]]; then
    echo "ROS 2工作区尚未构建，请先运行 ./scripts/build.sh。" >&2
    exit 1
fi
source /workspace/ros2_ws/install/setup.bash

if ! python3 -c 'import pycomm3' >/dev/null 2>&1; then
    echo "缺少真机通信依赖 pycomm3。" >&2
    echo "请重建开发容器，或执行：" >&2
    echo "  apt-get update && apt-get install -y python3-pip" >&2
    echo "  python3 -m pip install --break-system-packages 'pycomm3>=1.2,<2'" >&2
    exit 1
fi

echo "=============================================================="
echo " FANUC + Gocator 真机环境：motion_mode=$MOTION_MODE"
if [[ "$MOTION_WRITES" == "true" ]]; then
    echo " 运动桥具备写权限，但启动时保持 DISARMED；必须由第二终端明确解锁。"
    echo " TP要求：PC_TRACK_ALL正在运行，UF/UT=1/1，R[100]=0。"
else
    echo " 当前禁止运动写入。"
fi
echo " 激光默认关闭；J23未经验证时只发布原始控制器关节。"
echo "=============================================================="

if [[ "$MOTION_MODE" == "disabled" ]]; then
    exec ros2 launch fanuc_gocator_bridge observe_only.launch.py "${LAUNCH_ARGS[@]}"
fi

exec ros2 launch fanuc_gocator_bridge calibration_with_motion.launch.py \
    "motion_mode:=$MOTION_MODE" \
    "motion_writes_enabled:=$MOTION_WRITES" \
    "${LAUNCH_ARGS[@]}"
