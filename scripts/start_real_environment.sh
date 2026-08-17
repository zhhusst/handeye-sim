#!/usr/bin/env bash
# Start the real measurement chain, optionally with a safety-gated motion bridge.
set -eo pipefail

# Must match start_calibration.sh: all ROS 2 processes in this workspace use
# the same UDP-only Fast DDS profile so the console, the active-calibration
# node, MoveIt and RViz all share one DDS domain.  Without this, RViz started
# here cannot see markers/services published by nodes launched from the
# calibration console (different DDS transport -> discovery isolation).
export FASTDDS_DEFAULT_PROFILES_FILE=/workspace/ros2_ws/src/handeye_sim_bridge/config/fastdds_udp_only.xml

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

# --- MoveIt2 move_group for real hardware ----------------------------------
# The real rig has no Gazebo.  move_group runs with use_sim_time=false and
# does NOT manage controllers (execution goes through the TP motion bridge
# action, not a Gazebo trajectory controller).
URDF_PATH="/workspace/urdf/calib_robot.urdf"
SRDF_PATH="/workspace/ros2_ws/src/handeye_sim_bridge/config/fanuc.srdf"
if [[ ! -f "$URDF_PATH" ]]; then echo "URDF not found: $URDF_PATH" >&2; exit 1; fi
if [[ ! -f "$SRDF_PATH" ]]; then echo "SRDF not found: $SRDF_PATH" >&2; exit 1; fi

mkdir -p /tmp/ros_params
cat > /tmp/ros_params/mg_params_real.yaml << 'YAMLEOF'
/**:
  ros__parameters:
    robot_description: |
YAMLEOF
echo "$(sed 's/^/      /' "$URDF_PATH")" >> /tmp/ros_params/mg_params_real.yaml
cat >> /tmp/ros_params/mg_params_real.yaml << 'YAMLEOF'

    robot_description_semantic: |
YAMLEOF
echo "$(sed 's/^/      /' "$SRDF_PATH")" >> /tmp/ros_params/mg_params_real.yaml
cat >> /tmp/ros_params/mg_params_real.yaml << 'YAMLEOF'

    planning_pipelines: ["ompl"]
    default_planning_pipeline: ompl
    ompl.planning_plugins: ["ompl_interface/OMPLPlanner"]
    ompl.request_adapters: ["default_planning_request_adapters/ResolveConstraintFrames",
      "default_planning_request_adapters/ValidateWorkspaceBounds",
      "default_planning_request_adapters/CheckStartStateBounds",
      "default_planning_request_adapters/CheckStartStateCollision"]
    ompl.response_adapters: ["default_planning_response_adapters/AddTimeOptimalParameterization",
      "default_planning_response_adapters/ValidateSolution",
      "default_planning_response_adapters/DisplayMotionPath"]
    ompl.RRTConnect.type: geometric::RRTConnect
    ompl.RRTConnect.range: 0.0

    robot_description_kinematics:
      arm:
        kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
        kinematics_solver_search_resolution: 0.005
        kinematics_solver_timeout: 0.05
        kinematics_solver_attempts: 3
        planner_configs: [RRTConnect]
        default_planner_config: RRTConnect

    robot_description_planning:
      joint_limits:
        J1_joint:
          has_velocity_limits: true
          max_velocity: 1.57
          has_acceleration_limits: true
          max_acceleration: 3.14
        J2_joint:
          has_velocity_limits: true
          max_velocity: 1.57
          has_acceleration_limits: true
          max_acceleration: 3.14
        J3_joint:
          has_velocity_limits: true
          max_velocity: 1.57
          has_acceleration_limits: true
          max_acceleration: 3.14
        J4_joint:
          has_velocity_limits: true
          max_velocity: 2.09
          has_acceleration_limits: true
          max_acceleration: 4.19
        J5_joint:
          has_velocity_limits: true
          max_velocity: 2.09
          has_acceleration_limits: true
          max_acceleration: 4.19
        J6_joint:
          has_velocity_limits: true
          max_velocity: 3.14
          has_acceleration_limits: true
          max_acceleration: 6.28

    # Real rig: do NOT let MoveIt take over controllers; execution is routed
    # through the TP motion bridge action by the active-calibration node.
    moveit_manage_controllers: false
    use_sim_time: false
YAMLEOF

# Static TF: world -> base_link.  RViz uses the world frame as its fixed
# frame and the collision object / markers are published in base_link, so
# without this transform the plate box cannot be displayed on the real rig.
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 1 world base_link &
sleep 1

echo "  -> Starting MoveIt2 move_group (real, use_sim_time=false)..."
ros2 run moveit_ros_move_group move_group \
    --ros-args --params-file /tmp/ros_params/mg_params_real.yaml \
    > /tmp/move_group_real.log 2>&1 &
MOVE_GROUP_PID=$!
sleep 6

exec ros2 launch fanuc_gocator_bridge calibration_with_motion.launch.py \
    "motion_mode:=$MOTION_MODE" \
    "motion_writes_enabled:=$MOTION_WRITES" \
    "${LAUNCH_ARGS[@]}"
