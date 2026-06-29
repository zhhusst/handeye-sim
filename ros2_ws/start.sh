#!/bin/bash
# 一键启动 Gazebo + MoveIt2 手眼标定仿真
set -e

USE_TMUX=false
NO_RVIZ=false
for arg in "$@"; do
    case "$arg" in
        --tmux) USE_TMUX=true ;;
        --no-rviz) NO_RVIZ=true ;;
        -h|--help)
            echo "用法: ./start.sh [--tmux] [--no-rviz]"
            echo "  --tmux    分屏模式（可看各组件实时日志）"
            echo "  --no-rviz  不启动 RViz"
            exit 0 ;;
    esac
done

cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash

URDF_PATH=/workspace/urdf/calib_robot.urdf
SRDF_PATH=/workspace/ros2_ws/src/handeye_sim_bridge/config/fanuc.srdf
GZ_CTRL_CONFIG=/workspace/ros2_ws/src/handeye_sim_bridge/config/gz_controllers.yaml
SCENE_EXE=/workspace/ros2_ws/install/handeye_sim_bridge/lib/handeye_sim_bridge/scene_publisher_node.py
SRDF_PUB_EXE=/workspace/ros2_ws/install/handeye_sim_bridge/lib/handeye_sim_bridge/srdf_publisher_node.py

./stop.sh 2>/dev/null || true
sleep 1

echo "=========================================="
echo "  Gazebo + MoveIt2  手眼标定仿真 "
echo "=========================================="

if ! ros2 pkg list 2>/dev/null | grep -q ros_gz_bridge; then
    apt-get update -qq && apt-get install -y -qq ros-jazzy-ros-gz-bridge
fi
if [ ! -f "$URDF_PATH" ]; then echo "URDF not found"; exit 1; fi
if [ ! -f "$SRDF_PATH" ]; then echo "SRDF not found"; exit 1; fi

mkdir -p /tmp/ros_params

# robot_state_publisher params
cat > /tmp/ros_params/rsp_params.yaml << 'EOF'
/**:
  ros__parameters:
    robot_description: |
EOF
echo "$(sed 's/^/      /' "$URDF_PATH")" >> /tmp/ros_params/rsp_params.yaml
cat >> /tmp/ros_params/rsp_params.yaml << 'EOF'
    publish_frequency: 30.0
    use_sim_time: true
EOF

# move_group params
cat > /tmp/ros_params/mg_params.yaml << 'YAMLEOF'
/**:
  ros__parameters:
    robot_description: |
YAMLEOF
echo "$(sed 's/^/      /' "$URDF_PATH")" >> /tmp/ros_params/mg_params.yaml
cat >> /tmp/ros_params/mg_params.yaml << 'YAMLEOF'

    robot_description_semantic: |
YAMLEOF
echo "$(sed 's/^/      /' "$SRDF_PATH")" >> /tmp/ros_params/mg_params.yaml
cat >> /tmp/ros_params/mg_params.yaml << 'YAMLEOF'

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

    moveit_manage_controllers: true
    moveit_controller_manager: moveit_simple_controller_manager/MoveItSimpleControllerManager
    moveit_simple_controller_manager:
      controller_names: ["joint_trajectory_controller"]
      joint_trajectory_controller:
        type: FollowJointTrajectory
        joints: ["J1_joint","J2_joint","J3_joint","J4_joint","J5_joint","J6_joint"]
        action_ns: follow_joint_trajectory
        default: true
    use_sim_time: true
YAMLEOF

echo "  params generated"

# Generate kinematics.yaml (shared with RViz MotionPlanning display)
cat > /tmp/ros_params/kinematics.yaml << 'KEOF'
/**:
  ros__parameters:
    robot_description_kinematics:
      arm:
        kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
        kinematics_solver_search_resolution: 0.005
        kinematics_solver_timeout: 0.05
        kinematics_solver_attempts: 3
KEOF
echo "  kinematics.yaml generated"

if $USE_TMUX; then

# ==================== TMUX MODE ====================
echo ""
echo "tmux mode (Ctrl+B+arrows to switch panes)"

tmux new-session -d -s handeye_sim -x 200 -y 80
tmux rename-window -t handeye_sim:0 'handeye-sim'
tmux split-window -h -t handeye_sim:0
tmux split-window -v -t handeye_sim:0.0
tmux split-window -v -t handeye_sim:0.2

# Pane 0: Gazebo
tmux send-keys -t handeye_sim:0.0 "export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib && gz sim -r -v 4 empty.sdf" Enter
sleep 2

# Pane 1: Launch components sequentially
tmux send-keys -t handeye_sim:0.1 "echo 'Starting components...'" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file /tmp/ros_params/rsp_params.yaml" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "gz sdf -p '$URDF_PATH' > /tmp/robot_ready.sdf && ros2 run ros_gz_sim create -file /tmp/robot_ready.sdf -name fanuc_robot -world empty -allow_renaming true" Enter
sleep 3
tmux send-keys -t handeye_sim:0.1 "ros2 param load /controller_manager '$GZ_CTRL_CONFIG' 2>&1 | head -3" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 run controller_manager spawner joint_state_broadcaster" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 run controller_manager spawner joint_trajectory_controller --param-file '$GZ_CTRL_CONFIG'" Enter
sleep 2
tmux send-keys -t handeye_sim:0.1 "python3 '$SRDF_PUB_EXE' --ros-args -p use_sim_time:=true" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "python3 '$SCENE_EXE' --ros-args -p use_sim_time:=true" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "echo 'Components started. MoveGroup starting in other pane.'" Enter

sleep 5

# Pane 2: move_group
tmux send-keys -t handeye_sim:0.2 "ros2 run moveit_ros_move_group move_group --ros-args --params-file /tmp/ros_params/mg_params.yaml" Enter

# Pane 3: monitor
tmux send-keys -t handeye_sim:0.3 "echo 'Monitoring...' && watch -n 3 'echo Topics:; ros2 topic list 2>/dev/null | grep -E \"marker|plan|trajectory|controller\" | head -10; echo; echo Controllers:; ros2 control list_controllers 2>/dev/null; echo; echo Actions:; ros2 action list -t 2>/dev/null | head -10'" Enter

if $NO_RVIZ; then
    echo "(skipping RViz)"
else
    sleep 2
    ros2 run rviz2 rviz2 -d /workspace/ros2_ws/src/handeye_sim_bridge/rviz/handeye_sim_moveit.rviz --ros-args --params-file /tmp/ros_params/kinematics.yaml -p use_sim_time:=true
fi

echo ""
echo "  attach: tmux attach -t handeye_sim"
echo "  stop:   ./stop.sh"
echo "  detach: Ctrl+B, D"

else

# ==================== NORMAL MODE ====================
echo ""
echo "[1/7] Gazebo Sim..."
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib
gz sim -r -v 4 empty.sdf &
sleep 4

echo "[2/7] Bridges + RSP..."
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock &
sleep 1
ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file /tmp/ros_params/rsp_params.yaml &
sleep 1

echo "[3/7] Spawn robot..."
gz sdf -p "$URDF_PATH" > /tmp/robot_ready.sdf
ros2 run ros_gz_sim create -file /tmp/robot_ready.sdf -name fanuc_robot -world empty -allow_renaming true &
sleep 3

echo "[4/7] Controllers..."
ros2 param load /controller_manager "$GZ_CTRL_CONFIG" 2>&1 | head -3 || true
sleep 1
ros2 run controller_manager spawner joint_state_broadcaster &
sleep 1
ros2 run controller_manager spawner joint_trajectory_controller --param-file "$GZ_CTRL_CONFIG" &
sleep 2

echo "[5/7] MoveIt2 move_group..."
ros2 run moveit_ros_move_group move_group --ros-args --params-file /tmp/ros_params/mg_params.yaml &
sleep 5

echo "[6/7] SRDF + Scene publishers..."
python3 "$SRDF_PUB_EXE" --ros-args -p use_sim_time:=true &
sleep 1
python3 "$SCENE_EXE" --ros-args -p use_sim_time:=true &
sleep 1

if $NO_RVIZ; then
    echo "[7/7] Skipping RViz"
else
    echo "[7/7] RViz2..."
    ros2 run rviz2 rviz2 -d /workspace/ros2_ws/src/handeye_sim_bridge/rviz/handeye_sim_moveit.rviz --ros-args --params-file /tmp/ros_params/kinematics.yaml -p use_sim_time:=true &
    sleep 2
fi

echo ""
echo "================================"
echo "  All nodes started!"
echo "================================"
echo "  stop: ./stop.sh"
echo "================================"

trap 'echo "Shutting down..."; ./stop.sh' SIGINT SIGTERM
wait

fi
