#!/usr/bin/env bash
# 一键启动 Gazebo + MoveIt2 手眼标定仿真
set -eo pipefail

USE_TMUX=false
NO_RVIZ=false
HEADLESS=false
FORCE_WEB=false
WEB_VISUALIZATION=false
STRESS_NOISE=false
GZ_VERBOSITY=${GZ_VERBOSITY:-2}
WEB_VISUALIZATION_URL="http://localhost:6080/vnc.html?autoconnect=1&resize=scale"
for arg in "$@"; do
    case "$arg" in
        --tmux) USE_TMUX=true ;;
        --no-rviz) NO_RVIZ=true ;;
        --headless) HEADLESS=true; NO_RVIZ=true ;;
        --web) FORCE_WEB=true ;;
        --stress-noise) STRESS_NOISE=true ;;
        -h|--help)
            echo "用法: ./scripts/start_simulation.sh [--tmux] [--no-rviz] [--headless] [--web] [--stress-noise]"
            echo "  --tmux    分屏模式（可看各组件实时日志）"
            echo "  --no-rviz  不启动 RViz"
            echo "  --headless 不启动 Gazebo GUI 和 RViz（适合验证/无 X11 环境）"
            echo "  --web      在 http://localhost:6080 提供 Gazebo + RViz 可视化"
            echo "  --stress-noise 加载 0.5 mm RMS 固定非平面共享形貌压力测试"
            exit 0 ;;
        *)
            echo "未知参数: $arg" >&2
            exit 2 ;;
    esac
done

start_web_visualization() {
    /workspace/scripts/start_web_desktop.sh
    export DISPLAY=${WEB_DISPLAY:-:99}
    export QT_X11_NO_MITSHM=1
    WEB_VISUALIZATION=true
}

launch_gazebo_gui() {
    : > /tmp/gazebo_gui.log
    if $WEB_VISUALIZATION; then
        bash -c \
            'gz sim -g -v "$1" --render-engine-gui ogre; status=$?; exit "$status"' \
            _ "$GZ_VERBOSITY" > /tmp/gazebo_gui.log 2>&1 &
    else
        bash -c 'gz sim -g -v "$1"; status=$?; exit "$status"' _ "$GZ_VERBOSITY" \
            > /tmp/gazebo_gui.log 2>&1 &
    fi
    GZ_GUI_PID=$!
    sleep 3
    kill -0 "$GZ_GUI_PID" 2>/dev/null
}

ensure_gazebo_gui() {
    if $FORCE_WEB; then
        start_web_visualization
    fi
    if launch_gazebo_gui; then
        return
    fi

    if ! $WEB_VISUALIZATION; then
        echo "DISPLAY=${DISPLAY:-<unset>} is not authorized; switching to browser visualization."
        start_web_visualization
        if launch_gazebo_gui; then
            return
        fi
    fi

    echo "Gazebo GUI failed to start." >&2
    grep -E -m 5 \
        "No protocol specified|could not connect to display|Could not load the Qt|RenderingAPIException|Unable to create" \
        /tmp/gazebo_gui.log >&2 || true
    /workspace/scripts/stop_simulation.sh >/dev/null 2>&1
    exit 1
}

arrange_web_windows() {
    $WEB_VISUALIZATION || return 0
    command -v wmctrl >/dev/null 2>&1 || return 0
    for _ in $(seq 1 50); do
        window_list=$(wmctrl -l 2>/dev/null || true)
        if [[ "$window_list" == *"Gazebo Sim"* ]] \
            && { $NO_RVIZ || [[ "$window_list" == *"RViz"* ]]; }; then
            break
        fi
        sleep 0.2
    done
    wmctrl -r "Gazebo Sim" -b remove,maximized_vert,maximized_horz 2>/dev/null || true
    if $NO_RVIZ; then
        wmctrl -r "Gazebo Sim" -e 0,0,0,1910,1040 2>/dev/null || true
        return
    fi
    wmctrl -r "Gazebo Sim" -e 0,0,0,950,1040 2>/dev/null || true
    wmctrl -r "RViz" -b remove,maximized_vert,maximized_horz 2>/dev/null || true
    wmctrl -r "RViz" -e 0,960,0,950,1040 2>/dev/null || true
}

print_web_visualization_banner() {
    $WEB_VISUALIZATION || return 0
    echo ""
    echo "=============================================================="
    echo "  Gazebo 和 RViz 位于浏览器虚拟桌面，不会弹出宿主机原生窗口"
    printf "  打开: \033]8;;%s\033\\%s\033]8;;\033\\\n" \
        "$WEB_VISUALIZATION_URL" "$WEB_VISUALIZATION_URL"
    echo "=============================================================="
}

move_to_initial_observation_pose() {
    local action_name=/joint_trajectory_controller/follow_joint_trajectory
    local goal
    goal='{trajectory: {joint_names: [J1_joint, J2_joint, J3_joint, J4_joint, J5_joint, J6_joint], points: [{positions: [-0.2357, -0.0364, -0.6328, -0.4062, -1.0504, 0.8788], time_from_start: {sec: 3, nanosec: 0}}]}}'

    local active=false
    for _ in $(seq 1 60); do
        if ros2 control list_controllers 2>/dev/null \
            | grep -E -q '^joint_trajectory_controller[[:space:]].*[[:space:]]active$'; then
            active=true
            break
        fi
        sleep 0.2
    done
    if ! $active; then
        echo "joint_trajectory_controller did not become active." >&2
        ros2 control list_controllers >&2 || true
        return 1
    fi

    for attempt in 1 2 3; do
        if timeout 20s ros2 action send_goal \
            "$action_name" control_msgs/action/FollowJointTrajectory "$goal" \
            > /tmp/handeye_initial_pose.log 2>&1 \
            && grep -E -q \
                "status: SUCCEEDED|Goal finished with status: SUCCEEDED" \
                /tmp/handeye_initial_pose.log; then
            return 0
        fi
        sleep 0.5
    done
    echo "Initial observation pose was not reached after three attempts." >&2
    sed -n '1,40p' /tmp/handeye_initial_pose.log >&2
    return 1
}

cd /workspace
source /opt/ros/jazzy/setup.bash
export FASTDDS_DEFAULT_PROFILES_FILE=/workspace/ros2_ws/src/handeye_sim_bridge/config/fastdds_udp_only.xml
if [[ ! -f /workspace/ros2_ws/install/setup.bash ]]; then
    echo "ROS 2 workspace is not built; run ./scripts/build.sh first." >&2
    exit 1
fi
source /workspace/ros2_ws/install/setup.bash

URDF_PATH=/workspace/urdf/calib_robot.urdf
SRDF_PATH=/workspace/ros2_ws/src/handeye_sim_bridge/config/fanuc.srdf
GZ_CTRL_CONFIG=/workspace/ros2_ws/src/handeye_sim_bridge/config/gz_controllers.yaml
CALIBRATION_CONFIG=/workspace/ros2_ws/src/handeye_sim_bridge/config/calibration.yaml
STRESS_NOISE_CONFIG=/workspace/ros2_ws/src/handeye_sim_bridge/config/calibration_noise_stress.yaml
SCENE_EXE=/workspace/ros2_ws/install/handeye_sim_bridge/lib/handeye_sim_bridge/scene_publisher_node
SRDF_PUB_EXE=/workspace/ros2_ws/install/handeye_sim_bridge/lib/handeye_sim_bridge/srdf_publisher_node
PROFILE_VIZ_BIN=/workspace/ros2_ws/install/handeye_sim_bridge/lib/handeye_sim_bridge/profile_viz
ENDPOINT_DETECTOR_BIN=/workspace/ros2_ws/install/handeye_sim_bridge/lib/handeye_sim_bridge/profile_endpoint_detector

SCENE_PARAMETER_ARGS="--params-file '$CALIBRATION_CONFIG'"
if $STRESS_NOISE; then
    SCENE_PARAMETER_ARGS="$SCENE_PARAMETER_ARGS --params-file '$STRESS_NOISE_CONFIG'"
fi

/workspace/scripts/stop_simulation.sh 2>/dev/null || true
sleep 1

echo "=========================================="
echo "  Gazebo + MoveIt2  手眼标定仿真 "
echo "=========================================="
if $STRESS_NOISE; then
    echo "  噪声工况：0.5 mm RMS 固定非平面共享形貌压力测试"
else
    echo "  噪声工况：论文主实验（合格标定件 10 µm RMS）"
fi

if ! ros2 pkg prefix ros_gz_bridge >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq ros-jazzy-ros-gz-bridge
fi
if [ ! -f "$URDF_PATH" ]; then echo "URDF not found"; exit 1; fi
if [ ! -f "$SRDF_PATH" ]; then echo "SRDF not found"; exit 1; fi

mkdir -p /tmp/ros_params

# Fail before launching any ROS processes if URDF -> SDF conversion is invalid.
if ! gz sdf -p "$URDF_PATH" > /tmp/robot_ready.sdf; then
    echo "URDF validation failed: $URDF_PATH" >&2
    echo "Regenerate it with: python3 /workspace/urdf/generate_urdf.py" >&2
    exit 1
fi

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

# Pane 0: Gazebo server
tmux send-keys -t handeye_sim:0.0 "export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib && gz sim -s -r -v '$GZ_VERBOSITY' empty.sdf" Enter
sleep 2
if ! $HEADLESS; then
    ensure_gazebo_gui
fi

# Pane 1: Launch components sequentially
tmux send-keys -t handeye_sim:0.1 "echo 'Starting components...'" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock &" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file /tmp/ros_params/rsp_params.yaml &" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 1 world base_link &" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 run ros_gz_sim create -file /tmp/robot_ready.sdf -name fanuc_robot -world empty -allow_renaming true" Enter
sleep 2
tmux send-keys -t handeye_sim:0.1 "ros2 run ros_gz_sim create -file /workspace/ros2_ws/src/handeye_sim_bridge/config/calibration_plate.sdf -name calibration_plate -world empty -allow_renaming true" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 param load /controller_manager '$GZ_CTRL_CONFIG' 2>&1 | head -3" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 run controller_manager spawner joint_state_broadcaster" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "ros2 run controller_manager spawner joint_trajectory_controller --param-file '$GZ_CTRL_CONFIG'" Enter
sleep 2
echo "Restoring GitHub initial observation pose..."
move_to_initial_observation_pose
tmux send-keys -t handeye_sim:0.1 "'$SRDF_PUB_EXE' --ros-args -p use_sim_time:=true &" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "'$SCENE_EXE' --ros-args $SCENE_PARAMETER_ARGS -p use_sim_time:=true &" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "'$ENDPOINT_DETECTOR_BIN' --ros-args --params-file '$CALIBRATION_CONFIG' -p use_sim_time:=true &" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "'$PROFILE_VIZ_BIN' --ros-args -p use_sim_time:=true &" Enter
sleep 1
tmux send-keys -t handeye_sim:0.1 "echo 'Components started. MoveGroup starting in other pane.'" Enter

sleep 5

# Pane 2: move_group
tmux send-keys -t handeye_sim:0.2 "ros2 run moveit_ros_move_group move_group --ros-args --params-file /tmp/ros_params/mg_params.yaml" Enter

# Pane 3: monitor
tmux send-keys -t handeye_sim:0.3 "echo 'Monitoring...' && watch -n 3 'echo Topics:; ros2 topic list 2>/dev/null | grep -E \"marker|plan|trajectory|controller|profile\" | head -10; echo; echo Controllers:; ros2 control list_controllers 2>/dev/null; echo; echo Actions:; ros2 action list -t 2>/dev/null | head -10'" Enter

if $NO_RVIZ; then
    echo "(skipping RViz)"
else
    sleep 2
    if $WEB_VISUALIZATION; then
        ros2 run rviz2 rviz2 -d /workspace/ros2_ws/src/handeye_sim_bridge/rviz/handeye_sim_moveit.rviz --ros-args --params-file /tmp/ros_params/kinematics.yaml -p use_sim_time:=true &
        sleep 3
    else
        ros2 run rviz2 rviz2 -d /workspace/ros2_ws/src/handeye_sim_bridge/rviz/handeye_sim_moveit.rviz --ros-args --params-file /tmp/ros_params/kinematics.yaml -p use_sim_time:=true
    fi
fi
arrange_web_windows

echo ""
echo "  attach: tmux attach -t handeye_sim"
echo "  stop:   ./scripts/stop_simulation.sh"
echo "  detach: Ctrl+B, D"

else

# ==================== NORMAL MODE ====================
echo ""
echo "[1/7] Gazebo Sim..."
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib
gz sim -s -r -v "$GZ_VERBOSITY" empty.sdf &
GZ_PID=$!
sleep 3
if ! kill -0 "$GZ_PID" 2>/dev/null; then
    wait "$GZ_PID" || true
    echo "Gazebo server failed to start." >&2
    exit 1
fi
if ! $HEADLESS; then
    ensure_gazebo_gui
fi

echo "[2/7] Bridges + RSP + static TF..."
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock &
sleep 1
ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file /tmp/ros_params/rsp_params.yaml &
sleep 1
# 静态 TF: world -> base_link（标记/场景使用 world 帧）
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 1 world base_link &
sleep 1

echo "[3/7] Spawn robot + calibration plate..."
timeout 30s ros2 run ros_gz_sim create -file /tmp/robot_ready.sdf -name fanuc_robot -world empty -allow_renaming true
echo "  -> Spawn calibration plate..."
timeout 30s ros2 run ros_gz_sim create -file /workspace/ros2_ws/src/handeye_sim_bridge/config/calibration_plate.sdf -name calibration_plate -world empty -allow_renaming true

echo "[4/7] Controllers..."
ros2 param load /controller_manager "$GZ_CTRL_CONFIG" 2>&1 | head -3 || true
sleep 1
ros2 run controller_manager spawner joint_state_broadcaster &
sleep 1
ros2 run controller_manager spawner joint_trajectory_controller --param-file "$GZ_CTRL_CONFIG" &
sleep 2
echo "  -> Restoring GitHub initial observation pose..."
move_to_initial_observation_pose

echo "[5/7] MoveIt2 move_group..."
ros2 run moveit_ros_move_group move_group --ros-args --params-file /tmp/ros_params/mg_params.yaml &
sleep 5

echo "[6/7] SRDF + Scene + Endpoint Detector + Profile Viz..."
"$SRDF_PUB_EXE" --ros-args -p use_sim_time:=true &
sleep 1
if $STRESS_NOISE; then
    "$SCENE_EXE" --ros-args --params-file "$CALIBRATION_CONFIG" --params-file "$STRESS_NOISE_CONFIG" -p use_sim_time:=true &
else
    "$SCENE_EXE" --ros-args --params-file "$CALIBRATION_CONFIG" -p use_sim_time:=true &
fi
sleep 1
"$ENDPOINT_DETECTOR_BIN" --ros-args --params-file "$CALIBRATION_CONFIG" -p use_sim_time:=true &
sleep 1
"$PROFILE_VIZ_BIN" --ros-args -p use_sim_time:=true &
sleep 1

if $NO_RVIZ; then
    echo "[7/7] Skipping RViz"
else
    if $WEB_VISUALIZATION; then
        echo "[7/7] RViz2（浏览器虚拟桌面）..."
    else
        echo "[7/7] RViz2..."
    fi
    ros2 run rviz2 rviz2 -d /workspace/ros2_ws/src/handeye_sim_bridge/rviz/handeye_sim_moveit.rviz --ros-args --params-file /tmp/ros_params/kinematics.yaml -p use_sim_time:=true &
    sleep 2
fi
arrange_web_windows
print_web_visualization_banner

echo ""
echo "================================"
echo "  All nodes started!"
echo "================================"
echo "  stop: ./scripts/stop_simulation.sh"
if $WEB_VISUALIZATION; then
    echo "  web:  $WEB_VISUALIZATION_URL"
fi
echo "================================"

trap 'echo "Shutting down..."; /workspace/scripts/stop_simulation.sh' SIGINT SIGTERM
wait

fi
