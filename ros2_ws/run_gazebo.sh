#!/bin/bash
# Gazebo + MoveIt2 手眼标定仿真（v3 — 完全修复版）
set -e

source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash

URDF_PATH=/workspace/urdf/calib_robot.urdf
SRDF_PATH=/workspace/ros2_ws/src/handeye_sim_bridge/config/fanuc.srdf
GZ_CTRL_CONFIG=/workspace/ros2_ws/src/handeye_sim_bridge/config/gz_controllers.yaml
SCENE_EXE=/workspace/ros2_ws/install/handeye_sim_bridge/lib/handeye_sim_bridge/scene_publisher_node.py

echo "=========================================="
echo "  启动 Gazebo + MoveIt2 手眼标定仿真 v3"
echo "=========================================="

# 检查依赖
echo "[*] 检查依赖..."
if ! ros2 pkg list 2>/dev/null | grep -q ros_gz_bridge; then
    echo "  -> 安装 ros-jazzy-ros-gz-bridge..."
    apt-get update -qq && apt-get install -y -qq ros-jazzy-ros-gz-bridge
fi

mkdir -p /tmp/ros_params

# --- robot_state_publisher 参数 ---
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

# --- move_group 参数（关键修复：使用扁平点号键名） ---
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

    # ---- 规划流水线 ----
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

    # ---- 运动学（必须在 robot_description_kinematics 命名空间下）----
    robot_description_kinematics:
      arm:
        kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
        kinematics_solver_search_resolution: 0.005
        kinematics_solver_timeout: 0.05
        kinematics_solver_attempts: 3
        planner_configs: [RRTConnect]
        default_planner_config: RRTConnect

    # ---- 关节限位（必须在 robot_description_planning 命名空间下）----
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

    # ---- MoveIt 控制器管理 ----
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

# ===================== 启动顺序 =====================
# 1. Gazebo
echo "[1/8] 启动 Gazebo Sim..."
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib
gz sim -r -v 4 empty.sdf &
sleep 4

# 2. 时钟桥接
echo "[2/8] 启动 Gazebo ↔ ROS 时钟桥接..."
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock &
sleep 1
echo "  -> 确认时钟..."
timeout 3 ros2 topic echo /clock --once 2>&1 | grep -q "sec:" && echo "  -> 时钟正常！"

# 3. robot_state_publisher
echo "[3/8] 启动 robot_state_publisher..."
ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file /tmp/ros_params/rsp_params.yaml &
sleep 1

# 4. 生成机器人到 Gazebo
echo "[4/8] 生成机器人到 Gazebo..."
gz sdf -p "$URDF_PATH" > /tmp/robot_ready.sdf
ros2 run ros_gz_sim create -file /tmp/robot_ready.sdf -name fanuc_robot -world empty -allow_renaming true &
sleep 3

# 5. 加载控制器参数
echo "[5/8] 加载控制器参数到 controller_manager..."
ros2 param load /controller_manager "$GZ_CTRL_CONFIG" 2>&1 | head -3 || true
sleep 1

# 6. 加载控制器
echo "[6/8] 加载控制器..."
ros2 run controller_manager spawner joint_state_broadcaster &
sleep 1
ros2 run controller_manager spawner joint_trajectory_controller --param-file "$GZ_CTRL_CONFIG" &
sleep 2

# 7. move_group
echo "[7/8] 启动 MoveIt2 move_group..."
ros2 run moveit_ros_move_group move_group --ros-args --params-file /tmp/ros_params/mg_params.yaml &
sleep 5

# 8. SRDF 发布（RViz MotionPlanning 显示需要 /robot_description_semantic 话题）
echo "[8/8] 发布 SRDF..."
python3 /workspace/ros2_ws/install/handeye_sim_bridge/lib/handeye_sim_bridge/srdf_publisher_node.py --ros-args -p use_sim_time:=true &
sleep 1

# 9. scene_publisher + RViz
echo "[9/9] 启动场景发布 + RViz2..."
python3 "$SCENE_EXE" --ros-args -p use_sim_time:=true &
sleep 1
ros2 run rviz2 rviz2 -d /workspace/ros2_ws/src/handeye_sim_bridge/rviz/handeye_sim_moveit.rviz --ros-args -p use_sim_time:=true &

echo ""
echo "✅ 所有节点已启动！"
echo "  操作: RViz中拖拽小球 -> Plan & Execute"
echo "  停止: Ctrl+C"

wait
