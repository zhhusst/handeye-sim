import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('handeye_sim_bridge')
    workspace_dir = os.path.dirname(os.path.dirname(pkg_dir))

    urdf_path = '/workspace/urdf/calib_robot.urdf'
    srdf_path = os.path.join(workspace_dir, 'src', 'handeye_sim_bridge', 'config', 'fanuc.srdf')
    gz_ctrl_config = os.path.join(workspace_dir, 'src', 'handeye_sim_bridge', 'config', 'gz_controllers.yaml')
    rviz_config = os.path.join(workspace_dir, 'src', 'handeye_sim_bridge', 'rviz', 'handeye_sim_moveit.rviz')

    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    with open(srdf_path, 'r') as f:
        srdf_content = f.read()

    # Gazebo Sim
    gz_sim = ExecuteProcess(cmd=['gz', 'sim', '-r', '-v', '4', 'empty.sdf'], output='screen')

    # Clock bridge
    clock_bridge = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
             '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen')

    # robot_state_publisher
    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': urdf_content, 'use_sim_time': True}],
        output='screen')

    # Spawn robot in Gazebo
    spawn_gz = ExecuteProcess(
        cmd=['bash', '-c', 'gz sdf -p ' + urdf_path + ' > /tmp/robot_ready.sdf && '
             'ros2 run ros_gz_sim create -file /tmp/robot_ready.sdf -name fanuc_robot -world empty -allow_renaming true'],
        output='screen')

    # Gazebo controllers
    load_ctrl = ExecuteProcess(
        cmd=['ros2', 'param', 'load', '/controller_manager', gz_ctrl_config],
        output='screen')
    spawn_jsb = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster'],
        output='screen')
    spawn_jtc = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_trajectory_controller',
             '--param-file', gz_ctrl_config],
        output='screen')

    # === move_group with ALL params properly declared ===
    move_group_params = {
        'robot_description': urdf_content,
        'robot_description_semantic': srdf_content,
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl.planning_plugins': ['ompl_interface/OMPLPlanner'],
        'ompl.request_adapters': [
            'default_planning_request_adapters/ResolveConstraintFrames',
            'default_planning_request_adapters/ValidateWorkspaceBounds',
            'default_planning_request_adapters/CheckStartStateBounds',
            'default_planning_request_adapters/CheckStartStateCollision',
        ],
        'ompl.response_adapters': [
            'default_planning_response_adapters/AddTimeOptimalParameterization',
            'default_planning_response_adapters/ValidateSolution',
            'default_planning_response_adapters/DisplayMotionPath',
        ],
        'robot_description_kinematics.arm.kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
        'robot_description_kinematics.arm.kinematics_solver_search_resolution': 0.005,
        'robot_description_kinematics.arm.kinematics_solver_timeout': 0.05,
        'robot_description_kinematics.arm.kinematics_solver_attempts': 3,
        'robot_description_planning.joint_limits.J1_joint.max_velocity': 1.57,
        'robot_description_planning.joint_limits.J2_joint.max_velocity': 1.57,
        'robot_description_planning.joint_limits.J3_joint.max_velocity': 1.57,
        'robot_description_planning.joint_limits.J4_joint.max_velocity': 2.09,
        'robot_description_planning.joint_limits.J5_joint.max_velocity': 2.09,
        'robot_description_planning.joint_limits.J6_joint.max_velocity': 3.14,
        # === Controller config (will be properly declared by launch!) ===
        'moveit_manage_controllers': True,
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        'controller_names': ['joint_trajectory_controller'],
        'joint_trajectory_controller.type': 'FollowJointTrajectory',
        'joint_trajectory_controller.joints': ['J1_joint', 'J2_joint', 'J3_joint', 'J4_joint', 'J5_joint', 'J6_joint'],
        'joint_trajectory_controller.action_ns': 'follow_joint_trajectory',
        'joint_trajectory_controller.default': True,
        'use_sim_time': True,
    }

    move_group = Node(
        package='moveit_ros_move_group', executable='move_group',
        parameters=[move_group_params],
        output='screen')

    # Visualization nodes
    srdf_pub = Node(
        package='handeye_sim_bridge', executable='srdf_publisher_node',
        parameters=[{'use_sim_time': True}], output='screen')
    scene_pub = Node(
        package='handeye_sim_bridge', executable='scene_publisher_node',
        parameters=[{'use_sim_time': True}], output='screen')
    rviz = Node(
        package='rviz2', executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}], output='screen')

    return LaunchDescription([
        gz_sim,
        TimerAction(period=4.0, actions=[LogInfo(msg='Starting clock + RSP...'), clock_bridge, rsp]),
        TimerAction(period=5.0, actions=[LogInfo(msg='Spawning robot...'), spawn_gz]),
        TimerAction(period=7.0, actions=[LogInfo(msg='Loading controllers...'), load_ctrl, spawn_jsb, spawn_jtc]),
        TimerAction(period=9.0, actions=[LogInfo(msg='Starting MoveIt2...'), move_group]),
        TimerAction(period=14.0, actions=[LogInfo(msg='Starting visualization...'), srdf_pub, scene_pub, rviz]),
    ])
