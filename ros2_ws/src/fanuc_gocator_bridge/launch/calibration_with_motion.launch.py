"""Real measurement chain plus a DISARMED PC_TRACK_ALL motion bridge."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bridge_share = Path(get_package_share_directory("fanuc_gocator_bridge"))
    real_config = bridge_share / "config/real_calibration.yaml"
    robot_ip = LaunchConfiguration("robot_ip")
    sensor_ip = LaunchConfiguration("sensor_ip")
    j23_factor = LaunchConfiguration("j23_factor")
    j23_validated = LaunchConfiguration("j23_validated")
    start_rviz = LaunchConfiguration("start_rviz")
    motion_mode = LaunchConfiguration("motion_mode")
    motion_writes_enabled = LaunchConfiguration("motion_writes_enabled")

    measurement_chain = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(bridge_share / "launch/observe_only.launch.py")
        ),
        launch_arguments={
            "robot_ip": robot_ip,
            "sensor_ip": sensor_ip,
            "j23_factor": j23_factor,
            "j23_validated": j23_validated,
            "start_rviz": start_rviz,
        }.items(),
    )
    motion_bridge = Node(
        package="fanuc_gocator_bridge",
        executable="fanuc_motion_bridge",
        name="fanuc_motion_bridge",
        parameters=[str(real_config)],
        # CLI parameter overrides have higher precedence than node-specific
        # YAML.  Keep mode/write selection authoritative at the launch gate.
        ros_arguments=[
            "-p", ["robot_ip:=", robot_ip],
            "-p", ["mode:=", motion_mode],
            "-p", ["motion_writes_enabled:=", motion_writes_enabled],
        ],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="192.168.0.10"),
            DeclareLaunchArgument("sensor_ip", default_value="192.168.0.19"),
            DeclareLaunchArgument("j23_factor", default_value="1.0"),
            DeclareLaunchArgument("j23_validated", default_value="true"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument("motion_mode", default_value="plan_only"),
            DeclareLaunchArgument(
                "motion_writes_enabled", default_value="false"
            ),
            measurement_chain,
            motion_bridge,
        ]
    )
