"""Launch the simulated laser profile and optional Phase 0b collector.

The robot/controller simulation is started separately by ``scripts/start_simulation.sh``.
Keeping this launch file focused makes it useful with either Gazebo or real topics.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    collect_seeds = LaunchConfiguration("collect_seeds")
    run_active = LaunchConfiguration("run_active")
    config = os.path.join(
        get_package_share_directory("handeye_sim_bridge"), "config", "calibration.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("collect_seeds", default_value="false"),
            DeclareLaunchArgument("run_active", default_value="false"),
            Node(
                package="handeye_sim_backend",
                executable="scene_publisher",
                name="calibration_scene",
                parameters=[config, {"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="handeye_sim_bridge",
                executable="profile_endpoint_detector",
                name="profile_endpoint_detector",
                parameters=[config, {"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="handeye_sim_bridge",
                executable="profile_viz",
                name="profile_visualization",
                parameters=[config, {"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="handeye_sim_bridge",
                executable="seed_collection",
                name="bilateral_seed_collection",
                condition=IfCondition(collect_seeds),
                parameters=[config, {"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="handeye_sim_bridge",
                executable="active_calibration_sim",
                name="active_calibration_sim",
                condition=IfCondition(run_active),
                parameters=[config, {"use_sim_time": use_sim_time}],
                output="screen",
            ),
        ]
    )
