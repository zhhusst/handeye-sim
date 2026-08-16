"""Start the real FANUC/Gocator measurement chain with no motion writer."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bridge_share = Path(get_package_share_directory("fanuc_gocator_bridge"))
    calibration_share = Path(
        get_package_share_directory("handeye_sim_bridge")
    )
    common_config = calibration_share / "config/calibration.yaml"
    real_config = bridge_share / "config/real_calibration.yaml"
    robot_urdf = Path("/workspace/urdf/calib_robot.urdf")
    if not robot_urdf.exists():
        raise FileNotFoundError(robot_urdf)
    robot_description = robot_urdf.read_text(encoding="utf-8")

    robot_ip = LaunchConfiguration("robot_ip")
    sensor_ip = LaunchConfiguration("sensor_ip")
    j23_factor = LaunchConfiguration("j23_factor")
    j23_validated = LaunchConfiguration("j23_validated")
    start_rviz = LaunchConfiguration("start_rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="192.168.0.10"),
            DeclareLaunchArgument("sensor_ip", default_value="192.168.0.19"),
            DeclareLaunchArgument("j23_factor", default_value="1.0"),
            DeclareLaunchArgument("j23_validated", default_value="true"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                parameters=[
                    {
                        "robot_description": ParameterValue(
                            robot_description, value_type=str
                        ),
                        "use_sim_time": False,
                    }
                ],
                output="screen",
            ),
            Node(
                package="fanuc_gocator_bridge",
                executable="fanuc_joint_state",
                name="fanuc_joint_state",
                parameters=[str(real_config)],
                ros_arguments=[
                    "-p", ["robot_ip:=", robot_ip],
                    "-p", ["j23_factor:=", j23_factor],
                    "-p", ["j23_validated:=", j23_validated],
                ],
                output="screen",
            ),
            Node(
                package="gocator_profile_driver",
                executable="gocator_profile_node",
                name="gocator_profile_driver",
                parameters=[str(real_config)],
                ros_arguments=["-p", ["sensor_ip:=", sensor_ip]],
                output="screen",
            ),
            Node(
                package="fanuc_gocator_bridge",
                executable="gocator_metric_adapter",
                name="gocator_metric_adapter",
                parameters=[str(real_config)],
                output="screen",
            ),
            Node(
                package="fanuc_gocator_bridge",
                executable="measurement_sync",
                name="measurement_sync",
                parameters=[str(real_config)],
                output="screen",
            ),
            Node(
                package="handeye_sim_bridge",
                executable="profile_endpoint_detector",
                name="profile_endpoint_detector",
                parameters=[str(common_config), str(real_config)],
                output="screen",
            ),
            Node(
                package="handeye_sim_bridge",
                executable="profile_viz",
                name="profile_viz_node",
                parameters=[str(common_config), str(real_config)],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=[
                    "-d",
                    str(
                        calibration_share
                        / "rviz/handeye_sim_moveit.rviz"
                    ),
                ],
                parameters=[{"use_sim_time": False}],
                condition=IfCondition(start_rviz),
                output="screen",
            ),
        ]
    )
