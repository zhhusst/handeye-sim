from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument('sensor_ip', default_value='192.168.0.19'),
            DeclareLaunchArgument(
                'initial_laser_state', default_value='false'
            ),
            DeclareLaunchArgument(
                'output_topic', default_value='/gocator/profile_raw_mm'
            ),
            DeclareLaunchArgument('frame_id', default_value='gocator_sensor'),
            Node(
                package='gocator_profile_driver',
                executable='gocator_profile_node',
                name='gocator_profile_driver',
                output='screen',
                parameters=[
                    {
                        'sensor_ip': LaunchConfiguration('sensor_ip'),
                        'receive_timeout': 20000000,  # 20秒超时
                        'initial_laser_state': LaunchConfiguration(
                            'initial_laser_state'
                        ),
                        'output_topic': LaunchConfiguration('output_topic'),
                        'frame_id': LaunchConfiguration('frame_id'),
                    }
                ],
            )
        ]
    )
