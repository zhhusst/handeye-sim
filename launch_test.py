from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['cat', '/workspace/rsp_params.yaml'],
            name='test_cat', output='screen'),
    ])
