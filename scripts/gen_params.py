import yaml, os

def load_text(p):
    with open(p) as f:
        return f.read()
def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)

ws = '/workspace'
install = ws + '/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge'
urdf = load_text(ws + '/urdf/calib_robot.urdf')
srdf = load_text(install + '/config/fanuc.srdf')
kin = load_yaml(install + '/config/kinematics.yaml')
ompl = load_yaml(install + '/config/ompl_planning.yaml')
jl = load_yaml(install + '/config/joint_limits.yaml')

# RSP 参数
with open('/workspace/rsp_params.yaml', 'w') as f:
    yaml.dump({'robot_state_publisher': {'ros__parameters': {
        'robot_description': urdf, 'publish_frequency': 30.0}}}, f, default_flow_style=False)

# MG 参数
with open('/workspace/mg_params.yaml', 'w') as f:
    yaml.dump({'move_group': {'ros__parameters': {
        'robot_description': urdf,
        'robot_description_semantic': srdf,
        'robot_description_kinematics': kin,
        'robot_description_planning': jl,
        'ompl': ompl,
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'moveit_manage_controllers': False,
        'publish_planning_scene': True,
        'publish_robot_description': True,
        'publish_robot_description_semantic': True,
    }}}, f, default_flow_style=False)

# 验证前几行
for name in ['/workspace/rsp_params.yaml', '/workspace/mg_params.yaml']:
    with open(name) as f:
        first_line = f.readline().strip()
        print(f'{name}: first line = [{first_line}]')
