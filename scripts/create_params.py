import yaml
urdf_path = '/workspace/urdf/calib_robot.urdf'
srdf_path = '/workspace/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge/config/fanuc.srdf'
with open(urdf_path) as f:
    urdf_text = f.read()
rsp_params = {'robot_state_publisher': {'ros__parameters': {'robot_description': urdf_text, 'publish_frequency': 30.0}}}
with open('/workspace/rsp_params.yaml', 'w') as f:
    yaml.dump(rsp_params, f, default_flow_style=False)
with open('/workspace/rsp_params.yaml') as f2:
    head = f2.read()[:200]
    print('FIRST 200 CHARS:')
    print(head)
    print()
    print('STARTS WITH /** :', head.startswith('/'))
    print('STARTS WITH robot :', head.startswith('robot'))
