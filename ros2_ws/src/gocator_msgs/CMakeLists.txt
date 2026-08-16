cmake_minimum_required(VERSION 3.8)
project(gocator_msgs)

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 生成服务消息
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetLaserState.srv"
)

# 导出依赖
ament_export_dependencies(rosidl_default_runtime)

ament_package()
