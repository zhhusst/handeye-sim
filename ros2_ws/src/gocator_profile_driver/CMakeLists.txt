cmake_minimum_required(VERSION 3.8)
project(gocator_profile_driver)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(gocator_msgs REQUIRED)

set(_GOCATOR_SDK_DEFAULT
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/gocator_sdk")
if(DEFINED ENV{GOCATOR_SDK_ROOT})
  set(_GOCATOR_SDK_DEFAULT "$ENV{GOCATOR_SDK_ROOT}")
endif()
set(GOCATOR_SDK_ROOT "${_GOCATOR_SDK_DEFAULT}" CACHE PATH
  "Path to the local LMI Gocator SDK")

set(GOCATOR_INCLUDES ${GOCATOR_SDK_ROOT}/Gocator/GoSdk)
set(KAPI_INCLUDES    ${GOCATOR_SDK_ROOT}/Platform/kApi)

set(GOCATOR_LIBRARIES ${GOCATOR_SDK_ROOT}/lib/linux_x64d/libGoSdk.so)
set(KAPI_LIBRARIES    ${GOCATOR_SDK_ROOT}/lib/linux_x64d/libkApi.so)

foreach(_required_path
    "${GOCATOR_INCLUDES}/GoSdk/GoSdk.h"
    "${KAPI_INCLUDES}/kApi/kApi.h"
    "${GOCATOR_LIBRARIES}"
    "${KAPI_LIBRARIES}")
  if(NOT EXISTS "${_required_path}")
    message(FATAL_ERROR
      "Missing Gocator SDK component: ${_required_path}. "
      "Set GOCATOR_SDK_ROOT to a valid local SDK installation.")
  endif()
endforeach()

add_executable(gocator_profile_node
  src/gocator_profile_node.cpp
)

ament_target_dependencies(gocator_profile_node
  rclcpp
  sensor_msgs
  tf2_ros
  geometry_msgs
  gocator_msgs
)

target_include_directories(gocator_profile_node PRIVATE
  ${GOCATOR_INCLUDES}
  ${KAPI_INCLUDES}
)

target_link_libraries(gocator_profile_node
  ${GOCATOR_LIBRARIES}
  ${KAPI_LIBRARIES}
)

set_target_properties(gocator_profile_node PROPERTIES
  INSTALL_RPATH "$ORIGIN/.."
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS
  gocator_profile_node
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  ${GOCATOR_LIBRARIES}
  ${KAPI_LIBRARIES}
  DESTINATION lib
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
