cmake_minimum_required(VERSION 3.8)
project(ocams_ros2)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include/ocams_ros2
)

add_executable(ocams_stereo_imu_node
  src/ocams_stereo_imu_node.cpp
  src/withrobot_camera.cpp
  src/withrobot_utility.cpp
)

target_include_directories(ocams_stereo_imu_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(ocams_stereo_imu_node rclcpp sensor_msgs ament_index_cpp)
target_link_libraries(ocams_stereo_imu_node ${OpenCV_LIBS} v4l2 udev pthread)

install(TARGETS ocams_stereo_imu_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

ament_package()
