cmake_minimum_required(VERSION 3.8)
project(zmq_bridge_cpp)

add_compile_options(-O3)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Find Gazebo Dependencies
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)

# Find ZeroMQ via PkgConfig
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

add_executable(zmq_bridge_node src/zeromq_bridge.cpp)

# Use ament_target_dependencies for ROS packages. 
ament_target_dependencies(zmq_bridge_node
  rclcpp
  std_msgs
  geometry_msgs
)

# Manually link non-ROS system libraries (Gazebo & ZMQ)
target_link_libraries(zmq_bridge_node
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
  ${ZMQ_LIBRARIES}
)

# Add ZMQ include directories explicitly
target_include_directories(zmq_bridge_node PUBLIC
  ${ZMQ_INCLUDE_DIRS}
)

install(TARGETS zmq_bridge_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
