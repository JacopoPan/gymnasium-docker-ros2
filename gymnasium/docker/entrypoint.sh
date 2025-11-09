#!/bin/bash
set -e

# Source ROS and the built workspace
. /opt/ros/humble/setup.bash
. /ros_ws/install/setup.bash

# Execute the command passed to this script (e.g., ros2 run ...)
exec "$@"