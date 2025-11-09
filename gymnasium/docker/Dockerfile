# Base ROS2 image
FROM osrf/ros:humble-desktop

# Copy the entire ROS workspace
WORKDIR /ros_ws
COPY ./ros_workspace/src /ros_ws/src

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select sim_nodes

# Copy the entrypoint
COPY ./docker/world/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "sim_nodes", "world_node"]