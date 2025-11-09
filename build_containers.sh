#!/bin/bash
set -e

echo "Building ROS2 simulation images..."

# Build the world image
docker build -t ros-world-mwe -f docker/world/Dockerfile .

# Build the robot image
docker build -t ros-robot-mwe -f docker/robot/Dockerfile .

echo "Docker images built successfully:"
echo "  - ros-world-mwe"
echo "  - ros-robot-mwe"