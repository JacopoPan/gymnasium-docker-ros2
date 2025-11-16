FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS gdr2-base-image

# Tell apt (and other Debian tools) not to prompt for user input during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install basic dependencies
RUN apt update \
    && apt install -y --no-install-recommends \
        wget gosu htop vim ruby tmux net-tools iproute2 iputils-ping netcat-openbsd \
        python3-pip python3-venv \
        mesa-utils \
    && gem install tmuxinator \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 Humble
# Based on https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
RUN apt update \
    && apt install -y --no-install-recommends \
        locales \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
RUN apt update \
    && apt install -y --no-install-recommends \
        software-properties-common curl \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \
    $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update \
    && apt install -y --no-install-recommends \
        ros-humble-desktop ros-dev-tools \
        ros-humble-bondcpp ros-humble-ament-cmake-clang-format \
        ros-humble-vision-msgs \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN rosdep init

# Install Zenoh
RUN echo "deb [trusted=yes] https://download.eclipse.org/zenoh/debian-repo/ /" | sudo tee -a /etc/apt/sources.list > /dev/null
RUN apt-get update && \
    apt-get install -y zenoh-bridge-ros2dds \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Gazebo Harmonic
# Based on https://gazebosim.org/docs/harmonic/install_ubuntu/
RUN apt update \
    && apt install -y --no-install-recommends \
        lsb-release gnupg \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt update \
    && apt install -y --no-install-recommends \
        gz-harmonic ros-humble-ros-gzharmonic \
        libgz-transport13-* python3-gz-transport13 python3-gz-msgs10 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
# Run with $ gz sim

# Install ZeroMQ
RUN apt-get update && \
    apt-get install -y --no-install-recommends libzmq3-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir --resume-retries 5 pyzmq

# Copy resources and configuration files from this repository
FROM gdr2-base-image AS gdr2-dev-image

# Build the ROS 2 workspace
COPY resources/ros2_ws/src /gdr2/ros2_ws/src
WORKDIR /gdr2/ros2_ws
RUN rosdep update
RUN rosdep install --from-paths src/ --ignore-src --rosdistro humble -y
# Explicitly use bash, not sh, to source and build the workspace
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# Source the workspaces
RUN echo "source /gdr2/ros2_ws/install/setup.bash" >> /root/.bashrc
# If needed (but already in .bashrc) $ source /opt/ros/humble/setup.bash && source /gdr2/ros2_ws/install/setup.bash

# Copy resources
COPY resources/zeromq_bridge.py /gdr2/zeromq_bridge.py
COPY resources/gdr2-world.sdf /gdr2/gdr2-world.sdf
COPY resources/tmux_session.yml.erb /gdr2/tmux_session.yml.erb

# Final config
WORKDIR /gdr2
ENTRYPOINT ["tmuxinator", "start", "-p", "/gdr2/tmux_session.yml.erb"]
