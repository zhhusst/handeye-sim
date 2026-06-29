FROM osrf/ros:jazzy-desktop

# Base image has everything: numpy/scipy/pip3/rviz2/xacro/sudo

# Install matplotlib + Gazebo packages
RUN pip3 install --quiet --no-cache-dir matplotlib 2>/dev/null || true && \
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      ros-jazzy-gz-sim-vendor \
      ros-jazzy-ros-gz-sim \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-ros-gz-bridge \
    && rm -rf /var/lib/apt/lists/*

ENV SHELL /bin/bash
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
