FROM osrf/ros:jazzy-desktop

# Keep Python dependencies in apt so they share ROS 2's system interpreter.
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      python3-numpy \
      python3-scipy \
      python3-yaml \
      python3-matplotlib \
      python3-pytest \
      python3-pip \
      novnc \
      openbox \
      websockify \
      wmctrl \
      x11vnc \
      xvfb \
      ros-jazzy-gz-sim-vendor \
      ros-jazzy-ros-gz-sim \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-ros-gz-bridge \
    && rm -rf /var/lib/apt/lists/*

# FANUC explicit EtherNet/IP transport used by the real-hardware backend.
COPY requirements-real.txt /tmp/requirements-real.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir \
      -r /tmp/requirements-real.txt && rm /tmp/requirements-real.txt

ENV SHELL=/bin/bash
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
