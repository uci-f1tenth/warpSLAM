FROM autodriveecosystem/autodrive_roboracer_api:2026-icra-practice

RUN apt-get update && apt-get upgrade -y && apt-get install -y ros-$ROS_DISTRO-foxglove-bridge

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
