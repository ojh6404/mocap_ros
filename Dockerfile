FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list

RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN apt update

# install minimum tools:
RUN apt install -y build-essential sudo git

RUN \
  useradd user && \
  echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && \
  chmod 0440 /etc/sudoers.d/user && \
  mkdir -p /home/user && \
  chown user:user /home/user && \
  chsh -s /bin/bash user

RUN echo 'root:root' | chpasswd
RUN echo 'user:user' | chpasswd

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    curl \
    wget \
    libglu1-mesa \
    libxi-dev \
    libxmu-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libosmesa6-dev \
    ffmpeg \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN sudo apt-get update && apt-get install -y lsb-release
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# setup keys
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    python-is-python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install python3-osrf-pycommon python3-catkin-tools python3-wstool -y
RUN apt update && apt install ros-noetic-jsk-tools ros-noetic-jsk-recognition-utils ros-noetic-jsk-recognition-msgs -y

WORKDIR /home/user

USER user
SHELL ["/bin/bash", "-c"]


ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA=True
ENV CUDA_HOME="/usr/local/cuda-11.3"
# trick for cuda because cuda is not available when building docker image
ENV FORCE_CUDA="1" TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
# use gdown to download models from google drive
ENV PATH="$PATH:/home/user/.local/bin"

########################################
########### WORKSPACE BUILD ############
########################################
# Installing catkin package
RUN mkdir -p ~/catkin_ws/src
RUN sudo rosdep init && rosdep update && sudo apt update
RUN cd ~/catkin_ws/src && git clone https://github.com/ojh6404/hand_object_detection_ros.git
RUN cd ~/catkin_ws/src/hand_object_detection_ros && ./prepare.sh
RUN cd ~/catkin_ws/src/ &&\
    source /opt/ros/noetic/setup.bash &&\
    rosdep install --from-paths . --ignore-src -y -r &&\
    cd ~/catkin_ws/src/hand_object_detection_ros &&\
    cd ~/catkin_ws && catkin init && catkin build

COPY --chown=user node_scripts /home/user/catkin_ws/src/hand_object_detection_ros/node_scripts

# to avoid conflcit when mounting
RUN rm -rf ~/catkin_ws/src/hand_object_detection_ros/launch

#########################################
############ ENV VARIABLE STUFF #########
#########################################
RUN touch ~/.bashrc
RUN echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

CMD ["bash"]
