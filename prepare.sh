#!/usr/bin/bash

git submodule update --init --recursive
./motion_capture/prepare.sh

# for ros build
pip install psutil empy==3.3.2 rospkg gnupg pycryptodomex catkin-tools wheel cython ninja importlib_metadata netifaces
