# hand_object_detection_ros 

ROS1 wrapper package for hand detection and mocap with [hand_object_detector](https://github.com/ddshan/hand_object_detector.git) and [frankmocap](https://github.com/facebookresearch/frankmocap.git).

![Alt text](asset/hand_object_detection_example.gif)

## Setup

### Prerequisite
This package is build upon
- ROS1 (Noetic)
- `torch==1.12` and `cuda-11.3`
- (Optional) docker and nvidia-container-toolkit (for environment safety)

### Build package

#### on your workspace
It is better to use docker environment cause it needs specific cuda version and build environment. But you can build it directly if you want provided that you use `cuda-11.3` and `torch==1.12`. Instruction's are below.
```bash
mkdir -p ~/ros/catkin_ws/src && cd ~/ros/catkin_ws/src
git clone https://github.com/ojh6404/hand_object_detection_ros.git
cd ~/ros/catkin_ws/src/hand_object_detection_ros
./prepare.sh # install torch and build python submodules
cd ~/ros/catkin_ws && catkin b
```

#### using docker (Recommended)
Otherwise, you can build this package on docker environment.
```bash
git clone https://github.com/ojh6404/hand_object_detection_ros.git
cd hand_object_detection_ros && catkin bt # to build message
docker build -t hand_object_detection_ros .
```

## Usage
### 1. run directly
```bash
roslaunch hand_object_detection_ros sample.launch \
    input_image:=/kinect_head/rgb/image_rect_color \
    device:=cuda:0 \
    with_handmocap:=true
```
### 2. using docker
You can run on docker by
```bash
./run_docker -host pr1040 -mount ./launch -name sample.launch \
    input_image:=/kinect_head/rgb/image_rect_color \
    device:=cuda:0 \
    with_handmocap:=true
```
where
- `-host` : hostname like `pr1040` or `localhost`
- `-mount` : mount launch file directory for launch inside docker.
- `-name` : launch file name to run

launch args below.
- `input_image` : input image topic
- `device` : which device to use. `cpu` or `cuda`. default is `cuda:0`.
- `hand_threshold` : hand detection threshold. default is `0.9`.
- `object_threshold` : object detection threshold. default is `0.9`.
- `with_handmocap` : use frankmocap or not. if you need faster detection and don't need mocap, then set `false`. default is `true`.

### Output topic
- `~hand_detections` : `HandDetectionArray`. array of hand detection results. please refer to `msg`
- `~debug_image` : `Image`. image for visualization.

### TODO
add rostest and docker build test.
