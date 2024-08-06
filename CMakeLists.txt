cmake_minimum_required(VERSION 3.0.2)
project(mocap_ros)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  geometry_msgs
  jsk_recognition_msgs
  jsk_topic_tools
  message_generation
)

add_message_files(
    FILES
    Detection.msg
    DetectionArray.msg
    Mocap.msg
    MocapArray.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
    jsk_recognition_msgs
)


catkin_package(
  CATKIN_DEPENDS
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
