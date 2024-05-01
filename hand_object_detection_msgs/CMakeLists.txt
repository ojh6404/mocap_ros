cmake_minimum_required(VERSION 3.0.2)
project(hand_object_detection_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  geometry_msgs
  jsk_recognition_msgs
  message_generation
)

add_message_files(
  FILES
  HandDetection.msg
  HandDetectionArray.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs
    jsk_recognition_msgs
)

catkin_package(
  CATKIN_DEPENDS
  std_msgs
  sensor_msgs
  geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
