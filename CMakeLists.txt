cmake_minimum_required(VERSION 3.0.2)
project(mocap_ros)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  geometry_msgs
  jsk_recognition_msgs
  jsk_topic_tools
  message_generation
)

catkin_python_setup()

add_message_files(
    FILES
    MocapDetection.msg
    MocapDetectionArray.msg
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

# if(CATKIN_ENABLE_TESTING)
#   add_custom_target(${PROJECT_NAME}_download_image ALL
#     COMMAND ${PROJECT_SOURCE_DIR}/test/prepare_test_data.sh)
#   find_package(catkin REQUIRED COMPONENTS rostest)
#   add_rostest(test/test_node.test)
# endif()
