cmake_minimum_required(VERSION 3.0.2)
project(cvdnn_object_detection)

find_package(catkin REQUIRED
  COMPONENTS
    image_transport
    cv_bridge
    vision_msgs
    cvdnn_core
)

find_package(OpenCV)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  )

catkin_package(
  CATKIN_DEPENDS
    cvdnn_core
    roscpp
)

if (NOT EXISTS "${CMAKE_SOURCE_DIR}/models/yolov3-tiny.weights")
  file(DOWNLOAD
       https://pjreddie.com/media/files/yolov3.weights
       ${CMAKE_SOURCE_DIR}/models/yolov3.weights
       SHOW_PROGRESS)
endif()

add_executable(cvdnn_object_detection_node src/object_detection_node.cpp)
target_link_libraries(cvdnn_object_detection_node ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  )

install(TARGETS cvdnn_object_detection_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
  )