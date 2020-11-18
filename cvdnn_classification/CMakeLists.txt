cmake_minimum_required(VERSION 3.0.2)
project(cvdnn_classification)

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

if (NOT EXISTS "${CMAKE_SOURCE_DIR}/models/bvlc_googlenet.caffemodel")
  file(DOWNLOAD
       http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
       ${CMAKE_SOURCE_DIR}/models/bvlc_googlenet.caffemodel
       SHOW_PROGRESS)
endif()

add_executable(cvdnn_classification_node src/classification_node.cpp)
target_link_libraries(cvdnn_classification_node ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  )

install(TARGETS cvdnn_classification_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
  )