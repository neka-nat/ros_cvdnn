cmake_minimum_required(VERSION 3.0.2)
project(cvdnn_segmentation)

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

add_executable(cvdnn_segmentation_node src/segmentation_node.cpp)
target_link_libraries(cvdnn_segmentation_node ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  )

install(TARGETS cvdnn_segmentation_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
  )