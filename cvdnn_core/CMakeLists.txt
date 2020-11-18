cmake_minimum_required(VERSION 3.0.2)
project(cvdnn_core)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
)

find_package(OpenCV)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  )

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES cvdnn_core
  CATKIN_DEPENDS
    roscpp
)

add_library(cvdnn_core src/utils.cpp)
target_link_libraries(cvdnn_core ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  )

install(TARGETS cvdnn_core
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
  )