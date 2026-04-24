cmake_minimum_required(VERSION 2.8.3)
project(camera_models)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

# ---- RPATH configuration ----
# Use RPATH so it propagates to transitively loaded libraries
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--disable-new-dtags")

# Embed absolute paths to custom-built dependencies in the binary/library
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH   FALSE)   # use build-time RPATH while building
set(CMAKE_SKIP_BUILD_RPATH           FALSE)

#########################################################
# CHECK IF YOU SET ENVIRONMENT VARIABLES
#########################################################
if(NOT DEFINED ENV{OPENCV_LIB_FOLDER})
    MESSAGE(FATAL_ERROR "ERROR : Please set OPENCV_LIB_FOLDER in bashrc file.")
endif()

if(NOT DEFINED ENV{CERES_LIB_FOLDER})
    MESSAGE(FATAL_ERROR "ERROR : Please set CERES_LIB_FOLDER in bashrc file.")
endif()

set(OPENCV_LIB_FOLDER "$ENV{OPENCV_LIB_FOLDER}/build/install/lib/cmake/opencv4")
set(OpenCV_LIB_DIRS "${OPENCV_LIB_FOLDER}/../../")
set(CERES_LIB_FOLDER "$ENV{CERES_LIB_FOLDER}/build/install/lib/cmake/Ceres")
#########################################################

# find_package(catkin REQUIRED COMPONENTS
#     roscpp
#     std_msgs
#     )

find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV 4 REQUIRED PATHS ${OPENCV_LIB_FOLDER} NO_DEFAULT_PATH)
set(OPENCV_FULL_PATH_LIBS)
foreach(lib ${OpenCV_LIBS})
#   message(STATUS "OpenCV_LIB_DIRS: ${OpenCV_LIB_DIRS}")
  # This will look for libopencv_<xxx>.so under ${OpenCV_LIB_DIRS}
  find_library(_found_${lib}
               NAMES        ${lib}
               PATHS        ${OpenCV_LIB_DIRS}
               NO_DEFAULT_PATH)
  if(NOT _found_${lib})
    message(FATAL_ERROR "Couldn't locate ${lib} in ${OpenCV_LIB_DIRS}")
  endif()
  list(APPEND OPENCV_FULL_PATH_LIBS ${_found_${lib}})
endforeach()
include_directories(${OpenCV_INCLUDE_DIRS})

# set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
find_package(Ceres REQUIRED PATHS ${CERES_LIB_FOLDER} NO_DEFAULT_PATH)
include_directories(${CERES_INCLUDE_DIRS})


# catkin_package(
#     INCLUDE_DIRS include
#     LIBRARIES camera_models
#     CATKIN_DEPENDS roscpp std_msgs
# #    DEPENDS system_lib
#     )

# include_directories(
#     ${catkin_INCLUDE_DIRS}
#     )

#------------- ROS-ralated library -------------#
find_package(roscpp REQUIRED)
include_directories(${roscpp_INCLUDE_DIRS})

include_directories("include")

add_executable(Calibrations
    src/intrinsic_calib.cc
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/PinholeFullCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc)

add_library(camera_models SHARED
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/PinholeFullCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc)

target_link_libraries(Calibrations ${Boost_LIBRARIES} ${OPENCV_FULL_PATH_LIBS} ${CERES_LIBRARIES} ${roscpp_LIBRARIES})
target_link_libraries(camera_models ${Boost_LIBRARIES} ${OPENCV_FULL_PATH_LIBS} ${CERES_LIBRARIES} ${roscpp_LIBRARIES})


# ---- install rules ----
install(TARGETS camera_models Calibrations
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)