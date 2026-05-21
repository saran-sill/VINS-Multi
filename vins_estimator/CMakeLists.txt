cmake_minimum_required(VERSION 2.8.3)
project(vins_multi)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# ---- RPATH configuration ----
# Use RPATH so it propagates to transitively loaded libraries
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--disable-new-dtags")

# Embed absolute paths to custom-built dependencies in the binary/library
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH   FALSE)   # use build-time RPATH while building
set(CMAKE_SKIP_BUILD_RPATH           FALSE)

list(APPEND CMAKE_INSTALL_RPATH
    "$ENV{OPENCV_LIB_FOLDER}/build/install/lib"
    "$ENV{CERES_LIB_FOLDER}/build/install/lib"
    "${CMAKE_INSTALL_PREFIX}/lib"    # <-- so installed binaries find installed .so files
)

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
set(CERES_INCLUDE_DIRS "$ENV{CERES_LIB_FOLDER}/build/install/include")
#########################################################

set(ENABLE_BACKWARD false)
set(CUDA false)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# find_package(catkin REQUIRED COMPONENTS
#     roscpp
#     std_msgs
#     geometry_msgs
#     nav_msgs
#     tf
#     cv_bridge
#     camera_models
#     image_transport
#     nodelet
#     message_filters
#     sensor_msgs)

# find_package(OpenCV REQUIRED)
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

message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

# ---- camera_models (sibling subproject) ----
set(CAMERA_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../camera_models)
include_directories(${CAMERA_MODELS_DIR}/include)
# The `camera_models` target is defined by ../camera_models/CMakeLists.txt
# and is available because the top-level CMakeLists adds it first.
set(CAMERA_MODELS_LIB camera_models)

# find_package(Ceres REQUIRED)
find_package(Ceres REQUIRED PATHS ${CERES_LIB_FOLDER} NO_DEFAULT_PATH)
include_directories(${CERES_INCLUDE_DIRS})

if (CUDA)
  find_package(CUDA QUIET)
  if (CUDA_FOUND)
    message("-- Found CUDA version ${CUDA_VERSION}: "
        "${CUDA_LIBRARIES};"
        "${CUDA_cusolver_LIBRARY};"
        "${CUDA_cusparse_LIBRARY}")
    include_directories(
      ${CUDA_INCLUDE_DIRS}
    )
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      message("embed_platform on")
      include_directories(/usr/local/cuda/targets/aarch64-linux/include)
      link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
    else()
      message("embed_platform off")
      link_directories(/usr/local/cuda-11.4/targets/x86_64-linux/lib/)
    endif()

  else (CUDA_FOUND)
    message("-- Did not find CUDA library, disabling CUDA support.")
    update_cache_variable(CUDA OFF)
    list(APPEND CERES_COMPILE_OPTIONS CERES_NO_CUDA)
  endif (CUDA_FOUND)
else (CUDA)
  message("-- Building without CUDA.")
  list(APPEND CERES_COMPILE_OPTIONS CERES_NO_CUDA)
endif (CUDA)

if(CUDA)
    add_compile_definitions(WITH_CUDA)
endif (CUDA)

if(ENABLE_BACKWARD)
    add_definitions(-D USE_BACKWARD)
    set(LIBDW "dw")
else()
    set(LIBDW "")
endif()

# include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# find_package(Eigen3)
find_package(Eigen3 REQUIRED)
include_directories(
#   ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

#------------- DBow3 library -------------#
if(NOT DEFINED ENV{DBOW3_FOLDER})
    MESSAGE(FATAL_ERROR "ERROR : Please set DBOW3_FOLDER in bashrc file.")
endif()

set(DBOW3_FOLDER "$ENV{DBOW3_FOLDER}/lib/cmake/DBoW3")

find_package(DBoW3 REQUIRED PATHS ${DBOW3_FOLDER} NO_DEFAULT_PATH)

include_directories(${DBoW3_INCLUDE_DIRS})

#------------- ROS-ralated library -------------#
find_package(roscpp REQUIRED)
find_package(tf REQUIRED)
find_package(image_transport REQUIRED)

# cv_bridge
if(NOT DEFINED ENV{CV_BRIDGE_CMAKE_FOLDER})
    MESSAGE(FATAL_ERROR "ERROR : Please set CV_BRIDGE_CMAKE_FOLDER in bashrc file.")
endif()
set(CV_BRIDGE_LIB_FOLDER "$ENV{CV_BRIDGE_CMAKE_FOLDER}") # path to cv_bridgeConfig.cmake
find_package(cv_bridge REQUIRED PATHS ${CV_BRIDGE_LIB_FOLDER} NO_DEFAULT_PATH)

include_directories(${roscpp_INCLUDE_DIRS} ${cv_bridge_INCLUDE_DIRS} ${image_transport_INCLUDE_DIRS})

list(APPEND ros_libraries
    ${roscpp_LIBRARIES}
    ${tf_LIBRARIES}
    ${image_transport_LIBRARIES}
    ${cv_bridge_LIBRARIES}
)

# catkin_package()

add_library(colmap_util_lib SHARED
    ${CMAKE_SOURCE_DIR}/colmap/colmap_util.cpp
)
target_link_libraries(colmap_util_lib
    ${OPENCV_FULL_PATH_LIBS}
)

include_directories(
        ${CMAKE_SOURCE_DIR}/colmap
)

add_library(mesh_util_lib SHARED
    ${CMAKE_SOURCE_DIR}/mesh_utils/read_ply.cpp
)

include_directories(
        ${CMAKE_SOURCE_DIR}/mesh_utils
)

add_library(point_features_lib SHARED
    ${CMAKE_SOURCE_DIR}/point_features/point_features.cpp
)
target_link_libraries(point_features_lib
    ${OPENCV_FULL_PATH_LIBS})
include_directories(
        ${CMAKE_SOURCE_DIR}/point_features
)

add_library(dbow3_util_lib STATIC
    ${CMAKE_SOURCE_DIR}/dbow3/dbow3_util.cpp
)
target_link_libraries(dbow3_util_lib
    ${DBoW3_LIBS}
)

include_directories(
        ${CMAKE_SOURCE_DIR}/dbow3
)

add_library(parameter_lib_multi SHARED
    src/estimator/parameters.cpp
)
target_link_libraries(parameter_lib_multi
    ${ros_libraries} ${OPENCV_FULL_PATH_LIBS} ${LIBDW})

add_library(utility_lib_multi SHARED
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
)
target_link_libraries(utility_lib_multi
    ${ros_libraries} ${LIBDW})


add_library(frontend_lib_multi SHARED
    src/featureTracker/feature_tracker.cpp
)
target_link_libraries(frontend_lib_multi
    parameter_lib_multi ${CAMERA_MODELS_LIB} ${ros_libraries} ${OPENCV_FULL_PATH_LIBS} ${LIBDW})

add_library(init_lib_multi SHARED
    src/initial/initial_aligment.cpp
)
target_link_libraries(init_lib_multi
    ${ros_libraries} ${LIBDW})

add_library(factor_lib_multi SHARED
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/depthFactor.cpp
    src/factor/projectionTwoFrameOneCamDepthFactor.cpp
    src/factor/marginalization_factor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
)
target_link_libraries(factor_lib_multi
    ${ros_libraries} ${OPENCV_FULL_PATH_LIBS} ${CERES_LIBRARIES} ${LIBDW})


add_library(estimator_lib_multi SHARED
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
)
target_link_libraries(estimator_lib_multi
    parameter_lib_multi utility_lib_multi init_lib_multi factor_lib_multi mesh_util_lib
    ${CAMERA_MODELS_LIB} ${ros_libraries} ${OPENCV_FULL_PATH_LIBS} ${CERES_LIBRARIES} ${LIBDW})

add_library(gloc_lib SHARED
    src/gloc/gloc.cpp
    src/gloc/parameters.cpp
)
target_link_libraries(gloc_lib
    colmap_util_lib dbow3_util_lib point_features_lib
    parameter_lib_multi
    mesh_util_lib
    ${CAMERA_MODELS_LIB}
    ${CERES_LIBRARIES}
    ${ros_libraries}
)

add_library(rosnode_lib_multi SHARED
    src/rosNode.cpp
)
target_link_libraries(rosnode_lib_multi ${ros_libraries} ${OPENCV_FULL_PATH_LIBS} ${LIBDW})

add_executable(${PROJECT_NAME} src/rosNodeMain.cpp)
target_link_libraries(${PROJECT_NAME} rosnode_lib_multi
                                      parameter_lib_multi
                                      utility_lib_multi
                                      estimator_lib_multi
                                      frontend_lib_multi
                                      init_lib_multi
                                      factor_lib_multi
                                      colmap_util_lib
                                      point_features_lib
                                      mesh_util_lib
                                      dbow3_util_lib
                                      gloc_lib
                                      ${LIBDW})

# add_library(${PROJECT_NAME}_nodelet_lib src/rosNodelet.cpp)
# target_link_libraries(${PROJECT_NAME}_nodelet_lib rosnode_lib_multi parameter_lib_multi utility_lib_multi estimator_lib_multi frontend_lib_multi init_lib_multi factor_lib_multi ${LIBDW})

# ---- install rules ----
install(TARGETS
    ${PROJECT_NAME}                  # vins_multi executable
    # ${PROJECT_NAME}_nodelet_lib      # nodelet lib
    parameter_lib_multi
    utility_lib_multi
    frontend_lib_multi
    init_lib_multi
    factor_lib_multi
    estimator_lib_multi
    rosnode_lib_multi
    colmap_util_lib
    point_features_lib
    mesh_util_lib
    dbow3_util_lib
    gloc_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install launch files, config, and nodelet plugin descriptor so
# they're available alongside the binaries
install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
    OPTIONAL
)

install(FILES nodelet_plugins.xml
    DESTINATION share/${PROJECT_NAME}
    OPTIONAL
)

# install(FILES
#     ${CMAKE_SOURCE_DIR}/colmap/colmap_util.h
#     ${CMAKE_SOURCE_DIR}/point_features/point_features.h
#     ${CMAKE_SOURCE_DIR}/gloc/gloc.h
#     DESTINATION include
# )