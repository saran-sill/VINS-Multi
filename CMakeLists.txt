cmake_minimum_required(VERSION 3.10)
project(vins_multi_workspace)

# ---------------------------------------------------------------
# Sanity check environment variables before descending into subdirs
# (both subprojects need these; check once here for a clearer error)
# ---------------------------------------------------------------
if(NOT DEFINED ENV{OPENCV_LIB_FOLDER})
    message(FATAL_ERROR "ERROR : Please set OPENCV_LIB_FOLDER in bashrc file.")
endif()

if(NOT DEFINED ENV{CERES_LIB_FOLDER})
    message(FATAL_ERROR "ERROR : Please set CERES_LIB_FOLDER in bashrc file.")
endif()

# ---------------------------------------------------------------
# Default install prefix -> <build_dir>/install
# Only override if the user hasn't explicitly set one via -DCMAKE_INSTALL_PREFIX=...
# ---------------------------------------------------------------
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install"
        CACHE PATH "Install path prefix" FORCE)
endif()
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# ---------------------------------------------------------------
# Build order matters: camera_models must be built before
# vins_estimator, since vins_estimator links against libcamera_models
# CMake processes add_subdirectory() calls in the order they appear,
# so this ordering is enough.
# ---------------------------------------------------------------
add_subdirectory(camera_models)
add_subdirectory(vins_estimator)