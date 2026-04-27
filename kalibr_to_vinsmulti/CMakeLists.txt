cmake_minimum_required(VERSION 3.10)
project(kalibr_to_vinsmulti)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{OPENCV_LIB_FOLDER})
    MESSAGE(FATAL_ERROR "ERROR : Please set OPENCV_LIB_FOLDER in bashrc file.")
endif()
set(OpenCV_LIB_DIRS "${OPENCV_LIB_FOLDER}/../../")

# Match the OpenCV lookup style used by the rest of this workspace
set(OPENCV_LIB_FOLDER "$ENV{OPENCV_LIB_FOLDER}/build/install/lib/cmake/opencv4")
set(OpenCV_DIR $ENV{OPENCV_LIB_FOLDER})

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

find_package(Eigen3 REQUIRED)

add_executable(kalibr_to_vinsmulti kalibr_to_vinsmulti.cpp)

target_include_directories(kalibr_to_vinsmulti PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(kalibr_to_vinsmulti PRIVATE
    ${OPENCV_FULL_PATH_LIBS}
    stdc++fs            # std::filesystem on older GCC; harmless on newer
)

install(TARGETS kalibr_to_vinsmulti
    RUNTIME DESTINATION bin)