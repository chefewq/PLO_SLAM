cmake_minimum_required(VERSION 3.1)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(yolov8-seg LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /home/haochen/slam/environment/TensorRT-8.6.1.6/include)
set(TensorRT_LIBRARIES /home/haochen/slam/environment/TensorRT-8.6.1.6/lib)
#find_package(TensorRT REQUIRED)

find_package(Eigen3 3.1.0 REQUIRED)

message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

list(APPEND INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include
    ${EIGEN3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    include/deepsort
)

list(APPEND ALL_LIBS
    ${CUDA_LIBRARIES}
    ${CUDA_LIB_DIR}
    ${OpenCV_LIBRARIES}
    ${TensorRT_LIBRARIES}
)

# 包含头文件目录
include_directories(${INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include 
include/yolo
include/deepsort
)

message(STATUS "Source files: ${PROJECT_SOURCES}")
message(STATUS "Include dirs: ${INCLUDE_DIRS}")

# 移到前面
link_directories(${CUDA_LIB_DIR} ${TensorRT_LIBRARIES})


# 安装头文件
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
# 添加库
add_library(yolo_deepsort_lib STATIC
    
    src/ImgSeg.cc
    src/deepsort/deepsort.cpp
    src/deepsort/featuretensor.cpp

    src/deepsort/hungarianoper.cpp
    src/deepsort/kalmanfilter.cpp
    src/deepsort/linear_assignment.cpp
    src/deepsort/munkres.cpp
    src/deepsort/nn_matching.cpp
    src/deepsort/track.cpp
    src/deepsort/tracker.cpp
    src/YoloSort.cc
)


install(TARGETS yolo_deepsort_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)


# 设置特定的宏
if (${OpenCV_VERSION} VERSION_GREATER_EQUAL 4.7.0)
    message(STATUS "Build with -DBATCHED_NMS")
    add_definitions(-DBATCHED_NMS)
endif()
