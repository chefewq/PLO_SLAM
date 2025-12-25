cmake_minimum_required(VERSION 2.8)
project(ORBSLAM3_dense)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

#====================================
# options

set(OPENCV_VERSION "4.10.0" CACHE STRING "Desired OpenCV version") # this is the first target version that is searched for 

set(BUILD_WITH_MARCH_NATIVE ON  CACHE BOOL "Build with \"-march native\"")

set(WITH_G2O_NEW        OFF CACHE BOOL "Add g2o new support") # for using the new version of g2o with Cholmod and Csparse solvers (and more...)
set(WITH_OPENMP         ON  CACHE BOOL "Add OpenMP support") # useful for PCL (since we use the macro PCL_NO_PRECOMPILE!)
set(WITH_TRACY_PROFILER OFF CACHE BOOL "Add Tracy profiler support.")

#====================================
# c++ standard 

# Set the C++ standard 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#====================================
# general cmake options 

# NOTE: this is to remove the tons of cmake warning coming from importing ext packages 
# see here: https://github.com/PointCloudLibrary/pcl/issues/3680
# when this is fixed, we can remove the following 3 lines.
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

# This option allows the generations of a file compile_commands.json in our build folder: that file contains the full command line to compile individual source files
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#====================================
# compiler and build options 

set(LIBRARY_TYPE SHARED)
#set(LIBRARY_TYPE STATIC)  

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_BUILD_TYPE Debug)  # force Debug 

message("Build type: " ${CMAKE_BUILD_TYPE})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

if(CMAKE_BUILD_TYPE STREQUAL "Release")

    # Best release
    set(MY_FLAGS "-Wall -O3 -fPIC -DNDEBUG -Wno-unused-parameter -Wno-unused-function") # -Wextra -Wno-deprecated

    if(BUILD_WITH_MARCH_NATIVE)
        set(MY_FLAGS "${MY_FLAGS} -march=native")      
    endif()

    #set(MY_FLAGS "-Wall -Wextra -O3 -pthread -ftree-vectorize -funroll-loops")

    # Best release Nvidia Jetson TX2
    #set(MY_FLAGS "-Wall -O3 -ftree-vectorize -funroll-loops -pthread -Wextra")

    if(WITH_TX2)
        # Best release Nvidia Jetson TX2
        #set(MY_FLAGS "-Wall -O3 -march=armv8-a+crypto -mcpu=cortex-a57+crypto -Wextra -ffast-math -flto -fPIC -DNDEBUG")
    endif()
else()

    message(AUTHOR_WARNING "\n!!!Compiling in Debug mode!!!\n")
    # Debug
    #set(MY_FLAGS "-Wall -O3  -g -pthread")

    # Debug with g2o compiled with -march=native; this should avoid g2o crashes (see "Mismatched build flags" on https://github.com/ethz-asl/eigen_catkin/wiki/Eigen-Memory-Issues#memory-misalignment )
    set(MY_FLAGS "-Wall  -g -O3 -fPIC -pthread") # -fsanitize=address  

    if(BUILD_WITH_MARCH_NATIVE)
        set(MY_FLAGS "${MY_FLAGS} -march=native")      
    endif()

    # this makes g2o crash if g2o is compiled with optimizations 
    #set(MY_FLAGS "-Wall  -g")
endif()

# Add OpenMP flags
if( WITH_OPENMP )
    set(MY_FLAGS "${MY_FLAGS} -fopenmp")   
endif()

# get and store all the active flags 
set(MY_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(MY_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")    

# apply flags 
set(CMAKE_C_FLAGS "${MY_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${MY_CXX_FLAGS}")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(-DCOMPILEDWITHC11)  #TODO: replace the macro everywhere with COMPILEDWITHC14 or completely remove it

set(SHADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" CACHE PATH "Where the shaders live")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSHADERS_DIR=${SHADERS_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHADERS_DIR=${SHADERS_DIR}")

#====================================
# cmake modules

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#====================================

#====================================
# packages 




find_package(Boost REQUIRED COMPONENTS thread system serialization)

message(STATUS "Desired OpenCV version: ${OPENCV_VERSION}")
find_package(OpenCV 4.10.0 REQUIRED)
message(STATUS "found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "opencv include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv lib dirs: ${OpenCV_INSTALL_PATH}")
message(STATUS "opencv libs: ${OpenCV_LIBS}")



find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(Pangolin 0.6 REQUIRED)
find_package(realsense2)

find_package(CUDA REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(TensorRT_INCLUDE_DIRS /home/haochen/slam/environment/TensorRT-8.6.1.6/include)
set(TensorRT_LIBRARIES /home/haochen/slam/environment/TensorRT-8.6.1.6/lib)
#find_package(TensorRT REQUIRED)

# find_package(catkin REQUIRED COMPONENTS roscpp tf pcl_conversions)
link_directories(${TensorRT_LIBRARIES})

find_package(PCL 1.12 REQUIRED)
add_definitions(-DPCL_NO_PRECOMPILE)  # this is strictly required since we defined a new pcl point type

add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS})

set(GLOG_INCLUDE_DIR "/usr/local/include/glog")
set(GLOG_LIBRARIES "/usr/local/lib/libglog.a")
find_package(GLOG REQUIRED)
# if(NOT GLOG_FOUND)
# 	message(FATAL_ERROR "please run: sudo apt-get install libgoogle-glog-dev " )
# endif()

find_package(Protobuf REQUIRED)
message(STATUS "protobuf libs: ${PROTOBUF_LIBRARIES}")

find_package(LibCrypto REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)



if( WITH_G2O_NEW )
    # In order to set RPATH (instead of RUNPATH) for g2o and other libraries linking issue.
    # ISSUE: g2o binaries link to system g2o (here /opt/ros/noetic/lib/libg2o_core.so) and crash (different version); 
    # we need to set LD_LIBRARY_PATH for correct linking
    # https://stackoverflow.com/questions/47117443/dynamic-linking-with-rpath-not-working-under-ubuntu-11-10 
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--disable-new-dtags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--disable-new-dtags")          
    find_package(Cholmod)
    include_directories(${CHOLMOD_INCLUDE_DIR})        
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_G2O_NEW")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_G2O_NEW")    
    # NOTE: use the following block of lines to test latest g2o version support.
    #       If it crashes. Check this link: https://github.com/rst-tu-dortmund/teb_local_planner/issues/336
    set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/Thirdparty/g2o_new/install/lib/cmake/g2o/)
    set(G2O_ROOT ${PROJECT_SOURCE_DIR}/Thirdparty/g2o_new/install/)
    message(STATUS "set custom g2o root folder: ${G2O_ROOT}")
    find_package(G2O REQUIRED)
    # used in the latest g2o version 
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DG2O_USE_VENDORED_CERES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DG2O_USE_VENDORED_CERES")
else()
    # g2o provided with ORBSLAM3
    set(G2O_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/)
    set(G2O_LIBS ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so)
endif()
if(WITH_TRACY_PROFILER)
  if(EXISTS "/usr/local/lib/libTracyClient.a")
    message(STATUS "Found Tracy")
    add_definitions(-DTRACY_ENABLE)
    set(TRACY_LIBS /usr/local/lib/libTracyClient.a -ldl -lpthread)
  endif()
endif()



#====================================
# includes 
include_directories(
# ${catkin_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${G2O_INCLUDE_DIR}
${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/include
${PROJECT_SOURCE_DIR}/yolo_tensort/include
${LIBELAS_INCLUDES}
${LIBSGM_INCLUDES}
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${Pangolin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${GLOG_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
${GLFW_INCLUDE_DIRS}
${CHOLMOD_INCLUDE_DIR}
${CUDA_INCLUDE_DIRS}
${TensorRT_INCLUDE_DIRS}
)

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME} ${realsense_INCLUDE_DIR})
endif()

#====================================
# libs 

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


set(EXTERNAL_CORE_LIBS   # core external libs 
${PCL_LIBRARIES}
${Boost_LIBRARIES} boost_serialization boost_thread # FIXME: remove this and let cmake package finder get the info!
#Boost::boost # NOTE this does not work!
${Pangolin_LIBRARIES}
${G2O_LIBS}
${OpenCV_LIBS}
${GLOG_LIBRARIES}
${OCTOMAP_LIBRARIES}
${PROTOBUF_LIBRARIES}
${LibCrypto_LIBRARY}
${GLFW_LIBRARIES}
${CHOLMOD_LIBRARIES}
nvinfer
nvinfer_plugin
nvonnxparser
)

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    message(STATUS "found REALSENSE LIB: ${realsense2_LIBRARY}")
    set(EXTERNAL_CORE_LIBS ${EXTERNAL_CORE_LIBS} ${realsense2_LIBRARY})
endif()


set(EXTERNAL_LIBS  # Thirdparty libs 
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/lib/liblinedesc.a
${PROJECT_SOURCE_DIR}/yolo_tensort/build/libyolo_deepsort_lib.a
${LIBELAS_LIBRARIES}
${LIBSGM_LIBRARIES}
${CUDA_LIBRARIES} 
)

link_directories( 
  ${G2O_LIB_DIR}  
  ${PCL_LIBRARY_DIRS}
)

#====================================
# source files and folders to build 

#add_library(${PROJECT_NAME} SHARED
set(CPU_SOURCES 
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/G2oLineTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/OptimizableLineTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
src/PointCloudMapper.cpp
src/ObjectTrack.cc
src/Ellipse.cc
src/Ellipsoid.cc
include/PointCloudMapper.h
include/ObjectTrack.h
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/ShaderKannalaBrandtRawFeatures.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/G2oLineTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/OptimizableLineTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h
include/MapObject.h
include/SegObject.h
include/BoxUtils.h
include/Ellipse.h
include/Ellipsoid.h
include/Reconstruction.h
include/OptimizerObject.h
include/Distance.h
include/Kalmanfilter3d.h
###
src/MapLine.cc
src/LineExtractor.cc
src/LineMatcher.cc
src/Pointers.cc
src/MapObject.cc
src/SegObject.cc
src/KalmanFilter3d.cc
###
src/Ellipse.cc
src/Ellipsoid.cc
src/Reconstruction.cc
src/Distance.cc
###
src/LabelMap.cc
src/KeyFrameSearchTree.cc
###
src/g2o/types_sba_line.cpp
src/g2o/types_six_dof_expmap2.cpp
src/g2o/types_seven_dof_expmap2.cpp
)


add_library(${PROJECT_NAME} ${LIBRARY_TYPE}
    ${CUDA_OBJS}
    ${CPU_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
    ${EXTERNAL_LIBS} 
    ${EXTERNAL_CORE_LIBS}
)
target_compile_definitions(${PROJECT_NAME} PRIVATE FLANN_NO_SERIALIZATION)

set(CORE_LIBS ${PROJECT_NAME})

message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
message(STATUS "EXTERNAL_CORE_LIBS: ${EXTERNAL_CORE_LIBS}")


### Build examples



set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME}

)

if(EXISTS ${PROJECT_SOURCE_DIR}/test)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
    #add_subdirectory(${PROJECT_SOURCE_DIR}/test) # uncomment to build tests/examples
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

# add_executable(stereo_kitti
#         Examples/Stereo/stereo_kitti.cc)
# target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})