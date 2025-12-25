# POL-SLAM

A robust visual SLAM system integrating **Point**, **Object**, and **Line** features for dynamic environments.

## Overview

POL-SLAM is an advanced visual SLAM system built upon ORB-SLAM3, designed to operate robustly in dynamic environments by incorporating multiple geometric features:

- **Point Features**: Traditional ORB features for accurate visual odometry
- **Line Features**: Structural line segments for enhanced geometric constraints
- **Object Features**: Semantic object detection and tracking using YOLO with TensorRT acceleration

The system supports both RGB-D and stereo camera configurations and is capable of dense point cloud reconstruction.

## Key Features

- Multi-feature fusion (points, lines, objects) for robust tracking
- Real-time object detection with YOLO + TensorRT optimization
- Dynamic object handling for improved accuracy in crowded scenes
- Dense 3D point cloud mapping with PCL integration
- Support for multiple camera types (RealSense D435i, ZED, OAK-D, etc.)
- Compatible with popular benchmarks (EuRoC, TUM RGB-D, KITTI)

## Dependencies

### Required

- **CMake** >= 2.8
- **C++11** compiler (GCC, Clang)
- **OpenCV** 4.10.0 (specified in CMakeLists.txt)
- **Eigen3** (linear algebra)
- **Pangolin** >= 0.6 (visualization)
- **PCL** 1.12 (point cloud processing)
- **Boost** (thread, system, serialization)
- **CUDA** (for GPU acceleration)
- **TensorRT** 8.6.1.6 (for YOLO inference)
- **glog** (logging)
- **Protobuf** (serialization)
- **GLFW** (OpenGL support)

### Third-party Libraries (included)

- **DBoW2**: Place recognition
- **g2o**: Graph optimization
- **Sophus**: Lie algebra
- **line_descriptor**: Line feature extraction

### Optional

- **RealSense SDK** (if using Intel RealSense cameras)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chefewq/POL-SLAM.git
cd POL-SLAM
```

### 2. Build Third-party Libraries and Main System

Run the provided build script:

```bash
chmod +x build.sh
./build.sh
```

This script will:
- Build DBoW2 for vocabulary management
- Build g2o for graph optimization
- Build line_descriptor for line feature extraction
- Build Sophus for Lie algebra operations
- Extract the ORB vocabulary
- Build the main POL-SLAM library

### 3. Manual Build (Alternative)

If you prefer to build manually:

```bash
# Build third-party libraries
cd Thirdparty/DBoW2
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../g2o
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../line_descriptor
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../Sophus
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Extract vocabulary
cd ../../../Vocabulary
tar -xf ORBvoc.txt.tar.gz

# Build main system
cd ..
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## Usage

### RGB-D Mode (TUM Dataset)

```bash
./Examples/RGB-D/rgbd_tum \
    ./Vocabulary/ORBvoc.txt \
    ./Settings/InSeg.yaml \
    /path/to/dataset \
    /path/to/associations.txt
```

### Stereo Mode (EuRoC Dataset)

```bash
./Examples/Stereo/stereo_euroc \
    ./Vocabulary/ORBvoc.txt \
    ./Examples/Stereo/EuRoC.yaml \
    /path/to/euroc/dataset \
    ./Examples/Stereo/EuRoC_TimeStamps/MH05.txt
```

### Example Scripts

Several example run scripts are provided:

```bash
# EuRoC MH05 sequence
./run_mh05.sh

# EuRoC V101 sequence
./run_v101.sh

# Custom dataset
./run_mydataset.sh
```

## Configuration

Camera and system parameters are configured via YAML files in the `Settings/` directory:

- `d435.yaml`, `d435_hres.yaml`, `d435_mres.yaml`: Intel RealSense D435i configurations
- `zed.yaml`, `zed_mini.yaml`, `zed2i.yaml`: Stereolabs ZED camera configurations
- `oak-d.yaml`, `oak-d-vga.yaml`: Luxonis OAK-D camera configurations
- `InSeg.yaml`: RGB-D configuration with segmentation
- `EuRoC.yaml`: Configuration for EuRoC MAV datasets

### Key Parameters

Modify the YAML configuration files to adjust:
- Camera intrinsics and distortion parameters
- Feature extraction settings (ORB, line features)
- Tracking and mapping parameters
- Object detection thresholds
- Point cloud resolution

## Datasets

The system has been tested on:

- **EuRoC MAV Dataset**: Stereo + IMU sequences (MH01-05, V101-203)
- **TUM RGB-D Dataset**: RGB-D sequences for indoor environments
- **KITTI Dataset**: Outdoor stereo sequences (00-12)
- **TUM-VI Dataset**: Visual-inertial sequences

Download datasets:
- EuRoC: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
- TUM RGB-D: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- KITTI: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

## Output

The system generates:

- **CameraTrajectory.txt**: Full camera trajectory in TUM format
- **KeyFrameTrajectory.txt**: Keyframe trajectory for evaluation
- **resultPointCloudFile.pcd**: Dense point cloud map (PCL format)

### Trajectory Format (TUM)

```
timestamp tx ty tz qx qy qz qw
```

## Evaluation

Trajectory evaluation tools are provided in `=--evaluation/`:

```bash
# Evaluate Absolute Trajectory Error (ATE)
python =--evaluation/evaluate_ate_scale.py \
    /path/to/groundtruth.txt \
    POL_CameraTrajectory.txt
```

Ground truth files for EuRoC sequences are included in `=--evaluation/Ground_truth/`.

## Project Structure

```
POL-SLAM/
├── build.sh                  # Automated build script
├── CMakeLists.txt            # Main CMake configuration
├── Examples/
│   ├── RGB-D/                # RGB-D examples
│   └── Stereo/               # Stereo examples
├── include/                  # Header files
│   ├── System.h              # Main system interface
│   ├── Tracking.h            # Visual tracking
│   ├── LocalMapping.h        # Local mapping
│   ├── LoopClosing.h         # Loop closure detection
│   ├── PointCloudMapper.h    # Dense mapping
│   ├── ObjectTrack.h         # Object detection/tracking
│   ├── MapLine.h             # Line feature mapping
│   └── ...
├── src/                      # Implementation files
├── Settings/                 # Camera/system configurations
├── Thirdparty/               # Third-party libraries
│   ├── DBoW2/                # Bag of words
│   ├── g2o/                  # Graph optimization
│   ├── Sophus/               # Lie algebra
│   └── line_descriptor/      # Line features
├── Vocabulary/               # ORB vocabulary
├── yolo_tensort/             # YOLO+TensorRT implementation
├── =--evaluation/            # Evaluation scripts
└── run_*.sh                  # Example run scripts
```

## Camera Support

POL-SLAM supports various camera configurations:

- **Intel RealSense**: D435, D435i, R200
- **Stereolabs ZED**: ZED, ZED Mini, ZED 2i
- **Luxonis**: OAK-D
- **Stereo cameras**: Calibrated stereo rigs
- **RGB-D cameras**: Any RGB-D camera with proper calibration

## Notes

- Ensure CUDA and TensorRT are properly installed for object detection features
- For optimal performance, use GPU with CUDA capability >= 6.1
- TensorRT path is currently hardcoded in CMakeLists.txt (line 144-145), modify if needed:
  ```cmake
  set(TensorRT_INCLUDE_DIRS /path/to/TensorRT/include)
  set(TensorRT_LIBRARIES /path/to/TensorRT/lib)
  ```
- The system requires substantial computational resources for real-time operation with all features enabled

## Troubleshooting

### Common Issues

1. **TensorRT not found**: Update the TensorRT paths in `CMakeLists.txt` lines 144-145
2. **OpenCV version mismatch**: The system expects OpenCV 4.10.0, but should work with 4.x versions
3. **PCL errors**: Ensure PCL 1.12 is installed with all dependencies
4. **CUDA errors**: Verify CUDA installation and compatible GPU driver

### Build Flags

To disable certain features during build, modify `CMakeLists.txt`:

```cmake
set(WITH_G2O_NEW        OFF)  # Use bundled g2o instead of system g2o
set(WITH_OPENMP         ON)   # Enable OpenMP parallelization
set(WITH_TRACY_PROFILER OFF)  # Disable Tracy profiler
```

## License

POL-SLAM is released under GPLv3 license.

This project builds upon:
- **ORB-SLAM3** (GPLv3): Carlos Campos et al., University of Zaragoza
- **ORB-SLAM2** (GPLv3): Raúl Mur-Artal et al.
- Contributions from Luigi Freda

See individual source files for detailed copyright information.

## Citation

```bibtex
@article{campos2021orb,
  title={ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM},
  author={Campos, Carlos and Elvira, Richard and Rodr{\'\i}guez, Juan J G{\'o}mez and Montiel, Jos{\'e} MM and Tard{\'o}s, Juan D},
  journal={IEEE Transactions on Robotics},
  volume={37},
  number={6},
  pages={1874--1890},
  year={2021}
}
```

## Contact

For questions and issues, please open an issue on GitHub or contact 2405676953@qq.com

## Acknowledgments

We thank the authors of ORB-SLAM3, DBoW2, g2o, OA-SLAM, and all other open-source libraries used in this project.
