/*
 *
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef MAP_PLANAR_OBJECT_H
#define MAP_PLANAR_OBJECT_H

#include "BoostArchiver.h"

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Pointers.h"

#include "Eigen/Core"
#include "Thirdparty/Sophus/sophus/se3.hpp"
#include "Thirdparty/Sophus/sophus/sim3.hpp"

#include "Ellipsoid.h"

namespace ORB_SLAM3 {

class KeyFrame;
class Map;
class Frame;

class MapObject {
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version);

public:
  typedef std::shared_ptr<MapObject> Ptr;
  typedef std::shared_ptr<const MapObject> ConstPtr;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MapObject();
  MapObject(const int id, const int plabel, Ellipsoid &ellipsoid, Map *map);
  Ellipsoid &GetEllipsoid();
  void SetEllipsoid(Ellipsoid ell);
  virtual ~MapObject() = default;

public:
  Map *GetMap();
  void UpdateMap(Map *pMap);

public:
  int track_id;
  int label;
  Eigen::Vector3d max_axes;
  std::vector<Eigen::Vector3d> trajetory;
    Eigen::VectorXd kf_x;      // 状态向量 [px, py, pz, vx, vy, vz]
    Eigen::MatrixXd kf_P;      // 协方差矩阵
  std::vector<std::vector<Eigen::Vector3d>> trajectory_segments;
protected:
  Map *mpMap;

  std::mutex mMutexMap;
  std::mutex mutex_ellipsoid_;
  Ellipsoid mpEllipsoid; // 自动管理内存

  // Reference KeyFrame

  bool is_moving;
};

} // namespace ORB_SLAM3

#endif // MAP_PLANAR_OBJECT_H
