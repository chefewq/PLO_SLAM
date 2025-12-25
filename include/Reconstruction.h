#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include "Ellipsoid.h"
#include "BoxUtils.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "ImgSeg.h"
namespace ORB_SLAM3{

std::pair<bool, Ellipsoid>
ReconstructEllipsoidFromCenters(const std::vector<BBox2>& boxes,
                                std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>& Rts, 
                                const std::vector<float> depths,
                                const Eigen::Matrix3d& K);
}


#endif