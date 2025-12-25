//
// Created by yuwenlu on 2022/7/2.
//

#ifndef ORB_SLAM3_POINTCLOUDMAPPER_H
#define ORB_SLAM3_POINTCLOUDMAPPER_H
#include <iostream>
#include <fstream>
#include <pcl/common/transforms.h>
#include <Eigen/Geometry>
#include <boost/format.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <opencv2/core/core.hpp>
#include "ImgSeg.h"
#include <opencv2/highgui/highgui.hpp>
#include "KeyFrame.h"

using namespace ORB_SLAM3;
using namespace std;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class PointCloudMapper
{
public:
    PointCloudMapper();

    void InsertKeyFrame(KeyFrame* kf, cv::Mat& imRGB, cv::Mat& imDepth, std::vector<seg::Object> &objs);
    void Cloud_transform(pcl::PointCloud<pcl::PointXYZRGBA>& source, pcl::PointCloud<pcl::PointXYZRGBA>& out);

    PointCloud::Ptr GeneratePointCloud(KeyFrame* kf, cv::Mat& imRGB, cv::Mat& imDepth,std::vector<seg::Object> &objs);
    
    void RequestFinsh();
    bool CheckRequesFinsh();
    bool isFinshd();
    void run();

    queue<KeyFrame*> mqKeyFrame;
    queue<cv::Mat> mqRGB;
    queue <std::vector<seg::Object>> mqObject;
    queue<cv::Mat> mqDepth;
    std::mutex RequestMutex;
    bool requestFinsh = false;
    std::mutex mFinshMutex;
    bool mbFinshd = false;
    
    pcl::VoxelGrid<PointT>::Ptr mpVoxel;
 pcl::StatisticalOutlierRemoval<PointT>::Ptr sor_filter_;
    std::mutex mmLoadKFMutex;
    PointCloud::Ptr mpGlobalMap;
    int mKeyFrameSize;


};

#endif //ORB_SLAM3_POINTCLOUDMAPPER_H
