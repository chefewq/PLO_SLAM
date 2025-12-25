//
// Created by yuwenlu on 2022/7/2.
//
#include "PointCloudMapper.h"
//#include "ros/ros.h"
//#include "sensor_msgs/PointCloud2.h"
//#include <tf/transform_broadcaster.h>

//#include <pcl_conversions/pcl_conversions.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
//ros::Publisher pclPoint_pub;
//sensor_msgs::PointCloud2 pcl_point;
pcl::PointCloud<pcl::PointXYZRGBA> pcl_filter;

PointCloudMapper::PointCloudMapper()
{
    mpGlobalMap = pcl::make_shared<PointCloud>();
    cout << "voxel set start" << endl;
    mpVoxel = pcl::make_shared<pcl::VoxelGrid<PointT>>();
    sor_filter_ = pcl::make_shared<pcl::StatisticalOutlierRemoval<PointT>>();
    sor_filter_->setMeanK(50);               // 邻域点个数
    sor_filter_->setStddevMulThresh(1.0);    // 标准差倍数阈值
    // sor_filter_->setNegative(true);         // 是否保留离群点（false = 只保留内点）
    mpVoxel->setLeafSize(0.01, 0.01, 0.01);
    cout << "voxel set finish" << endl;

}

void PointCloudMapper::RequestFinsh(){
    unique_lock<mutex> lock(RequestMutex);
    requestFinsh=true;
}

bool PointCloudMapper::CheckRequesFinsh(){
    unique_lock<mutex> lock(RequestMutex);
    return requestFinsh;
}

bool PointCloudMapper::isFinshd()
{
    unique_lock<mutex> lock(mFinshMutex);
    return mbFinshd;
}

void PointCloudMapper::InsertKeyFrame(KeyFrame *kf, cv::Mat &imRGB, cv::Mat &imDepth,std::vector<seg::Object> &objs)
{
    std::lock_guard<std::mutex> lck_loadKF(mmLoadKFMutex);
    mqKeyFrame.push(kf);
    mqRGB.push(imRGB.clone());
    mqDepth.push(imDepth.clone());
    mqObject.push(objs);
}

PointCloud::Ptr PointCloudMapper::GeneratePointCloud(
    KeyFrame *kf, cv::Mat &imRGB, cv::Mat &imDepth, std::vector<seg::Object> &objs)
{
    PointCloud::Ptr pointCloud_temp(new PointCloud);

    for (int v = 10; v < imRGB.rows-10; v += 3) {
        for (int u = 10; u < imRGB.cols-10; u += 3) {
            float d = imDepth.at<float>(v, u);
            if (d < 0 || d > 10)
                continue;

            bool skip_point = false;
            bool colored = false;
            uint8_t r = 0, g = 0, b = 0;

            for (const auto& obj : objs) {
                cv::Rect box_int(
                    std::max(0, int(std::floor(obj.rect.x))),
                    std::max(0, int(std::floor(obj.rect.y))),
                    int(std::ceil(obj.rect.width)),
                    int(std::ceil(obj.rect.height))
                );

                if (box_int.x + box_int.width > imDepth.cols)
                    box_int.width = imDepth.cols - box_int.x;
                if (box_int.y + box_int.height > imDepth.rows)
                    box_int.height = imDepth.rows - box_int.y;

                cv::Rect img_rect(0, 0, imDepth.cols, imDepth.rows);
                cv::Rect box = box_int & img_rect;

                if (box.width <= 0 || box.height <= 0)
                    continue;

                if (box.contains(cv::Point(u, v))) {
                    int mask_x = u - box.x;
                    int mask_y = v - box.y;

                    if (!obj.boxMask.empty() &&
                        mask_x >= 0 && mask_x < obj.boxMask.cols &&
                        mask_y >= 0 && mask_y < obj.boxMask.rows)
                    {
                        if (obj.boxMask.at<uint8_t>(mask_y, mask_x) != 0) {
                            if (obj.label == 0) {
                                skip_point = true;  // 动态物体，跳过
                            } else {
                                int color_idx = (obj.label - 1) % COLORS.size();
                                const auto& color = COLORS[color_idx];
                                b = static_cast<uint8_t>(color[0]);
                                g = static_cast<uint8_t>(color[1]);
                                r = static_cast<uint8_t>(color[2]);
                                colored = true;
                            }
                            break;
                        }
                    }
                }
            }

            if (skip_point)
                continue;

            if (d < 2)
                continue;

            PointT p;
            p.z = d;
            p.x = (u - kf->cx) * p.z / kf->fx;
            p.y = (v - kf->cy) * p.z / kf->fy;

            if (0) {
                p.r = r;
                p.g = g;
                p.b = b;
            } else {
                const cv::Vec3b& c = imRGB.at<cv::Vec3b>(v, u);
                p.b = c[0];
                p.g = c[1];
                p.r = c[2];
            }

            pointCloud_temp->push_back(p);
        }
    }

    Eigen::Isometry3d T = ORB_SLAM3::Converter::toSE3Quat(kf->GetPose());
    PointCloud::Ptr pointCloud(new PointCloud);
    


    pcl::transformPointCloud(*pointCloud_temp, *pointCloud, T.inverse().matrix());
    if( pointCloud->points.size()==0)
    return pointCloud;
    if( pointCloud->points.size()!=0)
    {
pointCloud->width  = pointCloud->points.size();
pointCloud->height = 1;          // 无序点云
pointCloud->is_dense = false;
    }

    return pointCloud;
}

void PointCloudMapper::run()
{
    pcl::visualization::CloudViewer Viewer("Viewer");
    cout << endl << "PointCloudMapping thread start!" << endl;
    int ID = 0;
    //ros::NodeHandle n;
    //pclPoint_pub = n.advertise<sensor_msgs::PointCloud2>("/ORBSLAM3_PointMap/Point_Clouds",1000000);

    while (0)
    {
        {
            std::lock_guard<std::mutex> lck_loadKFSize(mmLoadKFMutex);
            mKeyFrameSize= mqKeyFrame.size();
        }
        if (mKeyFrameSize != 0)
        {
            PointCloud::Ptr pointCloud_new(new PointCloud);
            pointCloud_new = GeneratePointCloud(mqKeyFrame.front(), mqRGB.front(), mqDepth.front(),mqObject.front());
            // 检查 pointCloud_new 是否为空

            mqKeyFrame.pop();
            mqRGB.pop();
            mqDepth.pop();
            mqObject.pop();
            // cout << "==============Insert No. " << ID << "KeyFrame ================" << endl;
             if (pointCloud_new->empty()) {
                // 若为空，跳过当前循环的剩余部分   
                continue;
            }
            ID++;

            *mpGlobalMap += *pointCloud_new;
            PointCloud::Ptr temp(new PointCloud);
            pcl::copyPointCloud(*mpGlobalMap, *temp);
            mpVoxel->setInputCloud(temp);
            mpVoxel->filter(*mpGlobalMap);
            // sor_filter_->setInputCloud(temp);
            // sor_filter_->filter(*mpGlobalMap);
        }
        //std::cout << "点云的数量: " << mpGlobalMap->size() << std::endl;
        //pcl::toROSMsg(*mpGlobalMap, pcl_point);
        //pcl_point.header.frame_id = "map";
        //pclPoint_pub.publish(pcl_point);

        Viewer.showCloud(mpGlobalMap);
        if(CheckRequesFinsh())
        {
            break;
        }
    }
// //         // 存储点云
    string save_path = "./resultPointCloudFile.pcd";
    pcl::io::savePCDFileBinary(save_path, *mpGlobalMap);
    cout << "save pcd files to :  " << save_path << endl;
    unique_lock<mutex> lock(mFinshMutex);
    mbFinshd=true;
}
