#ifndef OBJECTTRACK_H
#define OBJECTTRACK_H
#include <iostream>
#include "ImgSeg.h"
#include "Map.h"
#include <map>
#include "Ellipse.h"
#include "YoloSort.h"
#include <mutex>
#include "Atlas.h"
#include "KeyFrame.h"
#include "MapObject.h"
#include "SegObject.h"
#include "BoxUtils.h"
#include <opencv2/core/core.hpp>
namespace ORB_SLAM3
{

    class Atlas;
    class KeyFrame;
    class MapObject;
    class SegObject;
    class Map;
    class ObjectTrack
    {
    public:
        ObjectTrack(const std::string p_yoloEnginePath, const std::string p_sortEnginePath, cv::Mat K);
        ~ObjectTrack();
        bool detect();
        bool updateObjectMap(KeyFramePtr KeyFrame);
        bool updateTrack(Frame frame, KeyFramePtr kf);
        bool getDynaMask();
        bool ImgArrive();
        void track(Map *mcurentMap, Frame frame);
        void trackDynamicObject(Map *mcurentMap, Frame frame);
        bool TrackFinsh();
        void setAtlas(Atlas *pAtlas);
        void run();
        bool update(const cv::Mat img);
        static BBox2 GetBox(seg::Object obj);
        float GetCenterDepth(seg::Object obj);
        void Show();
        std::map<int, cv::Rect> object_project(Matrix34d Rt);

        void visualize_projected_objects(cv::Mat &image, const std::map<int, cv::Rect> &res);
        std::vector<SegObject *> dynamicObjects;
        std::vector<cv::Rect> dynamicArea;
        cv::Mat dynaMask;
        cv::Mat deepth;
        std::vector<seg::Object> objs;
        std::vector<seg::Object> Objects;
        std::unordered_map<int, SegObject *> ObjMap;
        std::unordered_map<int, SegObject *> dynamicMap;
        Eigen::Matrix3f K_;

    protected:
        Atlas *mpAtlas;

    private:
        long unsigned int nNextId = -1;
        const std::string yoloEnginePath;
        const std::string sortEnginePath;
        ImgSeg *yolo;
        YoloSort *sort;
        bool newImg = false;
        bool trackFinsh = false;
        std::mutex mMutexNewImg;
        std::mutex mMutexFinsh;
        cv::Mat image;
        KeyFramePtr lastKeyFrame;
        std::vector<DetectBox> boxes;
        std::vector<std::pair<Ellipse, int>> ell_map;
    };
}

#endif