#include "ObjectTrack.h"
#include "MapObject.h"
#include "Reconstruction.h"
#include "dlib/matrix.h"
#include <Eigen/Core>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>

#include <dlib/optimization/max_cost_assignment.h>
#include <vector>

namespace ORB_SLAM3
{

  ObjectTrack::ObjectTrack(const std::string p_yoloEnginePath,
                           const std::string p_sortEnginePath, cv::Mat K)
      : yoloEnginePath(p_yoloEnginePath), sortEnginePath(p_sortEnginePath)
  {
    yolo = new ImgSeg(yoloEnginePath);
    sort = new YoloSort(sortEnginePath);
    K_ = Converter::toMatrix3f(K);
  }
  ObjectTrack::~ObjectTrack() {}

  bool ObjectTrack::detect()
  {

    ObjectTrack::nNextId++;
    objs.clear();
    boxes.clear();
    yolo->detect(image, objs);
    Objects = objs;


    return true;
  }

  bool ObjectTrack::updateObjectMap(KeyFramePtr keyFrame) {}

  float ObjectTrack::GetCenterDepth(seg::Object obj)
  {
    cv::Mat depth = deepth;
    cv::Rect rect = obj.rect;
    cv::Mat mask = obj.boxMask;

    if (depth.empty() || mask.empty())
      return 0.0f;

    // 1. 优先使用中心区域深度
    cv::Rect centerRegion(rect.x + rect.width / 4, rect.y + rect.height / 4,
                          rect.width / 2, rect.height / 2);

    std::vector<float> valid_depths;
    for (int y = centerRegion.y; y < centerRegion.y + centerRegion.height; ++y)
    {
      for (int x = centerRegion.x; x < centerRegion.x + centerRegion.width; ++x)
      {
        if (mask.at<uchar>(y - rect.y, x - rect.x) > 0)
        {
          float d = depth.at<float>(y, x);
          if (d > 0)
          {
            valid_depths.push_back(d);
          }
        }
      }
    }

    // 2. 如果中心区域无效，再尝试整个区域
    if (valid_depths.empty())
    {
      for (int y = rect.y; y < rect.y + rect.height; ++y)
      {
        for (int x = rect.x; x < rect.x + rect.width; ++x)
        {
          if (mask.at<uchar>(y - rect.y, x - rect.x) > 0)
          {
            float d = depth.at<float>(y, x);
            if (d > 0)
            {
              valid_depths.push_back(d);
            }
          }
        }
      }
    }

    if (valid_depths.empty())
      return 0.0f;

    // 使用加权平均而非简单中值
    float sum = 0.0f;
    float sumWeights = 0.0f;
    for (float d : valid_depths)
    {
      float weight =
          1.0f / (1.0f + std::abs(d - valid_depths[valid_depths.size() / 2]));
      sum += d * weight;
      sumWeights += weight;
    }

    return sum / sumWeights;
  }

  void draw_ellipses_dashed(cv::Mat &img,
                            const std::vector<std::pair<Ellipse, int>> &ellipses_with_labels,
                            int thickness)
  {
    int size = 8;
    int space = 16;

    for (const auto &item : ellipses_with_labels)
    {
      const Ellipse &ell = item.first;
      int label = item.second;

      // 椭球使用 MASK_COLORS
      const auto &color_vec = MASK_COLORS[label % MASK_COLORS.size()];
      cv::Scalar color(color_vec[2], color_vec[1], color_vec[0]); // BGR

      const auto &c = ell.GetCenter();
      const auto &axes = ell.GetAxes();
      double angle = ell.GetAngle();
float ax = axes[0];
float ay = axes[1];
if (!std::isfinite(ax) || !std::isfinite(ay)) continue;
ax = std::max(ax, 1.0f);
ay = std::max(ay, 1.0f);

// 检查中心坐标是否在图像范围
if (c[0] < 0 || c[0] >= img.cols || c[1] < 0 || c[1] >= img.rows)
    continue;

// 限制 thickness
int t = std::min(thickness, 10); // 不超过 10

for (int i = 0; i < 360; i += space)
{
    cv::ellipse(img,
                cv::Point2f(c[0], c[1]),
                cv::Size2f(ax, ay),
                TO_DEG(angle),
                i, i + size,
                color,
                t);
}
    }
  
  }

  std::map<int, cv::Rect> ObjectTrack::object_project(Matrix34d Rt)
  {
    Matrix34d P = K_.cast<double>() * Rt;
    ell_map.clear();
    BBox2 img_bbox(0, 0, image.cols, image.rows);
    std::map<int, cv::Rect> res;

    for (auto it : ObjMap)
    {
      
      int track_id = it.first;
      auto tr = it.second;

      if (tr->GetStatus() == ObjectTrackStatus::INITIALIZED ||
          tr->GetStatus() == ObjectTrackStatus::IN_MAP)
      {
        MapObject *obj = tr->GetMapObject();
        Eigen::Vector3d c = obj->GetEllipsoid().GetCenter();
        double z = Rt.row(2).dot(c.homogeneous());

        auto ell = obj->GetEllipsoid().project(P);
        BBox2 bb = ell.ComputeBbox();
        if (bboxes_intersection(bb, img_bbox) < 0.3 * bbox_area(bb)||z<=0)
          continue;
        // 椭球 + label
        ell_map.emplace_back(ell, obj->label);



        res[track_id] = cv::Rect(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]);
      }
    }
    return res;
  }

  void ObjectTrack::visualize_projected_objects(cv::Mat &image, const std::map<int, cv::Rect> &res)
  {
    cv::Size img_size = image.size();

    for (const auto &it : res)
    {
      int track_id = it.first;
      cv::Rect rect = it.second;

      rect &= cv::Rect(0, 0, img_size.width, img_size.height);
      if (rect.width <= 0 || rect.height <= 0)
        continue;

      // 根据 track_id 获取 label
      int label = ObjMap[track_id]->GetMapObject()->label;

      // 检测框颜色使用 COLORS
      const auto &color_vec = COLORS[label % COLORS.size()];
      cv::Scalar color(color_vec[2], color_vec[1], color_vec[0]); // BGR

      // 画检测框
      cv::rectangle(image, rect, color, 2);

      // 画 ID 文本
      std::string label_text = "ID: " + std::to_string(track_id);
      int baseline = 0;
      cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
      cv::Point text_origin(rect.x, std::max(0, rect.y - 5));
      cv::putText(image, label_text, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
    }

    cv::imshow("Projected Objects", image);
    cv::waitKey(1);
  }

  bool ObjectTrack::updateTrack(Frame frame, KeyFramePtr kf)
  {  
    Eigen::Matrix4d Tcw = frame.GetPose().matrix().cast<double>(); // 世界到相机
    std::vector<seg::Object> track;
    BBox2 img_bbox(0, 0, image.cols, image.rows);

    // 筛选检测
    for (auto &det : objs)
    {
      if (det.prob >= 0.3)
        track.push_back(det);
    }

    // 用 Tcw 投影物体
    Matrix34d Rt = Tcw.block<3, 4>(0, 0);
    Matrix34d P = K_.cast<double>() * Rt;
    std::map<int, cv::Rect> proj = object_project(Rt);
    sort->detect(image, track, proj);

    if (!proj.empty())
    {
      draw_ellipses_dashed(image, ell_map, 3);
      visualize_projected_objects(image, proj);
    }

    std::vector<SegObject *> new_tracks;
    for (auto &det : track)
    {
      int track_id = det.track_id;
      auto it = ObjMap.find(track_id);

      if (it != ObjMap.end() && it->second->label == det.label)
      {
        SegObject *associated_track = it->second;

        if (associated_track->GetStatus() == ObjectTrackStatus::ONLY_2D)
        {
          auto recentobj = ObjectTrack::GetBox(associated_track->obj_recent);
          auto bb = ObjectTrack::GetBox(det);
          if (bboxes_iou(bb, recentobj) > 0.7)
          {
            associated_track->AddDetection(det, Rt, frame.mnId, kf, GetCenterDepth(det));
          }
        }
        else if (associated_track->GetStatus() == ObjectTrackStatus::DYNAMIC)
        {
          // 动态物体不需要 IoU 检查，直接更新
          auto recentobj = ObjectTrack::GetBox(associated_track->obj_recent);
          auto bb = ObjectTrack::GetBox(det);
          // if (bboxes_iou(bb, recentobj) > 0.75)
          associated_track->AddDetection(det, Rt, frame.mnId, kf, GetCenterDepth(det));
        }
        else if (associated_track->GetStatus() == ObjectTrackStatus::INITIALIZED ||
                 associated_track->GetStatus() == ObjectTrackStatus::IN_MAP)
        {
          MapObject *obj = associated_track->GetMapObject();
          Eigen::Vector3d c = obj->GetEllipsoid().GetCenter();
          double z = Rt.row(2).dot(c.homogeneous()); // 正确使用 Tcw
          if (z <= 0 || z > 10)
            continue;

          auto ell = obj->GetEllipsoid().project(P);
          BBox2 bb = ell.ComputeBbox();
          if (bboxes_intersection(bb, img_bbox) < 0.3 * bbox_area(bb))
            continue;

          double score = bboxes_iou(bb, ObjectTrack::GetBox(det));
          if (score > 0.4)
          {
            associated_track->AddDetection(det, Rt, frame.mnId, kf, GetCenterDepth(det));
          }
        }
      }
      else
      {
        // 新建轨迹
        auto tr = SegObject::CreateNewSegObject(
            track_id, det.label, det, Rt, frame.mnId, kf,
            GetCenterDepth(det), K_.cast<double>());
        new_tracks.push_back(tr);
      }
    }

    for (auto tr : new_tracks)
    {
      ObjMap[tr->track_id] = tr;
    }
return true;
  }

  bool ObjectTrack::getDynaMask()
  {
    dynamicObjects.clear();
    dynaMask = cv::Mat::zeros(image.size(), CV_8U);
    for (auto obj : Objects)
    {
      // if (obj.label == 0)
      // {
        dynamicObjects.push_back(new SegObject(obj));
        if (obj.label == 0)
        dynaMask(obj.rect).setTo(255, obj.boxMask);
      // }
    }
    return true;
  }

  bool ObjectTrack::ImgArrive()
  {
    unique_lock<mutex> lock(mMutexNewImg);
    if (newImg)
    {
      newImg = false;
      return true;
    }
    return false;
  }

  bool ObjectTrack::TrackFinsh()
  {
    unique_lock<mutex> lock(mMutexFinsh);
    if (trackFinsh)
    {
      trackFinsh = false;
      return true;
    }
    return false;
  }

  void ObjectTrack::Show()
  {
    // yolo->visualizeObjects(image, objs);
    // cv::imshow("img", image);
    // cv::waitKey(0);
  }

  bool ObjectTrack::update(const cv::Mat img)
  {
    unique_lock<mutex> lock(mMutexNewImg);
    img.copyTo(image);
    newImg = true;
    return true;
  }

  BBox2 ObjectTrack::GetBox(seg::Object obj)
  {
    cv::Rect rect = obj.rect;
    return Eigen::Vector4f(static_cast<float>(rect.x),              // 左上角 x 坐标
                           static_cast<float>(rect.y),              // 左上角 y 坐标
                           static_cast<float>(rect.width + rect.x), // 右下角x坐标
                           static_cast<float>(rect.height + rect.y) // 右下角y坐标
    );
  }

  void ObjectTrack::run()
  {
    while (true)
    {
      if (!ImgArrive())
      {
        usleep(1);
        continue;
      }
      detect();
      unique_lock<mutex> lock(mMutexFinsh);
      trackFinsh = true;
    }
  }

  void ObjectTrack::setAtlas(Atlas *pAtlas) { mpAtlas = pAtlas; }

  void ObjectTrack::track(Map *mcurentMap, Frame frame)
  {

    for (auto &[track_id, tr] : ObjMap)
    {
      if (!tr || tr->label == 0)
        continue;
if (tr->GetStatus() == ObjectTrackStatus::INITIALIZED)
{
    // 判断是否已经超过 50 帧仍未进入地图
    if (frame.mnId - tr->last_obs_frame_id_ > 50)
    {
        // 删除该 track
        // tr->SetIsBad();    // 如果已有标记 bad 的机制
        // ObjMap.erase(track_id);  // 或者从 ObjMap 中移除
        tr->status_ = ObjectTrackStatus::ONLY_2D;
        continue;
    }
}
      if (tr->last_obs_frame_id_ == frame.mnId)
      {

        if ((tr->GetNbObservations() > 3 &&
             tr->GetStatus() == ObjectTrackStatus::ONLY_2D) ||
            (tr->GetNbObservations() % 2 == 0 &&
             tr->GetStatus() == ObjectTrackStatus::INITIALIZED))
        {

          bool status_rec = tr->ReconstructFromCenter(false, mcurentMap);
          if (status_rec)
            tr->OptimizeReconstruction(mcurentMap);
        }

        if (tr->GetNbObservations() >= 25 &&
            tr->GetStatus() == ObjectTrackStatus::INITIALIZED)
        {

          tr->OptimizeReconstruction(mcurentMap);
          auto checked = tr->CheckReprojectionIoU(0.3);

          if (checked)
          {
            tr->InsertInMap(mpAtlas);
            // 可选：送给 local mapper
            // if (local_object_mapper_)
            //   local_object_mapper_->InsertModifiedObject(tr->GetMapObject());
          }
          else
          {
            // 可选：设置为 bad 或回退状态
            // tr->SetIsBad();
          }
        }
        if (
            frame.mnId%40==0 &&
            tr->GetStatus() == ObjectTrackStatus::IN_MAP)
        {
          tr->OptimizeReconstruction(mcurentMap);
        }
      }
    }

  }

  void ObjectTrack::trackDynamicObject(Map *mcurentMap, Frame frame)
  {
    for (auto &[track_id, tr] : ObjMap)
    {
      if (!tr || tr->label != 0)
        continue;

      if (tr->last_obs_frame_id_ == frame.mnId)
      {
        tr->ReconstructTracjetory(mcurentMap,deepth);
      }
    }
  }

} // namespace ORB_SLAM3