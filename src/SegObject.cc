#include "SegObject.h"
#include "OptimizerObject.h"
namespace ORB_SLAM3 {
SegObject::SegObject():map_object_(nullptr) {}
SegObject::SegObject(seg::Object obj)
{
  label = obj.label;
  obj_recent = obj;
}

SegObject *SegObject::CreateNewSegObject(int id,int cat, seg::Object obj,
                                         const Matrix34d RT,
                                         unsigned int frame_idx, KeyFramePtr kf,
                                         float d, Eigen::Matrix3d mk) {
  SegObject *sob = new SegObject();
  sob->track_id = id;
  sob->label = cat;
  sob->K = mk;
  sob->poses.clear();
  sob->depth.clear();
  sob->track.clear();
  sob->scores.clear();
  sob->poses.push_back(RT);
  sob->depth.push_back(d);
  sob->track.push_back(obj);
  sob->scores.push_back(obj.prob);
  if (kf) {
    sob->keyframes_bboxes_[kf] = ObjectTrack::GetBox(obj);
    sob->keyframes_scores_[kf] = obj.prob;
    sob->keyframes_depths_[kf] = d;
  }
  sob->last_obs_frame_id_ = frame_idx;
  sob->status_ = ObjectTrackStatus::ONLY_2D;
  if(sob->label==0)
  sob->status_ = ObjectTrackStatus::DYNAMIC;
  sob->last_obs_score_ = obj.prob;
  sob->obj_recent = obj;
  sob->unc_ = 0.5;
  return sob;
}


void SegObject::AddDetection(seg::Object obj, const Matrix34d &RT,
                             unsigned int frame_idx, KeyFrame *kf, float d) {
  track.push_back(obj);
  poses.push_back(RT);
  depth.push_back(d);
  scores.push_back(obj.prob);
  last_obs_frame_id_ = frame_idx;
  last_obs_score_ = obj.prob;
  obj_recent = obj;
  if (kf) {
    keyframes_bboxes_[kf] = ObjectTrack::GetBox(obj);
    keyframes_scores_[kf] = obj.prob;
    keyframes_depths_[kf] = d;
    for (size_t i = 0; i < kf->mvKeys.size(); ++i) {
      MapPointPtr p = kf->GetMapPoint(i);
      if (p) {
        auto kp = kf->mvKeys[i];
        if (is_inside_bbox(kp.pt.x, kp.pt.y, ObjectTrack::GetBox(obj))) {
          if (associated_map_points_.find(p) == associated_map_points_.end())
            associated_map_points_[p] = 1;
          else
            associated_map_points_[p]++;
        }
      }
    }
  }

  if (status_ == ObjectTrackStatus::IN_MAP) {
    // kalman uncertainty update
    float k = unc_ / (unc_ + std::exp(-obj.prob));
    unc_ = unc_ * (1.0 - k);
  }
}

SegObject::~SegObject() {}

MapObject* SegObject::GetMapObject() { 
  return map_object_; 
}

std::unordered_map<MapPointPtr, int> SegObject::GetAssociatedMapPoints() {
  std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
  return associated_map_points_;
}

void SegObject::UpdateTrajectory(KeyFramePtr kf, seg::Object obj) {
  // trajectory.insert({kf,obj});
}

void SegObject::UpdateTrack(Matrix34d tcw, seg::Object obj, float pdepth) {
  poses.push_back(tcw);
  track.push_back(obj);
  depth.push_back(pdepth);
}

double SegObject::GetAngularDifference() {
  Eigen::Vector3d c0 =
      bbox_center(ObjectTrack::GetBox(track.front())).homogeneous();
  Eigen::Matrix3d K_inv = K.inverse();
  Eigen::Vector3d v0 = K_inv * c0;
  v0.normalize();
  v0 = poses.front().block<3, 3>(0, 0).transpose() * v0;
  Eigen::Vector3d c1 =
      bbox_center(ObjectTrack::GetBox(track.back())).homogeneous();
  Eigen::Vector3d v1 = K_inv * c1;
  v1.normalize();
  v1 = poses.back().block<3, 3>(0, 0).transpose() * v1;
  return std::atan2(v0.cross(v1).norm(), v0.dot(v1));
}

bool SegObject::ReconstructFromCenter(bool use_keyframes, Map *pmap, bool force_latest_only ) 
{
    std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts;
    std::vector<BBox2> bboxes;
    std::vector<float> z_;

    if (use_keyframes) {
        auto [bbs, poses, _, d] = this->CopyDetectionsInKeyFrames();
        bboxes = std::move(bbs);
        Rts = std::move(poses);
        z_ = std::move(d);
    } else {
        Rts = poses;
        for (auto obj : track)
            bboxes.push_back(ObjectTrack::GetBox(obj));
        z_ = depth;
    }

    // ===== 新增：只保留最新观测并清空历史 =====
    if (force_latest_only && !bboxes.empty() && !Rts.empty() && !z_.empty()) {
        bboxes = {bboxes.back()};
        Rts = {Rts.back()};
        z_ = {z_.back()};

        // 清空原有历史观测
        poses.clear();
        track.clear();
        depth.clear();
    }
    // =============================

    auto [status, ellipsoid] = ReconstructEllipsoidFromCenters(bboxes, Rts, z_, K);
    if (!status)
        return false;

    if (map_object_ != nullptr) {
        std::lock_guard<std::mutex> lock(mutex_map_object_);
        map_object_->SetEllipsoid(ellipsoid);
    } else {
        map_object_ = new MapObject(track_id, label, ellipsoid, pmap);
    }

    if (status_ == ObjectTrackStatus::ONLY_2D)
        status_ = ObjectTrackStatus::INITIALIZED;
    if(status_ == ObjectTrackStatus::IN_MAP)
    status_ = ObjectTrackStatus::INITIALIZED;
    return true;
}


Eigen::Vector3d ComputeCentroidFromMask(
    const cv::Mat &depth_img, const cv::Mat &mask, const Eigen::Matrix3d &K)
{
    double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    std::vector<Eigen::Vector3d> points;
    points.reserve(1000);

    for (int v = 0; v < mask.rows; ++v) {
        for (int u = 0; u < mask.cols; ++u) {
            if (mask.at<uchar>(v, u) == 0) continue;
            float d = depth_img.at<float>(v, u);
            if (d <= 0) continue;

            Eigen::Vector3d p;
            p[0] = (u - cx) * d / fx;
            p[1] = (v - cy) * d / fy;
            p[2] = d;
            points.push_back(p);
        }
    }

    if (points.empty()) return Eigen::Vector3d::Zero();

    Eigen::Vector3d centroid(0, 0, 0);
    for (auto &p : points) centroid += p;
    centroid /= points.size();

    return centroid;
}


bool SegObject::ReconstructTracjetory(Map *pmap, cv::Mat &depth_img)
{
    if (poses.empty() || depth.empty()) return false;

    Matrix34d RT = poses.back();
    double dep = depth.back();
    if (dep <= 0) return false;

    const cv::Mat &mask = obj_recent.boxMask;
    if (mask.empty()) return false;

    cv::Rect_<float> roi = obj_recent.rect;

    // --- mask 质心（像素坐标） ---
    cv::Moments m = cv::moments(mask, true);
    if (m.m00 < 1e-5) return false;
    double u_c = roi.x + m.m10 / m.m00;
    double v_c = roi.y + m.m01 / m.m00;

    // --- mask 内深度中值 ---
    std::vector<float> depths;
    for (int y = 0; y < mask.rows; ++y)
    {
        for (int x = 0; x < mask.cols; ++x)
        {
            if (mask.at<uchar>(y, x) == 0) continue;
            float d = depth_img.at<float>(roi.y + y, roi.x + x);
            if (d > 0) depths.push_back(d);
        }
    }
    if (depths.empty()) return false;
    std::nth_element(depths.begin(), depths.begin() + depths.size()/2, depths.end());
    double dep_median = depths[depths.size()/2];

    // --- 相机坐标 ---
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    Eigen::Vector3d center_cam_obs((u_c - cx) * dep_median / fx,
                                   (v_c - cy) * dep_median / fy,
                                   dep_median);

    // --- 相机到世界坐标 ---
    Eigen::Matrix3d R = RT.block<3,3>(0,0);
    Eigen::Vector3d t = RT.block<3,1>(0,3);
    Eigen::Matrix3d R_inv = R.transpose();
    Eigen::Vector3d t_inv = -R_inv * t;
    Eigen::Vector3d center_world_obs = R_inv * center_cam_obs + t_inv;

    // --- 遮挡检测 ---
    double mask_ratio = cv::countNonZero(mask) / double(mask.rows * mask.cols);
    bool obs_valid = mask_ratio > 0.1;

    // --- 初始化 MapObject ---
    if (!map_object_)
    {
        double rx = 0.5 * mask.cols * dep_median / fx;
        double ry = 0.5 * mask.rows * dep_median / fy;
        double rz = 0.5 * (rx + ry);
        Eigen::Vector3d axes(rx, ry, rz);
        Eigen::Matrix3d axes_rotation = R_inv;

        Ellipsoid ellipsoid(axes, axes_rotation, center_world_obs);
        map_object_ = new MapObject(track_id, label, ellipsoid, pmap);
        map_object_->trajectory_segments.emplace_back();
        map_object_->trajectory_segments.back().push_back(center_world_obs);
        pmap->AddMapObjects(map_object_);

        map_object_->kf_x = Eigen::VectorXd::Zero(6);
        map_object_->kf_x.head<3>() = center_world_obs;
        map_object_->kf_P = Eigen::MatrixXd::Identity(6,6) * 0.01;
        return true;
    }

    // --- 卡尔曼预测 ---
    double dt = 1.0;
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6,6);
    F(0,3) = dt; F(1,4) = dt; F(2,5) = dt;
    Eigen::VectorXd x_pred = F * map_object_->kf_x;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6,6) * 0.03;
    Eigen::MatrixXd P_pred = F * map_object_->kf_P * F.transpose() + Q;

    // --- 卡尔曼观测更新 ---
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3,6);
    H(0,0)=1; H(1,1)=1; H(2,2)=1;
    Eigen::MatrixXd Rk = Eigen::MatrixXd::Identity(3,3) * 0.001;
    if (!obs_valid) Rk *= 10.0;

    Eigen::Vector3d z = center_world_obs;
    Eigen::VectorXd y = z - H * x_pred;
    Eigen::MatrixXd S = H * P_pred * H.transpose() + Rk;
    Eigen::MatrixXd K = P_pred * H.transpose() * S.inverse();
    map_object_->kf_x = x_pred + K * y;
    map_object_->kf_P = (Eigen::MatrixXd::Identity(6,6) - K*H) * P_pred;

    Eigen::Vector3d center_new = map_object_->kf_x.head<3>();

    // --- 在可能分段前先计算最近窗口平均 y（避免空段导致 avg = 0） ---
    double avg_y = 0.0;
    int n = 0;
    const int window = 5;               // 最近帧窗口，可调
    const double y_outlier_thresh = 0.1; // 超出该差值视为异常（米），可调

    // 如果当前没有 segment（理论上不会），使用椭球中心
    if (map_object_->trajectory_segments.empty())
    {
        avg_y = map_object_->GetEllipsoid().GetCenter()(1);
    }
    else
    {
        auto &cur_seg_before = map_object_->trajectory_segments.back(); // 注意：在这里还未做 emplace
        if (!cur_seg_before.empty())
        {
            int start = std::max(0, (int)cur_seg_before.size() - window);
            for (int i = start; i < (int)cur_seg_before.size(); ++i)
            {
                avg_y += cur_seg_before[i](1);
                ++n;
            }
        }
        if (n > 0)
            avg_y /= n;
        else
            avg_y = map_object_->GetEllipsoid().GetCenter()(1); // 兜底
    }

    // --- 如果 y 是异常值，则用 avg_y 修正（不丢点） ---
    if (std::abs(center_new(1) - avg_y) > y_outlier_thresh)
    {
        center_new(1) = avg_y;
        // 同步修正卡尔曼状态中的 y，防止下一帧被 KF 拉回异常值
        map_object_->kf_x(1) = center_new(1);
    }

    // --- 超跳检测，分段（使用修正后的 center_new 与当前段最后点比较）---
    Eigen::Vector3d last_pos = map_object_->trajectory_segments.back().empty()
                                   ? map_object_->GetEllipsoid().GetCenter()
                                   : map_object_->trajectory_segments.back().back();
    double jump_thresh = 0.3;
    if ((center_new - last_pos).norm() > jump_thresh)
    {
        map_object_->trajectory_segments.emplace_back();
    }

    // --- 尺寸平滑 ---
    // --- 尺寸平滑 ---
    double rx = 0.5 * mask.cols * dep_median / fx;
    double ry = 0.5 * mask.rows * dep_median / fy;
    double rz = 0.5 * (rx + ry);
    Eigen::Vector3d axes_obs(rx, ry, rz);

    // 历史最大尺寸（如果你在 MapObject 里没有记录，可以在类里加一个成员 max_axes）
    if (map_object_->max_axes.norm() < 1e-6) {
        map_object_->max_axes = axes_obs; // 初始化最大值
    } else {
        for (int i = 0; i < 3; ++i) {
            // 限制下限：如果小于0.7*max，就用0.7*max
            double min_allowed = 0.7 * map_object_->max_axes(i);
            if (axes_obs(i) < min_allowed) {
                axes_obs(i) = min_allowed;
            }
            // 更新最大值
            if( map_object_->trajectory_segments.back().size()>5)
            map_object_->max_axes(i) = std::max(map_object_->max_axes(i), axes_obs(i));
        }
    }

    Eigen::Vector3d prev_axes = map_object_->GetEllipsoid().GetAxes();
    double beta = obs_valid ? 0.1 : 0.0;
    Eigen::Vector3d new_axes = (1 - beta) * prev_axes + beta * axes_obs;

    Eigen::Matrix3d axes_rotation = Eigen::Matrix3d::Identity();
    Ellipsoid ellipsoid(new_axes, axes_rotation, center_new);
    map_object_->SetEllipsoid(ellipsoid);


    // --- 更新轨迹（始终插入修正后的点） ---
    if (map_object_->trajectory_segments.empty())
        map_object_->trajectory_segments.emplace_back();
    map_object_->trajectory_segments.back().push_back(center_new);

    return true;
}

// bool SegObject::ReconstructTracjetory(Map *pmap)
// {
//     if (poses.empty() || depth.empty()) return false;

//     Matrix34d RT = poses.back();
//     double dep = depth.back();
//     if (dep <= 0) return false;

//     // 取检测框中心
//     BBox2 bb = ObjectTrack::GetBox(obj_recent);
//     Eigen::Vector2d cen = bbox_center(bb);

//     double fx = K(0, 0);
//     double fy = K(1, 1);
//     double cx = K(0, 2);
//     double cy = K(1, 2);

//     // ================== 可见性检测 ==================
//     int img_w = 640;
//     int img_h = 480;
//     bool fully_visible = (bb[0] >= 0 && bb[1] >= 0 &&
//                           bb[2] < img_w && bb[3] < img_h);

//     // 物体丢失 → 冻结轨迹（追加最后位置）
//     if (!fully_visible && map_object_ != nullptr)
//     {
//         if (!map_object_->trajectory_segments.empty())
//         {
//             map_object_->trajectory_segments.back().push_back(map_object_->GetEllipsoid().GetCenter());
//         }
//         out_of_view_ = true;
//         return true;
//     }

//     // 物体重新出现 → 新开一段轨迹
//     if (map_object_ && out_of_view_)
//     {
//         map_object_->trajectory_segments.emplace_back();
//         out_of_view_ = false;
//     }

//     // 反投影到相机坐标
//     Eigen::Vector3d center_cam;
//     center_cam[0] = (cen[0] - cx) * dep / fx;
//     center_cam[1] = (cen[1] - cy) * dep / fy;
//     center_cam[2] = dep;

//     // Tcw → Twc
//     Eigen::Matrix3d R = RT.block<3, 3>(0, 0);
//     Eigen::Vector3d t = RT.block<3, 1>(0, 3);
//     Eigen::Matrix3d R_inv = R.transpose();
//     Eigen::Vector3d t_inv = -R_inv * t;

//     // 相机坐标 → 世界坐标
//     Eigen::Vector3d center = R_inv * center_cam + t_inv;

//     // 计算物体 3D 尺寸
//     float rx = 0.5f * (bb[2] - bb[0]) * dep / fx;
//     float ry = 0.5f * (bb[3] - bb[1]) * dep / fy;
//     float rz = 0.5f * (rx + ry);

//     // 使用相机坐标系的旋转
//     Eigen::Matrix3d axes_rotation = R_inv;

//     if (map_object_ != nullptr)
//     {
//         Ellipsoid ellipsoid(Eigen::Vector3d(rx, ry, rz), axes_rotation, center);
//         map_object_->SetEllipsoid(ellipsoid);

//         // 添加到当前轨迹段
//         if (!map_object_->trajectory_segments.empty())
//             map_object_->trajectory_segments.back().push_back(center);
//     }
//     else
//     {
//         Ellipsoid ellipsoid(Eigen::Vector3d(rx, ry, rz), axes_rotation, center);
//         map_object_ = new MapObject(track_id, label, ellipsoid, pmap);

//         // 初始化第一段轨迹
//         map_object_->trajectory_segments.emplace_back();
//         map_object_->trajectory_segments.back().push_back(center);

//         pmap->AddMapObjects(map_object_);
//         out_of_view_ = false;
//     }

//     return true;
// }

void SegObject::OptimizeReconstruction(Map *map)
{
if (!map_object_) {
        std::cerr << "Impossible to optimize ellipsoid. It first requires a initial reconstruction." << std::endl;
        return ;
    }
    const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();

    // std::cout << "===============================> Start ellipsoid optimization " << id_ << std::endl;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolver_6_1;
    BlockSolver_6_1::LinearSolverType *linear_solver = new g2o::LinearSolverDense<BlockSolver_6_1::PoseMatrixType>();


    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        new BlockSolver_6_1(linear_solver)
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);


    VertexEllipsoidNoRot* vertex = new VertexEllipsoidNoRot();
    // VertexEllipsoid* vertex = new VertexEllipsoid();
    vertex->setId(0);
    Eigen::Matrix<double, 6, 1> e;
    e << ellipsoid.GetAxes(), ellipsoid.GetCenter();
    // EllipsoidQuat ellipsoid_quat = EllipsoidQuat::FromEllipsoid(*ellipsoid_);
    // std::cout << "before optim: " << e.transpose() << "\n";;
    // vertex->setEstimate(ellipsoid_quat);
    vertex->setEstimate(e);
    optimizer.addVertex(vertex);

double lambda = 0.025; 
int current_frame_id = last_obs_frame_id_; 
auto it_bb = track.begin();
auto it_Rt = poses.begin();
auto it_s = scores.begin();
int frame_idx = current_frame_id - track.size() + 1; // 估算对应frame index
for (size_t i = 0; i < track.size() && it_bb != track.end() && it_Rt != poses.end() && it_s != scores.end(); ++i, ++it_bb, ++it_Rt, ++it_s, ++frame_idx) {
    Eigen::Matrix<double, 3, 4> P = K * (*it_Rt);
    EdgeEllipsoidProjection *edge = new EdgeEllipsoidProjection(P, Ellipse::FromBbox(ObjectTrack::GetBox(*it_bb)), ellipsoid.GetOrientation());
    edge->setId(i);
    edge->setVertex(0, vertex);

    // ===== 动态权重计算 =====
    double d_j = std::abs(frame_idx); // 帧间距离
    double sigma_j = *it_s; // 置信度
    double rho_j =  sigma_j/(1.0 + lambda * d_j) ;

    Eigen::Matrix<double, 1, 1> information_matrix = Eigen::Matrix<double, 1, 1>::Identity();
    information_matrix *= rho_j;
    edge->setInformation(information_matrix);
    optimizer.addEdge(edge);
}

    optimizer.initializeOptimization();
    optimizer.optimize(8);
    Eigen::Matrix<double, 6, 1> ellipsoid_est = vertex->estimate();
    // EllipsoidQuat ellipsoid_quat_est = vertex->estimate();

    // std::cout << "after optim: " << vertex->estimate().transpose() << "\n\n";

    Ellipsoid new_ellipsoid(ellipsoid_est.head(3), Eigen::Matrix3d::Identity(), ellipsoid_est.tail(3));
    map_object_->SetEllipsoid(new_ellipsoid);

}


bool SegObject::CheckReprojectionIoU(double iou_threshold){
      if (status_ != ObjectTrackStatus::INITIALIZED)
        return false;
  const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
    bool valid = true;
    auto it_bb = track.begin();
    auto it_Rt = poses.begin();
    for (; it_bb != track.end() && it_Rt != poses.end(); ++it_bb, ++it_Rt) {
        Eigen::Matrix<double, 3, 4> P = K * (*it_Rt);
        Ellipse ell = ellipsoid.project(P);
        BBox2 proj_bb = ell.ComputeBbox();
        double iou = bboxes_iou(ObjectTrack::GetBox(*it_bb), proj_bb);
        if (iou < iou_threshold)
        {
            valid  = false;
            break;
        }
    }
    return valid;
}

void SegObject::InsertInMap(Atlas *alt)
{
   alt->AddMapObject(map_object_);
   status_ = ObjectTrackStatus::IN_MAP;
}


std::tuple<std::vector<BBox2>, std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>, std::vector<float>,std::vector<float>>
SegObject::CopyDetectionsInKeyFrames() {
  std::vector<BBox2> bboxes;
  std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> poses;
  std::vector<float> scores;
  std::vector<float> dep;
  if (true) {
    bboxes.reserve(keyframes_bboxes_.size());
    poses.reserve(keyframes_bboxes_.size());
    scores.reserve(keyframes_bboxes_.size());
    dep.reserve(keyframes_bboxes_.size());
    int i = 0;
    std::vector<KeyFramePtr> to_erase;
    for (auto &it : keyframes_bboxes_) {
      if (it.first->isBad()) {
        to_erase.push_back(it.first);
        continue;
      }
      bboxes.push_back(it.second);
      dep.push_back(keyframes_depths_[it.first]);
      poses.push_back(it.first->GetPose().matrix3x4().cast<double>());
      scores.push_back(keyframes_scores_[it.first]);
      ++i;
    }
    for (auto *pt : to_erase) {
      keyframes_bboxes_.erase(pt);
      keyframes_scores_.erase(pt);
      keyframes_depths_.erase(pt);
    }
  }
  return make_tuple(bboxes, poses, scores,dep);
};
size_t SegObject::GetNbObservations() { return track.size(); }
ObjectTrackStatus SegObject::GetStatus() { return status_; }

bool SegObject::updateMovingStatus(float pro_static) {
             pro_static_history.push_back(pro_static);

        // if (pro_static_history.size() > 1) {
        //     pro_static_history.pop_front();
        // }
        // float sum = 0.0f;
        // for (float val : pro_static_history) {
        //     sum += val;
        // }
        // float average_pro_static = sum / pro_static_history.size();
        if(label == 0)
        is_moving = pro_static <0.9;
        else 
         is_moving = pro_static <0.75;
        // cout<<average_pro_static <<endl;
        // 判断是否为动态物体，假设某种条件
        return is_moving;
    }
} // namespace ORB_SLAM3