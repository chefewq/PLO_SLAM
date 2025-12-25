#include "tracker.h"
#include "nn_matching.h"
#include "linear_assignment.h"
using namespace std;

#define MY_inner_DEBUG
#ifdef MY_inner_DEBUG
#include <string>
#include <iostream>
#endif
using namespace std;
tracker::tracker(/*NearNeighborDisMetric *metric, */
                 float max_cosine_distance, int nn_budget,
                 float max_iou_distance, int max_age, int n_init)
{
    this->metric = new NearNeighborDisMetric(
        NearNeighborDisMetric::METRIC_TYPE::cosine,
        max_cosine_distance, nn_budget);
    this->max_iou_distance = max_iou_distance;
    this->max_age = max_age;
    this->n_init = n_init;
    cout<<max_age<<endl;
    this->kf = new KalmanFilter();
    this->tracks.clear();
    this->_next_idx = 1;
}

void tracker::predict()
{
    for (Track &track : tracks)
    {
        track.predit(kf);
    }
}

void tracker::update(const DETECTIONS &detections)
{
    TRACHER_MATCHD res;
    _match(detections, res);

    vector<MATCH_DATA> &matches = res.matches;
    for (MATCH_DATA &data : matches)
    {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, detections[detection_idx]);
    }
    vector<int> &unmatched_tracks = res.unmatched_tracks;
    for (int &track_idx : unmatched_tracks)
    {
        this->tracks[track_idx].mark_missed();
    }
    vector<int> &unmatched_detections = res.unmatched_detections;
    for (int &detection_idx : unmatched_detections)
    {
        this->_initiate_track(detections[detection_idx]);
    }
    vector<Track>::iterator it;
    for (it = tracks.begin(); it != tracks.end();)
    {
        if ((*it).is_deleted())
            it = tracks.erase(it);
        else
            ++it;
    }
    vector<int> active_targets;
    vector<TRACKER_DATA> tid_features;
    for (Track &track : tracks)
    {
        if (track.is_confirmed() == false)
            continue;
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        FEATURESS t = FEATURESS(0, 256);
        track.features = t;
    }
    this->metric->partial_fit(tid_features, active_targets);
}

void tracker::update(const DETECTIONSV2 &detectionsv2)
{
    const vector<CLSCONF> &clsConf = detectionsv2.first;
    const DETECTIONS &detections = detectionsv2.second;
    TRACHER_MATCHD res;
    _match(detections, res);

    vector<MATCH_DATA> &matches = res.matches;
    for (MATCH_DATA &data : matches)
    {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, detections[detection_idx], clsConf[detection_idx]);
    }
    vector<int> &unmatched_tracks = res.unmatched_tracks;
    for (int &track_idx : unmatched_tracks)
    {
        this->tracks[track_idx].mark_missed();
    }
    vector<int> &unmatched_detections = res.unmatched_detections;

    for (int &detection_idx : unmatched_detections)
    {
        this->_initiate_track(detections[detection_idx], clsConf[detection_idx]);
    }
    vector<Track>::iterator it;
    for (it = tracks.begin(); it != tracks.end();)
    {
        if ((*it).is_deleted())
            it = tracks.erase(it);
        else
            ++it;
    }
    vector<int> active_targets;
    vector<TRACKER_DATA> tid_features;
    for (Track &track : tracks)
    {
        if (track.is_confirmed() == false)
            continue;
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        FEATURESS t = FEATURESS(0, 512);
        track.features = t;
    }
    this->metric->partial_fit(tid_features, active_targets);
}

void tracker::update(const DETECTIONSV2 &detectionsv2, std::map<int, cv::Rect> &obj_proj)
{
    const vector<CLSCONF> &clsConf = detectionsv2.first;
    const DETECTIONS &detections = detectionsv2.second;
    TRACHER_MATCHD res;
    _match(detections, res, obj_proj);

    vector<MATCH_DATA> &matches = res.matches;
    for (MATCH_DATA &data : matches)
    {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, detections[detection_idx], clsConf[detection_idx]);
    }

    //++++++++++++++++ proj_iou ++++++++++++++++++++++++++++++++
    vector<int> &unmatched_tracks = res.unmatched_tracks;
    vector<int> &unmatched_detections = res.unmatched_detections;

    //++++++++++++++++++++++++++++++++++++++++++++++++

    for (int &track_idx : unmatched_tracks)
    {
        this->tracks[track_idx].mark_missed();
    }
    for (int &detection_idx : unmatched_detections)
    {
        this->_initiate_track(detections[detection_idx], clsConf[detection_idx]);
    }
    vector<Track>::iterator it;
    for (it = tracks.begin(); it != tracks.end();)
    {
        if ((*it).is_deleted())
            it = tracks.erase(it);
        else
            ++it;
    }
    vector<int> active_targets;
    vector<TRACKER_DATA> tid_features;
    for (Track &track : tracks)
    {
        if (track.is_confirmed() == false)
            continue;
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        FEATURESS t = FEATURESS(0, 512);
        track.features = t;
    }
    this->metric->partial_fit(tid_features, active_targets);
}

void tracker::_match(const DETECTIONS &detections, TRACHER_MATCHD &res)
{
    vector<int> confirmed_tracks;
    vector<int> unconfirmed_tracks;
    int idx = 0;
    for (Track &t : tracks)
    {
        if (t.is_confirmed())
            confirmed_tracks.push_back(idx);
        else
            unconfirmed_tracks.push_back(idx);
        idx++;
    }

    // 第一阶段：Cascade 匹配
    TRACHER_MATCHD matcha = linear_assignment::getInstance()->matching_cascade(
        this, &tracker::gated_matric,
        this->metric->mating_threshold,
        this->max_age,
        this->tracks,
        detections,
        confirmed_tracks);

    // 第二阶段：IOU 匹配
    vector<int> iou_track_candidates(unconfirmed_tracks);
    for (auto it = matcha.unmatched_tracks.begin(); it != matcha.unmatched_tracks.end();)
    {
        int idx = *it;
        if (tracks[idx].time_since_update == 1)
        { // 刚丢失的confirmed也尝试IOU匹配
            iou_track_candidates.push_back(idx);
            it = matcha.unmatched_tracks.erase(it);
        }
        else
        {
            ++it;
        }
    }

    TRACHER_MATCHD matchb = linear_assignment::getInstance()->min_cost_matching(
        this, &tracker::iou_cost,
        this->max_iou_distance,
        this->tracks,
        detections,
        iou_track_candidates,
        matcha.unmatched_detections);

    // === 合并匹配结果 ===
    res.matches = matcha.matches;
    res.matches.insert(res.matches.end(), matchb.matches.begin(), matchb.matches.end());

    // === 重新计算 unmatched ===
    std::set<int> matched_tracks;
    std::set<int> matched_dets;
    for (auto &m : res.matches)
    {
        matched_tracks.insert(m.first);
        matched_dets.insert(m.second);
    }
    res.unmatched_tracks.clear();
    for (int i = 0; i < tracks.size(); ++i)
    {
        if (!matched_tracks.count(i))
            res.unmatched_tracks.push_back(i);
    }
    res.unmatched_detections.clear();
    for (int j = 0; j < detections.size(); ++j)
    {
        if (!matched_dets.count(j))
            res.unmatched_detections.push_back(j);
    }
}

void tracker::_match(const DETECTIONS &detections,
                     TRACHER_MATCHD &res,
                     std::map<int, cv::Rect> &obj_proj)
{
    // ================= 1. Cascade 匹配 =================
    vector<int> confirmed_tracks;
    vector<int> unconfirmed_tracks;
    for (int i = 0; i < tracks.size(); ++i)
    {
        if (tracks[i].is_confirmed())
            confirmed_tracks.push_back(i);
        else
            unconfirmed_tracks.push_back(i);
    }

    TRACHER_MATCHD matcha = linear_assignment::getInstance()->matching_cascade(
        this, &tracker::gated_matric,
        this->metric->mating_threshold,
        this->max_age,
        this->tracks,
        detections,
        confirmed_tracks);

    res.matches = matcha.matches;

    // ================= 2. 投影匹配 (提前) =================
    std::set<int> matched_tracks;
    std::set<int> matched_dets;
    for (auto &m : res.matches)
    {
        matched_tracks.insert(m.first);
        matched_dets.insert(m.second);
    }

    vector<int> unmatched_tracks;
    for (int i = 0; i < tracks.size(); ++i)
    {
        if (!matched_tracks.count(i))
            unmatched_tracks.push_back(i);
    }

    vector<int> unmatched_detections;
    for (int j = 0; j < detections.size(); ++j)
    {
        if (!matched_dets.count(j))
            unmatched_detections.push_back(j);
    }

    for (int i = 0; i < unmatched_tracks.size();)
    {
        int track_idx = unmatched_tracks[i];
        auto &track = this->tracks[track_idx];
        
        auto it = obj_proj.find(track.track_id);
        if (it == obj_proj.end() || !track.is_confirmed())
        {
            ++i;
            continue;
        }

        const cv::Rect &proj_box = it->second;
        bool matched = false;

        for (int j = 0; j < unmatched_detections.size(); j++)
        {
            int det_idx = unmatched_detections[j];
            const DETECTION_ROW &det = detections[det_idx];
            if (track.cur_det.label != det.label )
                continue;

            cv::Rect det_box(
                static_cast<int>(det.tlwh(0)),
                static_cast<int>(det.tlwh(1)),
                static_cast<int>(det.tlwh(2)),
                static_cast<int>(det.tlwh(3)));

            float inter_area = (proj_box & det_box).area();
            float union_area = (proj_box | det_box).area();
            float iou = (union_area > 0) ? (inter_area / union_area) : 0.f;

            cv::Point2f proj_center(proj_box.x + proj_box.width / 2.0f,
                                    proj_box.y + proj_box.height / 2.0f);
            cv::Point2f det_center(det_box.x + det_box.width / 2.0f,
                                   det_box.y + det_box.height / 2.0f);

            float dx = proj_center.x - det_center.x;
            float dy = proj_center.y - det_center.y;
            float distance = std::sqrt(dx * dx + dy * dy);
            float norm_dist = distance / std::sqrt(det_box.width * det_box.width + det_box.height * det_box.height);
            if (iou >= 0.2f || norm_dist <= 0.7f)
            {
                res.matches.emplace_back(track_idx, det_idx);
                matched_tracks.insert(track_idx);
                matched_dets.insert(det_idx);
                unmatched_tracks.erase(unmatched_tracks.begin() + i);
                unmatched_detections.erase(unmatched_detections.begin() + j);

                matched = true;
                break;
            }
        }

        if (!matched)
            ++i;
    }

    // ================= 3. IOU 匹配 =================
    vector<int> iou_track_candidates;
    for (int idx : unmatched_tracks)
    {
        if (!tracks[idx].is_confirmed() || tracks[idx].time_since_update == 1)
            iou_track_candidates.push_back(idx);
    }

    TRACHER_MATCHD matchb = linear_assignment::getInstance()->min_cost_matching(
        this, &tracker::iou_cost,
        this->max_iou_distance,
        this->tracks,
        detections,
        iou_track_candidates,
        unmatched_detections);

    res.matches.insert(res.matches.end(), matchb.matches.begin(), matchb.matches.end());

    // ================= 4. 重新计算 unmatched =================
    matched_tracks.clear();
    matched_dets.clear();
    for (auto &m : res.matches)
    {
        matched_tracks.insert(m.first);
        matched_dets.insert(m.second);
    }

    res.unmatched_tracks.clear();
    for (int i = 0; i < tracks.size(); ++i)
    {
        if (!matched_tracks.count(i))
            res.unmatched_tracks.push_back(i);
    }

    res.unmatched_detections.clear();
    for (int j = 0; j < detections.size(); ++j)
    {
        if (!matched_dets.count(j))
            res.unmatched_detections.push_back(j);
    }
}

void tracker::_initiate_track(const DETECTION_ROW &detection)
{
    KAL_DATA data = kf->initiate(detection.to_xyah());
    KAL_MEAN mean = data.first;
    KAL_COVA covariance = data.second;

    this->tracks.push_back(Track(mean, covariance, this->_next_idx, this->n_init,
                                 this->max_age, detection.feature, detection));
}
void tracker::_initiate_track(const DETECTION_ROW &detection, CLSCONF clsConf)
{
    KAL_DATA data = kf->initiate(detection.to_xyah());
    KAL_MEAN mean = data.first;
    KAL_COVA covariance = data.second;

    this->tracks.push_back(Track(mean, covariance, this->_next_idx, this->n_init,
                                 this->max_age, detection.feature, clsConf.cls, clsConf.conf, detection));
}

DYNAMICM tracker::gated_matric(
    std::vector<Track> &tracks,
    const DETECTIONS &dets,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices)
{
    int rows = track_indices.size();
    int cols = detection_indices.size();
    if (rows == 0 || cols == 0)
        return Eigen::MatrixXf::Zero(rows, cols);

    // 提取 ReID 特征
    FEATURESS features(cols, 512);
    for (int j = 0; j < cols; ++j)
    {
        features.row(j) = dets[detection_indices[j]].feature;
    }

    // 提取 Track IDs 用于匹配
    std::vector<int> targets;
    for (int i = 0; i < rows; ++i)
    {
        targets.push_back(tracks[track_indices[i]].track_id);
    }

    // 计算 ReID cosine 距离：rows x cols
    DYNAMICM cost_matrix = this->metric->distance(features, targets);

    // 构造最终代价矩阵
    DYNAMICM final_cost(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        int track_idx = track_indices[i];
        for (int j = 0; j < cols; ++j)
        {
            int det_idx = detection_indices[j];
            if (tracks[track_idx].cur_det.label == 0 && dets[det_idx].label == 0 )
            {
                final_cost(i, j) = cost_matrix(i, j); // 使用 ReID 距离
            }
            else
            {
                final_cost(i, j) = 100000.f; // 强制不匹配
            }
        }
    }

    // gating 操作（遮挡过滤）
    // DYNAMICM res = linear_assignment::getInstance()->gate_cost_matrix(
    //     this->kf, final_cost, tracks, dets, track_indices, detection_indices);
    return final_cost;
}

DYNAMICM
tracker::iou_cost(
    std::vector<Track> &tracks,
    const DETECTIONS &dets,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices)
{
    int rows = track_indices.size();
    int cols = detection_indices.size();
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        int track_idx = track_indices[i];

        if (tracks[track_idx].time_since_update > 1)
        {
            cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
            continue;
        }

        DETECTBOX bbox = tracks[track_idx].to_tlwh();
        auto track = tracks[track_idx].cur_det;

        int csize = cols;
        DETECTBOXSS candidates(csize, 4);
        for (int k = 0; k < csize; k++)
        {
            candidates.row(k) = dets[detection_indices[k]].tlwh;
        }

        Eigen::VectorXf ious = iou(bbox, candidates); // iou 返回的是 vector，大小为 cols
        for (int j = 0; j < cols; j++)
        {
            int det_idx = detection_indices[j];
            // 标签不一致，代价为无穷大
            if (dets[det_idx].label != track.label)
            {
                cost_matrix(i, j) = INFTY_COST;
            }
            else
            {
                cost_matrix(i, j) = 1.0f - ious[j];
            }
        }
    }

    return cost_matrix;
}

Eigen::VectorXf
tracker::iou(DETECTBOX &bbox, DETECTBOXSS &candidates)
{
    float bbox_tl_1 = bbox[0];
    float bbox_tl_2 = bbox[1];
    float bbox_br_1 = bbox[0] + bbox[2];
    float bbox_br_2 = bbox[1] + bbox[3];
    float area_bbox = bbox[2] * bbox[3];

    Eigen::Matrix<float, -1, 2> candidates_tl;
    Eigen::Matrix<float, -1, 2> candidates_br;
    candidates_tl = candidates.leftCols(2);
    candidates_br = candidates.rightCols(2) + candidates_tl;

    int size = int(candidates.rows());
    //    Eigen::VectorXf area_intersection(size);
    //    Eigen::VectorXf area_candidates(size);
    Eigen::VectorXf res(size);
    for (int i = 0; i < size; i++)
    {
        float tl_1 = std::max(bbox_tl_1, candidates_tl(i, 0));
        float tl_2 = std::max(bbox_tl_2, candidates_tl(i, 1));
        float br_1 = std::min(bbox_br_1, candidates_br(i, 0));
        float br_2 = std::min(bbox_br_2, candidates_br(i, 1));

        float w = br_1 - tl_1;
        w = (w < 0 ? 0 : w);
        float h = br_2 - tl_2;
        h = (h < 0 ? 0 : h);
        float area_intersection = w * h;
        float area_candidates = candidates(i, 2) * candidates(i, 3);
        res[i] = area_intersection / (area_bbox + area_candidates - area_intersection);
    }
    return res;
}
