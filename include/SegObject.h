#ifndef SEG_OBJECT_H
#define SEG_OBJECT_H
#include "Frame.h"
#include "ImgSeg.h"
#include "KeyFrame.h"
#include "ObjectTrack.h"
#include "Atlas.h"
#include "Kalmanfilter3d.h"
#include "Reconstruction.h"
#include <map>
static const size_t max_frames_history = 50;
namespace ORB_SLAM3 {
class KeyFrame;
class Map;
class Frame;
class Atlas;
class KalmanFilter3D;
class MapObject;
enum class ObjectTrackStatus { ONLY_2D, INITIALIZED, IN_MAP,DYNAMIC, BAD };
class SegObject {
public:
  SegObject();
  SegObject(seg::Object obj);
  SegObject(seg::Object obj, KeyFramePtr kf);
  static SegObject *CreateNewSegObject(int id, int cat, seg::Object obj,
                                       const Matrix34d RT,
                                       unsigned int frame_idx, KeyFramePtr kf,
                                       float d, Eigen::Matrix3d mk);
  ~SegObject();
  void UpdateTrajectory(KeyFramePtr kf, seg::Object obj);
  void UpdateTrack(Matrix34d tcw, seg::Object obj, float pDepth);
  void AddDetection(seg::Object obj, const Matrix34d &RT,
                    unsigned int frame_idx, KeyFrame *kf, float d);
  void AddCenter();
  size_t GetNbObservations();
  bool ReconstructFromCenter(bool use_keyframes, Map *map, bool force_latest_only=false);
  bool ReconstructTracjetory(Map *map, cv::Mat &depth_img);
  void OptimizeReconstruction(Map *map);
  bool CheckReprojectionIoU(double iou_threshold);
  void InsertInMap(Atlas *alt);
  bool updateMovingStatus(float pro_static);
  double GetAngularDifference();
  ObjectTrackStatus GetStatus();
  MapObject *GetMapObject();
  std::unordered_map<MapPointPtr, int> GetAssociatedMapPoints();
  std::tuple<std::vector<BBox2>,
             std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>,
             std::vector<float>, std::vector<float>>
  CopyDetectionsInKeyFrames();

public:
  static long unsigned int nNextId;
  int track_id;
  long unsigned int lastObsFrameId;
  int label = 0;
  float last_obs_score_ = 0.0;
  std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> poses;
  std::vector<seg::Object> track;
  std::vector<float> depth;
  std::vector<float> scores;
  std::unordered_map<KeyFramePtr, BBox2> keyframes_bboxes_;
  std::unordered_map<KeyFramePtr, float> keyframes_scores_;
  std::unordered_map<KeyFramePtr, float> keyframes_depths_;
  std::vector<pair<int, cv::KeyPoint>> KeyPointsInBox;
  std::vector<double> epiErr, epiErrChi2;
  seg::Object obj_recent;
  seg::Object obj_temp;
  std::vector<float> mvKeysDynam;
  std::vector<Eigen::Vector3d> ms;
  std::deque<float> pro_static_history;
  float average, median, stdcov, average_person;
  ObjectTrackStatus status_ = ObjectTrackStatus::ONLY_2D;
  Eigen::Matrix3d K;
  int framesSinceLastSeen;
  bool is_moving = false;
  int last_obs_frame_id_ = -1;
  float unc_ = 0.0;
  double pro_static;
  Eigen::Vector3d max_axes;
public:
private:
  MapObject *map_object_ = nullptr;
  bool  out_of_view_= false; 
  std::unordered_map<MapPointPtr, int> associated_map_points_;
  std::mutex mutex_associated_map_points_;
  std::mutex mutex_map_object_;
};

} // namespace ORB_SLAM3

#endif