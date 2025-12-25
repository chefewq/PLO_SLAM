#include "deepsort.h"

DeepSort::DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID, ILogger* gLogger) {
    this->gpuID = gpuID;
    this->enginePath = modelPath;
    this->batchSize = batchSize;
    this->featureDim = featureDim;
    this->imgShape = cv::Size(64, 128);
    this->maxBudget = 100;
    this->maxCosineDist = 0.5;
    this->gLogger = gLogger;
    init();
}

void DeepSort::init() {
    objTracker = new tracker(maxCosineDist, maxBudget);
    featureExtractor = new FeatureTensor(batchSize, imgShape, featureDim, gpuID, gLogger);
    int ret = enginePath.find(".onnx");
    if (ret != -1)
        featureExtractor->loadOnnx(enginePath);
    else
        featureExtractor->loadEngine(enginePath);
}

DeepSort::~DeepSort() {
    delete objTracker;
}

void DeepSort::sort(cv::Mat& frame, vector<DetectBox>& dets) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    vector<CLSCONF> clsConf;
    
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classID, i.confidence));
    }
    result.clear();
    results.clear();
    if (detections.size() > 0) {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2);
    }
    // postprocess DETECTION -> Mat
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2)+i(0), i(3)+i(1), 1.);
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    for (int i = 0; i < results.size(); ++i) {
        CLSCONF c = results[i].first;
        dets[i].classID = c.cls;
        dets[i].confidence = c.conf;
    }
}

float centerDistance(const cv::Rect2f& a, const cv::Rect2f& b) {
    cv::Point2f ca = (a.tl() + a.br()) * 0.5f;
    cv::Point2f cb = (b.tl() + b.br()) * 0.5f;
    return cv::norm(ca - cb);
}

void DeepSort::sort(cv::Mat& frame, std::vector<seg::Object>& objs) {
    // 备份原始 objs（保留 boxMask、label、prob）
    std::vector<seg::Object> original_objs = objs;

    // preprocess seg::Object -> DETECTION
    DETECTIONS detections;
    std::vector<CLSCONF> clsConf;

    for (const auto& obj : original_objs) {
        float x1 = obj.rect.x;
        float y1 = obj.rect.y;
        float w  = obj.rect.width;
        float h  = obj.rect.height;

        DETECTBOX box(x1, y1, w, h);  // xywh
        DETECTION_ROW d;
        d.prob = obj.prob;
        d.tlwh = box;
        d.rect = obj.rect;
        d.label = obj.label;
        d.boxMask = obj.boxMask;
        d.confidence = obj.prob;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)obj.label, obj.prob));
    }

    result.clear();
    results.clear();
    obj_res.clear();
    if (!detections.empty()) {
        DETECTIONSV2 detectionsv2 = std::make_pair(clsConf, detections);
        sort(frame, detectionsv2); // 调用内部跟踪逻辑
    }

    // postprocess -> 写回 objs
    objs.clear();
    for(size_t i =0;i<obj_res.size();++i)
    {
       const auto r = obj_res[i];
       DETECTION_ROW det = r.second;
       seg::Object obj;

       obj.track_id = r.first;
       obj.label = det.label;
       obj.rect = det.rect;
       obj.prob = det.prob;
       obj.track_rect = obj.rect;
       obj.boxMask = det.boxMask;
       objs.push_back(obj);
    }
}


void DeepSort::sort(cv::Mat& frame, std::vector<seg::Object>& objs,std::map<int, cv::Rect> &obj_proj) {
    // 备份原始 objs（保留 boxMask、label、prob）
    std::vector<seg::Object> original_objs = objs;

    // preprocess seg::Object -> DETECTION
    DETECTIONS detections;
    std::vector<CLSCONF> clsConf;

    for (const auto& obj : original_objs) {
        float x1 = obj.rect.x;
        float y1 = obj.rect.y;
        float w  = obj.rect.width;
        float h  = obj.rect.height;

        DETECTBOX box(x1, y1, w, h);  // xywh
        DETECTION_ROW d;
        d.tlwh = box;
        d.rect = obj.rect;
        d.label = obj.label;
        d.boxMask = obj.boxMask;
        d.confidence = obj.prob;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)obj.label, obj.prob));
    }

    result.clear();
    results.clear();
    obj_res.clear();
    if (!detections.empty()) {
        DETECTIONSV2 detectionsv2 = std::make_pair(clsConf, detections);
        sort(frame, detectionsv2,obj_proj); // 调用内部跟踪逻辑
    }

    // postprocess -> 写回 objs
    objs.clear();
    for(size_t i =0;i<obj_res.size();++i)
    {
       const auto r = obj_res[i];
       DETECTION_ROW det = r.second;
       seg::Object obj;
    
       obj.track_id = r.first;
       obj.label = det.label;
       obj.rect = det.rect;
       obj.prob = det.confidence;
       obj.track_rect = det.track_rect;
       obj.boxMask = det.boxMask;
       objs.push_back(obj);
    }
}


void DeepSort::sort(cv::Mat& frame, DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        //result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}

void DeepSort::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {

        objTracker->predict();
        objTracker->update(detectionsv2);

        result.clear();
        results.clear();
        obj_res.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            obj_res.push_back(make_pair(track.track_id, track.get_det()));
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf) ,track.to_tlwh()));
        }
    }
}

void DeepSort::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2,std::map<int, cv::Rect> &obj_proj) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {

        objTracker->predict();
        objTracker->update(detectionsv2,obj_proj);

        result.clear();
        results.clear();
        obj_res.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            obj_res.push_back(make_pair(track.track_id, track.get_det()));
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf) ,track.to_tlwh()));
        }
    }
}


void DeepSort::sort(vector<DetectBox>& dets) {
    DETECTIONS detections;
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
    }
    if (detections.size() > 0)
        sort(detections);
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2), i(3), 1.);
        b.trackID = r.first;
        dets.push_back(b);
    }
}

void DeepSort::sort(DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}
