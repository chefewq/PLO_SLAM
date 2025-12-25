#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <iostream>
#include "yolo/common.hpp"
#include <opencv2/opencv.hpp>
#include "featuretensor.h"
#include "tracker.h"
#include "datatype.h"
#include <vector>

using std::vector;
using nvinfer1::ILogger;

class DeepSort {
public:    
    DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID, ILogger* gLogger);
    ~DeepSort();

public:
    void sort(cv::Mat& frame, vector<DetectBox>& dets);
    void sort(cv::Mat& frame,std::vector<seg::Object>& detections);
        void sort(cv::Mat& frame,std::vector<seg::Object>& detections,std::map<int, cv::Rect> &obj_proj);
private:
    void sort(cv::Mat& frame, DETECTIONS& detections);
    void sort(cv::Mat& frame, DETECTIONSV2& detectionsv2);
    void sort(cv::Mat& frame, DETECTIONSV2& detectionsv2,std::map<int, cv::Rect> &obj_proj);        
    void sort(vector<DetectBox>& dets);
    void sort(DETECTIONS& detections);
    void init();

private:
    std::string enginePath;
    int batchSize;
    int featureDim;
    cv::Size imgShape;
    float confThres;
    float nmsThres;
    int maxBudget;
    float maxCosineDist;

private:
    vector<RESULT_DATA> result;
    vector<std::pair<CLSCONF, DETECTBOX>> results;
    vector<OBJECT_DATA> obj_res;
    tracker* objTracker;
    FeatureTensor* featureExtractor;
    ILogger* gLogger;
    int gpuID;
};

#endif  //deepsort.h
