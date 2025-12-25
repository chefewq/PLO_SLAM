#ifndef YOLOSORT_H
#define YOLOSORT_H

#include <vector>
#include "deepsort/deepsort.h"
#include "deepsort/datatype.h"
#include "logging.h"

class DeepSort;
class YoloSort
{
public:
YoloSort(std::string sort_engine_path);
bool detect(cv::Mat &frame,std::vector<DetectBox> &det);
bool detect(cv::Mat &frame,std::vector<seg::Object> &det);
bool detect(cv::Mat &frame,std::vector<seg::Object> &det,std::map<int, cv::Rect> &obj_proj);
void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes);
void showDetection(cv::Mat &img, std::vector<seg::Object> &objs);
private:
DeepSort* DS;
std::vector<DetectBox> t;

};

#endif