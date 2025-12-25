#include "opencv2/opencv.hpp"

#include <chrono>
#include "ImgSeg.h"
#include "YoloSort.h"
int main(){
    const std::string engine_path="../model/yolov8m-seg.engine";
    const std::string sort_engine_path="../model/deepsort.engine";
    ImgSeg yolo(engine_path);
    YoloSort yoloSort(sort_engine_path);

    std::string img_path="/home/haochen/slam/Dynamic_Object_Track_SLAM/data/rgbd_dataset_freiburg3_walking_xyz/rgb";
    std::vector<std::string> imagePathList;

 
    std::vector<seg::Object> objs;
    cv::glob(img_path+ "/*.png", imagePathList);
    for (auto& path : imagePathList) {
      cv::Mat image = cv::imread(path);
         objs.clear();
         yolo.detect(image, objs);
     std::vector<DetectBox> boxes;
    for (const auto& obj : objs) {
        if(obj.label!=0)
        continue;
        float x1 = obj.rect.x;
        float y1 = obj.rect.y;
        float x2 = obj.rect.x + obj.rect.width;
        float y2 = obj.rect.y + obj.rect.height;
        float confidence = obj.prob;
        float classID = obj.label;
        DetectBox box(x1, y1, x2, y2, confidence, classID, -1);
        boxes.push_back(box);
    }

    yoloSort.detect(image,boxes);
     int i =0;
    
        for (auto& obj : objs) {
        
        if(obj.label!=0)
        {

          continue;
        }
       
        // 将目标检测结果的坐标更新到 objs 中
        //const auto& obj = obj;
        DetectBox& box = boxes[i];

        obj.track_rect.x = box.x1;
        obj.track_rect.y = box.y1;
        obj.track_rect.width = box.x2 - box.x1;
        obj.track_rect.height = box.y2 - box.y1;

        // 更新 objs 中的 trackID
        obj.track_id = box.trackID;  // DeepSORT 给出的 trackID
        i++;
    }
    // yoloSort.showDetection(image, boxes);
    // yolo.visualizeObjects(image, objs);
    // cv::imshow("img",image);
    // cv::waitKey(1);
     }
   
    return 0;
}