#include "ImgSeg.h"
#include "yolo/yolov8-seg.hpp"
ImgSeg::ImgSeg (){
}
ImgSeg::ImgSeg(std::string path){
cudaSetDevice(0);
yolo = new YOLOv8_seg(path);
yolo->make_pipe(true);
}
bool ImgSeg::detect(cv::Mat image,std::vector<seg::Object> &objs)
{
            // cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
            objs.clear();
            //image = cv::imread(path);
            yolo->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolo->infer();
            auto end = std::chrono::system_clock::now();
            yolo->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);
            // yolo->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
            // // auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            // // printf("cost %2.4lf ms\n", tc);
            // cv::imshow("result", res);
            return true;
}


void ImgSeg::visualizeObjects(cv::Mat& image, const std::vector<seg::Object>& objs) {
cv::Mat mask = image.clone();  // 创建图像副本
for (const auto& obj : objs) {
    // if (obj.label != 0)
    //     continue;

    cv::Scalar color(255, 0, 0);  // 绿色框
    cv::rectangle(image, obj.rect, color, 2);  // 绘制目标框

    // 显示标签文本
    std::string label_text = "ID: " + std::to_string(obj.track_id) +
                             " Label: " + std::to_string(obj.label);
    int baseLine = 0;
    cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::putText(image, label_text, cv::Point(obj.rect.x, obj.rect.y - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

    if (!obj.boxMask.empty()) {
    // cv::waitKey(1);
    mask(obj.rect).setTo(cv::Scalar(255, 0, 0), obj.boxMask);

    }
}

double alpha = 0.5; 
double beta = 0.5;  
cv::addWeighted(image, alpha, mask, beta, 0, image);  // 进行加权叠加

}