#include "YoloSort.h"
#include "ImgSeg.h"
static Logger gLogger;
YoloSort::YoloSort(std::string sort_engine_path)
{
DS = new DeepSort(sort_engine_path, 128, 512, 0, &gLogger);
}

bool YoloSort::detect(cv::Mat &img, std::vector<DetectBox> &det)
{
     DS->sort(img, det);
    showDetection(img,det);
    return true;
}

bool YoloSort::detect(cv::Mat &img, std::vector<seg::Object> &det)
{
     DS->sort(img, det);
    showDetection(img,det);
    return true;
}

bool YoloSort::detect(cv::Mat &img, std::vector<seg::Object> &det,std::map<int, cv::Rect> &obj_proj)
{
     DS->sort(img, det,obj_proj);
    showDetection(img,det);
    return true;
}

void YoloSort::showDetection(cv::Mat &img, std::vector<DetectBox> &boxes)
{
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 1);
        //std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
		//std::string lbl = cv::format("ID:%d_C:%d", (int)box.trackID, (int)box.classID);
		std::string lbl = cv::format("ID:%d",(int)box.trackID);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,255,0));
    }
    temp.copyTo(img);
    // cv::imshow("track", temp);
    // cv::imshow("detect",img);
    // cv::waitKey(0);
}
void YoloSort::showDetection(cv::Mat &img, std::vector<seg::Object> &objs)
{
    cv::Mat temp = img.clone();

    for (const auto &obj : objs) {
        // 根据 label 取颜色，做模运算防止越界
        int color_idx = obj.label % COLORS.size();
        const auto& c = COLORS[color_idx];
        cv::Scalar box_color(c[0], c[1], c[2]);

        int mask_color_idx = obj.label % MASK_COLORS.size();
        const auto& mc = MASK_COLORS[mask_color_idx];
        cv::Scalar mask_color(mc[0], mc[1], mc[2]);

        if (!obj.boxMask.empty()) {
            cv::Mat mask;
            obj.boxMask.convertTo(mask, CV_8U);

            cv::Mat mask3c;
            cv::cvtColor(mask, mask3c, cv::COLOR_GRAY2BGR);
            // 使用掩膜颜色填充
            mask3c.setTo(mask_color, mask);

            cv::Rect intRect = obj.rect;
            cv::Rect imgRect(0, 0, img.cols, img.rows);
            cv::Rect roiRect = intRect & imgRect;

            if (roiRect.width > 0 && roiRect.height > 0) {
                cv::Mat roi = temp(roiRect);
                cv::Mat maskROI = mask3c(cv::Rect(0, 0, roiRect.width, roiRect.height));
                cv::addWeighted(roi, 1.0, maskROI, 0.5, 0, roi);
            }
        }

        cv::Point lt(obj.track_rect.x, obj.track_rect.y);
        cv::Point br(obj.track_rect.x + obj.track_rect.width, obj.track_rect.y + obj.track_rect.height);
        cv::rectangle(temp, lt, br, box_color, 2);

        std::string lbl = cv::format("ID:%d", obj.track_id);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0),1);
    }

    cv::imshow("track", temp);
}
