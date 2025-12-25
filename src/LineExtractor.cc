/*
 *  
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <functional>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <exception>
#include <opencv2/core/core.hpp>

#include "LineExtractor.h"

#define VERBOSE 0

namespace ORB_SLAM3
{

class Sharpener
{
public:

    static const float kSharpenValue;
public:

    Sharpener();

    void Sharpen(cv::Mat& frame);


public: /// setters

    void SetSharpenValue(float val);

protected:

    cv::Mat _sharpKernel;

    float _sharpenValue;
};

const float Sharpener::kSharpenValue = 0.4; // default value in the range [0,1]

Sharpener::Sharpener()
{
    _sharpKernel = 0;

    _sharpenValue = kSharpenValue;

    SetSharpenValue(kSharpenValue);

}

void Sharpener::SetSharpenValue(float val)
{
    _sharpenValue = val;
    _sharpKernel = (cv::Mat_<float>(3, 3) <<  0., -val, 0., 
                                            -val, 1.f + 4.f * val, -val, 
                                              0., -val, 0. ); 
}

void Sharpener::Sharpen(cv::Mat& frame)
{
    cv::filter2D(frame, frame, -1, _sharpKernel);
    //cvFilter2D(frame, frame, _sharpKernel);
}

struct sort_lines_by_response
{

    inline bool operator()(const cv::line_descriptor_c::KeyLine& a, const cv::line_descriptor_c::KeyLine& b)
    {
        return ( a.response > b.response);
    }
};


bool LineExtractor::skUseLsdExtractor = false; // use LSD exactor (slower)
    
const int LineExtractor::kBorderThreshold = 5;
const int LineExtractor::kNumOctavesForBdDefault = 2; 

const float LineExtractor::kScaleFactorDefault = sqrt(2);//1.2; /// sqrt(2) is the default value
const int LineExtractor::kGaussianFilterSize = 5; /// 5 is the default value
float LineExtractor::skSigma0 = 2; // base sigma for level 0 of the pyramid

LineExtractor::LineExtractor(int numLinefeatures_in, cv::line_descriptor_c::LSDDetectorC::LSDOptions& opts_in) : mOpts(opts_in)
{
    mnLinefeatures = numLinefeatures_in;

    mdSigmaScale = skSigma0; // base sigma for level 0 of the pyramid

    mnLevels = opts_in.numOctaves;
    mfScaleFactor = opts_in.scale;
    mvScaleFactor.resize(mnLevels);
    mvLevelSigma2.resize(mnLevels);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = mdSigmaScale*mdSigmaScale;
    for (int i = 1; i < mnLevels; i++)
    {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * mfScaleFactor;
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i] * mvLevelSigma2[0]; //mvLevelSigma2[i - 1];
    }

    mvInvScaleFactor.resize(mnLevels);
    mvInvLevelSigma2.resize(mnLevels);
    for (int i = 0; i < mnLevels; i++)
    {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }


    cv::line_descriptor_c::BinaryDescriptor::Params params;
    params.numOfOctave_ = mnLevels; 
    params.ksize_ = kGaussianFilterSize;
    params.scaleFactor_ = opts_in.scale;//kScaleFactorDefault;
    //params.widthOfBand_ = 7; // default value in Thirdparty/line_descriptor/src/binary_descriptor_custom.cpp
        
    mLbd = cv::line_descriptor_c::BinaryDescriptor::createBinaryDescriptor(params);
    if(skUseLsdExtractor)   
        mLsd = cv::line_descriptor_c::LSDDetectorC::createLSDDetectorC(opts_in);    
    
}

void LineExtractor::SetOptions(cv::line_descriptor_c::LSDDetectorC::LSDOptions& opts_in)
{
    mOpts = opts_in;
}

void LineExtractor::operator()(const cv::Mat& image, std::vector<cv::line_descriptor_c::KeyLine>& keylines, cv::Mat& descriptors)
{
    detectLineFeatures(image, keylines, descriptors, mOpts.min_length);
}

class Border
{
public:

    Border(const int& minX_in, const int& maxX_in, const int& minY_in, const int& maxY_in)
    {
        minX = minX_in;
        maxX = maxX_in;
        minY = minY_in;
        maxY = maxY_in;
    }

    int minX, maxX, minY, maxY;
};

void LineExtractor::SetGaussianPyramid(const std::vector<cv::Mat>& pyramid, int numOctaves, float scale, int borderX_in, int borderY_in)
{
    mLbd->setGaussianPyramid(pyramid, numOctaves, scale, borderX_in, borderY_in);
    if(skUseLsdExtractor) mLsd->setGaussianPyramid(pyramid, numOctaves, scale, borderX_in, borderY_in);
}

class BorderChecker
{
public:

    BorderChecker(const Border& border_in) : border(border_in)
    {
    }

    bool operator()(const cv::line_descriptor_c::KeyLine& line) const
    {
        const float& startPointX = line.startPointX;
        const float& startPointY = line.startPointY;
        const float& endPointX = line.endPointX;
        const float& endPointY = line.endPointY;
        return (((startPointX < border.minX) && (endPointX < border.minX)) ||
                ((startPointX > border.maxX) && (endPointX > border.maxX)) ||
                ((startPointY < border.minY) && (endPointY < border.minY)) ||
                ((startPointY > border.maxY) && (endPointY > border.maxY)));
    }

    const Border border;
};

bool isPointInMask(const cv::Point2f& point, const cv::Mat& mask) {
    // 检查点是否在图像范围内
    if (point.x < 0 || point.x >= mask.cols || point.y < 0 || point.y >= mask.rows) {
        return false;
    }
    // 检查点是否在掩码内
    return mask.at<uchar>(point.y, point.x) == 255;
}

bool isLineInMask(const cv::line_descriptor_c::KeyLine& line, const cv::Mat& masks) {
    cv::Point2f startPoint(line.startPointX, line.startPointY);
    cv::Point2f endPoint(line.endPointX, line.endPointY);

    // 检查两个端点是否都在掩码内
    bool startInMask = isPointInMask(startPoint, masks);
    bool endInMask = isPointInMask(endPoint, masks);

    return startInMask || endInMask;
}

void LineExtractor::detectLineFeatures(const cv::Mat& img, std::vector<cv::line_descriptor_c::KeyLine> &lines, cv::Mat &descriptors, double min_line_length)
{
    mOpts.min_length = min_line_length;

    if(skUseLsdExtractor)
    {
        mLsd->detect(img, lines, mOpts.scale, mOpts.numOctaves, mOpts); 
    }
    else
    {
        mLbd->detect(img, lines);
         
    }

//     if(lines.size()<500)
//     {
//         Sharpener shapener; 
//         cv::Mat imgSharpened = img.clone(); 
//         shapener.Sharpen(imgSharpened);
//         mLsd->detect(imgSharpened, lines, 2.0, 4.0, mOpts); 
//         // std::cout << "using sharper, found  " << lines.size() << " lines" << std::endl; 
//         // cv::Mat img_vis;
//         // cv::cvtColor(img, img_vis, cv::COLOR_GRAY2BGR);  // 确保三通道，方便画彩色线段

// // for (size_t i = 0; i < lines.size(); i++)
// // {
// //     cv::Point2f start(lines[i].startPointX, lines[i].startPointY);
// //     cv::Point2f end(lines[i].endPointX, lines[i].endPointY);

// //     // 画线段
// //     cv::line(img_vis, start, end, cv::Scalar(0, 255, 0), 2);

// //     // 可选：画端点
// //     cv::circle(img_vis, start, 3, cv::Scalar(0, 0, 255), -1);
// //     cv::circle(img_vis, end,   3, cv::Scalar(255, 0, 0), -1);

// //     // 可选：给每条线标上索引
// //     cv::putText(img_vis, std::to_string(i), start, cv::FONT_HERSHEY_SIMPLEX,
// //                 0.4, cv::Scalar(255, 255, 0), 1);
// // }

// // cv::imshow("Detected Lines", img_vis);
// // cv::waitKey(0);
//     }
    
#if VERBOSE    
    std::cout << "LineExtractor::detectLineFeatures() - detected " << lines.size()<< " lines" << std::endl; 
#endif
    

#if 1

        // 过滤掉 mvDynamicArea 内的线特征
    lines.erase(std::remove_if(lines.begin(), lines.end(),
                [&](const cv::line_descriptor_c::KeyLine& line) {
                    return isLineInMask(line, all_dynaMask);
                }),
                lines.end());

    // filter lines
    if (lines.size() > mnLinefeatures && mnLinefeatures != 0)
    {
        // sort lines by their response
        sort(lines.begin(), lines.end(), sort_lines_by_response());
        //sort( lines.begin(), lines.end(), sort_lines_by_length() );
        lines.resize(mnLinefeatures);
        // reassign index
        //for (int i = 0; i < mnLinefeatures; i++)
        //    lines[i].class_id = i;
    }

    Border border(kBorderThreshold, img.cols - kBorderThreshold, kBorderThreshold, img.rows - kBorderThreshold);
    lines.erase(std::remove_if(lines.begin(), lines.end(), BorderChecker(border)), lines.end());

    // reassign index and check min length
    bool bCutForMinLength = false;
    int iCut = 0;
    for (size_t i = 0,iEnd=lines.size(); i < iEnd; i++)
    {
        //std::cout  << "line " << i << " - octave:" << lines[i].octave << "- S (" << lines[i].startPointX << ", " << lines[i].startPointY << ")" << "- E (" << lines[i].endPointX << ", " << lines[i].endPointY << ")" << std::endl;
        //std::cout  << "line " << i << "- response: " << lines[i].response << std::endl;
        lines[i].class_id = i;
        if (lines[i].response < min_line_length)
        {
            iCut = i; 
            bCutForMinLength = true;
            break;
        }
    }
    
    if (bCutForMinLength)
    {
        if (iCut >= 0)
            lines.resize(iCut);//调整 lines 向量的大小，保留前 iCut 个元素
        else
            lines.clear();
    }

#endif
    
    if(lines.empty()) 
    {
        std::cout << "LineExtractor::detectLineFeatures() - no lines! **********" << std::endl; 
        return; 
    }
    
    if(skUseLsdExtractor)   
    {
        // N.B.: can't reuse detection data since we used the LSD extractor 
        mLbd->compute(img, lines, descriptors, false /*returnFloatDescr*/, false /*reuse detect data*/); 
    }
    else
    {
        mLbd->compute(img, lines, descriptors, false /*returnFloatDescr*/, true /*reuse detect data*/);
    }
    
#if VERBOSE    
    std::cout << "LineExtractor::detectLineFeatures() - extracted " << lines.size()<< " lines" << std::endl; 
#endif    
}

} //namespace ORB_SLAM3


