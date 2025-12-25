/*
 *  .
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
/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "Atlas.h"
#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "MapLine.h"
#include "Utils.h"
#include "MapObject.h"
#include "ImgSeg.h"
#include <pangolin/pangolin.h>
#include <mutex>
#include <unordered_set>

#define USE_ORIGINAL_MAP_LOCK_STYLE 0
#define DRAW_FOV_CENTERS 0

// #define COLOR_LINE_OBJECT 0.745,0.969,0.125
#define COLOR_LINE_OBJECT 0.471, 0.98, 0.125

namespace ORB_SLAM3
{

    MapDrawer::MapDrawer(Atlas *pAtlas, const string &strSettingPath, Settings *settings) : mpAtlas(pAtlas)
    {
        if (settings)
        {
            newParameterLoader(settings);
        }
        else
        {
            cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
            bool is_correct = ParseViewerParamFile(fSettings);

            if (!is_correct)
            {
                std::cerr << "**ERROR in the config file [MapDrawer], the format is not correct**" << std::endl;
                try
                {
                    throw -1;
                }
                catch (exception &e)
                {
                }
            }
        }
    }

    void MapDrawer::newParameterLoader(Settings *settings)
    {
        mKeyFrameSize = settings->keyFrameSize();
        mKeyFrameLineWidth = settings->keyFrameLineWidth();
        mGraphLineWidth = settings->graphLineWidth();
        mPointSize = settings->pointSize();
        mLineSize = settings->lineSize();
        mCameraSize = settings->cameraSize();
        mCameraLineWidth = settings->cameraLineWidth();

        mbUseAR = false;
    }

    bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings)
    {
        bool b_miss_params = false;

        cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
        if (!node.empty())
        {
            mKeyFrameSize = node.real();
        }
        else
        {
            std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.KeyFrameLineWidth"];
        if (!node.empty())
        {
            mKeyFrameLineWidth = node.real();
        }
        else
        {
            std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.GraphLineWidth"];
        if (!node.empty())
        {
            mGraphLineWidth = node.real();
        }
        else
        {
            std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.PointSize"];
        if (!node.empty())
        {
            mPointSize = node.real();
        }
        else
        {
            std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.CameraSize"];
        if (!node.empty())
        {
            mCameraSize = node.real();
        }
        else
        {
            std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.CameraLineWidth"];
        if (!node.empty())
        {
            mCameraLineWidth = node.real();
        }
        else
        {
            std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        mLineSize = Utils::GetParam(fSettings, "Viewer.LineSize", 1);

        mbUseAR = false;

        return !b_miss_params;
    }

    void MapDrawer::DrawMapPoints()
    {
        Map *pActiveMap = mpAtlas->GetCurrentMap();
        if (!pActiveMap)
            return;

#if USE_ORIGINAL_MAP_LOCK_STYLE
        // original version
        const vector<MapPointPtr> &vpMPs = pActiveMap->GetAllMapPoints();
        const vector<MapPointPtr> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

        set<MapPointPtr> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());
#else
        const vector<MapPointPtr> vpMPs = pActiveMap->GetAllMapPoints();
        const vector<MapPointPtr> vpRefMPs = pActiveMap->GetReferenceMapPoints();

        const unordered_set<MapPointPtr> spRefMPs(vpRefMPs.begin(), vpRefMPs.end()); // this makes the vector-to-set conversion occur in this thread
#endif

        if (vpMPs.empty())
            return;

        if (mbUseAR && mbUseKbRawFeatures)
        {
            // draw points with the KB shader

            std::vector<Eigen::Vector3f> points;
            points.reserve(vpMPs.size() + vpRefMPs.size());
            for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
            {
                if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
                    continue;
                points.push_back(vpMPs[i]->GetWorldPos());
            }
            for (auto sit = spRefMPs.begin(), send = spRefMPs.end(); sit != send; sit++)
            {
                if ((*sit)->isBad())
                    continue;
                points.push_back((*sit)->GetWorldPos());
            }
            kbFeaturesProgram_->DrawPoints(points);
        }
        else
        {
            // draw points in the standard way

            glPointSize(mPointSize);

            glBegin(GL_POINTS);
            glColor3f(0.0, 0.0, 0.0);
            for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
            {
                if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
                    continue;
                Eigen::Matrix<float, 3, 1> pos = vpMPs[i]->GetWorldPos();
                glVertex3f(pos(0), pos(1), pos(2));
            }
            glEnd();

            if (!mbUseAR)
            {
                glPointSize(mPointSize);
            }
            else
            {
                glPointSize(mPointSize + 1);
            }

            glBegin(GL_POINTS);
            if (!mbUseAR)
            {
                glColor3f(1.0, 0.0, 0.0);
            }

            for (auto sit = spRefMPs.begin(), send = spRefMPs.end(); sit != send; sit++)
            {
                if ((*sit)->isBad())
                    continue;
                Eigen::Matrix<float, 3, 1> pos = (*sit)->GetWorldPos();
                glVertex3f(pos(0), pos(1), pos(2));
            }
            glEnd();
        }
    }

// p0, p1, p2, p3 为控制点， t ∈ [0,1]
Eigen::Vector3d CatmullRom(const Eigen::Vector3d& p0,
                           const Eigen::Vector3d& p1,
                           const Eigen::Vector3d& p2,
                           const Eigen::Vector3d& p3,
                           float t)
{
    float t2 = t * t;
    float t3 = t2 * t;

    return 0.5f * ((2.0f * p1) +
                   (-p0 + p2) * t +
                   (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                   (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}

  
void DrawSmoothTrajectoryCatmullRom(const std::vector<Eigen::Vector3d>& points, 
                                    int interp_points = 10,
                                    float line_width = 3.0f) // 轨迹宽度参数
{
    if (points.size() < 4) return; // 至少4个点

    glLineWidth(line_width);   // 设置轨迹宽度
    glBegin(GL_LINE_STRIP);

    // 前两个点直接画，保证轨迹经过头尾
    glVertex3f(points[0].x(), points[0].y(), points[0].z());
    glVertex3f(points[1].x(), points[1].y(), points[1].z());

    for (size_t i = 1; i < points.size() - 2; ++i)
    {
        const auto& p0 = points[i - 1];
        const auto& p1 = points[i];
        const auto& p2 = points[i + 1];
        const auto& p3 = points[i + 2];

        for (int j = 1; j <= interp_points; ++j)
        {
            float t = j / float(interp_points);
            Eigen::Vector3d p = CatmullRom(p0, p1, p2, p3, t);
            glVertex3f(p.x(), p.y(), p.z());
        }
    }

    // 最后两个点直接画
    glVertex3f(points[points.size() - 2].x(), points[points.size() - 2].y(), points[points.size() - 2].z());
    glVertex3f(points[points.size() - 1].x(), points[points.size() - 1].y(), points[points.size() - 1].z());

    glEnd();
    glLineWidth(1.0f); // 复位，避免影响其他绘制
}


// 根据 track_id 生成可区分的颜色
Eigen::Vector3f GetColorFromTrackID(int track_id)
{
    // 使用简单哈希保证颜色分布
    int r = (track_id * 97) % 255;
    int g = (track_id * 57) % 255;
    int b = (track_id * 37) % 255;

    // 避免颜色过暗
    r = std::max(r, 50);
    g = std::max(g, 50);
    b = std::max(b, 50);

    return Eigen::Vector3f(r / 255.0f, g / 255.0f, b / 255.0f);
}


void MapDrawer::DrawMapObject()
    {
        Map *pActiveMap = mpAtlas->GetCurrentMap();
        if (!pActiveMap)
            return;

        const std::vector<MapObject *> objects = pActiveMap->GetAllMapObjects();
        if (objects.empty())
            return;

        glLineWidth(2);
        glPointSize(mPointSize);

        for (auto obj : objects)
        {

            const Ellipsoid &ell = obj->GetEllipsoid();
            int color_index = obj->label;

            const auto &color_rgb = COLORS[color_index];
            float r = color_rgb[0] / 255.0f;
            float g = color_rgb[1] / 255.0f;
            float b = color_rgb[2] / 255.0f;
            
            // ================= 绘制椭球 =================
            if(obj->label!=0)
            {
                            auto pts = ell.GeneratePointCloud();
            int i = 0;
            while (i < pts.rows())
            {
                glBegin(GL_LINE_STRIP);
                glColor3f(r, g, b);
                for (int k = 0; k < 50 && i < pts.rows(); ++k, ++i)
                {
                    glVertex3f(pts(i, 0), pts(i, 1), pts(i, 2));
                }
                glEnd();
            }
            }


            // ================= 绘制分段轨迹 =================
Eigen::Vector3f track_color = GetColorFromTrackID(obj->track_id); // 轨迹颜色
glColor3f(track_color.x(), track_color.y(), track_color.z());

for (const auto& segment : obj->trajectory_segments)
{
    if (segment.size() >= 10)
        DrawSmoothTrajectoryCatmullRom(segment);
    else
    {
        glBegin(GL_LINE_STRIP);
        for (const auto& p : segment)
            glVertex3f(p.x(), p.y(), p.z());
        glEnd();
    }
}
        }
    }

    
void splitLine(const Eigen::Vector3f &posStart, const Eigen::Vector3f &posEnd, std::vector<Eigen::Vector3f> &linePoints, const int numSplits)
    {
        // split the line into multiple line segments for having better rendering with the line shader
        const Eigen::Vector3f direction = posEnd - posStart;
        const Eigen::Vector3f delta = direction / numSplits;
        Eigen::Vector3f pi = posStart;
        for (size_t i = 0; i < numSplits; i++)
        {
            linePoints.push_back(pi);
            pi += delta;
            linePoints.push_back(pi);
        }
    }

    void MapDrawer::DrawMapLines()
    {
        constexpr int numSplits = 10;
        constexpr float distanceForSplit = 0.1f; // meters

#if USE_ORIGINAL_MAP_LOCK_STYLE
        const vector<MapLinePtr> &vpMLs = mpAtlas->GetAllMapLines();
        const vector<MapLinePtr> &vpRefMLs = mpAtlas->GetReferenceMapLines();

        set<MapLinePtr> spRefMLs(vpRefMLs.begin(), vpRefMLs.end());
#else
        const vector<MapLinePtr> vpMLs = mpAtlas->GetAllMapLines();
        const vector<MapLinePtr> vpRefMLs = mpAtlas->GetReferenceMapLines();

        const unordered_set<MapLinePtr> spRefMLs(vpRefMLs.begin(), vpRefMLs.end()); // this makes the vector-to-set conversion occur in this thread
#endif

        if (vpMLs.empty())
            return;

        if (mbUseAR && mbUseKbRawFeatures)
        {
            // draw lines with the KB shader

            std::vector<Eigen::Vector3f> linePoints;
            linePoints.reserve(2 * (vpMLs.size() + spRefMLs.size()) * numSplits);
            Eigen::Vector3f posStart, posEnd;
            float length = 0;

            for (size_t i = 0, iend = vpMLs.size(); i < iend; i++)
            {
                if (vpMLs[i]->isBad() || spRefMLs.count(vpMLs[i]))
                    continue;
                vpMLs[i]->GetWorldEndPointsAndLength(posStart, posEnd, length);
                if (length > distanceForSplit)
                {
                    splitLine(posStart, posEnd, linePoints, numSplits);
                }
                else
                {
                    linePoints.push_back(posStart);
                    linePoints.push_back(posEnd);
                }
            }
            for (auto sit = spRefMLs.begin(), send = spRefMLs.end(); sit != send; sit++)
            {
                if ((*sit)->isBad())
                    continue;
                (*sit)->GetWorldEndPointsAndLength(posStart, posEnd, length);
                if (length > distanceForSplit)
                {
                    splitLine(posStart, posEnd, linePoints, numSplits);
                }
                else
                {
                    linePoints.push_back(posStart);
                    linePoints.push_back(posEnd);
                }
            }
            kbFeaturesProgram_->DrawLines(linePoints);
        }
        else
        {
            // draw lines in the standard way

            glLineWidth(mLineSize);
            glBegin(GL_LINES);
            glColor3f(0.0, 0.0, 0.0);

            for (size_t i = 0, iend = vpMLs.size(); i < iend; i++)
            {
                if (vpMLs[i]->isBad() || spRefMLs.count(vpMLs[i]))
                    continue;

                // if(vpMLs[i]->Observations() < 3) continue;

                Eigen::Vector3f posStart, posEnd;
                vpMLs[i]->GetWorldEndPoints(posStart, posEnd);

                glVertex3f(posStart(0), posStart(1), posStart(2));
                glVertex3f(posEnd(0), posEnd(1), posEnd(2));
            }
            glEnd();

            if (!mbUseAR)
            {
                glLineWidth(mLineSize);
            }
            else
            {
                glLineWidth(mLineSize + 2);
            }

            glBegin(GL_LINES);
            if (!mbUseAR)
            {
                glColor3f(1.0, 0.0, 0.0);
            }
            for (auto sit = spRefMLs.begin(), send = spRefMLs.end(); sit != send; sit++)
            {
                if ((*sit)->isBad())
                    continue;

                // if((*sit)->Observations() < 3) continue;

                Eigen::Vector3f posStart, posEnd;
                (*sit)->GetWorldEndPoints(posStart, posEnd);

                glVertex3f(posStart(0), posStart(1), posStart(2));
                glVertex3f(posEnd(0), posEnd(1), posEnd(2));
            }
            glEnd();
        }
    }

    void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba)
    {
        const float &w = mKeyFrameSize;
        const float h = w * 0.75;
        const float z = w * 0.6;

        Map *pActiveMap = mpAtlas->GetCurrentMap();
        // DEBUG LBA
        const std::set<long unsigned int> sOptKFs = bDrawOptLba ? pActiveMap->msOptKFs : std::set<long unsigned int>();
        const std::set<long unsigned int> sFixedKFs = bDrawOptLba ? pActiveMap->msFixedKFs : std::set<long unsigned int>();

        if (!pActiveMap)
            return;

        const vector<KeyFramePtr> vpKFs = pActiveMap->GetAllKeyFrames();

        if (bDrawKF)
        {
            for (size_t i = 0; i < vpKFs.size(); i++)
            {
                KeyFramePtr pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                unsigned int index_color = pKF->mnOriginMapId;

                glPushMatrix();

                glMultMatrixf((GLfloat *)Twc.data());

                if (!pKF->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth * 5);
                    glColor3f(1.0f, 0.0f, 0.0f);
                    glBegin(GL_LINES);
                }
                else
                {
                    // cout << "Child KF: " << vpKFs[i]->mnId << endl;
                    glLineWidth(mKeyFrameLineWidth);
                    if (bDrawOptLba)
                    {
                        if (sOptKFs.find(pKF->mnId) != sOptKFs.end())
                        {
                            glColor3f(0.0f, 1.0f, 0.0f); // Green -> Opt KFs
                        }
                        else if (sFixedKFs.find(pKF->mnId) != sFixedKFs.end())
                        {
                            glColor3f(1.0f, 0.0f, 0.0f); // Red -> Fixed KFs
                        }
                        else
                        {
                            glColor3f(0.0f, 0.0f, 1.0f); // Basic color
                        }
                    }
                    else
                    {
                        glColor3f(0.0f, 0.0f, 1.0f); // Basic color
                    }
                    glBegin(GL_LINES);
                }

                glVertex3f(0, 0, 0);
                glVertex3f(w, h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, h, z);

                glVertex3f(w, h, z);
                glVertex3f(w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(-w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(w, h, z);

                glVertex3f(-w, -h, z);
                glVertex3f(w, -h, z);
                glEnd();

                glPopMatrix();

                // Draw lines with Loop and Merge candidates
                /*glLineWidth(mGraphLineWidth);
                glColor4f(1.0f,0.6f,0.0f,1.0f);
                glBegin(GL_LINES);
                cv::Mat Ow = pKF->GetCameraCenter();
                const vector<KeyFramePtr> vpLoopCandKFs = pKF->mvpLoopCandKFs;
                if(!vpLoopCandKFs.empty())
                {
                    for(vector<KeyFramePtr>::const_iterator vit=vpLoopCandKFs.begin(), vend=vpLoopCandKFs.end(); vit!=vend; vit++)
                    {
                        cv::Mat Ow2 = (*vit)->GetCameraCenter();
                        glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                        glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                    }
                }
                const vector<KeyFramePtr> vpMergeCandKFs = pKF->mvpMergeCandKFs;
                if(!vpMergeCandKFs.empty())
                {
                    for(vector<KeyFramePtr>::const_iterator vit=vpMergeCandKFs.begin(), vend=vpMergeCandKFs.end(); vit!=vend; vit++)
                    {
                        cv::Mat Ow2 = (*vit)->GetCameraCenter();
                        glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                        glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                    }
                }*/

                glEnd();
            }
        }

        if (bDrawGraph)
        {
            glLineWidth(mGraphLineWidth);
            glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
            glBegin(GL_LINES);

            // cout << "-----------------Draw graph-----------------" << endl;
            for (size_t i = 0; i < vpKFs.size(); i++)
            {
                // Covisibility Graph
                const vector<KeyFramePtr> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
                Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
                if (!vCovKFs.empty())
                {
                    for (vector<KeyFramePtr>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
                    {
                        if ((*vit)->mnId < vpKFs[i]->mnId)
                            continue;
                        Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                        glVertex3f(Ow(0), Ow(1), Ow(2));
                        glVertex3f(Ow2(0), Ow2(1), Ow2(2));
                    }
                }

                // Spanning tree
                KeyFramePtr pParent = vpKFs[i]->GetParent();
                if (pParent)
                {
                    Eigen::Vector3f Owp = pParent->GetCameraCenter();
                    glVertex3f(Ow(0), Ow(1), Ow(2));
                    glVertex3f(Owp(0), Owp(1), Owp(2));
                }

                // Loops
                set<KeyFramePtr> sLoopKFs = vpKFs[i]->GetLoopEdges();
                for (set<KeyFramePtr>::iterator sit = sLoopKFs.begin(), send = sLoopKFs.end(); sit != send; sit++)
                {
                    if ((*sit)->mnId < vpKFs[i]->mnId)
                        continue;
                    Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                    glVertex3f(Ow(0), Ow(1), Ow(2));
                    glVertex3f(Owl(0), Owl(1), Owl(2));
                }
            }

            glEnd();
        }

#if DRAW_FOV_CENTERS
        glColor3f(0.0, 0.0, 1.0);
        glPointSize(5);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            const Eigen::Vector3f fovCenter = vpKFs[i]->GetFovCenter();
            glVertex3f(fovCenter(0), fovCenter(1), fovCenter(2));
        }
        glEnd();
#endif

        if (bDrawInertialGraph && pActiveMap->isImuInitialized())
        {
            glLineWidth(mGraphLineWidth);
            // glColor4f(1.0f,0.0f,0.0f,0.6f);
            glColor4f(0.0f, 0.0f, 1.0f, 0.6f); // let's draw it blue
            glBegin(GL_LINES);

            // Draw inertial links
            for (size_t i = 0; i < vpKFs.size(); i++)
            {
                KeyFramePtr pKFi = vpKFs[i];
                Eigen::Vector3f Ow = pKFi->GetCameraCenter();
                KeyFramePtr pNext = pKFi->mNextKF;
                if (pNext)
                {
                    Eigen::Vector3f Owp = pNext->GetCameraCenter();
                    glVertex3f(Ow(0), Ow(1), Ow(2));
                    glVertex3f(Owp(0), Owp(1), Owp(2));
                }
            }

            glEnd();
        }

        vector<Map *> vpMaps = mpAtlas->GetAllMaps();

        if (bDrawKF)
        {
            for (Map *pMap : vpMaps)
            {
                if (pMap == pActiveMap)
                    continue;

                vector<KeyFramePtr> vpKFs = pMap->GetAllKeyFrames();

                for (size_t i = 0; i < vpKFs.size(); i++)
                {
                    KeyFramePtr pKF = vpKFs[i];
                    Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                    unsigned int index_color = pKF->mnOriginMapId;

                    glPushMatrix();

                    glMultMatrixf((GLfloat *)Twc.data());

                    if (!vpKFs[i]->GetParent()) // It is the first KF in the map
                    {
                        glLineWidth(mKeyFrameLineWidth * 5);
                        glColor3f(1.0f, 0.0f, 0.0f);
                        glBegin(GL_LINES);
                    }
                    else
                    {
                        glLineWidth(mKeyFrameLineWidth);
                        glColor3f(mfFrameColors[index_color][0], mfFrameColors[index_color][1], mfFrameColors[index_color][2]);
                        glBegin(GL_LINES);
                    }

                    glVertex3f(0, 0, 0);
                    glVertex3f(w, h, z);
                    glVertex3f(0, 0, 0);
                    glVertex3f(w, -h, z);
                    glVertex3f(0, 0, 0);
                    glVertex3f(-w, -h, z);
                    glVertex3f(0, 0, 0);
                    glVertex3f(-w, h, z);

                    glVertex3f(w, h, z);
                    glVertex3f(w, -h, z);

                    glVertex3f(-w, h, z);
                    glVertex3f(-w, -h, z);

                    glVertex3f(-w, h, z);
                    glVertex3f(w, h, z);

                    glVertex3f(-w, -h, z);
                    glVertex3f(w, -h, z);
                    glEnd();

                    glPopMatrix();
                }
            }
        }
    }

    void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
    {
        const float &w = mCameraSize;
        const float h = w * 0.75;
        const float z = w * 0.6;

        glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(w, h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, h, z);

        glVertex3f(w, h, z);
        glVertex3f(w, -h, z);

        glVertex3f(-w, h, z);
        glVertex3f(-w, -h, z);

        glVertex3f(-w, h, z);
        glVertex3f(w, h, z);

        glVertex3f(-w, -h, z);
        glVertex3f(w, -h, z);
        glEnd();

        glPopMatrix();
    }

    void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &Tcw)
    {
        unique_lock<mutex> lock(mMutexCamera);
        mCameraPose = Tcw.inverse();
    }

    void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
    {
        Eigen::Matrix4f Twc;
        {
            unique_lock<mutex> lock(mMutexCamera);
            Twc = mCameraPose.matrix();
        }

        for (int i = 0; i < 4; i++)
        {
            M.m[4 * i] = Twc(0, i);
            M.m[4 * i + 1] = Twc(1, i);
            M.m[4 * i + 2] = Twc(2, i);
            M.m[4 * i + 3] = Twc(3, i);
        }

        MOw.SetIdentity();
        MOw.m[12] = Twc(0, 3);
        MOw.m[13] = Twc(1, 3);
        MOw.m[14] = Twc(2, 3);
    }

    void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw, pangolin::OpenGlMatrix &MTwwp)
    {
        Eigen::Matrix4f Twc;
        Eigen::Matrix3f Rwc;
        Eigen::Vector3f twc;
        {
            unique_lock<mutex> lock(mMutexCamera);
            Twc = mCameraPose.matrix();

            Rwc = mCameraPose.rotationMatrix();
            twc = -Rwc * mCameraPose.translation();
        }

        for (int i = 0; i < 4; i++)
        {
            M.m[4 * i] = Twc(0, i);
            M.m[4 * i + 1] = Twc(1, i);
            M.m[4 * i + 2] = Twc(2, i);
            M.m[4 * i + 3] = Twc(3, i);
        }

        MOw.SetIdentity();
        MOw.m[12] = Twc(0, 3);
        MOw.m[13] = Twc(1, 3);
        MOw.m[14] = Twc(2, 3);

        Eigen::Matrix3f Rwwp = Eigen::Matrix3f::Identity();
        MTwwp.SetIdentity();
        MTwwp.m[0] = Rwwp(0, 0);
        MTwwp.m[1] = Rwwp(1, 0);
        MTwwp.m[2] = Rwwp(2, 0);

        MTwwp.m[4] = Rwwp(0, 1);
        MTwwp.m[5] = Rwwp(1, 1);
        MTwwp.m[6] = Rwwp(2, 1);

        MTwwp.m[8] = Rwwp(0, 2);
        MTwwp.m[9] = Rwwp(1, 2);
        MTwwp.m[10] = Rwwp(2, 2);

        MTwwp.m[12] = twc(0);
        MTwwp.m[13] = twc(1);
        MTwwp.m[14] = twc(2);
    }

} // namespace ORB_SLAM3
