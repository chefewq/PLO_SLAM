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
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez
 * Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

#include <include/CameraModels/KannalaBrandt8.h>
#include <include/CameraModels/Pinhole.h>

#include <thread>

#include "Converter.h"
#include "Frame.h"
#include "G2oTypes.h"
#include "Geom2DUtils.h"
#include "GeometricCamera.h"
#include "KeyFrame.h"
#include "LineMatcher.h"
#include "MapLine.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Stopwatch.h"
#include "Tracking.h"
#include "Utils.h"

#if RERUN_ENABLE
#endif

#define LOG_ASSIGN_FEATURES 0

#define CHECK_RGBD_ENDPOINTS_DEPTH_CONSISTENCY 1
#define USE_PARALLEL_POINTS_LINES_EXTRACTION 1
#define USE_UNFILTERED_PYRAMID_FOR_LINES \
  1 // N.B.: AT PRESENT TIME, it's better not to use the filtered pyramid!
    //       This is because a Gaussian filter is added in keylines extraction
    //       when a pre-computed pyramid is set. In fact, the keypoint Gaussian
    //       filter is too strong for keyline extraction (it blurs the image
    //       too much!)

#define DISABLE_STATIC_LINE_TRIANGULATION \
  0 // set this to 1 to check how local mapping is able to triangulate lines
    // without using any static stereo line triangulation

#if LOG_ASSIGN_FEATURES
#include "Logger.h"
static const string logFileName = "assign_feature.log";
static Logger logger(logFileName);
#endif

namespace ORB_SLAM3
{

  long unsigned int Frame::nNextId = 0;
  bool Frame::mbInitialComputations = true;
  bool Frame::mbUseFovCentersKfGenCriterion = false;
  float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
  float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
  float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;
  float Frame::mfLineGridElementThetaInv, Frame::mfLineGridElementDInv;
  float Frame::mnMaxDiag;
  std::vector<cv::Point2f> Prepoint, Curpoint;
  cv::Mat imGrayPre;
  std::vector<uchar> state;
  std::vector<float> Err;
  std::vector<double> epiErr, EpiErr;
  std::vector<uchar> status;
  const std::vector<size_t> Frame::kEmptyVecSizet = std::vector<size_t>();

  const float Frame::kDeltaTheta = 10 * M_PI / 180.f; // [rad]   10
  const float Frame::kDeltaD = 100;                   //[pixels]                100

  const float Frame::kTgViewZAngleMin = tan(30. * M_PI / 180.f);
  const float Frame::kCosViewZAngleMax = cos(30. * M_PI / 180.f);
  const float Frame::kDeltaZForCheckingViewZAngleMin = 0.02;           // [m]
  const float Frame::kDeltaZThresholdForRefiningEndPointsDepths = 0.0; // [m]

  const float Frame::kLinePointsMaxMisalignment = 0.03; // [m]
  const int Frame::kDeltaSizeForSearchingDepthMinMax = 1;
  const float Frame::kMinStereoLineOverlap =
      2; // [pixels]  (for pure horizontal lines one has minimum = 1)
  const float Frame::kLineNormalsDotProdThreshold =
      0.005; // this a percentage over unitary modulus
  const float Frame::kMinVerticalLineSpan =
      2; // [pixels] should be startY-endY>kMinVerticalLineSpan in order to avoid
         // line too close to an epipolar plane (degeneracy in line
         // triangulation)

  const int Frame::kMaxInt = std::numeric_limits<int>::max();

  float Frame::skMinLineLength3D = 0.01; // [m]

  // For stereo fisheye matching
  cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

  Frame::Frame()
      : mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL),
        mpImuPreintegratedFrame(NULL),
        mpReferenceKF(static_cast<KeyFramePtr>(NULL)), mbIsSet(false),
        mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
  {
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
  }

  // Copy Constructor
  Frame::Frame(const Frame &frame)
      : mpcpi(frame.mpcpi), mpORBvocabulary(frame.mpORBvocabulary),
        mpORBextractorLeft(frame.mpORBextractorLeft),
        mpORBextractorRight(frame.mpORBextractorRight),
        mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()),
        mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
        mbf(frame.mbf), mvbImg(frame.mvbImg), mvbDepth(frame.mvbDepth),
        mbfInv(frame.mbfInv), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N),
        mvKeys(frame.mvKeys), mvObjectsKeys(frame.mvObjectsKeys),
        mvObjectsDepth(frame.mvObjectsDepth), N_O(frame.N_O),
        mvObjectsDescriptors(frame.mvObjectsDescriptors),
        mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),
        mvuRight(frame.mvuRight), mvDepth(frame.mvDepth), mBowVec(frame.mBowVec),
        mFeatVec(frame.mFeatVec), mDescriptors(frame.mDescriptors.clone()),
        mDescriptorsRight(
            frame.mDescriptorsRight.clone()), // clone point descriptors
        mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier),
        mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
        mpImuPreintegrated(frame.mpImuPreintegrated),
        mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame),
        mImuBias(frame.mImuBias), mnId(frame.mnId), Nlines(frame.Nlines),
        mvKeyLines(frame.mvKeyLines), mvKeyLinesRight(frame.mvKeyLinesRight),
        mvKeyLinesUn(frame.mvKeyLinesUn),
        mvKeyLinesRightUn(frame.mvKeyLinesRightUn),
        mvuRightLineStart(frame.mvuRightLineStart),
        mvDepthLineStart(frame.mvDepthLineStart),
        mvuRightLineEnd(frame.mvuRightLineEnd),
        mvDepthLineEnd(frame.mvDepthLineEnd),
        mLineDescriptors(frame.mLineDescriptors.clone()),
        mLineDescriptorsRight(
            frame.mLineDescriptorsRight.clone()), // clone line descriptors
        mvpMapLines(frame.mvpMapLines), mvbLineOutlier(frame.mvbLineOutlier),
        mvuNumLinePosOptFailures(frame.mvuNumLinePosOptFailures),
        mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
        mfScaleFactor(frame.mfScaleFactor),
        mfLogScaleFactor(frame.mfLogScaleFactor),
        mvScaleFactors(frame.mvScaleFactors),
        mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile),
        mnDataset(frame.mnDataset), mvLevelSigma2(frame.mvLevelSigma2),
        mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame),
        mpLastKeyFrame(frame.mpLastKeyFrame), mbIsSet(frame.mbIsSet),
        mbImuPreintegrated(frame.mbImuPreintegrated),
        mpMutexImu(frame.mpMutexImu), mpCamera(frame.mpCamera),
        mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
        monoLeft(frame.monoLeft), monoRight(frame.monoRight),
        mvLeftToRightMatch(frame.mvLeftToRightMatch),
        mvRightToLeftMatch(frame.mvRightToLeftMatch),
        mvStereo3Dpoints(frame.mvStereo3Dpoints), NlinesLeft(frame.NlinesLeft),
        NlinesRight(frame.NlinesRight), monoLinesLeft(frame.monoLinesLeft),
        monoLinesRight(frame.monoLinesRight),
        mvLeftToRightLinesMatch(frame.mvLeftToRightLinesMatch),
        mvRightToLeftLinesMatch(frame.mvRightToLeftLinesMatch), mTlr(frame.mTlr),
        mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl), mTcw(frame.mTcw),
        mnLineScaleLevels(frame.mnLineScaleLevels),
        mfLineScaleFactor(frame.mfLineScaleFactor),
        mfLineLogScaleFactor(frame.mfLineLogScaleFactor),
        mvLineScaleFactors(frame.mvLineScaleFactors),
        mvLineLevelSigma2(frame.mvLineLevelSigma2),
        mvLineInvLevelSigma2(frame.mvLineInvLevelSigma2),
        mMedianDepth(frame.mMedianDepth), mbHasPose(false), mbHasVelocity(false)
  {
    for (int i = 0; i < FRAME_GRID_COLS; i++)
      for (int j = 0; j < FRAME_GRID_ROWS; j++)
      {
        mGrid[i][j] = frame.mGrid[i][j];
        if (frame.Nleft > 0)
        {
          mGridRight[i][j] = frame.mGridRight[i][j];
        }
      }

    if (frame.mpLineExtractorLeft)
    {
      for (int i = 0; i < LINE_D_GRID_COLS; i++)
        for (int j = 0; j < LINE_THETA_GRID_ROWS; j++)
        {
          mLineGrid[i][j] = frame.mLineGrid[i][j];
          if (frame.NlinesLeft > 0)
          {
            mLineGridRight[i][j] = frame.mLineGridRight[i][j];
          }
        }
    }

    if (frame.mbHasPose)
      SetPose(frame.GetPose());

    if (frame.HasVelocity())
    {
      SetVelocity(frame.GetVelocity());
    }

    // mmProjectPoints = frame.mmProjectPoints;
    // mmMatchedInImage = frame.mmMatchedInImage;

    // mmProjectLines = frame.mmProjectLines;
    // mmMatchedLinesInImage = frame.mmMatchedLinesInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
  }

  /// <  STEREO
  Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight,
               const double &timeStamp,
               std::shared_ptr<LineExtractor> &lineExtractorLeft,
               std::shared_ptr<LineExtractor> &lineExtractorRight,
               ORBextractor *extractorLeft, ORBextractor *extractorRight,
               ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
               const float &thDepth, GeometricCamera *pCamera, Frame *pPrevF,
               const IMU::Calib &ImuCalib)
      : mpcpi(NULL), mpLineExtractorLeft(lineExtractorLeft),
        mpLineExtractorRight(lineExtractorRight), mpORBvocabulary(voc),
        mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight),
        mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),
        mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
        mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),
        mpImuPreintegratedFrame(NULL),
        mpReferenceKF(static_cast<KeyFramePtr>(NULL)), Nlines(0),
        mMedianDepth(KeyFrame::skFovCenterDistance), mbIsSet(false),
        mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(nullptr),
        mbHasPose(false), mbHasVelocity(false)
  {
    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    if (mpLineExtractorLeft)
    {
      mnLineScaleLevels = mpLineExtractorLeft->GetLevels();
      mfLineScaleFactor = mpLineExtractorLeft->GetScaleFactor();
      mfLineLogScaleFactor = log(mfLineScaleFactor);
      mvLineScaleFactors = mpLineExtractorLeft->GetScaleFactors();
      mvLineLevelSigma2 = mpLineExtractorLeft->GetScaleSigmaSquares();
      mvLineInvLevelSigma2 = mpLineExtractorLeft->GetInverseScaleSigmaSquares();
    }

    // This is done only for the first Frame (or after a change in the
    // calibration)
    if (mbInitialComputations)
    {
      ComputeImageBounds(imLeft);

      mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) /
                              static_cast<float>(mnMaxX - mnMinX);
      mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) /
                               static_cast<float>(mnMaxY - mnMinY);

      mfLineGridElementThetaInv = static_cast<float>(LINE_THETA_GRID_ROWS) /
                                  static_cast<float>(LINE_THETA_SPAN);
      mfLineGridElementDInv =
          static_cast<float>(LINE_D_GRID_COLS) / (2.0f * mnMaxDiag);

      fx = K.at<float>(0, 0);
      fy = K.at<float>(1, 1);
      cx = K.at<float>(0, 2);
      cy = K.at<float>(1, 2);
      invfx = 1.0f / fx;
      invfy = 1.0f / fy;

      mbInitialComputations = false;
    }

    mb = mbf / fx;
    mbfInv = 1.f / mbf;

    if (pPrevF)
    {
      if (pPrevF->HasVelocity())
        SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
      mVw.setZero();
    }

    // Features extraction

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB =
        std::chrono::steady_clock::now();
#endif
    if (mpLineExtractorLeft)
    {
#ifndef USE_CUDA
      if (Tracking::skUsePyramidPrecomputation)
      {
        // pre-compute Gaussian pyramid to be used for the extraction of both
        // keypoints and keylines
        std::thread threadGaussianLeft(&Frame::PrecomputeGaussianPyramid, this, 0,
                                       imLeft);
        std::thread threadGaussianRight(&Frame::PrecomputeGaussianPyramid, this,
                                        1, imRight);
        threadGaussianLeft.join();
        threadGaussianRight.join();
      }
#else
      if (Tracking::skUsePyramidPrecomputation)
      {
        std::cout << IoColor::Yellow()
                  << "Can't use pyramid precomputation when CUDA is active!"
                  << std::endl;
      }
#endif

      // ORB extraction
      thread threadLeft(&Frame::ExtractORB, this, 0, imLeft, 0, 0);
      thread threadRight(&Frame::ExtractORB, this, 1, imRight, 0, 0);
      // Line extraction
      std::thread threadLinesLeft(&Frame::ExtractLSD, this, 0, imLeft);
      std::thread threadLinesRight(&Frame::ExtractLSD, this, 1, imRight);

      threadLeft.join();
      threadRight.join();
      threadLinesLeft.join();
      threadLinesRight.join();
    }
    else
    {
      // ORB extraction
      thread threadLeft(&Frame::ExtractORB, this, 0, imLeft, 0, 0);
      thread threadRight(&Frame::ExtractORB, this, 1, imRight, 0, 0);
      threadLeft.join();
      threadRight.join();
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB =
        std::chrono::steady_clock::now();

    mTimeORB_Ext =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            time_EndExtORB - time_StartExtORB)
            .count();
#endif

    N = mvKeys.size();
    if (mvKeys.empty())
      return;

    UndistortKeyPoints();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches =
        std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches =
        std::chrono::steady_clock::now();

    mTimeStereoMatch =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            time_EndStereoMatches - time_StartStereoMatches)
            .count();
#endif

    mvpMapPoints = vector<MapPointPtr>(N, static_cast<MapPointPtr>(NULL));
    mvbOutlier = vector<bool>(N, false);
    // mmProjectPoints.clear();
    // mmMatchedInImage.clear();

    // LSD line segments extraction
    if (mpLineExtractorLeft)
    {
      Nlines = mvKeyLines.size();

      if (mvKeyLines.empty())
      {
        std::cout << "frame " << mnId << " no lines dectected!" << std::endl;
      }
      else
      {
        UndistortKeyLines();

        ComputeStereoLineMatches();
      }

      mvpMapLines = vector<MapLinePtr>(Nlines, static_cast<MapLinePtr>(NULL));
      mvbLineOutlier = vector<bool>(Nlines, false);
      mvuNumLinePosOptFailures = vector<unsigned int>(Nlines, 0);

      // mmProjectLines.clear();// = map<long unsigned int, LineEndPoints>(N,
      // static_cast<LineEndPoints>(NULL)); mmMatchedLinesInImage.clear();
    }

    mpMutexImu = new std::mutex();

    // Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid(); // NOTE: this must stay after having initialized
                            // Nleft and Nright
  }

  /// <  RGBD
  Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth,
               const double &timeStamp,
               std::shared_ptr<LineExtractor> &lineExtractor,
               ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K,
               cv::Mat &distCoef, const float &bf, const float &thDepth,
               GeometricCamera *pCamera, Frame *pPrevF,
               const IMU::Calib &ImuCalib)
      : mpcpi(NULL), mpLineExtractorLeft(lineExtractor), mpLineExtractorRight(0),
        mpORBvocabulary(voc), mpORBextractorLeft(extractor),
        mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
        mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),
        mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), Nlines(0),
        mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),
        mpImuPreintegratedFrame(NULL),
        mpReferenceKF(static_cast<KeyFramePtr>(NULL)), mbIsSet(false),
        mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(nullptr),
        mMedianDepth(KeyFrame::skFovCenterDistance), mbHasPose(false),
        mbHasVelocity(false)
  {
    // Frame ID
    mnId = nNextId++;
    // imgLeft = imGray.clone(); // just needed for debugging and viz purposes

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    if (mpLineExtractorLeft)
    {
      mnLineScaleLevels = mpLineExtractorLeft->GetLevels();
      mfLineScaleFactor = mpLineExtractorLeft->GetScaleFactor();
      mfLineLogScaleFactor = log(mfLineScaleFactor);
      mvLineScaleFactors = mpLineExtractorLeft->GetScaleFactors();
      mvLineLevelSigma2 = mpLineExtractorLeft->GetScaleSigmaSquares();
      mvLineInvLevelSigma2 = mpLineExtractorLeft->GetInverseScaleSigmaSquares();
    }

    // This is done only for the first Frame (or after a change in the
    // calibration)
    if (mbInitialComputations)
    {
      ComputeImageBounds(imGray);

      mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) /
                              static_cast<float>(mnMaxX - mnMinX);
      mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) /
                               static_cast<float>(mnMaxY - mnMinY);

      mfLineGridElementThetaInv = static_cast<float>(LINE_THETA_GRID_ROWS) /
                                  static_cast<float>(LINE_THETA_SPAN);
      mfLineGridElementDInv =
          static_cast<float>(LINE_D_GRID_COLS) / (2.0f * mnMaxDiag);

      fx = K.at<float>(0, 0);
      fy = K.at<float>(1, 1);
      cx = K.at<float>(0, 2);
      cy = K.at<float>(1, 2);
      invfx = 1.0f / fx;
      invfy = 1.0f / fy;

      mbInitialComputations = false;
    }

    mb = mbf / fx;
    mbfInv = 1.f / mbf;

    if (pPrevF)
    {
      if (pPrevF->HasVelocity())
        SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
      mVw.setZero();
    }

    // Feature extraction

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB =
        std::chrono::steady_clock::now();
#endif

    TICKTRACK("ExtractFeatures");

#ifndef USE_CUDA
    if (Tracking::skUsePyramidPrecomputation)
    {
      // pre-compute Gaussian pyramid to be used for the extraction of both
      // keypoints and keylines
      TICKTRACK("PyramidPrecompute");
      this->PrecomputeGaussianPyramid(0, imGray);
      TOCKTRACK("PyramidPrecompute");
    }
#else
    if (Tracking::skUsePyramidPrecomputation)
    {
      std::cout << IoColor::Yellow()
                << "Can't use pyramid precomputation when CUDA is active!"
                << std::endl;
    }
#endif

#if USE_PARALLEL_POINTS_LINES_EXTRACTION

    if (mpLineExtractorLeft)
    {
      TICKTRACK("ExtractKeyPoints*");
      std::thread threadPoints(&Frame::ExtractORB, this, 0, imGray, 0, 0);
      TICKTRACK("ExtractKeyLines-");
      std::thread threadLines(&Frame::ExtractLSD, this, 0, imGray);
      threadPoints.join();
      TOCKTRACK("ExtractKeyPoints*");
      threadLines.join();
      TOCKTRACK("ExtractKeyLines-");
    }
    else
    {
      TOCKTRACK("ExtractKeyPoints*");
      // ORB extraction
      ExtractORB(0, imGray, 0, 0);
      TOCKTRACK("ExtractKeyLines-");
    }

#else

    TICKTRACK("ExtractKeyPoints*");
    // ORB extraction
    ExtractORB(0, imGray, 0, 0);
    TOCKTRACK("ExtractKeyPoints*");

    if (mpLineExtractorLeft)
    {
      TICKTRACK("ExtractKeyLines-");
      ExtractLSD(0, imGray);
      TOCKTRACK("ExtractKeyLines-");
    }

#endif

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB =
        std::chrono::steady_clock::now();

    mTimeORB_Ext =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            time_EndExtORB - time_StartExtORB)
            .count();
#endif
  }

  void reduceVector(std::vector<cv::Point2f> &v, std::vector<bool> status)
  {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
      if (status[i])
        v[j++] = v[i];
    v.resize(j);
  }

  void Frame::CalcEpiDistChi2(const std::vector<cv::KeyPoint> &kp1,
                              const std::vector<cv::Point2f> &kp2, cv::Mat &F,
                              std::vector<uchar> &state)
  {
    int N = kp1.size();
    //    EpiErr.assign(N,1);
    //    epiErr.assign(N,0);
    //    tbb::parallel_for(tbb::blocked_range<std::size_t>(0,N),[&](tbb::blocked_range<std::size_t>rN)
    //    {
    for (size_t i = 0; i < N; i++)
    {
      double A1 = F.at<double>(0, 0) * kp1[i].pt.x +
                  F.at<double>(0, 1) * kp1[i].pt.y + F.at<double>(0, 2);
      double B1 = F.at<double>(1, 0) * kp1[i].pt.x +
                  F.at<double>(1, 1) * kp1[i].pt.y + F.at<double>(1, 2);
      double C1 = F.at<double>(2, 0) * kp1[i].pt.x +
                  F.at<double>(2, 1) * kp1[i].pt.y + F.at<double>(2, 2);
      double dis1 =
          fabs(A1 * kp2[i].x + B1 * kp2[i].y + C1) / sqrt(A1 * A1 + B1 * B1);

      double A2 = F.at<double>(0, 0) * kp2[i].x + F.at<double>(1, 0) * kp2[i].y +
                  F.at<double>(2, 0);
      double B2 = F.at<double>(0, 1) * kp2[i].x + F.at<double>(1, 1) * kp2[i].y +
                  F.at<double>(2, 1);
      double C2 = F.at<double>(0, 2) * kp2[i].x + F.at<double>(1, 2) * kp2[i].y +
                  F.at<double>(2, 2);
      double dis2 = fabs(A2 * kp1[i].pt.x + B2 * kp1[i].pt.y + C2) /
                    sqrt(A2 * A2 + B2 * B2);

      double dis = (dis1 + dis2) / 2.0;
      EpiErr.emplace_back(dis);
      // 定义卡方分布变量
      boost::math::chi_squared_distribution<> dist(2);
      epiErr.emplace_back(1-boost::math::cdf(dist, dis * dis));
    }
    //    });
  }

  void Frame::EpiloarWithFundMatrix(const cv::Mat &ImGrayPre,
                                    const cv::Mat &ImGray)
  {
    Curpoint.clear();
    Prepoint.clear();
    epiErr.clear();
    state.clear();
    status.clear();
    Err.clear();
    for (auto it : mvKeys)
      Curpoint.emplace_back(it.pt);
    cv::calcOpticalFlowPyrLK(ImGray, ImGrayPre, Curpoint, Prepoint, state, Err,
                             cv::Size(21, 21), 3,cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), 0,1e-4);
    std::vector<cv::Point2f> PrepointT = Prepoint;
    std::vector<cv::Point2f> CurpointT = Curpoint;
    std::vector<float> flowErr = Err;
    sort(flowErr.begin(), flowErr.end());
    float th = flowErr[0.5 * flowErr.size()];
    std::vector<bool> isErase(Curpoint.size(), true);
    for (auto err = Err.begin(); err != Err.end(); ++err)
    {
      int i = err - Err.begin();
      if ((*err) > th)
      {
        isErase[i] = false;
      }
    }
    reduceVector(Curpoint, isErase);
    reduceVector(Prepoint, isErase);
        std::vector<cv::Point2f> Curpoint_static;
    std::vector<cv::Point2f> Prepoint_static;
    for (size_t i = 0; i < Curpoint.size(); i++)
    {
        if (!state[i])
            continue; // 跟踪失败

        const cv::Point2f &pt = Curpoint[i];
        // if (pt.x < 0 || pt.y < 0 || pt.x >= dynaMask.cols || pt.y >= dynaMask.rows)
        //     continue;
        if(dynaMask.at<uchar>(pt)==255)
        continue;
        // uchar mask_val = dynaMask.at<uchar>(cvRound(pt.y), cvRound(pt.x));
        Curpoint_static.push_back(Curpoint[i]);
        Prepoint_static.push_back(Prepoint[i]);
    }
    int N = Curpoint_static.size();
    if (N > 8)
    {
      cv::Mat FundMatrix;
      if (Curpoint.size() < 15)
      {
        FundMatrix = cv::findFundamentalMat(Curpoint_static, Prepoint_static, cv::FM_RANSAC,
                                            0.1, 0.99, status);
        CalcEpiDistChi2(mvKeys, PrepointT, FundMatrix, state);
      }
      else
      {
        FundMatrix = cv::findFundamentalMat(Curpoint, Prepoint, cv::FM_RANSAC,
                                            0.1, 0.99, status);
        CalcEpiDistChi2(mvKeys, PrepointT, FundMatrix, state);
      }
    }
    cv::Mat vis;
cv::cvtColor(ImGray, vis, cv::COLOR_GRAY2BGR);

// 绘制光流
for (size_t i = 0; i < Curpoint.size(); i++)
{
    if (!state[i]) continue; // 光流跟踪失败

    cv::Point2f pt1 = Curpoint[i];   // 当前帧点
    cv::Point2f pt2 = Prepoint[i];   // 前一帧点

    // 画箭头
    cv::arrowedLine(vis, pt1, pt2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    // 可选：画点
    cv::circle(vis, pt1, 2, cv::Scalar(0, 0, 255), -1);
}

// 显示
cv::imshow("Optical Flow", vis);
cv::waitKey(1);
  }

  bool isInBox(cv::KeyPoint &kp, cv::Rect tmpRect, bool extend = false)
  {
    if (!extend)
    {
      if (kp.pt.x > tmpRect.tl().x - 10 && kp.pt.y > tmpRect.tl().y - 10 &&
          kp.pt.x < tmpRect.br().x + 10 && kp.pt.y < tmpRect.br().y + 10)
      {
        return true;
      }
    }
    else
    {
      if (kp.pt.x > tmpRect.tl().x - 0.2 * tmpRect.width &&
          kp.pt.y > tmpRect.tl().y - 10 &&
          kp.pt.x < tmpRect.br().x + 0.2 * tmpRect.width &&
          kp.pt.y < tmpRect.br().y + 10)
      {
        return true;
      }
    }
    return false;
  }

  void Frame::CheckObjectStatus(std::vector<double> epiErr,
                                std::vector<SegObject *> &area)
  {
    
    if (epiErr.empty())
      return;

    if (area.empty())
    {
      // std::vector<cv::KeyPoint> _mvKeys;
      // cv::Mat _mDescriptors;
      // for (int i = 0; i < mvKeys.size(); i++)
      // {
      //   // 概率 <= 0.95 的点当作静态点保留
      //   if (epiErr[i] >= 0.9)
      //   {
      //     _mvKeys.emplace_back(mvKeys[i]);
      //     _mDescriptors.push_back(mDescriptors.row(i));
      //   }
      // }
      // mvKeys.swap(_mvKeys);
      // mDescriptors = _mDescriptors.clone();
      return;
    }
    else
    {
for (int i = 0; i < mvKeys.size(); i++)
{
    cv::Point2f pt = mvKeys[i].pt;

    for (int j = 0; j < area.size(); j++)
    {
        cv::Rect rect = area[j]->obj_recent.rect;   // 物体框
        cv::Mat boxMask = area[j]->obj_recent.boxMask; // 对应的局部掩膜

        // 检查点是否在物体矩形内
        if (!rect.contains(pt))
            continue;

        // 转换到局部掩膜坐标
        int localX = cvRound(pt.x - rect.x);
        int localY = cvRound(pt.y - rect.y);

        // 边界检查
        if (localX < 0 || localY < 0 || localX >= boxMask.cols || localY >= boxMask.rows)
            continue;

        // 掩膜前景判断
        if (boxMask.at<uchar>(localY, localX) > 0)
        {
            area[j]->KeyPointsInBox.emplace_back(i, mvKeys[i]);
            area[j]->epiErr.emplace_back(epiErr[i]);
        }
    }
}


      // 计算每个物体的静态密度
      for (auto &obj : area)
      {
        int M = obj->epiErr.size();
        if (M == 0)
          continue;

        double sumProb = 0.0;
        for (double p : obj->epiErr)
        {
          sumProb += p; // epiErr[i] 是概率值
        }
        double proStatic = sumProb / static_cast<double>(M);
        obj->pro_static = proStatic;
        // cout<<obj->pro_static<<endl;
        // 判断是否移动
        if(obj->label == 0)
        obj->is_moving = (proStatic <0.9); // 静态密度小于 50% 就当作动
        if(obj->label   != 0)
        obj->is_moving = (proStatic <0.65); // 静态密度小于 50% 就当作动
        if(!obj->is_moving || obj->label !=0)
        {
            dynaMask(obj->obj_recent.rect).setTo(135, obj->obj_recent.boxMask);
        }
        else
        {
            dynaMask(obj->obj_recent.rect).setTo(255, obj->obj_recent.boxMask);
        }
      }
      cv::imshow("dynamask",dynaMask);
    }
  }

void Frame::RemoveMovingKeyPoints(const cv::Mat &ImGray, const cv::Mat &ImDepth,
                                  std::vector<SegObject *> &area)
{
    if (area.empty() || epiErr.empty())
        return;
    cv::Mat maskDilated;
    int dilateSize = 10;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(2*dilateSize+1, 2*dilateSize+1));
    cv::dilate(dynaMask, maskDilated, kernel);
    // cv::imshow("mask",dynaMask);
    std::vector<double> sortedProb = epiErr;
    std::sort(sortedProb.begin(), sortedProb.end());
    double cutoffProb = sortedProb[static_cast<int>(0.05 * sortedProb.size())];
    
    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;
    
    for (size_t idx = 0; idx < mvKeys.size(); ++idx)
    {
        const cv::KeyPoint &kp = mvKeys[idx];
        bool keep = true;

        for (auto &obj : area)
        {

            if (!obj->is_moving)
                continue;

            // 点在物体框内
            if (obj->obj_recent.rect.contains(kp.pt))
            {
                // 点在膨胀后的掩膜中
                if (maskDilated.at<uchar>(cv::Point(kp.pt)) == 255)
                {
                    keep = false;
                    break;  
                }
                if(EpiErr[idx] > 0.5)
                {
                    keep = false;
                    break;
                }
                // cout<<epiErr[idx]<<endl;
            }
        }
        if (EpiErr[idx] > 1)
            keep = false;

        if (keep)
        {
            _mvKeys.emplace_back(kp);
            _mDescriptors.push_back(mDescriptors.row(idx));
        }
    }

    mvKeys.swap(_mvKeys);
    mDescriptors = _mDescriptors.clone();
}


  void Frame::RemoveMovingLine(const cv::Mat &ImGray, const cv::Mat &ImDepth,
                               std::vector<SegObject *> &area)
  {
    if (area.empty() || epiErr.empty())
    {
      return;
    }

    std::vector<cv::line_descriptor_c::KeyLine> _mvKeyLines;
    cv::Mat _mLineDescriptors;

    for (size_t i = 0; i < mvKeyLines.size(); ++i)
    {
      const auto &kl = mvKeyLines[i];
      bool to_keep = true;

      // 线段的起点和终点
      cv::KeyPoint kp1(kl.getStartPoint(), 1.f); // 1.f 是 size，随便给个默认值
      cv::KeyPoint kp2(kl.getEndPoint(), 1.f);
      for (SegObject *obj : area)
      {
        if (obj == nullptr)
          continue;

        // 如果该目标是动态物体
        if (obj->is_moving || (obj->label == 0 && obj->pro_static < 0.96))
        {
          // 如果线段的任意端点在目标框中，则认为是动态线段
          if (isInBox(kp1, obj->obj_recent.rect) ||
              isInBox(kp2, obj->obj_recent.rect))
          {
            to_keep = false;
            break;
          }
        }
      }

      if (to_keep)
      {
        _mvKeyLines.push_back(kl);
        _mLineDescriptors.push_back(mLineDescriptors.row(i));
      }
    }

    mvKeyLines.swap(_mvKeyLines);
    mLineDescriptors = _mLineDescriptors.clone();
  }

  bool Frame::PointFilter(cv::Mat imGray, std::vector<SegObject *> &dynamicArea,
                          cv::Mat imDepth)
  {
    
    if (!imGrayPre.empty())
    {
      EpiloarWithFundMatrix(imGrayPre, imGray);
      CheckObjectStatus(epiErr, dynamicArea);
      RemoveMovingKeyPoints(imGray, imDepth, dynamicArea);
    }

    N = mvKeys.size();
    if (mvKeys.empty())
      return false;
    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPointPtr>(N, static_cast<MapPointPtr>(NULL));

    // mmProjectPoints.clear();
    // mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N, false);
    if (mpLineExtractorLeft)
    {
      RemoveMovingLine(imGray, imDepth, dynamicArea);
      Nlines = mvKeyLines.size();

      if (mvKeyLines.empty())
      {
        std::cout << "frame " << mnId << " no lines dectected!" << std::endl;
      }
      else
      {
        UndistortKeyLines();

        // TICKTRACK("ComputeStereoLinesFromRGBD");
        ComputeStereoLinesFromRGBD(imDepth);
        // TOCKTRACK("ComputeStereoLinesFromRGBD");
      }

      mvpMapLines = vector<MapLinePtr>(Nlines, static_cast<MapLinePtr>(NULL));
      mvbLineOutlier = vector<bool>(Nlines, false);
      mvuNumLinePosOptFailures = vector<unsigned int>(Nlines, 0);

      // mmProjectLines.clear();// = map<long unsigned int, LineEndPoints>(N,
      // static_cast<LineEndPoints>(NULL)); mmMatchedLinesInImage.clear();
    }

    TOCKTRACK("ExtractFeatures");

    mpMutexImu = new std::mutex();

    // Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
    imGrayPre = imGray;
    return true;
  }

  /// < MONOCULAR
  Frame::Frame(const cv::Mat &imGray, const double &timeStamp,
               ORBextractor *extractor, ORBVocabulary *voc,
               GeometricCamera *pCamera, cv::Mat &distCoef, const float &bf,
               const float &thDepth, Frame *pPrevF, const IMU::Calib &ImuCalib)
      : mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractor),
        mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
        mTimeStamp(timeStamp), mK(pCamera->toLinearK()),
        mK_(pCamera->toLinearK_()), mDistCoef(distCoef.clone()), mbf(bf),
        mThDepth(thDepth), mImuCalib(ImuCalib), mpImuPreintegrated(NULL),
        mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL),
        mpReferenceKF(static_cast<KeyFramePtr>(NULL)), mbIsSet(false),
        mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(nullptr),
        mbHasPose(false), mbHasVelocity(false)
  {
    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB =
        std::chrono::steady_clock::now();
#endif
    ExtractORB(0, imGray, 0, 1000);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB =
        std::chrono::steady_clock::now();

    mTimeORB_Ext =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            time_EndExtORB - time_StartExtORB)
            .count();
#endif

    N = mvKeys.size();
    if (mvKeys.empty())
      return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N, -1);
    mvDepth = vector<float>(N, -1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPointPtr>(N, static_cast<MapPointPtr>(NULL));

    // mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N,
    // static_cast<cv::Point2f>(NULL)); mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N, false);

    // This is done only for the first Frame (or after a change in the
    // calibration)
    if (mbInitialComputations)
    {
      ComputeImageBounds(imGray);

      mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) /
                              static_cast<float>(mnMaxX - mnMinX);
      mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) /
                               static_cast<float>(mnMaxY - mnMinY);

      mfLineGridElementThetaInv = static_cast<float>(LINE_THETA_GRID_ROWS) /
                                  static_cast<float>(LINE_THETA_SPAN);
      mfLineGridElementDInv =
          static_cast<float>(LINE_D_GRID_COLS) / (2.0f * mnMaxDiag);

      fx = mK.at<float>(
          0, 0); // static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
      fy = mK.at<float>(
          1, 1); // static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
      cx = mK.at<float>(
          0, 2); // static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
      cy = mK.at<float>(
          1, 2); // static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
      invfx = 1.0f / fx;
      invfy = 1.0f / fy;

      mbInitialComputations = false;
    }

    mb = mbf / fx;
    mbfInv = 1.f / mbf;

    // Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if (pPrevF)
    {
      if (pPrevF->HasVelocity())
      {
        SetVelocity(pPrevF->GetVelocity());
      }
    }
    else
    {
      mVw.setZero();
    }

    if (mpLineExtractorLeft)
    {
      mvuRightLineStart = vector<float>(Nlines, -1);
      mvDepthLineStart = vector<float>(Nlines, -1);
      mvuRightLineEnd = vector<float>(Nlines, -1);
      mvDepthLineEnd = vector<float>(Nlines, -1);
    }

    mpMutexImu = new std::mutex();
  }

  void Frame::AssignFeaturesToGrid()
  {
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS * FRAME_GRID_ROWS;

    int nReserve = 0.5f * N / (nCells);

    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
      for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
      {
        mGrid[i][j].reserve(nReserve);
        if (Nleft != -1)
        {
          mGridRight[i][j].reserve(nReserve);
        }
      }

    for (int i = 0; i < N; i++)
    {
      const cv::KeyPoint &kp =
          (Nleft == -1) ? mvKeysUn[i]
          : (i < Nleft) ? mvKeys[i]
                        : mvKeysRight[i - Nleft];

      int nGridPosX, nGridPosY;
      if (PosInGrid(kp, nGridPosX, nGridPosY))
      {
        if (Nleft == -1 || i < Nleft)
          mGrid[nGridPosX][nGridPosY].push_back(i);
        else
          mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
      }
    }

    if (mpLineExtractorLeft)
    {
      int nReserve = 0.5f * Nlines / (LINE_D_GRID_COLS * LINE_THETA_GRID_ROWS);
      for (unsigned int i = 0; i < LINE_D_GRID_COLS; i++)
        for (unsigned int j = 0; j < LINE_THETA_GRID_ROWS; j++)
        {
          mLineGrid[i][j].reserve(nReserve);
          if (NlinesLeft != -1)
          {
            mLineGridRight[i][j].reserve(nReserve);
          }
        }

      for (int i = 0; i < Nlines; i++)
      {
        // const cv::line_descriptor_c::KeyLine& kl = mvKeyLinesUn[i];
        const cv::line_descriptor_c::KeyLine &kl =
            (NlinesLeft == -1)
                ? mvKeyLinesUn[i]
            : (i < NlinesLeft) ? mvKeyLinesUn[i]
                               : mvKeyLinesRightUn[i - NlinesLeft];

        int nGridPosX, nGridPosY;
        if (PosLineInGrid(kl, nGridPosX, nGridPosY))
        {
          if (NlinesLeft == -1 || i < NlinesLeft)
          {
            mLineGrid[nGridPosX][nGridPosY].push_back(i);
          }
          else
          {
            mLineGridRight[nGridPosX][nGridPosY].push_back(i - NlinesLeft);
          }
        }
      }

#if LOG_ASSIGN_FEATURES
      logger << "================================================================"
             << std::endl;
      logger << "frame: " << mnId << std::endl;
      for (int nGridPosX = 0; nGridPosX < LINE_D_GRID_COLS; nGridPosX++)
        for (int nGridPosY = 0; nGridPosY < LINE_THETA_GRID_ROWS; nGridPosY++)
        {
          std::vector<std::size_t> &vec = mLineGrid[nGridPosX][nGridPosY];
          for (int kk = 0; kk < vec.size(); kk++)
          {
            const cv::line_descriptor_c::KeyLine &kl = mvKeyLinesUn[vec[kk]];
            const float &xs = kl.startPointX;
            const float &ys = kl.startPointY;
            const float &xe = kl.endPointX;
            const float &ye = kl.endPointY;

            Line2DRepresentation lineRepresentation;
            Geom2DUtils::GetLine2dRepresentation(xs, ys, xe, ye,
                                                 lineRepresentation);

            logger << "line " << vec[kk]
                   << ", theta: " << lineRepresentation.theta * 180 / M_PI
                   << ", d: " << lineRepresentation.d << ", posX: " << nGridPosX
                   << ", posY: " << nGridPosY << std::endl;
          }
        }
#endif
    }
  }

  void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0,
                         const int x1)
  {
    
    vector<int> vLapping = {x0, x1};
    if (flag == 0)
      monoLeft =
          (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors, vLapping);
    else
      monoRight = (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight,
                                         mDescriptorsRight, vLapping);
  }

  void Frame::ExtractLSD(int flag, const cv::Mat &im)
  {
    if (flag == 0)
    {
      if (mpLineExtractorLeft)
      {
        (*mpLineExtractorLeft)(im, mvKeyLines, mLineDescriptors);
        monoLinesLeft = 0; // TODO Luigi
      }
    }
    else
    {
      if (mpLineExtractorRight)
      {
        (*mpLineExtractorRight)(im, mvKeyLinesRight, mLineDescriptorsRight);
        monoLinesRight = 0; // TODO Luigi
      }
    }
  }

  static void clonePyramid(const std::vector<cv::Mat> &imgsIn,
                           std::vector<cv::Mat> &imgsClone)
  {
    imgsClone.resize(imgsIn.size());
    for (size_t ii = 0; ii < imgsIn.size(); ii++)
      imgsClone[ii] = imgsIn[ii].clone();
  }

  void Frame::PrecomputeGaussianPyramid(int flag, const cv::Mat &im)
  {
    if (flag == 0)
    {
      mpORBextractorLeft->PrecomputeGaussianPyramid(im);
#ifndef USE_CUDA
#if USE_UNFILTERED_PYRAMID_FOR_LINES
      if (mpLineExtractorLeft)
        mpLineExtractorLeft->SetGaussianPyramid(
            mpORBextractorLeft->mvImagePyramid, mpLineExtractorLeft->GetLevels(),
            mpORBextractorLeft->GetScaleFactor());
#else
      if (mpLineExtractorLeft)
        mpLineExtractorLeft->SetGaussianPyramid(
            mpORBextractorLeft->mvImagePyramidFiltered,
            mpLineExtractorLeft->GetLevels(),
            mpORBextractorLeft->GetScaleFactor());
#endif
#endif
    }
    else
    {
      mpORBextractorRight->PrecomputeGaussianPyramid(im);
#ifndef USE_CUDA
#if USE_UNFILTERED_PYRAMID_FOR_LINES
      if (mpLineExtractorRight)
        mpLineExtractorRight->SetGaussianPyramid(
            mpORBextractorRight->mvImagePyramid,
            mpLineExtractorRight->GetLevels(),
            mpORBextractorRight->GetScaleFactor());
#else
      if (mpLineExtractorRight)
        mpLineExtractorRight->SetGaussianPyramid(
            mpORBextractorRight->mvImagePyramidFiltered,
            mpLineExtractorRight->GetLevels(),
            mpORBextractorRight->GetScaleFactor());
#endif
#endif
    }
  }

  bool Frame::isSet() const { return mbIsSet; }

  void Frame::SetPose(const Sophus::SE3<float> &Tcw)
  {
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
  }

  void Frame::SetNewBias(const IMU::Bias &b)
  {
    mImuBias = b;
    if (mpImuPreintegrated)
      mpImuPreintegrated->SetNewBias(b);
  }

  void Frame::SetVelocity(Eigen::Vector3f Vwb)
  {
    mVw = Vwb;
    mbHasVelocity = true;
  }

  Eigen::Vector3f Frame::GetVelocity() const { return mVw; }

  void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb,
                                 const Eigen::Vector3f &twb,
                                 const Eigen::Vector3f &Vwb)
  {
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
  }

  void Frame::UpdatePoseMatrices()
  {
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();

    // fovCw = Ow + Twc.rowRange(0,3).col(2) * skFovCenterDistance;
    fovCw = mOw + mRwc.col(2) * mMedianDepth;
  }

  Eigen::Matrix<float, 3, 1> Frame::GetImuPosition() const
  {
    return mRwc * mImuCalib.mTcb.translation() + mOw;
  }

  Eigen::Matrix<float, 3, 3> Frame::GetImuRotation()
  {
    return mRwc * mImuCalib.mTcb.rotationMatrix();
  }

  Sophus::SE3<float> Frame::GetImuPose()
  {
    return mTcw.inverse() * mImuCalib.mTcb;
  }

  Sophus::SE3f Frame::GetRelativePoseTrl() { return mTrl; }

  Sophus::SE3f Frame::GetRelativePoseTlr() { return mTlr; }

  Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation()
  {
    return mTlr.rotationMatrix();
  }

  Eigen::Vector3f Frame::GetRelativePoseTlr_translation()
  {
    return mTlr.translation();
  }

  bool Frame::isInFrustum(MapPointPtr &pMP, float viewingCosLimit)
  {
    if (Nleft == -1)
    {
      pMP->mbTrackInView = false;
      pMP->mTrackProjX = -1;
      pMP->mTrackProjY = -1;

      // 3D in absolute coordinates
      Eigen::Matrix<float, 3, 1> P = pMP->GetWorldPos();

      // 3D in camera coordinates
      const Eigen::Matrix<float, 3, 1> Pc = mRcw * P + mtcw;
      const float Pc_dist = Pc.norm();

      // Check positive depth
      const float &PcZ = Pc(2);
      const float invz = 1.0f / PcZ;
      if (PcZ < 0.0f)
        return false;

      const Eigen::Vector2f uv = mpCamera->project(Pc);

      if (uv(0) < mnMinX || uv(0) > mnMaxX)
        return false;
      if (uv(1) < mnMinY || uv(1) > mnMaxY)
        return false;

      pMP->mTrackProjX = uv(0);
      pMP->mTrackProjY = uv(1);

      // Check distance is in the scale invariance region of the MapPoint
      const float maxDistance = pMP->GetMaxDistanceInvariance();
      const float minDistance = pMP->GetMinDistanceInvariance();
      const Eigen::Vector3f PO = P - mOw;
      const float dist = PO.norm();

      if (dist < minDistance || dist > maxDistance)
        return false;

      // Check viewing angle
      Eigen::Vector3f Pn = pMP->GetNormal();

      const float viewCos = PO.dot(Pn) / dist;

      if (viewCos < viewingCosLimit)
        return false;

      // Predict scale in the image
      const int nPredictedLevel = pMP->PredictScale(dist, this);

      // Data used by the tracking
      pMP->mbTrackInView = true;
      pMP->mTrackProjX = uv(0);
      pMP->mTrackProjXR = uv(0) - mbf * invz;

      pMP->mTrackDepth = Pc_dist;

      pMP->mTrackProjY = uv(1);
      pMP->mnTrackScaleLevel = nPredictedLevel;
      pMP->mTrackViewCos = viewCos;

      return true;
    }
    else
    {
      pMP->mbTrackInView = false;
      pMP->mbTrackInViewR = false;
      pMP->mnTrackScaleLevel = -1;
      pMP->mnTrackScaleLevelR = -1;

      pMP->mbTrackInView = isInFrustumChecks(pMP, viewingCosLimit);
      pMP->mbTrackInViewR = isInFrustumChecks(pMP, viewingCosLimit, true);

      return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
  }

  bool Frame::isInFrustum(MapLinePtr &pML, float viewingCosLimit)
  {
    /// < NOTE: Luigi add management of Right lines

    if (NlinesLeft == -1)
    {
      // here we have pinhole cameras (no distortion model is used when projecting
      // or back-projecting)

      pML->mbTrackInView = false;
      pML->mbTrackInViewR = false;
      pML->mTrackProjStartX = -1;
      pML->mTrackProjStartY = -1;
      pML->mTrackProjStartXR = -1;
      pML->mTrackProjStartYR = -1;
      pML->mTrackProjEndX = -1;
      pML->mTrackProjEndY = -1;
      pML->mTrackProjEndXR = -1;
      pML->mTrackProjEndYR = -1;
      // pML->mTrackProjMiddleX = -1; pML->mTrackProjMiddleY = -1;
      // pML->mTrackProjMiddleXR = -1; pML->mTrackProjMiddleYR = -1;

      // 3D in absolute coordinates
      Eigen::Vector3f p3DStart, p3DEnd;
      pML->GetWorldEndPoints(p3DStart, p3DEnd);

      // consider the middle point
      const Eigen::Vector3f p3DMiddle = 0.5 * (p3DStart + p3DEnd);

      // 3D in camera coordinates
      const Eigen::Vector3f p3DMc = mRcw * p3DMiddle + mtcw;
      // const float &pMcX = p3DMc(0);
      // const float &pMcY = p3DMc(1);
      const float &pMcZ = p3DMc(2);

      // Check positive depth
      if (pMcZ < 0.0f)
        return false;

      // Project in image and check it is not outside
      const Eigen::Vector2f uvM =
          mpCamera->project(p3DMc); // pinhole projection with no distortion here

      if (uvM(0) < mnMinX || uvM(0) > mnMaxX)
        return false;
      if (uvM(1) < mnMinY || uvM(1) > mnMaxY)
        return false;

      // const float invMz = 1.0f/pMcZ;
      // pML->mTrackProjMiddleX = uvM(0); //fx*pMcX*invMz+cx;
      // pML->mTrackProjMiddleY = uvM(1); //fy*pMcY*invMz+cy;
      // pML->mTrackProjMiddleXR = pML->mTrackProjMiddleX - mbf*invMz;

      // Check distance is in the scale invariance region of the MapPoint
      const float maxDistance = pML->GetMaxDistanceInvariance();
      const float minDistance = pML->GetMinDistanceInvariance();
      const Eigen::Vector3f PO = p3DMiddle - mOw;
      const float dist = PO.norm();

      if (dist < minDistance || dist > maxDistance)
        return false;

      // Check viewing angle
      Eigen::Vector3f Pn = pML->GetNormal();

      const float viewCos = PO.dot(Pn) / dist;

      if (viewCos < viewingCosLimit)
        return false;

      // Now compute other data used by the tracking

      // 3D in camera coordinates
      const Eigen::Vector3f p3DSc = mRcw * p3DStart + mtcw;
      // const float &p3DScX = p3DSc(0);
      // const float &p3DScY = p3DSc(1);
      const float &p3DScZ = p3DSc(2);
      const Eigen::Vector2f uvS =
          mpCamera->project(p3DSc); // pinhole projection with no distortion here

      const Eigen::Vector3f p3DEc = mRcw * p3DEnd + mtcw;
      // const float &p3DEcX = p3DEc(0);
      // const float &p3DEcY = p3DEc(1);
      const float &p3DEcZ = p3DEc(2);
      const Eigen::Vector2f uvE =
          mpCamera->project(p3DEc); // pinhole projection with no distortion here

      // Project in image
      const float invSz = 1.0f / p3DScZ;
      pML->mTrackProjStartX = uvS(0); // fx*p3DScX*invSz+cx;
      pML->mTrackProjStartY = uvS(1); // fy*p3DScY*invSz+cy;
      pML->mTrackStartDepth = p3DScZ;
      pML->mTrackProjStartXR = pML->mTrackProjStartX - mbf * invSz;

      // Project in image
      const float invEz = 1.0f / p3DEcZ;
      pML->mTrackProjEndX = uvE(0); // fx*p3DEcX*invEz+cx;
      pML->mTrackProjEndY = uvE(1); // fy*p3DEcY*invEz+cy;
      pML->mTrackEndDepth = p3DEcZ;
      pML->mTrackProjEndXR = pML->mTrackProjEndX - mbf * invEz;

      /*const float deltaDepth = fabs(p3DScZ-p3DEcZ);
      if(deltaDepth > kDeltaZForCheckingViewZAngleMin)
      {
          const float tgViewZAngle = sqrt( Utils::Pow2(p3DScX-p3DEcX) +
      Utils::Pow2(p3DScY-p3DEcY) )/deltaDepth; if(tgViewZAngle<kTgViewZAngleMin)
              return false;
      }*/

      // Predict scale in the image
      const int nPredictedLevel = pML->PredictScale(dist, this);

      pML->mbTrackInView = true;
      pML->mnTrackScaleLevel = nPredictedLevel;
      pML->mTrackViewCos = viewCos;

      return true;
    }
    else
    {
      // here we have fisheye cameras

      pML->mbTrackInView = false;
      pML->mbTrackInViewR = false;
      pML->mnTrackScaleLevel = -1;
      pML->mnTrackScaleLevelR = -1;

      pML->mbTrackInView = isInFrustumChecks(pML, viewingCosLimit);
      pML->mbTrackInViewR = isInFrustumChecks(pML, viewingCosLimit, true);

      return pML->mbTrackInView || pML->mbTrackInViewR;
    }
  }

  bool Frame::ProjectPointDistort(MapPointPtr pMP, cv::Point2f &kp, float &u,
                                  float &v)
  {
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY = Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if (PcZ < 0.0f)
    {
      cout << "Negative depth: " << PcZ << endl;
      return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    u = fx * PcX * invz + cx;
    v = fy * PcY * invz + cy;

    if (u < mnMinX || u > mnMaxX)
      return false;
    if (v < mnMinY || v > mnMaxY)
      return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if (mDistCoef.total() == 5)
    {
      k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
  }

  Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
  {
    return mRcw * pCw + mtcw;
  }

  // TODO: Luigi optimization
  // bring (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy]; outside the loop!
  vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y,
                                          const float &r, const int minLevel,
                                          const int maxLevel,
                                          const bool bRight) const
  {
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX =
        max(0, (int)floor((x - mnMinX - factorX) * mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
    {
      return vIndices;
    }

    const int nMaxCellX =
        min((int)FRAME_GRID_COLS - 1,
            (int)ceil((x - mnMinX + factorX) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
    {
      return vIndices;
    }

    const int nMinCellY =
        max(0, (int)floor((y - mnMinY - factorY) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
    {
      return vIndices;
    }

    const int nMaxCellY =
        min((int)FRAME_GRID_ROWS - 1,
            (int)ceil((y - mnMinY + factorY) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
    {
      return vIndices;
    }

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

    const auto &grid = (!bRight) ? mGrid : mGridRight;
    const auto &keys =
        (Nleft == -1) ? mvKeysUn : (!bRight) ? mvKeys
                                             : mvKeysRight;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
      for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
      {
        // const vector<size_t>& vCell = (!bRight) ? mGrid[ix][iy] :
        // mGridRight[ix][iy];
        const vector<size_t> &vCell = grid[ix][iy];
        if (vCell.empty())
          continue;

        for (size_t j = 0, jend = vCell.size(); j < jend; j++)
        {
          // const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
          //                                          : (!bRight) ?
          //                                          mvKeys[vCell[j]]
          //                                                      :
          //                                                      mvKeysRight[vCell[j]];
          const cv::KeyPoint &kpUn = keys[vCell[j]];
          if (bCheckLevels)
          {
            if (kpUn.octave < minLevel)
              continue;
            // if(maxLevel>=0)
            if (kpUn.octave > maxLevel)
              continue;
          }

          const float distx = kpUn.pt.x - x;
          const float disty = kpUn.pt.y - y;

          if (fabs(distx) < factorX && fabs(disty) < factorY)
            vIndices.push_back(vCell[j]);
        }
      }
    }

    return vIndices;
  }

  bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
  {
    /// < N.B: shouldn't we use floor here? NO! since the features are searched
    /// with radius
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

    // Keypoint's coordinates are undistorted, which could cause to go out of the
    // image
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 ||
        posY >= FRAME_GRID_ROWS)
      return false;

    return true;
  }

  vector<size_t> Frame::GetLineFeaturesInArea(const float &xs, const float &ys,
                                              const float &xe, const float &ye,
                                              const float &dtheta,
                                              const float &dd, const int minLevel,
                                              const int maxLevel,
                                              const bool bRight) const
  {
    Line2DRepresentation lineRepresentation;
    Geom2DUtils::GetLine2dRepresentation(xs, ys, xe, ye, lineRepresentation);
    return GetLineFeaturesInArea(lineRepresentation, dtheta, dd, minLevel,
                                 maxLevel, bRight);
  }

  vector<size_t>
  Frame::GetLineFeaturesInArea(const Line2DRepresentation &lineRepresentation,
                               const float &dtheta, const float &dd,
                               const int minLevel, const int maxLevel,
                               const bool bRight) const
  {
    vector<size_t> vIndices;
    vIndices.reserve(Nlines);

    const bool bCheckLevels = (minLevel > 0) || (maxLevel < kMaxInt);

    const float thetaMin = lineRepresentation.theta - dtheta;
    const float thetaMax = lineRepresentation.theta + dtheta;

    if (fabs(thetaMin - thetaMax) > M_PI)
    {
      std::cout << "Frame::GetLineFeaturesInArea() - ERROR - you are searching "
                   "over the full theta interval!"
                << std::endl;
      quick_exit(-1);
      return vIndices;
    }

    const float dMin = lineRepresentation.d - dd;
    const float dMax = lineRepresentation.d + dd;

    GetLineFeaturesInArea(thetaMin, thetaMax, dMin, dMax, bCheckLevels, minLevel,
                          maxLevel, vIndices, bRight);

    return vIndices;
  }

  void Frame::GetLineFeaturesInArea(const float thetaMin, const float thetaMax,
                                    const float dMin, const float dMax,
                                    const bool bCheckLevels, const int minLevel,
                                    const int maxLevel, vector<size_t> &vIndices,
                                    const bool bRight) const
  {
    // std::cout << "GetLineFeaturesInArea - thetaMin: " << thetaMin << ",
    // thetaMax: " << thetaMax << ", dMin: " << dMin << ", dMax: " << dMax <<
    // std::endl;

    if (thetaMin < -M_PI_2)
    {
      // let's split the search interval in two intervals (we are searching over a
      // manifold here) 1) bring theta angles within [-pi/2, pi/2] 2) if you wrap
      // around +-pi/2 (i.e. add +-pi) then you have to invert the sign of d

      // std::cout << "thetaMin: " << thetaMin << " < -M_PI_2" << std::endl;
      // std::cout << "start split " << std::endl;

      const float thetaMin1 = thetaMin + M_PI;
      const float thetaMax1 = M_PI_2 - std::numeric_limits<float>::epsilon();
      const float dMin1 = -dMax;
      const float dMax1 = -dMin;
      GetLineFeaturesInArea(thetaMin1, thetaMax1, dMin1, dMax1, bCheckLevels,
                            minLevel, maxLevel, vIndices, bRight);

      const float thetaMin2 = -M_PI_2 + std::numeric_limits<float>::epsilon();
      const float thetaMax2 = thetaMax;
      const float dMin2 = dMin;
      const float dMax2 = dMax;
      GetLineFeaturesInArea(thetaMin2, thetaMax2, dMin2, dMax2, bCheckLevels,
                            minLevel, maxLevel, vIndices, bRight);

      // std::cout << "end split " << std::endl;

      return; // < EXIT
    }

    if (thetaMax > M_PI_2)
    {
      // let's split the search interval in two intervals (we are searching over a
      // manifold here) 1) bring theta angles within [-pi/2, pi/2] 2) if you wrap
      // around +-pi/2 (i.e. add +-pi) then you have to invert the sign of d

      // std::cout << "thetaMax: " << thetaMax <<  " > M_PI_2" << std::endl;
      // std::cout << "start split " << std::endl;

      const float thetaMin1 = -M_PI_2 + std::numeric_limits<float>::epsilon();
      const float thetaMax1 = thetaMax - M_PI;
      const float dMin1 = -dMax;
      const float dMax1 = -dMin;
      GetLineFeaturesInArea(thetaMin1, thetaMax1, dMin1, dMax1, bCheckLevels,
                            minLevel, maxLevel, vIndices, bRight);

      const float thetaMin2 = thetaMin;
      const float thetaMax2 = M_PI_2 - std::numeric_limits<float>::epsilon();
      const float dMin2 = dMin;
      const float dMax2 = dMax;
      GetLineFeaturesInArea(thetaMin2, thetaMax2, dMin2, dMax2, bCheckLevels,
                            minLevel, maxLevel, vIndices, bRight);

      // std::cout << "end split " << std::endl;

      return; // < EXIT
    }

    // const int nMinCellThetaRow = (int)floor((thetaMin
    // -LINE_THETA_MIN)*mfLineGridElementThetaInv); const int nMaxCellThetaRow =
    // (int)floor((thetaMax -LINE_THETA_MIN)*mfLineGridElementThetaInv);

    const int nMinCellThetaRow = std::max(
        0, (int)floor((thetaMin - LINE_THETA_MIN) * mfLineGridElementThetaInv));
    if (nMinCellThetaRow >= LINE_THETA_GRID_ROWS)
    {
      return;
    }
    const int nMaxCellThetaRow = std::min(
        LINE_THETA_GRID_ROWS - 1,
        (int)floor((thetaMax - LINE_THETA_MIN) * mfLineGridElementThetaInv));
    if (nMaxCellThetaRow < 0)
    {
      return;
    }

    const int nMinCellDCol = std::max(
        0, (int)floor((dMin + mnMaxDiag) *
                      mfLineGridElementDInv)); // + mnMaxDiag = - mnMinDiag
    if (nMinCellDCol >= LINE_D_GRID_COLS)
    {
      return;
    }
    const int nMaxCellDCol =
        std::min(LINE_D_GRID_COLS - 1,
                 (int)floor((dMax + mnMaxDiag) *
                            mfLineGridElementDInv)); // + mnMaxDiag = - mnMinDiag
    if (nMaxCellDCol < 0)
    {
      return;
    }

    const auto &grid = (!bRight) ? mLineGrid : mLineGridRight;
    const auto &keyLines = (NlinesLeft == -1)
                               ? mvKeyLinesUn
                           : (!bRight) ? mvKeyLinesUn
                                       : mvKeyLinesRightUn;

    for (int ix = nMinCellDCol; ix <= nMaxCellDCol; ix++)
    {
      for (int iy = nMinCellThetaRow; iy <= nMaxCellThetaRow; iy++)
      {
        // const int iyW = Utils::Modulus(iy,LINE_THETA_GRID_ROWS);
        const int iyW = iy;
        // const vector<size_t>& vCell = mLineGrid[ix][iyW];
        const vector<size_t> &vCell = grid[ix][iyW];
        if (vCell.empty())
          continue;

#if 0            
            vIndices.insert(vIndices.end(),vCell.begin(),vCell.end());
#else
        for (size_t j = 0, jend = vCell.size(); j < jend; j++)
        {
          // const cv::line_descriptor_c::KeyLine &klUn = mvKeyLinesUn[vCell[j]];
          const cv::line_descriptor_c::KeyLine &klUn = keyLines[vCell[j]];
          if (bCheckLevels)
          {
            if (klUn.octave < minLevel)
              continue;
            if (klUn.octave > maxLevel)
              continue;
          }

          // const float distx = kpUn.pt.x-x;
          // const float disty = kpUn.pt.y-y;

          // if(fabs(distx)<r && fabs(disty)<r)
          //    vIndices.push_back(vCell[j]);

          vIndices.push_back(vCell[j]);
        }
#endif
      }
    }
  }

  bool Frame::PosLineInGrid(const cv::line_descriptor_c::KeyLine &kl,
                            int &posXcol, int &posYrow)
  {
    const float &xs = kl.startPointX;
    const float &ys = kl.startPointY;
    const float &xe = kl.endPointX;
    const float &ye = kl.endPointY;

    Line2DRepresentation lineRepresentation;
    Geom2DUtils::GetLine2dRepresentation(xs, ys, xe, ye, lineRepresentation);

    posYrow = round((lineRepresentation.theta - LINE_THETA_MIN) *
                    mfLineGridElementThetaInv); // row
    posXcol = round((lineRepresentation.d + mnMaxDiag) *
                    mfLineGridElementDInv); // col     +mnMaxDiag=-mnMinDiag

    // Keyline coordinates are undistorted, which could cause to go out of the
    // image
    if (posYrow < 0 || posYrow >= LINE_THETA_GRID_ROWS || posXcol < 0 ||
        posXcol >= LINE_D_GRID_COLS)
      return false;

    return true;
  }

  void Frame::ComputeBoW()
  {
    if (mBowVec.empty())
    {
      vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
      mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
  }

  void Frame::UndistortKeyPoints()
  {
    // mDistCoef == 0.0 means no pinhole distortion
    if (mDistCoef.at<float>(0) == 0.0)
    {
      mvKeysUn = mvKeys;
      return;
    }

    // Fill matrix with points
    cv::Mat mat(N, 2, CV_32F);

    for (int i = 0; i < N; i++)
    {
      mat.at<float>(i, 0) = mvKeys[i].pt.x;
      mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points (we don't undistort right points here since they are not
    // used in tracking)
    mat = mat.reshape(2);
    if (mpCamera->GetType() == GeometricCamera::CAM_PINHOLE)
    {
      cv::undistortPoints(mat, mat, mpCamera->toK(), mDistCoef, cv::Mat(),
                          mpCamera->toLinearK());
    }
    else
    {
      MSG_ASSERT(mpCamera->GetType() == GeometricCamera::CAM_FISHEYE,
                 "Invalid camera type - It should be fisheye!");
      cv::Mat D = (cv::Mat_<float>(4, 1) << mpCamera->getParameter(4),
                   mpCamera->getParameter(5), mpCamera->getParameter(6),
                   mpCamera->getParameter(7));
      cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
      cv::fisheye::undistortPoints(mat, mat, mpCamera->toK(), D, R,
                                   mpCamera->toLinearK());
    }
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for (int i = 0; i < N; i++)
    {
      cv::KeyPoint &kp = mvKeysUn[i];
      kp = mvKeys[i];
      kp.pt.x = mat.at<float>(i, 0);
      kp.pt.y = mat.at<float>(i, 1);
      // mvKeysUn[i]=kp;
    }
  }

  void Frame::UndistortKeyLines()
  {
    constexpr float DEG2RAD = M_PI / 180.0f;

    if (Nlines == 0)
      return;

    // mDistCoef == 0.0 means no pinhole distortion
    // If no pinehole distortion and no second camera (i.e. fisheye), no need to
    // undistort
    if (mDistCoef.at<float>(0) == 0.0 && !mpCamera2)
    {
      mvKeyLinesUn = mvKeyLines;
      if (mvKeyLinesRight.size() > 0)
        mvKeyLinesRightUn = mvKeyLinesRight;
      return;
    }

    // if(mpCamera2)
    // {
    //     std::cout << "undistorting " << Nlines << " keylines for fisheye
    //     cameras" << std::endl;
    // }

    // with fisheye Nlines = NlinesLeft + NlinesRight
    const size_t numLines = NlinesLeft == -1 ? Nlines : NlinesLeft;

    // Fill matrix with points
    cv::Mat mat(2 * numLines, 2, CV_32F); // left keylines
    for (size_t i = 0, j = 0; i < numLines; i++, j += 2)
    {
      mat.at<float>(j, 0) = mvKeyLines[i].startPointX;
      mat.at<float>(j, 1) = mvKeyLines[i].startPointY;
      mat.at<float>(j + 1, 0) = mvKeyLines[i].endPointX;
      mat.at<float>(j + 1, 1) = mvKeyLines[i].endPointY;
    }
    cv::Mat matRight = NlinesRight > 0 ? cv::Mat(2 * NlinesRight, 2, CV_32F)
                                       : cv::Mat(); // right keylines
    if (NlinesRight > 0)
    {
      for (size_t i = 0, j = 0; i < NlinesRight; i++, j += 2)
      {
        matRight.at<float>(j, 0) = mvKeyLinesRight[i].startPointX;
        matRight.at<float>(j, 1) = mvKeyLinesRight[i].startPointY;
        matRight.at<float>(j + 1, 0) = mvKeyLinesRight[i].endPointX;
        matRight.at<float>(j + 1, 1) = mvKeyLinesRight[i].endPointY;
      }
    }

    // Undistort points
    mat = mat.reshape(2);
    if (NlinesRight > 0)
      matRight = matRight.reshape(2);

    if (mpCamera->GetType() == GeometricCamera::CAM_PINHOLE)
    {
      cv::undistortPoints(mat, mat, mpCamera->toK(), mDistCoef, cv::Mat(),
                          mpCamera->toLinearK());
      // NOTE: here we should have NlinesRight == 0
      MSG_ASSERT(NlinesRight == -1, "Should be NlinesRight == -1!");
    }
    else
    {
      MSG_ASSERT(mpCamera->GetType() == GeometricCamera::CAM_FISHEYE,
                 "Invalid camera type - It should be fisheye!");
      const cv::Mat D = (cv::Mat_<float>(4, 1) << mpCamera->getParameter(4),
                         mpCamera->getParameter(5), mpCamera->getParameter(6),
                         mpCamera->getParameter(7));
      cv::fisheye::undistortPoints(mat, mat, mpCamera->toK(), D, cv::Mat(),
                                   mpCamera->toLinearK());

      if (NlinesRight > 0)
      {
        const cv::Mat D2 =
            (cv::Mat_<float>(4, 1) << mpCamera2->getParameter(4),
             mpCamera2->getParameter(5), mpCamera2->getParameter(6),
             mpCamera2->getParameter(7));
        cv::fisheye::undistortPoints(matRight, matRight, mpCamera2->toK(), D2,
                                     cv::Mat(), mpCamera2->toLinearK());
      }
    }
    mat = mat.reshape(1);
    if (NlinesRight > 0)
      matRight = matRight.reshape(1);

    std::vector<cv::line_descriptor_c::KeyLine> vKeyLines;
    cv::Mat lineDescriptors;
    lineDescriptors.reserve(numLines);

    // Fill undistorted keypoint vector
    mvKeyLinesUn.reserve(numLines);
    vKeyLines.reserve(numLines);
    for (size_t i = 0, j = 0; i < numLines; i++, j += 2)
    {
      // cv::line_descriptor_c::KeyLine& kl = mvKeyLinesUn[i];
      cv::line_descriptor_c::KeyLine kl;
      kl = mvKeyLines[i];
      kl.startPointX = mat.at<float>(j, 0);
      kl.startPointY = mat.at<float>(j, 1);
      kl.endPointX = mat.at<float>(j + 1, 0);
      kl.endPointY = mat.at<float>(j + 1, 1);
      kl.angle = cv::fastAtan2((kl.endPointY - kl.startPointY),
                               (kl.endPointX - kl.startPointX)) *
                 DEG2RAD; // kl.angle is originally in radians
      // std::cout << "kl[" << i <<  "]: " << kl.startPointX <<", "<<
      // kl.startPointY <<"  -  " << kl.endPointX << ", " << kl.endPointY <<
      // std::endl;
      // TODO: add check with linear fov scale
      if (kl.startPointX < mnMinX || kl.startPointX >= mnMaxX ||
          kl.startPointY < mnMinY || kl.startPointY >= mnMaxY ||
          kl.endPointX < mnMinX || kl.endPointX >= mnMaxX ||
          kl.endPointY < mnMinY || kl.endPointY >= mnMaxY)
      {
        continue;
      }
      mvKeyLinesUn.push_back(kl);
      vKeyLines.push_back(mvKeyLines[i]);
      lineDescriptors.push_back(mLineDescriptors.row(i));
    }
    mvKeyLines = std::move(vKeyLines);
    mLineDescriptors = lineDescriptors;
    MSG_ASSERT(mvKeyLines.size() == mvKeyLinesUn.size(),
               "mvKeyLines.size() != mvKeyLinesUn.size()");
    if (NlinesLeft != -1)
    {
      NlinesLeft = mvKeyLinesUn.size();
      NlinesRight = mvKeyLinesRight.size();
      Nlines = NlinesLeft + NlinesRight;
    }
    else
    {
      Nlines = mvKeyLinesUn.size();
    }

    if (NlinesRight > 0)
    {
      std::vector<cv::line_descriptor_c::KeyLine> vKeyLinesRight;
      cv::Mat lineDescriptorsRight;
      lineDescriptorsRight.reserve(NlinesRight);
      mvKeyLinesRightUn.reserve(NlinesRight);
      vKeyLinesRight.reserve(NlinesRight);
      for (size_t i = 0, j = 0; i < NlinesRight; i++, j += 2)
      {
        // cv::line_descriptor_c::KeyLine& kl = mvKeyLinesRightUn[i];
        cv::line_descriptor_c::KeyLine kl;
        kl = mvKeyLinesRight[i];
        kl.startPointX = matRight.at<float>(j, 0);
        kl.startPointY = matRight.at<float>(j, 1);
        kl.endPointX = matRight.at<float>(j + 1, 0);
        kl.endPointY = matRight.at<float>(j + 1, 1);
        kl.angle = cv::fastAtan2((kl.endPointY - kl.startPointY),
                                 (kl.endPointX - kl.startPointX)) *
                   DEG2RAD; // kl.angle is originally in radians
        // std::cout << "kl[" << i <<  "]: " << kl.startPointX <<", "<<
        // kl.startPointY <<"  -  " << kl.endPointX << ", " << kl.endPointY <<
        // std::endl;
        // TODO: add check with linear fov scale
        if (kl.startPointX < mnMinX || kl.startPointX >= mnMaxX ||
            kl.startPointY < mnMinY || kl.startPointY >= mnMaxY ||
            kl.endPointX < mnMinX || kl.endPointX >= mnMaxX ||
            kl.endPointY < mnMinY || kl.endPointY >= mnMaxY)
        {
          continue;
        }
        mvKeyLinesRightUn.push_back(kl);
        vKeyLinesRight.push_back(mvKeyLinesRight[i]);
        lineDescriptorsRight.push_back(mLineDescriptorsRight.row(i));
      }
      mvKeyLinesRight = std::move(vKeyLinesRight);
      mLineDescriptorsRight = lineDescriptorsRight;
      MSG_ASSERT(mvKeyLinesRight.size() == mvKeyLinesRightUn.size(),
                 "mvKeyLinesRight.size() != mvKeyLinesRightUn.size()");
      NlinesRight = mvKeyLinesRightUn.size();
      Nlines = NlinesLeft + NlinesRight;
    }
  }

  void Frame::ComputeImageBounds(const cv::Mat &imLeft)
  {
    // mDistCoef == 0.0 means no pinhole distortion
    if (mDistCoef.at<float>(0) != 0.0)
    {
      cv::Mat mat(4, 2, CV_32F);
      mat.at<float>(0, 0) = 0.0;
      mat.at<float>(0, 1) = 0.0;
      mat.at<float>(1, 0) = imLeft.cols;
      mat.at<float>(1, 1) = 0.0;
      mat.at<float>(2, 0) = 0.0;
      mat.at<float>(2, 1) = imLeft.rows;
      mat.at<float>(3, 0) = imLeft.cols;
      mat.at<float>(3, 1) = imLeft.rows;

      mat = mat.reshape(2);
      cv::undistortPoints(mat, mat, mpCamera->toK(), mDistCoef, cv::Mat(),
                          mpCamera->toLinearK());
      mat = mat.reshape(1);

      // Undistort corners
      mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
      mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
      mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
      mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
    }
    else
    {
      mnMinX = 0.0f;
      mnMaxX = imLeft.cols;
      mnMinY = 0.0f;
      mnMaxY = imLeft.rows;
    }
    mnMaxDiag = sqrt(pow(mnMaxX - mnMinX, 2) + pow(mnMaxY - mnMinY, 2));
  }

  void Frame::ComputeStereoMatches()
  {
    mvuRight = vector<float>(N, -1.0f);
    mvDepth = vector<float>(N, -1.0f);

    mMedianDepth = KeyFrame::skFovCenterDistance;

    const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    // Assign keypoints to row table
    vector<vector<size_t>> vRowIndices(nRows, vector<size_t>());

    for (int i = 0; i < nRows; i++)
      vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for (int iR = 0; iR < Nr; iR++)
    {
      const cv::KeyPoint &kp =
          mvKeysRight[iR]; // we are assuming images are already rectified
      const float &kpY = kp.pt.y;
      const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
      const int maxr = ceil(kpY + r);
      const int minr = floor(kpY - r);

      for (int yi = minr; yi <= maxr; yi++)
        vRowIndices[yi].push_back(iR);
    }

    // Set limits for search (here we use the fact that uR = uL - fx*b/zL )
    const float minZ = mb; // baseline in meters
    const float minD = 0;
    const float maxD = mbf / minZ; // here maxD = fx

    // For each left keypoint search a match in the right image
    vector<pair<int, int>> vDistIdx;
    vDistIdx.reserve(N);

    for (int iL = 0; iL < N; iL++)
    {
      const cv::KeyPoint &kpL = mvKeys[iL];
      const int &levelL = kpL.octave;
      const float &vL = kpL.pt.y;
      const float &uL = kpL.pt.x;

      const vector<size_t> &vCandidates = vRowIndices[vL];

      if (vCandidates.empty())
        continue;

      const float minU = uL - maxD;
      const float maxU = uL - minD;

      if (maxU < 0)
        continue;

      int bestDist = ORBmatcher::TH_HIGH;
      size_t bestIdxR = 0;

      const cv::Mat &dL = mDescriptors.row(iL);

      // Compare descriptor to right keypoints
      for (size_t iC = 0; iC < vCandidates.size(); iC++)
      {
        const size_t iR = vCandidates[iC];
        const cv::KeyPoint &kpR = mvKeysRight[iR];

        if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
          continue;

        const float &uR = kpR.pt.x;

        if (uR >= minU && uR <= maxU)
        {
          const cv::Mat &dR = mDescriptorsRight.row(iR);
          const int dist = ORBmatcher::DescriptorDistance(dL, dR);

          if (dist < bestDist)
          {
            bestDist = dist;
            bestIdxR = iR;
          }
        }
      }

      // Subpixel match by correlation
      if (bestDist < thOrbDist)
      {
        // coordinates in image pyramid at keypoint scale
        const float uR0 = mvKeysRight[bestIdxR].pt.x;
        const float scaleFactor = mvInvScaleFactors[kpL.octave];
        const float scaleduL = round(kpL.pt.x * scaleFactor);
        const float scaledvL = round(kpL.pt.y * scaleFactor);
        const float scaleduR0 = round(uR0 * scaleFactor);

        // sliding window search
        const int w = 5;
        // cv::Mat IL =
        // mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
#ifdef USE_CUDA
        cv::cuda::GpuMat gMat = mpORBextractorLeft->mvImagePyramid[kpL.octave]
                                    .rowRange(scaledvL - w, scaledvL + w + 1)
                                    .colRange(scaleduL - w, scaleduL + w + 1);
        cv::Mat IL(gMat.rows, gMat.cols, gMat.type(), gMat.data, gMat.step);
#else
        cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave]
                         .rowRange(scaledvL - w, scaledvL + w + 1)
                         .colRange(scaleduL - w, scaleduL + w + 1);
#endif
        // Originally in ORBSLAM2
        // IL.convertTo(IL,CV_32F);
        // IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

        int bestDist = INT_MAX;
        int bestincR = 0;
        const int L = 5;
        vector<float> vDists;
        vDists.resize(2 * L + 1);

        const float iniu = scaleduR0 + L - w;
        const float endu = scaleduR0 + L + w + 1;
        if (iniu < 0 ||
            endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
          continue;

        for (int incR = -L; incR <= +L; incR++)
        {
          // cv::Mat IR =
          // mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
#ifdef USE_CUDA
          cv::cuda::GpuMat gMat =
              mpORBextractorRight->mvImagePyramid[kpL.octave]
                  .rowRange(scaledvL - w, scaledvL + w + 1)
                  .colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
          cv::Mat IR(gMat.rows, gMat.cols, gMat.type(), gMat.data, gMat.step);
#else
          cv::Mat IR =
              mpORBextractorRight->mvImagePyramid[kpL.octave]
                  .rowRange(scaledvL - w, scaledvL + w + 1)
                  .colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
#endif
          // Originally in ORBSLAM2
          // IR.convertTo(IR,CV_32F);
          // IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

          float dist = cv::norm(IL, IR, cv::NORM_L1);
          if (dist < bestDist)
          {
            bestDist = dist;
            bestincR = incR;
          }

          vDists[L + incR] = dist;
        }

        if (bestincR == -L || bestincR == L)
          continue;

        // Sub-pixel match (Parabola fitting)
        const float dist1 =
            vDists[L + bestincR - 1]; // here we assume (x1,y1) = (-1, dist1)
        const float dist2 =
            vDists[L + bestincR]; // here we assume (x2,y2) = ( 0, dist2)
        const float dist3 =
            vDists[L + bestincR + 1]; // here we assume (x3,y3) = ( 1, dist3)

        const float deltaR =
            (dist1 - dist3) /
            (2.0f *
             (dist1 + dist3 -
              2.0f * dist2)); // this is equivalent to computing of x0 = -b/2a

        if (deltaR < -1 ||
            deltaR > 1) // we are outside the parabola interpolation range
          continue;

        // Re-scaled coordinate
        float bestuR = mvScaleFactors[kpL.octave] *
                       ((float)scaleduR0 + (float)bestincR + deltaR);

        float disparity = (uL - bestuR);

        if (disparity >= minD && disparity < maxD)
        {
          if (disparity <= 0)
          {
            disparity = 0.01;
            bestuR = uL - 0.01;
          }
          mvDepth[iL] = mbf / disparity;
          mvuRight[iL] = bestuR;
          vDistIdx.push_back(pair<int, int>(bestDist, iL));
        }
      }
    }

    sort(vDistIdx.begin(), vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size() / 2].first;
    const float thDist = 1.5f * 1.4f * median;

    for (int i = vDistIdx.size() - 1; i >= 0; i--)
    {
      if (vDistIdx[i].first < thDist)
        break;
      else
      {
        mvuRight[vDistIdx[i].second] = -1;
        mvDepth[vDistIdx[i].second] = -1;
      }
    }

    if (mbUseFovCentersKfGenCriterion)
    {
      mMedianDepth = ComputeSceneMedianDepth();
    }
  }

  // we assume images are rectified => compute pixel overlap along y
  static double lineSegmentOverlapStereo(double ys1, double ye1, double ys2,
                                         double ye2)
  {
    double ymin1 = std::min(ys1, ye1);
    double yMax1 = std::max(ys1, ye1);

    double ymin2 = std::min(ys2, ye2);
    double yMax2 = std::max(ys2, ye2);

    double overlap = 0.;
    if ((yMax2 < ymin1) || (ymin2 > yMax1)) // lines do not intersect
    {
      overlap = 0.;
    }
    else
    {
      overlap = min(yMax1, yMax2) - max(ymin1, ymin2);
    }

    return overlap;
  }

  // NOTE: here we assume images have been rectified
  void Frame::ComputeStereoLineMatches()
  {
    mvuRightLineStart = vector<float>(Nlines, -1);
    mvDepthLineStart = vector<float>(Nlines, -1);

    mvuRightLineEnd = vector<float>(Nlines, -1);
    mvDepthLineEnd = vector<float>(Nlines, -1);

#if DISABLE_STATIC_LINE_TRIANGULATION
    return; // just for testing: disable static stereo triangulation
#endif

    const int thDescriptorDist = LineMatcher::TH_LOW_STEREO;

    // Set limits for search (here we use the fact that uR = uL - fx*b/zL =>
    // disparity=fx*b/zL)
    const float minZ = mb; // baseline in meters
    const float maxZ = std::min(mbf, Tracking::skLineStereoMaxDist);
    const float minD = mbf / maxZ; // minimum disparity
    const float maxD = mbf / minZ; // maximum disparity, here maxD = fx
    // std::cout << " stereo line matching - minZ: " << minZ << ", maxZ: " << maxZ
    // << ", minD: " << minD << ", maxD: " << maxD << std::endl;

    // For each left keyline search a match in the right image
    std::vector<cv::DMatch> vMatches;
    std::vector<bool> vValidMatches;
    vMatches.reserve(Nlines);
    vValidMatches.reserve(Nlines);

    int nLineMatches = 0;
    LineMatcher lineMatcher(0.7);
    if (Nlines > 0 && mvKeyLinesRight.size() > 0)
    {
      nLineMatches = lineMatcher.SearchStereoMatchesByKnn(
          *this, vMatches, vValidMatches, thDescriptorDist);
      MSG_ASSERT(vMatches.size() == vValidMatches.size(),
                 "The two sizes must be equal");
    }
    else
    {
      return;
    }

    // std::cout << "matched " << nLineMatches << " lines " << std::endl;

    int numStereolines = 0;

    // for storing matched right lines
    std::vector<bool> vFlagMatchedRight(mvKeyLinesRight.size(), false);

    // For each left keyline search a match in the right image
    vector<pair<int, int>> vDistIdx;
    vDistIdx.reserve(Nlines);

    for (size_t ii = 0; ii < vMatches.size(); ii++)
    {
      if (!vValidMatches[ii])
        continue;

      const cv::DMatch &match = vMatches[ii]; // get first match from knn search

      const int &idxL = match.queryIdx;
      const int &idxR = match.trainIdx;

      // if already matched
      if (vFlagMatchedRight[idxR])
        continue;

      const cv::line_descriptor_c::KeyLine &kll = mvKeyLines[idxL];
      const cv::line_descriptor_c::KeyLine &klr = mvKeyLinesRight[idxR];
      const float sigma = sqrt(mvLineLevelSigma2[kll.octave]);

      // we assume the distances of the camera centers to a line are approximately
      // equal => the lines should be matched in the same octave or close octaves
#if 0        
        if( kll.octave != klr.octave ) continue;
#else
      if (klr.octave < (kll.octave - 1) || klr.octave > (kll.octave + 1))
        continue;
#endif

      const float &vS = kll.startPointY;
      const float &uS = kll.startPointX;

      const float &vE = kll.endPointY;
      const float &uE = kll.endPointX;

      const float dyl = fabs(kll.startPointY - kll.endPointY);
      const float dyr = fabs(klr.startPointY - klr.endPointY);

      // check line triangulation degeneracy
      // we don't want lines that are too close to an epipolar plane (i.e. dy=0)
      const float minVerticalLineSpan = kMinVerticalLineSpan * sigma;
      if ((dyl <= minVerticalLineSpan) || (dyr <= minVerticalLineSpan))
        continue;

      const double overlap = lineSegmentOverlapStereo(
          kll.startPointY, kll.endPointY, klr.startPointY, klr.endPointY);
      if (overlap <= kMinStereoLineOverlap * sigma)
        continue; // we modulate the overlap threshold by sigma considering the
                  // detection uncertainty

      // estimate the disparity of the line endpoints

      Eigen::Vector3d startL(kll.startPointX, kll.startPointY, 1.0);
      Eigen::Vector3d endL(kll.endPointX, kll.endPointY, 1.0);
      Eigen::Vector3d ll = startL.cross(endL);
      ll = ll / sqrt(ll(0) * ll(0) +
                     ll(1) * ll(1)); // now ll = [nx ny -d] with (nx^2 + ny^2) = 1

      Eigen::Vector3d startR(klr.startPointX, klr.startPointY, 1.0);
      Eigen::Vector3d endR(klr.endPointX, klr.endPointY, 1.0);
      Eigen::Vector3d lr = startR.cross(endR);
      lr = lr / sqrt(lr(0) * lr(0) +
                     lr(1) * lr(1)); // now lr = [nx ny -d] with (nx^2 + ny^2) = 1

      const float dotProductThreshold =
          kLineNormalsDotProdThreshold *
          sigma; // this a percentage over unitary modulus ( we modulate
                 // threshold by sigma)
      const float distThreshold =
          2.f * sigma; // 1.f * sigma // 1 pixel (we modulate threshold by sigma)

      // an alternative way to check the degeneracy
      if (fabs(ll(0)) < dotProductThreshold)
        continue; // ll = [nx ny -d] with (nx^2 + ny^2) = 1
      if (fabs(lr(0)) < dotProductThreshold)
        continue; // lr = [nx ny -d] with (nx^2 + ny^2) = 1

      // detect if lines are equal => if yes the backprojected planes are parallel
      // and we must discard the matched lines
      if (Geom2DUtils::areLinesEqual(ll, lr, dotProductThreshold,
                                     distThreshold))
      {
        // std::cout << "detected equal lines[" << ii <<"] - ll " << ll << ", lr:
        // " << lr <<  std::endl;
        continue;
      }

#if 0        
        if( fabs( lr(0) ) < 0.01 ) continue;
        
        startR << - (lr(2)+lr(1)*kll.startPointY )/lr(0) , kll.startPointY ,  1.0;
        endR << - (lr(2)+lr(1)*kll.endPointY   )/lr(0) , kll.endPointY ,    1.0;
        double disparity_s = kll.startPointX - startR(0);
        double disparity_e = kll.endPointX   - endR(0);     
                        
        if( disparity_s>=minD && disparity_s<=maxD && disparity_e>=minD && disparity_e<=maxD )
        {
            if(disparity_s<=0)
            {
                disparity_s=0.01;
            }
            if(disparity_e<=0)
            {
                disparity_e=0.01;
            }            

            mvuRightLineStart[idxL] = startR(0);
            mvDepthLineStart[idxL]  = mbf/disparity_s;
            mvuRightLineEnd[idxL] = endR(0);
            mvDepthLineEnd[idxL]  = mbf/disparity_e;
        }
        else
        {
            continue; 
        }
#else

      // < TODO: compute covariance and use chi square check ?

      const double disparity_s = lr.dot(startL) / lr(0);
      const double disparity_e = lr.dot(endL) / lr(0);

      if (disparity_s >= minD && disparity_s <= maxD && disparity_e >= minD &&
          disparity_e <= maxD)
      {
        const double depth_s = mbf / disparity_s;
        const double depth_e = mbf / disparity_e;

        mvuRightLineStart[idxL] = startL(0) - disparity_s;
        mvDepthLineStart[idxL] = depth_s;
        mvuRightLineEnd[idxL] = endL(0) - disparity_e;
        mvDepthLineEnd[idxL] = depth_e;
      }
      else
      {
        continue;
      }

#endif

      float &dS = mvDepthLineStart[idxL];
      float &dE = mvDepthLineEnd[idxL];
      if (dS > 0 && dE > 0)
      {
        // use undistorted coordinates here
        const float xS = (uS - cx) * dS * invfx;
        const float yS = (vS - cy) * dS * invfy;
        const float xE = (uE - cx) * dE * invfx;
        const float yE = (vE - cy) * dE * invfy;

        Eigen::Vector3d camRay(xS, yS, dS);
        camRay.normalize();

        Eigen::Vector3d lineES(xS - xE, yS - yE, dS - dE);
        const double lineLength = lineES.norm();
        if (lineLength < Frame::skMinLineLength3D)
        {
          // force line as mono
          dS = dE = -1; // set depths as not valid
        }
        else
        {
          lineES /= lineLength;
          const float cosViewAngle = fabs((float)camRay.dot(lineES));

          if (cosViewAngle > kCosViewZAngleMax)
          {
            dS = dE = -1; // set depths as not valid
          }
        }
      }

      if (dS > 0 && dE > 0)
      {
        numStereolines++;
        vFlagMatchedRight[idxR] = true;
        vDistIdx.push_back(pair<int, int>(match.distance, idxL));
      }
      else
      {
        mvuRightLineStart[idxL] = mvuRightLineEnd[idxL] = -1;
        mvDepthLineStart[idxL] = mvDepthLineEnd[idxL] = -1;
      }
    }

#if 1
    float median = -1;
    int numFiltered = 0;
    if (vDistIdx.size() > 0)
    {
      sort(vDistIdx.begin(), vDistIdx.end());
      median = vDistIdx[vDistIdx.size() / 2].first;
      const float thDist = 1.5f * 1.48f * median;
      for (int i = vDistIdx.size() - 1; i >= 0; i--)
      {
        if (vDistIdx[i].first < thDist)
          break;
        else
        {
          numFiltered++;
          const int idx = vDistIdx[i].second;
          mvDepthLineStart[idx] = mvDepthLineEnd[idx] = -1;
          mvuRightLineStart[idx] = mvuRightLineEnd[idx] = -1;
        }
      }
    }

    // std::cout << "num lines: "<< Nlines<< ", stereo lines %: " <<
    // 100.*numStereolines/Nlines << ", filtered: " << numFiltered << ", median
    // dist: " << median << std::endl;
#endif
  }

  void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
  {
    mvuRight = vector<float>(N, -1);
    mvDepth = vector<float>(N, -1);
    mvObjectsDepth = vector<float>(N_O, -1);
    mMedianDepth = KeyFrame::skFovCenterDistance;

    for (int i = 0; i < N; i++)
    {
      const cv::KeyPoint &kp = mvKeys[i];
      const cv::KeyPoint &kpU = mvKeysUn[i];

      const float &v = kp.pt.y;
      const float &u = kp.pt.x;

      const float d = imDepth.at<float>(v, u);

      if (d > 0)
      {
        mvDepth[i] = d;
        mvuRight[i] = kpU.pt.x - mbf / d;
      }
    }
    for (int i = 0; i < N_O; i++)
    {
      const cv::KeyPoint &kp = mvObjectsKeys[i];
      const float &v = kp.pt.y;
      const float &u = kp.pt.x;
      const float d = imDepth.at<float>(v, u);
      if (d > 0)
      {
        mvObjectsDepth[i] = d;
      }
    }

    if (mbUseFovCentersKfGenCriterion)
    {
      mMedianDepth = ComputeSceneMedianDepth();
    }
  }

  static void computeLocalAverageDepth(const cv::Mat &imDepth, const int &u,
                                       const int &v, const int &deltau,
                                       const int &deltav, float &outDepth)
  {
    outDepth = 0;

    // const float depth = imDepth.at<float>(v,u);
    // if (depth>0 && std::isfinite(depth))
    {
      float sum = 0.0f;
      int count = 0;
      for (int du = -deltau; du <= deltau; du++)
      {
        for (int dv = -deltav; dv <= deltav; dv++)
        {
          const int otheru = u + du;
          const int otherv = v + dv;
          if ((otheru >= 0) && (otheru < imDepth.cols) && (otherv >= 0) &&
              (otherv < imDepth.rows))
          {
            const float &otherval = imDepth.at<float>(otherv, otheru);
            if (std::isfinite(otherval)) //&& fabs(val - otherval) < kDeltaDepth)
            {
              sum += otherval;
              count++;
            }
          }
        }
      }
      outDepth = sum / std::max(count, (int)1);
    }
  }

  static void computeLocalMinDepth(const cv::Mat &imDepth, const int &u,
                                   const int &v, const int &deltau,
                                   const int &deltav, float &outDepth)
  {
    outDepth = 0;
    float min = std::numeric_limits<float>::max();

    for (int du = -deltau; du <= deltau; du++)
    {
      for (int dv = -deltav; dv <= deltav; dv++)
      {
        const int otheru = u + du;
        const int otherv = v + dv;
        if ((otheru >= 0) && (otheru < imDepth.cols) && (otherv >= 0) &&
            (otherv < imDepth.rows))
        {
          const float &otherval = imDepth.at<float>(otherv, otheru);
          if (std::isfinite(otherval) && (otherval > 0))
          {
            if (min > otherval)
              min = otherval;
          }
        }
      }
    }
    if (min < std::numeric_limits<float>::max())
      outDepth = min;
  }

  static void computeLocalMinMaxDepth(const cv::Mat &imDepth, const int &u,
                                      const int &v, const int &deltau,
                                      const int &deltav, float &minDepth,
                                      float &maxDepth)
  {
    minDepth = 0;
    maxDepth = 0;
    float min = std::numeric_limits<float>::max();
    float max = 0;

    for (int du = -deltau; du <= deltau; du++)
    {
      for (int dv = -deltav; dv <= deltav; dv++)
      {
        const int otheru = u + du;
        const int otherv = v + dv;
        if ((otheru >= 0) && (otheru < imDepth.cols) && (otherv >= 0) &&
            (otherv < imDepth.rows))
        {
          const float &otherval = imDepth.at<float>(otherv, otheru);
          if (std::isfinite(otherval) && (otherval > 0))
          {
            if (min > otherval)
              min = otherval;
            if (max < otherval)
              max = otherval;
          }
        }
      }
    }

    if (min < std::numeric_limits<float>::max())
      minDepth = min;
    if (max > 0)
      maxDepth = max;
  }

  static void searchClosest(const cv::Mat &imDepth, const int &u, const int &v,
                            const int &deltau, const int &deltav,
                            const float &refDepth, float &outDepth)
  {
    float minDepthDist = fabs(refDepth - outDepth);
    float closestDepth = outDepth;

    for (int du = -deltau; du <= deltau; du++)
    {
      for (int dv = -deltav; dv <= deltav; dv++)
      {
        const int otheru = u + du;
        const int otherv = v + dv;
        if ((otheru >= 0) && (otheru < imDepth.cols) && (otherv >= 0) &&
            (otherv < imDepth.rows))
        {
          const float &otherDepth = imDepth.at<float>(otherv, otheru);
          if (std::isfinite(otherDepth) && (otherDepth > 0))
          {
            const float depthDist = fabs(refDepth - otherDepth);
            if (minDepthDist > depthDist)
            {
              closestDepth = otherDepth;
              minDepthDist = depthDist;
            }
          }
        }
      }
    }

    outDepth = closestDepth;
  }

  static void searchClosestAlongLine(const cv::Mat &imDepth, const cv::Point &pt1,
                                     const cv::Point &pt2, const int delta,
                                     const float &refDepth, float &outDepth)
  {
    cv::LineIterator it(imDepth, pt1, pt2, 8);
    cv::LineIterator it2 = it;

    float minDepthDist = fabs(refDepth - outDepth);
    float closestDepth = outDepth;

    // cout << endl;

    for (int i = 0; (i < it2.count) && (i < delta); i++, ++it2)
    {
      float otherDepth = imDepth.at<float>(it2.pos());
      if (std::isfinite(otherDepth) && (otherDepth > 0))
      {
        // cout << "pos: " << it2.pos() <<  ", depth: " << otherDepth << endl;
        const float depthDist = fabs(refDepth - otherDepth);
        if (minDepthDist > depthDist)
        {
          closestDepth = otherDepth;
          minDepthDist = depthDist;
        }
      }
    }

    outDepth = closestDepth;
  }

void Frame::ComputeStereoLinesFromRGBD(const cv::Mat &imDepth)
{
    mvuRightLineStart = vector<float>(Nlines, -1);
    mvDepthLineStart = vector<float>(Nlines, -1);

    mvuRightLineEnd = vector<float>(Nlines, -1);
    mvDepthLineEnd = vector<float>(Nlines, -1);

#if DISABLE_STATIC_LINE_TRIANGULATION
    return; // just for testing
#endif

    // 局部稳健深度函数：中位数 + 剔除异常值
    auto computeLocalRobustDepth = [&](int u, int v, int winSize = 3) -> float {
        std::vector<float> depths;
        for(int i=-winSize; i<=winSize; i++)
            for(int j=-winSize; j<=winSize; j++)
            {
                int x = u+i, y = v+j;
                if(x>=0 && x<imDepth.cols && y>=0 && y<imDepth.rows)
                {
                    float d = imDepth.at<float>(y,x);
                    if(d>0 && std::isfinite(d))
                        depths.push_back(d);
                }
            }
        if(depths.empty()) return -1.0f;

        // 中位数
        std::nth_element(depths.begin(), depths.begin()+depths.size()/2, depths.end());
        float med = depths[depths.size()/2];

        // 剔除离散异常
        std::vector<float> filtered;
        for(auto dd : depths)
            if(fabs(dd-med) < 0.05*med) // 可调比例
                filtered.push_back(dd);

        if(filtered.empty()) return -1.0f;
        std::nth_element(filtered.begin(), filtered.begin()+filtered.size()/2, filtered.end());
        return filtered[filtered.size()/2];
    };

    for(int i=0; i<Nlines; i++)
    {
        const cv::line_descriptor_c::KeyLine &klU = mvKeyLinesUn[i];

        int uS = static_cast<int>(klU.startPointX);
        int vS = static_cast<int>(klU.startPointY);
        int uE = static_cast<int>(klU.endPointX);
        int vE = static_cast<int>(klU.endPointY);

        int uM = (uS + uE)/2;
        int vM = (vS + vE)/2;

        // 获取端点和中点深度
        float dS = computeLocalRobustDepth(uS, vS);
        float dE = computeLocalRobustDepth(uE, vE);
        float dM = computeLocalRobustDepth(uM, vM);

        // 中点补偿异常端点
        if(dS <= 0 || !std::isfinite(dS)) dS = dM;
        if(dE <= 0 || !std::isfinite(dE)) dE = dM;

        // 如果深度仍异常，丢弃线段
        if(dS <= 0 || dE <= 0 || fabs(dS-dE) > 0.5*dM) 
        {
            mvDepthLineStart[i] = -1;
            mvDepthLineEnd[i] = -1;
            mvuRightLineStart[i] = klU.startPointX;
            mvuRightLineEnd[i] = klU.endPointX;
            continue;
        }

        // 三维坐标
        float xS = (uS - cx) * dS * invfx;
        float yS = (vS - cy) * dS * invfy;
        float xE = (uE - cx) * dE * invfx;
        float yE = (vE - cy) * dE * invfy;

        Eigen::Vector3d lineVec(xE-xS, yE-yS, dE-dS);

        // 最小线段长度过滤
        if(lineVec.norm() < Frame::skMinLineLength3D)
        {
            mvDepthLineStart[i] = -1;
            mvDepthLineEnd[i] = -1;
            continue;
        }

        // 方向过滤：线段中点向量与相机光线夹角
        // Eigen::Vector3d midPoint(0.5*(xS+xE), 0.5*(yS+yE), 0.5*(dS+dE));
        // Eigen::Vector3d camRay = midPoint.normalized();
        // Eigen::Vector3d lineDir = lineVec.normalized();
        // float cosAngle = fabs(camRay.dot(lineDir));
        // if(cosAngle > kCosViewZAngleMax)
        // {
        //     mvDepthLineStart[i] = -1;
        //     mvDepthLineEnd[i] = -1;
        //     continue;
        // }

        // 保存深度和右相机投影
        mvDepthLineStart[i] = dS;
        mvDepthLineEnd[i] = dE;
        mvuRightLineStart[i] = klU.startPointX - mbf/dS;
        mvuRightLineEnd[i] = klU.endPointX - mbf/dE;
    }
}

  bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
  {
    const float z = mvDepth[i];
    if (z > 0)
    {
      const float u = mvKeysUn[i].pt.x;
      const float v = mvKeysUn[i].pt.y;
      const float x = (u - cx) * z * invfx;
      const float y = (v - cy) * z * invfy;
      Eigen::Vector3f x3Dc(x, y, z);
      x3D = mRwc * x3Dc + mOw;
      return true;
    }
    else
      return false;
  }

  bool Frame::UnprojectStereoLine(const int &i, Eigen::Vector3f &p3DStart,
                                  Eigen::Vector3f &p3DEnd)
  {
    bool res = true;
    const float &zS = mvDepthLineStart[i];
    const float &zE = mvDepthLineEnd[i];
    if ((zS > 0) && (zE > 0))
    {
      const float uS = mvKeyLinesUn[i].startPointX;
      const float vS = mvKeyLinesUn[i].startPointY;
      const float xS = (uS - cx) * zS * invfx;
      const float yS = (vS - cy) * zS * invfy;
      Eigen::Vector3f xs3Dc(xS, yS, zS);
      p3DStart = mRwc * xs3Dc + mOw;

      const float uE = mvKeyLinesUn[i].endPointX;
      const float vE = mvKeyLinesUn[i].endPointY;
      const float xE = (uE - cx) * zE * invfx;
      const float yE = (vE - cy) * zE * invfy;
      Eigen::Vector3f xe3Dc(xE, yE, zE);
      p3DEnd = mRwc * xe3Dc + mOw;

      res = true;
    }
    else
    {
      // std::cout
      //     << "*****************************************************************"
      //     << std::endl;
      // std::cout
      //     << "Frame::UnprojectStereoLine() - WARNING unprojecting invalid line!"
      //     << std::endl;
      // std::cout
      //     << "*****************************************************************"
      //     << std::endl;
      p3DStart.setZero();
      p3DEnd.setZero();

      res = false;
    }

    return res;
  }

  float Frame::ComputeSceneMedianDepth(const int q)
  {
    float res = KeyFrame::skFovCenterDistance;

    std::vector<float> vDepths;
    vDepths.reserve(N);

    for (size_t i = 0; i < N; i++)
    {
      const float &d = mvDepth[i];
      if (d > 0)
        vDepths.push_back(d);
    }
    if (!vDepths.empty())
    {
      sort(vDepths.begin(), vDepths.end());
      res = vDepths[(vDepths.size() - 1) / 2];
    }

    // std::cout << "median depth: " << res << std::endl;

    return res;
  }

  bool Frame::imuIsPreintegrated()
  {
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
  }

  void Frame::setIntegrated()
  {
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
  }

  /// < FISHEYE
  Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight,
               const double &timeStamp,
               std::shared_ptr<LineExtractor> &lineExtractorLeft,
               std::shared_ptr<LineExtractor> &lineExtractorRight,
               ORBextractor *extractorLeft, ORBextractor *extractorRight,
               ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
               const float &thDepth, GeometricCamera *pCamera,
               GeometricCamera *pCamera2, Sophus::SE3f &Tlr, Frame *pPrevF,
               const IMU::Calib &ImuCalib)
      : mpcpi(NULL), mpLineExtractorLeft(lineExtractorLeft),
        mpLineExtractorRight(lineExtractorRight), mpORBvocabulary(voc),
        mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight),
        mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),
        mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
        mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),
        mpImuPreintegratedFrame(NULL),
        mpReferenceKF(static_cast<KeyFramePtr>(NULL)), Nlines(0),
        mMedianDepth(KeyFrame::skFovCenterDistance), mbImuPreintegrated(false),
        mpCamera(pCamera), mpCamera2(pCamera2), mbHasPose(false),
        mbHasVelocity(false)
  {
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    if (mpLineExtractorLeft)
    {
      mnLineScaleLevels = mpLineExtractorLeft->GetLevels();
      mfLineScaleFactor = mpLineExtractorLeft->GetScaleFactor();
      mfLineLogScaleFactor = log(mfLineScaleFactor);
      mvLineScaleFactors = mpLineExtractorLeft->GetScaleFactors();
      mvLineLevelSigma2 = mpLineExtractorLeft->GetScaleSigmaSquares();
      mvLineInvLevelSigma2 = mpLineExtractorLeft->GetInverseScaleSigmaSquares();
    }

    // This is done only for the first Frame (or after a change in the
    // calibration)
    if (mbInitialComputations)
    {
      ComputeImageBounds(imLeft);

      mfGridElementWidthInv =
          static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
      mfGridElementHeightInv =
          static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

      mfLineGridElementThetaInv = static_cast<float>(LINE_THETA_GRID_ROWS) /
                                  static_cast<float>(LINE_THETA_SPAN);
      mfLineGridElementDInv =
          static_cast<float>(LINE_D_GRID_COLS) / (2.0f * mnMaxDiag);

      fx = K.at<float>(0, 0);
      fy = K.at<float>(1, 1);
      cx = K.at<float>(0, 2);
      cy = K.at<float>(1, 2);
      invfx = 1.0f / fx;
      invfy = 1.0f / fy;

      mbInitialComputations = false;
    }

    mb = mbf / fx;
    mbfInv = 1.f / mbf;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

    // Features extraction

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB =
        std::chrono::steady_clock::now();
#endif
    if (mpLineExtractorLeft)
    {
#ifndef USE_CUDA
      if (Tracking::skUsePyramidPrecomputation)
      {
        // pre-compute Gaussian pyramid to be used for the extraction of both
        // keypoints and keylines
        std::thread threadGaussianLeft(&Frame::PrecomputeGaussianPyramid, this, 0,
                                       imLeft);
        std::thread threadGaussianRight(&Frame::PrecomputeGaussianPyramid, this,
                                        1, imRight);
        threadGaussianLeft.join();
        threadGaussianRight.join();
      }
#else
      if (Tracking::skUsePyramidPrecomputation)
      {
        std::cout << IoColor::Yellow()
                  << "Can't use pyramid precomputation when CUDA is active!"
                  << std::endl;
      }
#endif

      // ORB extraction
      thread threadLeft(
          &Frame::ExtractORB, this, 0, imLeft,
          static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[0],
          static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[1]);
      thread threadRight(
          &Frame::ExtractORB, this, 1, imRight,
          static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[0],
          static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[1]);
      // Line extraction
      std::thread threadLinesLeft(&Frame::ExtractLSD, this, 0, imLeft);
      std::thread threadLinesRight(&Frame::ExtractLSD, this, 1, imRight);

      threadLeft.join();
      threadRight.join();
      threadLinesLeft.join();
      threadLinesRight.join();
    }
    else
    {
      // ORB extraction
      thread threadLeft(
          &Frame::ExtractORB, this, 0, imLeft,
          static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[0],
          static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[1]);
      thread threadRight(
          &Frame::ExtractORB, this, 1, imRight,
          static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[0],
          static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[1]);
      threadLeft.join();
      threadRight.join();
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB =
        std::chrono::steady_clock::now();

    mTimeORB_Ext =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            time_EndExtORB - time_StartExtORB)
            .count();
#endif

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if (N == 0)
      return;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches =
        std::chrono::steady_clock::now();
#endif
    ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches =
        std::chrono::steady_clock::now();

    mTimeStereoMatch =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            time_EndStereoMatches - time_StartStereoMatches)
            .count();
#endif

    // Put all descriptors in the same matrix
    cv::vconcat(mDescriptors, mDescriptorsRight, mDescriptors);

    // Put all the points in the same vector (N = Nleft + Nright)
    mvpMapPoints = vector<MapPointPtr>(N, static_cast<MapPointPtr>(nullptr));
    mvbOutlier = vector<bool>(N, false);

    // LSD line segments extraction
    if (mpLineExtractorLeft)
    {
      NlinesLeft = mvKeyLines.size();
      NlinesRight = mvKeyLinesRight.size();
      Nlines = NlinesLeft + NlinesRight;

      if (Nlines == 0)
      {
        std::cout << "frame " << mnId << " no lines dectected!" << std::endl;
      }
      else
      {
        UndistortKeyLines();

        ComputeStereoFishEyeLineMatches(); // must be called before
                                           // mLineDescriptorsRight are
                                           // vconcatenated into mLineDescriptors
      }

      // Put all descriptors in the same matrix
      if (mLineDescriptorsRight.rows > 0)
      {
        if (mLineDescriptors.rows == 0)
          mLineDescriptors = mLineDescriptorsRight;
        else
          cv::vconcat(mLineDescriptors, mLineDescriptorsRight, mLineDescriptors);
      }

      mvpMapLines = vector<MapLinePtr>(Nlines, static_cast<MapLinePtr>(NULL));
      mvbLineOutlier = vector<bool>(Nlines, false);
      mvuNumLinePosOptFailures = vector<unsigned int>(Nlines, 0);
    }

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();
  }

  void Frame::ComputeStereoFishEyeMatches()
  {
    // Speed it up by matching keypoints in the lapping area
    // vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    // vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight,
    // mvKeysRight.end());

    if (mDescriptors.rows == 0 || mDescriptorsRight.rows == 0)
    {
      MSG_WARN_STREAM("No keypoints detected in frame " << mnId);
      return;
    }

    cv::Mat stereoDescLeft = mDescriptors.rowRange(
        monoLeft,
        mDescriptors
            .rows); // NOTE: here we haven't vconcatenated the descriptors yet!
    cv::Mat stereoDescRight =
        mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft, -1);
    mvRightToLeftMatch = vector<int>(Nright, -1);
    mvDepth = vector<float>(Nleft, -1.0f);
    mvuRight = vector<float>(Nleft, -1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    // Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    BFmatcher.knnMatch(stereoDescLeft, stereoDescRight, matches, 2);

    int nMatches = 0;
    int descMatches = 0;

    // Check matches using Lowe's ratio
    for (vector<vector<cv::DMatch>>::iterator it = matches.begin();
         it != matches.end(); ++it)
    {
      if ((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7)
      {
        // For every good match, check parallax and reprojection error to discard
        // spurious matches
        Eigen::Vector3f p3D;
        descMatches++;
        const size_t leftIdx = (*it)[0].queryIdx + monoLeft;
        const size_t rightIdx = (*it)[0].trainIdx + monoRight;
        const float sigma1 = mvLevelSigma2[mvKeys[leftIdx].octave];
        const float sigma2 = mvLevelSigma2[mvKeysRight[rightIdx].octave];
        float depth = static_cast<KannalaBrandt8 *>(mpCamera)->TriangulateMatches(
            mpCamera2, mvKeys[leftIdx], mvKeysRight[rightIdx], mRlr, mtlr, sigma1,
            sigma2, p3D);
        if (depth > 0.0001f)
        {
          mvLeftToRightMatch[leftIdx] = rightIdx;
          mvRightToLeftMatch[rightIdx] = leftIdx;
          mvStereo3Dpoints[leftIdx] = p3D;
          mvDepth[leftIdx] = depth;
          nMatches++;
        }
      }
    }
  }

  // NOTE: we assume we haven't vconcatenated the line descriptors yet!
  void Frame::ComputeStereoFishEyeLineMatches()
  {
    MSG_ASSERT(mpCamera2, "No second camera");
    MSG_ASSERT(NlinesLeft == mLineDescriptors.rows &&
                   NlinesRight == mLineDescriptorsRight.rows,
               "Line descriptors must be concatenated in "
               "Frame::ComputeStereoFishEyeLineMatches()");

    if (NlinesLeft == 0 || NlinesRight == 0)
    {
      MSG_WARN_STREAM("No lines detected in frame " << mnId);
      return;
    }

    // NOTE: here, we assume we haven't vconcatenated the descriptors yet!
    cv::Mat stereoDescLinesLeft =
        mLineDescriptors.rowRange(monoLinesLeft, mLineDescriptors.rows);
    cv::Mat stereoDescLinesRight = mLineDescriptorsRight.rowRange(
        monoLinesRight, mLineDescriptorsRight.rows);

    mvLeftToRightLinesMatch = vector<int>(NlinesLeft, -1);
    mvRightToLeftLinesMatch = vector<int>(NlinesRight, -1);
    mvDepthLineStart = vector<float>(NlinesLeft, -1.0f);
    mvDepthLineEnd = vector<float>(NlinesLeft, -1.0f);

    // NOTE: We store in mvuRightLineStart/End fake positive values (+1) where
    // depths are available (for both left and right lines).
    //       We need to enable stereo backprojection error in Optimization.cc when
    //       we have fisheye cameras. We can't really use this info though.
    mvuRightLineStart =
        vector<float>(Nlines, -1); // Nlines = NlinesLeft + NlinesRight
    mvuRightLineEnd =
        vector<float>(Nlines, -1); // Nlines = NlinesLeft + NlinesRight

    mvStereo3DLineStartPoints = vector<Eigen::Vector3f>(NlinesLeft);
    mvStereo3DLineEndPoints = vector<Eigen::Vector3f>(NlinesLeft);
    // mnCloseMLs = 0;

#if DISABLE_STATIC_LINE_TRIANGULATION
    return; // just for testing: disable static stereo triangulation
#endif

    const int thDescriptorDist = LineMatcher::TH_LOW_STEREO;

    // Set limits for search
    const float minZ = mb; // baseline in meters
    const float maxZ = std::min(mbf, Tracking::skLineStereoMaxDist);

    // Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

#define USE_LINE_MATCHER 1
#if USE_LINE_MATCHER
    // For each left keyline search a match in the right image
    std::vector<cv::DMatch> vMatches;
    std::vector<bool> vValidMatches;
    vMatches.reserve(NlinesLeft);
    vValidMatches.reserve(NlinesLeft);

    int nLineMatches = 0;
    LineMatcher lineMatcher(0.8);
    if (mvKeyLines.size() > 0 && mvKeyLinesRight.size() > 0)
    {
      nLineMatches = lineMatcher.SearchStereoMatchesByKnn(
          *this, vMatches, vValidMatches, thDescriptorDist);
      MSG_ASSERT(vMatches.size() == vValidMatches.size(),
                 "The two sizes must be equal");
    }
    else
    {
      return;
    }
#else
    BFmatcher.knnMatch(stereoDescLinesLeft, stereoDescLinesRight, matches, 2);
#endif

    int nMatches = 0;
    int descMatches = 0;

    CameraPairTriangulationInput camPairData;
    camPairData.K1 = mpCamera->toLinearK_();
    camPairData.K2 = mpCamera2->toLinearK_();
    camPairData.R12 = mRlr;
    camPairData.t12 = mtlr;
    camPairData.R21 = mRlr.transpose();
    camPairData.t21 = -mRlr.transpose() * mtlr;
    camPairData.e2 =
        camPairData.K2 * camPairData.t21; // epipole in image 2 (right)

    Eigen::Matrix3f K1inv;
    const float invfx = 1.0f / camPairData.K1(0, 0);
    const float invfy = 1.0f / camPairData.K1(1, 1);
    const float cx = camPairData.K1(0, 2);
    const float cy = camPairData.K1(1, 2);
    K1inv << invfx, 0, -cx * invfx, 0, invfy, -cy * invfy, 0, 0, 1.;
    camPairData.H21 = camPairData.K2 * camPairData.R21 * K1inv;

    camPairData.minZ = mb; // baseline in meters
    camPairData.maxZ = Tracking::skLineStereoMaxDist;

    LineTriangulationOutput lineTriangData;
#if USE_LINE_MATCHER
    for (size_t ii = 0; ii < vMatches.size(); ii++)
    {
      if (vValidMatches[ii])
      {
        descMatches++;
        const cv::DMatch &match = vMatches[ii]; // get first match from knn search
        const size_t leftIdx = match.queryIdx + monoLinesLeft;
        const size_t rightIdx = match.trainIdx + monoLinesRight;
#else
    // Check matches using Lowe's ratio
    for (vector<vector<cv::DMatch>>::iterator it = matches.begin();
         it != matches.end(); ++it)
    {
      if ((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7)
      {
        descMatches++;
        const size_t leftIdx = (*it)[0].queryIdx + monoLinesLeft;
        const size_t rightIdx = (*it)[0].trainIdx + monoLinesRight;
#endif
        const float sigma1 = mvLineLevelSigma2[mvKeyLinesUn[leftIdx].octave];
        const float sigma2 =
            mvLineLevelSigma2[mvKeyLinesRightUn[rightIdx].octave];

        if (static_cast<KannalaBrandt8 *>(mpCamera)->TriangulateLineMatches(
                mpCamera2, mvKeyLinesUn[leftIdx], mvKeyLinesRightUn[rightIdx],
                sigma1, sigma2, camPairData, lineTriangData))
        {
          mvLeftToRightLinesMatch[leftIdx] = rightIdx;
          mvRightToLeftLinesMatch[rightIdx] = leftIdx;
          mvStereo3DLineStartPoints[leftIdx] = lineTriangData.p3DS;
          mvStereo3DLineEndPoints[leftIdx] = lineTriangData.p3DE;
          mvDepthLineStart[leftIdx] = lineTriangData.depthS;
          mvDepthLineEnd[leftIdx] = lineTriangData.depthE;

          // const double disparity_s = mbf / lineTriangData.depthS;
          // const double disparity_e = mbf / lineTriangData.depthE;
          // mvKeyLinesUn[leftIdx].startPointX - disparity_s;
          // mvKeyLinesUn[leftIdx].endPointX - disparity_e;

          // Add virtual disparity
          // NOTE: We store in mvuRightLineStart/End fake positive values (+1)
          // where depths are available (for both left and right lines).
          //       We need to enable stereo backprojection error in
          //       Optimization.cc when we have fisheye cameras. We can't really
          //       use this info though.
          mvuRightLineStart[leftIdx] = 1;
          mvuRightLineStart[rightIdx + NlinesLeft] = 1;
          mvuRightLineEnd[leftIdx] = 1;
          mvuRightLineEnd[rightIdx + NlinesLeft] = 1;

          nMatches++;
        }
      }
    }
    std::cout << "Frame::ComputeStereoFishEyeLineMatches() - #triangulated line "
                 "matches: "
              << nMatches << ", #desc matches: " << descMatches
              << ", perc: " << std::setprecision(2) << std::fixed
              << 100.0f * nMatches / descMatches << "%" << std::endl;
  }

  bool Frame::isInFrustumChecks(MapPointPtr pMP, float viewingCosLimit,
                                bool bRight)
  {
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if (bRight)
    {
      const Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
      const Eigen::Vector3f trl = mTrl.translation();
      mR = Rrl * mRcw;
      mt = Rrl * mtcw + trl;
      twc = mRwc * mTlr.translation() + mOw;
    }
    else
    {
      mR = mRcw;
      mt = mtcw;
      twc = mOw;
    }

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    if (PcZ < 0.0f)
      return false;

    // Project in image and check it is not outside
    const Eigen::Vector2f uv =
        bRight ? mpCamera2->project(Pc) : mpCamera->project(Pc);

    if (uv(0) < mnMinX || uv(0) > mnMaxX)
      return false;
    if (uv(1) < mnMinY || uv(1) > mnMaxY)
      return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - twc;
    const float dist = PO.norm();

    if (dist < minDistance || dist > maxDistance)
      return false;

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if (viewCos < viewingCosLimit)
      return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist, this);

    if (bRight)
    {
      pMP->mTrackProjXR = uv(0);
      pMP->mTrackProjYR = uv(1);
      pMP->mnTrackScaleLevelR = nPredictedLevel;
      pMP->mTrackViewCosR = viewCos;
      pMP->mTrackDepthR = Pc_dist;
    }
    else
    {
      pMP->mTrackProjX = uv(0);
      pMP->mTrackProjY = uv(1);
      pMP->mnTrackScaleLevel = nPredictedLevel;
      pMP->mTrackViewCos = viewCos;
      pMP->mTrackDepth = Pc_dist;
    }

    return true;
  }

  // used for fisheye cameras instead of Frame::isInFrustum()
  bool Frame::isInFrustumChecks(MapLinePtr pML, float viewingCosLimit,
                                bool bRight)
  {
    // NOTE: here we have fisheye cameras
    const GeometricCamera *cam = bRight ? mpCamera2 : mpCamera;

    // 3D in absolute coordinates
    Eigen::Vector3f PS, PE;
    pML->GetWorldEndPoints(PS, PE);

    // consider the middle point
    const Eigen::Vector3f PM = 0.5 * (PS + PE);

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if (bRight)
    {
      const Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
      const Eigen::Vector3f trl = mTrl.translation();
      mR = Rrl * mRcw;
      mt = Rrl * mtcw + trl;
      twc = mRwc * mTlr.translation() + mOw;
    }
    else
    {
      mR = mRcw;
      mt = mtcw;
      twc = mOw;
    }

    // 3D in camera coordinates

    const Eigen::Vector3f PMc = mR * PM + mt;
    // const float PMc_dist = PMc.norm();
    const float &PMcZ = PMc(2);
    // Check positive depth
    if (PMcZ < 0.0f)
      return false;

    // Project in image and check it is not outside
    const Eigen::Vector2f uvM =
        cam->project(PMc); // projection with distortion model (if we enter here
                           // we have fisheye cameras)

    // check image bounds with distortion model (with fisheye )
    if (uvM(0) < mnMinX || uvM(0) > mnMaxX)
      return false;
    if (uvM(1) < mnMinY || uvM(1) > mnMaxY)
      return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pML->GetMaxDistanceInvariance();
    const float minDistance = pML->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = PM - twc;
    const float dist = PO.norm();

    if (dist < minDistance || dist > maxDistance)
      return false;

    // Check viewing angle
    Eigen::Vector3f Pn = pML->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if (viewCos < viewingCosLimit)
      return false;

    // Predict scale in the image
    const int nPredictedLevel = pML->PredictScale(dist, this);

    const Eigen::Vector3f PSc = mR * PS + mt;
    const Eigen::Vector2f uvS =
        cam->projectLinear(PSc); // projection without distortion model
    const float &PScZ = PSc(2);
    // Check positive depth
    if (PScZ < 0.0f)
      return false;

    const Eigen::Vector3f PEc = mR * PE + mt;
    const Eigen::Vector2f uvE =
        cam->projectLinear(PEc); // projection without distortion model
    const float &PEcZ = PEc(2);
    // Check positive depth
    if (PEcZ < 0.0f)
      return false;

    if (bRight)
    {
      // pML->mTrackProjMiddleXR = uv(0);
      // pML->mTrackProjMiddleYR = uv(1);
      pML->mnTrackScaleLevelR = nPredictedLevel;
      pML->mTrackViewCosR = viewCos;
      // pML->mTrackMiddleDepthR = PMc_dist;

      // undistorted
      pML->mTrackProjStartXR = uvS(0);
      pML->mTrackProjStartYR = uvS(1);
      pML->mTrackStartDepthR = PScZ;

      // undistorted
      pML->mTrackProjEndXR = uvE(0);
      pML->mTrackProjEndYR = uvE(1);
      pML->mTrackEndDepthR = PEcZ;
    }
    else
    {
      // pML->mTrackProjMiddleX = uv(0);
      // pML->mTrackProjMiddleY = uv(1);
      pML->mnTrackScaleLevel = nPredictedLevel;
      pML->mTrackViewCos = viewCos;
      // pML->mTrackMiddleDepth = PMc_dist;

      // undistorted
      pML->mTrackProjStartX = uvS(0);
      pML->mTrackProjStartY = uvS(1);
      pML->mTrackStartDepth = PScZ;

      // undistorted
      pML->mTrackProjEndX = uvE(0);
      pML->mTrackProjEndY = uvE(1);
      pML->mTrackEndDepth = PEcZ;
    }

    return true;
  }

  Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i)
  {
    return mRwc * mvStereo3Dpoints[i] + mOw;
  }

  bool Frame::UnprojectStereoLineFishEye(const int &i, Eigen::Vector3f &p3DStart,
                                         Eigen::Vector3f &p3DEnd)
  {
    int idx = i;
    if (NlinesLeft != -1 && idx >= NlinesLeft)
    {
      idx = mvRightToLeftLinesMatch[idx - NlinesLeft]; // get the corresponding
                                                       // left line if any
      if (idx < 0)
        return false;
    }
    bool res = (mvDepthLineStart[idx] > 0) && (mvDepthLineEnd[idx] > 0);
    if (res)
    {
      p3DStart = mRwc * mvStereo3DLineStartPoints[idx] + mOw;
      p3DEnd = mRwc * mvStereo3DLineEndPoints[idx] + mOw;
    }
    return res;
  }

} // namespace ORB_SLAM3