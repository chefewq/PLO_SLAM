#include "Reconstruction.h"
#include <algorithm>
#include <random>
namespace ORB_SLAM3
{

    Eigen::Vector4d RecttoVector(cv::Rect rect)
    {
        return Eigen::Vector4d(static_cast<float>(rect.x),              // 左上角 x 坐标
                               static_cast<float>(rect.y),              // 左上角 y 坐标
                               static_cast<float>(rect.width + rect.x), // 右下角x坐标
                               static_cast<float>(rect.height + rect.y) // 右下角y坐标
        );
    }

    Eigen::Vector3d TriangulatePoints(const std::vector<Eigen::Vector2d> &points,
                                      std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> &projections)
    {
        const int n = projections.size();
        Eigen::Matrix<double, Eigen::Dynamic, 4> A(2 * n, 4);
        for (int i = 0; i < n; ++i)
        {
            A.row(i * 2) = points[i][0] * projections[i].row(2) - projections[i].row(0);
            A.row(i * 2 + 1) =
                points[i][1] * projections[i].row(2) - projections[i].row(1);
        }
        Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 4>> svd(
            A, Eigen::ComputeFullV);
        Eigen::MatrixXd V = svd.matrixV();
        Eigen::Vector4d X = V.col(3);
        Eigen::Vector3d center = X.head(3) / X[3];
        return center;
    }
    Eigen::Vector3d Get3DPointFromDepth(
        const std::vector<Eigen::Vector2d> &points2d,
        const std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> &projections,
        const std::vector<float> &depths,
        const std::string &depth_frame = "camera")
    {
        if (points2d.size() != projections.size() || points2d.size() != depths.size())
        {
            throw std::invalid_argument("Input sizes must match!");
        }

        Eigen::Vector3d point3d_avg = Eigen::Vector3d::Zero();
        const size_t n = points2d.size();

        for (size_t i = 0; i < n; ++i)
        {
            const auto &P = projections[i];
            const auto &point2d = points2d[i];
            const double depth = depths[i];

            // 1. 反投影到相机坐标系：X_cam = depth * (K⁻¹ * [u, v, 1]ᵀ)
            Eigen::Vector3d point_cam = P.leftCols<3>().inverse() * point2d.homogeneous();
            point_cam *= depth;

            // 2. 如果深度是世界坐标系的，转换到相机坐标系
            if (depth_frame == "world")
            {
                // X_cam = R * X_world + t → X_world = R⁻¹(X_cam - t)
                Eigen::Matrix3d R = P.block<3, 3>(0, 0);
                Eigen::Vector3d t = P.block<3, 1>(0, 3);
                point_cam = R * (point_cam - t);
            }

            // 3. 转到世界坐标系：X_world = R⁻¹ (X_cam - t)
            Eigen::Matrix3d R = P.block<3, 3>(0, 0);
            Eigen::Vector3d t = P.block<3, 1>(0, 3);
            point3d_avg += R.transpose() * (point_cam - t);
        }

        return point3d_avg / n;
    }

    Eigen::Vector3d TriangulatePoints(const std::vector<Eigen::Vector2d> &pixels,
                                      const std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> &Tcw,
                                      const Eigen::Matrix3d &K,
                                      const std::vector<float> &depths,
                                      const std::vector<float> &confidences = {})
    {
        const size_t n = pixels.size();
        assert(n >= 2 && n == Tcw.size() && n == depths.size());

        const float fx = K(0, 0), fy = K(1, 1);
        const float cx = K(0, 2), cy = K(1, 2);
        bool use_confidence = !confidences.empty() && (confidences.size() == n);

        Eigen::Vector3d result = Eigen::Vector3d::Zero();
        float total_weight = 0.0f;
        int valid_count = 0;

        for (size_t i = 0; i < n; ++i)
        {
            if (depths[i] <= 0 || !std::isfinite(pixels[i].x()))
                continue;

            double u = pixels[i].x();
            double v = pixels[i].y();
            double d = depths[i];

            // 1. 像素 → 相机坐标
            Eigen::Vector3d cam_point((u - cx) / fx * d, (v - cy) / fy * d, d);

            // 2. 相机 → 世界
            Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
            Twc.block<3, 4>(0, 0) = Tcw[i]; // 这里的 Tcw[i] 是 Matrix34d
            Twc = Twc.inverse();
            Eigen::Vector4d world_homo = Twc * Eigen::Vector4d(cam_point.x(), cam_point.y(), cam_point.z(), 1.0);

            if (world_homo.w() == 0)
                continue;
            Eigen::Vector3d world_point = world_homo.head<3>() / world_homo.w();

            // 3. 加权平均
            float weight = use_confidence ? confidences[i] : 1.0f;
            result += world_point * weight;
            total_weight += weight;
            valid_count++;
        }

        if (valid_count >= 2 && total_weight > 0)
        {
            return result / total_weight;
        }

        return Eigen::Vector3d(NAN, NAN, NAN); // 重建失败
    }

    std::pair<bool, Ellipsoid> ReconstructEllipsoidFromCenters(
        const std::vector<BBox2> &boxes,
        std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> &Rts,
        const std::vector<float> depths, const Eigen::Matrix3d &K)
    {
        size_t n = boxes.size();
        std::vector<float> sizes(n);
        std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> projections(n);
        std::vector<Eigen::Vector2d> points2d(n);
        for (size_t i = 0; i < n; ++i)
        {
            const auto &bb = boxes[i];
            points2d[i] = bbox_center(bb);
            projections[i] = K * Rts[i];
            sizes[i] = 0.5 * ((bb[2] - bb[0]) / (float)K(0, 0) + (bb[3] - bb[1]) / (float)K(1, 1));
        }
        // return {false,Ellipsoid()};
        Eigen::Vector3d center = TriangulatePoints(points2d, Rts, K, depths);
        // cout<<"center1"<<center<<endl;
        // cout<<"center2"<<center<<endl;
        float mean_3D_size = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            Eigen::Vector3d X_cam = Rts[i] * center.homogeneous();
            // Reconstruction failed if the object is behind a camera
            if (X_cam.z() <= 0  || X_cam.z()>=15)
            {
                std::cerr << "Reconstruction failed: z is negative" << std::endl;
                return {false, Ellipsoid()};
            }
            Eigen::Vector3d X_img = K * X_cam;
            float u = X_img[0] / X_cam[2];
            float v = X_img[1] / X_cam[2];

            // cout<<center<<endl;
            // if ((points2d[i] - Eigen::Vector2d(u, v)).norm() > 100) {
            //     // std::cerr << "Reconstruction failed: reconstructed center is too far from a detection" << std::endl;
            //     return {false, Ellipsoid()};
            // }
            mean_3D_size += X_cam.z() * sizes[i];
        }
        mean_3D_size /= sizes.size();
        return {true, Ellipsoid(Eigen::Vector3d::Ones() * mean_3D_size * 0.5, Eigen::Matrix3d::Identity(), center)};
    }

} // namespace ORB_SLAM3