#include "Kalmanfilter3d.h"
#include <Eigen/Cholesky>
#include <iostream>

namespace ORB_SLAM3
{

KalmanFilter3D::KalmanFilter3D()
{
    float dt = 1.0f;

    // 6x6 运动矩阵 (x,y,z,vx,vy,vz)
    _motion_mat = Eigen::Matrix<float, 6, 6>::Identity();
    for (int i = 0; i < 3; ++i)
    {
        _motion_mat(i, i + 3) = dt; // 位置受速度影响
    }

    // 3x6 观测矩阵 (x,y,z)
    _update_mat = Eigen::Matrix<float, 3, 6>::Zero();
    _update_mat(0, 0) = 1.0f;
    _update_mat(1, 1) = 1.0f;
    _update_mat(2, 2) = 1.0f;

    // 噪声权重
    _std_weight_position = 1.0f / 2.0f;
    _std_weight_velocity = 1.0f / 80.0f;
}

// ==================== 初始化 ====================
KAL_DATA3D KalmanFilter3D::initiate(const DETECT3D &measurement)
{
    KAL_MEAN3D mean;
    mean << measurement(0), measurement(1), measurement(2), 0, 0, 0;

    KAL_MEAN3D std;
    std(0) = 2 * _std_weight_position;
    std(1) = 2 * _std_weight_position;
    std(2) = 2 * _std_weight_position;
    std(3) = 10 * _std_weight_velocity;
    std(4) = 10 * _std_weight_velocity;
    std(5) = 10 * _std_weight_velocity;

    KAL_COVA3D covariance = std.array().square().matrix().asDiagonal();

    return std::make_pair(mean, covariance);
}

// ==================== 预测 ====================
void KalmanFilter3D::predict(KAL_MEAN3D &mean, KAL_COVA3D &covariance)
{
    // 预测噪声
    KAL_MEAN3D std;
    std(0) = _std_weight_position;
    std(1) = _std_weight_position;
    std(2) = _std_weight_position;
    std(3) = _std_weight_velocity;
    std(4) = _std_weight_velocity;
    std(5) = _std_weight_velocity;

    KAL_COVA3D motion_cov = std.array().square().matrix().asDiagonal();

    // 状态预测
    mean = (mean * _motion_mat.transpose()).eval();
    covariance = _motion_mat * covariance * _motion_mat.transpose() + motion_cov;
}

// ==================== 投影 ====================
KAL_HDATA3D KalmanFilter3D::project(const KAL_MEAN3D &mean, const KAL_COVA3D &covariance)
{
    Eigen::Matrix<float, 1, 3> projected_mean = mean * _update_mat.transpose();
    Eigen::Matrix<float, 3, 3> projected_cov = _update_mat * covariance * _update_mat.transpose();

    Eigen::Matrix<float, 3, 1> std;
    std << _std_weight_position, _std_weight_position, _std_weight_position;
    projected_cov += std.array().square().matrix().asDiagonal();

    return std::make_pair(projected_mean, projected_cov);
}

// ==================== 更新 ====================
KAL_DATA3D KalmanFilter3D::update(
    const KAL_MEAN3D &mean,
    const KAL_COVA3D &covariance,
    const DETECT3D &measurement)
{
    KAL_HDATA3D proj = project(mean, covariance);
    auto &proj_mean = proj.first;
    auto &proj_cov = proj.second;

    proj_cov += 1e-6f * Eigen::Matrix<float, 3, 3>::Identity(); // 确保正定

    // 卡尔曼增益
    Eigen::Matrix<float, 6, 3> kalman_gain = covariance * _update_mat.transpose() * proj_cov.inverse();

    // 更新状态
    KAL_MEAN3D new_mean = mean + (measurement - proj_mean) * kalman_gain.transpose();

    // 更新协方差
    KAL_COVA3D new_cov = covariance - kalman_gain * proj_cov * kalman_gain.transpose();

    return std::make_pair(new_mean, new_cov);
}

// ==================== 门控距离 ====================
Eigen::Matrix<float, 1, -1> KalmanFilter3D::gating_distance(
    const KAL_MEAN3D &mean,
    const KAL_COVA3D &covariance,
    const std::vector<DETECT3D> &measurements)
{
    KAL_HDATA3D proj = project(mean, covariance);
    auto &proj_mean = proj.first;
    auto &proj_cov = proj.second;

    Eigen::LLT<Eigen::Matrix<float, 3, 3>> llt(proj_cov);
    Eigen::Matrix<float, 3, 3> L = llt.matrixL();

    Eigen::Matrix<float, 1, -1> distances(measurements.size());
    for (size_t i = 0; i < measurements.size(); ++i)
    {
        Eigen::Matrix<float, 1, 3> d = measurements[i] - proj_mean;
        Eigen::Matrix<float, 3, 1> z = L.triangularView<Eigen::Lower>().solve(d.transpose());
        distances(0, i) = z.squaredNorm();
    }

    return distances;
}

} // namespace ORB_SLAM3
