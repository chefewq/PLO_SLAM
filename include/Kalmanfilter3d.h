#ifndef KALMANFILTER3D_H
#define KALMANFILTER3D_H

#include <Eigen/Dense>
#include <vector>
#include <utility>

namespace ORB_SLAM3
{

// ================== 类型定义 ==================
typedef Eigen::Matrix<float, 1, 3> DETECT3D;        // 3D 检测 (x, y, z)
typedef Eigen::Matrix<float, 1, 6> KAL_MEAN3D;      // 状态 (x, y, z, vx, vy, vz)
typedef Eigen::Matrix<float, 6, 6, Eigen::RowMajor> KAL_COVA3D;
typedef std::pair<KAL_MEAN3D, KAL_COVA3D> KAL_DATA3D;
typedef std::pair<Eigen::Matrix<float, 1, 3>, Eigen::Matrix<float, 3, 3>> KAL_HDATA3D;

class KalmanFilter3D {
public:
    KalmanFilter3D();

    // 初始化：根据测量点创建状态
    KAL_DATA3D initiate(const DETECT3D &measurement);

    // 预测：基于运动模型更新状态
    void predict(KAL_MEAN3D &mean, KAL_COVA3D &covariance);

    // 投影到测量空间
    KAL_HDATA3D project(const KAL_MEAN3D &mean, const KAL_COVA3D &covariance);

    // 更新：用新的测量修正状态
    KAL_DATA3D update(const KAL_MEAN3D &mean,
                      const KAL_COVA3D &covariance,
                      const DETECT3D &measurement);

    // 门控距离：用于数据关联
    Eigen::Matrix<float, 1, -1> gating_distance(
        const KAL_MEAN3D &mean,
        const KAL_COVA3D &covariance,
        const std::vector<DETECT3D> &measurements);

private:
    Eigen::Matrix<float, 6, 6, Eigen::RowMajor> _motion_mat;  // 运动模型矩阵
    Eigen::Matrix<float, 3, 6, Eigen::RowMajor> _update_mat;  // 更新矩阵
    float _std_weight_position;                               // 位置噪声权重
    float _std_weight_velocity;                               // 速度噪声权重
};

} // namespace ORB_SLAM3

#endif // KALMANFILTER3D_H
