#include "kalmanfilter.h"
#include <Eigen/Cholesky>
#include <iostream>

const double KalmanFilter::chi2inv95[10] = {
    0,
    3.8415,
    5.9915,
    7.8147,
    9.4877,
    11.070,
    12.592,
    14.067,
    15.507,
    16.919
};
KalmanFilter::KalmanFilter() {
    int ndim = 4;
    double dt = 1.;

    _motion_mat = Eigen::MatrixXf::Identity(8, 8);
    for(int i = 0; i < ndim; i++) {
        _motion_mat(i, ndim+i) = dt;
    }
    _update_mat = Eigen::MatrixXf::Identity(4, 8);

    this->_std_weight_position = 1. / 20 ;
    this->_std_weight_velocity = 1. / 160;
}

KAL_DATA KalmanFilter::initiate(const DETECTBOX& measurement) {
    DETECTBOX mean_pos = measurement;
    DETECTBOX mean_vel;
    for(int i = 0; i < 4; i++) mean_vel(i) = 0;

    KAL_MEAN mean;
    for(int i = 0; i < 8; i++){
        if(i < 4) mean(i) = mean_pos(i);
        else mean(i) = mean_vel(i - 4);
    }

    KAL_MEAN std;
    std(0) = 2 * _std_weight_position * measurement[3];
    std(1) = 2 * _std_weight_position * measurement[3];
    std(2) = 1e-2;
    std(3) = 2 * _std_weight_position * measurement[3];
    std(4) = 10 * _std_weight_velocity * measurement[3];
    std(5) = 10 * _std_weight_velocity * measurement[3];
    std(6) = 1e-5;
    std(7) = 10 * _std_weight_velocity * measurement[3];

    KAL_MEAN tmp = std.array().square();
    KAL_COVA var = tmp.asDiagonal();
    return std::make_pair(mean, var);
}

void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance) {
    //revise the data;
    DETECTBOX std_pos;
    std_pos << _std_weight_position * mean(3),
            _std_weight_position * mean(3),
            1e-2,
            _std_weight_position * mean(3);
    DETECTBOX std_vel;
    std_vel << _std_weight_velocity * mean(3),
            _std_weight_velocity * mean(3),
            1e-5,
            _std_weight_velocity * mean(3);
    KAL_MEAN tmp;
    tmp.block<1,4>(0,0) = std_pos;
    tmp.block<1,4>(0,4) = std_vel;
    tmp = tmp.array().square();
    KAL_COVA motion_cov = tmp.asDiagonal();
    KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
    KAL_COVA covariance1 = this->_motion_mat * covariance *(_motion_mat.transpose());
    covariance1 += motion_cov;

    mean = mean1;
    covariance = covariance1;
}

KAL_HDATA KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance) {
    DETECTBOX std;
    std << _std_weight_position * mean(3), _std_weight_position * mean(3),
            1e-1, _std_weight_position * mean(3);
    KAL_HMEAN mean1 = _update_mat * mean.transpose();
    KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
    Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
    diag = diag.array().square().matrix();
    covariance1 += diag;
//    covariance1.diagonal() << diag;
    return std::make_pair(mean1, covariance1);
}

KAL_DATA KalmanFilter::update(
        const KAL_MEAN &mean,
        const KAL_COVA &covariance,
        const DETECTBOX &measurement) {
    
    // 1. 投影观测空间
    KAL_HDATA pa = project(mean, covariance);
    KAL_HMEAN projected_mean = pa.first;
    KAL_HCOVA projected_cov = pa.second;

    // 2. 为避免数值不稳定，添加微小值到对角线确保正定性
    projected_cov += 1e-6 * Eigen::Matrix<float, 4, 4>::Identity();

    // 3. 计算增益项 B（4x8） -> transpose 为 8x4
    Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();

    // 4. 使用 LDLT 分解（比 LLT 更稳定，支持半正定矩阵）
    Eigen::Matrix<float, 8, 4> kalman_gain;
    Eigen::LDLT<Eigen::Matrix<float, 4, 4>> ldlt(projected_cov);
    if (ldlt.info() != Eigen::Success) {
        std::cerr << "Kalman update failed: projected_cov is not positive semi-definite!\n";
        std::cerr << "projected_cov:\n" << projected_cov << std::endl;
        std::exit(EXIT_FAILURE);  // or return previous state
    }
    kalman_gain = ldlt.solve(B).transpose();

    // 5. 更新状态向量
    Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; // eg. 1x4
    KAL_MEAN new_mean = mean + innovation * kalman_gain.transpose();      // 1x4 * 4x8 = 1x8

    // 6. 更新协方差矩阵
    KAL_COVA new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();

    return std::make_pair(new_mean, new_covariance);
}


Eigen::Matrix<float, 1, -1>
KalmanFilter::gating_distance(
        const KAL_MEAN &mean,
        const KAL_COVA &covariance,
        const std::vector<DETECTBOX> &measurements,
        bool only_position) {
    only_position = false;
    KAL_HDATA pa = this->project(mean, covariance);
    KAL_HMEAN mean1 = pa.first;       // shape: 1x4
    KAL_HCOVA covariance1 = pa.second; // shape: 4x4

    if (only_position) {
        Eigen::Matrix<float, 1, 2> mean_pos = mean1.leftCols(2);
        Eigen::Matrix2f covariance_pos = covariance1.topLeftCorner<2, 2>();

        Eigen::Matrix<float, Eigen::Dynamic, 2> d(measurements.size(), 2);
        for (size_t i = 0; i < measurements.size(); ++i) {
            d(i, 0) = measurements[i][0]  - mean_pos(0);
            d(i, 1) = measurements[i][1]  - mean_pos(1);
        }

        Eigen::Matrix2f L = covariance_pos.llt().matrixL();
        Eigen::Matrix<float, 2, Eigen::Dynamic> z = L.triangularView<Eigen::Lower>().solve(d.transpose());
        Eigen::Matrix<float, 1, Eigen::Dynamic> squared_maha = z.array().square().colwise().sum();

        return squared_maha;
    } else {
        DETECTBOXSS d(measurements.size(), 4);
        int pos = 0;
        for (const auto& box : measurements) {
            d.row(pos++) = box - mean1;
        }
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
        Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
        auto zz = (z.array() * z.array()).matrix();
        auto square_maha = zz.colwise().sum();
        return square_maha;
    }
}


