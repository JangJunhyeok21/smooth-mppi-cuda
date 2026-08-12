#pragma once

#include <algorithm>
#include <cmath>

namespace mppi {

struct LateralVelocityKFParams {
    float cornering_stiffness_front{110.0f};  // [N/rad], not Pacejka C_f
    float cornering_stiffness_rear{199.0f};   // [N/rad], not Pacejka C_r
    float mass{3.74f};
    float yaw_inertia{0.04712f};
    float l_f{0.163f};
    float l_r{0.161f};
    float dt{0.02f};
    float min_longitudinal_speed{0.5f};
    float low_speed_threshold{0.0f};
    float max_abs_vy{2.0f};
    float process_var_vy{0.02f};
    float process_var_yaw_rate{0.02f};
    float measurement_var_lateral_accel{0.5f};
    float measurement_var_yaw_rate{0.01f};
    float initial_var_vy{0.25f};
    float initial_var_yaw_rate{0.10f};
    // sensor_msgs/Imu.linear_acceleration.y must be vehicle body +Y (left).
    // Set to -1 when the installed IMU reports +Y to the right.
    float imu_lateral_accel_sign{1.0f};
};

// Allocation-free 2-state KF: state = [body lateral velocity, yaw rate].
// The prediction model is linear bicycle dynamics; the nonlinear Pacejka
// rollout remains completely separate in mppi_core.cu.
// lateral vy 부호 : 왼쪽 +Y, 오른쪽 -Y. IMU는 FRD(x-forward, y-right, z-down) 좌표계이므로
class LateralVelocityKF {
public:
    void initialize(const LateralVelocityKFParams &params) {
        params_ = params;
        params_.mass = std::max(params_.mass, 1.0e-4f);
        params_.yaw_inertia = std::max(params_.yaw_inertia, 1.0e-5f);
        params_.dt = std::max(params_.dt, 1.0e-4f);
        params_.min_longitudinal_speed =
            std::max(params_.min_longitudinal_speed, 0.05f);
        // low_speed_threshold controls only the optional vy=0 prior.  It is
        // intentionally independent of min_longitudinal_speed, which still
        // protects the 1/vx terms in the bicycle matrices.  A threshold of
        // zero therefore disables the hard vy reset without allowing 1/0.
        params_.low_speed_threshold = std::max(
            params_.low_speed_threshold, 0.0f);
        params_.max_abs_vy = std::max(params_.max_abs_vy, 0.05f);
        params_.cornering_stiffness_front =
            std::max(params_.cornering_stiffness_front, 0.0f);
        params_.cornering_stiffness_rear =
            std::max(params_.cornering_stiffness_rear, 0.0f);
        params_.process_var_vy = std::max(params_.process_var_vy, 0.0f);
        params_.process_var_yaw_rate =
            std::max(params_.process_var_yaw_rate, 0.0f);
        params_.measurement_var_lateral_accel =
            std::max(params_.measurement_var_lateral_accel, 1.0e-8f);
        params_.measurement_var_yaw_rate =
            std::max(params_.measurement_var_yaw_rate, 1.0e-8f);
        reset(0.0f);
    }

    void reset(float measured_yaw_rate) {
        vy_ = 0.0f;
        yaw_rate_ = std::isfinite(measured_yaw_rate) ? measured_yaw_rate : 0.0f;
        p00_ = std::max(params_.initial_var_vy, 1.0e-8f);
        p01_ = 0.0f;
        p10_ = 0.0f;
        p11_ = std::max(params_.initial_var_yaw_rate, 1.0e-8f);
        initialized_ = true;
    }

    float update(float measured_vx, float steering_angle,
                 float measured_yaw_rate, float measured_lateral_accel) {
        if (!initialized_) reset(measured_yaw_rate);

        const float abs_vx = std::fabs(std::isfinite(measured_vx) ? measured_vx : 0.0f);
        const bool yaw_valid = std::isfinite(measured_yaw_rate);
        const bool ay_valid = std::isfinite(measured_lateral_accel);

        // Euler-discretized dynamic bicycle is ill-conditioned near standstill.
        // The configured no-slip/kinematic low-speed prior is vy=0.
        if (abs_vx < params_.low_speed_threshold) {
            vy_ = 0.0f;
            if (yaw_valid) yaw_rate_ = measured_yaw_rate;
            p00_ = std::min(p00_ + params_.process_var_vy,
                            std::max(params_.initial_var_vy, 1.0e-8f));
            p01_ = p10_ = 0.0f;
            p11_ = yaw_valid ? std::max(params_.measurement_var_yaw_rate, 1.0e-8f)
                             : p11_ + params_.process_var_yaw_rate;
            return vy_;
        }

        const float vx = std::max(abs_vx, params_.min_longitudinal_speed);
        const float cf = params_.cornering_stiffness_front;
        const float cr = params_.cornering_stiffness_rear;
        const float m = params_.mass;
        const float iz = params_.yaw_inertia;
        const float lf = params_.l_f;
        const float lr = params_.l_r;
        const float inv_vx = 1.0f / vx;

        const float a00 = -(cf + cr) * inv_vx / m;
        const float a01 = -(vx + (lf * cf - lr * cr) * inv_vx / m);
        const float a10 = -(lf * cf - lr * cr) * inv_vx / iz;
        const float a11 = -(lf * lf * cf + lr * lr * cr) * inv_vx / iz;
        const float ad00 = 1.0f + params_.dt * a00;
        const float ad01 = params_.dt * a01;
        const float ad10 = params_.dt * a10;
        const float ad11 = 1.0f + params_.dt * a11;
        const float bd0 = params_.dt * cf / m;
        const float bd1 = params_.dt * lf * cf / iz;

        const float vy_pred = ad00 * vy_ + ad01 * yaw_rate_ + bd0 * steering_angle;
        const float yaw_pred = ad10 * vy_ + ad11 * yaw_rate_ + bd1 * steering_angle;

        // P- = Ad P Ad' + Q, expanded explicitly (no Eigen/heap/inverse).
        const float ap00 = ad00 * p00_ + ad01 * p10_;
        const float ap01 = ad00 * p01_ + ad01 * p11_;
        const float ap10 = ad10 * p00_ + ad11 * p10_;
        const float ap11 = ad10 * p01_ + ad11 * p11_;
        float pp00 = ap00 * ad00 + ap01 * ad01 + params_.process_var_vy;
        float pp01 = ap00 * ad10 + ap01 * ad11;
        float pp10 = ap10 * ad00 + ap11 * ad01;
        float pp11 = ap10 * ad10 + ap11 * ad11 + params_.process_var_yaw_rate;

        vy_ = vy_pred;
        yaw_rate_ = yaw_pred;
        // Requirement: if either IMU channel is NaN/Inf, prediction only.
        if (!ay_valid || !yaw_valid) {
            assignPredictedCovariance(pp00, pp01, pp10, pp11);
            clampState();
            return vy_;
        }

        const float h00 = -(cf + cr) * inv_vx / m;
        const float h01 = (-lf * cf + lr * cr) * inv_vx / m;
        // Second H row is [0, 1]. D = [cf/m, 0]'.
        const float predicted_ay = h00 * vy_ + h01 * yaw_rate_
                                 + (cf / m) * steering_angle;
        const float innovation0 = params_.imu_lateral_accel_sign
                                * measured_lateral_accel - predicted_ay;
        const float innovation1 = measured_yaw_rate - yaw_rate_;

        // S = H P- H' + R, explicitly expanded.
        const float hp00 = h00 * pp00 + h01 * pp10;
        const float hp01 = h00 * pp01 + h01 * pp11;
        const float s00 = hp00 * h00 + hp01 * h01
                        + params_.measurement_var_lateral_accel;
        const float s01 = hp01;
        const float s10 = pp10 * h00 + pp11 * h01;
        const float s11 = pp11 + params_.measurement_var_yaw_rate;
        const float determinant = s00 * s11 - s01 * s10;
        if (!std::isfinite(determinant) || std::fabs(determinant) < 1.0e-10f) {
            assignPredictedCovariance(pp00, pp01, pp10, pp11);
            clampState();
            return vy_;
        }
        const float inv_det = 1.0f / determinant;
        const float si00 = s11 * inv_det;
        const float si01 = -s01 * inv_det;
        const float si10 = -s10 * inv_det;
        const float si11 = s00 * inv_det;

        // PH' = [[pp00*h00+pp01*h01, pp01],
        //         [pp10*h00+pp11*h01, pp11]].
        const float ph00 = pp00 * h00 + pp01 * h01;
        const float ph01 = pp01;
        const float ph10 = pp10 * h00 + pp11 * h01;
        const float ph11 = pp11;
        const float k00 = ph00 * si00 + ph01 * si10;
        const float k01 = ph00 * si01 + ph01 * si11;
        const float k10 = ph10 * si00 + ph11 * si10;
        const float k11 = ph10 * si01 + ph11 * si11;

        vy_ += k00 * innovation0 + k01 * innovation1;
        yaw_rate_ += k10 * innovation0 + k11 * innovation1;

        // P = (I-KH)P-. Keep a symmetric, positive diagonal against roundoff.
        const float ikh00 = 1.0f - k00 * h00;
        const float ikh01 = -(k00 * h01 + k01);
        const float ikh10 = -k10 * h00;
        const float ikh11 = 1.0f - k10 * h01 - k11;
        p00_ = ikh00 * pp00 + ikh01 * pp10;
        p01_ = ikh00 * pp01 + ikh01 * pp11;
        p10_ = ikh10 * pp00 + ikh11 * pp10;
        p11_ = ikh10 * pp01 + ikh11 * pp11;
        symmetrizeCovariance();
        clampState();
        return vy_;
    }

    float getVy() const { return vy_; }
    float getYawRate() const { return yaw_rate_; }
    bool isInitialized() const { return initialized_; }

private:
    void assignPredictedCovariance(float p00, float p01, float p10, float p11) {
        p00_ = p00; p01_ = p01; p10_ = p10; p11_ = p11;
        symmetrizeCovariance();
    }
    void symmetrizeCovariance() {
        const float off_diagonal = 0.5f * (p01_ + p10_);
        p01_ = p10_ = off_diagonal;
        p00_ = std::max(p00_, 1.0e-8f);
        p11_ = std::max(p11_, 1.0e-8f);
    }
    void clampState() {
        if (!std::isfinite(vy_) || !std::isfinite(yaw_rate_)) {
            reset(0.0f);
            return;
        }
        vy_ = std::max(-params_.max_abs_vy,
                       std::min(params_.max_abs_vy, vy_));
    }

    LateralVelocityKFParams params_{};
    float vy_{0.0f}, yaw_rate_{0.0f};
    float p00_{1.0f}, p01_{0.0f}, p10_{0.0f}, p11_{1.0f};
    bool initialized_{false};
};

}  // namespace mppi
