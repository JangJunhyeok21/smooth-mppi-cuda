#pragma once

#include <algorithm>
#include <cmath>

namespace mppi {

struct LateralVelocityKFParams {
    float mass{3.74f};
    float yaw_inertia{0.04712f};
    float l_f{0.163f};
    float l_r{0.161f};
    // Same normalized Pacejka parameters used by MPPI dynamic_mlp_*.
    float pacejka_b_front{2.9844349f}, pacejka_c_front{1.3f};
    float pacejka_d_front{0.36261123f}, pacejka_e_front{0.0f};
    float pacejka_b_rear{0.31731659f}, pacejka_c_rear{1.3f};
    float pacejka_d_rear{2.7999999f}, pacejka_e_rear{0.0f};
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
    float process_var_ay_bias{2.8456300e-6f};
    float initial_var_ay_bias{0.008792814f};
    float max_abs_ay_bias{0.7880429f};
    float measurement_var_pose_vy{0.07933411f};
    float pose_vy_gate{1.3457935f};
};

// Allocation-free 2-state EKF: state = [body lateral velocity, yaw rate].
// Prediction and ay observation use the same normalized Pacejka model as MPPI.
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
        params_.process_var_vy = std::max(params_.process_var_vy, 0.0f);
        params_.process_var_yaw_rate =
            std::max(params_.process_var_yaw_rate, 0.0f);
        params_.measurement_var_lateral_accel =
            std::max(params_.measurement_var_lateral_accel, 1.0e-8f);
        params_.measurement_var_yaw_rate =
            std::max(params_.measurement_var_yaw_rate, 1.0e-8f);
        params_.process_var_ay_bias = std::max(params_.process_var_ay_bias, 0.0f);
        params_.initial_var_ay_bias = std::max(params_.initial_var_ay_bias, 1.0e-8f);
        params_.max_abs_ay_bias = std::max(params_.max_abs_ay_bias, 0.0f);
        params_.measurement_var_pose_vy = std::max(params_.measurement_var_pose_vy, 1.0e-8f);
        params_.pose_vy_gate = std::max(params_.pose_vy_gate, 0.0f);
        reset(0.0f);
    }

    void reset(float measured_yaw_rate) {
        vy_ = 0.0f;
        yaw_rate_ = std::isfinite(measured_yaw_rate) ? measured_yaw_rate : 0.0f;
        p00_ = std::max(params_.initial_var_vy, 1.0e-8f);
        p01_ = 0.0f;
        p10_ = 0.0f;
        p11_ = std::max(params_.initial_var_yaw_rate, 1.0e-8f);
        ay_bias_ = 0.0f;
        p_ay_bias_ = params_.initial_var_ay_bias;
        initialized_ = true;
    }

    float update(float measured_vx, float steering_angle,
                 float measured_yaw_rate, float measured_lateral_accel,
                 float measured_pose_vy = NAN) {
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
        const float previous_vy = vy_;
        const float previous_yaw_rate = yaw_rate_;
        float dvy, rdot, model_ay;
        pacejkaDynamics(vx, steering_angle, previous_vy, previous_yaw_rate,
                        dvy, rdot, model_ay);
        const float pacejka_vy_pred = previous_vy + params_.dt * dvy;
        const float yaw_pred = previous_yaw_rate + params_.dt * rdot;
        constexpr float jacobian_eps = 1.0e-3f;
        float dvy_v, rdot_v, ay_v, dvy_r, rdot_r, ay_r;
        pacejkaDynamics(vx, steering_angle, previous_vy + jacobian_eps,
                        previous_yaw_rate, dvy_v, rdot_v, ay_v);
        pacejkaDynamics(vx, steering_angle, previous_vy,
                        previous_yaw_rate + jacobian_eps, dvy_r, rdot_r, ay_r);
        float ad00 = 1.0f + params_.dt * (dvy_v-dvy)/jacobian_eps;
        float ad01 = params_.dt * (dvy_r-dvy)/jacobian_eps;
        const float ad10 = params_.dt * (rdot_v-rdot)/jacobian_eps;
        const float ad11 = 1.0f + params_.dt * (rdot_r-rdot)/jacobian_eps;
        const float vy_pred = pacejka_vy_pred;

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
            applyPoseVyMeasurement(measured_pose_vy);
            clampState();
            return vy_;
        }

        float predicted_dvy, predicted_rdot, predicted_ay_no_bias;
        pacejkaDynamics(vx, steering_angle, vy_, yaw_rate_,
                        predicted_dvy, predicted_rdot, predicted_ay_no_bias);
        pacejkaDynamics(vx, steering_angle, vy_ + jacobian_eps, yaw_rate_,
                        dvy_v, rdot_v, ay_v);
        pacejkaDynamics(vx, steering_angle, vy_, yaw_rate_ + jacobian_eps,
                        dvy_r, rdot_r, ay_r);
        const float h00 = (ay_v-predicted_ay_no_bias)/jacobian_eps;
        const float h01 = (ay_r-predicted_ay_no_bias)/jacobian_eps;
        const float predicted_ay = predicted_ay_no_bias + ay_bias_;
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
            applyPoseVyMeasurement(measured_pose_vy);
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
        p_ay_bias_ += params_.process_var_ay_bias;
        const float bias_gain = p_ay_bias_ /
            (p_ay_bias_ + params_.measurement_var_lateral_accel);
        ay_bias_ = std::clamp(ay_bias_ + bias_gain * innovation0,
                              -params_.max_abs_ay_bias, params_.max_abs_ay_bias);
        p_ay_bias_ = std::max((1.0f-bias_gain)*p_ay_bias_,1.0e-8f);
        applyPoseVyMeasurement(measured_pose_vy);
        clampState();
        return vy_;
    }

    float getVy() const { return vy_; }
    float getYawRate() const { return yaw_rate_; }
    bool isInitialized() const { return initialized_; }

private:
    static float pacejkaForce(float slip, float fz, float b, float c,
                              float d, float e) {
        const float bs = b*slip;
        const float inner = bs-e*(bs-std::atan(bs));
        return fz*d*std::sin(c*std::atan(inner));
    }
    void pacejkaDynamics(float vx, float steering, float vy, float yaw_rate,
                         float &dvy, float &rdot, float &ay) const {
        const float safe_vx=std::max(std::fabs(vx),params_.min_longitudinal_speed);
        const float alpha_f=steering-std::atan2(vy+params_.l_f*yaw_rate,safe_vx);
        const float alpha_r=-std::atan2(vy-params_.l_r*yaw_rate,safe_vx);
        const float wheelbase=params_.l_f+params_.l_r;
        const float fzf=params_.mass*9.81f*params_.l_r/wheelbase;
        const float fzr=params_.mass*9.81f*params_.l_f/wheelbase;
        const float fyf=pacejkaForce(alpha_f,fzf,params_.pacejka_b_front,
            params_.pacejka_c_front,params_.pacejka_d_front,params_.pacejka_e_front);
        const float fyr=pacejkaForce(alpha_r,fzr,params_.pacejka_b_rear,
            params_.pacejka_c_rear,params_.pacejka_d_rear,params_.pacejka_e_rear);
        ay=(fyf*std::cos(steering)+fyr)/params_.mass;
        dvy=ay-vx*yaw_rate;
        rdot=(params_.l_f*fyf*std::cos(steering)-params_.l_r*fyr)/params_.yaw_inertia;
    }
    void applyPoseVyMeasurement(float measured_pose_vy) {
        if(!std::isfinite(measured_pose_vy)) return;
        const float innovation=measured_pose_vy-vy_;
        if(std::fabs(innovation)>params_.pose_vy_gate) return;
        const float gain=p00_/(p00_+params_.measurement_var_pose_vy);
        vy_+=gain*innovation;
        p00_=std::max((1.0f-gain)*p00_,1.0e-8f);
        p01_*=1.0f-gain;
        p10_=p01_;
    }
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
    float ay_bias_{0.0f},p_ay_bias_{1.0f};
    float p00_{1.0f}, p01_{0.0f}, p10_{0.0f}, p11_{1.0f};
    bool initialized_{false};
};

}  // namespace mppi
