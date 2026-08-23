#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace mppi {

struct LateralVelocityKFParams {
    float dt{0.04f};

    Eigen::Matrix<float, 6, 1> process_var{
        (Eigen::Matrix<float, 6, 1>() <<
            2e-5f, 2e-5f, 2e-5f, 3e-3f, 1e-2f, 3e-3f).finished()};

    Eigen::Matrix<float, 8, 1> measurement_var{
        (Eigen::Matrix<float, 8, 1>() <<
            0.015f * 0.015f, 0.015f * 0.015f, 0.01f * 0.01f,
            0.025f * 0.025f, 0.12f * 0.12f, 0.02f * 0.02f,
            0.35f * 0.35f, 0.35f * 0.35f).finished()};

    Eigen::Matrix<float, 6, 1> initial_var{
        (Eigen::Matrix<float, 6, 1>() <<
            0.01f, 0.01f, 0.005f, 0.03f, 0.12f, 0.02f).finished()};

    float mass{3.74f};
    float iz{0.04712f};
    float lf{0.163f};
    float lr{0.161f};

    float bf{6.0926f};
    float cf{1.2447f};
    float df{0.7955f};
    float ef{0.7815f};
    float br{6.6457f};
    float cr{2.2129f};
    float dr{0.7317f};
    float er{0.0597f};

    float speed_kp{27.85168694f};
    float min_accel{-9.0f};
    float max_accel{9.0f};

    float steer_scale{1.0f};
    float steer_bias{0.0f};
    float steer_tau{0.1551485f};
    float max_steer{0.4788f};
    float max_steer_rate{6.5449847f};

    float speed_accel_tau{0.09013387f};
    float speed_brake_tau{0.09717008f};
    float max_speed_rate{5.895775f};
};

// Causal six-state EKF using the same MPPI classic dynamics as step_1.
// State order: [x, y, yaw, vx, vy, yaw_rate]
// Measurement order: [MCL x, MCL y, MCL yaw, odom vx, causal MCL vy, IMU yaw_rate, IMU ax, IMU ay]
class LateralVelocityKF {
public:
    static constexpr int N = 6; 
    using State = Eigen::Matrix<float, N, 1>;
    using Covariance = Eigen::Matrix<float, N, N>;
    
    void initialize(const LateralVelocityKFParams &params) {
        params_ = params;
        params_.dt = std::max(params_.dt, 1e-4f);
        initialized_ = false;
    }

    void reset(float x, float y, float yaw, float vx, float vy, float yaw_rate,
               float steer_command, float speed_command) {
        state_ << x, y, wrap(yaw), vx, vy, yaw_rate;
        covariance_ = params_.initial_var.asDiagonal();

        applied_steer_ = std::clamp(
            steer_command, -params_.max_steer, params_.max_steer);
        speed_reference_ = vx;
        last_steer_command_ = steer_command;
        last_speed_command_ = speed_command;
        initialized_ = true;
    }

    float update(float x, float y, float yaw, float vx, float vy,
                 float yaw_rate, float ax, float ay,
                 float steer_command, float speed_command) {
        if (!initialized_) {
            reset(x, y, yaw, vx,
                  std::isfinite(vy) ? vy : 0.0f,
                  std::isfinite(yaw_rate) ? yaw_rate : 0.0f,
                  steer_command, speed_command);
        }

        updateActuators(last_steer_command_, last_speed_command_);
        last_steer_command_ = steer_command;
        last_speed_command_ = speed_command;

        const auto transition = [this](const State &candidate) {
            return step(candidate);
        };
        const Covariance transition_jacobian = jacobian6(transition, state_);

        state_ = transition(state_);
        covariance_ = transition_jacobian * covariance_ * transition_jacobian.transpose();
        covariance_.diagonal() += params_.process_var;

        Eigen::Matrix<float, 8, 1> measurement;
        measurement << x, y, yaw, vx, vy, yaw_rate, ax, ay;

        const auto observe = [this](const State &candidate) {
            Eigen::Matrix<float, 8, 1> observation;
            const Eigen::Vector3f acceleration = accelerations(candidate);
            observation << candidate(0), candidate(1), candidate(2),
                candidate(3), candidate(4), candidate(5),
                acceleration(0), acceleration(1);
            return observation;
        };

        const Eigen::Matrix<float, 8, 1> predicted_measurement = observe(state_);
        const Eigen::Matrix<float, 8, 6> observation_jacobian =
            jacobian8(observe, state_);

        Eigen::Matrix<float, 8, 1> innovation = measurement - predicted_measurement;
        innovation(2) = wrap(innovation(2));

        Eigen::Matrix<float, 8, 8> measurement_covariance =
            params_.measurement_var.asDiagonal();
        for (int index = 0; index < 8; ++index) {
            if (!std::isfinite(measurement(index))) {
                innovation(index) = 0.0f;
                measurement_covariance(index, index) = 1e9f;
            }
        }

        const Eigen::Matrix<float, 8, 8> innovation_covariance =
            observation_jacobian * covariance_ * observation_jacobian.transpose()
            + measurement_covariance;
        const Eigen::Matrix<float, 6, 8> kalman_gain =
            covariance_ * observation_jacobian.transpose()
            * innovation_covariance.ldlt().solve(
                Eigen::Matrix<float, 8, 8>::Identity());

        state_ += kalman_gain * innovation;
        state_(2) = wrap(state_(2));

        const Covariance identity = Covariance::Identity();
        covariance_ =
            (identity - kalman_gain * observation_jacobian)
            * covariance_
            * (identity - kalman_gain * observation_jacobian).transpose()
            + kalman_gain * measurement_covariance * kalman_gain.transpose();
        covariance_ = 0.5f * (covariance_ + covariance_.transpose());

        return state_(4);
    }

    float getVy() const { return state_(4); }
    float getYawRate() const { return state_(5); }
    float getAx() const { return accelerations(state_)(0); }
    float getAy() const { return accelerations(state_)(1); }

    float getState(int index) const {
        return index >= 0 && index < N ? state_(index) : 0.0f;
    }

    float getCovariance(int row, int column) const {
        const bool valid = row >= 0 && row < N && column >= 0 && column < N;
        return valid ? covariance_(row, column) : 0.0f;
    }

    bool isInitialized() const { return initialized_; }

private:
    static float wrap(float angle) {
        constexpr float two_pi = 2.0f * 3.14159265358979323846f;
        return std::remainder(angle, two_pi);
    }

    float pacejka(float slip_angle, float normal_force,
                  float b, float c, float d, float e) const {
        const float scaled_slip = b * slip_angle;
        return normal_force * d * std::sin(
            c * std::atan(scaled_slip
                - e * (scaled_slip - std::atan(scaled_slip))));
    }

    Eigen::Vector3f accelerations(const State &state) const {
        const float safe_vx = std::max(std::abs(state(3)), 0.5f);
        const float front_slip = applied_steer_ - std::atan2(
            state(4) + params_.lf * state(5), safe_vx);
        const float rear_slip = -std::atan2(
            state(4) - params_.lr * state(5), safe_vx);

        const float wheelbase = params_.lf + params_.lr;
        const float front_normal_force =
            params_.mass * 9.81f * params_.lr / wheelbase;
        const float rear_normal_force =
            params_.mass * 9.81f * params_.lf / wheelbase;

        const float front_lateral_force = pacejka(
            front_slip, front_normal_force,
            params_.bf, params_.cf, params_.df, params_.ef);
        const float rear_lateral_force = pacejka(
            rear_slip, rear_normal_force,
            params_.br, params_.cr, params_.dr, params_.er);

        const float ax = std::clamp(
            params_.speed_kp * (speed_reference_ - state(3)),
            params_.min_accel, params_.max_accel);
        const float dynamic_ay =
            (front_lateral_force * std::cos(applied_steer_)
                + rear_lateral_force) / params_.mass;
        const float dynamic_yaw_acceleration =
            (params_.lf * front_lateral_force * std::cos(applied_steer_)
                - params_.lr * rear_lateral_force) / params_.iz;

        const float blend_input = std::clamp(
            (std::abs(state(3)) - 0.2f) / 0.3f, 0.0f, 1.0f);
        const float dynamic_blend = blend_input * blend_input
            * (3.0f - 2.0f * blend_input);
        constexpr float low_speed_time_constant = 0.1f;
        const float kinematic_yaw_rate = state(3) * std::tan(applied_steer_)
            / std::max(params_.lf + params_.lr, 1.0e-6f);
        const float kinematic_ay = state(3) * state(5)
            - state(4) / low_speed_time_constant;
        const float kinematic_yaw_acceleration =
            (kinematic_yaw_rate - state(5)) / low_speed_time_constant;
        const float ay = dynamic_blend * dynamic_ay
            + (1.0f - dynamic_blend) * kinematic_ay;
        const float yaw_acceleration = dynamic_blend * dynamic_yaw_acceleration
            + (1.0f - dynamic_blend) * kinematic_yaw_acceleration;

        return {ax, ay, yaw_acceleration};
    }

    State step(const State &state) const {
        const Eigen::Vector3f acceleration = accelerations(state);
        const float dt = params_.dt;
        const float yaw = state(2);

        const float next_vx =
            state(3) + (acceleration(0) + state(4) * state(5)) * dt;
        const float next_vy =
            state(4) + (acceleration(1) - state(3) * state(5)) * dt;
        const float next_yaw_rate = state(5) + acceleration(2) * dt;

        State next_state;
        next_state <<
            state(0) + (next_vx * std::cos(yaw) - next_vy * std::sin(yaw)) * dt,
            state(1) + (next_vx * std::sin(yaw) + next_vy * std::cos(yaw)) * dt,
            wrap(yaw + next_yaw_rate * dt),
            next_vx,
            next_vy,
            next_yaw_rate;
        return next_state;
    }

    template <class Function>
    Covariance jacobian6(Function function, const State &state) const {
        const State base = function(state);
        Covariance jacobian;

        for (int column = 0; column < N; ++column) {
            const float epsilon =
                1e-5f * std::max(1.0f, std::abs(state(column)));
            State perturbed = state;
            perturbed(column) += epsilon;

            State difference = function(perturbed) - base;
            difference(2) = wrap(difference(2));
            jacobian.col(column) = difference / epsilon;
        }
        return jacobian;
    }

    template <class Function>
    Eigen::Matrix<float, 8, 6> jacobian8(
        Function function, const State &state) const {
        const Eigen::Matrix<float, 8, 1> base = function(state);
        Eigen::Matrix<float, 8, 6> jacobian;

        for (int column = 0; column < N; ++column) {
            const float epsilon =
                1e-5f * std::max(1.0f, std::abs(state(column)));
            State perturbed = state;
            perturbed(column) += epsilon;

            Eigen::Matrix<float, 8, 1> difference = function(perturbed) - base;
            difference(2) = wrap(difference(2));
            jacobian.col(column) = difference / epsilon;
        }
        return jacobian;
    }

    void updateActuators(float steer_command, float speed_command) {
        const float target_steer = std::clamp(
            steer_command, -params_.max_steer, params_.max_steer);
        const float steer_rate = std::clamp(
            (target_steer - applied_steer_)
                / std::max(params_.steer_tau, 1e-3f),
            -params_.max_steer_rate,
            params_.max_steer_rate);
        applied_steer_ = std::clamp(
            applied_steer_ + steer_rate * params_.dt,
            -params_.max_steer, params_.max_steer);

        const float speed_tau = speed_command >= speed_reference_
            ? params_.speed_accel_tau
            : params_.speed_brake_tau;
        const float speed_rate = std::clamp(
            (speed_command - speed_reference_) / std::max(speed_tau, 1e-3f),
            -params_.max_speed_rate,
            params_.max_speed_rate);
        speed_reference_ += speed_rate * params_.dt;
    }

    LateralVelocityKFParams params_{};
    State state_{State::Zero()};
    Covariance covariance_{Covariance::Identity()};

    float applied_steer_{0.0f};
    float speed_reference_{0.0f};
    float last_steer_command_{0.0f};
    float last_speed_command_{0.0f};
    bool initialized_{false};
};

}  // namespace mppi
