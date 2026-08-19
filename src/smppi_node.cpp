#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "f1_msgs/msg/f1state_arr.hpp"
#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include "cuda_mppi_controller/kinematic_residual_weights.hpp"
#include "cuda_mppi_controller/lateral_velocity_kf.hpp"
#include "smppi_cuda_controller/msg/mppi_trajectory.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <cmath>
#include <array>
#include <deque>
#include <limits>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

using namespace std::chrono_literals;

class MPPINode : public rclcpp::Node {
public:
    MPPINode() : Node("smppi_controller") {
        load_parameters();
        validate_parameters();
        cache_fixed_model_properties();

        if (uses_lateral_velocity_kf_) {
            lateral_velocity_kf_.initialize(lateral_velocity_kf_params_);
        }

        solver_ = std::make_unique<mppi::MPPISolver>(num_samples_, horizon_steps_, mppi_params_);
        if (mppi_params_.dynamics_model == mppi::KINEMATIC_RESIDUAL)
            solver_->load_residual_weights(residual_weights_path_);
        if (mppi_params_.dynamics_model == mppi::KINEMATIC_MLP_RESIDUAL)
            solver_->load_mlp_residual_weights(mlp_weights_path_);
        if (mppi_params_.dynamics_model == mppi::KINEMATIC_MLP_NO_IMU_RESIDUAL)
            solver_->load_mlp_no_imu_residual_weights(mlp_weights_path_);
        if (mppi_params_.dynamics_model == mppi::KINEMATIC_NOSLIP_NO_IMU_DIRECT_SPEED)
            solver_->load_kinematic_noslip_noimu_direct_weights(kinematic_noslip_noimu_weights_path_);
        if (mppi_params_.dynamics_model == mppi::SLIP_KINEMATIC_WITH_IMU_DIRECT_SPEED)
            solver_->load_slip_kinematic_with_imu_direct_weights(slip_kinematic_with_imu_weights_path_);
        if (mppi_params_.dynamics_model == mppi::DYNAMIC_IMU_RECURSIVE)
            solver_->load_dynamic_imu_recursive_weights(
                dynamics_model_name_ == "e2e_mlp" ? e2e_weights_path_ : dynamic_imu_weights_path_);
        if (mppi_params_.dynamics_model == mppi::DYNAMIC_MLP_RESIDUAL ||
            mppi_params_.dynamics_model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG)
            solver_->load_dynamic_mlp_residual_weights(
                mppi_params_.dynamics_model == mppi::DYNAMIC_MLP_RESIDUAL
                    ? dynamic_mlp_weights_path_ : dynamic_mlp_servo_lag_weights_path_);
        if (mppi_params_.dynamics_model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D)
            solver_->load_dynamic_mlp_vx_delta_residual_weights(dynamic_mlp_vx_delta_weights_path_);
        if (mppi_params_.dynamics_model == mppi::EFFECTIVE_HISTORY_STATE_RESIDUAL)
            solver_->load_effective_history_state_residual_weights(effective_history_weights_path_);

        load_track_csv();

        selected_drive_topic_ = is_simulation_
            ? simulation_drive_topic_ : real_drive_topic_;
        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            selected_drive_topic_, 10);
        vis_pub_   = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi_viz", 50);
        boundary_vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            boundary_visualization_topic_,
            rclcpp::QoS(rclcpp::KeepLast(5)).reliable().transient_local());
        traj_pub_  = this->create_publisher<smppi_cuda_controller::msg::MppiTrajectory>("/mppi_optimal_trajectory", 10);

        if (!is_simulation_) {
            // Real car: localization pose is in map frame, while wheel odom
            // supplies body-frame velocity only.
            mcl_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
                real_pose_topic_, 10,
                std::bind(&MPPINode::mcl_pose_callback, this, std::placeholders::_1));
            velocity_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                real_odom_topic_, 10,
                std::bind(&MPPINode::velocity_callback, this, std::placeholders::_1));
        } else {
            // Simulator: pose and twist already share one odometry frame.
            odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                simulation_odom_topic_, 10,
                std::bind(&MPPINode::odom_callback, this, std::placeholders::_1));
        }
        if (obstacle_avoidance_enabled_) {
            if (is_simulation_) {
                obstacle_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                    simulation_obstacle_odom_topic_, 10,
                    std::bind(&MPPINode::obstacle_odom_callback, this, std::placeholders::_1));
                RCLCPP_INFO(this->get_logger(),
                    "Simulation obstacle input: %s [nav_msgs/Odometry]",
                    simulation_obstacle_odom_topic_.c_str());
            } else {
                perception_obstacles_sub_ =
                    this->create_subscription<f1_msgs::msg::F1stateArr>(
                        real_perception_obstacles_topic_, 10,
                        std::bind(&MPPINode::perception_obstacles_callback,
                                  this, std::placeholders::_1));
                RCLCPP_INFO(this->get_logger(),
                    "Real perception obstacle input: %s [f1_msgs/F1stateArr, frame=%s]",
                    real_perception_obstacles_topic_.c_str(),
                    real_perception_obstacles_frame_.c_str());
            }
        }
        // The no-IMU rollout neither subscribes to nor waits for IMU.  Keep the
        // subscription only for the legacy 21-feature MLP checkpoint.
        if (mppi_params_.dynamics_model == mppi::KINEMATIC_MLP_RESIDUAL ||
            mppi_params_.dynamics_model == mppi::SLIP_KINEMATIC_WITH_IMU_DIRECT_SPEED ||
            mppi_params_.dynamics_model == mppi::DYNAMIC_IMU_RECURSIVE ||
            mppi_params_.dynamics_model == mppi::DYNAMIC_MLP_RESIDUAL ||
            mppi_params_.dynamics_model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
            mppi_params_.dynamics_model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D ||
            mppi_params_.dynamics_model == mppi::EFFECTIVE_HISTORY_STATE_RESIDUAL) {
            imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                imu_topic_, 20, std::bind(&MPPINode::imu_callback,this,std::placeholders::_1));
        }

        const auto control_period = std::chrono::milliseconds(static_cast<int>(1000.0 / std::max(1.0, control_rate_hz_)));
        timer_ = this->create_wall_timer(
            control_period, std::bind(&MPPINode::timer_callback, this));

        if (!is_simulation_) {
            RCLCPP_INFO(this->get_logger(), "MPPI Node Started — pose: %s | velocity: %s",
                        real_pose_topic_.c_str(), real_odom_topic_.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "MPPI Node Started — single odom topic: %s",
                        simulation_odom_topic_.c_str());
        }
    }

private:
    void activate_sudden_replan_if_needed(
        float obstacle_x, float obstacle_y, bool newly_observed,
        float obstacle_jump) {
        const float ego_distance = std::hypot(
            obstacle_x-current_state_.x, obstacle_y-current_state_.y);
        if (sudden_obstacle_replan_enabled_ &&
            (newly_observed || obstacle_jump >= sudden_obstacle_jump_threshold_) &&
            ego_distance <= sudden_obstacle_replan_distance_) {
            sudden_obstacle_replan_until_ = this->now() +
                rclcpp::Duration::from_seconds(sudden_obstacle_replan_duration_s_);
            RCLCPP_WARN(this->get_logger(),
                "Sudden obstacle replan: jump=%.2f m, range=%.2f m, duration=%.2f s",
                obstacle_jump, ego_distance, sudden_obstacle_replan_duration_s_);
        }
    }

    void obstacle_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (!std::isfinite(msg->pose.pose.position.x) ||
            !std::isfinite(msg->pose.pose.position.y)) return;
        const float obstacle_x = static_cast<float>(msg->pose.pose.position.x);
        const float obstacle_y = static_cast<float>(msg->pose.pose.position.y);
        const bool was_active = mppi_params_.num_obstacles > 0;
        const float obstacle_jump = has_obstacle_measurement_
            ? std::hypot(obstacle_x-last_obstacle_x_,obstacle_y-last_obstacle_y_)
            : 0.0f;
        activate_sudden_replan_if_needed(
            obstacle_x, obstacle_y, !was_active, obstacle_jump);
        mppi_params_.obs_x[0] = obstacle_x;
        mppi_params_.obs_y[0] = obstacle_y;
        mppi_params_.num_obstacles = 1;
        last_obstacle_x_ = obstacle_x;
        last_obstacle_y_ = obstacle_y;
        has_obstacle_measurement_ = true;
        obstacle_stamp_ = this->now();
    }

    void perception_obstacles_callback(
        const f1_msgs::msg::F1stateArr::SharedPtr msg) {
        // The perception contract from integration/full-driving-stack reports
        // global x/y. Empty frame_id was used by the original publisher; a
        // non-empty frame must match the configured MPPI map frame.
        if (!msg->header.frame_id.empty() &&
            msg->header.frame_id != real_perception_obstacles_frame_) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Ignoring perception obstacles in frame '%s'; expected '%s'",
                msg->header.frame_id.c_str(),
                real_perception_obstacles_frame_.c_str());
            return;
        }

        std::array<float, MAX_OBS> previous_x{};
        std::array<float, MAX_OBS> previous_y{};
        const int previous_count = mppi_params_.num_obstacles;
        for (int index=0; index<previous_count; ++index) {
            previous_x[index] = mppi_params_.obs_x[index];
            previous_y[index] = mppi_params_.obs_y[index];
        }

        int count = 0;
        for (const auto &obstacle : msg->f1_state_arr) {
            if (count >= MAX_OBS) break;
            const float obstacle_x = static_cast<float>(obstacle.x);
            const float obstacle_y = static_cast<float>(obstacle.y);
            if (!std::isfinite(obstacle_x) || !std::isfinite(obstacle_y)) continue;

            float nearest_previous_jump = std::numeric_limits<float>::infinity();
            for (int previous=0; previous<previous_count; ++previous) {
                nearest_previous_jump = std::min(nearest_previous_jump,
                    std::hypot(obstacle_x-previous_x[previous],
                               obstacle_y-previous_y[previous]));
            }
            const bool newly_observed = previous_count == 0 ||
                !std::isfinite(nearest_previous_jump) ||
                nearest_previous_jump >= sudden_obstacle_jump_threshold_;
            activate_sudden_replan_if_needed(
                obstacle_x, obstacle_y, newly_observed,
                previous_count > 0 ? nearest_previous_jump : 0.0f);
            mppi_params_.obs_x[count] = obstacle_x;
            mppi_params_.obs_y[count] = obstacle_y;
            ++count;
        }
        mppi_params_.num_obstacles = count;
        has_obstacle_measurement_ = count > 0;
        obstacle_stamp_ = this->now();
    }

    void cache_fixed_model_properties() {
        const int model = mppi_params_.dynamics_model;
        is_kinematic_residual_model_ = model == mppi::KINEMATIC_RESIDUAL;
        uses_legacy_mlp_imu_ = model == mppi::KINEMATIC_MLP_RESIDUAL;
        direct_speed_model_ =
            model == mppi::KINEMATIC_NOSLIP_NO_IMU_DIRECT_SPEED ||
            model == mppi::SLIP_KINEMATIC_WITH_IMU_DIRECT_SPEED ||
            model == mppi::DYNAMIC_IMU_RECURSIVE ||
            model == mppi::DYNAMIC_MLP_RESIDUAL ||
            model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
            model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D ||
            model == mppi::EFFECTIVE_HISTORY_STATE_RESIDUAL;
        uses_command_history_ = uses_legacy_mlp_imu_ || direct_speed_model_ ||
            model == mppi::KINEMATIC_MLP_NO_IMU_RESIDUAL;
        uses_actuator_state_ =
            model == mppi::SLIP_KINEMATIC_WITH_IMU_DIRECT_SPEED ||
            model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
            model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D;
        uses_lateral_velocity_kf_ =
            model == mppi::SLIP_KINEMATIC_WITH_IMU_DIRECT_SPEED ||
            model == mppi::DYNAMIC_IMU_RECURSIVE ||
            model == mppi::DYNAMIC_MLP_RESIDUAL ||
            model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
            model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D ||
            model == mppi::EFFECTIVE_HISTORY_STATE_RESIDUAL;
        wheelbase_ = mppi_params_.l_f + mppi_params_.l_r;
        direct_speed_step_ = mppi_params_.max_accel_rate * mppi_params_.dt;
    }

    void mcl_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        rclcpp::Time stamp(msg->header.stamp);
        if (stamp.nanoseconds() == 0) stamp = this->now();
        const auto &ori = msg->pose.orientation;
        const double yaw = std::atan2(
            2.0 * (ori.w * ori.z + ori.x * ori.y),
            1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z));
        current_state_.x = static_cast<float>(msg->pose.position.x);
        current_state_.y = static_cast<float>(msg->pose.position.y);
        current_state_.yaw = static_cast<float>(yaw);
        update_pose_velocity_observation(
            stamp, msg->pose.position.x, msg->pose.position.y, yaw);
        pose_received_ = true;
    }

    void update_pose_velocity_observation(const rclcpp::Time &stamp, double x,
                                          double y, double yaw) {
        if (!kf_pose_vy_enabled_) return;
        if (!pose_history_.empty()) {
            const double gap = (stamp-pose_history_.back().stamp).seconds();
            if (gap <= 0.0 || gap > kf_reset_gap_s_) pose_history_.clear();
        }
        pose_history_.push_back({stamp,x,y});
        while (pose_history_.size()>2 &&
               (stamp-pose_history_.front().stamp).seconds()>kf_pose_vy_window_s_)
            pose_history_.pop_front();
        if (pose_history_.size()<3) return;

        const auto origin=pose_history_.front().stamp;
        double mean_t=0.0,mean_x=0.0,mean_y=0.0;
        for(const auto &sample:pose_history_) {
            mean_t+=(sample.stamp-origin).seconds();
            mean_x+=sample.x; mean_y+=sample.y;
        }
        const double inv_n=1.0/static_cast<double>(pose_history_.size());
        mean_t*=inv_n; mean_x*=inv_n; mean_y*=inv_n;
        double denominator=0.0,numerator_x=0.0,numerator_y=0.0;
        for(const auto &sample:pose_history_) {
            const double centered_t=(sample.stamp-origin).seconds()-mean_t;
            denominator+=centered_t*centered_t;
            numerator_x+=centered_t*(sample.x-mean_x);
            numerator_y+=centered_t*(sample.y-mean_y);
        }
        if (denominator<=1.0e-8) return;
        const double world_vx=numerator_x/denominator;
        const double world_vy=numerator_y/denominator;
        const float body_vy=static_cast<float>(
            -std::sin(yaw)*world_vx+std::cos(yaw)*world_vy);
        if (!std::isfinite(body_vy)) return;
        pose_vy_buffer_.push_back({stamp,body_vy});
        while(pose_vy_buffer_.size()>pose_vy_buffer_capacity_)
            pose_vy_buffer_.pop_front();
    }

    float aligned_pose_vy(const rclcpp::Time &stamp) {
        if (!kf_pose_vy_enabled_) return NAN;
        const PoseVySample *chosen=nullptr;
        for(const auto &sample:pose_vy_buffer_) {
            if(sample.stamp<=stamp) chosen=&sample;
            else break;
        }
        if(!chosen) return NAN;
        const double age=(stamp-chosen->stamp).seconds();
        if(age<0.0 || age>kf_pose_vy_max_age_s_) return NAN;
        const float vy=chosen->vy;
        while(pose_vy_buffer_.size()>1 && pose_vy_buffer_[1].stamp<=stamp)
            pose_vy_buffer_.pop_front();
        return vy;
    }

    void velocity_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        rclcpp::Time stamp(msg->header.stamp);
        if (stamp.nanoseconds() == 0) stamp = this->now();
        align_imu_to_pose(stamp);

        const double measured_vx = msg->twist.twist.linear.x;
        const double measured_vy = msg->twist.twist.linear.y;
        const double measured_omega = msg->twist.twist.angular.z;
        if (has_prev_velocity_) {
            const double dt = (stamp - last_velocity_stamp_).seconds();
            if (dt > 1e-6 && std::isfinite(dt))
                current_ax_ = static_cast<float>((measured_vx - last_body_vx_) / dt);
        }

        current_state_.v = static_cast<float>(measured_vx);
        current_state_.vy = static_cast<float>(measured_vy);
        current_state_.omega = static_cast<float>(measured_omega);
        if (uses_lateral_velocity_kf_) {
            const float steer = std::clamp(kf_steer_scale_ * last_steer_cmd_ + kf_steer_bias_,
                                           -kf_max_steer_, kf_max_steer_);
            const float wz = aligned_imu_valid_ ? aligned_imu_[0] : NAN;
            const float ay = aligned_imu_valid_ ? aligned_imu_[2] : NAN;
            current_state_.vy = lateral_velocity_kf_.update(
                current_state_.v, steer, wz, ay, aligned_pose_vy(stamp));
            // vy is unobservable and comes from the KF. Yaw-rate is directly
            // observable: keep the signed/causally aligned IMU measurement.
            // The coupled KF yaw state was ~2x too small in the 08-08 turns.
            current_state_.omega = aligned_imu_valid_ ? aligned_imu_[0]
                                                       : static_cast<float>(measured_omega);
        }
        current_state_.slip_angle =
            std::atan2(current_state_.vy, std::fabs(current_state_.v) + 1e-5f);
        current_state_.ay = current_state_.v * current_state_.omega;

        last_body_vx_ = measured_vx;
        last_velocity_stamp_ = stamp;
        has_prev_velocity_ = true;
        velocity_received_ = true;
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        rclcpp::Time stamp(msg->header.stamp);
        if (stamp.nanoseconds() == 0) stamp = this->now();
        // 0807 IMU is FRD (x-forward, y-right, z-down), while the MPPI state
        // uses ROS FLU (x-forward, y-left, z-up). Convert once at the callback
        // boundary so every downstream consumer (EMA, vy KF, residual model)
        // sees the same body-frame convention as the training pipeline.
        imu_buffer_.push_back({
            stamp,
            imu_wz_sign_*static_cast<float>(msg->angular_velocity.z),
            imu_ax_sign_*static_cast<float>(msg->linear_acceleration.x),
            imu_ay_sign_*static_cast<float>(msg->linear_acceleration.y)});
        while(imu_buffer_.size()>imu_buffer_capacity_) imu_buffer_.pop_front();
        imu_received_=true;
    }

    void align_imu_to_pose(const rclcpp::Time &pose_stamp) {
        // Online-causal alignment: newest IMU whose timestamp is <= pose time.
        // Do not interpolate with a future message, which would add control
        // latency and leak unavailable information relative to deployment.
        const ImuSample *chosen=nullptr;
        for(const auto &sample:imu_buffer_) {
            if(sample.stamp<=pose_stamp) chosen=&sample;
            else break;
        }
        if(!chosen) { aligned_imu_valid_=false; return; }
        const double age=(pose_stamp-chosen->stamp).seconds();
        if(age<0. || age>imu_sync_max_age_s_) { aligned_imu_valid_=false; return; }
        const float raw[3]={chosen->wz,chosen->ax,chosen->ay};
        if(!imu_ema_initialized_) {
            for(int i=0;i<3;++i) aligned_imu_[i]=raw[i];
            imu_ema_initialized_=true;
        } else {
            for(int i=0;i<3;++i)
                aligned_imu_[i]=imu_ema_alpha_*raw[i]+(1.f-imu_ema_alpha_)*aligned_imu_[i];
        }
        aligned_imu_valid_=true;
        while(imu_buffer_.size()>1 && imu_buffer_[1].stamp<=pose_stamp) imu_buffer_.pop_front();
    }
    // ════════════════════════════════════════════════════════════════
    //  단일 odom_callback
    //  /ekf_odom 에서 pose(x,y,yaw) + twist(vx,vy,omega) 동시 처리
    // ════════════════════════════════════════════════════════════════
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        rclcpp::Time pose_stamp(msg->header.stamp);
        if(pose_stamp.nanoseconds()==0) pose_stamp=this->now();
        align_imu_to_pose(pose_stamp);
        // 위치 & 헤딩 (EKF가 map 프레임으로 보장)
        const auto &ori = msg->pose.pose.orientation;
        double yaw = std::atan2(
            2.0 * (ori.w * ori.z + ori.x * ori.y),
            1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z));

        const double x = msg->pose.pose.position.x;
        const double y = msg->pose.pose.position.y;
        const double measured_vx = msg->twist.twist.linear.x;
        const double measured_vy = msg->twist.twist.linear.y;
        const double measured_omega = msg->twist.twist.angular.z;

        current_state_.x   = static_cast<float>(x);
        current_state_.y   = static_cast<float>(y);
        current_state_.yaw = static_cast<float>(yaw);
        update_pose_velocity_observation(pose_stamp,x,y,yaw);

        double estimated_vx = measured_vx;
        double estimated_vy = measured_vy;
        double estimated_omega = measured_omega;

        if (has_prev_odom_) {
            const rclcpp::Time current_stamp(msg->header.stamp);
            const double dt = (current_stamp - last_odom_stamp_).seconds();
            if (dt > 1e-6 && std::isfinite(dt)) {
                current_ax_ = static_cast<float>((estimated_vx - last_body_vx_) / dt);
                const double dx = x - last_odom_x_;
                const double dy = y - last_odom_y_;
                const double world_vx = dx / dt;
                const double world_vy = dy / dt;
                const double body_vx = world_vx * std::cos(last_odom_yaw_) + world_vy * std::sin(last_odom_yaw_);
                const double body_vy = -world_vx * std::sin(last_odom_yaw_) + world_vy * std::cos(last_odom_yaw_);
                const double yaw_rate = (yaw - last_odom_yaw_) / dt;

                if (std::fabs(measured_vx) < 1e-3 && std::fabs(measured_vy) < 1e-3) {
                    estimated_vx = body_vx;
                    estimated_vy = body_vy;
                }
                if (std::fabs(measured_omega) < 1e-3) {
                    estimated_omega = yaw_rate;
                }
            }
        }

        current_state_.v     = static_cast<float>(estimated_vx);
        current_state_.vy    = static_cast<float>(estimated_vy);
        current_state_.omega = static_cast<float>(estimated_omega);
        if (uses_lateral_velocity_kf_) {
            const float steer = std::clamp(kf_steer_scale_*last_steer_cmd_+kf_steer_bias_,
                                           -kf_max_steer_,kf_max_steer_);
            const float wz = aligned_imu_valid_ ? aligned_imu_[0] : NAN;
            const float ay = aligned_imu_valid_ ? aligned_imu_[2] : NAN;
            current_state_.vy = lateral_velocity_kf_.update(
                current_state_.v,steer,wz,ay,aligned_pose_vy(pose_stamp));
            current_state_.omega = aligned_imu_valid_ ? aligned_imu_[0]
                                                       : static_cast<float>(measured_omega);
        }
        last_body_vx_ = estimated_vx;

        // 파생 상태
        current_state_.slip_angle =
            std::atan2(current_state_.vy, std::fabs(current_state_.v) + 1e-5f);
        current_state_.ay = current_state_.v * current_state_.omega;

        last_odom_stamp_ = rclcpp::Time(msg->header.stamp);
        last_odom_x_ = x;
        last_odom_y_ = y;
        last_odom_yaw_ = yaw;
        has_prev_odom_ = true;
        odom_received_ = true;
    }

    float compute_min_boundary_distance(const mppi::State &s, int current_path_idx) {
        if (left_xs_.empty() || right_xs_.empty() ||
            left_xs_.size() != right_xs_.size() || ref_path_xs_.empty()) return 1e9f;

        float dx = s.x - ref_path_xs_[current_path_idx];
        float dy = s.y - ref_path_ys_[current_path_idx];
        float ref_yaw = ref_path_yaws_[current_path_idx];
        float nx = -std::sin(ref_yaw), ny = std::cos(ref_yaw);
        float e_y = dx * nx + dy * ny;

        float dx_l = left_xs_[current_path_idx]  - ref_path_xs_[current_path_idx];
        float dy_l = left_ys_[current_path_idx]  - ref_path_ys_[current_path_idx];
        float dx_r = right_xs_[current_path_idx] - ref_path_xs_[current_path_idx];
        float dy_r = right_ys_[current_path_idx] - ref_path_ys_[current_path_idx];
        return std::min(std::hypot(dx_l, dy_l) - e_y, std::hypot(dx_r, dy_r) + e_y);
    }

    int update_nearest_index(const mppi::State &s) {
        if (ref_path_xs_.empty()) return 0;
        int nearest = 0; float min_d = 1e9f;
        for (int i = 0; i < (int)ref_path_xs_.size(); ++i) {
            float d = (s.x - ref_path_xs_[i]) * (s.x - ref_path_xs_[i])
                    + (s.y - ref_path_ys_[i]) * (s.y - ref_path_ys_[i]);
            if (d < min_d) { min_d = d; nearest = i; }
        }
        return nearest;
    }

    void append_best_traj_costs(
        const std::vector<mppi::State>   &best_traj,
        const std::vector<mppi::Control> &optimal_controls,
        smppi_cuda_controller::msg::MppiTrajectory &msg)
    {
        if (best_traj.empty() || optimal_controls.empty() ||
            ref_path_xs_.empty() || ref_path_yaws_.empty()) return;

        int t_idx = (best_traj.size() > 1 && optimal_controls.size() > 1) ? 1 : 0;
        const auto &s      = best_traj[t_idx];
        const auto &u      = optimal_controls[t_idx];
        const auto &u_prev = (t_idx == 0) ? optimal_controls[0] : optimal_controls[t_idx - 1];
        int idx = update_nearest_index(s);

        float dx = s.x - ref_path_xs_[idx], dy = s.y - ref_path_ys_[idx];
        float dist_error = dx*dx + dy*dy;
        float speed_err  = s.v - mppi_params_.max_speed;
        float overspeed  = (speed_err > 0.f) ? mppi_params_.q_v * speed_err * speed_err : 0.f;

        float d_steer = u.steer - u_prev.steer, d_accel = u.accel - u_prev.accel;
        float ay_abs = fabsf(s.ay);
        float lat_g  = (ay_abs >= 9.5f) ? mppi_params_.q_lat_g * expf(-3.f*(ay_abs-9.5f)) : 0.f;

        float min_bnd   = compute_min_boundary_distance(s, idx);
        float safe_dist = mppi_params_.collision_radius;
        float bnd_cost  = 0.f;
        if (min_bnd < safe_dist) {
            float pen = safe_dist - min_bnd;
            bnd_cost = mppi_params_.q_boundary_slack * pen * pen;
        }

        msg.dist_cost       = mppi_params_.q_dist * dist_error;
        msg.vel_cost        = overspeed;
        msg.steer_rate_cost = mppi_params_.q_du * 2.f * d_steer * d_steer;
        msg.accel_rate_cost = mppi_params_.q_du * std::fabs(d_accel);
        msg.steer_cost      = mppi_params_.q_steer * u.steer * u.steer;
        msg.slip_cost       = lat_g;
        msg.boundary_cost   = bnd_cost;
        msg.yaw             = s.yaw;
        msg.ref_yaw         = ref_path_yaws_[idx];
    }

    void load_parameters() {
        this->declare_parameter("dynamics_model", "legacy_hybrid");
        dynamics_model_name_ = this->get_parameter("dynamics_model").as_string();
        this->declare_parameter("residual_weights_path", "/home/a/smooth-mppi-cuda/config/kinematic_residual_gru.bin");
        residual_weights_path_ = this->get_parameter("residual_weights_path").as_string();
        this->declare_parameter("mlp_weights_path", "/home/a/smooth-mppi-cuda/config/kinematic_mlp_residual.bin");
        mlp_weights_path_=this->get_parameter("mlp_weights_path").as_string();
        this->declare_parameter("kinematic_noslip_noimu_weights_path", "/home/a/smooth-mppi-cuda/config/kinematic_noslip_noimu_direct_speed.bin");
        kinematic_noslip_noimu_weights_path_=this->get_parameter("kinematic_noslip_noimu_weights_path").as_string();
        this->declare_parameter("slip_kinematic_with_imu_weights_path", "/home/a/smooth-mppi-cuda/config/slip_kinematic_with_imu_direct_speed.bin");
        slip_kinematic_with_imu_weights_path_=this->get_parameter("slip_kinematic_with_imu_weights_path").as_string();
        this->declare_parameter("dynamic_imu_weights_path", "/home/a/smooth-mppi-cuda/config/dynamic_imu_recursive.bin");
        dynamic_imu_weights_path_=this->get_parameter("dynamic_imu_weights_path").as_string();
        this->declare_parameter("dynamic_mlp_weights_path", "/home/a/smooth-mppi-cuda/config/dynamic_MLP.bin");
        dynamic_mlp_weights_path_=this->get_parameter("dynamic_mlp_weights_path").as_string();
        this->declare_parameter("dynamic_mlp_servo_lag_weights_path", "/home/a/smooth-mppi-cuda/config/dynamic_MLP_all_drive_fixed_iz.bin");
        dynamic_mlp_servo_lag_weights_path_=this->get_parameter("dynamic_mlp_servo_lag_weights_path").as_string();
        this->declare_parameter("dynamic_mlp_vx_delta_weights_path", "/home/a/smooth-mppi-cuda/config/dynamic_40ms_vx_delta_history_24d.bin");
        dynamic_mlp_vx_delta_weights_path_=this->get_parameter("dynamic_mlp_vx_delta_weights_path").as_string();
        this->declare_parameter("e2e_weights_path", "/home/a/smooth-mppi-cuda/config/E2E.bin");
        e2e_weights_path_=this->get_parameter("e2e_weights_path").as_string();
        this->declare_parameter("effective_history_weights_path", "/home/a/smooth-mppi-cuda/model_tuning/results/effective_history_recursive_correct_history_seed31/effective_history_state_residual.bin");
        effective_history_weights_path_=this->get_parameter("effective_history_weights_path").as_string();
        if (dynamics_model_name_ == "legacy_hybrid") {
            mppi_params_.dynamics_model = mppi::LEGACY_HYBRID;
        } else if (dynamics_model_name_ == "kinematic") {
            mppi_params_.dynamics_model = mppi::KINEMATIC;
        } else if (dynamics_model_name_ == "kinematic_residual") {
            mppi_params_.dynamics_model = mppi::KINEMATIC_RESIDUAL;
        } else if (dynamics_model_name_ == "kinematic_mlp_residual") {
            mppi_params_.dynamics_model = mppi::KINEMATIC_MLP_RESIDUAL;
        } else if (dynamics_model_name_ == "kinematic_mlp_no_imu_residual") {
            mppi_params_.dynamics_model = mppi::KINEMATIC_MLP_NO_IMU_RESIDUAL;
        } else if (dynamics_model_name_ == "kinematic_noslip_noimu_direct_speed") {
            mppi_params_.dynamics_model = mppi::KINEMATIC_NOSLIP_NO_IMU_DIRECT_SPEED;
        } else if (dynamics_model_name_ == "slip_kinematic_with_imu_direct_speed") {
            mppi_params_.dynamics_model = mppi::SLIP_KINEMATIC_WITH_IMU_DIRECT_SPEED;
        } else if (dynamics_model_name_ == "dynamic_imu_recursive") {
            mppi_params_.dynamics_model = mppi::DYNAMIC_IMU_RECURSIVE;
        } else if (dynamics_model_name_ == "e2e_mlp") {
            mppi_params_.dynamics_model = mppi::DYNAMIC_IMU_RECURSIVE;
        } else if (dynamics_model_name_ == "dynamic_mlp_residual") {
            mppi_params_.dynamics_model = mppi::DYNAMIC_MLP_RESIDUAL;
        } else if (dynamics_model_name_ == "dynamic_mlp_residual_servo_lag") {
            mppi_params_.dynamics_model = mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG;
        } else if (dynamics_model_name_ == "dynamic_mlp_residual_servo_lag_vx_delta_24d") {
            mppi_params_.dynamics_model = mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D;
        } else if (dynamics_model_name_ == "effective_history_state_residual") {
            mppi_params_.dynamics_model = mppi::EFFECTIVE_HISTORY_STATE_RESIDUAL;
        } else {
            throw std::invalid_argument("Unknown dynamics_model: " + dynamics_model_name_);
        }
        this->declare_parameter("num_samples",          8000);
        num_samples_ = this->get_parameter("num_samples").as_int();
        this->declare_parameter("horizon_steps",        80);
        horizon_steps_ = this->get_parameter("horizon_steps").as_int();
        this->declare_parameter("max_steer",            0.507);  mppi_params_.max_steer     = this->get_parameter("max_steer").as_double();
        this->declare_parameter("min_accel",            -9.0);   mppi_params_.min_accel     = this->get_parameter("min_accel").as_double();
        this->declare_parameter("max_accel",            9.0);    mppi_params_.max_accel     = this->get_parameter("max_accel").as_double();
        this->declare_parameter("min_speed",            0.0);    mppi_params_.min_speed     = this->get_parameter("min_speed").as_double();
        this->declare_parameter("max_speed",            10.0);   mppi_params_.max_speed     = this->get_parameter("max_speed").as_double();
        this->declare_parameter("heading_speed_limit_gain", 8.0);
        heading_speed_limit_gain_ = this->get_parameter("heading_speed_limit_gain").as_double();
        this->declare_parameter("contour_speed_limit_gain", 2.0);
        contour_speed_limit_gain_ = this->get_parameter("contour_speed_limit_gain").as_double();
        this->declare_parameter("q_dist",               1.5);    mppi_params_.q_dist        = this->get_parameter("q_dist").as_double();
        this->declare_parameter("q_contour",            0.5);    mppi_params_.q_contour     = this->get_parameter("q_contour").as_double();
        this->declare_parameter("q_lag",                5.0);    mppi_params_.q_lag         = this->get_parameter("q_lag").as_double();
        this->declare_parameter("q_heading",            12.0);   mppi_params_.q_heading     = this->get_parameter("q_heading").as_double();
        this->declare_parameter("q_error_speed",        8.0);    mppi_params_.q_error_speed = this->get_parameter("q_error_speed").as_double();
        this->declare_parameter("q_v",                  2.0);    mppi_params_.q_v           = this->get_parameter("q_v").as_double();
        this->declare_parameter("q_du",                 0.8);    mppi_params_.q_du          = this->get_parameter("q_du").as_double();
        this->declare_parameter("q_steer",              0.3);    mppi_params_.q_steer       = this->get_parameter("q_steer").as_double();
        this->declare_parameter("q_collision",          400.0);  mppi_params_.q_collision   = this->get_parameter("q_collision").as_double();
        this->declare_parameter("q_lat_g",              200.0);  mppi_params_.q_lat_g       = this->get_parameter("q_lat_g").as_double();
        this->declare_parameter("lat_g_soft_limit",     9.81);   mppi_params_.lat_g_soft_limit = this->get_parameter("lat_g_soft_limit").as_double();
        this->declare_parameter("longitudinal_accel_soft_limit", 4.0); mppi_params_.longitudinal_accel_soft_limit = this->get_parameter("longitudinal_accel_soft_limit").as_double();
        this->declare_parameter("q_rear_slip",          800.0);  mppi_params_.q_rear_slip = this->get_parameter("q_rear_slip").as_double();
        this->declare_parameter("rear_slip_soft_limit_deg", 8.0);
        mppi_params_.rear_slip_soft_limit =
            this->get_parameter("rear_slip_soft_limit_deg").as_double()
            * static_cast<double>(M_PI) / 180.0;
        this->declare_parameter("rear_slip_cost_min_speed", 1.5);
        mppi_params_.rear_slip_cost_min_speed =
            this->get_parameter("rear_slip_cost_min_speed").as_double();
        this->declare_parameter("q_progress",           13.0);   mppi_params_.q_progress    = this->get_parameter("q_progress").as_double();
        this->declare_parameter("q_escape_vel",         6.5);    mppi_params_.q_escape_vel  = this->get_parameter("q_escape_vel").as_double();
        this->declare_parameter("collision_radius",     0.19);   mppi_params_.collision_radius = this->get_parameter("collision_radius").as_double();
        this->declare_parameter("q_boundary_slack", 15000.0);
        mppi_params_.q_boundary_slack =
            this->get_parameter("q_boundary_slack").as_double();
        this->declare_parameter("q_boundary_terminal_slack", 15000.0);
        mppi_params_.q_boundary_terminal_slack =
            this->get_parameter("q_boundary_terminal_slack").as_double();
        this->declare_parameter("weighted_trajectory_safety_enabled", true);
        mppi_params_.weighted_trajectory_safety_enabled =
            this->get_parameter("weighted_trajectory_safety_enabled").as_bool();
        this->declare_parameter("car_radius",           0.15);   mppi_params_.car_radius    = this->get_parameter("car_radius").as_double();
        this->declare_parameter("obstacle_influence_distance", 1.2); mppi_params_.obstacle_influence_distance=this->get_parameter("obstacle_influence_distance").as_double();
        this->declare_parameter("sudden_obstacle_influence_distance", 2.0); mppi_params_.sudden_obstacle_influence_distance=this->get_parameter("sudden_obstacle_influence_distance").as_double();
        this->declare_parameter("sudden_obstacle_min_clearance", 0.65); mppi_params_.sudden_obstacle_min_clearance=this->get_parameter("sudden_obstacle_min_clearance").as_double();
        this->declare_parameter("sudden_obstacle_candidate_clearance", 0.65); mppi_params_.sudden_obstacle_candidate_clearance=this->get_parameter("sudden_obstacle_candidate_clearance").as_double();
        this->declare_parameter("sudden_obstacle_cost_multiplier", 3.0); mppi_params_.sudden_obstacle_cost_multiplier=this->get_parameter("sudden_obstacle_cost_multiplier").as_double();
        this->declare_parameter("q_obs",                50.0);   mppi_params_.q_obs         = this->get_parameter("q_obs").as_double();
        this->declare_parameter("obstacle_avoidance_enabled", false); obstacle_avoidance_enabled_=this->get_parameter("obstacle_avoidance_enabled").as_bool();
        this->declare_parameter("simulation_obstacle_odom_topic", "/opp_racecar/odom"); simulation_obstacle_odom_topic_=this->get_parameter("simulation_obstacle_odom_topic").as_string();
        this->declare_parameter("real_perception_obstacles_topic", "/f1/perception/object/obstacles/arr"); real_perception_obstacles_topic_=this->get_parameter("real_perception_obstacles_topic").as_string();
        this->declare_parameter("real_perception_obstacles_frame", "map"); real_perception_obstacles_frame_=this->get_parameter("real_perception_obstacles_frame").as_string();
        this->declare_parameter("obstacle_timeout", 0.5); obstacle_timeout_s_=this->get_parameter("obstacle_timeout").as_double();
        this->declare_parameter("sudden_obstacle_replan_enabled", true); sudden_obstacle_replan_enabled_=this->get_parameter("sudden_obstacle_replan_enabled").as_bool();
        this->declare_parameter("sudden_obstacle_jump_threshold", 0.75); sudden_obstacle_jump_threshold_=this->get_parameter("sudden_obstacle_jump_threshold").as_double();
        this->declare_parameter("sudden_obstacle_replan_distance", 3.0); sudden_obstacle_replan_distance_=this->get_parameter("sudden_obstacle_replan_distance").as_double();
        this->declare_parameter("sudden_obstacle_replan_duration", 0.6); sudden_obstacle_replan_duration_s_=this->get_parameter("sudden_obstacle_replan_duration").as_double();
        this->declare_parameter("noise_steer_std",      0.4);    mppi_params_.noise_steer_std  = this->get_parameter("noise_steer_std").as_double();
        this->declare_parameter("noise_accel_std",      2.0);    mppi_params_.noise_accel_std  = this->get_parameter("noise_accel_std").as_double();
        this->declare_parameter("max_steer_rate",       0.5236); mppi_params_.max_steer_rate   = this->get_parameter("max_steer_rate").as_double();
        this->declare_parameter("max_accel_rate",       1000.0); mppi_params_.max_accel_rate   = this->get_parameter("max_accel_rate").as_double();
        this->declare_parameter("lambda",               10.0);   mppi_params_.lambda        = this->get_parameter("lambda").as_double();
        this->declare_parameter("visualize_candidates", true);   mppi_params_.visualize_candidates = this->get_parameter("visualize_candidates").as_bool();
        this->declare_parameter("boundary_visualization_topic", "/mppi_boundary_viz");
        boundary_visualization_topic_ =
            this->get_parameter("boundary_visualization_topic").as_string();
        this->declare_parameter("mass",   3.74);   mppi_params_.mass = this->get_parameter("mass").as_double();
        this->declare_parameter("l_f",    0.163);  mppi_params_.l_f  = this->get_parameter("l_f").as_double();
        this->declare_parameter("l_r",    0.162);  mppi_params_.l_r  = this->get_parameter("l_r").as_double();
        this->declare_parameter("I_z",    0.04712);mppi_params_.I_z  = this->get_parameter("I_z").as_double();
        this->declare_parameter("kinematic_steer_scale",1.2896732099);
        mppi_params_.kinematic_steer_scale=this->get_parameter("kinematic_steer_scale").as_double();
        this->declare_parameter("kinematic_steer_bias",-0.0347926021);
        mppi_params_.kinematic_steer_bias=this->get_parameter("kinematic_steer_bias").as_double();
        this->declare_parameter("kinematic_position_speed_scale",1.0);
        mppi_params_.kinematic_position_speed_scale=
            this->get_parameter("kinematic_position_speed_scale").as_double();
        this->declare_parameter("kinematic_no_slip",true);
        mppi_params_.kinematic_no_slip=this->get_parameter("kinematic_no_slip").as_bool();
        this->declare_parameter("Cm0",    0.04);   mppi_params_.Cm0  = this->get_parameter("Cm0").as_double();
        this->declare_parameter("speed_servo_kp",8.0);mppi_params_.speed_servo_kp=this->get_parameter("speed_servo_kp").as_double();
        this->declare_parameter("kinematic_yaw_rate_time_constant",0.10);mppi_params_.kinematic_yaw_rate_time_constant=this->get_parameter("kinematic_yaw_rate_time_constant").as_double();
        this->declare_parameter("kinematic_max_yaw_accel",15.0);mppi_params_.kinematic_max_yaw_accel=this->get_parameter("kinematic_max_yaw_accel").as_double();
        this->declare_parameter("steer_servo_time_constant",0.08);mppi_params_.steer_servo_time_constant=this->get_parameter("steer_servo_time_constant").as_double();
        this->declare_parameter("actuator_max_steer_rate",6.0);mppi_params_.actuator_max_steer_rate=this->get_parameter("actuator_max_steer_rate").as_double();
        mppi_params_.actuator_steer_state=0.0f;
        mppi_params_.actuator_speed_reference_state=0.0f;
        this->declare_parameter("speed_reference_accel_time_constant",0.04);mppi_params_.speed_reference_accel_time_constant=this->get_parameter("speed_reference_accel_time_constant").as_double();
        this->declare_parameter("speed_reference_brake_time_constant",0.02);mppi_params_.speed_reference_brake_time_constant=this->get_parameter("speed_reference_brake_time_constant").as_double();
        this->declare_parameter("actuator_max_speed_reference_rate",8.0);mppi_params_.actuator_max_speed_reference_rate=this->get_parameter("actuator_max_speed_reference_rate").as_double();
        // Pacejka 4-파라미터 (ForzaETH On-Track-SysID 규약).
        // D 는 무차원 마찰계수이며 실제 힘은 아래에서 계산하는 정하중 F_z 가 만든다.
        // E=0 이면 예전 3-파라미터 수식과 완전히 동일하다 (mppi_core.cu 주석 참고).
        this->declare_parameter("B_f",    7.2);    mppi_params_.B_f  = this->get_parameter("B_f").as_double();
        this->declare_parameter("C_f",    1.5);    mppi_params_.C_f  = this->get_parameter("C_f").as_double();
        this->declare_parameter("D_f",    0.65);   mppi_params_.D_f  = this->get_parameter("D_f").as_double();
        this->declare_parameter("E_f",    0.0);    mppi_params_.E_f  = this->get_parameter("E_f").as_double();
        this->declare_parameter("B_r",    7.5);    mppi_params_.B_r  = this->get_parameter("B_r").as_double();
        this->declare_parameter("C_r",    1.5);    mppi_params_.C_r  = this->get_parameter("C_r").as_double();
        this->declare_parameter("D_r",    0.65);   mppi_params_.D_r  = this->get_parameter("D_r").as_double();
        this->declare_parameter("E_r",    0.0);    mppi_params_.E_r  = this->get_parameter("E_r").as_double();

        // Dedicated parameters used only by dynamic_mlp_residual.  They must
        // match tune_dynamic_model.py and must not overwrite the KF/legacy I_z.
        this->declare_parameter("dynamic_mlp_B_f", 3.7417837566); mppi_params_.dynamic_mlp_B_f=this->get_parameter("dynamic_mlp_B_f").as_double();
        this->declare_parameter("dynamic_mlp_C_f", 1.5797653815); mppi_params_.dynamic_mlp_C_f=this->get_parameter("dynamic_mlp_C_f").as_double();
        this->declare_parameter("dynamic_mlp_D_f", 0.2488725065); mppi_params_.dynamic_mlp_D_f=this->get_parameter("dynamic_mlp_D_f").as_double();
        this->declare_parameter("dynamic_mlp_E_f",-1.0);          mppi_params_.dynamic_mlp_E_f=this->get_parameter("dynamic_mlp_E_f").as_double();
        this->declare_parameter("dynamic_mlp_B_r", 3.3027640068); mppi_params_.dynamic_mlp_B_r=this->get_parameter("dynamic_mlp_B_r").as_double();
        this->declare_parameter("dynamic_mlp_C_r", 1.9942488896); mppi_params_.dynamic_mlp_C_r=this->get_parameter("dynamic_mlp_C_r").as_double();
        this->declare_parameter("dynamic_mlp_D_r", 0.4480617878); mppi_params_.dynamic_mlp_D_r=this->get_parameter("dynamic_mlp_D_r").as_double();
        this->declare_parameter("dynamic_mlp_E_r",-1.0);          mppi_params_.dynamic_mlp_E_r=this->get_parameter("dynamic_mlp_E_r").as_double();
        this->declare_parameter("dynamic_mlp_I_z", 0.5);          mppi_params_.dynamic_mlp_I_z=this->get_parameter("dynamic_mlp_I_z").as_double();
        this->declare_parameter("dynamic_mlp_min_speed", 0.8);    mppi_params_.dynamic_mlp_min_speed=this->get_parameter("dynamic_mlp_min_speed").as_double();
        this->declare_parameter("model_dt",0.04);mppi_params_.model_dt=this->get_parameter("model_dt").as_double();
        this->declare_parameter("effective_steer_scale",0.51);mppi_params_.effective_steer_scale=this->get_parameter("effective_steer_scale").as_double();
        this->declare_parameter("effective_steer_bias",0.01);mppi_params_.effective_steer_bias=this->get_parameter("effective_steer_bias").as_double();
        this->declare_parameter("effective_yaw_response_tau",0.10);mppi_params_.effective_yaw_response_tau=this->get_parameter("effective_yaw_response_tau").as_double();
        this->declare_parameter("effective_max_yaw_accel",15.0);mppi_params_.effective_max_yaw_accel=this->get_parameter("effective_max_yaw_accel").as_double();
        this->declare_parameter("effective_speed_response_gain",0.76);mppi_params_.effective_speed_response_gain=this->get_parameter("effective_speed_response_gain").as_double();
        this->declare_parameter("effective_max_accel",1.0);mppi_params_.effective_max_accel=this->get_parameter("effective_max_accel").as_double();
        this->declare_parameter("effective_vy_decay_tau",0.12);mppi_params_.effective_vy_decay_tau=this->get_parameter("effective_vy_decay_tau").as_double();

        // 정하중 (pacejka_sysid/helpers/generate_predictions.py 와 동일한 규약)
        //   F_zf = m*g*l_r/l_wb,  F_zr = m*g*l_f/l_wb,  l_wb = l_f + l_r
        {
            const double g_    = 9.81;
            const double l_wb  = mppi_params_.l_f + mppi_params_.l_r;
            mppi_params_.F_zf  = mppi_params_.mass * g_ * mppi_params_.l_r / l_wb;
            mppi_params_.F_zr  = mppi_params_.mass * g_ * mppi_params_.l_f / l_wb;
            RCLCPP_INFO(this->get_logger(),
                        "Pacejka 정하중: F_zf=%.3f N, F_zr=%.3f N (m=%.3f, l_f=%.3f, l_r=%.3f)",
                        mppi_params_.F_zf, mppi_params_.F_zr,
                        mppi_params_.mass, mppi_params_.l_f, mppi_params_.l_r);
        }

        this->declare_parameter("control_rate_hz", 50.0);
        control_rate_hz_ = this->get_parameter("control_rate_hz").as_double();

        this->declare_parameter("is_simulation", true);
        is_simulation_ = this->get_parameter("is_simulation").as_bool();
        this->declare_parameter("simulation_odom_topic", "/ego_racecar/odom");
        simulation_odom_topic_ =
            this->get_parameter("simulation_odom_topic").as_string();
        this->declare_parameter("simulation_drive_topic", "/drive");
        simulation_drive_topic_ =
            this->get_parameter("simulation_drive_topic").as_string();
        this->declare_parameter("real_pose_topic", "/newmcl_pose");
        real_pose_topic_ = this->get_parameter("real_pose_topic").as_string();
        this->declare_parameter("real_odom_topic", "/odom");
        real_odom_topic_ = this->get_parameter("real_odom_topic").as_string();
        this->declare_parameter("real_drive_topic", "/drive");
        real_drive_topic_ = this->get_parameter("real_drive_topic").as_string();
        this->declare_parameter("imu_topic","/imu/data");imu_topic_=this->get_parameter("imu_topic").as_string();
        this->declare_parameter("imu_sync_max_age_s",0.05);
        imu_sync_max_age_s_=this->get_parameter("imu_sync_max_age_s").as_double();
        this->declare_parameter("imu_ema_alpha",0.25);
        imu_ema_alpha_=static_cast<float>(std::clamp(
            this->get_parameter("imu_ema_alpha").as_double(),0.0,1.0));
        this->declare_parameter("imu_wz_sign",1.0);
        imu_wz_sign_=static_cast<float>(this->get_parameter("imu_wz_sign").as_double());
        this->declare_parameter("imu_ax_sign",1.0);
        imu_ax_sign_=static_cast<float>(this->get_parameter("imu_ax_sign").as_double());
        this->declare_parameter("imu_ay_sign",1.0);
        imu_ay_sign_=static_cast<float>(this->get_parameter("imu_ay_sign").as_double());
        this->declare_parameter("kf_min_vx",0.5);lateral_velocity_kf_params_.min_longitudinal_speed=this->get_parameter("kf_min_vx").as_double();
        this->declare_parameter("kf_low_speed_threshold",0.0);
        lateral_velocity_kf_params_.low_speed_threshold=this->get_parameter("kf_low_speed_threshold").as_double();
        mppi_params_.kf_low_speed_threshold=std::max(
            0.0, this->get_parameter("kf_low_speed_threshold").as_double());
        this->declare_parameter("kf_max_abs_vy",2.0);lateral_velocity_kf_params_.max_abs_vy=this->get_parameter("kf_max_abs_vy").as_double();
        this->declare_parameter("kf_q_vy",0.02);lateral_velocity_kf_params_.process_var_vy=this->get_parameter("kf_q_vy").as_double();
        this->declare_parameter("kf_q_yaw_rate",0.02);lateral_velocity_kf_params_.process_var_yaw_rate=this->get_parameter("kf_q_yaw_rate").as_double();
        this->declare_parameter("kf_r_lateral_accel",0.5);lateral_velocity_kf_params_.measurement_var_lateral_accel=this->get_parameter("kf_r_lateral_accel").as_double();
        this->declare_parameter("kf_r_yaw_rate",0.01);lateral_velocity_kf_params_.measurement_var_yaw_rate=this->get_parameter("kf_r_yaw_rate").as_double();
        this->declare_parameter("kf_initial_p_vy",0.25);lateral_velocity_kf_params_.initial_var_vy=this->get_parameter("kf_initial_p_vy").as_double();
        this->declare_parameter("kf_initial_p_yaw_rate",0.10);lateral_velocity_kf_params_.initial_var_yaw_rate=this->get_parameter("kf_initial_p_yaw_rate").as_double();
        this->declare_parameter("imu_lateral_accel_sign",1.0);lateral_velocity_kf_params_.imu_lateral_accel_sign=this->get_parameter("imu_lateral_accel_sign").as_double();
        this->declare_parameter("kf_q_ay_bias",2.8456300e-6);lateral_velocity_kf_params_.process_var_ay_bias=this->get_parameter("kf_q_ay_bias").as_double();
        this->declare_parameter("kf_initial_p_ay_bias",0.008792814);lateral_velocity_kf_params_.initial_var_ay_bias=this->get_parameter("kf_initial_p_ay_bias").as_double();
        this->declare_parameter("kf_max_abs_ay_bias",0.7880429);lateral_velocity_kf_params_.max_abs_ay_bias=this->get_parameter("kf_max_abs_ay_bias").as_double();
        this->declare_parameter("kf_r_pose_vy",0.07933411);lateral_velocity_kf_params_.measurement_var_pose_vy=this->get_parameter("kf_r_pose_vy").as_double();
        this->declare_parameter("kf_pose_vy_gate",1.3457935);lateral_velocity_kf_params_.pose_vy_gate=this->get_parameter("kf_pose_vy_gate").as_double();
        this->declare_parameter("kf_pose_vy_enabled",true);kf_pose_vy_enabled_=this->get_parameter("kf_pose_vy_enabled").as_bool();
        this->declare_parameter("kf_pose_vy_window_s",0.12);kf_pose_vy_window_s_=this->get_parameter("kf_pose_vy_window_s").as_double();
        this->declare_parameter("kf_pose_vy_max_age_s",0.06);kf_pose_vy_max_age_s_=this->get_parameter("kf_pose_vy_max_age_s").as_double();
        this->declare_parameter("kf_reset_gap_s",0.5);kf_reset_gap_s_=this->get_parameter("kf_reset_gap_s").as_double();
        this->declare_parameter("kf_steer_scale",1.1058064699);kf_steer_scale_=this->get_parameter("kf_steer_scale").as_double();
        this->declare_parameter("kf_steer_bias",-0.0300696939);kf_steer_bias_=this->get_parameter("kf_steer_bias").as_double();
        this->declare_parameter("kf_max_steer",0.4788);kf_max_steer_=this->get_parameter("kf_max_steer").as_double();
        this->declare_parameter("csv_file_path", "data/map1/map1_centerline.csv");
        csv_file_path_ = this->get_parameter("csv_file_path").as_string();

        mppi_params_.dt            = 1.0 / std::max(1.0, control_rate_hz_);
        mppi_params_.control_dt    = mppi_params_.dt;
        lateral_velocity_kf_params_.mass=mppi_params_.mass;lateral_velocity_kf_params_.yaw_inertia=mppi_params_.I_z;
        lateral_velocity_kf_params_.l_f=mppi_params_.l_f;lateral_velocity_kf_params_.l_r=mppi_params_.l_r;
        lateral_velocity_kf_params_.pacejka_b_front=mppi_params_.dynamic_mlp_B_f;
        lateral_velocity_kf_params_.pacejka_c_front=mppi_params_.dynamic_mlp_C_f;
        lateral_velocity_kf_params_.pacejka_d_front=mppi_params_.dynamic_mlp_D_f;
        lateral_velocity_kf_params_.pacejka_e_front=mppi_params_.dynamic_mlp_E_f;
        lateral_velocity_kf_params_.pacejka_b_rear=mppi_params_.dynamic_mlp_B_r;
        lateral_velocity_kf_params_.pacejka_c_rear=mppi_params_.dynamic_mlp_C_r;
        lateral_velocity_kf_params_.pacejka_d_rear=mppi_params_.dynamic_mlp_D_r;
        lateral_velocity_kf_params_.pacejka_e_rear=mppi_params_.dynamic_mlp_E_r;
        lateral_velocity_kf_params_.dt=mppi_params_.dt;
        mppi_params_.num_obstacles = 0;
    }

    void validate_parameters() {
        if (mppi_params_.min_speed > mppi_params_.max_speed)
            std::swap(mppi_params_.min_speed, mppi_params_.max_speed);
        if (horizon_steps_ < 1) horizon_steps_ = 1;
        if (mppi_params_.lambda <= 0.0f) mppi_params_.lambda = 1.0f;
        if (mppi_params_.max_accel_rate <= 0.0f)
            mppi_params_.max_accel_rate = 1.5f;
        heading_speed_limit_gain_ = std::max(0.0f, heading_speed_limit_gain_);
        contour_speed_limit_gain_ = std::max(0.0f, contour_speed_limit_gain_);
        if (mppi_params_.collision_radius < 0.0f)
            mppi_params_.collision_radius = std::abs(mppi_params_.collision_radius);
        if (mppi_params_.q_boundary_slack < 0.0f)
            mppi_params_.q_boundary_slack = 0.0f;
        if (mppi_params_.q_boundary_terminal_slack < 0.0f)
            mppi_params_.q_boundary_terminal_slack = 0.0f;
        if (mppi_params_.dynamic_mlp_min_speed < 0.0f)
            mppi_params_.dynamic_mlp_min_speed = 0.0f;
        if(mppi_params_.dynamics_model==mppi::EFFECTIVE_HISTORY_STATE_RESIDUAL &&
           (std::abs(mppi_params_.control_dt-.02f)>1e-6f || std::abs(mppi_params_.model_dt-.04f)>1e-6f))
            throw std::invalid_argument("effective_history_state_residual requires control_dt=0.02 and model_dt=0.04");
        if((mppi_params_.dynamics_model==mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
            mppi_params_.dynamics_model==mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D) &&
           (std::abs(mppi_params_.control_dt-.02f)>1e-6f || std::abs(mppi_params_.model_dt-.04f)>1e-6f))
            throw std::invalid_argument("dynamic_mlp_residual_servo_lag requires control_dt=0.02 and model_dt=0.04");
        if(mppi_params_.dynamics_model==mppi::EFFECTIVE_HISTORY_STATE_RESIDUAL &&
           (mppi_params_.effective_yaw_response_tau<=0.f ||
            mppi_params_.effective_max_yaw_accel<=0.f ||
            mppi_params_.effective_speed_response_gain<=0.f ||
            mppi_params_.effective_max_accel<=0.f ||
            mppi_params_.effective_vy_decay_tau<=0.f))
            throw std::invalid_argument("effective_history_state_residual effective response parameters must be positive");
    }

    bool path_received_{false}, left_bnd_received_{false}, right_bnd_received_{false};
    int slack_boundary_publish_remaining_{0};

    static std::string trim_csv_cell(const std::string &value) {
        const auto first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return {};
        const auto last = value.find_last_not_of(" \t\r\n");
        return value.substr(first, last-first+1);
    }

    static std::vector<std::string> split_csv_row(const std::string &line) {
        std::vector<std::string> cells;
        std::stringstream stream(line);
        std::string cell;
        while(std::getline(stream,cell,',')) cells.push_back(trim_csv_cell(cell));
        return cells;
    }

    static int csv_column(const std::vector<std::string> &headers,
                          std::initializer_list<const char*> names) {
        for(const char *name:names)
            for(std::size_t i=0;i<headers.size();++i)
                if(headers[i]==name) return static_cast<int>(i);
        return -1;
    }

    static float csv_float(const std::vector<std::string> &cells,int column,
                           const std::string &path,std::size_t line_number) {
        if(column<0 || column>=static_cast<int>(cells.size()))
            throw std::runtime_error("Missing CSV value at "+path+":"+std::to_string(line_number));
        try { return std::stof(cells[column]); }
        catch(const std::exception&) {
            throw std::runtime_error("Invalid CSV number at "+path+":"+std::to_string(line_number));
        }
    }

    void load_track_csv() {
        if(csv_file_path_.empty()) throw std::invalid_argument("csv_file_path must not be empty");
        std::string path=csv_file_path_;
        if(path.front()!='/')
            path=ament_index_cpp::get_package_share_directory("smppi_cuda_controller")+"/"+path;
        std::ifstream file(path);
        if(!file) throw std::runtime_error("Cannot open MPPI track CSV: "+path);

        std::string line;
        if(!std::getline(file,line)) throw std::runtime_error("Empty MPPI track CSV: "+path);
        auto headers=split_csv_row(line);
        for(auto &header:headers)
            std::transform(header.begin(),header.end(),header.begin(),
                [](unsigned char c){return static_cast<char>(std::tolower(c));});
        const int ix=csv_column(headers,{"x_m","x","x_map"});
        const int iy=csv_column(headers,{"y_m","y","y_map"});
        const int iyaw=csv_column(headers,{"psi_rad","psi","yaw","heading_rad"});
        const int iwl=csv_column(headers,{"w_tr_left_m","w_left_m","left_width_m"});
        const int iwr=csv_column(headers,{"w_tr_right_m","w_right_m","right_width_m"});
        const int ilx=csv_column(headers,{"left_x_m","left_x"});
        const int ily=csv_column(headers,{"left_y_m","left_y"});
        const int irx=csv_column(headers,{"right_x_m","right_x"});
        const int iry=csv_column(headers,{"right_y_m","right_y"});
        if(ix<0 || iy<0 || iwl<0 || iwr<0)
            throw std::runtime_error("Track CSV requires x_m,y_m,w_tr_left_m,w_tr_right_m: "+path);

        std::vector<float> xs,ys,yaws,wl,wr,lx,ly,rx,ry;
        std::size_t line_number=1;
        while(std::getline(file,line)) {
            ++line_number;
            if(trim_csv_cell(line).empty() || trim_csv_cell(line).front()=='#') continue;
            const auto cells=split_csv_row(line);
            xs.push_back(csv_float(cells,ix,path,line_number));
            ys.push_back(csv_float(cells,iy,path,line_number));
            wl.push_back(csv_float(cells,iwl,path,line_number));
            wr.push_back(csv_float(cells,iwr,path,line_number));
            yaws.push_back(iyaw>=0?csv_float(cells,iyaw,path,line_number):0.0f);
            if(ilx>=0 && ily>=0 && irx>=0 && iry>=0) {
                lx.push_back(csv_float(cells,ilx,path,line_number));
                ly.push_back(csv_float(cells,ily,path,line_number));
                rx.push_back(csv_float(cells,irx,path,line_number));
                ry.push_back(csv_float(cells,iry,path,line_number));
            }
        }
        if(xs.size()<3) throw std::runtime_error("Track CSV needs at least 3 rows: "+path);
        if(iyaw<0) for(std::size_t i=0;i<xs.size();++i) {
            const std::size_t prev=(i+xs.size()-1)%xs.size(),next=(i+1)%xs.size();
            yaws[i]=std::atan2(ys[next]-ys[prev],xs[next]-xs[prev]);
        }
        if(lx.size()!=xs.size()) for(std::size_t i=0;i<xs.size();++i) {
            const float nx=-std::sin(yaws[i]),ny=std::cos(yaws[i]);
            lx.push_back(xs[i]+nx*wl[i]); ly.push_back(ys[i]+ny*wl[i]);
            rx.push_back(xs[i]-nx*wr[i]); ry.push_back(ys[i]-ny*wr[i]);
        }

        constexpr std::size_t MAX_CUDA_PATH_POINTS=1000;
        const std::size_t source_count=xs.size();
        const std::size_t output_count=std::min(source_count,MAX_CUDA_PATH_POINTS);
        auto sample=[&](const std::vector<float>& source) {
            std::vector<float> output; output.reserve(output_count);
            for(std::size_t i=0;i<output_count;++i)
                output.push_back(source[(i*source_count)/output_count]);
            return output;
        };
        ref_path_xs_=sample(xs); ref_path_ys_=sample(ys); ref_path_yaws_=sample(yaws);
        left_xs_=sample(lx); left_ys_=sample(ly); right_xs_=sample(rx); right_ys_=sample(ry);
        solver_->set_reference_path(ref_path_xs_,ref_path_ys_,ref_path_yaws_);
        solver_->set_boundaries(left_xs_,left_ys_,right_xs_,right_ys_);
        path_received_=left_bnd_received_=right_bnd_received_=true;
        slack_boundary_publish_remaining_=5;
        RCLCPP_INFO(this->get_logger(),"Loaded MPPI track CSV: %s (%zu -> %zu points)",
                    path.c_str(),source_count,output_count);
    }

    void publish_slack_boundaries() {
        const size_t count = std::min({
            ref_path_xs_.size(), ref_path_ys_.size(), ref_path_yaws_.size(),
            left_xs_.size(), left_ys_.size(), right_xs_.size(), right_ys_.size()});
        if (count < 2) return;

        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker left_marker;
        visualization_msgs::msg::Marker right_marker;
        left_marker.header.frame_id = right_marker.header.frame_id = "map";
        left_marker.header.stamp = right_marker.header.stamp = this->now();
        left_marker.ns = right_marker.ns = "boundary_slack_zero";
        left_marker.id = 200;
        right_marker.id = 201;
        left_marker.type = right_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        left_marker.action = right_marker.action = visualization_msgs::msg::Marker::ADD;
        left_marker.pose.orientation.w = right_marker.pose.orientation.w = 1.0;
        left_marker.scale.x = right_marker.scale.x = 0.045;
        left_marker.color.r = 1.0f;
        left_marker.color.g = 0.35f;
        left_marker.color.a = 1.0f;
        right_marker.color.r = 0.85f;
        right_marker.color.b = 1.0f;
        right_marker.color.a = 1.0f;
        left_marker.points.reserve(count + 1);
        right_marker.points.reserve(count + 1);

        for (size_t index = 0; index < count; ++index) {
            const float normal_x = -std::sin(ref_path_yaws_[index]);
            const float normal_y =  std::cos(ref_path_yaws_[index]);
            const float left_width = std::hypot(
                left_xs_[index] - ref_path_xs_[index],
                left_ys_[index] - ref_path_ys_[index]);
            const float right_width = std::hypot(
                right_xs_[index] - ref_path_xs_[index],
                right_ys_[index] - ref_path_ys_[index]);
            const float allowed_left = left_width - mppi_params_.collision_radius;
            const float allowed_right = right_width - mppi_params_.collision_radius;

            geometry_msgs::msg::Point left_point;
            geometry_msgs::msg::Point right_point;
            left_point.x = ref_path_xs_[index] + normal_x * allowed_left;
            left_point.y = ref_path_ys_[index] + normal_y * allowed_left;
            right_point.x = ref_path_xs_[index] - normal_x * allowed_right;
            right_point.y = ref_path_ys_[index] - normal_y * allowed_right;
            left_point.z = right_point.z = 0.04;
            left_marker.points.push_back(left_point);
            right_marker.points.push_back(right_point);
        }
        left_marker.points.push_back(left_marker.points.front());
        right_marker.points.push_back(right_marker.points.front());
        markers.markers.push_back(std::move(left_marker));
        markers.markers.push_back(std::move(right_marker));
        boundary_vis_pub_->publish(markers);
    }

    void timer_callback() {
        if (slack_boundary_publish_remaining_ > 0 && path_received_ &&
            left_bnd_received_ && right_bnd_received_) {
            publish_slack_boundaries();
            --slack_boundary_publish_remaining_;
        }
        mppi_params_.sudden_obstacle_replan = sudden_obstacle_replan_enabled_ &&
            this->now() < sudden_obstacle_replan_until_;
        if (obstacle_avoidance_enabled_ && mppi_params_.num_obstacles > 0 &&
            obstacle_timeout_s_ > 0.0 &&
            (this->now() - obstacle_stamp_).seconds() > obstacle_timeout_s_) {
            mppi_params_.num_obstacles = 0;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Obstacle input is stale; disabling obstacle cost until a new sample arrives");
        }
        if (!is_simulation_) {
            if (!pose_received_ || !velocity_received_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Waiting for pose/velocity (%s, %s)...",
                    real_pose_topic_.c_str(), real_odom_topic_.c_str());
                return;
            }
        } else if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for simulator odom (%s)...",
                simulation_odom_topic_.c_str());
            return;
        }
        auto start = std::chrono::high_resolution_clock::now();
        if(uses_legacy_mlp_imu_) {
            mppi_params_.residual_imu[0]=aligned_imu_valid_?aligned_imu_[0]:current_state_.omega;
            mppi_params_.residual_imu[1]=aligned_imu_valid_?aligned_imu_[1]:current_ax_;
            mppi_params_.residual_imu[2]=aligned_imu_valid_?aligned_imu_[2]:current_state_.ay;
            if(command_history_.empty()) command_history_.push_back({last_steer_cmd_,last_speed_cmd_});
            while(command_history_.size()<5) command_history_.push_front(command_history_.front());
            for(int i=0;i<5;++i){mppi_params_.residual_command_history[2*i]=command_history_[i][0];mppi_params_.residual_command_history[2*i+1]=command_history_[i][1];}
        }
        if(uses_command_history_ && !uses_legacy_mlp_imu_) {
            if(command_history_.empty()) command_history_.push_back({last_steer_cmd_,last_speed_cmd_});
            while(command_history_.size()<5) command_history_.push_front(command_history_.front());
            for(int i=0;i<5;++i){mppi_params_.residual_command_history[2*i]=command_history_[i][0];mppi_params_.residual_command_history[2*i+1]=command_history_[i][1];}
            if(mppi_params_.dynamics_model==mppi::EFFECTIVE_HISTORY_STATE_RESIDUAL) {
                while(command_history_.size()<10)command_history_.push_front(command_history_.front());
                for(int i=0;i<10;++i){mppi_params_.effective_command_history[2*i]=command_history_[i][0];mppi_params_.effective_command_history[2*i+1]=command_history_[i][1];}
                current_state_.ax=aligned_imu_valid_?aligned_imu_[1]:current_ax_;
                current_state_.ay=aligned_imu_valid_?aligned_imu_[2]:current_state_.v*current_state_.omega;
                current_state_.vy=0.f; // matches the causal 0813 training input contract
                if(aligned_imu_valid_)current_state_.omega=aligned_imu_[0];
            }
        }
        if(uses_actuator_state_) {
            const float target=std::clamp(mppi_params_.kinematic_steer_scale*last_steer_cmd_+
                mppi_params_.kinematic_steer_bias,-.55f,.55f);
            const float rate=std::clamp((target-mppi_params_.actuator_steer_state)/
                std::max(1e-3f,mppi_params_.steer_servo_time_constant),
                -mppi_params_.actuator_max_steer_rate,mppi_params_.actuator_max_steer_rate);
            mppi_params_.actuator_steer_state=std::clamp(
                mppi_params_.actuator_steer_state+rate*mppi_params_.dt,-.55f,.55f);
            const float speed_tau=last_speed_cmd_>=mppi_params_.actuator_speed_reference_state
                ? mppi_params_.speed_reference_accel_time_constant
                : mppi_params_.speed_reference_brake_time_constant;
            const float speed_reference_rate=std::clamp(
                (last_speed_cmd_-mppi_params_.actuator_speed_reference_state)/std::max(1e-3f,speed_tau),
                -mppi_params_.actuator_max_speed_reference_rate,
                mppi_params_.actuator_max_speed_reference_rate);
            mppi_params_.actuator_speed_reference_state+=speed_reference_rate*mppi_params_.dt;
        }
        if(mppi_params_.dynamics_model==mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D) {
            // The checkpoint was trained with vx samples spaced by one 40 ms
            // model knot. The node runs at 50 Hz, hence append every second
            // callback. Each CUDA candidate then advances this history using
            // its own predicted vx, never a future measurement.
            if(!vx_history_initialized_) {
                for(float &vx:mppi_params_.residual_vx_history)vx=current_state_.v;
                vx_history_initialized_=true;
                vx_history_phase_=0;
            } else {
                vx_history_phase_=(vx_history_phase_+1)&1;
                if(vx_history_phase_==0) {
                    for(int i=0;i<4;++i)
                        mppi_params_.residual_vx_history[i]=mppi_params_.residual_vx_history[i+1];
                    mppi_params_.residual_vx_history[4]=current_state_.v;
                }
            }
        }
        if (is_kinematic_residual_model_) {
            const float beta=std::atan((mppi_params_.l_r/wheelbase_)*std::tan(last_steer_cmd_));
            const float classic_v=std::min(mppi_params_.max_speed,std::max(mppi_params_.min_speed,last_speed_cmd_));
            const float classic_w=current_state_.v*std::cos(beta)*std::tan(last_steer_cmd_)/wheelbase_;
            std::array<float,11> raw={current_state_.v,current_state_.vy,current_state_.omega,
                // The selected checkpoint's three IMU channels are effectively
                // constant in the extracted training set (std was clipped to
                // 1e-5). Keep them at the training mean; feeding odometry into
                // those slots would create O(1e5) normalized outliers.
                last_steer_cmd_,last_speed_cmd_,mppi::residual_weights::feature_mean[5],
                mppi::residual_weights::feature_mean[6],mppi::residual_weights::feature_mean[7],
                classic_v,classic_v*std::sin(beta),classic_w};
            std::array<float,11> normalized{};
            for(int i=0;i<11;++i) normalized[i]=(raw[i]-mppi::residual_weights::feature_mean[i])/
                                                   mppi::residual_weights::feature_std[i];
            residual_history_.push_back(normalized);
            while(residual_history_.size()>RESIDUAL_HISTORY) residual_history_.pop_front();
            std::vector<float> flat; flat.reserve(RESIDUAL_HISTORY*RESIDUAL_FEATURES);
            const auto &pad=residual_history_.front();
            for(size_t n=residual_history_.size();n<RESIDUAL_HISTORY;++n) flat.insert(flat.end(),pad.begin(),pad.end());
            for(const auto &f:residual_history_) flat.insert(flat.end(),f.begin(),f.end());
            solver_->set_residual_history(flat);
        }
        solver_->update_params(mppi_params_);
        mppi::Control u = solver_->solve(current_state_);
        float next_v;
        float published_accel;
        if(direct_speed_model_) {
            const float previous_speed_command = has_published_command_
                ? last_speed_cmd_ : current_state_.v;
            const int nearest_idx = update_nearest_index(current_state_);
            const float dx = current_state_.x - ref_path_xs_[nearest_idx];
            const float dy = current_state_.y - ref_path_ys_[nearest_idx];
            const float ref_yaw = ref_path_yaws_[nearest_idx];
            const float contour_error = -std::sin(ref_yaw) * dx + std::cos(ref_yaw) * dy;
            const float heading_error = std::atan2(
                std::sin(current_state_.yaw - ref_yaw),
                std::cos(current_state_.yaw - ref_yaw));
            const float heading_speed_limit = mppi_params_.max_speed /
                (1.0f + heading_speed_limit_gain_ * std::abs(heading_error));
            const float contour_speed_limit = mppi_params_.max_speed /
                (1.0f + contour_speed_limit_gain_ * std::abs(contour_error));
            const float safety_speed_limit = std::clamp(
                std::min(heading_speed_limit, contour_speed_limit),
                0.0f, mppi_params_.max_speed);
            const float recovery_speed_floor = std::min(
                mppi_params_.min_speed, safety_speed_limit);
            const float desired_speed = std::clamp(
                u.accel, recovery_speed_floor, safety_speed_limit);
            const float requested_speed = std::clamp(desired_speed,
                previous_speed_command-direct_speed_step_,
                previous_speed_command+direct_speed_step_);
            next_v=std::clamp(requested_speed, 0.0f, mppi_params_.max_speed);
            published_accel=std::clamp(
                (next_v-current_state_.v)/mppi_params_.dt,
                mppi_params_.min_accel,
                mppi_params_.max_accel);
        } else {
            next_v=current_state_.v+u.accel*mppi_params_.dt;
            if(next_v<=mppi_params_.min_speed){u.accel=(mppi_params_.min_speed-current_state_.v)/mppi_params_.dt;next_v=mppi_params_.min_speed;}
            else if(next_v>=mppi_params_.max_speed){u.accel=(mppi_params_.max_speed-current_state_.v)/mppi_params_.dt;next_v=mppi_params_.max_speed;}
            published_accel=u.accel;
        }

        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->now(); drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle          = u.steer;
        drive_msg.drive.steering_angle_velocity = 1.0;
        drive_msg.drive.speed                   = next_v;
        drive_msg.drive.acceleration            = published_accel;
        drive_pub_->publish(drive_msg);
        has_published_command_=true;
        last_steer_cmd_=u.steer; last_speed_cmd_=next_v;
        command_history_.push_back({last_steer_cmd_,last_speed_cmd_});
        const size_t command_history_limit=mppi_params_.dynamics_model==mppi::EFFECTIVE_HISTORY_STATE_RESIDUAL?10:5;
        while(command_history_.size()>command_history_limit)command_history_.pop_front();

        if (mppi_params_.visualize_candidates) { publish_path_visualization(); publish_mppi_trajectory(); }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        static int count = 0;
        if (count++ % 10 == 0)
            RCLCPP_INFO(this->get_logger(), "MPPI: %.2fms | V: %.2f", elapsed.count(), current_state_.v);
    }

    void publish_mppi_trajectory() {
        const auto &bt = solver_->get_weighted_control_trajectory();
        const auto &oc = solver_->get_optimal_controls();
        if (bt.empty() || oc.empty()) return;
        smppi_cuda_controller::msg::MppiTrajectory msg;
        msg.header.stamp = this->now(); msg.header.frame_id = "map";
        int T = solver_->get_T();
        msg.steer.reserve(T); msg.accel.reserve(T);
        msg.predicted_x.reserve(T); msg.predicted_y.reserve(T);
        msg.predicted_yaw.reserve(T); msg.predicted_v.reserve(T);
        msg.predicted_vy.reserve(T); msg.predicted_yaw_rate.reserve(T);
        for (int t = 0; t < T; ++t) {
            msg.steer.push_back(oc[t].steer);
            msg.accel.push_back(oc[t].accel);
            msg.predicted_x.push_back(bt[t].x);
            msg.predicted_y.push_back(bt[t].y);
            msg.predicted_yaw.push_back(bt[t].yaw);
            msg.predicted_v.push_back(bt[t].v);
            msg.predicted_vy.push_back(bt[t].vy);
            msg.predicted_yaw_rate.push_back(bt[t].omega);
        }
        append_best_traj_costs(bt, oc, msg);
        traj_pub_->publish(msg);
    }

    std_msgs::msg::ColorRGBA get_speed_color(float v, float alpha) {
        std_msgs::msg::ColorRGBA c; c.a = alpha;
        float t = std::max(0.f, std::min(1.f, v / mppi_params_.max_speed));
        if (t < 0.5f) { c.r = 2.f*t; c.g = 2.f*t; c.b = 1.f-2.f*t; }
        else           { c.r = 1.f;   c.g = 2.f*(1.f-t); c.b = 0.f; }
        return c;
    }

    void publish_path_visualization() {
        visualization_msgs::msg::MarkerArray markers;
        const auto &states = solver_->get_generated_trajectories();
        const auto &costs  = solver_->get_costs();
        int K = solver_->get_K(), T = solver_->get_T();

        visualization_msgs::msg::Marker tm;
        tm.header.frame_id = "map"; tm.header.stamp = this->now();
        tm.ns = "candidates"; tm.id = 0;
        tm.type = visualization_msgs::msg::Marker::LINE_LIST;
        tm.action = visualization_msgs::msg::Marker::ADD; tm.scale.x = 0.02;
        if ((int)costs.size() == K) {
            std::vector<int> idx(K); for (int k=0;k<K;++k) idx[k]=k;
            std::sort(idx.begin(), idx.end(), [&costs](int a, int b){ return costs[a]<costs[b]; });
            for (int i = 0; i < std::min(50,K); ++i) {
                int k = idx[i];
                for (int t = 1; t < T-2; ++t) {
                    int id = k*T+t;
                    geometry_msgs::msg::Point p1, p2;
                    p1.x=states[id].x; p1.y=states[id].y;
                    p2.x=states[id+1].x; p2.y=states[id+1].y;
                    auto col = get_speed_color(states[id].v, 0.3f);
                    tm.points.push_back(p1); tm.colors.push_back(col);
                    tm.points.push_back(p2); tm.colors.push_back(col);
                }
            }
        }
        markers.markers.push_back(tm);

        visualization_msgs::msg::Marker bm;
        bm.header.frame_id = "map"; bm.header.stamp = this->now();
        bm.ns = "best_trajectory"; bm.id = 1;
        bm.type = visualization_msgs::msg::Marker::LINE_LIST;
        bm.action = visualization_msgs::msg::Marker::ADD; bm.scale.x = 0.06;
        const auto &bt = solver_->get_best_trajectory();
        if (!bt.empty()) {
            for (int t = 0; t < (int)bt.size()-1; ++t) {
                geometry_msgs::msg::Point p1, p2;
                p1.x=bt[t].x; p1.y=bt[t].y; p2.x=bt[t+1].x; p2.y=bt[t+1].y;
                auto bc = get_speed_color(bt[t].v, 1.0f);
                bm.points.push_back(p1); bm.colors.push_back(bc);
                bm.points.push_back(p2); bm.colors.push_back(bc);
            }
            markers.markers.push_back(bm);
        }

        visualization_msgs::msg::Marker wm;
        wm.header.frame_id = "map"; wm.header.stamp = this->now();
        wm.ns = "weighted_control_trajectory"; wm.id = 2;
        wm.type = visualization_msgs::msg::Marker::LINE_LIST;
        wm.action = visualization_msgs::msg::Marker::ADD; wm.scale.x = 0.075;
        const auto &wt = solver_->get_weighted_control_trajectory();
        if (!wt.empty()) {
            std_msgs::msg::ColorRGBA green;
            green.r = 0.05f; green.g = 1.0f; green.b = 0.15f; green.a = 1.0f;
            for (int t = 0; t < (int)wt.size()-1; ++t) {
                geometry_msgs::msg::Point p1, p2;
                p1.x=wt[t].x; p1.y=wt[t].y; p2.x=wt[t+1].x; p2.y=wt[t+1].y;
                wm.points.push_back(p1); wm.colors.push_back(green);
                wm.points.push_back(p2); wm.colors.push_back(green);
            }
            markers.markers.push_back(wm);
        }
        for (int i = 0; i < mppi_params_.num_obstacles; ++i) {
            visualization_msgs::msg::Marker obstacle;
            obstacle.header.frame_id = "map";
            obstacle.header.stamp = this->now();
            obstacle.ns = "mppi_obstacles";
            obstacle.id = 100 + i;
            obstacle.type = visualization_msgs::msg::Marker::CYLINDER;
            obstacle.action = visualization_msgs::msg::Marker::ADD;
            obstacle.pose.position.x = mppi_params_.obs_x[i];
            obstacle.pose.position.y = mppi_params_.obs_y[i];
            obstacle.pose.position.z = 0.05;
            obstacle.pose.orientation.w = 1.0;
            obstacle.scale.x = 2.0f * mppi_params_.car_radius;
            obstacle.scale.y = 2.0f * mppi_params_.car_radius;
            obstacle.scale.z = 0.10;
            obstacle.color.r = 1.0f;
            obstacle.color.g = 0.1f;
            obstacle.color.b = 0.05f;
            obstacle.color.a = 0.35f;
            markers.markers.push_back(obstacle);
        }

        // MarkerArray does not replace the previous marker set.  An ADD for
        // the current obstacles therefore leaves a marker behind whenever a
        // perception message contains fewer obstacles (including an empty
        // array), or when the obstacle input times out.  Explicitly delete
        // the IDs that were published in the previous cycle but are no
        // longer active so RViz reflects the same obstacle set used by MPPI.
        for (int i = mppi_params_.num_obstacles;
             i < published_obstacle_marker_count_; ++i) {
            visualization_msgs::msg::Marker deleted_obstacle;
            deleted_obstacle.header.frame_id = "map";
            deleted_obstacle.header.stamp = this->now();
            deleted_obstacle.ns = "mppi_obstacles";
            deleted_obstacle.id = 100 + i;
            deleted_obstacle.action = visualization_msgs::msg::Marker::DELETE;
            markers.markers.push_back(deleted_obstacle);
        }
        published_obstacle_marker_count_ = mppi_params_.num_obstacles;
        vis_pub_->publish(markers);
    }

    std::int16_t num_samples_;
    std::int16_t horizon_steps_{80};
    mppi::Params mppi_params_;
    std::unique_ptr<mppi::MPPISolver> solver_;
    mppi::State  current_state_;

    std::vector<float> left_xs_, left_ys_, right_xs_, right_ys_;
    std::vector<float> ref_path_xs_, ref_path_ys_, ref_path_yaws_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;  // 단일 구독
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr obstacle_odom_sub_;
    rclcpp::Subscription<f1_msgs::msg::F1stateArr>::SharedPtr perception_obstacles_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mcl_pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr velocity_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr       vis_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr       boundary_vis_pub_;
    rclcpp::Publisher<smppi_cuda_controller::msg::MppiTrajectory>::SharedPtr traj_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    std::string simulation_odom_topic_, simulation_drive_topic_;
    std::string real_pose_topic_, real_odom_topic_, real_drive_topic_;
    std::string selected_drive_topic_, csv_file_path_, imu_topic_, dynamics_model_name_, residual_weights_path_,mlp_weights_path_;
    std::string boundary_visualization_topic_;
    std::string simulation_obstacle_odom_topic_;
    std::string real_perception_obstacles_topic_, real_perception_obstacles_frame_;
    std::string kinematic_noslip_noimu_weights_path_,slip_kinematic_with_imu_weights_path_,dynamic_imu_weights_path_,dynamic_mlp_weights_path_,dynamic_mlp_servo_lag_weights_path_,dynamic_mlp_vx_delta_weights_path_,e2e_weights_path_,effective_history_weights_path_;
    mppi::LateralVelocityKF lateral_velocity_kf_;
    mppi::LateralVelocityKFParams lateral_velocity_kf_params_;
    float kf_steer_scale_{1.1058064699f},kf_steer_bias_{-0.0300696939f},kf_max_steer_{0.4788f};
    bool odom_received_ = false;
    bool obstacle_avoidance_enabled_{false};
    double obstacle_timeout_s_{0.5};
    rclcpp::Time obstacle_stamp_{0, 0, RCL_ROS_TIME};
    int published_obstacle_marker_count_{0};
    bool sudden_obstacle_replan_enabled_{true};
    bool has_obstacle_measurement_{false};
    float last_obstacle_x_{0.0f},last_obstacle_y_{0.0f};
    double sudden_obstacle_jump_threshold_{0.75};
    double sudden_obstacle_replan_distance_{3.0};
    double sudden_obstacle_replan_duration_s_{0.6};
    rclcpp::Time sudden_obstacle_replan_until_{0, 0, RCL_ROS_TIME};
    bool is_simulation_{true}, pose_received_{false}, velocity_received_{false};
    bool has_prev_velocity_{false};
    rclcpp::Time last_velocity_stamp_{0, 0, RCL_ROS_TIME};
    bool has_prev_odom_{false};
    rclcpp::Time last_odom_stamp_{0, 0, RCL_ROS_TIME};
    double last_odom_x_{0.0};
    double last_odom_y_{0.0};
    double last_odom_yaw_{0.0};
    double control_rate_hz_{50.0};
    float heading_speed_limit_gain_{8.0f};
    float contour_speed_limit_gain_{2.0f};
    bool has_published_command_{false};
    bool direct_speed_model_{false};
    bool uses_command_history_{false};
    bool uses_legacy_mlp_imu_{false};
    bool uses_actuator_state_{false};
    bool uses_lateral_velocity_kf_{false};
    bool vx_history_initialized_{false};
    int vx_history_phase_{0};
    bool is_kinematic_residual_model_{false};
    float wheelbase_{0.0f};
    float direct_speed_step_{0.0f};
    double last_body_vx_{0.0};
    float last_steer_cmd_{0.f}, last_speed_cmd_{0.f}, current_ax_{0.f};
    std::deque<std::array<float,RESIDUAL_FEATURES>> residual_history_;
    std::deque<std::array<float,2>> command_history_;
    struct ImuSample { rclcpp::Time stamp; float wz,ax,ay; };
    std::deque<ImuSample> imu_buffer_;
    static constexpr std::size_t imu_buffer_capacity_=400;
    std::array<float,3> aligned_imu_{0.f,0.f,0.f};
    double imu_sync_max_age_s_{0.05};
    float imu_ema_alpha_{0.25f};
    float imu_wz_sign_{1.f},imu_ax_sign_{1.f},imu_ay_sign_{1.f};
    bool imu_received_{false},aligned_imu_valid_{false},imu_ema_initialized_{false};
    struct PoseSample { rclcpp::Time stamp; double x,y; };
    struct PoseVySample { rclcpp::Time stamp; float vy; };
    std::deque<PoseSample> pose_history_;
    std::deque<PoseVySample> pose_vy_buffer_;
    static constexpr std::size_t pose_vy_buffer_capacity_=100;
    double kf_pose_vy_window_s_{0.12},kf_pose_vy_max_age_s_{0.06},kf_reset_gap_s_{0.5};
    bool kf_pose_vy_enabled_{true};
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPINode>());
    rclcpp::shutdown();
    return 0;
}
