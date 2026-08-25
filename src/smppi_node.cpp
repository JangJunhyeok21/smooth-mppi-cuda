#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include "cuda_mppi_controller/kinematic_residual_weights.hpp"
#include "cuda_mppi_controller/lateral_velocity_kf.hpp"
#include "smppi_cuda_controller/msg/mppi_trajectory.hpp"
#include "smppi_cuda_controller/msg/kf_state.hpp"
#include "smppi_cuda_controller/msg/dynamic_obstacle_trajectory.hpp"
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
        selected_drive_topic_ = is_simulation_
            ? simulation_drive_topic_ : real_drive_topic_;
        log_startup_io_configuration();

        if (uses_lateral_velocity_kf_) {
            lateral_velocity_kf_.initialize(lateral_velocity_kf_params_);
        }

        solver_ = std::make_unique<mppi::MPPISolver>(num_samples_, horizon_steps_, mppi_params_);
        if (mppi_params_.dynamics_model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG)
            solver_->load_dynamic_mlp_residual_weights(dynamic_mlp_servo_lag_weights_path_);

        load_track_csv();
        load_safe_set_csv();
        RCLCPP_INFO(this->get_logger(),"MPPI objective mode: %s",
                    objective_mode_name_.c_str());

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            selected_drive_topic_, 10);
        vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            visualization_topic_, 50);
        if (publish_optimal_trajectory_) {
            traj_pub_ = this->create_publisher<smppi_cuda_controller::msg::MppiTrajectory>(
                optimal_trajectory_topic_, 10);
        }
        // Sensor inputs must prefer the newest sample instead of draining a
        // stale queue after a busy MPPI cycle.
        const auto live_state_qos = rclcpp::SensorDataQoS().keep_last(1);
        // Keep reliable compatibility with Foxglove/ros2 topic subscribers,
        // while depth 1 prevents old estimator telemetry from accumulating.
        const auto kf_state_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
        kf_state_pub_ = this->create_publisher<smppi_cuda_controller::msg::KfState>(
            kf_state_topic_, kf_state_qos);

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
                simulation_odom_topic_, live_state_qos,
                std::bind(&MPPINode::odom_callback, this, std::placeholders::_1));
        }
        if (obstacle_avoidance_enabled_) {
            if (dynamic_obstacle_prediction_enabled_) {
                dynamic_obstacle_sub_ = this->create_subscription<
                    smppi_cuda_controller::msg::DynamicObstacleTrajectory>(
                    dynamic_obstacle_trajectory_topic_, 10,
                    std::bind(&MPPINode::dynamic_obstacle_callback, this,
                              std::placeholders::_1));
                RCLCPP_INFO(this->get_logger(),
                    "Dynamic obstacle trajectory input: %s",
                    dynamic_obstacle_trajectory_topic_.c_str());
            }
            if (!dynamic_obstacle_prediction_enabled_) {
                RCLCPP_WARN(this->get_logger(),
                    "Obstacle avoidance has no direct simulator/perception fallback; "
                    "enable the predictor trajectory input");
            }
        }
        // The no-IMU rollout neither subscribes to nor waits for IMU.  Keep the
        // subscription only for the legacy 21-feature MLP checkpoint.
        if (mppi_params_.dynamics_model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
            mppi_params_.dynamics_model == mppi::DYNAMIC_SERVO_LAG ||
            uses_lateral_velocity_kf_) {
            imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                imu_topic_, live_state_qos,
                std::bind(&MPPINode::imu_callback,this,std::placeholders::_1));
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
    void log_startup_io_configuration() const {
        RCLCPP_INFO(
            this->get_logger(),
            "MPPI startup mode: %s (is_simulation=%s)",
            is_simulation_ ? "SIMULATOR" : "REAL CAR",
            is_simulation_ ? "true" : "false");

        if (is_simulation_) {
            RCLCPP_INFO(
                this->get_logger(),
                "State input: %s [nav_msgs/msg/Odometry; pose + velocity]",
                simulation_odom_topic_.c_str());
        } else {
            RCLCPP_INFO(
                this->get_logger(),
                "State inputs: pose=%s [geometry_msgs/msg/PoseStamped], "
                "velocity=%s [nav_msgs/msg/Odometry]",
                real_pose_topic_.c_str(), real_odom_topic_.c_str());
        }

        const bool subscribes_to_imu =
            mppi_params_.dynamics_model == mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
            mppi_params_.dynamics_model == mppi::DYNAMIC_SERVO_LAG ||
            uses_lateral_velocity_kf_;
        if (subscribes_to_imu) {
            RCLCPP_INFO(this->get_logger(),
                        "IMU input: %s [sensor_msgs/msg/Imu]",
                        imu_topic_.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "IMU input: disabled");
        }

        if (obstacle_avoidance_enabled_) {
            RCLCPP_INFO(
                this->get_logger(), "Obstacle input: %s%s",
                dynamic_obstacle_prediction_enabled_
                    ? dynamic_obstacle_trajectory_topic_.c_str()
                    : "disabled (predictor required)",
                dynamic_obstacle_prediction_enabled_ &&
                        predictor_static_obstacles_only_
                    ? " [static entries only]" : "");
        } else {
            RCLCPP_INFO(this->get_logger(), "Obstacle input: disabled");
        }
        RCLCPP_INFO(
            this->get_logger(),
            "Drive command output: %s [ackermann_msgs/msg/AckermannDriveStamped]",
            selected_drive_topic_.c_str());
    }

    void publish_kf_state(const rclcpp::Time &stamp) {
        if(!kf_state_pub_||!lateral_velocity_kf_.isInitialized())return;
        smppi_cuda_controller::msg::KfState msg;
        msg.header.stamp=stamp;msg.header.frame_id="map";
        msg.x=lateral_velocity_kf_.getState(0);msg.y=lateral_velocity_kf_.getState(1);
        msg.yaw=lateral_velocity_kf_.getState(2);msg.vx=lateral_velocity_kf_.getState(3);
        msg.vy=lateral_velocity_kf_.getState(4);msg.yaw_rate=lateral_velocity_kf_.getYawRate();
        msg.ax=lateral_velocity_kf_.getAx();msg.ay=lateral_velocity_kf_.getAy();
        for(int row=0;row<6;++row)for(int col=0;col<6;++col)
            msg.covariance[row*6+col]=lateral_velocity_kf_.getCovariance(row,col);
        kf_state_pub_->publish(msg);
    }

    void dynamic_obstacle_callback(const
        smppi_cuda_controller::msg::DynamicObstacleTrajectory::SharedPtr msg) {
        const int horizon = static_cast<int>(msg->horizon);
        const int obstacle_count = static_cast<int>(msg->obstacle_ids.size());
        // An empty predictor message explicitly clears a just-expired
        // clicked-point obstacle instead of waiting for the stale timeout.
        if (obstacle_count == 0) {
            solver_->set_dynamic_obstacles({}, {}, {}, {}, {}, {}, 0, 0);
            dynamic_obstacle_active_ = false;
            dynamic_obstacle_count_ = 0;
            dynamic_obstacle_horizon_ = 0;
            return;
        }
        std::size_t expected = 0;
        if (msg->is_dynamic.size() == static_cast<std::size_t>(obstacle_count)) {
            for (const bool is_dynamic : msg->is_dynamic) {
                expected += is_dynamic ? static_cast<std::size_t>(
                    std::max(0, horizon)) : 1U;
            }
        }
        if (horizon <= 0 || horizon > MAX_DYNAMIC_OBSTACLE_HORIZON ||
            obstacle_count <= 0 || obstacle_count > MAX_OBS ||
            msg->x.size() != expected || msg->y.size() != expected ||
            msg->yaw.size() != expected || msg->semi_major.size() != expected ||
            msg->semi_minor.size() != expected ||
            msg->is_dynamic.size() != static_cast<std::size_t>(obstacle_count)) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Invalid dynamic obstacle trajectory: obstacles=%d horizon=%d "
                "x/y/yaw/major/minor/dynamic=%zu/%zu/%zu/%zu/%zu/%zu",
                obstacle_count, horizon, msg->x.size(), msg->y.size(),
                msg->yaw.size(), msg->semi_major.size(), msg->semi_minor.size(),
                msg->is_dynamic.size());
            return;
        }
        const float expected_dt = mppi_params_.model_dt;
        if (std::abs(msg->dt - expected_dt) > 1.0e-4f) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Dynamic obstacle dt %.6f does not match MPPI model_dt %.6f",
                msg->dt, expected_dt);
            return;
        }
        const std::size_t expanded_size = static_cast<std::size_t>(
            obstacle_count * horizon);
        dynamic_obs_x_.clear(); dynamic_obs_x_.reserve(expanded_size);
        dynamic_obs_y_.clear(); dynamic_obs_y_.reserve(expanded_size);
        dynamic_obs_yaw_.clear(); dynamic_obs_yaw_.reserve(expanded_size);
        dynamic_obs_semi_major_.clear();
        dynamic_obs_semi_major_.reserve(expanded_size);
        dynamic_obs_semi_minor_.clear();
        dynamic_obs_semi_minor_.reserve(expanded_size);
        dynamic_obs_is_dynamic_.clear();
        dynamic_obs_is_dynamic_.reserve(obstacle_count);
        std::size_t packed_index = 0;
        int selected_obstacle_count = 0;
        for (int obstacle = 0; obstacle < obstacle_count; ++obstacle) {
            const int packed_points = msg->is_dynamic[obstacle] ? horizon : 1;
            if (predictor_static_obstacles_only_ && msg->is_dynamic[obstacle]) {
                packed_index += static_cast<std::size_t>(packed_points);
                continue;
            }
            for (int step = 0; step < horizon; ++step) {
                const std::size_t source = packed_index +
                    (msg->is_dynamic[obstacle] ? step : 0);
                dynamic_obs_x_.push_back(msg->x[source]);
                dynamic_obs_y_.push_back(msg->y[source]);
                dynamic_obs_yaw_.push_back(msg->yaw[source]);
                dynamic_obs_semi_major_.push_back(msg->semi_major[source]);
                dynamic_obs_semi_minor_.push_back(msg->semi_minor[source]);
            }
            dynamic_obs_is_dynamic_.push_back(msg->is_dynamic[obstacle]);
            ++selected_obstacle_count;
            packed_index += static_cast<std::size_t>(packed_points);
        }
        if (selected_obstacle_count == 0) {
            solver_->set_dynamic_obstacles({}, {}, {}, {}, {}, {}, 0, 0);
            dynamic_obstacle_active_ = false;
            dynamic_obstacle_count_ = 0;
            dynamic_obstacle_horizon_ = 0;
            return;
        }
        dynamic_obstacle_count_ = selected_obstacle_count;
        dynamic_obstacle_horizon_ = horizon;
        solver_->set_dynamic_obstacles(dynamic_obs_x_, dynamic_obs_y_,
            dynamic_obs_yaw_, dynamic_obs_semi_major_, dynamic_obs_semi_minor_,
            dynamic_obs_is_dynamic_, selected_obstacle_count, horizon);
        dynamic_obstacle_stamp_ = this->now();
        dynamic_obstacle_active_ = true;
        // Do not charge the current-pose fallback in addition to its predicted
        // trajectory.
        mppi_params_.num_obstacles = 0;
    }

    void cache_fixed_model_properties() {
        is_kinematic_residual_model_ = false;
        uses_legacy_mlp_imu_ = false;
        direct_speed_model_ = true;
        uses_command_history_ = true;
        uses_actuator_state_ = true;
        uses_lateral_velocity_kf_ = true;
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
        latest_mcl_x_=static_cast<float>(msg->pose.position.x);
        latest_mcl_y_=static_cast<float>(msg->pose.position.y);
        latest_mcl_yaw_=static_cast<float>(yaw);
        if(!uses_lateral_velocity_kf_){current_state_.x=latest_mcl_x_;current_state_.y=latest_mcl_y_;current_state_.yaw=latest_mcl_yaw_;}
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
            const float wz = aligned_imu_valid_ ? aligned_imu_[0] : NAN;
            const float ax = aligned_imu_valid_ ? aligned_imu_[1] : NAN;
            const float ay = aligned_imu_valid_ ? aligned_imu_[2] : NAN;
            const float pose_vy=aligned_pose_vy(stamp);
            lateral_velocity_kf_.update(latest_mcl_x_,latest_mcl_y_,latest_mcl_yaw_,
                current_state_.v,pose_vy,wz,ax,ay,last_steer_cmd_,last_speed_cmd_);
            current_state_.x=lateral_velocity_kf_.getState(0);current_state_.y=lateral_velocity_kf_.getState(1);
            current_state_.yaw=lateral_velocity_kf_.getState(2);current_state_.v=lateral_velocity_kf_.getState(3);
            current_state_.vy=lateral_velocity_kf_.getVy();current_state_.omega=lateral_velocity_kf_.getYawRate();
            current_state_.ax=lateral_velocity_kf_.getAx();current_state_.ay=lateral_velocity_kf_.getAy();
            publish_kf_state(stamp);
        }
        current_state_.slip_angle =
            std::atan2(current_state_.vy, std::fabs(current_state_.v) + 1e-5f);
        if(!uses_lateral_velocity_kf_)current_state_.ay=current_state_.v*current_state_.omega;

        last_body_vx_ = measured_vx;
        last_velocity_stamp_ = stamp;
        has_prev_velocity_ = true;
        velocity_received_ = true;
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        rclcpp::Time stamp(msg->header.stamp);
        if (stamp.nanoseconds() == 0) stamp = this->now();
        // Convert the configured sensor signs to MPPI FLU and remove the
        // stationary bias once at the callback boundary. Every downstream
        // consumer (EMA, vy EKF, residual model) therefore sees exactly the
        // same convention and offset as the training pipeline.
        // Simulator IMU is generated without the stationary bias measured on
        // the real sensor. Applying the real-car calibration here created a
        // constant artificial wz/ax/ay offset in simulation.
        const float wz_bias=is_simulation_?0.f:imu_wz_bias_;
        const float ax_bias=is_simulation_?0.f:imu_ax_bias_;
        const float ay_bias=is_simulation_?0.f:imu_ay_bias_;
        imu_buffer_.push_back({
            stamp,
            imu_wz_sign_*static_cast<float>(msg->angular_velocity.z)-wz_bias,
            imu_ax_sign_*static_cast<float>(msg->linear_acceleration.x)-ax_bias,
            imu_ay_sign_*static_cast<float>(msg->linear_acceleration.y)-ay_bias});
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
            const float wz = aligned_imu_valid_ ? aligned_imu_[0] : NAN;
            const float ax = aligned_imu_valid_ ? aligned_imu_[1] : NAN;
            const float ay = aligned_imu_valid_ ? aligned_imu_[2] : NAN;
            const float pose_vy=aligned_pose_vy(pose_stamp);
            lateral_velocity_kf_.update(current_state_.x,current_state_.y,current_state_.yaw,
                current_state_.v,pose_vy,wz,ax,ay,last_steer_cmd_,last_speed_cmd_);
            current_state_.x=lateral_velocity_kf_.getState(0);current_state_.y=lateral_velocity_kf_.getState(1);
            current_state_.yaw=lateral_velocity_kf_.getState(2);current_state_.v=lateral_velocity_kf_.getState(3);
            current_state_.vy=lateral_velocity_kf_.getVy();current_state_.omega=lateral_velocity_kf_.getYawRate();
            current_state_.ax=lateral_velocity_kf_.getAx();current_state_.ay=lateral_velocity_kf_.getAy();
            publish_kf_state(pose_stamp);
        }
        last_body_vx_ = estimated_vx;

        // 파생 상태
        current_state_.slip_angle =
            std::atan2(current_state_.vy, std::fabs(current_state_.v) + 1e-5f);
        if(!uses_lateral_velocity_kf_)current_state_.ay=current_state_.v*current_state_.omega;

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

        const size_t count = std::min(best_traj.size(), optimal_controls.size());
        float applied_steer = mppi_params_.actuator_steer_state;
        float previous_steer = last_steer_cmd_;
        float previous_accel = last_speed_cmd_;
        float heading=0.f, speed_reward=0.f, overspeed=0.f, error_speed=0.f;
        float friction=0.f, front_slip=0.f, rear_slip=0.f, steer=0.f;
        float control_rate=0.f, boundary=0.f, obstacle=0.f;
        const int initial_path_idx = update_nearest_index(current_state_);
        int last_idx = initial_path_idx;

        for (size_t t=0; t<count; ++t) {
            const auto &s=best_traj[t];
            const auto &u=optimal_controls[t];
            const float max_wheel_steer=mppi_params_.max_steer;
            const float target=std::max(-max_wheel_steer,std::min(
                max_wheel_steer,mppi_params_.kinematic_steer_scale*u.steer+
                mppi_params_.kinematic_steer_bias));
            const float applied_rate=std::clamp((target-applied_steer)/std::max(
                mppi_params_.steer_servo_time_constant,1.0e-3f),
                -mppi_params_.actuator_max_steer_rate,
                mppi_params_.actuator_max_steer_rate);
            applied_steer=std::max(-max_wheel_steer,std::min(
                max_wheel_steer,
                applied_steer+applied_rate*mppi_params_.model_dt));

            // CUDA와 같은 forward local search를 사용한다.
            float nearest_distance=std::numeric_limits<float>::infinity();
            int idx=last_idx;
            for (int offset=0;offset<30;++offset) {
                const int candidate=(last_idx+offset)%static_cast<int>(ref_path_xs_.size());
                const float dx=s.x-ref_path_xs_[candidate];
                const float dy=s.y-ref_path_ys_[candidate];
                const float distance=dx*dx+dy*dy;
                if(distance<nearest_distance){nearest_distance=distance;idx=candidate;}
            }
            last_idx=idx;
            const float dx=s.x-ref_path_xs_[idx],dy=s.y-ref_path_ys_[idx];
            const float ref_cos=std::cos(ref_path_yaws_[idx]);
            const float ref_sin=std::sin(ref_path_yaws_[idx]);
            const float contour=-ref_sin*dx+ref_cos*dy;
            const float lag=ref_cos*dx+ref_sin*dy;
            const float heading_error=std::atan2(
                std::sin(s.yaw-ref_path_yaws_[idx]),std::cos(s.yaw-ref_path_yaws_[idx]));
            if(mppi_params_.objective_mode==mppi::MPCC_OBJECTIVE) {
                heading += mppi_params_.q_dist*nearest_distance
                    +mppi_params_.q_contour*contour*contour
                    +mppi_params_.q_lag*lag*lag
                    +mppi_params_.q_heading*heading_error*heading_error;
                speed_reward -= (mppi_params_.q_v*0.2f)*s.v*std::cos(heading_error);
                const float speed_excess=std::max(0.f,s.v-mppi_params_.max_speed);
                overspeed += mppi_params_.q_v*speed_excess*speed_excess;
                error_speed += mppi_params_.q_error_speed*s.v*s.v*
                    (contour*contour+heading_error*heading_error);
            }

            const float utilization=(s.ax/std::max(mppi_params_.longitudinal_accel_soft_limit,1.e-3f))*
                (s.ax/std::max(mppi_params_.longitudinal_accel_soft_limit,1.e-3f))+
                (s.ay/std::max(mppi_params_.lat_g_soft_limit,1.e-3f))*
                (s.ay/std::max(mppi_params_.lat_g_soft_limit,1.e-3f));
            const float friction_excess=std::max(0.f,utilization-1.f);
            friction += mppi_params_.q_lat_g*friction_excess*friction_excess;
            const float safe_vx=std::max(std::fabs(s.v),0.5f);
            const float alpha_f=applied_steer-std::atan2(s.vy+mppi_params_.l_f*s.omega,safe_vx);
            const float alpha_r=-std::atan2(s.vy-mppi_params_.l_r*s.omega,safe_vx);
            const float front_ratio=std::min(1.0f,std::fabs(s.v)/
                std::max(mppi_params_.front_slip_cost_min_speed,1.0e-3f));
            const float front_gate=front_ratio*front_ratio*(3.0f-2.0f*front_ratio);
            const float front_excess=std::max(
                0.f,std::fabs(alpha_f)-mppi_params_.front_slip_soft_limit);
            front_slip += front_gate*mppi_params_.q_front_slip
                *front_excess*front_excess;
            const float rear_ratio=std::min(1.0f,std::fabs(s.v)/
                std::max(mppi_params_.rear_slip_cost_min_speed,1.0e-3f));
            const float rear_gate=rear_ratio*rear_ratio*(3.0f-2.0f*rear_ratio);
            const float rear_excess=std::max(
                0.f,std::fabs(alpha_r)-mppi_params_.rear_slip_soft_limit);
            rear_slip += rear_gate*mppi_params_.q_rear_slip
                *rear_excess*rear_excess;
            steer += mppi_params_.q_steer*u.steer*u.steer;
            const float ds=u.steer-previous_steer,da=u.accel-previous_accel;
            control_rate += mppi_params_.q_du*(ds*ds+da*da);
            previous_steer=u.steer;previous_accel=u.accel;
            const float slack=std::max(0.f,mppi_params_.collision_radius-
                compute_min_boundary_distance(s,idx));
            boundary += (t+1==count?mppi_params_.q_boundary_terminal_slack:
                mppi_params_.q_boundary_slack)*slack*slack;
            const float obstacle_soft_radius=
                mppi_params_.car_radius+mppi_params_.obstacle_soft_margin;
            for(int i=0;i<mppi_params_.num_obstacles;++i){
                const float ox=s.x-mppi_params_.obs_x[i],oy=s.y-mppi_params_.obs_y[i];
                const float obs_slack=std::max(
                    0.f,obstacle_soft_radius-std::hypot(ox,oy));
                obstacle += mppi_params_.q_obs*obs_slack*obs_slack;
            }
        }

        // Per-step 반환 항목과 terminal progress reward를 분리해 노출한다.
        const float tracking=heading+speed_reward+overspeed+error_speed;
        float progress_cost=0.f;
        if(mppi_params_.objective_mode==mppi::MPCC_OBJECTIVE && count>0) {
            int steps=last_idx-initial_path_idx;
            if(steps<0)steps+=static_cast<int>(ref_path_xs_.size());
            if(steps<=static_cast<int>(ref_path_xs_.size()/2)) {
                float progress_m=0.f;
                int previous=initial_path_idx;
                for(int n=0;n<steps;++n) {
                    const int next=(previous+1)%static_cast<int>(ref_path_xs_.size());
                    progress_m+=std::hypot(ref_path_xs_[next]-ref_path_xs_[previous],
                                           ref_path_ys_[next]-ref_path_ys_[previous]);
                    previous=next;
                }
                const auto tangent_lag=[this](const mppi::State &state,int idx) {
                    const float dx=state.x-ref_path_xs_[idx];
                    const float dy=state.y-ref_path_ys_[idx];
                    return std::max(-0.15f,std::min(0.15f,
                        dx*std::cos(ref_path_yaws_[idx])+dy*std::sin(ref_path_yaws_[idx])));
                };
                progress_m=std::max(0.f,progress_m+
                    tangent_lag(best_traj.back(),last_idx)-
                    tangent_lag(current_state_,initial_path_idx));
                progress_cost=-mppi_params_.q_progress*progress_m;
            }
        }
        msg.tracking_cost=tracking;
        msg.friction_ellipse_cost=friction;
        msg.front_slip_cost=front_slip;
        msg.rear_slip_cost=rear_slip;
        msg.steer_cost=steer;
        msg.rate_cost=control_rate;
        msg.boundary_cost=boundary;
        msg.obs_cost=obstacle;
        msg.progress_cost=progress_cost;
    }

    void load_parameters() {
        this->declare_parameter("dynamics_model", "dynamic_mlp_residual_servo_lag");
        dynamics_model_name_ = this->get_parameter("dynamics_model").as_string();
        this->declare_parameter("dynamic_mlp_servo_lag_weights_path", "config/dynamic_40ms_residual_servo_lag.bin");
        dynamic_mlp_servo_lag_weights_path_=this->get_parameter("dynamic_mlp_servo_lag_weights_path").as_string();
        if (!dynamic_mlp_servo_lag_weights_path_.empty() &&
            dynamic_mlp_servo_lag_weights_path_.front() != '/') {
            dynamic_mlp_servo_lag_weights_path_ =
                ament_index_cpp::get_package_share_directory("smppi_cuda_controller") +
                "/" + dynamic_mlp_servo_lag_weights_path_;
        }
        if (dynamics_model_name_ == "dynamic_mlp_residual_servo_lag") {
            mppi_params_.dynamics_model = mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG;
        } else if (dynamics_model_name_ == "DYNAMIC_SERVO_LAG") {
            mppi_params_.dynamics_model = mppi::DYNAMIC_SERVO_LAG;
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
        this->declare_parameter("q_dist",               1.5);    mppi_params_.q_dist        = this->get_parameter("q_dist").as_double();
        this->declare_parameter("q_contour",            0.5);    mppi_params_.q_contour     = this->get_parameter("q_contour").as_double();
        this->declare_parameter("q_lag",                5.0);    mppi_params_.q_lag         = this->get_parameter("q_lag").as_double();
        this->declare_parameter("q_heading",            12.0);   mppi_params_.q_heading     = this->get_parameter("q_heading").as_double();
        this->declare_parameter("q_error_speed",        8.0);    mppi_params_.q_error_speed = this->get_parameter("q_error_speed").as_double();
        this->declare_parameter("q_v",                  2.0);    mppi_params_.q_v           = this->get_parameter("q_v").as_double();
        this->declare_parameter("q_du",                 0.8);    mppi_params_.q_du          = this->get_parameter("q_du").as_double();
        this->declare_parameter("q_steer",              0.3);    mppi_params_.q_steer       = this->get_parameter("q_steer").as_double();
        this->declare_parameter("q_lat_g",              200.0);  mppi_params_.q_lat_g       = this->get_parameter("q_lat_g").as_double();
        this->declare_parameter("lat_g_soft_limit",     9.81);   mppi_params_.lat_g_soft_limit = this->get_parameter("lat_g_soft_limit").as_double();
        this->declare_parameter("longitudinal_accel_soft_limit", 4.0); mppi_params_.longitudinal_accel_soft_limit = this->get_parameter("longitudinal_accel_soft_limit").as_double();
        this->declare_parameter("q_front_slip", 300.0);
        mppi_params_.q_front_slip = this->get_parameter("q_front_slip").as_double();
        this->declare_parameter("front_slip_soft_limit_deg", 8.0);
        mppi_params_.front_slip_soft_limit =
            this->get_parameter("front_slip_soft_limit_deg").as_double()
            * static_cast<double>(M_PI) / 180.0;
        this->declare_parameter("front_slip_cost_min_speed", 1.5);
        mppi_params_.front_slip_cost_min_speed =
            this->get_parameter("front_slip_cost_min_speed").as_double();
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
        this->declare_parameter("objective_mode","mpcc");
        objective_mode_name_=this->get_parameter("objective_mode").as_string();
        if(objective_mode_name_=="mpcc")mppi_params_.objective_mode=mppi::MPCC_OBJECTIVE;
        else if(objective_mode_name_=="lmpc")mppi_params_.objective_mode=mppi::LMPC_OBJECTIVE;
        else throw std::invalid_argument("objective_mode must be 'mpcc' or 'lmpc'");
        this->declare_parameter("safe_set_file_path","");
        safe_set_file_path_=this->get_parameter("safe_set_file_path").as_string();
        this->declare_parameter("safe_set_k_near",20);
        safe_set_k_near_=this->get_parameter("safe_set_k_near").as_int();
        this->declare_parameter("q_terminal_safe_set_slack",1000.0);
        mppi_params_.q_terminal_safe_set_slack=this->get_parameter("q_terminal_safe_set_slack").as_double();
        this->declare_parameter("safe_set_cost_coefficient",10.0);
        mppi_params_.safe_set_cost_coefficient=this->get_parameter("safe_set_cost_coefficient").as_double();
        this->declare_parameter("safe_set_state_scale_x",1.0);
        this->declare_parameter("safe_set_state_scale_y",1.0);
        this->declare_parameter("safe_set_state_scale_yaw",0.5);
        mppi_params_.safe_set_inv_x_scale=1.0/std::max(1.0e-6,this->get_parameter("safe_set_state_scale_x").as_double());
        mppi_params_.safe_set_inv_y_scale=1.0/std::max(1.0e-6,this->get_parameter("safe_set_state_scale_y").as_double());
        mppi_params_.safe_set_inv_yaw_scale=1.0/std::max(1.0e-6,this->get_parameter("safe_set_state_scale_yaw").as_double());
        mppi_params_.safe_set_count=0;
        this->declare_parameter("car_radius",           0.15);   mppi_params_.car_radius    = this->get_parameter("car_radius").as_double();
        this->declare_parameter("obstacle_soft_margin", 1.0);
        mppi_params_.obstacle_soft_margin =
            this->get_parameter("obstacle_soft_margin").as_double();
        this->declare_parameter("q_obs",             15000.0);  mppi_params_.q_obs         = this->get_parameter("q_obs").as_double();
        this->declare_parameter("obstacle_avoidance_enabled", false); obstacle_avoidance_enabled_=this->get_parameter("obstacle_avoidance_enabled").as_bool();
        this->declare_parameter("dynamic_obstacle_prediction_enabled", false);
        dynamic_obstacle_prediction_enabled_ = this->get_parameter(
            "dynamic_obstacle_prediction_enabled").as_bool();
        this->declare_parameter("predictor_static_obstacles_only", false);
        predictor_static_obstacles_only_ = this->get_parameter(
            "predictor_static_obstacles_only").as_bool();
        this->declare_parameter("dynamic_obstacle_trajectory_topic",
                                "/mppi/dynamic_obstacle_trajectory");
        dynamic_obstacle_trajectory_topic_ = this->get_parameter(
            "dynamic_obstacle_trajectory_topic").as_string();
        this->declare_parameter("obstacle_timeout", 0.5); obstacle_timeout_s_=this->get_parameter("obstacle_timeout").as_double();
        this->declare_parameter("noise_steer_std",      0.4);    mppi_params_.noise_steer_std  = this->get_parameter("noise_steer_std").as_double();
        this->declare_parameter("noise_accel_std",      2.0);    mppi_params_.noise_accel_std  = this->get_parameter("noise_accel_std").as_double();
        this->declare_parameter("max_steer_rate",       0.5236); mppi_params_.max_steer_rate   = this->get_parameter("max_steer_rate").as_double();
        this->declare_parameter("max_accel_rate",       1000.0); mppi_params_.max_accel_rate   = this->get_parameter("max_accel_rate").as_double();
        this->declare_parameter("lambda",               10.0);   mppi_params_.lambda        = this->get_parameter("lambda").as_double();
        this->declare_parameter("visualize_candidates", false);  mppi_params_.visualize_candidates = this->get_parameter("visualize_candidates").as_bool();
        this->declare_parameter("publish_visualization", false);
        publish_visualization_ =
            this->get_parameter("publish_visualization").as_bool();
        this->declare_parameter("visualization_topic", "/mppi_viz");
        visualization_topic_ = this->get_parameter("visualization_topic").as_string();
        this->declare_parameter("optimal_trajectory_topic", "/mppi_optimal_trajectory");
        optimal_trajectory_topic_ = this->get_parameter(
            "optimal_trajectory_topic").as_string();
        this->declare_parameter("kf_state_topic", "/kf_state");
        kf_state_topic_ = this->get_parameter("kf_state_topic").as_string();
        this->declare_parameter("publish_optimal_trajectory", true);
        publish_optimal_trajectory_=
            this->get_parameter("publish_optimal_trajectory").as_bool();
        this->declare_parameter("mass",   3.74);   mppi_params_.mass = this->get_parameter("mass").as_double();
        this->declare_parameter("l_f",    0.163);  mppi_params_.l_f  = this->get_parameter("l_f").as_double();
        this->declare_parameter("l_r",    0.162);  mppi_params_.l_r  = this->get_parameter("l_r").as_double();
        this->declare_parameter("I_z",    0.04712);mppi_params_.I_z  = this->get_parameter("I_z").as_double();
        this->declare_parameter("kinematic_steer_scale",1.0);
        mppi_params_.kinematic_steer_scale=
            this->get_parameter("kinematic_steer_scale").as_double();
        this->declare_parameter("kinematic_steer_bias",0.0);
        mppi_params_.kinematic_steer_bias=
            this->get_parameter("kinematic_steer_bias").as_double();
        this->declare_parameter("kinematic_no_slip",true);
        mppi_params_.kinematic_no_slip=this->get_parameter("kinematic_no_slip").as_bool();
        this->declare_parameter("Cm0",    0.04);   mppi_params_.Cm0  = this->get_parameter("Cm0").as_double();
        this->declare_parameter("speed_servo_kp",8.0);mppi_params_.speed_servo_kp=this->get_parameter("speed_servo_kp").as_double();
        this->declare_parameter("kinematic_yaw_rate_time_constant",0.10);mppi_params_.kinematic_yaw_rate_time_constant=this->get_parameter("kinematic_yaw_rate_time_constant").as_double();
        this->declare_parameter("kinematic_max_yaw_accel",15.0);mppi_params_.kinematic_max_yaw_accel=this->get_parameter("kinematic_max_yaw_accel").as_double();
        this->declare_parameter("steer_servo_time_constant",0.08);mppi_params_.steer_servo_time_constant=this->get_parameter("steer_servo_time_constant").as_double();
        this->declare_parameter("actuator_max_steer_rate",6.544984694978735);mppi_params_.actuator_max_steer_rate=this->get_parameter("actuator_max_steer_rate").as_double();
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
        this->declare_parameter("mlp_max_residual_ax",0.0);mppi_params_.mlp_max_residual_ax=this->get_parameter("mlp_max_residual_ax").as_double();
        this->declare_parameter("mlp_max_residual_ay",8.0);mppi_params_.mlp_max_residual_ay=this->get_parameter("mlp_max_residual_ay").as_double();
        this->declare_parameter("mlp_max_residual_yaw_accel",12.0);mppi_params_.mlp_max_residual_yaw_accel=this->get_parameter("mlp_max_residual_yaw_accel").as_double();
        this->declare_parameter("model_dt",0.04);mppi_params_.model_dt=this->get_parameter("model_dt").as_double();

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
        this->declare_parameter("imu_wz_bias",0.0);imu_wz_bias_=this->get_parameter("imu_wz_bias").as_double();
        this->declare_parameter("imu_ax_bias",0.0);imu_ax_bias_=this->get_parameter("imu_ax_bias").as_double();
        this->declare_parameter("imu_ay_bias",0.0);imu_ay_bias_=this->get_parameter("imu_ay_bias").as_double();
        // Keep the runtime EKF noise contract identical to Step 1/Step 2.
        this->declare_parameter("classic_kf_process_var",
            std::vector<double>{2e-5,2e-5,2e-5,3e-3,1e-2,3e-3});
        this->declare_parameter("classic_kf_measurement_var",
            std::vector<double>{0.015*0.015,0.015*0.015,0.01*0.01,0.025*0.025,
                                0.12*0.12,0.02*0.02,0.35*0.35,0.35*0.35});
        this->declare_parameter("classic_kf_initial_var",
            std::vector<double>{0.01,0.01,0.005,0.03,0.12,0.02});
        const auto kf_q=this->get_parameter("classic_kf_process_var").as_double_array();
        const auto kf_r=this->get_parameter("classic_kf_measurement_var").as_double_array();
        const auto kf_p0=this->get_parameter("classic_kf_initial_var").as_double_array();
        if(kf_q.size()!=6||kf_r.size()!=8||kf_p0.size()!=6)
            throw std::runtime_error("classic KF variance lengths must be Q=6, R=8, P0=6");
        for(int i=0;i<6;++i){
            lateral_velocity_kf_params_.process_var(i)=static_cast<float>(kf_q[i]);
            lateral_velocity_kf_params_.initial_var(i)=static_cast<float>(kf_p0[i]);
        }
        for(int i=0;i<8;++i)
            lateral_velocity_kf_params_.measurement_var(i)=static_cast<float>(kf_r[i]);
        lateral_velocity_kf_params_.mass=mppi_params_.mass;lateral_velocity_kf_params_.lf=mppi_params_.l_f;
        lateral_velocity_kf_params_.lr=mppi_params_.l_r;lateral_velocity_kf_params_.iz=mppi_params_.dynamic_mlp_I_z;
        lateral_velocity_kf_params_.bf=mppi_params_.dynamic_mlp_B_f;lateral_velocity_kf_params_.cf=mppi_params_.dynamic_mlp_C_f;
        lateral_velocity_kf_params_.df=mppi_params_.dynamic_mlp_D_f;lateral_velocity_kf_params_.ef=mppi_params_.dynamic_mlp_E_f;
        lateral_velocity_kf_params_.br=mppi_params_.dynamic_mlp_B_r;lateral_velocity_kf_params_.cr=mppi_params_.dynamic_mlp_C_r;
        lateral_velocity_kf_params_.dr=mppi_params_.dynamic_mlp_D_r;lateral_velocity_kf_params_.er=mppi_params_.dynamic_mlp_E_r;
        lateral_velocity_kf_params_.speed_kp=mppi_params_.speed_servo_kp;
        lateral_velocity_kf_params_.min_accel=mppi_params_.min_accel;lateral_velocity_kf_params_.max_accel=mppi_params_.max_accel;
        lateral_velocity_kf_params_.steer_scale=mppi_params_.kinematic_steer_scale;
        lateral_velocity_kf_params_.steer_bias=mppi_params_.kinematic_steer_bias;
        lateral_velocity_kf_params_.steer_tau=mppi_params_.steer_servo_time_constant;lateral_velocity_kf_params_.max_steer=mppi_params_.max_steer;lateral_velocity_kf_params_.max_steer_rate=mppi_params_.actuator_max_steer_rate;
        lateral_velocity_kf_params_.speed_accel_tau=mppi_params_.speed_reference_accel_time_constant;
        lateral_velocity_kf_params_.speed_brake_tau=mppi_params_.speed_reference_brake_time_constant;
        lateral_velocity_kf_params_.max_speed_rate=mppi_params_.actuator_max_speed_reference_rate;
        this->declare_parameter("kf_pose_vy_enabled",true);kf_pose_vy_enabled_=this->get_parameter("kf_pose_vy_enabled").as_bool();
        this->declare_parameter("kf_pose_vy_window_s",0.12);kf_pose_vy_window_s_=this->get_parameter("kf_pose_vy_window_s").as_double();
        this->declare_parameter("kf_pose_vy_max_age_s",0.06);kf_pose_vy_max_age_s_=this->get_parameter("kf_pose_vy_max_age_s").as_double();
        this->declare_parameter("kf_reset_gap_s",0.5);kf_reset_gap_s_=this->get_parameter("kf_reset_gap_s").as_double();
        this->declare_parameter("csv_file_path", "data/map1/map1_centerline.csv");
        csv_file_path_ = this->get_parameter("csv_file_path").as_string();

        mppi_params_.dt            = 1.0 / std::max(1.0, control_rate_hz_);
        mppi_params_.control_dt    = mppi_params_.dt;
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
        if (mppi_params_.collision_radius < 0.0f)
            mppi_params_.collision_radius = std::abs(mppi_params_.collision_radius);
        mppi_params_.obstacle_soft_margin =
            std::max(0.0f, mppi_params_.obstacle_soft_margin);
        if (mppi_params_.q_boundary_slack < 0.0f)
            mppi_params_.q_boundary_slack = 0.0f;
        if (mppi_params_.q_boundary_terminal_slack < 0.0f)
            mppi_params_.q_boundary_terminal_slack = 0.0f;
        mppi_params_.q_front_slip = std::max(0.0f, mppi_params_.q_front_slip);
        mppi_params_.front_slip_soft_limit =
            std::max(0.0f, mppi_params_.front_slip_soft_limit);
        mppi_params_.front_slip_cost_min_speed =
            std::max(0.0f, mppi_params_.front_slip_cost_min_speed);
        mppi_params_.mlp_max_residual_ax=std::max(0.0f,mppi_params_.mlp_max_residual_ax);
        mppi_params_.mlp_max_residual_ay=std::max(0.0f,mppi_params_.mlp_max_residual_ay);
        mppi_params_.mlp_max_residual_yaw_accel=std::max(0.0f,mppi_params_.mlp_max_residual_yaw_accel);
        if((mppi_params_.dynamics_model==mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
            mppi_params_.dynamics_model==mppi::DYNAMIC_SERVO_LAG) &&
           (std::abs(mppi_params_.control_dt-.02f)>1e-6f || std::abs(mppi_params_.model_dt-.04f)>1e-6f))
            throw std::invalid_argument("servo-lag models require control_dt=0.02 and model_dt=0.04");
    }

    bool path_received_{false}, left_bnd_received_{false}, right_bnd_received_{false};

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
        const int ibrx=csv_column(headers,{"boundary_ref_x_m","boundary_ref_x"});
        const int ibry=csv_column(headers,{"boundary_ref_y_m","boundary_ref_y"});
        if(ix<0 || iy<0 || iwl<0 || iwr<0)
            throw std::runtime_error("Track CSV requires x_m,y_m,w_tr_left_m,w_tr_right_m: "+path);

        std::vector<float> xs,ys,yaws,wl,wr,lx,ly,rx,ry,brx,bry;
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
            if(ibrx>=0 && ibry>=0) {
                brx.push_back(csv_float(cells,ibrx,path,line_number));
                bry.push_back(csv_float(cells,ibry,path,line_number));
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
        if(brx.size()==xs.size()) {
            boundary_ref_xs_=sample(brx); boundary_ref_ys_=sample(bry);
            boundary_ref_yaws_.resize(boundary_ref_xs_.size());
            for(std::size_t i=0;i<boundary_ref_xs_.size();++i) {
                const std::size_t prev=(i+boundary_ref_xs_.size()-1)%boundary_ref_xs_.size();
                const std::size_t next=(i+1)%boundary_ref_xs_.size();
                boundary_ref_yaws_[i]=std::atan2(
                    boundary_ref_ys_[next]-boundary_ref_ys_[prev],
                    boundary_ref_xs_[next]-boundary_ref_xs_[prev]);
            }
        } else {
            boundary_ref_xs_=ref_path_xs_; boundary_ref_ys_=ref_path_ys_;
            boundary_ref_yaws_=ref_path_yaws_;
        }
        solver_->set_reference_path(ref_path_xs_,ref_path_ys_,ref_path_yaws_);
        solver_->set_boundaries(left_xs_,left_ys_,right_xs_,right_ys_);
        path_received_=left_bnd_received_=right_bnd_received_=true;
        // One initial publication is sufficient. A late RViz subscriber is
        // handled event-wise in timer_callback(), without periodic boundary
        // reconstruction or serialization.
        RCLCPP_INFO(this->get_logger(),"Loaded MPPI track CSV: %s (%zu -> %zu points)",
                    path.c_str(),source_count,output_count);
    }

    struct SafeSetSample { float x,y,yaw,v,s,cost; };

    void load_safe_set_csv() {
        if(mppi_params_.objective_mode!=mppi::LMPC_OBJECTIVE) return;
        if(safe_set_file_path_.empty())
            throw std::invalid_argument("objective_mode=lmpc requires safe_set_file_path");
        std::string path=safe_set_file_path_;
        if(path.front()!='/')
            path=ament_index_cpp::get_package_share_directory("smppi_cuda_controller")+"/"+path;
        std::ifstream file(path);
        if(!file) throw std::runtime_error("Cannot open LMPC safe-set CSV: "+path);
        std::string line;if(!std::getline(file,line))throw std::runtime_error("Empty safe-set CSV: "+path);
        auto headers=split_csv_row(line);
        for(auto &h:headers)std::transform(h.begin(),h.end(),h.begin(),
            [](unsigned char c){return static_cast<char>(std::tolower(c));});
        const int it=csv_column(headers,{"time"}),ix=csv_column(headers,{"x"}),
            iy=csv_column(headers,{"y"}),ia=csv_column(headers,{"yaw"}),
            iv=csv_column(headers,{"vel","v"}),is=csv_column(headers,{"theta","s"});
        if(it<0||ix<0||iy<0||ia<0||iv<0||is<0)
            throw std::runtime_error("Safe-set CSV requires time,x,y,yaw,vel,theta: "+path);
        std::vector<SafeSetSample> lap;float previous_time=-1.0f;std::size_t row=1;
        auto finish_lap=[&]() {
            if(lap.size()>100) {
                for(std::size_t i=0;i<lap.size();++i)lap[i].cost=float(lap.size()-1-i);
                safe_set_laps_.push_back(lap);
            }
            lap.clear();
        };
        while(std::getline(file,line)) {
            ++row;if(trim_csv_cell(line).empty())continue;const auto cells=split_csv_row(line);
            const float time=csv_float(cells,it,path,row);if(previous_time>=0.0f&&time<previous_time)finish_lap();
            lap.push_back({csv_float(cells,ix,path,row),csv_float(cells,iy,path,row),
                csv_float(cells,ia,path,row),csv_float(cells,iv,path,row),
                csv_float(cells,is,path,row),0.0f});previous_time=time;
        }
        finish_lap();
        if(safe_set_laps_.size()<2)
            throw std::runtime_error("LMPC requires at least two complete safe-set laps: "+path);
        safe_set_track_length_=0.0f;
        for(const auto &samples:safe_set_laps_)
            for(const auto &sample:samples)
                safe_set_track_length_=std::max(safe_set_track_length_,sample.s);
        if(safe_set_track_length_<=0.0f)
            throw std::runtime_error("Safe-set theta/s must contain a positive lap length: "+path);
        safe_set_k_near_=std::clamp(safe_set_k_near_,2,MAX_SAFE_SET_POINTS/2);
        RCLCPP_INFO(this->get_logger(),"Loaded LMPC safe set: %s (%zu laps, K_NEAR=%d, length=%.3fm)",
            path.c_str(),safe_set_laps_.size(),safe_set_k_near_,safe_set_track_length_);
    }

    void update_local_safe_set() {
        if(mppi_params_.objective_mode!=mppi::LMPC_OBJECTIVE||safe_set_laps_.size()<2)return;
        const auto &reference=safe_set_laps_.back();
        const float lap_length=safe_set_track_length_;
        auto circular_distance=[lap_length](float a,float b) {
            float distance=std::abs(a-b);
            return std::min(distance,lap_length-distance);
        };

        // LMPC uses the previous solution's terminal state as terminal_candidate.
        // Doing the same prevents current-v noise from moving the selected safe
        // set by several samples on every 50 Hz callback.
        const auto &previous_optimal=solver_->get_weighted_control_trajectory();
        const bool have_previous_terminal=!previous_optimal.empty();
        const float query_x=have_previous_terminal?previous_optimal.back().x:current_state_.x;
        const float query_y=have_previous_terminal?previous_optimal.back().y:current_state_.y;
        std::size_t nearest=0;float nearest_d=std::numeric_limits<float>::infinity();
        for(std::size_t i=0;i<reference.size();++i) {
            if(safe_set_terminal_s_initialized_) {
                // Equivalent to SLCMPC local_find_s(): terminal progress may
                // move only as far as the car can travel during one control tick.
                const float max_update=std::max(0.10f,
                    1.5f*mppi_params_.max_speed*mppi_params_.control_dt);
                if(circular_distance(reference[i].s,safe_set_terminal_s_)>max_update)continue;
            }
            const float dx=query_x-reference[i].x,dy=query_y-reference[i].y;
            const float distance=dx*dx+dy*dy;
            if(distance<nearest_d){nearest_d=distance;nearest=i;}
        }
        float terminal_s=reference[nearest].s;
        if(!have_previous_terminal&&!safe_set_terminal_s_initialized_) {
            terminal_s=std::fmod(terminal_s+
                std::max(0.0f,current_state_.v)*
                horizon_steps_*mppi_params_.model_dt,lap_length);
            float best=std::numeric_limits<float>::infinity();
            for(std::size_t i=0;i<reference.size();++i) {
                const float distance=circular_distance(reference[i].s,terminal_s);
                if(distance<best){best=distance;nearest=i;}
            }
            terminal_s=reference[nearest].s;
        }
        safe_set_terminal_s_=terminal_s;
        safe_set_terminal_s_initialized_=true;
        if(have_previous_terminal) {
            safe_set_terminal_candidate_={previous_optimal.back().x,previous_optimal.back().y,
                previous_optimal.back().yaw,previous_optimal.back().v,terminal_s,0.0f};
        } else {
            safe_set_terminal_candidate_=reference[nearest];
        }
        safe_set_terminal_candidate_valid_=true;
        int output=0;
        for(std::size_t lap_index=safe_set_laps_.size()-2;lap_index<safe_set_laps_.size();++lap_index){
            const auto &samples=safe_set_laps_[lap_index];std::size_t center=0;float best=std::numeric_limits<float>::infinity();
            for(std::size_t i=0;i<samples.size();++i){float ds=std::abs(samples[i].s-terminal_s);
                if(lap_length>0.0f)ds=std::min(ds,std::abs(lap_length-ds));
                if(ds<best){best=ds;center=i;}}
            const int first=int(center)-safe_set_k_near_/2+1;
            float minimum_cost=std::numeric_limits<float>::infinity();
            for(int k=0;k<safe_set_k_near_;++k){int index=(first+k)%int(samples.size());if(index<0)index+=samples.size();
                minimum_cost=std::min(minimum_cost,samples[index].cost);}
            for(int k=0;k<safe_set_k_near_;++k){int index=(first+k)%int(samples.size());if(index<0)index+=samples.size();
                const auto &sample=samples[index];mppi_params_.safe_set_x[output]=sample.x;
                mppi_params_.safe_set_y[output]=sample.y;mppi_params_.safe_set_yaw[output]=sample.yaw;
                mppi_params_.safe_set_cost[output]=sample.cost-minimum_cost;++output;}
        }
        mppi_params_.safe_set_count=output;
    }

    void timer_callback() {
        if (dynamic_obstacle_active_ && obstacle_timeout_s_ > 0.0 &&
            (this->now() - dynamic_obstacle_stamp_).seconds() > obstacle_timeout_s_) {
            solver_->set_dynamic_obstacles({}, {}, {}, {}, {}, {}, 0, 0);
            dynamic_obstacle_active_ = false;
            dynamic_obstacle_count_ = 0;
            dynamic_obstacle_horizon_ = 0;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Dynamic obstacle prediction is stale; disabling its cost");
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
        }
        if(uses_actuator_state_) {
            // The deployed 22-D residual was trained with causal IMU ax/ay.
            // Populate the same fields before CUDA rollout.
            if(!uses_lateral_velocity_kf_){
                current_state_.ax=aligned_imu_valid_?aligned_imu_[1]:current_ax_;
                current_state_.ay=aligned_imu_valid_?aligned_imu_[2]
                    :current_state_.v*current_state_.omega;
            }
            const float target=std::clamp(
                mppi_params_.kinematic_steer_scale*last_steer_cmd_+
                mppi_params_.kinematic_steer_bias,
                -mppi_params_.max_steer,mppi_params_.max_steer);
            const float steer_rate=std::clamp(
                (target-mppi_params_.actuator_steer_state)/
                    std::max(1e-3f,mppi_params_.steer_servo_time_constant),
                -mppi_params_.actuator_max_steer_rate,
                mppi_params_.actuator_max_steer_rate);
            mppi_params_.actuator_steer_state=std::clamp(
                mppi_params_.actuator_steer_state+steer_rate*mppi_params_.dt,
                -mppi_params_.max_steer,mppi_params_.max_steer);
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
        update_local_safe_set();
        solver_->update_params(mppi_params_);
        mppi::Control u = solver_->solve(current_state_);
        float next_v;
        float published_accel;
        if(direct_speed_model_) {
            const float previous_speed_command = has_published_command_
                ? last_speed_cmd_ : current_state_.v;
            const float requested_speed = std::clamp(u.accel,
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
        constexpr size_t command_history_limit=5;
        while(command_history_.size()>command_history_limit)command_history_.pop_front();

        // Candidate 복사는 옵션으로 끄되 weighted optimal Marker는 항상 발행한다.
        publish_path_visualization();
        // 대회 모드에서는 trajectory 복사/목적함수 재계산/ROS 직렬화를 모두 생략한다.
        if (publish_optimal_trajectory_) publish_mppi_trajectory();

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
        if (!publish_visualization_) return;
        visualization_msgs::msg::MarkerArray markers;
        if (mppi_params_.visualize_candidates) {
            const auto &states = solver_->get_generated_trajectories();
            const auto &costs  = solver_->get_costs();
            const int K = solver_->get_K(), T = solver_->get_T();
            visualization_msgs::msg::Marker tm;
            tm.header.frame_id = "map"; tm.header.stamp = this->now();
            tm.ns = "candidates"; tm.id = 0;
            tm.type = visualization_msgs::msg::Marker::LINE_LIST;
            tm.action = visualization_msgs::msg::Marker::ADD; tm.scale.x = 0.02;
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
            markers.markers.push_back(std::move(tm));

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
                markers.markers.push_back(std::move(bm));
            }
        }

        // SLCMPC visualize_mpc_solution()와 같은 pose 기반 ARROW 배열이다.
        // 각 화살표는 weighted optimal rollout의 위치·yaw를 직접 나타내고,
        // 색은 저속(blue) -> 중속(green) -> 고속(red)으로 표현한다.
        const auto &wt = solver_->get_weighted_control_trajectory();
        const auto stamp=this->now();
        visualization_msgs::msg::Marker old_weighted_line;
        old_weighted_line.header.frame_id="map";old_weighted_line.header.stamp=stamp;
        old_weighted_line.ns="weighted_control_trajectory";old_weighted_line.id=2;
        old_weighted_line.action=visualization_msgs::msg::Marker::DELETE;
        markers.markers.push_back(std::move(old_weighted_line));
        for (int t=0;t<(int)wt.size();++t) {
            visualization_msgs::msg::Marker arrow;
            arrow.header.frame_id="map";arrow.header.stamp=stamp;
            arrow.ns="mppi_optimal_path";arrow.id=t;
            arrow.type=visualization_msgs::msg::Marker::ARROW;
            arrow.action=visualization_msgs::msg::Marker::ADD;
            arrow.pose.position.x=wt[t].x;arrow.pose.position.y=wt[t].y;
            arrow.pose.position.z=0.04;
            const float half_yaw=0.5f*wt[t].yaw;
            arrow.pose.orientation.z=std::sin(half_yaw);
            arrow.pose.orientation.w=std::cos(half_yaw);
            arrow.scale.x=0.20; // SLCMPC와 동일한 arrow length
            arrow.scale.y=0.05; // shaft diameter
            arrow.scale.z=0.05; // head diameter
            arrow.color=get_speed_color(wt[t].v,1.0f);
            markers.markers.push_back(std::move(arrow));
        }
        // Horizon을 줄였을 때 이전 frame의 남은 화살표를 RViz에서 제거한다.
        for(int id=(int)wt.size();id<published_optimal_arrow_count_;++id) {
            visualization_msgs::msg::Marker deleted;
            deleted.header.frame_id="map";deleted.header.stamp=stamp;
            deleted.ns="mppi_optimal_path";deleted.id=id;
            deleted.action=visualization_msgs::msg::Marker::DELETE;
            markers.markers.push_back(std::move(deleted));
        }
        published_optimal_arrow_count_=static_cast<int>(wt.size());

        // SLCMPC LMPC visualization과 동일한 selected safe-set POINTS.
        visualization_msgs::msg::Marker safe_points;
        safe_points.header.frame_id="map";safe_points.header.stamp=stamp;
        safe_points.ns="safe_set";safe_points.id=300;
        safe_points.type=visualization_msgs::msg::Marker::POINTS;
        safe_points.action=mppi_params_.objective_mode==mppi::LMPC_OBJECTIVE
            ? visualization_msgs::msg::Marker::ADD
            : visualization_msgs::msg::Marker::DELETE;
        safe_points.pose.orientation.w=1.0;
        safe_points.scale.x=safe_points.scale.y=safe_points.scale.z=0.10;
        if(mppi_params_.objective_mode==mppi::LMPC_OBJECTIVE) {
            safe_points.points.reserve(mppi_params_.safe_set_count);
            safe_points.colors.reserve(mppi_params_.safe_set_count);
            for(int i=0;i<mppi_params_.safe_set_count;++i) {
                geometry_msgs::msg::Point point;
                point.x=mppi_params_.safe_set_x[i];point.y=mppi_params_.safe_set_y[i];
                point.z=0.06;safe_points.points.push_back(point);
                std_msgs::msg::ColorRGBA color;color.a=1.0f;
                if(i<safe_set_k_near_){color.r=0.45f;color.b=1.0f;}
                else {color.r=1.0f;color.b=0.55f;}
                safe_points.colors.push_back(color);
            }
        }
        markers.markers.push_back(std::move(safe_points));

        // SLCMPC의 terminal_candidate SPHERE: 이 점 주변에서 K_NEAR를 뽑는다.
        visualization_msgs::msg::Marker terminal_candidate;
        terminal_candidate.header.frame_id="map";terminal_candidate.header.stamp=stamp;
        terminal_candidate.ns="terminal_candidate";terminal_candidate.id=301;
        terminal_candidate.type=visualization_msgs::msg::Marker::SPHERE;
        terminal_candidate.action=(mppi_params_.objective_mode==mppi::LMPC_OBJECTIVE&&safe_set_terminal_candidate_valid_)
            ? visualization_msgs::msg::Marker::ADD
            : visualization_msgs::msg::Marker::DELETE;
        terminal_candidate.pose.orientation.w=1.0;
        terminal_candidate.pose.position.x=safe_set_terminal_candidate_.x;
        terminal_candidate.pose.position.y=safe_set_terminal_candidate_.y;
        terminal_candidate.pose.position.z=0.08;
        terminal_candidate.scale.x=terminal_candidate.scale.y=terminal_candidate.scale.z=0.30;
        terminal_candidate.color.r=0.5f;terminal_candidate.color.g=1.0f;
        terminal_candidate.color.b=0.1f;terminal_candidate.color.a=0.85f;
        markers.markers.push_back(std::move(terminal_candidate));
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
        if (dynamic_obstacle_active_) {
            for (int obstacle_index = 0; obstacle_index < dynamic_obstacle_count_;
                 ++obstacle_index) {
                visualization_msgs::msg::Marker line;
                line.header.frame_id = "map"; line.header.stamp = this->now();
                line.ns = "mppi_dynamic_obstacles"; line.id = 500 + obstacle_index;
                line.type = visualization_msgs::msg::Marker::LINE_STRIP;
                line.action = visualization_msgs::msg::Marker::ADD;
                line.pose.orientation.w = 1.0; line.scale.x = 0.035;
                line.color.r = 1.0; line.color.g = 0.55; line.color.a = 0.9;
                for (int step = 0; step < dynamic_obstacle_horizon_; ++step) {
                    const int index = obstacle_index * dynamic_obstacle_horizon_ + step;
                    geometry_msgs::msg::Point point;
                    point.x = dynamic_obs_x_[index]; point.y = dynamic_obs_y_[index];
                    point.z = 0.08; line.points.push_back(point);
                }
                markers.markers.push_back(std::move(line));
            }
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
    // Raceline 추종과 무관하게 slack-zero 경계를 원본 centerline 기준으로 표시한다.
    std::vector<float> boundary_ref_xs_, boundary_ref_ys_, boundary_ref_yaws_;
    std::vector<float> ref_path_xs_, ref_path_ys_, ref_path_yaws_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;  // 단일 구독
    rclcpp::Subscription<smppi_cuda_controller::msg::DynamicObstacleTrajectory>::SharedPtr
        dynamic_obstacle_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mcl_pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr velocity_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr       vis_pub_;
    rclcpp::Publisher<smppi_cuda_controller::msg::MppiTrajectory>::SharedPtr traj_pub_;
    rclcpp::Publisher<smppi_cuda_controller::msg::KfState>::SharedPtr kf_state_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    std::string simulation_odom_topic_, simulation_drive_topic_;
    std::string real_pose_topic_, real_odom_topic_, real_drive_topic_;
    std::string selected_drive_topic_, csv_file_path_, imu_topic_, dynamics_model_name_;
    std::string visualization_topic_;
    std::string optimal_trajectory_topic_, kf_state_topic_;
    std::string dynamic_obstacle_trajectory_topic_;
    std::string dynamic_mlp_servo_lag_weights_path_;
    std::string safe_set_file_path_,objective_mode_name_;
    int safe_set_k_near_{20};
    std::vector<std::vector<SafeSetSample>> safe_set_laps_;
    SafeSetSample safe_set_terminal_candidate_{};
    bool safe_set_terminal_candidate_valid_{false};
    float safe_set_track_length_{0.0f};
    float safe_set_terminal_s_{0.0f};
    bool safe_set_terminal_s_initialized_{false};
    mppi::LateralVelocityKF lateral_velocity_kf_;
    mppi::LateralVelocityKFParams lateral_velocity_kf_params_;
    bool odom_received_ = false;
    bool publish_optimal_trajectory_{true};
    bool publish_visualization_{false};
    bool obstacle_avoidance_enabled_{false};
    bool dynamic_obstacle_prediction_enabled_{false};
    bool predictor_static_obstacles_only_{false};
    bool dynamic_obstacle_active_{false};
    int dynamic_obstacle_count_{0};
    int dynamic_obstacle_horizon_{0};
    std::vector<float> dynamic_obs_x_, dynamic_obs_y_, dynamic_obs_yaw_;
    std::vector<float> dynamic_obs_semi_major_, dynamic_obs_semi_minor_;
    std::vector<bool> dynamic_obs_is_dynamic_;
    rclcpp::Time dynamic_obstacle_stamp_{0, 0, RCL_ROS_TIME};
    double obstacle_timeout_s_{0.5};
    int published_obstacle_marker_count_{0};
    int published_optimal_arrow_count_{0};
    bool is_simulation_{true}, pose_received_{false}, velocity_received_{false};
    bool has_prev_velocity_{false};
    rclcpp::Time last_velocity_stamp_{0, 0, RCL_ROS_TIME};
    bool has_prev_odom_{false};
    rclcpp::Time last_odom_stamp_{0, 0, RCL_ROS_TIME};
    double last_odom_x_{0.0};
    double last_odom_y_{0.0};
    double last_odom_yaw_{0.0};
    float latest_mcl_x_{0.f},latest_mcl_y_{0.f},latest_mcl_yaw_{0.f};
    double control_rate_hz_{50.0};
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
    float imu_wz_bias_{0.f},imu_ax_bias_{0.f},imu_ay_bias_{0.f};
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
