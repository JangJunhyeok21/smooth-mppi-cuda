#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include "cuda_mppi_controller/kinematic_residual_weights.hpp"
#include "smppi_cuda_controller/msg/mppi_trajectory.hpp"
#include <algorithm>
#include <cmath>
#include <array>
#include <deque>

using namespace std::chrono_literals;

class MPPINode : public rclcpp::Node {
public:
    MPPINode() : Node("smppi_controller") {
        load_parameters();
        validate_parameters();

        solver_ = std::make_unique<mppi::MPPISolver>(num_samples_, 50, mppi_params_);
        if (mppi_params_.dynamics_model == mppi::KINEMATIC_RESIDUAL)
            solver_->load_residual_weights(residual_weights_path_);
        if (mppi_params_.dynamics_model == mppi::KINEMATIC_MLP_RESIDUAL)
            solver_->load_mlp_residual_weights(mlp_weights_path_);
        if (mppi_params_.dynamics_model == mppi::KINEMATIC_MLP_NO_IMU_RESIDUAL)
            solver_->load_mlp_no_imu_residual_weights(mlp_weights_path_);

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic_, 10);
        vis_pub_   = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi_viz", 50);
        traj_pub_  = this->create_publisher<smppi_cuda_controller::msg::MppiTrajectory>("/mppi_optimal_trajectory", 10);

        auto path_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            path_topic_, path_qos,
            std::bind(&MPPINode::path_callback, this, std::placeholders::_1));

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        left_bnd_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/mppi_left_boundary", qos,
            std::bind(&MPPINode::left_bnd_callback, this, std::placeholders::_1));
        right_bnd_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/mppi_right_boundary", qos,
            std::bind(&MPPINode::right_bnd_callback, this, std::placeholders::_1));

        // ── EKF odom 단일 구독 ──────────────────────────────────────
        // /ekf_odom 한 토픽에서 pose(x,y,yaw) + twist(vx,vy,omega) 모두 수신
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 10,
            std::bind(&MPPINode::odom_callback, this, std::placeholders::_1));
        // The no-IMU rollout neither subscribes to nor waits for IMU.  Keep the
        // subscription only for the legacy 21-feature MLP checkpoint.
        if (mppi_params_.dynamics_model == mppi::KINEMATIC_MLP_RESIDUAL) {
            imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                imu_topic_, 20, std::bind(&MPPINode::imu_callback,this,std::placeholders::_1));
        }

        const auto control_period = std::chrono::milliseconds(static_cast<int>(1000.0 / std::max(1.0, control_rate_hz_)));
        timer_ = this->create_wall_timer(
            control_period, std::bind(&MPPINode::timer_callback, this));

        RCLCPP_INFO(this->get_logger(),
            "MPPI Node Started — single EKF odom topic: %s", odom_topic_.c_str());
    }

private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        rclcpp::Time stamp(msg->header.stamp);
        if (stamp.nanoseconds() == 0) stamp = this->now();
        imu_buffer_.push_back({stamp,static_cast<float>(msg->angular_velocity.z),
                              static_cast<float>(msg->linear_acceleration.x),
                              static_cast<float>(msg->linear_acceleration.y)});
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
        float speed_err  = s.v - mppi_params_.target_speed;
        float overspeed  = (speed_err > 0.f) ? mppi_params_.q_v * speed_err * speed_err : 0.f;

        float d_steer = u.steer - u_prev.steer, d_accel = u.accel - u_prev.accel;
        float ay_abs = fabsf(s.ay);
        float lat_g  = (ay_abs >= 9.5f) ? mppi_params_.q_lat_g * expf(-3.f*(ay_abs-9.5f)) : 0.f;

        float min_bnd   = compute_min_boundary_distance(s, idx);
        float safe_dist = mppi_params_.collision_radius + 0.4f;
        float bnd_cost  = 0.f;
        if (min_bnd < safe_dist) {
            float pen = safe_dist - min_bnd;
            float hrd = 0.f;
            if (min_bnd < mppi_params_.collision_radius * 1.2f)
                hrd = mppi_params_.q_collision *
                      std::log(1.f + std::exp(-40.f*(min_bnd - mppi_params_.collision_radius)));
            bnd_cost = 150.f * pen * pen + hrd;
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
        } else {
            throw std::invalid_argument("Unknown dynamics_model: " + dynamics_model_name_);
        }
        this->declare_parameter("num_samples",          8000);
        num_samples_ = this->get_parameter("num_samples").as_int();
        this->declare_parameter("max_steer",            0.507);  mppi_params_.max_steer     = this->get_parameter("max_steer").as_double();
        this->declare_parameter("min_accel",            -9.0);   mppi_params_.min_accel     = this->get_parameter("min_accel").as_double();
        this->declare_parameter("max_accel",            9.0);    mppi_params_.max_accel     = this->get_parameter("max_accel").as_double();
        this->declare_parameter("min_speed",            0.0);    mppi_params_.min_speed     = this->get_parameter("min_speed").as_double();
        this->declare_parameter("target_speed",         6.0);    mppi_params_.target_speed  = this->get_parameter("target_speed").as_double();
        this->declare_parameter("max_speed",            10.0);   mppi_params_.max_speed     = this->get_parameter("max_speed").as_double();
        this->declare_parameter("q_dist",               1.5);    mppi_params_.q_dist        = this->get_parameter("q_dist").as_double();
        this->declare_parameter("q_contour",            0.5);    mppi_params_.q_contour     = this->get_parameter("q_contour").as_double();
        this->declare_parameter("q_lag",                5.0);    mppi_params_.q_lag         = this->get_parameter("q_lag").as_double();
        this->declare_parameter("q_v",                  2.0);    mppi_params_.q_v           = this->get_parameter("q_v").as_double();
        this->declare_parameter("q_du",                 0.8);    mppi_params_.q_du          = this->get_parameter("q_du").as_double();
        this->declare_parameter("q_steer",              0.3);    mppi_params_.q_steer       = this->get_parameter("q_steer").as_double();
        this->declare_parameter("q_collision",          400.0);  mppi_params_.q_collision   = this->get_parameter("q_collision").as_double();
        this->declare_parameter("q_lat_g",              200.0);  mppi_params_.q_lat_g       = this->get_parameter("q_lat_g").as_double();
        this->declare_parameter("q_progress",           13.0);   mppi_params_.q_progress    = this->get_parameter("q_progress").as_double();
        this->declare_parameter("q_escape_vel",         6.5);    mppi_params_.q_escape_vel  = this->get_parameter("q_escape_vel").as_double();
        this->declare_parameter("collision_radius",     0.19);   mppi_params_.collision_radius = this->get_parameter("collision_radius").as_double();
        this->declare_parameter("car_radius",           0.15);   mppi_params_.car_radius    = this->get_parameter("car_radius").as_double();
        this->declare_parameter("q_obs",                50.0);   mppi_params_.q_obs         = this->get_parameter("q_obs").as_double();
        this->declare_parameter("noise_steer_std",      0.4);    mppi_params_.noise_steer_std  = this->get_parameter("noise_steer_std").as_double();
        this->declare_parameter("noise_accel_std",      2.0);    mppi_params_.noise_accel_std  = this->get_parameter("noise_accel_std").as_double();
        this->declare_parameter("max_steer_rate",       0.5236); mppi_params_.max_steer_rate   = this->get_parameter("max_steer_rate").as_double();
        this->declare_parameter("max_accel_rate",       1000.0); mppi_params_.max_accel_rate   = this->get_parameter("max_accel_rate").as_double();
        this->declare_parameter("lambda",               10.0);   mppi_params_.lambda        = this->get_parameter("lambda").as_double();
        this->declare_parameter("visualize_candidates", true);   mppi_params_.visualize_candidates = this->get_parameter("visualize_candidates").as_bool();
        this->declare_parameter("mass",   3.74);   mppi_params_.mass = this->get_parameter("mass").as_double();
        this->declare_parameter("l_f",    0.163);  mppi_params_.l_f  = this->get_parameter("l_f").as_double();
        this->declare_parameter("l_r",    0.162);  mppi_params_.l_r  = this->get_parameter("l_r").as_double();
        this->declare_parameter("I_z",    0.04712);mppi_params_.I_z  = this->get_parameter("I_z").as_double();
        this->declare_parameter("kinematic_steer_scale",1.2896732099);
        mppi_params_.kinematic_steer_scale=this->get_parameter("kinematic_steer_scale").as_double();
        this->declare_parameter("kinematic_steer_bias",-0.0347926021);
        mppi_params_.kinematic_steer_bias=this->get_parameter("kinematic_steer_bias").as_double();
        this->declare_parameter("kinematic_no_slip",true);
        mppi_params_.kinematic_no_slip=this->get_parameter("kinematic_no_slip").as_bool();
        this->declare_parameter("Cm0",    0.04);   mppi_params_.Cm0  = this->get_parameter("Cm0").as_double();
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

        // simulator 기본 토픽
        this->declare_parameter("odom_topic",   "/ego_racecar/odom");
        odom_topic_  = this->get_parameter("odom_topic").as_string();
        this->declare_parameter("drive_topic",  "/drive");
        drive_topic_ = this->get_parameter("drive_topic").as_string();
        this->declare_parameter("imu_topic","/imu/data");imu_topic_=this->get_parameter("imu_topic").as_string();
        this->declare_parameter("imu_sync_max_age_s",0.05);
        imu_sync_max_age_s_=this->get_parameter("imu_sync_max_age_s").as_double();
        this->declare_parameter("imu_ema_alpha",0.25);
        imu_ema_alpha_=static_cast<float>(std::clamp(
            this->get_parameter("imu_ema_alpha").as_double(),0.0,1.0));
        this->declare_parameter("path_topic",   "/mppi_target_path");
        path_topic_  = this->get_parameter("path_topic").as_string();

        mppi_params_.dt            = 1.0 / std::max(1.0, control_rate_hz_);
        mppi_params_.num_obstacles = 0;
    }

    void validate_parameters() {
        if (mppi_params_.min_speed > mppi_params_.max_speed)
            std::swap(mppi_params_.min_speed, mppi_params_.max_speed);
        if (mppi_params_.lambda <= 0.0f) mppi_params_.lambda = 1.0f;
        if (mppi_params_.collision_radius < 0.0f)
            mppi_params_.collision_radius = std::abs(mppi_params_.collision_radius);
    }

    bool path_received_{false}, left_bnd_received_{false}, right_bnd_received_{false};

    void path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (path_received_ || msg->poses.empty()) return;
        std::vector<float> xs, ys, yaws;
        xs.reserve(msg->poses.size()); ys.reserve(msg->poses.size());
        yaws.reserve(msg->poses.size());
        for (const auto &p : msg->poses) {
            xs.push_back(p.pose.position.x); ys.push_back(p.pose.position.y);
            double yaw = atan2(2.0*(p.pose.orientation.w*p.pose.orientation.z
                                  + p.pose.orientation.x*p.pose.orientation.y),
                               1.0 - 2.0*(p.pose.orientation.y*p.pose.orientation.y
                                         + p.pose.orientation.z*p.pose.orientation.z));
            yaws.push_back((float)yaw);
        }
        ref_path_xs_ = xs; ref_path_ys_ = ys; ref_path_yaws_ = yaws;
        solver_->set_reference_path(xs, ys, yaws);
        RCLCPP_INFO_ONCE(this->get_logger(), "Path received: %zu points", xs.size());
        path_received_ = true;
    }

    void left_bnd_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (left_bnd_received_ || msg->poses.empty()) return;
        left_xs_.clear(); left_ys_.clear();
        for (const auto &p : msg->poses) { left_xs_.push_back(p.pose.position.x); left_ys_.push_back(p.pose.position.y); }
        update_boundaries(); left_bnd_received_ = true;
    }

    void right_bnd_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (right_bnd_received_ || msg->poses.empty()) return;
        right_xs_.clear(); right_ys_.clear();
        for (const auto &p : msg->poses) { right_xs_.push_back(p.pose.position.x); right_ys_.push_back(p.pose.position.y); }
        update_boundaries(); right_bnd_received_ = true;
    }

    void update_boundaries() {
        if (!left_xs_.empty() && !right_xs_.empty() && left_xs_.size() == right_xs_.size())
            solver_->set_boundaries(left_xs_, left_ys_, right_xs_, right_ys_);
    }

    void timer_callback() {
        if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for EKF odom (%s)...", odom_topic_.c_str());
            return;
        }
        if (!path_received_ || !left_bnd_received_ || !right_bnd_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for path/boundaries (%s)...", path_topic_.c_str());
            return;
        }
        auto start = std::chrono::high_resolution_clock::now();
        if(mppi_params_.dynamics_model==mppi::KINEMATIC_MLP_RESIDUAL) {
            mppi_params_.residual_imu[0]=aligned_imu_valid_?aligned_imu_[0]:current_state_.omega;
            mppi_params_.residual_imu[1]=aligned_imu_valid_?aligned_imu_[1]:current_ax_;
            mppi_params_.residual_imu[2]=aligned_imu_valid_?aligned_imu_[2]:current_state_.ay;
            if(command_history_.empty()) command_history_.push_back({last_steer_cmd_,last_speed_cmd_});
            while(command_history_.size()<5) command_history_.push_front(command_history_.front());
            for(int i=0;i<5;++i){mppi_params_.residual_command_history[2*i]=command_history_[i][0];mppi_params_.residual_command_history[2*i+1]=command_history_[i][1];}
        }
        if(mppi_params_.dynamics_model==mppi::KINEMATIC_MLP_NO_IMU_RESIDUAL) {
            if(command_history_.empty()) command_history_.push_back({last_steer_cmd_,last_speed_cmd_});
            while(command_history_.size()<5) command_history_.push_front(command_history_.front());
            for(int i=0;i<5;++i){mppi_params_.residual_command_history[2*i]=command_history_[i][0];mppi_params_.residual_command_history[2*i+1]=command_history_[i][1];}
        }
        if (mppi_params_.dynamics_model == mppi::KINEMATIC_RESIDUAL) {
            const float wb=mppi_params_.l_f+mppi_params_.l_r;
            const float beta=std::atan((mppi_params_.l_r/wb)*std::tan(last_steer_cmd_));
            const float classic_v=std::min(mppi_params_.max_speed,std::max(mppi_params_.min_speed,last_speed_cmd_));
            const float classic_w=current_state_.v*std::cos(beta)*std::tan(last_steer_cmd_)/wb;
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
        float next_v = current_state_.v + u.accel * mppi_params_.dt;
        if      (next_v <= mppi_params_.min_speed) { u.accel = (mppi_params_.min_speed - current_state_.v) / mppi_params_.dt; next_v = mppi_params_.min_speed; }
        else if (next_v >= mppi_params_.max_speed) { u.accel = (mppi_params_.max_speed - current_state_.v) / mppi_params_.dt; next_v = mppi_params_.max_speed; }

        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->now(); drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle          = u.steer;
        drive_msg.drive.steering_angle_velocity = 1.0;
        drive_msg.drive.speed                   = next_v;
        drive_msg.drive.acceleration            = u.accel;
        drive_pub_->publish(drive_msg);
        last_steer_cmd_=u.steer; last_speed_cmd_=next_v;
        command_history_.push_back({last_steer_cmd_,last_speed_cmd_});while(command_history_.size()>5)command_history_.pop_front();

        if (mppi_params_.visualize_candidates) { publish_path_visualization(); publish_mppi_trajectory(); }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        static int count = 0;
        if (count++ % 10 == 0)
            RCLCPP_INFO(this->get_logger(), "MPPI: %.2fms | V: %.2f", elapsed.count(), current_state_.v);
    }

    void publish_mppi_trajectory() {
        const auto &bt = solver_->get_best_trajectory();
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
            msg.steer.push_back(oc[t].steer); msg.accel.push_back(oc[t].accel);
            msg.predicted_x.push_back(bt[t].x); msg.predicted_y.push_back(bt[t].y);
            msg.predicted_yaw.push_back(bt[t].yaw); msg.predicted_v.push_back(bt[t].v);
            msg.predicted_vy.push_back(bt[t].vy); msg.predicted_yaw_rate.push_back(bt[t].omega);
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
        vis_pub_->publish(markers);
    }

    std::int16_t num_samples_;
    mppi::Params mppi_params_;
    std::unique_ptr<mppi::MPPISolver> solver_;
    mppi::State  current_state_;

    std::vector<float> left_xs_, left_ys_, right_xs_, right_ys_;
    std::vector<float> ref_path_xs_, ref_path_ys_, ref_path_yaws_;

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     left_bnd_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     right_bnd_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;  // 단일 구독
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr       vis_pub_;
    rclcpp::Publisher<smppi_cuda_controller::msg::MppiTrajectory>::SharedPtr traj_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    std::string odom_topic_, drive_topic_, path_topic_, imu_topic_, dynamics_model_name_, residual_weights_path_,mlp_weights_path_;
    bool odom_received_ = false;
    bool has_prev_odom_{false};
    rclcpp::Time last_odom_stamp_{0, 0, RCL_ROS_TIME};
    double last_odom_x_{0.0};
    double last_odom_y_{0.0};
    double last_odom_yaw_{0.0};
    double control_rate_hz_{50.0};
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
    bool imu_received_{false},aligned_imu_valid_{false},imu_ema_initialized_{false};
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPINode>());
    rclcpp::shutdown();
    return 0;
}
