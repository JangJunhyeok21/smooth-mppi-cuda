#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include "smppi_cuda_controller/msg/mppi_trajectory.hpp"
#include "f1_msgs/msg/f1state_arr.hpp"
#include "f1_msgs/msg/f1state.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

using namespace std::chrono_literals;

class MPPINode : public rclcpp::Node {
public:
    MPPINode() : Node("smppi_controller") {
        load_parameters();
        validate_parameters();

        solver_ = std::make_unique<mppi::MPPISolver>(num_samples_, 50, mppi_params_);

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic_, 10);
        vis_pub_   = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi_viz", 50);
        traj_pub_  = this->create_publisher<smppi_cuda_controller::msg::MppiTrajectory>("/mppi_optimal_trajectory", 10);

        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            path_topic_, 1,
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

        // ── 동적 장애물 구독 ─────────────────────────────────────────
        obs_sub_ = this->create_subscription<f1_msgs::msg::F1stateArr>(
            "/f1/perception/object/obstacles/arr", 10,
            std::bind(&MPPINode::obs_callback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(
            35ms, std::bind(&MPPINode::timer_callback, this));

        RCLCPP_INFO(this->get_logger(),
            "MPPI Node Started — single EKF odom topic: %s", odom_topic_.c_str());
    }

private:
    // ════════════════════════════════════════════════════════════════
    //  단일 odom_callback
    //  /ekf_odom 에서 pose(x,y,yaw) + twist(vx,vy,omega) 동시 처리
    // ════════════════════════════════════════════════════════════════
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // 위치 & 헤딩 (EKF가 map 프레임으로 보장)
        const auto &ori = msg->pose.pose.orientation;
        double yaw = std::atan2(
            2.0 * (ori.w * ori.z + ori.x * ori.y),
            1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z));

        current_state_.x   = static_cast<float>(msg->pose.pose.position.x);
        current_state_.y   = static_cast<float>(msg->pose.pose.position.y);
        current_state_.yaw = static_cast<float>(yaw);

        // 속도 (body 프레임)
        current_state_.v     = static_cast<float>(msg->twist.twist.linear.x);
        current_state_.vy    = static_cast<float>(msg->twist.twist.linear.y);
        current_state_.omega = static_cast<float>(msg->twist.twist.angular.z);

        // 파생 상태
        current_state_.slip_angle =
            std::atan2(current_state_.vy, std::fabs(current_state_.v) + 1e-5f);
        current_state_.ay = current_state_.v * current_state_.omega;

        odom_received_ = true;
    }

    // ── 장애물 콜백: 최신 장애물 목록 저장 ──────────────────────────
    void obs_callback(const f1_msgs::msg::F1stateArr::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(obs_mutex_);
        latest_obstacles_ = msg->f1_state_arr;
    }

    // ── CPU 베지에 제어점 계산 ────────────────────────────────────────
    void compute_bezier_obstacles()
    {
        std::lock_guard<std::mutex> lock(obs_mutex_);
        mppi_params_.num_obs = 0;
        for (int i = 0; i < MAX_OBS; ++i) mppi_params_.obstacles[i].detected = false;

        if (ref_path_xs_.empty()) return;

        float horizon = 50 * mppi_params_.dt;   // T * dt = 1.75s
        int n = std::min((int)latest_obstacles_.size(), MAX_OBS);

        for (int i = 0; i < n; ++i) {
            const auto &obs = latest_obstacles_[i];
            auto &os = mppi_params_.obstacles[i];

            os.x = (float)obs.x;
            os.y = (float)obs.y;
            os.theta = (float)obs.yaw;

            // perception이 v_x/v_y를 채우지 않으므로 v + yaw로 직접 유도
            float speed = (float)obs.v;
            os.vx = speed * std::cos(os.theta);
            os.vy = speed * std::sin(os.theta);

            if (std::abs(speed) < 0.2f) {
                // 정적 장애물: 베지에를 현재 위치로 고정
                os.p0 = os.p1 = os.p2 = os.p3 = {os.x, os.y};
            } else {
                // 동적 장애물: 등속 예측 → 센터라인 스냅 → Hermite 베지에
                float px = os.x + os.vx * horizon;
                float py = os.y + os.vy * horizon;

                int snap_idx = 0;
                float min_d2 = std::numeric_limits<float>::max();
                for (int j = 0; j < (int)ref_path_xs_.size(); ++j) {
                    float dx = px - ref_path_xs_[j];
                    float dy = py - ref_path_ys_[j];
                    float d2 = dx*dx + dy*dy;
                    if (d2 < min_d2) { min_d2 = d2; snap_idx = j; }
                }
                float sx = ref_path_xs_[snap_idx];
                float sy = ref_path_ys_[snap_idx];
                float theta_T = ref_path_yaws_[snap_idx];
                // 제어점이 트랙 밖으로 벗어나지 않도록 alpha 클램핑
                float alpha = std::min(0.333f * std::hypot(sx - os.x, sy - os.y), 2.0f);

                os.p0 = {os.x, os.y};
                os.p1 = {os.x + alpha * std::cos(os.theta),
                          os.y + alpha * std::sin(os.theta)};
                os.p3 = {sx, sy};
                os.p2 = {sx - alpha * std::cos(theta_T),
                          sy - alpha * std::sin(theta_T)};
            }
            os.detected = true;
            mppi_params_.num_obs++;
        }
    }

    // ── 멀티모달 히스테리시스 활성화 로직 ────────────────────────────
    void update_multimodal_hysteresis()
    {
        std::lock_guard<std::mutex> lock(obs_mutex_);
        if (latest_obstacles_.empty()) {
            multimodal_active_ = false;
            mppi_params_.multimodal_enabled = false;
            return;
        }

        float dist_min = std::numeric_limits<float>::max();
        float v_obs = 0.0f;
        for (const auto &obs : latest_obstacles_) {
            float d = std::hypot(current_state_.x - (float)obs.x,
                                 current_state_.y - (float)obs.y);
            if (d < dist_min) { dist_min = d; v_obs = (float)obs.v; }
        }

        float activate_dist   = mppi_params_.modal_activation_dist;  // 2.0m
        float deactivate_dist = activate_dist + 0.5f;                 // 2.5m
        bool gaining = (current_state_.v - v_obs) > 0.3f;

        if (multimodal_active_ && dist_min > deactivate_dist)
            multimodal_active_ = false;
        else if (!multimodal_active_ && dist_min < activate_dist && gaining)
            multimodal_active_ = true;

        mppi_params_.multimodal_enabled = multimodal_active_;
    }

    // ────────────────────────────────────────────────────────────────
    //  이하 기존 로직 (변경 없음)
    // ────────────────────────────────────────────────────────────────
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
        float speed_err  = s.v - ref_path_vs_[idx];
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
        this->declare_parameter("num_samples",          8000);
        num_samples_ = this->get_parameter("num_samples").as_int();
        this->declare_parameter("max_steer",            0.507);  mppi_params_.max_steer     = this->get_parameter("max_steer").as_double();
        this->declare_parameter("min_accel",            -9.0);   mppi_params_.min_accel     = this->get_parameter("min_accel").as_double();
        this->declare_parameter("max_accel",            9.0);    mppi_params_.max_accel     = this->get_parameter("max_accel").as_double();
        this->declare_parameter("min_speed",            0.0);    mppi_params_.min_speed     = this->get_parameter("min_speed").as_double();
        this->declare_parameter("target_speed",         6.0);    mppi_params_.target_speed  = this->get_parameter("target_speed").as_double();
        this->declare_parameter("max_speed",            10.0);   mppi_params_.max_speed     = this->get_parameter("max_speed").as_double();
        this->declare_parameter("q_dist",               1.5);    mppi_params_.q_dist        = this->get_parameter("q_dist").as_double();
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
        this->declare_parameter("q_obs_gauss",          200.0);  mppi_params_.q_obs_gauss          = this->get_parameter("q_obs_gauss").as_double();
        this->declare_parameter("sigma_x",              1.0);    mppi_params_.sigma_x               = this->get_parameter("sigma_x").as_double();
        this->declare_parameter("sigma_y",              0.5);    mppi_params_.sigma_y               = this->get_parameter("sigma_y").as_double();
        this->declare_parameter("modal_steer_offset",   0.15);   mppi_params_.modal_steer_offset    = this->get_parameter("modal_steer_offset").as_double();
        this->declare_parameter("modal_activation_dist",2.0);    mppi_params_.modal_activation_dist = this->get_parameter("modal_activation_dist").as_double();
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
        this->declare_parameter("Cm0",    0.04);   mppi_params_.Cm0  = this->get_parameter("Cm0").as_double();
        this->declare_parameter("B_f",    14.0);   mppi_params_.B_f  = this->get_parameter("B_f").as_double();
        this->declare_parameter("C_f",    1.5);    mppi_params_.C_f  = this->get_parameter("C_f").as_double();
        this->declare_parameter("D_f",    19.0);   mppi_params_.D_f  = this->get_parameter("D_f").as_double();
        this->declare_parameter("B_r",    14.0);   mppi_params_.B_r  = this->get_parameter("B_r").as_double();
        this->declare_parameter("C_r",    1.5);    mppi_params_.C_r  = this->get_parameter("C_r").as_double();
        this->declare_parameter("D_r",    17.0);   mppi_params_.D_r  = this->get_parameter("D_r").as_double();

        // 기본값을 /ekf_odom 으로 변경 — EKF가 pose+twist를 하나로 발행
        this->declare_parameter("odom_topic",   "/ekf_odom");
        odom_topic_  = this->get_parameter("odom_topic").as_string();
        this->declare_parameter("drive_topic",  "/ackermann_cmd0");
        drive_topic_ = this->get_parameter("drive_topic").as_string();
        this->declare_parameter("path_topic",   "/mppi_target_path");
        path_topic_  = this->get_parameter("path_topic").as_string();

        mppi_params_.dt            = 0.035;
        mppi_params_.num_obstacles = 0;
        mppi_params_.num_obs       = 0;
        mppi_params_.multimodal_enabled = false;
        for (int i = 0; i < MAX_OBS; ++i) mppi_params_.obstacles[i].detected = false;
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
        std::vector<float> xs, ys, yaws, vs;
        xs.reserve(msg->poses.size()); ys.reserve(msg->poses.size());
        yaws.reserve(msg->poses.size()); vs.reserve(msg->poses.size());
        for (const auto &p : msg->poses) {
            xs.push_back(p.pose.position.x); ys.push_back(p.pose.position.y);
            double yaw = atan2(2.0*(p.pose.orientation.w*p.pose.orientation.z
                                  + p.pose.orientation.x*p.pose.orientation.y),
                               1.0 - 2.0*(p.pose.orientation.y*p.pose.orientation.y
                                         + p.pose.orientation.z*p.pose.orientation.z));
            yaws.push_back((float)yaw);
            float rv = (float)p.pose.position.z;
            vs.push_back(rv > 0.1f ? rv : mppi_params_.target_speed);
        }
        ref_path_xs_ = xs; ref_path_ys_ = ys; ref_path_yaws_ = yaws; ref_path_vs_ = vs;
        solver_->set_reference_path(xs, ys, yaws, vs);
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
        auto start = std::chrono::high_resolution_clock::now();
        update_multimodal_hysteresis();
        compute_bezier_obstacles();
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
        for (int t = 0; t < T; ++t) { msg.steer.push_back(oc[t].steer); msg.accel.push_back(oc[t].accel); }
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

    // 동적 장애물 관련
    rclcpp::Subscription<f1_msgs::msg::F1stateArr>::SharedPtr obs_sub_;
    std::vector<f1_msgs::msg::F1state> latest_obstacles_;
    std::mutex obs_mutex_;
    bool multimodal_active_ = false;

    std::vector<float> left_xs_, left_ys_, right_xs_, right_ys_;
    std::vector<float> ref_path_xs_, ref_path_ys_, ref_path_yaws_, ref_path_vs_;

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     left_bnd_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     right_bnd_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;  // 단일 구독

    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr       vis_pub_;
    rclcpp::Publisher<smppi_cuda_controller::msg::MppiTrajectory>::SharedPtr traj_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    std::string odom_topic_, drive_topic_, path_topic_;
    bool odom_received_ = false;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPINode>());
    rclcpp::shutdown();
    return 0;
}