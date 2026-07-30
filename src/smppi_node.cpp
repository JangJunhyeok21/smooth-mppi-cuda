#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/string.hpp"
#include "f1_msgs/msg/f1state_arr.hpp"
#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include "cuda_mppi_controller/overtake_fsm.hpp"
#include "smppi_cuda_controller/msg/mppi_trajectory.hpp"
#include <algorithm>
#include <cmath>

using namespace std::chrono_literals;

class MPPINode : public rclcpp::Node {
public:
    MPPINode() : Node("smppi_controller") {
        load_parameters();
        validate_parameters();

        solver_ = std::make_unique<mppi::MPPISolver>(num_samples_, 50, mppi_params_);
        overtake_fsm_ = std::make_unique<mppi::OvertakeFsm>(fsm_cfg_);

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic_, 10);
        vis_pub_   = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi_viz", 50);
        traj_pub_  = this->create_publisher<smppi_cuda_controller::msg::MppiTrajectory>("/mppi_optimal_trajectory", 10);
        fsm_state_pub_ = this->create_publisher<std_msgs::msg::String>("/fsm/state", 10);

        f1_obstacles_sub_ = this->create_subscription<f1_msgs::msg::F1stateArr>(
            f1_obstacles_topic_, 10,
            std::bind(&MPPINode::f1_obstacles_callback, this, std::placeholders::_1));

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

        if (use_mcl_pose_) {
            // ── mcl_pose + odom 분리 구독 (ekf_pose 비의존) ──────────────
            // pose(x,y,yaw)는 MCL에서, twist(vx,vy,omega)는 휠 오도메트리에서 직접 수신
            mcl_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
                pose_topic_, 10,
                std::bind(&MPPINode::mcl_pose_callback, this, std::placeholders::_1));
            velocity_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                velocity_topic_, 10,
                std::bind(&MPPINode::velocity_callback, this, std::placeholders::_1));

            RCLCPP_INFO(this->get_logger(),
                "MPPI Node Started — pose: %s | velocity: %s",
                pose_topic_.c_str(), velocity_topic_.c_str());
        } else {
            // ── odom0 단일 구독 (시뮬레이터 모드) ────────────────────
            // /odom0 한 토픽에서 pose(x,y,yaw) + twist(vx,vy,omega) 모두 수신
            odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                odom_topic_, 10,
                std::bind(&MPPINode::odom_callback, this, std::placeholders::_1));

            RCLCPP_INFO(this->get_logger(),
                "MPPI Node Started — single odom topic: %s", odom_topic_.c_str());
        }

        timer_ = this->create_wall_timer(
            35ms, std::bind(&MPPINode::timer_callback, this));
    }

private:
    // ════════════════════════════════════════════════════════════
    //  mcl_pose_callback — 위치 & 헤딩만 갱신 (use_mcl_pose 모드)
    // ════════════════════════════════════════════════════════════
    void mcl_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        const auto &ori = msg->pose.orientation;
        double yaw = std::atan2(
            2.0 * (ori.w * ori.z + ori.x * ori.y),
            1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z));

        current_state_.x   = static_cast<float>(msg->pose.position.x);
        current_state_.y   = static_cast<float>(msg->pose.position.y);
        current_state_.yaw = static_cast<float>(yaw);

        pose_received_ = true;
    }

    // ════════════════════════════════════════════════════════════════
    //  velocity_callback — 속도만 갱신 (use_mcl_pose 모드)
    //  주의: 이 차량의 휠 오도메트리는 횡슬립을 감지하지 못해
    //  twist.linear.y가 항상 0이므로 vy/slip_angle은 항상 0이 된다.
    // ════════════════════════════════════════════════════════════════
    void velocity_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        current_state_.v     = static_cast<float>(msg->twist.twist.linear.x);
        current_state_.vy    = static_cast<float>(msg->twist.twist.linear.y);
        current_state_.omega = static_cast<float>(msg->twist.twist.angular.z);

        current_state_.slip_angle =
            std::atan2(current_state_.vy, std::fabs(current_state_.v) + 1e-5f);
        current_state_.ay = current_state_.v * current_state_.omega;

        velocity_received_ = true;
    }

    // ════════════════════════════════════════════════════════════════
    //  단일 odom_callback (시뮬레이터 모드 전용, 기존과 동일)
    //  /odom0 에서 pose(x,y,yaw) + twist(vx,vy,omega) 동시 처리
    // ════════════════════════════════════════════════════════════════
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // 위치 & 헤딩
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

    int nearest_index_xy(float x, float y) {
        if (ref_path_xs_.empty()) return 0;
        int nearest = 0; float min_d = 1e9f;
        for (int i = 0; i < (int)ref_path_xs_.size(); ++i) {
            float d = (x - ref_path_xs_[i]) * (x - ref_path_xs_[i])
                    + (y - ref_path_ys_[i]) * (y - ref_path_ys_[i]);
            if (d < min_d) { min_d = d; nearest = i; }
        }
        return nearest;
    }

    int update_nearest_index(const mppi::State &s) {
        return nearest_index_xy(s.x, s.y);
    }

    // 상대차 위치를 센터라인 법선에 투영해 좌/우 여유폭(h_pl/h_pr) 계산.
    // opp_yaw/opp_v로 fsm_side_pred_time_ 뒤의 예측 횡위치를 반영한다.
    bool compute_opp_clearances(float opp_x, float opp_y, float opp_yaw, float opp_v,
                                float &h_pl, float &h_pr) {
        if (left_xs_.empty() || right_xs_.empty() ||
            left_xs_.size() != right_xs_.size() || ref_path_xs_.empty()) return false;

        int idx = nearest_index_xy(opp_x, opp_y);
        float dx = opp_x - ref_path_xs_[idx], dy = opp_y - ref_path_ys_[idx];
        float ref_yaw = ref_path_yaws_[idx];
        float nx = -std::sin(ref_yaw), ny = std::cos(ref_yaw);
        float e_y = dx * nx + dy * ny;

        float dx_l = left_xs_[idx]  - ref_path_xs_[idx], dy_l = left_ys_[idx]  - ref_path_ys_[idx];
        float dx_r = right_xs_[idx] - ref_path_xs_[idx], dy_r = right_ys_[idx] - ref_path_ys_[idx];
        float l_dist = std::hypot(dx_l, dy_l);
        float r_dist = std::hypot(dx_r, dy_r);

        float e_y_pred = e_y + opp_v * std::sin(opp_yaw - ref_yaw) * fsm_side_pred_time_;
        e_y_pred = std::max(-r_dist, std::min(l_dist, e_y_pred));

        float clearance_needed = mppi_params_.collision_radius + 0.2f + 0.1f;
        h_pl = (l_dist - e_y_pred) - clearance_needed;
        h_pr = (r_dist + e_y_pred) - clearance_needed;
        return true;
    }

    // ════════════════════════════════════════════════════════════════
    //  f1_obstacles_callback — perception 노드가 발행하는 동적/정적
    //  트랙(장애물+상대차) 목록 수신. header.stamp가 비어있으므로
    //  수신 시각(now())을 별도로 기록해 신선도 판정에 쓴다.
    // ════════════════════════════════════════════════════════════════
    void f1_obstacles_callback(const f1_msgs::msg::F1stateArr::SharedPtr msg) {
        latest_obstacles_ = msg;
        obstacles_recv_time_ = this->now();
    }

    // ════════════════════════════════════════════════════════════════
    //  update_overtake_fsm — 매 제어 주기, solve() 직전에 호출.
    //  1) 최신 장애물(정적+동적 모두)을 반발 비용(obs_x/obs_y)에 반영
    //  2) v>=opp_min_speed_ && ego 전방에 있는 최근접 트랙을 "상대차"로 선택
    //  3) FSM tick → speed_cap/바이패스 경로를 solver_에 적용
    // ════════════════════════════════════════════════════════════════
    void update_overtake_fsm() {
        mppi_params_.num_obstacles = 0;
        bool opp_detected = false;
        float opp_x = 0.f, opp_y = 0.f, opp_yaw = 0.f, opp_v = 0.f;
        float best_dist = 1e9f;

        bool fresh = latest_obstacles_ &&
            (this->now() - obstacles_recv_time_).seconds() < obstacle_timeout_s_;

        if (fresh) {
            const float cy = std::cos(current_state_.yaw), sy = std::sin(current_state_.yaw);
            for (const auto &e : latest_obstacles_->f1_state_arr) {
                float ex = static_cast<float>(e.x), ey = static_cast<float>(e.y);

                if (mppi_params_.num_obstacles < MAX_OBS) {
                    mppi_params_.obs_x[mppi_params_.num_obstacles] = ex;
                    mppi_params_.obs_y[mppi_params_.num_obstacles] = ey;
                    mppi_params_.num_obstacles++;
                }

                float fwd = (ex - current_state_.x) * cy + (ey - current_state_.y) * sy;
                if (fwd > 0.f && e.v >= opp_min_speed_) {
                    float d = std::hypot(ex - current_state_.x, ey - current_state_.y);
                    if (d < best_dist) {
                        best_dist    = d;
                        opp_detected = true;
                        opp_x   = ex;
                        opp_y   = ey;
                        opp_yaw = static_cast<float>(e.yaw);
                        opp_v   = static_cast<float>(e.v);
                    }
                }
            }
        }

        float h_pl = 0.f, h_pr = 0.f;
        if (opp_detected) {
            if (!compute_opp_clearances(opp_x, opp_y, opp_yaw, opp_v, h_pl, h_pr))
                h_pl = h_pr = 0.f;
        }

        mppi::FsmCommand cmd = overtake_fsm_->tick(
            current_state_, opp_detected, opp_x, opp_y, opp_v, h_pl, h_pr,
            ref_path_xs_, ref_path_ys_, ref_path_yaws_);

        mppi_params_.max_speed = std::min(base_max_speed_, cmd.speed_cap);

        if (cmd.has_bypass_path()) {
            solver_->set_reference_path(cmd.bypass_xs, cmd.bypass_ys, cmd.bypass_yaws);
            bypass_active_ = true;
        } else if (bypass_active_) {
            solver_->set_reference_path(ref_path_xs_, ref_path_ys_, ref_path_yaws_);
            bypass_active_ = false;
        }

        std_msgs::msg::String state_msg;
        state_msg.data = mppi::fsm_state_to_str(cmd.state);
        fsm_state_pub_->publish(state_msg);
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

        float d_steer = u.steer - u_prev.steer, d_accel = u.accel - u_prev.accel;
        float ay_abs = fabsf(s.ay);
        // 실제 플래닝 비용함수(mppi_core.cu compute_cost_cuda)와 동일한 형태(제곱 증가, lat_g_threshold 파라미터 공유)
        float lat_g_over = ay_abs - mppi_params_.lat_g_threshold;
        float lat_g = (lat_g_over > 0.f) ? mppi_params_.q_lat_g * lat_g_over * lat_g_over : 0.f;

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
        this->declare_parameter("max_speed",            10.0);   mppi_params_.max_speed     = this->get_parameter("max_speed").as_double();
        base_max_speed_ = mppi_params_.max_speed;  // FSM이 매 틱 덮어써도 넘지 않을 절대 상한
        this->declare_parameter("q_dist",               1.5);    mppi_params_.q_dist        = this->get_parameter("q_dist").as_double();
        this->declare_parameter("q_v",                  2.0);    mppi_params_.q_v           = this->get_parameter("q_v").as_double();
        this->declare_parameter("q_du",                 0.8);    mppi_params_.q_du          = this->get_parameter("q_du").as_double();
        this->declare_parameter("q_steer",              0.3);    mppi_params_.q_steer       = this->get_parameter("q_steer").as_double();
        this->declare_parameter("q_collision",          400.0);  mppi_params_.q_collision   = this->get_parameter("q_collision").as_double();
        this->declare_parameter("q_lat_g",              200.0);  mppi_params_.q_lat_g       = this->get_parameter("q_lat_g").as_double();
        this->declare_parameter("lat_g_threshold",       5.5);   mppi_params_.lat_g_threshold       = this->get_parameter("lat_g_threshold").as_double();
        this->declare_parameter("lat_g_fault_threshold", 9.0);   mppi_params_.lat_g_fault_threshold = this->get_parameter("lat_g_fault_threshold").as_double();
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

        this->declare_parameter("odom_topic",   "/odom0");
        odom_topic_  = this->get_parameter("odom_topic").as_string();
        this->declare_parameter("drive_topic",  "/ackermann_cmd0");
        drive_topic_ = this->get_parameter("drive_topic").as_string();
        this->declare_parameter("path_topic",   "/mppi_target_path");
        path_topic_  = this->get_parameter("path_topic").as_string();

        // ekf_pose 비의존 모드: mcl_pose + odom을 직접 구독해 state를 구성
        this->declare_parameter("use_mcl_pose",   true);
        use_mcl_pose_   = this->get_parameter("use_mcl_pose").as_bool();
        this->declare_parameter("pose_topic",     "/mcl_pose");
        pose_topic_     = this->get_parameter("pose_topic").as_string();
        this->declare_parameter("velocity_topic", "/odom");
        velocity_topic_ = this->get_parameter("velocity_topic").as_string();

        // ── 추월(Overtake) FSM: 상대차 인지 및 상태별 파라미터 ──────────
        this->declare_parameter("f1_obstacles_topic", "/f1/perception/object/obstacles/arr");
        f1_obstacles_topic_ = this->get_parameter("f1_obstacles_topic").as_string();
        this->declare_parameter("obstacle_timeout_s", 0.5);
        obstacle_timeout_s_ = this->get_parameter("obstacle_timeout_s").as_double();
        this->declare_parameter("opp_min_speed", 0.5);
        opp_min_speed_ = this->get_parameter("opp_min_speed").as_double();
        this->declare_parameter("fsm_side_pred_time", 0.6);
        fsm_side_pred_time_ = this->get_parameter("fsm_side_pred_time").as_double();

        this->declare_parameter("fsm_follow_dist",        5.0);
        this->declare_parameter("fsm_prep_dist",          3.5);
        this->declare_parameter("fsm_clear_dist",         7.0);
        this->declare_parameter("fsm_merge_dist",         2.0);
        this->declare_parameter("fsm_prep_timeout_s",     2.5);
        this->declare_parameter("fsm_emergency_dist",     0.5);
        this->declare_parameter("fsm_clear_threshold",    0.8);
        this->declare_parameter("fsm_lateral_offset",     0.5);
        this->declare_parameter("fsm_side_confirm_ticks", 3);
        this->declare_parameter("fsm_side_switch_margin", 0.2);
        this->declare_parameter("fsm_abort_clear_factor", 0.5);
        this->declare_parameter("fsm_solo_speed",         6.0);
        this->declare_parameter("fsm_follow_speed",       4.5);
        this->declare_parameter("fsm_overtake_speed",     6.5);
        this->declare_parameter("fsm_emergency_speed",    0.0);

        fsm_cfg_.follow_dist        = this->get_parameter("fsm_follow_dist").as_double();
        fsm_cfg_.prep_dist          = this->get_parameter("fsm_prep_dist").as_double();
        fsm_cfg_.clear_dist         = this->get_parameter("fsm_clear_dist").as_double();
        fsm_cfg_.merge_dist         = this->get_parameter("fsm_merge_dist").as_double();
        fsm_cfg_.prep_timeout_s     = this->get_parameter("fsm_prep_timeout_s").as_double();
        fsm_cfg_.emergency_dist     = this->get_parameter("fsm_emergency_dist").as_double();
        fsm_cfg_.clear_threshold    = this->get_parameter("fsm_clear_threshold").as_double();
        fsm_cfg_.lateral_offset     = this->get_parameter("fsm_lateral_offset").as_double();
        fsm_cfg_.side_confirm_ticks = this->get_parameter("fsm_side_confirm_ticks").as_int();
        fsm_cfg_.side_switch_margin = this->get_parameter("fsm_side_switch_margin").as_double();
        fsm_cfg_.abort_clear_factor = this->get_parameter("fsm_abort_clear_factor").as_double();
        fsm_cfg_.solo_speed         = this->get_parameter("fsm_solo_speed").as_double();
        fsm_cfg_.follow_speed       = this->get_parameter("fsm_follow_speed").as_double();
        fsm_cfg_.overtake_speed     = this->get_parameter("fsm_overtake_speed").as_double();
        fsm_cfg_.emergency_speed    = this->get_parameter("fsm_emergency_speed").as_double();

        mppi_params_.dt            = 0.035;
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
        if (use_mcl_pose_) {
            if (!pose_received_ || !velocity_received_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Waiting for pose (%s) / velocity (%s)...",
                    pose_topic_.c_str(), velocity_topic_.c_str());
                return;
            }
        } else if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for odom (%s)...", odom_topic_.c_str());
            return;
        }
        auto start = std::chrono::high_resolution_clock::now();
        update_overtake_fsm();
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

    std::vector<float> left_xs_, left_ys_, right_xs_, right_ys_;
    std::vector<float> ref_path_xs_, ref_path_ys_, ref_path_yaws_;

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     left_bnd_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     right_bnd_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr        odom_sub_;      // 시뮬레이터 모드 단일 구독
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mcl_pose_sub_; // use_mcl_pose 모드
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr        velocity_sub_;  // use_mcl_pose 모드
    rclcpp::Subscription<f1_msgs::msg::F1stateArr>::SharedPtr      f1_obstacles_sub_;

    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr       vis_pub_;
    rclcpp::Publisher<smppi_cuda_controller::msg::MppiTrajectory>::SharedPtr traj_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr                     fsm_state_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    std::string odom_topic_, drive_topic_, path_topic_;
    std::string pose_topic_, velocity_topic_;
    bool use_mcl_pose_ = true;
    bool odom_received_ = false;
    bool pose_received_ = false;
    bool velocity_received_ = false;

    // ── 추월(Overtake) FSM ────────────────────────────────────────────
    std::unique_ptr<mppi::OvertakeFsm> overtake_fsm_;
    mppi::OvertakeFsm::Config fsm_cfg_;
    std::string f1_obstacles_topic_;
    double obstacle_timeout_s_ = 0.5;
    double opp_min_speed_ = 0.5;
    double fsm_side_pred_time_ = 0.6;
    f1_msgs::msg::F1stateArr::SharedPtr latest_obstacles_;
    rclcpp::Time obstacles_recv_time_;
    float base_max_speed_ = 10.0f;   // yaml에 설정된 절대 속도 상한 (FSM이 넘지 못함)
    bool  bypass_active_ = false;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPINode>());
    rclcpp::shutdown();
    return 0;
}