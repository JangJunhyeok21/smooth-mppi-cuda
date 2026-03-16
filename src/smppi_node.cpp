#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "cuda_mppi_controller/cuda_mppi_core.hpp"
// [추가] 커스텀 메시지 헤더 포함
#include "smppi_cuda_controller/msg/mppi_trajectory.hpp"
#include <algorithm>
#include <cmath>

using namespace std::chrono_literals;

class MPPINode : public rclcpp::Node {
public:
    MPPINode() : Node("smppi_controller") {
        load_parameters();
        validate_parameters();

        solver_ = std::make_unique<mppi::MPPISolver>(10000, 150, mppi_params_);

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic_, 10);
        vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi_viz", 50);
        
        // [추가] MPPI 최적 궤적 퍼블리셔 초기화
        traj_pub_ = this->create_publisher<smppi_cuda_controller::msg::MppiTrajectory>("/mppi_optimal_trajectory", 10);

        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            path_topic_, 1, std::bind(&MPPINode::path_callback, this, std::placeholders::_1));
        
        // 시뮬레이터와 동일한 QoS(Transient Local) 설정
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

        left_bnd_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/left_boundary", qos, std::bind(&MPPINode::left_bnd_callback, this, std::placeholders::_1));
            
        right_bnd_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/right_boundary", qos, std::bind(&MPPINode::right_bnd_callback, this, std::placeholders::_1));

        if (use_mcl_pose_) {
            pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
                pose_topic_, 10, std::bind(&MPPINode::pose_callback, this, std::placeholders::_1));

            vel_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                velocity_topic_, 10, std::bind(&MPPINode::velocity_callback, this, std::placeholders::_1));
        } else {
            odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                odom_topic_, 10, std::bind(&MPPINode::odom_callback, this, std::placeholders::_1));
        }
        
        timer_ = this->create_wall_timer(use_mcl_pose_?40ms:20ms, std::bind(&MPPINode::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "MPPI Node Started: Optimization & Boundary Monitor Enabled");
    }

private:
    float compute_min_boundary_distance(const mppi::State &s, int current_path_idx) {
        if (left_xs_.empty() || right_xs_.empty() || left_xs_.size() != right_xs_.size()) {
            return 1e9f;
        }

        float min_dist_sq = 1e9f;
        int bnd_len = static_cast<int>(left_xs_.size());
        int search_window = 30;
        int start_search = current_path_idx - 5;
        if (start_search < 0) start_search += bnd_len;

        for (int offset = 0; offset < search_window; ++offset) {
            int i = start_search + offset;
            if (i >= bnd_len) i -= bnd_len;

            float dx_l = s.x - left_xs_[i];
            float dy_l = s.y - left_ys_[i];
            float dist_sq_l = dx_l * dx_l + dy_l * dy_l;

            float dx_r = s.x - right_xs_[i];
            float dy_r = s.y - right_ys_[i];
            float dist_sq_r = dx_r * dx_r + dy_r * dy_r;

            if (dist_sq_l < min_dist_sq) min_dist_sq = dist_sq_l;
            if (dist_sq_r < min_dist_sq) min_dist_sq = dist_sq_r;
        }

        return std::sqrt(min_dist_sq);
    }

    int update_nearest_index(const mppi::State &s) {
        if (ref_path_xs_.empty()) return 0;

        int path_len = static_cast<int>(ref_path_xs_.size());
        int nearest_idx = 0;
        float min_dist_sq = 1e9f;

        // 제어기(mppi_core.cu)의 CPU 단 전역 탐색과 완전히 동일한 알고리즘
        for (int i = 0; i < path_len; ++i) {
            float dx = s.x - ref_path_xs_[i];
            float dy = s.y - ref_path_ys_[i];
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }

        return nearest_idx;
    }

    void append_best_traj_costs(
        const std::vector<mppi::State> &best_traj,
        const std::vector<mppi::Control> &optimal_controls,
        smppi_cuda_controller::msg::MppiTrajectory &msg) {
        
        if (best_traj.empty() || optimal_controls.empty() || ref_path_xs_.empty() || ref_path_yaws_.empty()) {
            return;
        }

        int t_idx = (best_traj.size() > 1 && optimal_controls.size() > 1) ? 1 : 0;
        const auto &s = best_traj[t_idx];
        const auto &u = optimal_controls[t_idx];
        const auto &u_prev = (t_idx == 0) ? optimal_controls[0] : optimal_controls[t_idx - 1];

        // 차량의 상태(s)를 기준으로 전체 맵에서 가장 가까운 인덱스 추출
        int local_path_idx = update_nearest_index(s);

        float dx = s.x - ref_path_xs_[local_path_idx];
        float dy = s.y - ref_path_ys_[local_path_idx];
        float dist_error = dx * dx + dy * dy;

        // GPU에서 사용하는 것과 완벽히 동일한 인덱스의 yaw 값 참조
        float vel_cost = -mppi_params_.q_v * (s.v * std::cos(s.yaw - ref_path_yaws_[local_path_idx]));

        float d_steer = u.steer - u_prev.steer;
        float d_accel = u.accel - u_prev.accel;
        float steer_rate_cost = mppi_params_.q_du * 2.0f * (d_steer * d_steer);
        float accel_rate_cost = mppi_params_.q_du * std::fabs(d_accel);
        float steer_cost = mppi_params_.q_steer * (u.steer * u.steer);

        float lat_g_cost = 0.0f;
        float ay_abs = fabsf(s.ay);
        if (ay_abs >= 9.5f) {
            float excess = ay_abs - 9.5f;
            lat_g_cost = mppi_params_.q_lat_g * (expf(-3.0f * excess));
        }

        float min_bnd_dist = compute_min_boundary_distance(s, local_path_idx);
        // 기존 boundary_cost 계산 코드 삭제 후 아래로 교체
        float boundary_cost = 0.0f;
        float safe_dist = mppi_params_.collision_radius + 0.4f;

        if (min_bnd_dist < safe_dist) {
            float penetration = safe_dist - min_bnd_dist;
            float soft_cost = 150.0f * (penetration * penetration);

            float hard_cost = 0.0f;
            if (min_bnd_dist < mppi_params_.collision_radius * 1.2f) {
                float diff = min_bnd_dist - mppi_params_.collision_radius;
                float capped = std::min(diff, 1.0e-5f);
                hard_cost = mppi_params_.q_collision * std::log(1.0f + std::exp(-40.0f * capped));
            }
            boundary_cost = soft_cost + hard_cost;
        }

        msg.dist_cost= mppi_params_.q_dist * dist_error;
        msg.vel_cost = vel_cost;
        msg.steer_rate_cost = steer_rate_cost;
        msg.accel_rate_cost = accel_rate_cost;
        msg.steer_cost = steer_cost;
        msg.slip_cost = lat_g_cost;
        msg.boundary_cost = boundary_cost;    
        msg.yaw = s.yaw;
        msg.ref_yaw = ref_path_yaws_[local_path_idx];  
    }

    void load_parameters() {
        this->declare_parameter("max_steer", 0.507);
        mppi_params_.max_steer = this->get_parameter("max_steer").as_double();
        
        this->declare_parameter("min_accel", -9.0);
        mppi_params_.min_accel = this->get_parameter("min_accel").as_double();
        
        this->declare_parameter("max_accel", 9.0);
        mppi_params_.max_accel = this->get_parameter("max_accel").as_double();
        
        this->declare_parameter("min_speed", 0.0);
        mppi_params_.min_speed = this->get_parameter("min_speed").as_double();
        
        this->declare_parameter("target_speed", 6.0);
        mppi_params_.target_speed = this->get_parameter("target_speed").as_double();
        
        this->declare_parameter("max_speed", 10.0);
        mppi_params_.max_speed = this->get_parameter("max_speed").as_double();
        
        this->declare_parameter("q_dist", 1.5);
        mppi_params_.q_dist = this->get_parameter("q_dist").as_double();
        this->declare_parameter("q_v", 2.0);
        mppi_params_.q_v = this->get_parameter("q_v").as_double();
        this->declare_parameter("q_du", 0.8);
        mppi_params_.q_du = this->get_parameter("q_du").as_double();
        this->declare_parameter("q_steer", 0.3);
        mppi_params_.q_steer = this->get_parameter("q_steer").as_double();
        this->declare_parameter("q_collision", 400.0);
        mppi_params_.q_collision = this->get_parameter("q_collision").as_double();
        this->declare_parameter("q_lat_g", 200.0);
        mppi_params_.q_lat_g = this->get_parameter("q_lat_g").as_double();
        this->declare_parameter("collision_radius", 0.28);
        mppi_params_.collision_radius = this->get_parameter("collision_radius").as_double();
        
        this->declare_parameter("noise_steer_std", 0.4);
        mppi_params_.noise_steer_std = this->get_parameter("noise_steer_std").as_double();
        this->declare_parameter("noise_accel_std", 2.0); 
        mppi_params_.noise_accel_std = this->get_parameter("noise_accel_std").as_double();
        
        this->declare_parameter("max_steer_rate", 4.0); 
        mppi_params_.max_steer_rate = this->get_parameter("max_steer_rate").as_double();
        this->declare_parameter("max_accel_rate", 1000.0); 
        mppi_params_.max_accel_rate = this->get_parameter("max_accel_rate").as_double();

        this->declare_parameter("lambda", 10.0);
        mppi_params_.lambda = this->get_parameter("lambda").as_double();
        this->declare_parameter("visualize_candidates", true);
        mppi_params_.visualize_candidates = this->get_parameter("visualize_candidates").as_bool();

        this->declare_parameter("mass", 3.5);
        mppi_params_.mass = this->get_parameter("mass").as_double();
        this->declare_parameter("l_f", 0.17);
        mppi_params_.l_f = this->get_parameter("l_f").as_double();
        this->declare_parameter("l_r", 0.17);
        mppi_params_.l_r = this->get_parameter("l_r").as_double();
        this->declare_parameter("I_z", 0.07);
        mppi_params_.I_z = this->get_parameter("I_z").as_double();
        
        this->declare_parameter("B_f", 1.5); mppi_params_.B_f = this->get_parameter("B_f").as_double();
        this->declare_parameter("C_f", 1.5); mppi_params_.C_f = this->get_parameter("C_f").as_double();
        this->declare_parameter("D_f", 20.0); mppi_params_.D_f = this->get_parameter("D_f").as_double();
        
        this->declare_parameter("B_r", 1.5); mppi_params_.B_r = this->get_parameter("B_r").as_double();
        this->declare_parameter("C_r", 1.5); mppi_params_.C_r = this->get_parameter("C_r").as_double();
        this->declare_parameter("D_r", 20.0); mppi_params_.D_r = this->get_parameter("D_r").as_double();

        this->declare_parameter("odom_topic", "/odom0"); 
        odom_topic_ = this->get_parameter("odom_topic").as_string();
        this->declare_parameter("use_mcl_pose", false);
        use_mcl_pose_ = this->get_parameter("use_mcl_pose").as_bool();
        this->declare_parameter("pose_topic", "/mcl_pose");
        pose_topic_ = this->get_parameter("pose_topic").as_string();
        this->declare_parameter("velocity_topic", "/odom");
        velocity_topic_ = this->get_parameter("velocity_topic").as_string();
        this->declare_parameter("drive_topic", "/ackermann_cmd0");
        drive_topic_ = this->get_parameter("drive_topic").as_string();
        this->declare_parameter("path_topic", "/center_path");
        path_topic_ = this->get_parameter("path_topic").as_string();      
        if(use_mcl_pose_){  
            mppi_params_.dt = 0.04;
        } else {
            mppi_params_.dt = 0.02;
        }
    }

    void validate_parameters() {
        if (mppi_params_.min_speed > mppi_params_.max_speed) std::swap(mppi_params_.min_speed, mppi_params_.max_speed);
        if (mppi_params_.lambda <= 0.0f) mppi_params_.lambda = 1.0f;
        if (mppi_params_.collision_radius < 0.0f) mppi_params_.collision_radius = std::abs(mppi_params_.collision_radius);
    }

    void path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (msg->poses.empty()) return;
        std::vector<float> xs, ys, yaws, vs;
        xs.reserve(msg->poses.size()); ys.reserve(msg->poses.size());
        yaws.reserve(msg->poses.size()); vs.reserve(msg->poses.size());

        for (const auto& pose_stamped : msg->poses) {
            xs.push_back(pose_stamped.pose.position.x);
            ys.push_back(pose_stamped.pose.position.y);
            double qx = pose_stamped.pose.orientation.x;
            double qy = pose_stamped.pose.orientation.y;
            double qz = pose_stamped.pose.orientation.z;
            double qw = pose_stamped.pose.orientation.w;
            double yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
            yaws.push_back((float)yaw);
            vs.push_back(mppi_params_.target_speed);
        }
        
        ref_path_xs_ = xs;
        ref_path_ys_ = ys;
        ref_path_yaws_ = yaws;
        
        solver_->set_reference_path(xs, ys, yaws, vs);
        RCLCPP_INFO_ONCE(this->get_logger(), "Path Received: %zu points", xs.size());
    }

    void left_bnd_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        left_xs_.clear(); left_ys_.clear();
        for (const auto& p : msg->poses) {
            left_xs_.push_back(p.pose.position.x);
            left_ys_.push_back(p.pose.position.y);
        }
        update_boundaries();
    }

    void right_bnd_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        right_xs_.clear(); right_ys_.clear();
        for (const auto& p : msg->poses) {
            right_xs_.push_back(p.pose.position.x);
            right_ys_.push_back(p.pose.position.y);
        }
        update_boundaries();
    }

    void update_boundaries() {
        if (!left_xs_.empty() && !right_xs_.empty() && left_xs_.size() == right_xs_.size()) {
            solver_->set_boundaries(left_xs_, left_ys_, right_xs_, right_ys_);
        }
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;
        double yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));

        current_state_.x = msg->pose.pose.position.x;
        current_state_.y = msg->pose.pose.position.y;
        current_state_.yaw = (float)yaw;
        current_state_.v = msg->twist.twist.linear.x;
        current_state_.vy = msg->twist.twist.linear.y;
        current_state_.slip_angle = atan2(current_state_.vy, fabs(current_state_.v) + 1e-5f);
        current_state_.omega = msg->twist.twist.angular.z;
        current_state_.ay = current_state_.v * current_state_.omega; 
        
        odom_received_ = true;
    }

    void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        pose_ = *msg;
        has_pose_ = true;
        update_state_from_pose_velocity();
    }

    void velocity_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        velocity_odom_ = *msg;
        has_velocity_ = true;
        update_state_from_pose_velocity();
    }

    void update_state_from_pose_velocity() {
        if (!has_pose_ || !has_velocity_) return;

        const auto &p = pose_.pose.position;
        const auto &q = pose_.pose.orientation;
        double yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));

        current_state_.x = p.x;
        current_state_.y = p.y;
        current_state_.yaw = (float)yaw;
        current_state_.v = velocity_odom_.twist.twist.linear.x;
        current_state_.vy = velocity_odom_.twist.twist.linear.y;
        current_state_.ay = 0.0f;
        current_state_.slip_angle = atan2(current_state_.vy, fabs(current_state_.v) + 1e-5f);
        current_state_.omega = velocity_odom_.twist.twist.angular.z;

        odom_received_ = true;
    }

    void timer_callback() {
        if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for Odom...");
            return;
        }
        
        auto start = std::chrono::high_resolution_clock::now();

        // MPPI 실행
        mppi::Control u = solver_->solve(current_state_);
        
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->now();
        drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle = u.steer;
        
        // 최소 속도 제한: 후진 방지
        float next_v = current_state_.v + u.accel * mppi_params_.dt;
        if (next_v <= mppi_params_.min_speed) {
            u.accel = (mppi_params_.min_speed - current_state_.v) / mppi_params_.dt;
            next_v = mppi_params_.min_speed;
        }
        if(use_mcl_pose_){
            drive_msg.drive.speed = next_v;
        } else {
            drive_msg.drive.acceleration = u.accel;
        }
        drive_pub_->publish(drive_msg);
        
        // [추가] 시각화 및 최적 궤적 데이터 퍼블리시
        publish_path_visualization();
        publish_mppi_trajectory();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        static int count = 0;
        if (count++ % 10 == 0) {
          RCLCPP_INFO(this->get_logger(), "MPPI: %.2fms | V: %.2f", elapsed.count(), current_state_.v);
        }
    }

    // [추가] 최적 호라이즌 데이터를 Custom Message로 퍼블리시하는 함수
    void publish_mppi_trajectory() {
        const auto& best_traj = solver_->get_best_trajectory();
        const auto& optimal_controls = solver_->get_optimal_controls();

        if (!best_traj.empty() && !optimal_controls.empty()) {
            smppi_cuda_controller::msg::MppiTrajectory msg;
            msg.header.stamp = this->now();
            msg.header.frame_id = "map";

            int T = solver_->get_T();
            
            // 데이터 삽입 속도 최적화를 위한 메모리 사전 할당
            msg.steer.reserve(T); 
            msg.accel.reserve(T);

            for (int t = 0; t < T; ++t) {
                msg.steer.push_back(optimal_controls[t].steer);
                msg.accel.push_back(optimal_controls[t].accel);
            }

            append_best_traj_costs(best_traj, optimal_controls, msg);

            traj_pub_->publish(msg);
        }
    }

    void publish_path_visualization() {
        if (!mppi_params_.visualize_candidates) {
            return;
        }
        
        visualization_msgs::msg::MarkerArray markers;
        
        const auto& states = solver_->get_generated_trajectories();
        const auto& costs = solver_->get_costs();
        int K = solver_->get_K();
        int T = solver_->get_T();

        // 정상 궤적 (녹색)
        visualization_msgs::msg::Marker traj_marker;
        traj_marker.header.frame_id = "map";
        traj_marker.header.stamp = this->now();
        traj_marker.ns = "candidates";
        traj_marker.id = 0;
        traj_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        traj_marker.action = visualization_msgs::msg::Marker::ADD;
        traj_marker.scale.x = 0.02; 
        traj_marker.color.r = 0.0; traj_marker.color.g = 1.0; traj_marker.color.b = 0.0; traj_marker.color.a = 0.3;

        if ((int)costs.size() == K) {
            std::vector<int> indices(K);
            for (int k = 0; k < K; ++k) indices[k] = k;
            std::sort(indices.begin(), indices.end(), [&costs](int a, int b) {
                return costs[a] < costs[b];
            });

            int top_n = std::min(50, K);
            for (int i = 0; i < top_n; ++i) {
                int k = indices[i];
                for (int t = 1; t < T - 2; ++t) {
                    int idx = k * T + t;
                    geometry_msgs::msg::Point p1, p2;
                    p1.x = states[idx].x; p1.y = states[idx].y;
                    p2.x = states[idx+1].x; p2.y = states[idx+1].y;
                    traj_marker.points.push_back(p1);
                    traj_marker.points.push_back(p2);
                }
            }
        }
        markers.markers.push_back(traj_marker);
        

        visualization_msgs::msg::Marker best_traj_marker;
        best_traj_marker.header.frame_id = "map";
        best_traj_marker.header.stamp = this->now();
        best_traj_marker.ns = "best_trajectory";
        best_traj_marker.id = 1;
        best_traj_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        best_traj_marker.action = visualization_msgs::msg::Marker::ADD;
        best_traj_marker.scale.x = 0.05;
        best_traj_marker.color.r = 1.0; best_traj_marker.color.g = 0.0; best_traj_marker.color.b = 0.0; best_traj_marker.color.a = 0.8;

        const auto& best_trajectory = solver_->get_best_trajectory();
        if (!best_trajectory.empty()) {
            for (int t = 0; t < (int)best_trajectory.size() - 1; ++t) {
                geometry_msgs::msg::Point p1, p2;
                p1.x = best_trajectory[t].x; p1.y = best_trajectory[t].y;
                p2.x = best_trajectory[t+1].x; p2.y = best_trajectory[t+1].y;
                best_traj_marker.points.push_back(p1);
                best_traj_marker.points.push_back(p2);
            }

            markers.markers.push_back(best_traj_marker);
        }
        vis_pub_->publish(markers);
    }

    mppi::Params mppi_params_;
    std::unique_ptr<mppi::MPPISolver> solver_;
    mppi::State current_state_;

    std::vector<float> left_xs_, left_ys_, right_xs_, right_ys_;
    std::vector<float> ref_path_xs_, ref_path_ys_, ref_path_yaws_;

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr left_bnd_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr right_bnd_sub_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr vel_sub_;
    
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub_;
    // [추가] 커스텀 메시지 퍼블리셔 포인터
    rclcpp::Publisher<smppi_cuda_controller::msg::MppiTrajectory>::SharedPtr traj_pub_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    
    bool use_mcl_pose_{false};
    bool has_pose_{false};
    bool has_velocity_{false};
    geometry_msgs::msg::PoseStamped pose_;
    nav_msgs::msg::Odometry velocity_odom_;
    std::string odom_topic_, pose_topic_, velocity_topic_, drive_topic_, path_topic_;
    bool odom_received_ = false;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPINode>());
    rclcpp::shutdown();
    return 0;
}