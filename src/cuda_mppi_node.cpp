#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <algorithm>
#include <cmath>

using namespace std::chrono_literals;

class MPPINode : public rclcpp::Node {
public:
    MPPINode() : Node("mppi_controller") {
        load_parameters();
        validate_parameters();

        solver_ = std::make_unique<mppi::MPPISolver>(10000, 150, mppi_params_);

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic_, 10);
        vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi_viz", 10);

        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            path_topic_, 1, std::bind(&MPPINode::path_callback, this, std::placeholders::_1));
        
        // [추가] 시뮬레이터와 동일한 QoS(Transient Local) 설정
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
        
        // 경로 데이터 저장 (monitor_costs에서 사용)
        ref_path_xs_ = xs;
        ref_path_ys_ = ys;
        ref_path_yaws_ = yaws;
        
        solver_->set_reference_path(xs, ys, yaws, vs);
        RCLCPP_INFO_ONCE(this->get_logger(), "Path Received: %zu points", xs.size());
    }

    // [추가] 왼쪽 바운더리 콜백
    void left_bnd_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        left_xs_.clear(); left_ys_.clear();
        for (const auto& p : msg->poses) {
            left_xs_.push_back(p.pose.position.x);
            left_ys_.push_back(p.pose.position.y);
        }
        update_boundaries();
    }

    // [추가] 오른쪽 바운더리 콜백
    void right_bnd_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        right_xs_.clear(); right_ys_.clear();
        for (const auto& p : msg->poses) {
            right_xs_.push_back(p.pose.position.x);
            right_ys_.push_back(p.pose.position.y);
        }
        update_boundaries();
    }

    // [추가] 좌우 바운더리가 모두 수신되면 GPU로 전송
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
        current_state_.ay = 0.0f;
        current_state_.slip_angle = atan2(current_state_.vy, fabs(current_state_.v) + 1e-5f);
        current_state_.omega = msg->twist.twist.angular.z;
        
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

    void monitor_costs() {
        const auto& traj = solver_->get_best_trajectory();
        const auto& controls = solver_->get_optimal_controls();
        
        if (traj.empty() || controls.empty()) return;

        float total_dist_cost = 0.0f;
        float total_vel_cost = 0.0f;
        float total_heading_cost = 0.0f;
        float total_input_cost = 0.0f;
        float total_rate_cost = 0.0f;
        float total_lat_cost = 0.0f;
        float total_collision_cost = 0.0f;
        
        for (size_t t = 0; t < traj.size(); ++t) {
            const auto& s = traj[t];
            const auto& u = controls[t];
            const auto& u_prev = (t == 0) ? controls[0] : controls[t-1];
            
            // 1. Distance Cost (경로와의 거리)
            if (!ref_path_xs_.empty()) {
                float min_dist_sq = 1e9f;
                for (size_t i = 0; i < ref_path_xs_.size(); ++i) {
                    float dx = s.x - ref_path_xs_[i];
                    float dy = s.y - ref_path_ys_[i];
                    float dist_sq = dx * dx + dy * dy;
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                    }
                }
                total_dist_cost += mppi_params_.q_dist * min_dist_sq;
            }
            
            // 2. Velocity Cost
            // float v_error = (s.v - mppi_params_.target_speed);
            // total_vel_cost += mppi_params_.q_v * (v_error * v_error);
            total_vel_cost = -mppi_params_.q_v * s.v;
            
            // 5. Rate Cost (제어 변화율)
            float d_steer = u.steer - u_prev.steer;
            float d_accel = u.accel - u_prev.accel;
            total_rate_cost += mppi_params_.q_du * (d_steer * d_steer + d_accel * d_accel);
            
            // 6. Lateral Acceleration Cost
            float lat_accel = std::abs(s.ay);
            float g_limit = 9.8f;
            if (lat_accel > g_limit) {
                total_lat_cost = 1.0e9f;
            }
            
            // 7. Collision Cost
            if (!left_xs_.empty() && !right_xs_.empty()) {
                float min_bnd_dist = 1e9f;
                // 왼쪽 바운더리와의 거리
                for (size_t i = 0; i < left_xs_.size(); ++i) {
                    float dx = s.x - left_xs_[i];
                    float dy = s.y - left_ys_[i];
                    float dist = std::sqrt(dx * dx + dy * dy);
                    if (dist < min_bnd_dist) min_bnd_dist = dist;
                }
                // 오른쪽 바운더리와의 거리
                for (size_t i = 0; i < right_xs_.size(); ++i) {
                    float dx = s.x - right_xs_[i];
                    float dy = s.y - right_ys_[i];
                    float dist = std::sqrt(dx * dx + dy * dy);
                    if (dist < min_bnd_dist) min_bnd_dist = dist;
                }
                
                if (min_bnd_dist < mppi_params_.collision_radius) {
                    float diff = mppi_params_.collision_radius - min_bnd_dist;
                    total_collision_cost += mppi_params_.q_collision * std::exp(diff * 15.0f);
                }
            }
        }

        static int print_count = 0;
        if (print_count++ % 10 == 0) {
            printf("========== Real-time Cost Monitor ==========\n");
            printf("Dist Cost     : %10.2f (q=%.1f)\n", total_dist_cost, mppi_params_.q_dist);
            printf("Vel Cost      : %10.2f (q=%.1f)\n", total_vel_cost, mppi_params_.q_v);
            printf("Rate Cost     : %10.2f (q=%.1f)\n", total_rate_cost, mppi_params_.q_du);
            printf("Collision Cost: %10.2f (q=%.1f)\n", total_collision_cost, mppi_params_.q_collision);
            printf("TOTAL         : %10.2f\n", total_dist_cost + total_vel_cost + total_heading_cost + 
                   total_input_cost + total_rate_cost + total_lat_cost + total_collision_cost);
            printf("============================================\n");
        }
    }

    void timer_callback() {
        if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for Odom...");
            return;
        }
        
        auto start = std::chrono::high_resolution_clock::now();

        // MPPI 실행
        mppi::Control u = solver_->solve(current_state_);
        
        // 3. 비용 모니터링 (디버깅)
        // monitor_costs();
        
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
        publish_path_visualization();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        static int count = 0;
        if (count++ % 10 == 0) {
          RCLCPP_INFO(this->get_logger(), "MPPI: %.2fms | V: %.2f", elapsed.count(), current_state_.v);
        }
    }

    void publish_path_visualization() {
        if (!mppi_params_.visualize_candidates) {
            return;
        }
        
        visualization_msgs::msg::MarkerArray markers;
        
        const auto& states = solver_->get_generated_trajectories();
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

        for (int k = 0; k < K; k += 100) { 
            for (int t = 1; t < T - 2; ++t) {
                int idx = k * T + t;
                geometry_msgs::msg::Point p1, p2;
                p1.x = states[idx].x; p1.y = states[idx].y;
                p2.x = states[idx+1].x; p2.y = states[idx+1].y;
                traj_marker.points.push_back(p1);
                traj_marker.points.push_back(p2);
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

    // [추가] 바운더리 저장을 위한 벡터
    std::vector<float> left_xs_, left_ys_, right_xs_, right_ys_;
    
    // [추가] 경로 데이터 저장 (monitor_costs용)
    std::vector<float> ref_path_xs_, ref_path_ys_, ref_path_yaws_;

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    // [추가] 바운더리 구독 포인터
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr left_bnd_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr right_bnd_sub_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr vel_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub_;
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