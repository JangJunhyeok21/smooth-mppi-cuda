#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <algorithm>
#include <cmath>

using namespace std::chrono_literals;

class MPPINode : public rclcpp::Node {
public:
    MPPINode() : Node("mppi_controller") {
        load_parameters();
        validate_parameters();

        solver_ = std::make_unique<mppi::MPPISolver>(3000, 100, mppi_params_);

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic_, 10);
        vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi_viz", 10);

        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            path_topic_, 1, std::bind(&MPPINode::path_callback, this, std::placeholders::_1));
        
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 10, std::bind(&MPPINode::odom_callback, this, std::placeholders::_1));
        
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            scan_topic_, 10, std::bind(&MPPINode::scan_callback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(20ms, std::bind(&MPPINode::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "MPPI Node Started: Optimization & Monitor Enabled");
    }

private:
    void load_parameters() {
        mppi_params_.dt = 0.02;

        this->declare_parameter("max_steer", 4.0);
        mppi_params_.max_steer = this->get_parameter("max_steer").as_double();
        
        this->declare_parameter("min_accel", -40.0);
        mppi_params_.min_accel = this->get_parameter("min_accel").as_double();
        
        this->declare_parameter("max_accel", 40.0);
        mppi_params_.max_accel = this->get_parameter("max_accel").as_double();
        
        this->declare_parameter("min_speed", 0.0);
        mppi_params_.min_speed = this->get_parameter("min_speed").as_double();
        
        this->declare_parameter("target_speed", 10.0);
        mppi_params_.target_speed = this->get_parameter("target_speed").as_double();
        
        this->declare_parameter("max_speed", 20.0);
        mppi_params_.max_speed = this->get_parameter("max_speed").as_double();
        
        this->declare_parameter("q_dist", 10.0);
        mppi_params_.q_dist = this->get_parameter("q_dist").as_double();
        this->declare_parameter("q_v", 2.0);
        mppi_params_.q_v = this->get_parameter("q_v").as_double();
        this->declare_parameter("q_u", 3.0);
        mppi_params_.q_u = this->get_parameter("q_u").as_double();
        this->declare_parameter("q_du", 5.0);
        mppi_params_.q_du = this->get_parameter("q_du").as_double();
        this->declare_parameter("q_heading", 3.0);
        mppi_params_.q_heading = this->get_parameter("q_heading").as_double();
        this->declare_parameter("q_lat", 3.0);
        mppi_params_.q_lat = this->get_parameter("q_lat").as_double();
        this->declare_parameter("q_collision", 3.0);
        mppi_params_.q_collision = this->get_parameter("q_collision").as_double();
        this->declare_parameter("collision_radius", 0.4);
        mppi_params_.collision_radius = this->get_parameter("collision_radius").as_double();

        this->declare_parameter("noise_steer_std", 0.5);
        mppi_params_.noise_steer_std = this->get_parameter("noise_steer_std").as_double();
        this->declare_parameter("noise_accel_std", 5.0); 
        mppi_params_.noise_accel_std = this->get_parameter("noise_accel_std").as_double();
        
        this->declare_parameter("max_steer_rate", 4.0); 
        mppi_params_.max_steer_rate = this->get_parameter("max_steer_rate").as_double();
        this->declare_parameter("max_accel_rate", 1000.0); 
        mppi_params_.max_accel_rate = this->get_parameter("max_accel_rate").as_double();

        this->declare_parameter("lambda", 15.0);
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
        this->declare_parameter("D_f", 30.0); mppi_params_.D_f = this->get_parameter("D_f").as_double();
        
        this->declare_parameter("B_r", 1.5); mppi_params_.B_r = this->get_parameter("B_r").as_double();
        this->declare_parameter("C_r", 1.5); mppi_params_.C_r = this->get_parameter("C_r").as_double();
        this->declare_parameter("D_r", 30.0); mppi_params_.D_r = this->get_parameter("D_r").as_double();

        this->declare_parameter("odom_topic", "/state0"); 
        odom_topic_ = this->get_parameter("odom_topic").as_string();
        this->declare_parameter("scan_topic", "/scan0");
        scan_topic_ = this->get_parameter("scan_topic").as_string();
        this->declare_parameter("drive_topic", "/ackermann_cmd0");
        drive_topic_ = this->get_parameter("drive_topic").as_string();
        this->declare_parameter("path_topic", "/center_path");
        path_topic_ = this->get_parameter("path_topic").as_string();        
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
        solver_->set_reference_path(xs, ys, yaws, vs);
        RCLCPP_INFO_ONCE(this->get_logger(), "Path Received: %zu points", xs.size());
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
        current_state_.omega = msg->twist.twist.angular.z;
        
        odom_received_ = true;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        solver_->set_scan_data(msg->ranges, msg->angle_min, msg->angle_increment);
        scan_received_ = true;
    }

    // [추가] 실시간 비용 모니터링 함수
    void monitor_costs() {
        const auto& traj = solver_->get_best_trajectory();
        const auto& controls = solver_->get_optimal_controls();
        
        if (traj.empty() || controls.empty()) return;

        float total_vel_cost = 0.0f;
        float total_lat_cost = 0.0f;
        float total_input_cost = 0.0f;
        
        for (size_t t = 0; t < traj.size(); ++t) {
            const auto& s = traj[t];
            const auto& u = controls[t];
            
            // 1. 횡G 비용 (Lat Cost)
            float lat_g = s.v * s.omega;
            float g_limit = 9.8f * 0.85f;
            if (std::abs(lat_g) > g_limit) {
                float violation = std::abs(lat_g) - g_limit;
                total_lat_cost += mppi_params_.q_lat * (violation * violation);
            }

            // 2. 입력 비용 (Input Cost)
            total_input_cost += mppi_params_.q_u * (u.steer * u.steer + u.accel * u.accel);

            // 3. 속도 비용 (Velocity Cost)
            float v_error = (s.v - mppi_params_.target_speed);
            total_vel_cost += mppi_params_.q_v * (v_error * v_error);
        }

        // 터미널 출력 (디버깅용)
        static int print_count = 0;
        if (print_count++ % 10 == 0) {
            printf("=== Real-time Cost Monitor ===\n");
            printf("Vel Cost : %10.2f\n", total_vel_cost);
            printf("Lat Cost : %10.2f\n", total_lat_cost);
            printf("InputCost: %10.2f\n", total_input_cost);
            printf("------------------------------\n");
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
        drive_msg.drive.acceleration = u.accel;
        
        drive_pub_->publish(drive_msg);
        publish_path_visualization();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        static int count = 0;
        if (count++ % 20 == 0) {
          RCLCPP_INFO(this->get_logger(), "MPPI: %.2fms | V: %.2f", elapsed.count(), current_state_.v);
        }
    }

    void publish_path_visualization() {
        if (!mppi_params_.visualize_candidates) {
            // ...
        }
        
        visualization_msgs::msg::MarkerArray markers;
        
        if (mppi_params_.visualize_candidates) {
            const auto& states = solver_->get_generated_trajectories();
            int K = solver_->get_K();
            int T = solver_->get_T();

            visualization_msgs::msg::Marker traj_marker;
            traj_marker.header.frame_id = "map";
            traj_marker.header.stamp = this->now();
            traj_marker.ns = "candidates";
            traj_marker.id = 0;
            traj_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            traj_marker.action = visualization_msgs::msg::Marker::ADD;
            traj_marker.scale.x = 0.02; 
            traj_marker.color.r = 0.0; traj_marker.color.g = 1.0; traj_marker.color.b = 0.0; traj_marker.color.a = 0.2;

            for (int k = 0; k < K; k += 100) { 
                for (int t = 0; t < T - 1; ++t) {
                    int idx = k * T + t;
                    geometry_msgs::msg::Point p1, p2;
                    p1.x = states[idx].x; p1.y = states[idx].y;
                    p2.x = states[idx+1].x; p2.y = states[idx+1].y;
                    traj_marker.points.push_back(p1);
                    traj_marker.points.push_back(p2);
                }
            }
            markers.markers.push_back(traj_marker);
        }

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

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::string odom_topic_, scan_topic_, drive_topic_, path_topic_;
    bool odom_received_ = false;
    bool scan_received_ = false;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPINode>());
    rclcpp::shutdown();
    return 0;
}