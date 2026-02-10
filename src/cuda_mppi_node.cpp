#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <algorithm>

using namespace std::chrono_literals;

class MPPINode : public rclcpp::Node {
public:
    MPPINode() : Node("mppi_controller") {
        load_parameters();

        // K=3000, T=100 (약 2초 예측 - 반응성 향상)
        solver_ = std::make_unique<mppi::MPPISolver>(6000, 150, mppi_params_);

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic_, 10);
        vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mppi_viz", 10);

        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            path_topic_, 1, std::bind(&MPPINode::path_callback, this, std::placeholders::_1));
        
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 10, std::bind(&MPPINode::odom_callback, this, std::placeholders::_1));
        
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            scan_topic_, 10, std::bind(&MPPINode::scan_callback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(20ms, std::bind(&MPPINode::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "MPPI Node Started: SAFETY OFF (Fast Mode)");
    }

private:
    void load_parameters() {
        this->declare_parameter("max_steer", 0.52);
        this->declare_parameter("min_accel", -3.0);
        this->declare_parameter("max_accel", 3.0);
        this->declare_parameter("min_speed", 0.0);
        this->declare_parameter("target_speed", 2.5); // 적당한 속도로 시작
        this->declare_parameter("max_speed", 5.0);
        
        // --- [중요] 고착 해결을 위한 가중치 설정 ---
        this->declare_parameter("q_dist", 10.0);  // 경로 따라가기 (Main)
        this->declare_parameter("q_v", 4.0);      // 속도 내기 (Main) - 값을 높여서 앞으로 가게 유도
        
        // ** 방해 요소 모두 끄기 **
        this->declare_parameter("q_lat", 0.0);       // 코너 감속 OFF
        this->declare_parameter("q_collision", 0.0); // 충돌 방지 OFF
        
        // 부드러운 주행
        this->declare_parameter("q_u", 0.1);
        this->declare_parameter("q_du", 0.1);
        this->declare_parameter("collision_radius", 0.2);

        this->declare_parameter("odom_topic", "/odom0"); 
        this->declare_parameter("scan_topic", "/scan0");
        this->declare_parameter("drive_topic", "/ackermann_cmd0");
        this->declare_parameter("path_topic", "/center_path");

        odom_topic_ = this->get_parameter("odom_topic").as_string();
        scan_topic_ = this->get_parameter("scan_topic").as_string();
        drive_topic_ = this->get_parameter("drive_topic").as_string();
        path_topic_ = this->get_parameter("path_topic").as_string();

        mppi_params_.dt = 0.02;
        mppi_params_.wheel_base = 0.33;
        mppi_params_.max_steer = this->get_parameter("max_steer").as_double();
        mppi_params_.min_accel = this->get_parameter("min_accel").as_double();
        mppi_params_.max_accel = this->get_parameter("max_accel").as_double();
        mppi_params_.min_speed = this->get_parameter("min_speed").as_double();
        mppi_params_.target_speed = this->get_parameter("target_speed").as_double();
        mppi_params_.max_speed = this->get_parameter("max_speed").as_double();
        
        mppi_params_.q_dist = this->get_parameter("q_dist").as_double();
        mppi_params_.q_v = this->get_parameter("q_v").as_double();
        mppi_params_.q_lat = this->get_parameter("q_lat").as_double();
        mppi_params_.q_u = this->get_parameter("q_u").as_double();
        mppi_params_.q_du = this->get_parameter("q_du").as_double();
        mppi_params_.q_collision = this->get_parameter("q_collision").as_double();
        mppi_params_.collision_radius = this->get_parameter("collision_radius").as_double();
    }

    void path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (msg->poses.empty()) return;
        std::vector<float> xs, ys, yaws, vs;
        xs.reserve(msg->poses.size()); ys.reserve(msg->poses.size());
        yaws.reserve(msg->poses.size()); vs.reserve(msg->poses.size());

        for (const auto& pose_stamped : msg->poses) {
            xs.push_back(pose_stamped.pose.position.x);
            ys.push_back(pose_stamped.pose.position.y);
            // Yaw는 혹시 모르니 받아두지만 계산에는 안 씀
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
        odom_received_ = true;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        solver_->set_scan_data(msg->ranges, msg->angle_min, msg->angle_increment);

        scan_received_ = true;
    }

    void timer_callback() {
        if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for Odom...");
            return;
        }

        if (!scan_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for LaserScan...");
        }

        auto start = std::chrono::high_resolution_clock::now();
        mppi::Control u = solver_->solve(current_state_);
        
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->now();
        drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle = u.steer;
        drive_msg.drive.acceleration = u.accel;
        
        float v_cmd = current_state_.v + u.accel * mppi_params_.dt;

        if (v_cmd > mppi_params_.max_speed) {
            v_cmd = mppi_params_.max_speed;
        }
        
        // 최소 속도 제한 (기존 코드)
        drive_msg.drive.speed = std::max(mppi_params_.min_speed, v_cmd);
        
        drive_pub_->publish(drive_msg);
        publish_visualization();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        static int count = 0;
        if (count++ % 20 == 0) {
          RCLCPP_INFO(this->get_logger(), "MPPI: %.2fms | V: %.2f | Steer: %.2f", 
          elapsed.count(), current_state_.v, u.steer);
        }
    }

    void publish_visualization() {
        visualization_msgs::msg::MarkerArray markers;
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
        traj_marker.color.r = 0.0; traj_marker.color.g = 1.0; traj_marker.color.b = 0.0; traj_marker.color.a = 0.15;

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