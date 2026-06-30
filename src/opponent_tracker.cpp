#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include <cmath>
#include <vector>
#include <limits>

using namespace std::chrono_literals;

// ── 단순 유클리드 클러스터링 ──────────────────────────────────────────
struct Cluster {
    float cx, cy;   // 중심
    float vx, vy;   // 추정 속도 (이전 중심과의 차분)
    int   count;    // 포인트 수
};

class OpponentTracker : public rclcpp::Node {
public:
    OpponentTracker() : Node("opponent_tracker") {
        this->declare_parameter("cluster_dist_thresh",   0.15);
        this->declare_parameter("min_cluster_points",    3);
        this->declare_parameter("max_cluster_points",    200);
        this->declare_parameter("max_detection_range",   8.0);
        this->declare_parameter("min_detection_range",   0.3);
        this->declare_parameter("scan_topic",  "/scan");
        this->declare_parameter("odom_topic",  "/opponent_odom");
        this->declare_parameter("frame_id",    "laser");

        cluster_dist_thresh_  = get_parameter("cluster_dist_thresh").as_double();
        min_cluster_points_   = get_parameter("min_cluster_points").as_int();
        max_cluster_points_   = get_parameter("max_cluster_points").as_int();
        max_range_            = get_parameter("max_detection_range").as_double();
        min_range_            = get_parameter("min_detection_range").as_double();
        std::string scan_topic = get_parameter("scan_topic").as_string();
        std::string odom_topic = get_parameter("odom_topic").as_string();
        frame_id_             = get_parameter("frame_id").as_string();

        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            scan_topic, 10,
            std::bind(&OpponentTracker::scan_callback, this, std::placeholders::_1));

        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10);
        viz_pub_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("/opponent_viz", 10);

        RCLCPP_INFO(this->get_logger(), "OpponentTracker started — scan: %s", scan_topic.c_str());
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        // 1. 극좌표 → 직교 좌표 변환 (유효 포인트만)
        std::vector<std::pair<float,float>> pts;
        float angle = msg->angle_min;
        for (float r : msg->ranges) {
            if (r > min_range_ && r < max_range_ && std::isfinite(r)) {
                pts.emplace_back(r * std::cos(angle), r * std::sin(angle));
            }
            angle += msg->angle_increment;
        }

        if (pts.empty()) return;

        // 2. 유클리드 클러스터링
        std::vector<int> labels(pts.size(), -1);
        int cluster_id = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (labels[i] >= 0) continue;
            labels[i] = cluster_id;
            for (size_t j = i + 1; j < pts.size(); ++j) {
                if (labels[j] >= 0) continue;
                float dx = pts[i].first  - pts[j].first;
                float dy = pts[i].second - pts[j].second;
                if (dx*dx + dy*dy < cluster_dist_thresh_ * cluster_dist_thresh_)
                    labels[j] = cluster_id;
            }
            ++cluster_id;
        }

        // 3. 클러스터 중심 계산
        std::vector<Cluster> clusters(cluster_id, {0,0,0,0,0});
        for (size_t i = 0; i < pts.size(); ++i) {
            int id = labels[i];
            if (id < 0) continue;
            clusters[id].cx += pts[i].first;
            clusters[id].cy += pts[i].second;
            clusters[id].count++;
        }
        for (auto &c : clusters) {
            if (c.count > 0) { c.cx /= c.count; c.cy /= c.count; }
        }

        // 4. 유효 클러스터 필터링 (포인트 수 기준)
        std::vector<Cluster> valid;
        for (auto &c : clusters) {
            if (c.count >= min_cluster_points_ && c.count <= max_cluster_points_)
                valid.push_back(c);
        }
        if (valid.empty()) return;

        // 5. 가장 가까운 클러스터 선택
        Cluster best = valid[0];
        float min_r2 = best.cx*best.cx + best.cy*best.cy;
        for (auto &c : valid) {
            float r2 = c.cx*c.cx + c.cy*c.cy;
            if (r2 < min_r2) { min_r2 = r2; best = c; }
        }

        // 6. 속도 추정 (이전 위치와의 차분)
        rclcpp::Time t_now = this->now();
        if (prev_detected_) {
            double dt = (t_now - prev_time_).seconds();
            if (dt > 1e-3 && dt < 0.5) {
                best.vx = (best.cx - prev_cx_) / dt;
                best.vy = (best.cy - prev_cy_) / dt;
            }
        }
        prev_cx_ = best.cx; prev_cy_ = best.cy;
        prev_time_ = t_now;
        prev_detected_ = true;

        // 7. Odometry 발행
        nav_msgs::msg::Odometry odom;
        odom.header.stamp    = t_now;
        odom.header.frame_id = frame_id_;
        odom.child_frame_id  = "opponent";
        odom.pose.pose.position.x = best.cx;
        odom.pose.pose.position.y = best.cy;
        odom.pose.pose.position.z = 0.0;
        odom.pose.pose.orientation.w = 1.0;
        odom.twist.twist.linear.x = best.vx;
        odom.twist.twist.linear.y = best.vy;
        odom_pub_->publish(odom);

        // 8. RViz 마커 발행
        publish_marker(best, t_now);
    }

    void publish_marker(const Cluster &c, const rclcpp::Time &t) {
        visualization_msgs::msg::MarkerArray ma;
        visualization_msgs::msg::Marker m;
        m.header.stamp    = t;
        m.header.frame_id = frame_id_;
        m.ns     = "opponent";
        m.id     = 0;
        m.type   = visualization_msgs::msg::Marker::CYLINDER;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = c.cx;
        m.pose.position.y = c.cy;
        m.pose.position.z = 0.15;
        m.pose.orientation.w = 1.0;
        m.scale.x = 0.3; m.scale.y = 0.3; m.scale.z = 0.3;
        m.color.r = 1.0f; m.color.g = 0.3f; m.color.b = 0.0f; m.color.a = 0.8f;
        ma.markers.push_back(m);
        viz_pub_->publish(ma);
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        odom_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub_;

    float cluster_dist_thresh_ {0.15f};
    int   min_cluster_points_  {3};
    int   max_cluster_points_  {200};
    float max_range_           {8.0f};
    float min_range_           {0.3f};
    std::string frame_id_      {"laser"};

    bool  prev_detected_ {false};
    float prev_cx_ {0.f}, prev_cy_ {0.f};
    rclcpp::Time prev_time_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OpponentTracker>());
    rclcpp::shutdown();
    return 0;
}
