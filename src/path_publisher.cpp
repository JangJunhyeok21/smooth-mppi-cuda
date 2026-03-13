#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

class PathPublisher : public rclcpp::Node
{
public:
    PathPublisher() : Node("path_publisher")
    {
        declare_parameter<std::string>("csv_file_path", "");
        declare_parameter<std::string>("frame_id", "map");
        declare_parameter<double>("publish_rate", 10.0);

        get_parameter("csv_file_path", csv_path_);
        get_parameter("frame_id", frame_id_);
        get_parameter("publish_rate", publish_rate_);

        if (csv_path_.empty()) {
            RCLCPP_WARN(get_logger(), "csv_file_path not set, using default");
            csv_path_ = "map1_centerline.csv";
        }

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        path_center_pub_ = create_publisher<nav_msgs::msg::Path>("/center_path", qos);
        path_left_pub_ = create_publisher<nav_msgs::msg::Path>("/left_boundary", qos);
        path_right_pub_ = create_publisher<nav_msgs::msg::Path>("/right_boundary", qos);

        timer_ = create_wall_timer(
            std::chrono::milliseconds((int)(1000.0 / publish_rate_)),
            std::bind(&PathPublisher::publishAll, this));

        loadCSV();
    }

private:
    void loadCSV()
    {
        std::ifstream file(csv_path_);
        if (!file.is_open()) {
            RCLCPP_FATAL(get_logger(), "Cannot open CSV file: %s", csv_path_.c_str());
            rclcpp::shutdown();
            return;
        }

        std::string line;
        std::vector<std::string> headers;
        
        // 헤더 파싱
        if (!std::getline(file, line)) {
            RCLCPP_WARN(get_logger(), "CSV file is empty");
            return;
        }

        headers = splitCSV(line);
        for (auto &h : headers) {
            h = trim(h);
            std::transform(h.begin(), h.end(), h.begin(), ::tolower);
        }

        // 컬럼 인덱스 찾기
        int ix = findColumn(headers, {"x_m", "x", "x_map"});
        int iy = findColumn(headers, {"y_m", "y", "y_map"});
        int ipsi = findColumn(headers, {"psi_rad", "psi", "yaw", "heading_rad"});
        int ileft = findColumn(headers, {"w_tr_left_m", "w_left_m", "left_width_m"});
        int iright = findColumn(headers, {"w_tr_right_m", "w_right_m", "right_width_m"});

        if (ix < 0 || iy < 0) {
            RCLCPP_FATAL(get_logger(), "CSV must have X and Y columns");
            rclcpp::shutdown();
            return;
        }

        std::vector<double> xs, ys, psis, lefts, rights;

        // 데이터 파싱
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            auto cols = splitCSV(line);
            
            double x, y, psi = 0.0, left = 0.0, right = 0.0;
            
            if (!tryParse(cols, ix, x) || !tryParse(cols, iy, y)) continue;
            tryParse(cols, ipsi, psi);
            tryParse(cols, ileft, left);
            tryParse(cols, iright, right);

            xs.push_back(x);
            ys.push_back(y);
            psis.push_back(psi);
            lefts.push_back(left);
            rights.push_back(right);
        }

        if (xs.size() < 2) {
            RCLCPP_FATAL(get_logger(), "Too few valid rows in CSV");
            rclcpp::shutdown();
            return;
        }

        // yaw 계산 (CSV에 없는 경우)
        if(ipsi < 0){
            for (size_t i = 0; i < xs.size(); ++i) {
                if (i == 0 && xs.size() > 1) {
                    double dx = xs[1] - xs[0];
                    double dy = ys[1] - ys[0];
                    psis[0] = std::atan2(dy, dx);
                } else if (i > 0) {
                    double dx = xs[i] - xs[i-1];
                    double dy = ys[i] - ys[i-1];
                    double norm = std::hypot(dx, dy);
                    if (norm > 1e-6) {
                        psis[i] = std::atan2(dy, dx);
                    } else {
                        psis[i] = psis[i-1];
                    }
                }
            }
        }

        // yaw unwrap (±π 점프 제거)
        for (size_t i = 1; i < psis.size(); ++i) {
            double diff = psis[i] - psis[i-1];
            while (diff > M_PI) diff -= 2*M_PI;
            while (diff < -M_PI) diff += 2*M_PI;
            psis[i] = psis[i-1] + diff;
        }

        // Path 생성
        buildPath(xs, ys, psis, lefts, rights, path_center_, path_left_, path_right_);

        RCLCPP_INFO(get_logger(), "Loaded %zu waypoints from %s", xs.size(), csv_path_.c_str());
    }

    void buildPath(
        const std::vector<double> &xs,
        const std::vector<double> &ys,
        const std::vector<double> &psis,
        const std::vector<double> &lefts,
        const std::vector<double> &rights,
        nav_msgs::msg::Path &pc,
        nav_msgs::msg::Path &pl,
        nav_msgs::msg::Path &pr)
    {
        pc.header.frame_id = frame_id_;
        pl.header.frame_id = frame_id_;
        pr.header.frame_id = frame_id_;

        pc.poses.reserve(xs.size());
        pl.poses.reserve(xs.size());
        pr.poses.reserve(xs.size());

        for (size_t i = 0; i < xs.size(); ++i) {
            double psi = psis[i];
            // 좌측 법선: (-sin(psi), cos(psi))
            double nx = -std::sin(psi);
            double ny = std::cos(psi);

            // center
            geometry_msgs::msg::PoseStamped c;
            c.header.frame_id = frame_id_;
            c.pose.position.x = xs[i];
            c.pose.position.y = ys[i];
            c.pose.position.z = 0.0;
            double half_yaw = psi * 0.5;
            c.pose.orientation.z = std::sin(half_yaw);
            c.pose.orientation.w = std::cos(half_yaw);
            pc.poses.push_back(c);

            // left boundary
            geometry_msgs::msg::PoseStamped l;
            l.header.frame_id = frame_id_;
            l.pose.position.x = xs[i] + nx * lefts[i];
            l.pose.position.y = ys[i] + ny * lefts[i];
            l.pose.position.z = 0.0;
            l.pose.orientation = c.pose.orientation;
            pl.poses.push_back(l);

            // right boundary
            geometry_msgs::msg::PoseStamped r;
            r.header.frame_id = frame_id_;
            r.pose.position.x = xs[i] - nx * rights[i];
            r.pose.position.y = ys[i] - ny * rights[i];
            r.pose.position.z = 0.0;
            r.pose.orientation = c.pose.orientation;
            pr.poses.push_back(r);
        }
    }

    void publishAll()
    {
        if (path_center_.poses.empty()) return;

        auto now = this->get_clock()->now();
        path_center_.header.stamp = now;
        path_left_.header.stamp = now;
        path_right_.header.stamp = now;

        path_center_pub_->publish(path_center_);
        path_left_pub_->publish(path_left_);
        path_right_pub_->publish(path_right_);
    }

    // 유틸리티 함수들
    std::string trim(const std::string &s)
    {
        auto start = s.begin();
        while (start != s.end() && std::isspace(*start)) ++start;
        
        auto end = s.end();
        do { --end; } while (std::distance(start, end) > 0 && std::isspace(*end));
        
        return std::string(start, end + 1);
    }

    std::vector<std::string> splitCSV(const std::string &line)
    {
        std::vector<std::string> result;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            result.push_back(cell);
        }
        return result;
    }

    int findColumn(const std::vector<std::string> &headers, 
                   const std::initializer_list<const char*> &names)
    {
        for (const auto *name : names) {
            std::string lname = name;
            std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
            for (size_t i = 0; i < headers.size(); ++i) {
                if (headers[i] == lname) return (int)i;
            }
        }
        return -1;
    }

    bool tryParse(const std::vector<std::string> &cols, int idx, double &out)
    {
        if (idx < 0 || idx >= (int)cols.size() || cols[idx].empty()) return false;
        try {
            out = std::stod(cols[idx]);
            return true;
        } catch (...) {
            return false;
        }
    }

    // 멤버
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_center_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_left_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_right_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string csv_path_, frame_id_;
    double publish_rate_;

    nav_msgs::msg::Path path_center_, path_left_, path_right_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathPublisher>());
    rclcpp::shutdown();
    return 0;
}
