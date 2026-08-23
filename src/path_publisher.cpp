#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
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
        // Absolute paths are accepted. Relative paths are resolved from this
        // package's installed share directory, so params.yaml remains portable
        // between the development PC and the Orin deployment.
        declare_parameter<std::string>("csv_file_path", "data/map1/map1_centerline.csv");
        declare_parameter<std::string>("frame_id", "map");
        declare_parameter<double>("publish_rate", 10.0);
        declare_parameter<std::string>("boundary_visualization_topic", "/mppi_boundary_viz");
        declare_parameter<double>("boundary_publish_period_s", 5.0);
        declare_parameter<double>("collision_radius", 0.2);
        
        get_parameter("csv_file_path", csv_path_);
        get_parameter("frame_id", frame_id_);
        get_parameter("publish_rate", publish_rate_);
        get_parameter("boundary_visualization_topic", boundary_topic_);
        get_parameter("boundary_publish_period_s", boundary_publish_period_s_);
        get_parameter("collision_radius", collision_radius_);

        if (csv_path_.empty()) {
            throw std::invalid_argument("csv_file_path must not be empty");
        }
        if (csv_path_.front() != '/') {
            csv_path_ = ament_index_cpp::get_package_share_directory(
                "smppi_cuda_controller") + "/" + csv_path_;
        }

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        // 🚨 토픽 이름 분리 (시뮬레이터 충돌 방지)
        path_center_pub_ = create_publisher<nav_msgs::msg::Path>("/mppi_target_path", qos);
        boundary_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            boundary_topic_, qos);

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
        
        if (!std::getline(file, line)) {
            RCLCPP_WARN(get_logger(), "CSV file is empty");
            return;
        }

        headers = splitCSV(line);
        for (auto &h : headers) {
            h = trim(h);
            std::transform(h.begin(), h.end(), h.begin(), ::tolower);
        }

        int ix = findColumn(headers, {"x_m", "x", "x_map"});
        int iy = findColumn(headers, {"y_m", "y", "y_map"});
        int ipsi = findColumn(headers, {"psi_rad", "psi", "yaw", "heading_rad"});
        int ileft = findColumn(headers, {"w_tr_left_m", "w_left_m", "left_width_m"});
        int iright = findColumn(headers, {"w_tr_right_m", "w_right_m", "right_width_m"});
        int ilx = findColumn(headers, {"left_x_m", "left_x"});
        int ily = findColumn(headers, {"left_y_m", "left_y"});
        int irx = findColumn(headers, {"right_x_m", "right_x"});
        int iry = findColumn(headers, {"right_y_m", "right_y"});
        int ibrx = findColumn(headers, {"boundary_ref_x_m", "boundary_ref_x"});
        int ibry = findColumn(headers, {"boundary_ref_y_m", "boundary_ref_y"});

        if (ix < 0 || iy < 0) {
            RCLCPP_FATAL(get_logger(), "CSV must have X and Y columns");
            rclcpp::shutdown();
            return;
        }
        if (ileft < 0 || iright < 0) {
            RCLCPP_FATAL(
                get_logger(),
                "CSV must contain left/right lane widths "
                "(w_tr_left_m and w_tr_right_m) for MPPI boundary constraints");
            rclcpp::shutdown();
            return;
        }

        const bool has_explicit_boundaries = ilx >= 0 && ily >= 0 && irx >= 0 && iry >= 0;
        std::vector<double> xs, ys, psis, lefts, rights, left_xs, left_ys, right_xs, right_ys;

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
            if (has_explicit_boundaries) {
                double lx, ly, rx, ry;
                if (!tryParse(cols, ilx, lx) || !tryParse(cols, ily, ly) ||
                    !tryParse(cols, irx, rx) || !tryParse(cols, iry, ry)) {
                    RCLCPP_FATAL(get_logger(), "Invalid explicit boundary coordinate in %s", csv_path_.c_str());
                    rclcpp::shutdown();
                    return;
                }
                left_xs.push_back(lx); left_ys.push_back(ly);
                right_xs.push_back(rx); right_ys.push_back(ry);
            }
            double brx = x, bry = y;
            if (ibrx >= 0 && ibry >= 0) {
                if (!tryParse(cols, ibrx, brx) || !tryParse(cols, ibry, bry)) {
                    RCLCPP_FATAL(get_logger(), "Invalid boundary reference coordinate in %s", csv_path_.c_str());
                    rclcpp::shutdown();
                    return;
                }
            }
            boundary_ref_xs_.push_back(brx);
            boundary_ref_ys_.push_back(bry);
        }

        if (xs.size() < 2) {
            RCLCPP_FATAL(get_logger(), "Too few valid rows in CSV");
            rclcpp::shutdown();
            return;
        }

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

        for (size_t i = 1; i < psis.size(); ++i) {
            double diff = psis[i] - psis[i-1];
            while (diff > M_PI) diff -= 2*M_PI;
            while (diff < -M_PI) diff += 2*M_PI;
            psis[i] = psis[i-1] + diff;
        }

        buildPath(xs, ys, psis, lefts, rights, left_xs, left_ys,
                  right_xs, right_ys, path_center_, path_left_, path_right_);

        RCLCPP_INFO(
            get_logger(),
            "Loaded %zu waypoints with left/right lane widths from %s",
            xs.size(), csv_path_.c_str());
    }

    void buildPath(
        const std::vector<double> &xs,
        const std::vector<double> &ys,
        const std::vector<double> &psis,
        const std::vector<double> &lefts,
        const std::vector<double> &rights,
        const std::vector<double> &left_xs,
        const std::vector<double> &left_ys,
        const std::vector<double> &right_xs,
        const std::vector<double> &right_ys,
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
            double nx = -std::sin(psi);
            double ny = std::cos(psi);

            geometry_msgs::msg::PoseStamped c;
            c.header.frame_id = frame_id_;
            c.pose.position.x = xs[i];
            c.pose.position.y = ys[i];
            c.pose.position.z = 0.0;
            double half_yaw = psi * 0.5;
            c.pose.orientation.z = std::sin(half_yaw);
            c.pose.orientation.w = std::cos(half_yaw);
            pc.poses.push_back(c);

            geometry_msgs::msg::PoseStamped l;
            l.header.frame_id = frame_id_;
            const bool explicit_boundaries = left_xs.size() == xs.size();
            l.pose.position.x = explicit_boundaries ? left_xs[i] : xs[i] + nx * lefts[i];
            l.pose.position.y = explicit_boundaries ? left_ys[i] : ys[i] + ny * lefts[i];
            l.pose.position.z = 0.0;
            l.pose.orientation = c.pose.orientation;
            pl.poses.push_back(l);

            geometry_msgs::msg::PoseStamped r;
            r.header.frame_id = frame_id_;
            r.pose.position.x = explicit_boundaries ? right_xs[i] : xs[i] - nx * rights[i];
            r.pose.position.y = explicit_boundaries ? right_ys[i] : ys[i] - ny * rights[i];
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
        const auto steady_now = std::chrono::steady_clock::now();
        if (last_boundary_publish_.time_since_epoch().count() == 0 ||
            boundary_publish_period_s_ <= 0.0 ||
            std::chrono::duration<double>(steady_now-last_boundary_publish_).count() >=
                boundary_publish_period_s_) {
            const double elapsed_ms = publishBoundaryMarkers(now);
            last_boundary_publish_ = steady_now;
            ++boundary_publish_count_;
            boundary_publish_total_ms_ += elapsed_ms;
            if (boundary_publish_count_ == 1 || boundary_publish_count_ % 12 == 0) {
                RCLCPP_INFO(get_logger(),
                    "Boundary MarkerArray: latest %.3f ms, average %.3f ms, period %.1f s",
                    elapsed_ms, boundary_publish_total_ms_/boundary_publish_count_,
                    boundary_publish_period_s_);
            }
        }
    }

    double publishBoundaryMarkers(const rclcpp::Time &stamp)
    {
        const auto begin = std::chrono::steady_clock::now();
        visualization_msgs::msg::MarkerArray output;
        visualization_msgs::msg::Marker left, right;
        left.header.frame_id = right.header.frame_id = frame_id_;
        left.header.stamp = right.header.stamp = stamp;
        left.ns = right.ns = "boundary_slack_zero";
        left.id = 200; right.id = 201;
        left.type = right.type = visualization_msgs::msg::Marker::LINE_STRIP;
        left.action = right.action = visualization_msgs::msg::Marker::ADD;
        left.pose.orientation.w = right.pose.orientation.w = 1.0;
        left.scale.x = right.scale.x = 0.045;
        left.color.r = 1.0f; left.color.g = 0.35f; left.color.a = 1.0f;
        right.color.r = 0.85f; right.color.b = 1.0f; right.color.a = 1.0f;

        const std::size_t count = std::min({path_left_.poses.size(),
            path_right_.poses.size(), boundary_ref_xs_.size(), boundary_ref_ys_.size()});
        left.points.reserve(count+1); right.points.reserve(count+1);
        for (std::size_t i=0; i<count; ++i) {
            const double lx=path_left_.poses[i].pose.position.x;
            const double ly=path_left_.poses[i].pose.position.y;
            const double rx=path_right_.poses[i].pose.position.x;
            const double ry=path_right_.poses[i].pose.position.y;
            const double cx=boundary_ref_xs_[i], cy=boundary_ref_ys_[i];
            const double lw=std::hypot(lx-cx,ly-cy), rw=std::hypot(rx-cx,ry-cy);
            geometry_msgs::msg::Point lp,rp;
            const double ls=std::max(0.0,lw-collision_radius_)/std::max(lw,1e-9);
            const double rs=std::max(0.0,rw-collision_radius_)/std::max(rw,1e-9);
            lp.x=cx+(lx-cx)*ls; lp.y=cy+(ly-cy)*ls; lp.z=0.04;
            rp.x=cx+(rx-cx)*rs; rp.y=cy+(ry-cy)*rs; rp.z=0.04;
            left.points.push_back(lp); right.points.push_back(rp);
        }
        if (!left.points.empty()) {
            left.points.push_back(left.points.front());
            right.points.push_back(right.points.front());
        }
        output.markers.push_back(std::move(left));
        output.markers.push_back(std::move(right));
        boundary_pub_->publish(output);
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()-begin).count();
    }

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

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_center_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr boundary_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string csv_path_, frame_id_, boundary_topic_;
    double publish_rate_, boundary_publish_period_s_, collision_radius_;
    std::chrono::steady_clock::time_point last_boundary_publish_{};
    std::size_t boundary_publish_count_{0};
    double boundary_publish_total_ms_{0.0};
    std::vector<double> boundary_ref_xs_, boundary_ref_ys_;

    nav_msgs::msg::Path path_center_, path_left_, path_right_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathPublisher>());
    rclcpp::shutdown();
    return 0;
}
