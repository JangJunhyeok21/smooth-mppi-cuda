#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ATen/Parallel.h>
#include <torch/script.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include "smppi_cuda_controller/msg/dynamic_obstacle_trajectory.hpp"
#ifdef SMPPI_HAS_F1_MSGS
#include <f1_msgs/msg/f1state_arr.hpp>
#endif

namespace {
constexpr int kHistory=6,kLookahead=10,kHorizon=60,kLegacyInput=66,kSpeedInput=72;
constexpr double kDt=0.04,kPi=3.14159265358979323846;
double wrap(double a){return std::remainder(a,2.0*kPi);}
std::string package_relative_path(const std::string&path){
 if(path.empty()||path.front()=='/')return path;
 return ament_index_cpp::get_package_share_directory("smppi_cuda_controller")+"/"+path;
}
std::vector<std::string> split(const std::string&s){std::vector<std::string>v;std::stringstream ss(s);std::string x;while(std::getline(ss,x,','))v.push_back(x);return v;}
struct Raw{double t,x,y,yaw,v;};
struct State{double t,s,d,e,v,k,left,right;};
struct Track {
 std::vector<double>x,y,psi,k,left,right,s;double length{};
 explicit Track(const std::string&path){std::ifstream f(path);if(!f)throw std::runtime_error("track open failed: "+path);std::string line;std::getline(f,line);auto h=split(line);std::unordered_map<std::string,int>c;for(size_t i=0;i<h.size();++i)c[h[i]]=i;
  for(const char*n:{"x_m","y_m","psi_rad","kappa_radpm","w_tr_left_m","w_tr_right_m"})if(!c.count(n))throw std::runtime_error(std::string("missing track column ")+n);
  while(std::getline(f,line)){auto q=split(line);if(q.size()<h.size())continue;x.push_back(std::stod(q[c["x_m"]]));y.push_back(std::stod(q[c["y_m"]]));psi.push_back(std::stod(q[c["psi_rad"]]));k.push_back(std::stod(q[c["kappa_radpm"]]));left.push_back(std::stod(q[c["w_tr_left_m"]]));right.push_back(std::stod(q[c["w_tr_right_m"]]));}
  s.resize(x.size());for(size_t i=1;i<x.size();++i)s[i]=s[i-1]+std::hypot(x[i]-x[i-1],y[i]-y[i-1]);length=s.back()+std::hypot(x.front()-x.back(),y.front()-y.back());}
 int nearest(double px,double py)const{int best=0;double bd=std::numeric_limits<double>::max();for(size_t i=0;i<x.size();++i){double d=std::hypot(px-x[i],py-y[i]);if(d<bd){bd=d;best=i;}}return best;}
 int at(double sv)const{sv=std::fmod(sv,length);if(sv<0)sv+=length;auto it=std::upper_bound(s.begin(),s.end(),sv);return it==s.begin()?0:static_cast<int>(it-s.begin()-1);}
 State project(const Raw&r)const{int i=nearest(r.x,r.y);double d=(r.x-x[i])*(-std::sin(psi[i]))+(r.y-y[i])*std::cos(psi[i]);return {r.t,s[i],d,wrap(r.yaw-psi[i]),r.v,k[i],left[i],right[i]};}
};
}

class Predictor:public rclcpp::Node{
 using Msg=smppi_cuda_controller::msg::DynamicObstacleTrajectory;
 Track track_;torch::jit::script::Module model_;std::unordered_map<long,std::deque<Raw>> histories_;
 rclcpp::Publisher<Msg>::SharedPtr pub_;rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
 rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
#ifdef SMPPI_HAS_F1_MSGS
 rclcpp::Subscription<f1_msgs::msg::F1stateArr>::SharedPtr perception_sub_;
#endif
 rclcpp::TimerBase::SharedPtr timer_;double physical_,max_radius_,long_gain_,lat_gain_,dynamic_speed_threshold_;bool include_speed_feature_;std::string mode_;
 std::uint64_t timing_samples_{0};double prediction_ms_sum_{0.0},publish_ms_sum_{0.0};
 static std::string track_path(rclcpp::Node*n){return package_relative_path(n->declare_parameter<std::string>("track_csv","data/map2/map2_mppi_track_optimal.csv"));}
 static double yaw(const geometry_msgs::msg::Quaternion&q){return std::atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));}
 std::vector<float> features(const std::deque<State>&h)const{std::vector<float>v;v.reserve(include_speed_feature_?kSpeedInput:kLegacyInput);for(int i=0;i<kHistory;++i){double ds=0,dt=0;if(i){ds=std::remainder(h[i].s-h[i-1].s,track_.length);dt=h[i].t-h[i-1].t;}v.push_back(float(ds));v.push_back(float(h[i].d));if(include_speed_feature_)v.push_back(float(h[i].v));v.insert(v.end(),{float(h[i].k),float(h[i].left),float(h[i].right),float(dt)});}for(int j=0;j<kLookahead;++j){int i=track_.at(h.back().s+.5*j);v.insert(v.end(),{float(track_.k[i]),float(track_.left[i]),float(track_.right[i])});}return v;}
 bool resample(const std::deque<Raw>&raw,std::deque<State>&out)const{if(raw.size()<2)return false;double now=raw.back().t,first=now-(kHistory-1)*kDt;if(first<raw.front().t)return false;size_t cursor=0;for(int j=0;j<kHistory;++j){double q=first+j*kDt;while(cursor+1<raw.size()&&raw[cursor+1].t<q)++cursor;if(cursor+1>=raw.size())return false;const auto&a=raw[cursor];const auto&b=raw[cursor+1];double u=(q-a.t)/std::max(1e-9,b.t-a.t);Raw r{q,a.x+u*(b.x-a.x),a.y+u*(b.y-a.y),a.yaw+u*wrap(b.yaw-a.yaw),a.v+u*(b.v-a.v)};out.push_back(track_.project(r));}return true;}
 void predict_one(long id,const std::deque<Raw>&raw,Msg&msg,visualization_msgs::msg::MarkerArray&markers,int obstacle){
  std::deque<State>h;if(!resample(raw,h))return;
  const bool is_dynamic=std::abs(raw.back().v)>=dynamic_speed_threshold_;
  double var_s=0,var_d=0;msg.obstacle_ids.push_back(id);msg.is_dynamic.push_back(is_dynamic);
  for(int step=0;step<kHorizon;++step){
   auto old=h.back();double sv=old.s,d=old.d,e=old.e;
   if(is_dynamic){
    auto f=features(h);auto input=torch::from_blob(f.data(),{1,static_cast<long>(f.size())},torch::kFloat32).clone();auto tuple=model_.forward({input}).toTuple();auto logits=tuple->elements()[0].toTensor()[0];auto mu=tuple->elements()[1].toTensor()[0];auto sigma=tuple->elements()[2].toTensor()[0];auto p=torch::softmax(logits,0);auto mean=(p.unsqueeze(1)*mu).sum(0);auto variance=(p.unsqueeze(1)*(sigma.square()+(mu-mean).square())).sum(0);var_s+=variance[0].item<double>();var_d+=variance[1].item<double>();int m=logits.argmax().item<int>();auto z=mu[m];sv=std::fmod(old.s+z[0].item<double>()+track_.length,track_.length);d=old.d+z[1].item<double>();e=wrap(old.e+z[2].item<double>());int next=track_.at(sv);h.pop_front();h.push_back({old.t+kDt,sv,d,e,std::max(0.,z[3].item<double>()),track_.k[next],track_.left[next],track_.right[next]});
   }
   int ti=track_.at(sv);double angle=track_.psi[ti]+e,px=track_.x[ti]-d*std::sin(track_.psi[ti]),py=track_.y[ti]+d*std::cos(track_.psi[ti]);double a=is_dynamic?std::min(max_radius_,physical_+long_gain_*std::sqrt(std::max(0.,var_s))):0.0,b=is_dynamic?std::min(max_radius_,physical_+lat_gain_*std::sqrt(std::max(0.,var_d))):0.0;msg.x.push_back(px);msg.y.push_back(py);msg.yaw.push_back(angle);msg.semi_major.push_back(a);msg.semi_minor.push_back(b);
   visualization_msgs::msg::Marker mk;mk.header=msg.header;mk.ns="mdn_prediction_"+std::to_string(id);mk.id=obstacle*kHorizon+step;mk.type=mk.CYLINDER;mk.action=mk.ADD;mk.pose.position.x=px;mk.pose.position.y=py;mk.pose.position.z=.05;mk.pose.orientation.z=std::sin(angle/2);mk.pose.orientation.w=std::cos(angle/2);mk.scale.x=2*(is_dynamic?a:physical_);mk.scale.y=2*(is_dynamic?b:physical_);mk.scale.z=.05;mk.color.r=is_dynamic?1.f:.6f;mk.color.g=is_dynamic?.15f:.6f;mk.color.b=is_dynamic?.05f:.6f;mk.color.a=.12f+.45f*step/kHorizon;markers.markers.push_back(mk);
  }
 }
 void tick(){using Clock=std::chrono::steady_clock;const auto begin=Clock::now();Msg msg;msg.header.stamp=now();msg.header.frame_id="map";msg.dt=kDt;msg.horizon=kHorizon;visualization_msgs::msg::MarkerArray markers;int n=0;double current=now().seconds();for(auto&[id,h]:histories_){if(n>=5||h.empty()||current-h.back().t>.5)continue;predict_one(id,h,msg,markers,n++);}const auto predicted=Clock::now();if(!msg.obstacle_ids.empty()){pub_->publish(msg);marker_pub_->publish(markers);const auto published=Clock::now();const double prediction_ms=std::chrono::duration<double,std::milli>(predicted-begin).count();const double publish_ms=std::chrono::duration<double,std::milli>(published-predicted).count();prediction_ms_sum_+=prediction_ms;publish_ms_sum_+=publish_ms;if(++timing_samples_%100==0){RCLCPP_INFO(get_logger(),"MDN timing (last 100): prediction/message %.3f ms, ROS publish %.3f ms, total %.3f ms, obstacles=%d",prediction_ms_sum_/100.0,publish_ms_sum_/100.0,(prediction_ms_sum_+publish_ms_sum_)/100.0,n);prediction_ms_sum_=0.0;publish_ms_sum_=0.0;}}}
public:
 Predictor():Node("dynamic_obstacle_predictor"),track_(track_path(this)){
  // This network is tiny and is invoked recursively. A single intra-op thread
  // avoids paying a thread-pool synchronization cost at every horizon knot.
  at::set_num_threads(1);at::set_num_interop_threads(1);
  mode_=declare_parameter<std::string>("input_mode","simulation");include_speed_feature_=declare_parameter<bool>("include_speed_feature",true);auto model_path=package_relative_path(declare_parameter<std::string>("model_path","config/predictor/dynamic_obstacle_frenet_speed_mdn/frenet_mdn.ts"));physical_=declare_parameter<double>("opponent_radius",.24);max_radius_=declare_parameter<double>("maximum_radius",.75);long_gain_=declare_parameter<double>("longitudinal_ellipse_gain",3.1);lat_gain_=declare_parameter<double>("lateral_ellipse_gain",2.1);dynamic_speed_threshold_=declare_parameter<double>("dynamic_speed_threshold",1.0);auto output=declare_parameter<std::string>("output_topic","/mppi/dynamic_obstacle_trajectory");auto marker_topic=declare_parameter<std::string>("marker_topic","/mppi/dynamic_obstacle_prediction_markers");model_=torch::jit::load(model_path);model_.eval();auto expected=include_speed_feature_?kSpeedInput:kLegacyInput;for(const auto&buffer:model_.named_buffers())if(buffer.name=="input_mean"&&buffer.value.numel()!=expected)throw std::runtime_error("predictor model/YAML input mismatch: expected "+std::to_string(expected)+" features");pub_=create_publisher<Msg>(output,10);marker_pub_=create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic,10);
  if(mode_=="simulation"||mode_=="both"){auto topic=declare_parameter<std::string>("simulation_odom_topic","/opp_racecar/odom");odom_sub_=create_subscription<nav_msgs::msg::Odometry>(topic,20,[this](nav_msgs::msg::Odometry::SharedPtr m){double t=rclcpp::Time(m->header.stamp).seconds();histories_[1].push_back({t,m->pose.pose.position.x,m->pose.pose.position.y,yaw(m->pose.pose.orientation),std::hypot(m->twist.twist.linear.x,m->twist.twist.linear.y)});while(histories_[1].size()>100)histories_[1].pop_front();});}
#ifdef SMPPI_HAS_F1_MSGS
  if(mode_=="perception"||mode_=="both"){auto topic=declare_parameter<std::string>("perception_topic","/f1/perception/object/obstacles/arr");perception_sub_=create_subscription<f1_msgs::msg::F1stateArr>(topic,20,[this](f1_msgs::msg::F1stateArr::SharedPtr m){double t=rclcpp::Time(m->header.stamp).seconds();for(auto&o:m->f1_state_arr){if(!std::isfinite(o.x+o.y+o.yaw+o.v))continue;auto&h=histories_[static_cast<long>(o.id)];h.push_back({t,o.x,o.y,o.yaw,o.v});while(h.size()>100)h.pop_front();}});}
#else
  if(mode_!="simulation")throw std::runtime_error("perception mode requires f1_msgs");
#endif
  double hz=declare_parameter<double>("publish_rate_hz",50);timer_=create_wall_timer(std::chrono::duration<double>(1/hz),[this]{tick();});RCLCPP_INFO(get_logger(),"C++ Frenet recursive MDN loaded: %s; speed_feature=%s input=%d dt=%.2f horizon=%d, publish=%.1f Hz",model_path.c_str(),include_speed_feature_?"true":"false",include_speed_feature_?kSpeedInput:kLegacyInput,kDt,kHorizon,hz);
 }
};
int main(int argc,char**argv){rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<Predictor>());}catch(const std::exception&e){fprintf(stderr,"predictor fatal: %s\n",e.what());rclcpp::shutdown();return 1;}rclcpp::shutdown();return 0;}
