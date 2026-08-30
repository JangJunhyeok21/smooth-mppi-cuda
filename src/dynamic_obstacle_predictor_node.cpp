#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <onnxruntime_cxx_api.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include "smppi_cuda_controller/msg/dynamic_obstacle_trajectory.hpp"
#ifdef SMPPI_HAS_F1_MSGS
#include <f1_msgs/msg/f1state_arr.hpp>
#endif

namespace {
constexpr int kHistory=6,kLookahead=10,kHorizon=60,kLegacyInput=66,kSpeedInput=72;
constexpr int kMixtures=3,kOutputDim=4;
constexpr double kDt=0.04,kPi=3.14159265358979323846;
double wrap(double a){return std::remainder(a,2.0*kPi);}
std::string package_relative_path(const std::string&path){
 if(path.empty()||path.front()=='/')return path;
 return ament_index_cpp::get_package_share_directory("smppi_cuda_controller")+"/"+path;
}
std::vector<std::string> split(const std::string&s){std::vector<std::string>v;std::stringstream ss(s);std::string x;while(std::getline(ss,x,','))v.push_back(x);return v;}
struct Raw{double t,x,y,yaw,v;};
struct ClickedObstacle{long id;std::deque<Raw> raw;std::chrono::steady_clock::time_point expires_at;};
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
 Track track_;std::unordered_map<long,std::deque<Raw>> histories_;
 Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING,"smppi_dynamic_obstacle_predictor"};
 Ort::SessionOptions ort_options_;std::unique_ptr<Ort::Session> ort_session_;
 Ort::MemoryInfo ort_memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault)};
 std::array<float,kSpeedInput> input_buffer_{};
 std::array<float,kMixtures> logits_buffer_{};
 std::array<float,kMixtures*kOutputDim> mu_buffer_{};
 std::array<float,kMixtures*kOutputDim> sigma_buffer_{};
 Ort::Value input_tensor_{nullptr};
 std::array<Ort::Value,3> output_tensors_{Ort::Value{nullptr},Ort::Value{nullptr},Ort::Value{nullptr}};
 rclcpp::Publisher<Msg>::SharedPtr pub_;rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
 rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
 rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr clicked_point_sub_;
#ifdef SMPPI_HAS_F1_MSGS
 rclcpp::Subscription<f1_msgs::msg::F1stateArr>::SharedPtr perception_sub_;
#endif
 rclcpp::TimerBase::SharedPtr timer_;double physical_,static_car_radius_,max_radius_,long_gain_,lat_gain_,dynamic_speed_threshold_,marker_period_,pose_noise_std_,pose_noise_max_,clicked_static_lifetime_s_,constant_velocity_sigma_step_;int max_obstacles_;bool include_speed_feature_,publish_markers_,constant_velocity_;int marker_stride_,max_clicked_static_;std::string mode_;
 std::deque<ClickedObstacle> clicked_static_;
 long clicked_id_counter_{-100001};
 std::mt19937 noise_rng_;std::normal_distribution<double> pose_noise_{0.0,1.0};
 std::chrono::steady_clock::time_point last_marker_publish_{};
 std::uint64_t timing_samples_{0};double prediction_ms_sum_{0.0},publish_ms_sum_{0.0};
 static std::string track_path(rclcpp::Node*n){return package_relative_path(n->declare_parameter<std::string>("track_csv","data/map2/map2_mppi_track_optimal.csv"));}
 static double yaw(const geometry_msgs::msg::Quaternion&q){return std::atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));}
 Raw noisy_raw(double t,double x,double y,double heading,double speed){
  if(pose_noise_std_>0.0&&pose_noise_max_>0.0&&std::abs(speed)>dynamic_speed_threshold_){
   x+=std::clamp(pose_noise_(noise_rng_)*pose_noise_std_,-pose_noise_max_,pose_noise_max_);
   y+=std::clamp(pose_noise_(noise_rng_)*pose_noise_std_,-pose_noise_max_,pose_noise_max_);
  }
  return {t,x,y,heading,speed};
 }
 builtin_interfaces::msg::Duration marker_lifetime()const{const auto ns=static_cast<std::int64_t>(1.0e9*std::max(.25,1.5*marker_period_));builtin_interfaces::msg::Duration value;value.sec=static_cast<std::int32_t>(ns/1000000000LL);value.nanosec=static_cast<std::uint32_t>(ns%1000000000LL);return value;}
 void fill_features(const std::deque<State>&h){int n=0;for(int i=0;i<kHistory;++i){double ds=0,dt=0;if(i){ds=std::remainder(h[i].s-h[i-1].s,track_.length);dt=h[i].t-h[i-1].t;}input_buffer_[n++]=float(ds);input_buffer_[n++]=float(h[i].d);if(include_speed_feature_)input_buffer_[n++]=float(h[i].v);input_buffer_[n++]=float(h[i].k);input_buffer_[n++]=float(h[i].left);input_buffer_[n++]=float(h[i].right);input_buffer_[n++]=float(dt);}for(int j=0;j<kLookahead;++j){int i=track_.at(h.back().s+.5*j);input_buffer_[n++]=float(track_.k[i]);input_buffer_[n++]=float(track_.left[i]);input_buffer_[n++]=float(track_.right[i]);}}
 void infer(const std::deque<State>&h,float&var_s,float&var_d,int&best){fill_features(h);const char*inputs[]={"features"};const char*outputs[]={"logits","mu","sigma"};ort_session_->Run(Ort::RunOptions{nullptr},inputs,&input_tensor_,1,outputs,output_tensors_.data(),3);float max_logit=logits_buffer_[0];best=0;for(int m=1;m<kMixtures;++m){if(logits_buffer_[m]>max_logit){max_logit=logits_buffer_[m];best=m;}}float sum=0,mean_s=0,mean_d=0;std::array<float,kMixtures>prob{};for(int m=0;m<kMixtures;++m){prob[m]=std::exp(logits_buffer_[m]-max_logit);sum+=prob[m];}for(int m=0;m<kMixtures;++m){prob[m]/=sum;mean_s+=prob[m]*mu_buffer_[m*kOutputDim];mean_d+=prob[m]*mu_buffer_[m*kOutputDim+1];}float step_var_s=0,step_var_d=0;for(int m=0;m<kMixtures;++m){const float ds=mu_buffer_[m*kOutputDim]-mean_s,dd=mu_buffer_[m*kOutputDim+1]-mean_d;step_var_s+=prob[m]*(sigma_buffer_[m*kOutputDim]*sigma_buffer_[m*kOutputDim]+ds*ds);step_var_d+=prob[m]*(sigma_buffer_[m*kOutputDim+1]*sigma_buffer_[m*kOutputDim+1]+dd*dd);}var_s+=step_var_s;var_d+=step_var_d;}
 bool resample(const std::deque<Raw>&raw,std::deque<State>&out)const{if(raw.size()<2)return false;double now=raw.back().t,first=now-(kHistory-1)*kDt;if(first<raw.front().t)return false;size_t cursor=0;for(int j=0;j<kHistory;++j){double q=first+j*kDt;while(cursor+1<raw.size()&&raw[cursor+1].t<q)++cursor;if(cursor+1>=raw.size())return false;const auto&a=raw[cursor];const auto&b=raw[cursor+1];double u=(q-a.t)/std::max(1e-9,b.t-a.t);Raw r{q,a.x+u*(b.x-a.x),a.y+u*(b.y-a.y),a.yaw+u*wrap(b.yaw-a.yaw),a.v+u*(b.v-a.v)};out.push_back(track_.project(r));}return true;}
 void predict_one(long id,const std::deque<Raw>&raw,Msg&msg,visualization_msgs::msg::MarkerArray&markers,int obstacle,bool visualize){
  if(raw.empty())return;
  const bool is_dynamic=std::abs(raw.back().v)>dynamic_speed_threshold_;
  float var_s=0,var_d=0;msg.obstacle_ids.push_back(id);msg.is_dynamic.push_back(is_dynamic);
  if(!is_dynamic){
   const auto&current=raw.back();msg.x.push_back(current.x);msg.y.push_back(current.y);msg.yaw.push_back(current.yaw);msg.semi_major.push_back(static_car_radius_);msg.semi_minor.push_back(static_car_radius_);
   if(visualize){visualization_msgs::msg::Marker mk;mk.header=msg.header;mk.ns="mdn_prediction_"+std::to_string(id);mk.id=obstacle*kHorizon;mk.type=mk.CYLINDER;mk.action=mk.ADD;mk.lifetime=marker_lifetime();mk.pose.position.x=current.x;mk.pose.position.y=current.y;mk.pose.position.z=.05;mk.pose.orientation.z=std::sin(current.yaw/2);mk.pose.orientation.w=std::cos(current.yaw/2);mk.scale.x=2*static_car_radius_;mk.scale.y=2*static_car_radius_;mk.scale.z=.05;mk.color.r=.6f;mk.color.g=.6f;mk.color.b=.6f;mk.color.a=.57f;markers.markers.push_back(mk);}return;
  }
  std::deque<State>h;if(!resample(raw,h)){msg.obstacle_ids.pop_back();msg.is_dynamic.pop_back();return;}
  for(int step=0;step<kHorizon;++step){
   auto old=h.back();double sv=old.s,d=old.d,e=old.e;
   if(is_dynamic){
    if(constant_velocity_){
     // Constant speed along the raceline, holding the lateral offset. The MDN is
     // trained per track (config/predictor/*/metadata.json: data_dir is a
     // simulator MPPI run on one specific track) and degrades off it. Measured on
     // four track_20260814 bags, 238 rollout windows, median horizon error:
     //   MDN (ifac2026 weights)  1.0 s 2.49  2.0 s 4.67  2.4 s 5.67 m
     //   constant velocity       1.0 s 1.48  2.0 s 2.38  2.4 s 2.75 m
     // There the MDN is worse than assuming the obstacle does not move at all
     // (3.72 m at 2.4 s), so on a track without its own weights this is the
     // honest mode. On the track the weights were trained for it is the other way
     // round -- ifac2026 MDN 2.67 m vs constant velocity 2.76 m -- so leave it off
     // there. Default false: nothing changes until a config asks for it.
     //
     // The lateral offset is HELD rather than decayed onto the raceline. An
     // opponent is not obliged to take the racing line, and pulling the
     // prediction onto it invents a cut to the inside that does not happen.
     const double speed=std::max(0.,h.back().v);
     sv=std::fmod(old.s+speed*kDt+track_.length,track_.length);d=old.d;e=old.e;
     // No mixture to read a variance from, so grow one as a random walk whose
     // per-step sigma reproduces the measured 2.75 m error at the 2.4 s horizon.
     // maximum_radius caps the ellipse long before the horizon, which is the
     // point: the axes say "somewhere along here", not a calibrated covariance.
     const float step_var=float(constant_velocity_sigma_step_*constant_velocity_sigma_step_);
     var_s+=step_var;var_d+=step_var;
     int next=track_.at(sv);h.pop_front();h.push_back({old.t+kDt,sv,d,e,speed,track_.k[next],track_.left[next],track_.right[next]});
    }else{
     int m=0;infer(h,var_s,var_d,m);const float*z=mu_buffer_.data()+m*kOutputDim;sv=std::fmod(old.s+z[0]+track_.length,track_.length);d=old.d+z[1];e=wrap(old.e+z[2]);int next=track_.at(sv);h.pop_front();h.push_back({old.t+kDt,sv,d,e,std::max(0.,double(z[3])),track_.k[next],track_.left[next],track_.right[next]});
    }
   }
   int ti=track_.at(sv);double angle=track_.psi[ti]+e,px=track_.x[ti]-d*std::sin(track_.psi[ti]),py=track_.y[ti]+d*std::cos(track_.psi[ti]);double a=is_dynamic?std::min(max_radius_,physical_+long_gain_*std::sqrt(std::max(0.f,var_s))):static_car_radius_,b=is_dynamic?std::min(max_radius_,physical_+lat_gain_*std::sqrt(std::max(0.f,var_d))):static_car_radius_;msg.x.push_back(px);msg.y.push_back(py);msg.yaw.push_back(angle);msg.semi_major.push_back(a);msg.semi_minor.push_back(b);
   if(visualize&&step%marker_stride_==0){visualization_msgs::msg::Marker mk;mk.header=msg.header;mk.ns="mdn_prediction_"+std::to_string(id);mk.id=obstacle*kHorizon+step;mk.type=mk.CYLINDER;mk.action=mk.ADD;mk.lifetime=marker_lifetime();mk.pose.position.x=px;mk.pose.position.y=py;mk.pose.position.z=.05;mk.pose.orientation.z=std::sin(angle/2);mk.pose.orientation.w=std::cos(angle/2);mk.scale.x=2*a;mk.scale.y=2*b;mk.scale.z=.05;mk.color.r=is_dynamic?1.f:.6f;mk.color.g=is_dynamic?.15f:.6f;mk.color.b=is_dynamic?.05f:.6f;mk.color.a=.12f+.45f*step/kHorizon;markers.markers.push_back(mk);}
  }
 }
 void tick(){using Clock=std::chrono::steady_clock;const auto begin=Clock::now();bool clicked_expired=false;if(clicked_static_lifetime_s_>0.0){while(!clicked_static_.empty()&&begin>=clicked_static_.front().expires_at){RCLCPP_INFO(get_logger(),"RViz clicked-point static obstacle #%ld expired after %.2f s",clicked_static_.front().id,clicked_static_lifetime_s_);clicked_static_.pop_front();clicked_expired=true;}}const bool visualize=publish_markers_&&(last_marker_publish_.time_since_epoch().count()==0||std::chrono::duration<double>(begin-last_marker_publish_).count()>=marker_period_);Msg msg;msg.header.stamp=now();msg.header.frame_id="map";msg.dt=kDt;msg.horizon=kHorizon;visualization_msgs::msg::MarkerArray markers;if(visualize){markers.markers.reserve(1+5*((kHorizon+marker_stride_-1)/marker_stride_));visualization_msgs::msg::Marker clear;clear.header=msg.header;clear.action=clear.DELETEALL;markers.markers.push_back(clear);}int n=0;double current=now().seconds();for(auto&[id,h]:histories_){if(n>=max_obstacles_||h.empty()||current-h.back().t>.5)continue;predict_one(id,h,msg,markers,n++,visualize);}for(auto&obs:clicked_static_){if(n>=max_obstacles_)break;predict_one(obs.id,obs.raw,msg,markers,n++,visualize);}const auto predicted=Clock::now();if(visualize){marker_pub_->publish(markers);last_marker_publish_=begin;}if(!msg.obstacle_ids.empty()||clicked_expired){pub_->publish(msg);const auto published=Clock::now();const double prediction_ms=std::chrono::duration<double,std::milli>(predicted-begin).count();const double publish_ms=std::chrono::duration<double,std::milli>(published-predicted).count();prediction_ms_sum_+=prediction_ms;publish_ms_sum_+=publish_ms;if(++timing_samples_%100==0){RCLCPP_INFO(get_logger(),"ONNX MDN timing (last 100): prediction/message %.3f ms, ROS publish %.3f ms, total %.3f ms, obstacles=%d",prediction_ms_sum_/100.0,publish_ms_sum_/100.0,(prediction_ms_sum_+publish_ms_sum_)/100.0,n);prediction_ms_sum_=0.0;publish_ms_sum_=0.0;}}}
public:
 Predictor():Node("dynamic_obstacle_predictor"),track_(track_path(this)){
  mode_=declare_parameter<std::string>("input_mode","simulation");include_speed_feature_=declare_parameter<bool>("include_speed_feature",true);const auto use_speed_model_path=declare_parameter<std::string>("use_speed_model_path","config/predictor/dynamic_obstacle_frenet_speed_mdn/frenet_mdn.onnx");const auto no_speed_model_path=declare_parameter<std::string>("no_speed_model_path","config/predictor/dynamic_obstacle_frenet_mdn_pose_noise/frenet_mdn.onnx");const auto selected_model_path=include_speed_feature_?use_speed_model_path:no_speed_model_path;if(selected_model_path.empty())throw std::invalid_argument(include_speed_feature_?"use_speed_model_path is empty":"no_speed_model_path is empty");auto model_path=package_relative_path(selected_model_path);physical_=declare_parameter<double>("opponent_radius",.24);static_car_radius_=declare_parameter<double>("static_car_radius",.24);max_radius_=declare_parameter<double>("maximum_radius",.75);long_gain_=declare_parameter<double>("longitudinal_ellipse_gain",3.1);lat_gain_=declare_parameter<double>("lateral_ellipse_gain",2.1);dynamic_speed_threshold_=declare_parameter<double>("dynamic_speed_threshold",1.0);// Must match the controller's max_obstacles: it rejects a whole message that
  // carries more, and anything this loop drops is never seen at all. Was
  // hardcoded 5 while params.yaml already said 7, so two of the seven silently
  // never left here.
  max_obstacles_=std::max(1,int(declare_parameter<int64_t>("max_obstacles",15)));constant_velocity_=declare_parameter<bool>("constant_velocity_prediction",false);constant_velocity_sigma_step_=declare_parameter<double>("constant_velocity_sigma_step_m",0.355);if(constant_velocity_sigma_step_<0.0)throw std::invalid_argument("constant_velocity_sigma_step_m must be >= 0");clicked_static_lifetime_s_=declare_parameter<double>("clicked_point_static_obstacle_lifetime_s",5.0);max_clicked_static_=std::max<int64_t>(1,declare_parameter<int64_t>("max_clicked_point_static_obstacles",1));pose_noise_std_=declare_parameter<double>("input_pose_noise_std_m",0.0);pose_noise_max_=declare_parameter<double>("input_pose_noise_max_m",0.10);noise_rng_.seed(static_cast<std::mt19937::result_type>(declare_parameter<int64_t>("input_pose_noise_seed",20260824)));publish_markers_=declare_parameter<bool>("publish_markers",true);const double marker_hz=declare_parameter<double>("marker_publish_rate_hz",5.0);marker_stride_=std::max(1,static_cast<int>(declare_parameter<int64_t>("marker_horizon_stride",4)));marker_period_=1.0/std::max(.1,marker_hz);if(!(static_car_radius_>0.0)||!(dynamic_speed_threshold_>=0.0)||clicked_static_lifetime_s_<0.0||pose_noise_std_<0.0||pose_noise_max_<0.0)throw std::invalid_argument("invalid obstacle, clicked-point lifetime, or input-pose-noise parameter");auto output=declare_parameter<std::string>("output_topic","/mppi/dynamic_obstacle_trajectory");auto marker_topic=declare_parameter<std::string>("marker_topic","/mppi/dynamic_obstacle_prediction_markers");
  const int expected=include_speed_feature_?kSpeedInput:kLegacyInput;ort_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);ort_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);ort_options_.SetIntraOpNumThreads(1);ort_options_.SetInterOpNumThreads(1);ort_session_=std::make_unique<Ort::Session>(ort_env_,model_path.c_str(),ort_options_);const std::array<int64_t,2>input_shape{1,expected};const std::array<int64_t,2>logits_shape{1,kMixtures};const std::array<int64_t,3>output_shape{1,kMixtures,kOutputDim};input_tensor_=Ort::Value::CreateTensor<float>(ort_memory_,input_buffer_.data(),expected,input_shape.data(),input_shape.size());output_tensors_[0]=Ort::Value::CreateTensor<float>(ort_memory_,logits_buffer_.data(),logits_buffer_.size(),logits_shape.data(),logits_shape.size());output_tensors_[1]=Ort::Value::CreateTensor<float>(ort_memory_,mu_buffer_.data(),mu_buffer_.size(),output_shape.data(),output_shape.size());output_tensors_[2]=Ort::Value::CreateTensor<float>(ort_memory_,sigma_buffer_.data(),sigma_buffer_.size(),output_shape.data(),output_shape.size());pub_=create_publisher<Msg>(output,10);marker_pub_=create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic,10);
  if(mode_=="simulation"||mode_=="both"){auto topic=declare_parameter<std::string>("simulation_odom_topic","/opp_racecar/odom");odom_sub_=create_subscription<nav_msgs::msg::Odometry>(topic,20,[this](nav_msgs::msg::Odometry::SharedPtr m){double t=rclcpp::Time(m->header.stamp).seconds(),v=std::hypot(m->twist.twist.linear.x,m->twist.twist.linear.y);histories_[1].push_back(noisy_raw(t,m->pose.pose.position.x,m->pose.pose.position.y,yaw(m->pose.pose.orientation),v));while(histories_[1].size()>100)histories_[1].pop_front();});}
  if(declare_parameter<bool>("enable_clicked_point_static_obstacle",true)){
   const auto topic=declare_parameter<std::string>("clicked_point_topic","/clicked_point");
   clicked_point_sub_=create_subscription<geometry_msgs::msg::PointStamped>(topic,10,[this](geometry_msgs::msg::PointStamped::SharedPtr m){
    if(!std::isfinite(m->point.x)||!std::isfinite(m->point.y))return;
    while(static_cast<int>(clicked_static_.size())>=max_clicked_static_)clicked_static_.pop_front();
    ClickedObstacle obs;obs.id=clicked_id_counter_--;obs.raw.push_back({now().seconds(),m->point.x,m->point.y,0.0,0.0});
    if(clicked_static_lifetime_s_>0.0)obs.expires_at=std::chrono::steady_clock::now()+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(clicked_static_lifetime_s_));
    clicked_static_.push_back(std::move(obs));
    RCLCPP_INFO(get_logger(),"RViz clicked point registered as static obstacle #%ld (%zu/%d active): (%.3f, %.3f), radius=%.3f m, lifetime=%.2f s",clicked_static_.back().id,clicked_static_.size(),max_clicked_static_,m->point.x,m->point.y,static_car_radius_,clicked_static_lifetime_s_);
   });
  }
#ifdef SMPPI_HAS_F1_MSGS
  if(mode_=="perception"||mode_=="both"){auto topic=declare_parameter<std::string>("perception_topic","/f1/perception/object/obstacles/arr");perception_sub_=create_subscription<f1_msgs::msg::F1stateArr>(topic,20,[this](f1_msgs::msg::F1stateArr::SharedPtr m){double t=rclcpp::Time(m->header.stamp).seconds();for(auto&o:m->f1_state_arr){if(!std::isfinite(o.x+o.y+o.yaw+o.v))continue;auto&h=histories_[static_cast<long>(o.id)];h.push_back(noisy_raw(t,o.x,o.y,o.yaw,o.v));while(h.size()>100)h.pop_front();}});}
#else
  if(mode_!="simulation")throw std::runtime_error("perception mode requires f1_msgs");
#endif
  double hz=declare_parameter<double>("publish_rate_hz",50);timer_=create_wall_timer(std::chrono::duration<double>(1/hz),[this]{tick();});RCLCPP_INFO(get_logger(),"C++ ONNX Frenet recursive MDN loaded: %s; speed_feature=%s input=%d dt=%.2f horizon=%d, trajectory=%.1f Hz, markers=%.1f Hz stride=%d, dynamic pose noise std=%.3f m max=%.3f m",model_path.c_str(),include_speed_feature_?"true":"false",expected,kDt,kHorizon,hz,publish_markers_?marker_hz:0.0,marker_stride_,pose_noise_std_,pose_noise_max_);
 }
};
int main(int argc,char**argv){rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<Predictor>());}catch(const std::exception&e){fprintf(stderr,"predictor fatal: %s\n",e.what());rclcpp::shutdown();return 1;}rclcpp::shutdown();return 0;}
