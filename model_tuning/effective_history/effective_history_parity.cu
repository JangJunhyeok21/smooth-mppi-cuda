#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc,char**argv){
 if(argc!=2)throw std::runtime_error("usage: effective_history_parity WEIGHTS.bin");
 int count=0;auto status=cudaGetDeviceCount(&count);if(status!=cudaSuccess||count<1){std::cerr<<"CUDA unavailable: "<<cudaGetErrorString(status)<<'\n';return 2;}
 mppi::Params p{};p.dt=.02f;p.control_dt=.02f;p.model_dt=.04f;p.dynamics_model=mppi::EFFECTIVE_HISTORY_STATE_RESIDUAL;
 p.min_speed=.5f;p.max_speed=3.f;p.min_accel=-1.f;p.max_accel=1.f;p.l_f=.163f;p.l_r=.161f;
 p.kinematic_position_speed_scale=.8633491306389823f;p.effective_steer_scale=.51f;p.effective_steer_bias=.01f;
 p.effective_yaw_response_tau=.10f;p.effective_max_yaw_accel=15.f;p.effective_speed_response_gain=.76f;
 p.effective_max_accel=1.f;p.effective_vy_decay_tau=.12f;
 mppi::MPPISolver solver(1,60,p);solver.load_effective_history_state_residual_weights(argv[1]);
 mppi::State s{1.f,-.5f,.3f,2.2f,0.f,-.4f,.2f,-.1f,0.f};
 std::array<float,20> h{};for(int i=0;i<10;++i){h[2*i]=-.09f+.02f*i;h[2*i+1]=1.7f+.05f*i;}
 std::cout<<std::setprecision(9);
 for(int step=1;step<=60;++step){mppi::Control u{.22f*sinf(.17f*step),2.25f+.55f*cosf(.11f*step)};s=solver.debug_effective_history_state_residual_step(s,u,h);
  if(step==1||step==5||step==30||step==60){std::cout<<step;const float q[9]={s.x,s.y,s.yaw,s.v,s.vy,s.omega,s.ax,s.ay,s.slip_angle};for(float v:q)std::cout<<' '<<v;for(float v:h)std::cout<<' '<<v;std::cout<<'\n';}}
}
