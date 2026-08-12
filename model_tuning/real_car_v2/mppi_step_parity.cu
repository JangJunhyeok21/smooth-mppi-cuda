#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
    if (argc != 3) throw std::runtime_error("usage: mppi_step_parity WEIGHTS.bin lag|no_lag");
    const bool use_servo_lag=std::string(argv[2])=="lag";
    if(!use_servo_lag && std::string(argv[2])!="no_lag")
        throw std::runtime_error("second argument must be lag or no_lag");
    int device_count=0;
    const cudaError_t device_status=cudaGetDeviceCount(&device_count);
    if(device_status!=cudaSuccess || device_count<1){
        std::cerr<<"CUDA parity unavailable: "<<cudaGetErrorString(device_status)<<'\n';
        return 2;
    }
    mppi::Params p{};
    p.dt=.02f;p.dynamics_model=mppi::DYNAMIC_MLP_RESIDUAL;
    p.min_speed=.5f;p.max_speed=2.f;p.min_accel=-1.f;p.max_accel=1.f;
    p.mass=3.74f;p.l_f=.163f;p.l_r=.161f;
    p.kinematic_steer_scale=.50927964f;p.kinematic_steer_bias=.01015773f;
    p.kinematic_position_speed_scale=.8633491306389823f;
    p.speed_servo_kp=.7616888694734905f;
    p.steer_servo_time_constant=.15514851356820727f;
    p.actuator_max_steer_rate=.8344090950084138f;
    p.speed_reference_accel_time_constant=.04f;
    p.speed_reference_brake_time_constant=.02f;
    p.actuator_max_speed_reference_rate=8.f;
    p.dynamic_mlp_B_f=3.879070152566808f;p.dynamic_mlp_C_f=1.6471076687680233f;
    p.dynamic_mlp_D_f=.0710062229162444f;p.dynamic_mlp_E_f=-1.f;
    p.dynamic_mlp_B_r=2.321287285513187f;p.dynamic_mlp_C_r=1.9234527357451916f;
    p.dynamic_mlp_D_r=.05906540313616536f;p.dynamic_mlp_E_r=-1.f;
    p.dynamic_mlp_I_z=.04712f;p.dynamic_mlp_min_speed=.8f;
    const float wheelbase=p.l_f+p.l_r;
    p.F_zf=p.mass*9.81f*p.l_r/wheelbase;p.F_zr=p.mass*9.81f*p.l_f/wheelbase;
    mppi::MPPISolver solver(1,2,p);solver.load_dynamic_mlp_residual_weights(argv[1]);
    mppi::State s{1.f,-.5f,.3f,2.2f,.15f,-.4f,.2f,-.1f,std::atan2(.15f,2.2f)};
    mppi::Control u{.25f,3.5f}; // exact CUDA step clamps this to 2 m/s
    std::array<float,12> h{{-.10f,2.0f,-.05f,2.2f,.0f,2.5f,.08f,2.8f,.12f,3.0f,.07f,2.0f}};
    const auto n=solver.debug_dynamic_mlp_residual_step(s,u,h,use_servo_lag);
    std::cout<<std::setprecision(9)<<n.x<<' '<<n.y<<' '<<n.yaw<<' '<<n.v<<' '
             <<n.vy<<' '<<n.omega<<' '<<n.ax<<' '<<n.ay<<' '<<n.slip_angle;
    for(float value:h)std::cout<<' '<<value;std::cout<<'\n';
}
