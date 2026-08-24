#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Fixed-input fixture for the production 40 ms servo-lag CUDA step. Runtime
// parameters are supplied by the Python checker from params.yaml/regression.
int main(int argc,char **argv){
    if(argc!=32)throw std::runtime_error(
        "usage: mppi_step_parity WEIGHTS STEPS Bf Cf Df Ef Br Cr Dr Er Iz "
        "min_speed max_speed min_accel max_accel mass lf lr steer_scale steer_bias "
        "steer_tau max_steer max_steer_rate speed_kp accel_tau brake_tau max_ref_rate "
        "position_scale residual_ax residual_ay residual_yaw_accel");
    int devices=0;auto status=cudaGetDeviceCount(&devices);
    if(status!=cudaSuccess||devices<1){std::cerr<<"CUDA parity unavailable: "<<cudaGetErrorString(status)<<'\n';return 2;}
    const int steps=std::atoi(argv[2]);mppi::Params p{};p.dt=.02f;p.control_dt=.02f;p.model_dt=.04f;
    std::ifstream weight_file(argv[1],std::ios::binary|std::ios::ate);
    const bool vx_delta_24d=weight_file && weight_file.tellg()==15308;
    p.dynamics_model=vx_delta_24d
        ?mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D
        :mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG;
    p.min_speed=std::atof(argv[12]);p.max_speed=std::atof(argv[13]);
    p.min_accel=std::atof(argv[14]);p.max_accel=std::atof(argv[15]);
    p.mass=std::atof(argv[16]);p.l_f=std::atof(argv[17]);p.l_r=std::atof(argv[18]);
    p.kinematic_steer_scale=std::atof(argv[19]);p.kinematic_steer_bias=std::atof(argv[20]);
    p.steer_servo_time_constant=std::atof(argv[21]);p.max_steer=std::atof(argv[22]);
    p.actuator_max_steer_rate=std::atof(argv[23]);p.speed_servo_kp=std::atof(argv[24]);
    p.speed_reference_accel_time_constant=std::atof(argv[25]);
    p.speed_reference_brake_time_constant=std::atof(argv[26]);
    p.actuator_max_speed_reference_rate=std::atof(argv[27]);
    p.kinematic_position_speed_scale=std::atof(argv[28]);
    p.dynamic_mlp_B_f=std::atof(argv[3]);p.dynamic_mlp_C_f=std::atof(argv[4]);
    p.dynamic_mlp_D_f=std::atof(argv[5]);p.dynamic_mlp_E_f=std::atof(argv[6]);
    p.dynamic_mlp_B_r=std::atof(argv[7]);p.dynamic_mlp_C_r=std::atof(argv[8]);
    p.dynamic_mlp_D_r=std::atof(argv[9]);p.dynamic_mlp_E_r=std::atof(argv[10]);p.dynamic_mlp_I_z=std::atof(argv[11]);
    p.mlp_max_residual_ax=std::atof(argv[29]);p.mlp_max_residual_ay=std::atof(argv[30]);
    p.mlp_max_residual_yaw_accel=std::atof(argv[31]);
    const float wb=p.l_f+p.l_r;p.F_zf=p.mass*9.81f*p.l_r/wb;p.F_zr=p.mass*9.81f*p.l_f/wb;
    mppi::MPPISolver solver(1,std::max(2,steps),p);
    if(vx_delta_24d)solver.load_dynamic_mlp_vx_delta_residual_weights(argv[1]);
    else solver.load_dynamic_mlp_residual_weights(argv[1]);
    mppi::State s{1.f,-.5f,.3f,2.2f,.15f,-.4f,.2f,-.1f,std::atan2(.15f,2.2f)};
    std::array<float,12> h{{-.10f,2.f,-.05f,2.2f,0.f,2.5f,.08f,2.8f,.12f,3.f,.07f,2.f}};
    std::array<float,5> vxh{{1.95f,2.0f,2.08f,2.14f,2.2f}};
    std::cout<<std::setprecision(9);
    for(int k=0;k<steps;++k){
        const mppi::Control u{.25f-.012f*k,3.5f-.025f*k};
        s=vx_delta_24d?solver.debug_dynamic_mlp_vx_delta_residual_step(s,u,h,vxh)
                      :solver.debug_dynamic_mlp_residual_step(s,u,h,true);
        std::cout<<s.x<<' '<<s.y<<' '<<s.yaw<<' '<<s.v<<' '<<s.vy<<' '<<s.omega<<' '<<s.ax<<' '<<s.ay<<' '<<s.slip_angle;
        for(float v:h)std::cout<<' '<<v;std::cout<<'\n';
    }
}
