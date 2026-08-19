#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const int samples = argc > 1 ? std::stoi(argv[1]) : 3200;
    const int horizon = argc > 2 ? std::stoi(argv[2]) : 80;
    const bool use_mlp = argc <= 4 || std::string(argv[4]) != "classic";
    const bool use_safe_set = argc > 5 && std::string(argv[5]) == "safe";
    mppi::Params p{};
    p.dt=.02f; p.control_dt=.02f; p.model_dt=.04f;
    p.dynamics_model=use_mlp ? mppi::DYNAMIC_MLP_RESIDUAL_SERVO_LAG
                            : mppi::DYNAMIC_RESIDUAL_SERVO_LAG;
    p.max_steer=.4788f; p.min_accel=-10.f; p.max_accel=10.f;
    p.min_speed=0.f; p.max_speed=10.f; p.kf_low_speed_threshold=.5f;
    p.q_v=12.f; p.q_heading=1.f; p.q_du=.12f; p.q_steer=.3f;
    p.q_obs=15000.f; p.q_lat_g=60.f; p.lat_g_soft_limit=9.5f;
    p.longitudinal_accel_soft_limit=4.f; p.q_rear_slip=800.f;
    p.rear_slip_soft_limit=.13962634f; p.rear_slip_cost_min_speed=1.5f;
    p.q_progress=85.f; p.q_escape_vel=28.f; p.collision_radius=.4f;
    p.q_boundary_slack=15000.f; p.q_boundary_terminal_slack=15000.f;
    p.noise_steer_std=.35f; p.noise_accel_std=2.f;
    p.max_steer_rate=50.f; p.max_accel_rate=4.f; p.lambda=20.f;
    p.mass=3.74f; p.I_z=.04712f; p.dynamic_mlp_I_z=.04712f;
    p.l_f=.163f; p.l_r=.161f; p.F_zf=p.mass*9.81f*p.l_r/(p.l_f+p.l_r);
    p.F_zr=p.mass*9.81f*p.l_f/(p.l_f+p.l_r);
    p.kinematic_steer_scale=.50927964f; p.kinematic_steer_bias=.01015773f;
    p.kinematic_position_speed_scale=.86334913f;
    p.steer_servo_time_constant=.15514851f; p.actuator_max_steer_rate=.8344091f;
    p.speed_servo_kp=.76168889f; p.speed_reference_accel_time_constant=.04f;
    p.speed_reference_brake_time_constant=.02f; p.actuator_max_speed_reference_rate=8.f;
    p.kinematic_yaw_rate_time_constant=.1f; p.kinematic_max_yaw_accel=15.f;
    p.dynamic_mlp_B_f=9.12282176f; p.dynamic_mlp_C_f=.50023478f;
    p.dynamic_mlp_D_f=.66502247f; p.dynamic_mlp_E_f=.66150913f;
    p.dynamic_mlp_B_r=4.20691398f; p.dynamic_mlp_C_r=.75200864f;
    p.dynamic_mlp_D_r=.52704648f; p.dynamic_mlp_E_r=-.99773961f;
    p.dynamic_mlp_min_speed=.8f;
    p.dynamic_mlp_max_residual_yaw_accel=12.f;p.dynamic_mlp_residual_gate_steer_start=.40f;
    p.dynamic_mlp_residual_gate_steer_end=.4788f;p.dynamic_mlp_max_total_yaw_accel=12.f;
    p.dynamic_mlp_yaw_rate_kinematic_scale=.75f;p.dynamic_mlp_yaw_rate_margin=.35f;
    p.dynamic_mlp_yaw_rate_lateral_accel_limit=9.5f;
    p.visualize_candidates=argc > 3 && std::stoi(argv[3]) != 0;
    p.actuator_speed_reference_state=3.f;
    p.objective_mode=use_safe_set?mppi::LMPC_OBJECTIVE:mppi::MPCC_OBJECTIVE;
    p.safe_set_count=use_safe_set?40:0;
    p.q_terminal_safe_set_slack=1000.f;p.safe_set_cost_coefficient=10.f;
    p.safe_set_inv_x_scale=1.f;p.safe_set_inv_y_scale=1.f;p.safe_set_inv_yaw_scale=2.f;
    for(int i=0;i<p.safe_set_count;++i){const float a=.02f*i;
        p.safe_set_x[i]=8.f*cosf(a);p.safe_set_y[i]=8.f*sinf(a);
        p.safe_set_yaw[i]=a+1.57079632679f;p.safe_set_cost[i]=float(19-(i%20));}
    for(int i=0;i<5;++i) p.residual_command_history[2*i+1]=3.f;

    mppi::MPPISolver solver(samples,horizon,p);
    if (use_mlp) solver.load_dynamic_mlp_residual_weights(
        "config/dynamic_40ms_residual_servo_lag.bin");
    constexpr int n=400; constexpr float radius=8.f;
    std::vector<float>x(n),y(n),yaw(n),lx(n),ly(n),rx(n),ry(n);
    for(int i=0;i<n;++i){
        const float a=2.f*3.14159265358979323846f*i/n;
        x[i]=radius*cosf(a); y[i]=radius*sinf(a); yaw[i]=a+1.57079632679f;
        lx[i]=(radius-1.f)*cosf(a);ly[i]=(radius-1.f)*sinf(a);
        rx[i]=(radius+1.f)*cosf(a);ry[i]=(radius+1.f)*sinf(a);
    }
    solver.set_reference_path(x,y,yaw);solver.set_boundaries(lx,ly,rx,ry);
    mppi::State s{};s.x=radius;s.y=0.f;s.yaw=1.57079632679f;s.v=3.f;
    for(int i=0;i<20;++i) solver.solve(s);
    constexpr int iterations=100;
    const auto begin=std::chrono::steady_clock::now();
    for(int i=0;i<iterations;++i) solver.solve(s);
    const auto end=std::chrono::steady_clock::now();
    const double ms=std::chrono::duration<double,std::milli>(end-begin).count()/iterations;
    std::cout<<"model="<<(use_mlp?"mlp":"classic")<<" samples="<<samples
             <<" horizon="<<horizon<<" safe_set="<<use_safe_set
             <<" mean_solve_ms="<<ms<<"\n";
}
