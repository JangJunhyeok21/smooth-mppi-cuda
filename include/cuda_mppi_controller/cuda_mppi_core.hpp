#ifndef MPPI_CORE_HPP_
#define MPPI_CORE_HPP_

#include <vector>
#include <random>
#include <memory>
#include <iostream>
#include <cstdio>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// 에러 체크 매크로
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while (0)

namespace mppi {

// [수정] Dynamic Model을 위한 상태 변수 확장
struct alignas(16) State {
    float x;
    float y;
    float yaw;
    float v;      // vx (Longitudinal velocity)
    float vy;     // [추가] vy (Lateral velocity)
    float omega;  // [추가] Yaw rate
};

struct alignas(8) Control {
    float steer;
    float accel;
};

struct Params {
    float dt;
    float wheel_base; // Kinematic용 (Legacy)
    
    // Limits
    float max_steer;
    float min_accel;
    float max_accel;
    float min_speed;
    float target_speed;
    float max_speed;
    
    // Cost Weights
    float q_dist;
    float q_v;
    float q_u;
    float q_du;
    float q_heading;
    float q_lat;
    float q_collision;
    float collision_radius;
    
    // Noise & Tuning
    float noise_steer_std;
    float noise_accel_std;
    float max_steer_rate; // steer_vel_max * dt
    float max_accel_rate; // jerk_max * dt
    float lambda;
    bool visualize_candidates;

    // [추가] Vehicle Dynamics Params (User Provided)
    float mass;  // kg
    float I_z;   // kg*m^2
    float l_f;   // m
    float l_r;   // m
    
    // Pacejka Magic Formula Coefficients
    // Force = D * sin(C * atan(B * alpha))
    float B_f, C_f, D_f; // Front Tire
    float B_r, C_r, D_r; // Rear Tire
};

class MPPISolver {
public:
    MPPISolver(int K, int T, Params params);
    ~MPPISolver();

    void update_params(Params p);
    void set_reference_path(const std::vector<float>& xs, const std::vector<float>& ys,
                            const std::vector<float>& yaws, const std::vector<float>& vs);
    void set_scan_data(const std::vector<float>& ranges, float angle_min, float angle_inc);
    
    Control solve(const State& current_state);
    
    const std::vector<State>& get_generated_trajectories() const;
    const std::vector<State>& get_best_trajectory() const;
    int get_best_k() const;
    int get_K() const;
    int get_T() const;

private:
    void allocate_cuda_memory();
    void cleanup_cuda_memory();
    Control compute_optimal_control();

    int K_, T_;
    Params params_;
    
    // --- Host Memory ---
    std::vector<State> h_states_;       
    std::vector<Control> h_controls_;   
    std::vector<Control> h_prev_controls_; 
    std::vector<float> h_costs_;        
    std::vector<float> h_weights_;      
    int best_k_ = 0;
    std::vector<State> best_trajectory_;
    std::vector<float> h_ref_xs_;
    std::vector<float> h_ref_ys_;

    // --- Device Memory ---
    void* d_rng_states_;     
    State* d_states_;        
    Control* d_controls_;    
    Control* d_prev_controls_; 
    float* d_costs_;         
    
    float* d_ref_xs_;
    float* d_ref_ys_;
    float* d_ref_yaws_;
    float* d_ref_vs_;
    int ref_path_len_ = 0;

    float* d_scan_ranges_;
    int scan_len_ = 0;
    float scan_angle_min_ = 0.0f; 
    float scan_angle_inc_ = 0.0f;  
};
} // namespace mppi
#endif