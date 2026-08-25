#ifndef MPPI_CORE_HPP_
#define MPPI_CORE_HPP_

#include <vector>
#include <random>
#include <memory>
#include <iostream>
#include <cstdio>
#include <string>
#include <array>
#include <cstdint>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while (0)

#define MAX_DYNAMIC_OBSTACLE_HORIZON 120
#define MAX_SAFE_SET_POINTS 40 // LMPC: recent 2 laps x K_NEAR(20)
#define RESIDUAL_HISTORY 50
#define RESIDUAL_FEATURES 11
#define RESIDUAL_HIDDEN 96

namespace mppi {

enum DynamicsModel : int {
    LEGACY_HYBRID = 0,
    KINEMATIC = 1,
    KINEMATIC_RESIDUAL = 2,
    KINEMATIC_MLP_RESIDUAL = 3,
    KINEMATIC_MLP_NO_IMU_RESIDUAL = 4,
    KINEMATIC_NOSLIP_NO_IMU_DIRECT_SPEED = 5,
    DYNAMIC_IMU_RECURSIVE = 6,
    SLIP_KINEMATIC_WITH_IMU_DIRECT_SPEED = 7,
    DYNAMIC_MLP_RESIDUAL = 8,
    DYNAMIC_MLP_RESIDUAL_SERVO_LAG = 9,
    DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D = 11,
    DYNAMIC_SERVO_LAG = 12,
};

enum ObjectiveMode : int {
    MPCC_OBJECTIVE = 0,
    LMPC_OBJECTIVE = 1,
};

struct alignas(16) ButterworthCoeffs {
    float b0, b1, b2, a1, a2;
};

struct alignas(16) State {
    float x;
    float y;
    float yaw;
    float v;
    float vy;
    float omega;
    float ax;
    float ay; 
    float slip_angle;
};

struct alignas(8) Control {
    float steer;
    float accel;
};

struct Params {
    // Controller/publisher period and rollout state-transition period are
    // deliberately separate. Legacy models continue to use dt.
    float dt;
    float control_dt;
    float model_dt;

    // Runtime-selectable rollout model (DynamicsModel).
    int dynamics_model;
    float residual_imu[3];
    float residual_command_history[10];
    float residual_vx_history[5];
    float actuator_steer_state;
    float actuator_speed_reference_state;
    float steer_servo_time_constant;
    float actuator_max_steer_rate;
    float speed_reference_accel_time_constant;
    float speed_reference_brake_time_constant;
    float actuator_max_speed_reference_rate;
    
    // Limits
    float max_steer;
    float min_accel;
    float max_accel;
    float min_speed;
    float max_speed;
    // Cost Weights
    float q_dist;
    float q_contour;
    float q_lag;
    float q_heading;
    float q_error_speed;
    float q_v;
    float q_du;
    float q_steer;
    float q_lat_g;
    float lat_g_soft_limit;
    float longitudinal_accel_soft_limit;
    // Front-tire slip soft constraint. The applied (servo-lagged) steering
    // angle is used, not the raw command.
    float q_front_slip;
    float front_slip_soft_limit;
    float front_slip_cost_min_speed;
    // Rear-tire slip soft constraint. The threshold is stored in radians.
    float q_rear_slip;
    float rear_slip_soft_limit;
    float rear_slip_cost_min_speed;
    float q_progress;
    float q_escape_vel;
    float collision_radius;
    float q_boundary_slack;
    float q_boundary_terminal_slack;

    // LMPC terminal safe set. The host selects the same local 2*K_NEAR
    // samples as lmpc.cpp; each rollout solves the convex terminal projection.
    int objective_mode;
    int safe_set_count;
    float q_terminal_safe_set_slack;
    float safe_set_cost_coefficient;
    float safe_set_inv_x_scale;
    float safe_set_inv_y_scale;
    float safe_set_inv_yaw_scale;
    float safe_set_x[MAX_SAFE_SET_POINTS];
    float safe_set_y[MAX_SAFE_SET_POINTS];
    float safe_set_yaw[MAX_SAFE_SET_POINTS];
    float safe_set_cost[MAX_SAFE_SET_POINTS];
    
    // Obstacle Avoidance Params
    int num_obstacles;
    float car_radius;
    // Runtime cap on predictor-reported obstacles. Sizes the dynamic-obstacle
    // CUDA buffers at construction time (allocate_cuda_memory()) and bounds
    // set_dynamic_obstacles()/the incoming DynamicObstacleTrajectory message.
    int max_obstacles;
    // Extra distance outside the hard collision radius where the smooth
    // obstacle cost starts acting.
    float obstacle_soft_margin;
    float q_obs;
    bool rollout_obstacle_ahead;

    // Noise & Tuning
    float noise_steer_std;
    float noise_accel_std;
    float max_steer_rate;
    float max_accel_rate;
    float lambda;
    bool visualize_candidates;

    // Dynamics
    float mass;
    float I_z;
    float kinematic_steer_scale;
    float kinematic_steer_bias;
    float kinematic_position_speed_scale;
    bool kinematic_no_slip;
    float l_f;
    float l_r;
    float Cm0;
    float speed_servo_kp;
    // First-order yaw-rate response toward the algebraic kinematic target.
    float kinematic_yaw_rate_time_constant;
    float kinematic_max_yaw_accel;
    
    // Pacejka (ForzaETH On-Track-SysID 와 동일한 4-파라미터 매직 포뮬러)
    //   F_y = F_z * D * sin( C * atan( B*a - E*(B*a - atan(B*a)) ) )
    // D 는 무차원(마찰계수 성격)이고 실제 힘은 정하중 F_z 를 곱해 나온다.
    // 기존 3-파라미터 형태(D 가 뉴턴 단위)에서 바뀌었다 — E=0, D_new=D_old/F_z 로 두면
    // 수식이 정확히 예전 형태로 환원되므로 무손실 마이그레이션이 가능하다.
    // F_zf/F_zr 은 smppi_node 가 F_zf = m*g*l_r/l_wb, F_zr = m*g*l_f/l_wb 로 채워 준다
    // (pacejka_sysid/helpers/generate_predictions.py 와 동일한 정하중 규약).
    float B_f, C_f, D_f, E_f;
    float B_r, C_r, D_r, E_r;
    float F_zf, F_zr;

    // Parameters identified specifically for DYNAMIC_MLP_RESIDUAL.  Keep
    // these separate from the legacy Pacejka model and the lateral KF: the
    // latter legitimately uses the physical I_z above.
    float dynamic_mlp_B_f, dynamic_mlp_C_f, dynamic_mlp_D_f, dynamic_mlp_E_f;
    float dynamic_mlp_B_r, dynamic_mlp_C_r, dynamic_mlp_D_r, dynamic_mlp_E_r;
    float dynamic_mlp_I_z;
    // Limits only the learned correction, never a measured/predicted state.
    float mlp_max_residual_ax;
    float mlp_max_residual_ay;
    float mlp_max_residual_yaw_accel;

    ButterworthCoeffs filter_coeffs;
};

class MPPISolver {
public:
    MPPISolver(int K, int T, Params params);
    ~MPPISolver();

    void update_params(Params p);
    void set_residual_history(const std::vector<float>& normalized_features);
    void load_residual_weights(const std::string& path);
    void load_mlp_residual_weights(const std::string& path);
    void load_mlp_no_imu_residual_weights(const std::string& path);
    void load_kinematic_noslip_noimu_direct_weights(const std::string& path);
    void load_slip_kinematic_with_imu_direct_weights(const std::string& path);
    void load_dynamic_imu_recursive_weights(const std::string& path);
    void load_dynamic_mlp_residual_weights(const std::string& path);
    void load_dynamic_mlp_vx_delta_residual_weights(const std::string& path);
    // Validation-only entry point. It launches the exact CUDA rollout step,
    // including actuator states, feature construction, MLP and integration.
    State debug_dynamic_mlp_residual_step(
        const State& state, const Control& control,
        std::array<float, 12>& command_history, bool use_servo_lag);
    State debug_dynamic_mlp_vx_delta_residual_step(
        const State& state, const Control& control,
        std::array<float, 12>& command_history,
        std::array<float, 5>& vx_history);
    
    // 경로 및 바운더리 설정
    void set_reference_path(const std::vector<float>& xs, const std::vector<float>& ys,
                            const std::vector<float>& yaws);
    void set_boundaries(const std::vector<float>& left_xs, const std::vector<float>& left_ys,
                        const std::vector<float>& right_xs, const std::vector<float>& right_ys);
    void set_dynamic_obstacles(const std::vector<float>& xs,
                               const std::vector<float>& ys,
                               const std::vector<float>& yaws,
                               const std::vector<float>& semi_major,
                               const std::vector<float>& semi_minor,
                               const std::vector<bool>& is_dynamic,
                               int obstacle_count, int horizon);
    
    Control solve(const State& current_state);
    
    const std::vector<State>& get_generated_trajectories() const;
    const std::vector<State>& get_best_trajectory() const;
    // Rollout produced by feeding the actually published MPPI weighted controls
    // through the same CUDA dynamics model used by the candidates.
    const std::vector<State>& get_weighted_control_trajectory() const;
    const std::vector<Control>& get_optimal_controls() const;
    const std::vector<float>& get_costs() const;
    
    int get_best_k() const;
    int get_K() const;
    int get_T() const;

private:
    void allocate_cuda_memory();
    void cleanup_cuda_memory();
    Control compute_optimal_control(const State &current_state);
    float trajectory_min_boundary_clearance(const std::vector<State>& trajectory) const;
    float trajectory_min_obstacle_clearance(const std::vector<State>& trajectory) const;

    int K_, T_;
    Params params_;
    
    // --- Host Memory ---
    std::vector<State> h_states_;       
    std::vector<Control> h_prev_controls_; 
    bool direct_speed_warm_start_initialized_{false};
    int model_knot_phase_{0};
    std::vector<float> h_costs_;        
    std::vector<float> h_weights_;      
    std::vector<float> h_min_boundary_clearances_;
    std::vector<float> h_min_obstacle_clearances_;
    int best_k_ = 0;
    std::vector<State> best_trajectory_;
    std::vector<State> weighted_control_trajectory_;
    std::vector<Control> optimal_controls_;

    // Host Reference Path
    std::vector<float> h_ref_xs_;
    std::vector<float> h_ref_ys_;
    std::vector<float> h_ref_yaws_;
    std::vector<float> h_left_bnd_xs_;
    std::vector<float> h_left_bnd_ys_;
    std::vector<float> h_right_bnd_xs_;
    std::vector<float> h_right_bnd_ys_;
    std::vector<float> h_dynamic_obs_x_;
    std::vector<float> h_dynamic_obs_y_;
    std::vector<float> h_dynamic_obs_yaw_;
    std::vector<float> h_dynamic_obs_semi_major_;
    std::vector<float> h_dynamic_obs_semi_minor_;
    std::vector<bool> h_dynamic_obs_is_dynamic_;
    int dynamic_obstacle_count_ = 0;
    int dynamic_obstacle_horizon_ = 0;

    // --- Device Memory ---
    void* d_rng_states_;     
    State* d_states_;        
    Control* d_controls_;    
    Control* d_prev_controls_; 
    State* d_weighted_states_;
    Control* d_weighted_controls_;
    float* d_costs_;         
    float* d_weights_;
    float* d_min_boundary_clearances_;
    float* d_min_obstacle_clearances_;
    float* d_residual_history_;
    float* d_residual_hidden_;
    
    float* d_ref_xs_;
    float* d_ref_ys_;
    float* d_ref_yaws_;
    int ref_path_len_ = 0;

    // Boundary Device Memory
    float* d_left_bnd_xs_;
    float* d_left_bnd_ys_;
    float* d_right_bnd_xs_;
    float* d_right_bnd_ys_;
    float* d_dynamic_obs_x_;
    float* d_dynamic_obs_y_;
    float* d_dynamic_obs_yaw_;
    float* d_dynamic_obs_semi_major_;
    float* d_dynamic_obs_semi_minor_;
    std::uint8_t* d_dynamic_obs_is_dynamic_;
    int bnd_len_ = 0;
};
} // namespace mppi
#endif
