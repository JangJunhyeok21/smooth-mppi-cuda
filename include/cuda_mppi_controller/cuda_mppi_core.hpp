#ifndef MPPI_CORE_HPP_
#define MPPI_CORE_HPP_

#include <vector>
#include <random>
#include <memory>
#include <iostream>
#include <cstdio>
#include <string>

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

#define MAX_OBS 5      // 처리 가능한 최대 장애물 개수
#define MAX_CIRCLES 8  // 차체 형상 근사에 쓰는 최대 원 개수 (변경 시 smppi_lib/smppi_node 재컴파일 필요)

namespace mppi {

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
    float ay; 
    float slip_angle;
};

struct alignas(8) Control {
    float steer;
    float accel;
};

struct Params {
    float dt;
    
    // Limits
    float max_steer;
    float min_accel;
    float max_accel;
    float min_speed;
    float max_speed;
    
    // Cost Weights
    float q_dist;
    float q_v;
    float q_du;
    float q_steer;
    float q_collision;
    float q_lat_g;
    float lat_g_threshold;       // 횡가속도 소프트 페널티 시작 임계값 [m/s^2]
    float lat_g_fault_threshold; // 횡가속도 하드 페일(is_fault) 임계값 [m/s^2]
    float q_progress;
    float q_escape_vel;
    float collision_radius;
    float boundary_margin;     // collision_radius에 더해지는 소프트 페널티 시작 여유거리 [m]
    float boundary_soft_gain;  // 소프트 페널티 이차항 계수
    
    // Obstacle Avoidance Params
    int num_obstacles;
    float obs_x[MAX_OBS];
    float obs_y[MAX_OBS];
    float car_radius;
    float q_obs;

    // Multi-Circle 차체 형상 근사 (base_link=후륜축 기준, +x 전방 로컬 오프셋)
    int num_circles;
    float circle_offset[MAX_CIRCLES];
    float circle_radius;  // 모든 원 공통 반지름 [m] = vehicle_width/2 * safety_margin

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
    float l_f;
    float l_r;
    float Cm0;
    
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

    ButterworthCoeffs filter_coeffs;
};

// base_link(후륜축) 기준 +x 방향으로 [-rear_overhang, wheelbase+front_overhang] 구간을
// num_circles개의 원으로 등간격 커버하는 오프셋을 생성한다.
// num_circles<=1이면 base_link 원점(offset=0) 하나로 수렴한다 (단일 원 기존 동작과 동일).
// (호스트 전용 함수 — 정의부인 mppi_core.cu에서는 __host__로 컴파일되지만, 이 헤더는
//  smppi_node.cpp 등 nvcc가 아닌 일반 C++ 번역 단위에서도 include되므로 선언에는
//  __host__를 붙이지 않는다 — MPPISolver 멤버 함수들과 동일한 관례)
std::vector<float> compute_circle_offsets(
    int num_circles, float rear_overhang, float front_overhang, float wheelbase);

// 원들이 차량 폭(vehicle_width)과 종방향 간격을 빈틈없이 커버하는지 점검한다.
// 실패 시 false를 반환하고, reason_out이 non-null이면 사유를 덧붙인다.
bool validate_circle_coverage(
    const std::vector<float>& offsets, float circle_radius, float vehicle_width,
    std::string* reason_out = nullptr);

class MPPISolver {
public:
    MPPISolver(int K, int T, Params params);
    ~MPPISolver();

    void update_params(Params p);
    
    // 경로 및 바운더리 설정
    void set_reference_path(const std::vector<float>& xs, const std::vector<float>& ys,
                            const std::vector<float>& yaws);
    void set_boundaries(const std::vector<float>& left_xs, const std::vector<float>& left_ys,
                        const std::vector<float>& right_xs, const std::vector<float>& right_ys);
    
    Control solve(const State& current_state);
    
    const std::vector<State>& get_generated_trajectories() const;
    const std::vector<State>& get_best_trajectory() const;
    const std::vector<Control>& get_optimal_controls() const;
    const std::vector<float>& get_costs() const;
    
    int get_best_k() const;
    int get_K() const;
    int get_T() const;

private:
    void allocate_cuda_memory();
    void cleanup_cuda_memory();
    Control compute_optimal_control(const State &current_state);

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
    std::vector<Control> optimal_controls_;

    // Host Reference Path
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
    int ref_path_len_ = 0;

    // Boundary Device Memory
    float* d_left_bnd_xs_;
    float* d_left_bnd_ys_;
    float* d_right_bnd_xs_;
    float* d_right_bnd_ys_;
    int bnd_len_ = 0;
};
} // namespace mppi
#endif