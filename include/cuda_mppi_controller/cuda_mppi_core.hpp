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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while (0)

#define MAX_OBS 5 // 처리 가능한 최대 장애물 개수

namespace mppi {

struct alignas(16) ButterworthCoeffs {
    float b0, b1, b2, a1, a2;
};

// λ(lambda) 튠 진단용 — compute_optimal_control()에서 매 틱 K개 샘플에 대해 계산.
// ESS(Kish's effective sample size) = (Σw_k)^2 / Σw_k^2 : λ가 너무 작으면 최소비용
// 샘플 근처로만 쏠려 ESS가 작아지고(과적합), 너무 크면 균등가중(ESS→K)에 가까워져
// 비용 정보를 거의 반영하지 못한다.
struct EssStats {
    float mean_dcost = 0.0f;  // K개 샘플의 평균 코스트 갭 mean(C_k - C_min)
    float ess = 0.0f;         // 유효 샘플 수
    float ess_ratio = 0.0f;   // ess / K
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
    
    // Obstacle Avoidance Params
    int num_obstacles;
    float obs_x[MAX_OBS];
    float obs_y[MAX_OBS];
    float car_radius;
    float q_obs;

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

    // 저속(kinematic) / 고속(Pacejka dynamic) 모델을 tanh로 연속 블렌딩하기
    // 위한 전이 구간 중심 속도와 폭. sigma=0.5*(1+tanh((v-center)/width)).
    float v_blend_center;
    float v_blend_width;

    ButterworthCoeffs filter_coeffs;
};

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
    const EssStats& get_ess_stats() const;

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
    EssStats ess_stats_;
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