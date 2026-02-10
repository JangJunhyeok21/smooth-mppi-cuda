#ifndef MPPI_CORE_HPP_
#define MPPI_CORE_HPP_

#include <vector>
#include <random>
#include <memory>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace mppi {

struct alignas(16) State {
    float x;
    float y;
    float yaw;
    float v;
};

struct alignas(8) Control {
    float steer;
    float accel;
};

struct Params {
    float dt;
    float wheel_base;
    float max_steer;
    float min_accel;
    float max_accel;
    float min_speed;
    float target_speed;
    float max_speed;
    
    // Cost Weights
    float q_dist;
    float q_v;
    float q_lat;
    float q_u;
    float q_du;
    float collision_radius;
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
    
    // Visualization & Debugging
    const std::vector<State>& get_generated_trajectories() const;
    int get_best_k() const;
    bool is_cuda_active() const { return true; } // 에러 해결용
    int get_K() const;
    int get_T() const;

private:
    void allocate_cuda_memory();
    void cleanup_cuda_memory();
    Control compute_optimal_control();

    int K_, T_;
    Params params_;
    
    // --- Host Memory (CPU) ---
    std::vector<State> h_states_;       // 시각화용 궤적 데이터
    std::vector<Control> h_controls_;   // 샘플 제어 입력 (K * T)
    std::vector<Control> h_prev_controls_; // 이전 제어 입력 (T개)
    std::vector<float> h_costs_;        // 비용 데이터
    std::vector<float> h_weights_;      // 가중치
    int best_k_ = 0;

    // --- Device Memory Pointers (GPU) ---
    void* d_rng_states_;     // curandState*
    
    State* d_states_;        // K * T (Trajectory States)
    Control* d_controls_;    // K * T (Sampled Controls)
    Control* d_prev_controls_; // T (Previous Mean Controls)
    float* d_costs_;         // K (Costs)
    
    // Reference Path (Device)
    float* d_ref_xs_;
    float* d_ref_ys_;
    float* d_ref_vs_;
    int ref_path_len_ = 0;

    // Scan Data (Device) - 필요 시 확장
    float* d_scan_ranges_;
    int scan_len_ = 0;
    float scan_angle_min_ = 0.0f; // 스캔 각도 최소값
    float scan_angle_inc_ = 0.0f;  // 스캔 각도 증가값
};
} // namespace mppi
#endif