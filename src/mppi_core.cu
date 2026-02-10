#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace mppi
{

    // --- CUDA Device Functions ---

    __device__ float angle_normalize_cuda(float angle)
    {
        while (angle > M_PI)
            angle -= 2.0f * M_PI;
        while (angle < -M_PI)
            angle += 2.0f * M_PI;
        return angle;
    }

    __device__ State update_dynamics_cuda(const State &s, const Control &u, const Params &p)
    {
        State next_s;
        float v = s.v;
        float yaw = s.yaw;

        next_s.x = s.x + v * cosf(yaw) * p.dt;
        next_s.y = s.y + v * sinf(yaw) * p.dt;
        next_s.yaw = angle_normalize_cuda(yaw + (v / p.wheel_base) * tanf(u.steer) * p.dt);
        next_s.v = v + u.accel * p.dt;

        if (next_s.v < p.min_speed)
            next_s.v = p.min_speed;
        return next_s;
    }

    // [수정됨] 유턴 방지 로직이 추가된 비용 함수
    __device__ float compute_cost_cuda(
        const State& s,
        const float* ref_xs, const float* ref_ys, const float* ref_vs, int path_len,
        const Control& u, const Control& u_prev,
        const Params& p) 
    {
        float min_dist_sq = 1e9f;
        int nearest_idx = -1;

        // [핵심 수정 1] 전체 경로를 다 뒤지지 말고, 주변만 탐색 (연산 속도 + 유턴 방지)
        // 하지만 GPU 병렬 처리 특성상 로컬 인덱스를 알기 어려우므로,
        // "차량의 진행 방향(Yaw)과 경로의 방향이 비슷한지" 체크하는 로직 추가.

        for (int i = 0; i < path_len; ++i) {
            float dx = s.x - ref_xs[i];
            float dy = s.y - ref_ys[i];
            float dist_sq = dx * dx + dy * dy;

            // 거리가 너무 멀면(예: 10m 이상) 아예 후보에서 제외 (넓은 공터에서 엉뚱한 점 잡는 것 방지)
            if (dist_sq > 100.0f) continue; 

            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }
        
        // 만약 유효한 경로점을 못 찾았다면(너무 멀어서), 그냥 엄청 큰 비용 부과
        if (nearest_idx == -1) return 1.0e9f;

        // [핵심 수정 2] 유턴 방지: 진행 방향 벡터 내적 (Dot Product)
        // 차량이 바라보는 방향(cos, sin)과 
        // 경로점이 위치한 방향(path_x - car_x, path_y - car_y)이 반대면 페널티
        
        // 경로의 접선 방향(Tangent)을 구해서 차량 헤딩과 비교하는 것이 가장 정확함.
        // 여기서는 간단하게 "경로점 인덱스가 진행하는 방향"을 따르도록 유도.
        // 하지만 GPU 내에서 인덱스 추적은 복잡하므로, "Heading Error"를 추가합니다.

        // 1. 거리 오차 (기존)
        float dist_error = min_dist_sq;
        
        // 2. 속도 오차 (기존)
        float ref_v = ref_vs[nearest_idx];
        float v_error = (s.v - ref_v) * (s.v - ref_v);

        // 3. [추가] 헤딩(Heading) 오차 비용
        // 경로 상의 다음 점을 보고 방향을 구함
        int next_idx = (nearest_idx + 1 < path_len) ? nearest_idx + 1 : nearest_idx;
        float path_dx = ref_xs[next_idx] - ref_xs[nearest_idx];
        float path_dy = ref_ys[next_idx] - ref_ys[nearest_idx];
        float path_yaw = atan2f(path_dy, path_dx);
        
        float yaw_diff = angle_normalize_cuda(s.yaw - path_yaw);
        // 방향이 90도 이상 틀어지면 비용 폭증 -> 유턴 방지
        float heading_cost = 0.0f;
        if (abs(yaw_diff) > 1.57f) { // 90도(PI/2) 이상
            heading_cost = 1000.0f; 
        } else {
            heading_cost = 2.0f * (yaw_diff * yaw_diff); // 평소에는 부드럽게
        }

        // 4. 제어 입력 비용들 (기존)
        float lat_accel = (s.v * s.v) * tanf(u.steer) / p.wheel_base;
        float lat_cost = p.q_lat * (lat_accel * lat_accel);
        float input_cost = p.q_u * (u.steer * u.steer + u.accel * u.accel);
        float d_steer = u.steer - u_prev.steer;
        float d_accel = u.accel - u_prev.accel;
        float input_delta_cost = p.q_du * (d_steer * d_steer + d_accel * d_accel);

        // 최종 비용 합산
        return p.q_dist * dist_error 
            + p.q_v * v_error 
            + heading_cost    // [추가됨]
            + lat_cost + input_cost + input_delta_cost;
    }
    __device__ bool violates_scan_hard_constraint(
        const State &s,
        const float *scan_ranges, int scan_len,
        float scan_angle_min, float scan_angle_inc,
        const State &robot_pose,
        float collision_radius)
    {
        if (scan_ranges == nullptr || scan_len <= 0)
            return false;

        float dx = s.x - robot_pose.x;
        float dy = s.y - robot_pose.y;
        float local_x = dx * cosf(robot_pose.yaw) + dy * sinf(robot_pose.yaw);
        float local_y = -dx * sinf(robot_pose.yaw) + dy * cosf(robot_pose.yaw);

        const float min_dist_sq = collision_radius * collision_radius;

        for (int i = 0; i < scan_len; ++i)
        {
            float r = scan_ranges[i];
            if (isinf(r) || isnan(r))
                continue;

            float ang = scan_angle_min + i * scan_angle_inc;
            float obs_x = r * cosf(ang);
            float obs_y = r * sinf(ang);

            float dx_obs = local_x - obs_x;
            float dy_obs = local_y - obs_y;
            float dist_sq_obs = dx_obs * dx_obs + dy_obs * dy_obs;

            if (dist_sq_obs < min_dist_sq)
            {
                return true;
            }
        }
        return false;
    }

    // --- CUDA Kernels ---

    __global__ void init_rng_kernel(curandState *states, long seed, int K, int T)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < K * T)
        {
            curand_init(seed, idx, 0, &states[idx]);
        }
    }

    __global__ void rollout_kernel(
        State *states,           // [Output] Generated Trajectories (K * T)
        Control *controls,       // [Output] Sampled Controls (K * T)
        float *costs,            // [Output] Costs per sample (K)
        curandState *rng_states, // [Input/Output] RNG states
        const State start_state,
        const Control *prev_controls, // [Input] Previous Mean Controls (T)
        const Params p,
        const float *ref_xs, const float *ref_ys, const float *ref_vs, int path_len,
        const float *scan_ranges, int scan_len, float scan_angle_min, float scan_angle_inc,
        int K, int T)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= K)
            return;

        State x = start_state;
        float total_cost = 0.0f;
        Control last_u = prev_controls[0];

        for (int t = 0; t < T; ++t)
        {
            int idx = k * T + t;

            // 1. 노이즈 생성
            float noise_steer = curand_normal(&rng_states[idx]) * 0.1f;
            float noise_accel = curand_normal(&rng_states[idx]) * 0.2f;

            // 2. 제어 입력 적용 (Mean + Noise)
            Control u = prev_controls[t];
            u.steer += noise_steer;
            u.accel += noise_accel;

            // Clamp
            if (u.steer > p.max_steer)
                u.steer = p.max_steer;
            else if (u.steer < -p.max_steer)
                u.steer = -p.max_steer;
            if (u.accel > p.max_accel)
                u.accel = p.max_accel;
            else if (u.accel < p.min_accel)
                u.accel = p.min_accel;

            // 3. 상태 업데이트
            x = update_dynamics_cuda(x, u, p);
            states[idx] = x; // 시각화용 저장

            // 제어 입력 저장 (최적 제어 복원용)
            controls[idx] = u;

            // 4. Hard Collision Constraint
            if (violates_scan_hard_constraint(x, scan_ranges, scan_len, scan_angle_min, scan_angle_inc,
                                              start_state, p.collision_radius))
            {
                total_cost = 1.0e9f;
                break;
            }

            // 5. 비용 계산
            if (path_len > 0)
            {
                total_cost += compute_cost_cuda(
                    x, ref_xs, ref_ys, ref_vs, path_len,
                    u, last_u, p);
            }
            last_u = u;
        }
        costs[k] = total_cost;
    }

    // --- Host Implementation ---

    MPPISolver::MPPISolver(int K, int T, Params params)
        : K_(K), T_(T), params_(params)
    {

        h_states_.resize(K * T);
        h_controls_.resize(K * T);
        h_prev_controls_.resize(T, {0.0f, 0.0f});
        h_costs_.resize(K);
        h_weights_.resize(K);

        allocate_cuda_memory();
    }

    MPPISolver::~MPPISolver()
    {
        cleanup_cuda_memory();
    }

    void MPPISolver::allocate_cuda_memory()
    {
        cudaMalloc(&d_states_, K_ * T_ * sizeof(State));
        cudaMalloc(&d_controls_, K_ * T_ * sizeof(Control));
        cudaMalloc(&d_prev_controls_, T_ * sizeof(Control));
        cudaMalloc(&d_costs_, K_ * sizeof(float));
        cudaMalloc(&d_rng_states_, K_ * T_ * sizeof(curandState));

        int max_path = 2000;
        cudaMalloc(&d_ref_xs_, max_path * sizeof(float));
        cudaMalloc(&d_ref_ys_, max_path * sizeof(float));
        cudaMalloc(&d_ref_vs_, max_path * sizeof(float));

        // Scan data는 set_scan_data에서 동적으로 할당
        d_scan_ranges_ = nullptr;

        // RNG 초기화
        int threads = 256;
        int blocks = (K_ * T_ + threads - 1) / threads;
        init_rng_kernel<<<blocks, threads>>>((curandState *)d_rng_states_, 1234UL, K_, T_);
        cudaDeviceSynchronize();
    }

    void MPPISolver::cleanup_cuda_memory()
    {
        cudaFree(d_states_);
        cudaFree(d_controls_);
        cudaFree(d_prev_controls_);
        cudaFree(d_costs_);
        cudaFree(d_rng_states_);
        cudaFree(d_ref_xs_);
        cudaFree(d_ref_ys_);
        cudaFree(d_ref_vs_);
        if (d_scan_ranges_ != nullptr)
        {
            cudaFree(d_scan_ranges_);
            d_scan_ranges_ = nullptr;
        }
    }

    void MPPISolver::update_params(Params p) { params_ = p; }

    void MPPISolver::set_reference_path(const std::vector<float> &xs, const std::vector<float> &ys,
                                        const std::vector<float> &yaws, const std::vector<float> &vs)
    {
        ref_path_len_ = xs.size();
        if (ref_path_len_ > 2000)
            ref_path_len_ = 2000;

        if (ref_path_len_ > 0)
        {
            cudaMemcpy(d_ref_xs_, xs.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ref_ys_, ys.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ref_vs_, vs.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    void MPPISolver::set_scan_data(const std::vector<float> &ranges, float angle_min, float angle_inc)
    {
        // 1. 파라미터 업데이트
        scan_angle_min_ = angle_min;
        scan_angle_inc_ = angle_inc;

        int new_len = static_cast<int>(ranges.size());
        if (new_len <= 0)
        {
            scan_len_ = 0;
            return;
        }

        // 2. GPU 메모리 재할당 (크기가 변했을 때만)
        if (new_len != scan_len_)
        {
            if (d_scan_ranges_ != nullptr)
            {
                cudaFree(d_scan_ranges_);
            }
            cudaError_t err = cudaMalloc(&d_scan_ranges_, new_len * sizeof(float));
            if (err != cudaSuccess)
            {
                scan_len_ = 0;
                return;
            }
            scan_len_ = new_len;
        }

        // 3. 데이터 복사 (Host -> Device)
        cudaMemcpy(d_scan_ranges_, ranges.data(), scan_len_ * sizeof(float), cudaMemcpyHostToDevice);
    }

    Control MPPISolver::solve(const State &current_state)
    {
        // 1. 이전 제어 입력 복사 (Host -> Device)
        cudaMemcpy(d_prev_controls_, h_prev_controls_.data(), T_ * sizeof(Control), cudaMemcpyHostToDevice);

        // 2. 롤아웃 커널 실행
        int threadsPerBlock = 128;
        int blocksPerGrid = (K_ + threadsPerBlock - 1) / threadsPerBlock;

        rollout_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_states_,
            d_controls_,
            d_costs_,
            (curandState *)d_rng_states_,
            current_state,
            d_prev_controls_,
            params_,
            d_ref_xs_, d_ref_ys_, d_ref_vs_, ref_path_len_,
            d_scan_ranges_, scan_len_, scan_angle_min_, scan_angle_inc_,
            K_, T_);

        // 3. 결과 복사 (Device -> Host)
        cudaMemcpy(h_costs_.data(), d_costs_, K_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_states_.data(), d_states_, K_ * T_ * sizeof(State), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_controls_.data(), d_controls_, K_ * T_ * sizeof(Control), cudaMemcpyDeviceToHost);

        // 4. 최적 제어 계산 (Host에서 수행)
        return compute_optimal_control();
    }

    Control MPPISolver::compute_optimal_control()
    {
        // 1. 최적(최소 비용) 궤적 선택
        int best_k = 0;
        float min_cost = 1e9f;
        for (int k = 0; k < K_; ++k)
        {
            if (h_costs_[k] < min_cost)
            {
                min_cost = h_costs_[k];
                best_k = k;
            }
        }
        best_k_ = best_k;

        // 2. 최적 궤적의 첫 제어 입력 사용
        Control output = h_controls_[best_k * T_ + 0];
        // t번째 칸에 t+1번째 최적 입력을 당겨와서 저장 (Shift)
        for (int t = 0; t < T_ - 1; ++t)
        {
            h_prev_controls_[t] = h_controls_[best_k * T_ + (t + 1)];
        }
        h_prev_controls_[T_ - 1] = h_prev_controls_[T_ - 2];
        return output;
    }

    const std::vector<State> &MPPISolver::get_generated_trajectories() const { return h_states_; }
    int MPPISolver::get_best_k() const { return best_k_; }
    int MPPISolver::get_K() const { return K_; }
    int MPPISolver::get_T() const { return T_; }

} // namespace mppi