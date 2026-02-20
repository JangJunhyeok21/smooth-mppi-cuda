#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace mppi
{
    __device__ float angle_normalize_cuda(float angle)
    {
        while (angle > M_PI) angle -= 2.0f * M_PI;
        while (angle < -M_PI) angle += 2.0f * M_PI;
        return angle;
    }

    __device__ State update_dynamics_cuda(const State &s, const Control &u, const Params &p)
    {
        float px = s.x; float py = s.y; float yaw = s.yaw;
        float vx = s.v; float vy = s.vy; float omega = s.omega;
        
        if (vx < 0.1f) {
            State next_s;
            float beta = atanf(p.l_r * tanf(u.steer) / (p.l_f + p.l_r));
            next_s.x = px + vx * __cosf(yaw + beta) * p.dt;
            next_s.y = py + vx * __sinf(yaw + beta) * p.dt;
            next_s.yaw = angle_normalize_cuda(yaw + (vx / p.l_r) * __sinf(beta) * p.dt);
            next_s.v = vx + u.accel * p.dt;
            next_s.vy = 0.0f; next_s.omega = 0.0f;
            return next_s;
        }

        float alpha_f = u.steer - atan2f(vy + p.l_f * omega, vx);
        float alpha_r = -atan2f(vy - p.l_r * omega, vx);

        float F_fy = p.D_f * __sinf(p.C_f * atanf(p.B_f * alpha_f));
        float F_ry = p.D_r * __sinf(p.C_r * atanf(p.B_r * alpha_r));

        float F_rx = p.mass * u.accel; 
        float dot_vx = (F_rx - F_fy * __sinf(u.steer) + p.mass * vy * omega) / p.mass;
        float dot_vy = (F_ry + F_fy * __cosf(u.steer) - p.mass * vx * omega) / p.mass;
        float dot_omega = (F_fy * p.l_f * __cosf(u.steer) - F_ry * p.l_r) / p.I_z;

        float dot_x = vx * __cosf(yaw) - vy * __sinf(yaw);
        float dot_y = vx * __sinf(yaw) + vy * __cosf(yaw);

        State next_s;
        next_s.x = px + dot_x * p.dt;
        next_s.y = py + dot_y * p.dt;
        next_s.yaw = angle_normalize_cuda(yaw + omega * p.dt);
        next_s.v = vx + dot_vx * p.dt;
        next_s.vy = vy + dot_vy * p.dt;
        next_s.ay = dot_vy + vx * omega;
        next_s.omega = omega + dot_omega * p.dt;
        next_s.slip_angle = atan2f(next_s.vy, fabsf(next_s.v) + 1e-5f);

        return next_s;
    }

    __device__ float compute_cost_cuda(
        const State &s,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws, const float *ref_vs, int path_len,
        const Control &u, const Control &u_prev,
        const Params &p,
        float min_bnd_dist,
        int* last_idx)  
    {
        float min_dist_sq = 1e9f;
        int nearest_idx = -1;

        int start_search = *last_idx; 
        int search_window = 50; 
        
        if (start_search >= path_len) start_search %= path_len;
        if (start_search < 0) start_search = 0;

        for (int offset = 0; offset < search_window; ++offset)
        {
            int i = start_search + offset;
            if (i >= path_len) i -= path_len; 

            float dx = s.x - ref_xs[i];
            float dy = s.y - ref_ys[i];
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }

        if (nearest_idx == -1) nearest_idx = start_search;
        *last_idx = nearest_idx; 

        // 1. Reference Tracking Cost
        float dist_error = min_dist_sq;

        float vel_cost = 0.0f;
        // float ref_v = p.target_speed;=
        // vel_cost=p.q_v * (s.v - ref_v) * (s.v - ref_v);

        // 2. 빠른 속도 보상
        vel_cost = -p.q_v * (s.v*__cosf(s.yaw - ref_yaws[nearest_idx]));

        // 3. Control Input Cost
        float d_steer = u.steer - u_prev.steer;
        float d_accel = u.accel - u_prev.accel;
        float rate_cost = p.q_du * (d_steer * d_steer + d_accel * d_accel);


        // 5. Boundary Collision Cost
        float boundary_cost = 0.0f;
        
        boundary_cost= p.q_collision * logf(1.0f + expf(-50.0f * (min_bnd_dist - p.collision_radius))); // 바운더리 근접 시 급격히 증가하는 비용
        
        return p.q_dist * dist_error + vel_cost + rate_cost + boundary_cost;
    }
    
    // O(1) 윈도우 기반 바운더리 거리 계산 (슬라럼 대응 윈도우 100 확장)
    __device__ float compute_min_boundary_distance(
        const State &s,
        const float *left_xs, const float *left_ys,
        const float *right_xs, const float *right_ys,
        int bnd_len,
        int current_path_idx) 
    {
        if (left_xs == nullptr || right_xs == nullptr || bnd_len <= 0) return 1e9f;

        float min_dist_sq = 1e9f;
        
        // 탐색 창(Window) 확장: 뒤로 20, 앞으로 280칸을 확인하여 깊은 급커브 회피 보장
        int search_window = 300; 
        int start_search = current_path_idx - 20;
        
        if (start_search < 0) start_search += bnd_len; 

        for (int offset = 0; offset < search_window; ++offset)
        {
            int i = start_search + offset;
            if (i >= bnd_len) i -= bnd_len;

            float dx_l = s.x - left_xs[i];
            float dy_l = s.y - left_ys[i];
            float dist_sq_l = dx_l * dx_l + dy_l * dy_l;

            float dx_r = s.x - right_xs[i];
            float dy_r = s.y - right_ys[i];
            float dist_sq_r = dx_r * dx_r + dy_r * dy_r;

            if (dist_sq_l < min_dist_sq) min_dist_sq = dist_sq_l;
            if (dist_sq_r < min_dist_sq) min_dist_sq = dist_sq_r;
        }

        return sqrtf(min_dist_sq);
    }

    __global__ void init_rng_kernel(curandState *states, long seed, int K, int T)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < K * T) curand_init(seed, idx, 0, &states[idx]);
    }

    __global__ void rollout_kernel(
        State *states, Control *controls, float *costs, curandState *rng_states,
        const State start_state, const Control *prev_controls, const Params p,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws, const float *ref_vs, int path_len,
        const float *left_bnd_xs, const float *left_bnd_ys,
        const float *right_bnd_xs, const float *right_bnd_ys,
        int bnd_len,
        int K, int T, int start_path_idx)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= K) return;

        State x = start_state;
        float total_cost = 0.0f;
        Control current_action = prev_controls[0]; 
        Control last_u = current_action;
        int local_path_idx = start_path_idx;
        float steer_noise_state = 0.0f; 
        float accel_noise_state = 0.0f;
        for (int t = 0; t < T; ++t)
        {
            int idx = k * T + t;

            Control u_mean_curr = prev_controls[t];
            Control u_mean_prev = (t == 0) ? prev_controls[0] : prev_controls[t-1];
            
            // 1. Mean Trajectory의 '틱당 변화량(Delta)' 산출
            float mean_delta_steer = u_mean_curr.steer - u_mean_prev.steer;  // [rad]
            float mean_delta_accel = u_mean_curr.accel - u_mean_prev.accel;  // [m/s^2]

            // 2. 파라미터(시간 단위)에 dt를 곱해 '틱당 노이즈 변화량(Delta)' 산출
            float noise_delta_steer = curand_normal(&rng_states[idx]) * p.noise_steer_std * p.dt;   // [rad]
            float noise_delta_accel = curand_normal(&rng_states[idx]) * p.noise_accel_std * p.dt;   // [m/s^2]

            // 3. 물리적 모터/구동계 한계(rate * dt)를 적용하여 클램핑 후 적분(+=)
            current_action.steer += fminf(fmaxf(mean_delta_steer + noise_delta_steer, -p.max_steer_rate * p.dt), p.max_steer_rate * p.dt);    
            current_action.accel += fminf(fmaxf(mean_delta_accel + noise_delta_accel, -p.max_accel_rate * p.dt), p.max_accel_rate * p.dt);


            // // ========OU Process SMPPI 노이즈 적용 방식========
            // // 1. OU Process 기반 Colored Noise 생성 (핵심)
            // // theta 값이 1.0에 가까울수록 이전 노이즈 성향을 강하게 유지(관성)하여 
            // // 슬라럼을 위한 크고 굵은 조향 궤적을 탐색해 냅니다. (보통 0.8 ~ 0.9 권장)
            // float raw_steer_noise = curand_normal(&rng_states[idx]) * p.noise_steer_std;
            // float raw_accel_noise = curand_normal(&rng_states[idx]) * p.noise_accel_std;

            // steer_noise_state = 0.85f * steer_noise_state + (1.0f - 0.85f) * raw_steer_noise;
            // accel_noise_state = 0.85f * accel_noise_state + (1.0f - 0.85f) * raw_accel_noise;

            // // 2. 부드럽게 관성을 가진 노이즈를 변화량(dt)에 곱해 적용
            // float noise_delta_steer = steer_noise_state * p.dt;
            // float noise_delta_accel = accel_noise_state * p.dt;

            // // 3. Rate Limit 적용 및 누적
            // current_action.steer += fminf(fmaxf(mean_delta_steer + noise_delta_steer, -p.max_steer_rate * p.dt), p.max_steer_rate * p.dt);    
            // current_action.accel += fminf(fmaxf(mean_delta_accel + noise_delta_accel, -p.max_accel_rate * p.dt), p.max_accel_rate * p.dt);


            // 4. 절대 제어값 한계 적용
            Control u_clamped = current_action;
            u_clamped.steer = fminf(fmaxf(u_clamped.steer, -p.max_steer), p.max_steer);
            u_clamped.accel = fminf(fmaxf(u_clamped.accel, p.min_accel), p.max_accel);

            // 5. 속도 제한 적용
            float v_next = x.v + u_clamped.accel * p.dt;
            if (v_next >= p.max_speed && u_clamped.accel > 0.0f) u_clamped.accel = 0.0;
            else if (v_next <= p.min_speed && u_clamped.accel < 0.0f) u_clamped.accel = 0.0;

            current_action = u_clamped; 

            x = update_dynamics_cuda(x, u_clamped, p);
            states[idx] = x;
            controls[idx] = u_clamped; 

            // 하드 제약: 횡가속도 초과 시 후보군에서 완전 제외
            if(fabsf(x.ay) > 9.8f){
                total_cost = INFINITY;
                break;
            }

            if(fabsf(x.slip_angle) > 0.2f){ // 슬립각 0.2rad 이상 시 제약 위반으로 간주
                total_cost = INFINITY;
                break;
            }

            // 바운더리 기반 최소 거리 확인
            float min_dist = compute_min_boundary_distance(
                x, left_bnd_xs, left_bnd_ys, right_bnd_xs, right_bnd_ys, bnd_len, local_path_idx);
            
            // 하드 제약: 충돌 감지 시 후보군에서 완전 제외
            if (min_dist < p.collision_radius) {
                total_cost = INFINITY;
                break;
            }

            if (path_len > 0)
            {
                total_cost += compute_cost_cuda(
                    x, ref_xs, ref_ys, ref_yaws, ref_vs, path_len,
                    u_clamped, last_u, p, min_dist, &local_path_idx); 
            }
            last_u = u_clamped;
        }
        costs[k] = total_cost;
    }

    // --- Host Functions ---

    MPPISolver::MPPISolver(int K, int T, Params params) : K_(K), T_(T), params_(params) {
        h_states_.resize(K * T);
        h_controls_.resize(K * T);
        h_prev_controls_.resize(T, {0.0f, 0.0f});
        h_costs_.resize(K);
        h_weights_.resize(K);
        allocate_cuda_memory();
    }

    MPPISolver::~MPPISolver() { cleanup_cuda_memory(); }

    void MPPISolver::allocate_cuda_memory() {
        CUDA_CHECK(cudaMalloc(&d_states_, K_ * T_ * sizeof(State)));
        CUDA_CHECK(cudaMalloc(&d_controls_, K_ * T_ * sizeof(Control)));
        CUDA_CHECK(cudaMalloc(&d_prev_controls_, T_ * sizeof(Control)));
        CUDA_CHECK(cudaMalloc(&d_costs_, K_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_states_, K_ * T_ * sizeof(curandState)));
        
        // 메모리 한계 확장: 최대 10000개 포인트 처리 가능
        int max_path = 10000;
        CUDA_CHECK(cudaMalloc(&d_ref_xs_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_ys_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_yaws_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_vs_, max_path * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_left_bnd_xs_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_left_bnd_ys_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_right_bnd_xs_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_right_bnd_ys_, max_path * sizeof(float)));
        
        int threads = 256;
        int blocks = (K_ * T_ + threads - 1) / threads;
        init_rng_kernel<<<blocks, threads>>>((curandState *)d_rng_states_, 1234UL, K_, T_);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void MPPISolver::cleanup_cuda_memory() {
        cudaFree(d_states_); cudaFree(d_controls_); cudaFree(d_prev_controls_);
        cudaFree(d_costs_); cudaFree(d_rng_states_);
        cudaFree(d_ref_xs_); cudaFree(d_ref_ys_); cudaFree(d_ref_yaws_); cudaFree(d_ref_vs_);
        cudaFree(d_left_bnd_xs_); cudaFree(d_left_bnd_ys_);
        cudaFree(d_right_bnd_xs_); cudaFree(d_right_bnd_ys_);
    }

    void MPPISolver::update_params(Params p) { params_ = p; }

    void MPPISolver::set_reference_path(const std::vector<float> &xs, const std::vector<float> &ys,
                                        const std::vector<float> &yaws, const std::vector<float> &vs) {
        h_ref_xs_ = xs; h_ref_ys_ = ys; 
        ref_path_len_ = xs.size();
        if (ref_path_len_ > 10000) ref_path_len_ = 10000; // 메모리 제한 10000 적용
        if (ref_path_len_ > 0) {
            CUDA_CHECK(cudaMemcpy(d_ref_xs_, xs.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_ys_, ys.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_yaws_, yaws.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_vs_, vs.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    void MPPISolver::set_boundaries(const std::vector<float>& left_xs, const std::vector<float>& left_ys,
                                    const std::vector<float>& right_xs, const std::vector<float>& right_ys) {
        bnd_len_ = left_xs.size();
        if (bnd_len_ > 10000) bnd_len_ = 10000; // 메모리 제한 10000 적용
        if (bnd_len_ > 0) {
            CUDA_CHECK(cudaMemcpy(d_left_bnd_xs_, left_xs.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_left_bnd_ys_, left_ys.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_right_bnd_xs_, right_xs.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_right_bnd_ys_, right_ys.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    Control MPPISolver::solve(const State &current_state) {
        CUDA_CHECK(cudaMemcpy(d_prev_controls_, h_prev_controls_.data(), T_ * sizeof(Control), cudaMemcpyHostToDevice));

        int start_path_idx = 0;
        if (!h_ref_xs_.empty()) {
            float min_dist_sq = 1e9f;
            for (int i = 0; i < (int)h_ref_xs_.size(); ++i) {
                float dx = current_state.x - h_ref_xs_[i];
                float dy = current_state.y - h_ref_ys_[i];
                float d_sq = dx*dx + dy*dy;
                if (d_sq < min_dist_sq) { min_dist_sq = d_sq; start_path_idx = i; }
            }
        }

        int threadsPerBlock = 128;
        int blocksPerGrid = (K_ + threadsPerBlock - 1) / threadsPerBlock;

        rollout_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_states_, d_controls_, d_costs_, (curandState *)d_rng_states_,
            current_state, d_prev_controls_, params_,
            d_ref_xs_, d_ref_ys_, d_ref_yaws_, d_ref_vs_, ref_path_len_,
            d_left_bnd_xs_, d_left_bnd_ys_, d_right_bnd_xs_, d_right_bnd_ys_, bnd_len_, 
            K_, T_, start_path_idx);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_costs_.data(), d_costs_, K_ * sizeof(float), cudaMemcpyDeviceToHost));
        
        if (params_.visualize_candidates) {
            CUDA_CHECK(cudaMemcpy(h_states_.data(), d_states_, K_ * T_ * sizeof(State), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaMemcpy(h_controls_.data(), d_controls_, K_ * T_ * sizeof(Control), cudaMemcpyDeviceToHost));

        return compute_optimal_control();
    }

    Control MPPISolver::compute_optimal_control() {
        auto min_it = std::min_element(h_costs_.begin(), h_costs_.end());
        float min_cost = *min_it;
        best_k_ = static_cast<int>(std::distance(h_costs_.begin(), min_it));

        // 모든 샘플이 제약 조건 위반 시 정지 제어 반환
        if (std::isinf(min_cost) || min_cost >= 1.0e8f) { 
            Control stop_control = {0.0f, -5.0f};
            std::fill(h_prev_controls_.begin(), h_prev_controls_.end(), stop_control);
            return stop_control;
        }

        float lambda = params_.lambda; 
        float sum_weights = 0.0f;
        // 유효한 샘플(제약 조건을 만족하는 샘플)만 가중치 계산에 포함
        for (int k = 0; k < K_; ++k) {
            if (std::isinf(h_costs_[k]) || h_costs_[k] >= 1.0e8f) {
                h_weights_[k] = 0.0f;  // 무효 샘플은 가중치 0
            } else {
                h_weights_[k] = expf(-(h_costs_[k] - min_cost) / lambda);
            }
            sum_weights += h_weights_[k];
        }
        if (sum_weights < 1e-6) sum_weights = 1e-6;

        std::vector<Control> weighted_controls(T_, {0.0f, 0.0f});
        for (int k = 0; k < K_; ++k) {
            float w = h_weights_[k] / sum_weights;
            for (int t = 0; t < T_; ++t) {
                Control u_k = h_controls_[k * T_ + t];
                weighted_controls[t].steer += w * u_k.steer;
                weighted_controls[t].accel += w * u_k.accel;
            }
        }

        optimal_controls_ = weighted_controls; 
        Control output = weighted_controls[0];

        for (int t = 0; t < T_ - 1; ++t) h_prev_controls_[t] = weighted_controls[t + 1];
        h_prev_controls_[T_ - 1] = weighted_controls[T_ - 1];

        if (params_.visualize_candidates) {
            best_trajectory_.resize(T_);
            int base = best_k_ * T_;
            for (int t = 0; t < T_; ++t) best_trajectory_[t] = h_states_[base + t];
        }
        return output;
    }
    const std::vector<State> &MPPISolver::get_generated_trajectories() const { return h_states_; }
    const std::vector<State> &MPPISolver::get_best_trajectory() const { return best_trajectory_; }
    int MPPISolver::get_best_k() const { return best_k_; }
    const std::vector<Control>& MPPISolver::get_optimal_controls() const { return optimal_controls_; }
    int MPPISolver::get_K() const { return K_; }
    int MPPISolver::get_T() const { return T_; }
}