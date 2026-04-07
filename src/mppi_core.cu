#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace mppi
{
    __host__ __device__ inline float fast_cos(float x) {
    #ifdef __CUDA_ARCH__
        return __cosf(x); 
    #else
        return cosf(x); 
    #endif
    }

    __host__ __device__ inline float fast_sin(float x) {
    #ifdef __CUDA_ARCH__
        return __sinf(x);
    #else
        return sinf(x);
    #endif
    }

    __host__ __device__ inline float fast_exp(float x) {
    #ifdef __CUDA_ARCH__
        return __expf(x);
    #else
        return expf(x);
    #endif
    }

    __host__ __device__ float angle_normalize(float angle)
    {
        while (angle > M_PI) angle -= 2.0f * M_PI;
        while (angle < -M_PI) angle += 2.0f * M_PI;
        return angle;
    }

    __host__ __device__ State update_dynamics(const State &s, const Control &u, const Params &p)
    {
        float px = s.x; float py = s.y; float yaw = s.yaw;
        float vx = s.v; float vy = s.vy; float omega = s.omega;
        
        if (vx < 0.5f) {
            State next_s;
            float beta = atanf(p.l_r * tanf(u.steer) / (p.l_f + p.l_r));
            next_s.x = px + vx * fast_cos(yaw + beta) * p.dt;
            next_s.y = py + vx * fast_sin(yaw + beta) * p.dt;
            next_s.yaw = angle_normalize(yaw + (vx / p.l_r) * fast_sin(beta) * p.dt);
            next_s.v = vx + u.accel * p.dt;
            next_s.vy = 0.0f; next_s.omega = 0.0f;
            return next_s;
        }

        float alpha_f = u.steer - atan2f(vy + p.l_f * omega, vx);
        float alpha_r = -atan2f(vy - p.l_r * omega, vx);

        float F_fy = p.D_f * fast_sin(p.C_f * atanf(p.B_f * alpha_f));
        float F_ry = p.D_r * fast_sin(p.C_r * atanf(p.B_r * alpha_r));

        float F_rx = p.mass * u.accel; 
        float dot_vx = (F_rx - F_fy * fast_sin(u.steer) + p.mass * vy * omega) / p.mass;
        float dot_vy = (F_ry + F_fy * fast_cos(u.steer) - p.mass * vx * omega) / p.mass;
        float dot_omega = (F_fy * p.l_f * fast_cos(u.steer) - F_ry * p.l_r) / p.I_z;

        float dot_x = vx * fast_cos(yaw) - vy * fast_sin(yaw);
        float dot_y = vx * fast_sin(yaw) + vy * fast_cos(yaw);

        State next_s;
        next_s.x = px + dot_x * p.dt;
        next_s.y = py + dot_y * p.dt;
        next_s.yaw = angle_normalize(yaw + omega * p.dt);
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

        // 2. 공격적 속도 보상 (무조건 빠르게 가도록 유도하여 거북이 주행 방지)
        float vel_cost = -p.q_v * (s.v * fast_cos(s.yaw - ref_yaws[nearest_idx]));

        // 3. 오버스피드 방지 패널티 (곡률 기반 한계 속도인 ref_vs를 넘었을 때만 브레이크 강제)
        float overspeed_cost = 0.0f;
        if (s.v > ref_vs[nearest_idx]) {
            float excess = s.v - ref_vs[nearest_idx] * 0.9f;
            overspeed_cost = p.q_v * 20.0f * (excess * excess); // 20.0f는 브레이킹 강도 튜닝 계수
        }

        // 4. Control Input Cost
        float d_steer = u.steer - u_prev.steer;
        float d_accel = u.accel - u_prev.accel;
        float steer_rate_cost = p.q_du * 2.0f * (d_steer * d_steer);
        float accel_rate_cost = p.q_du * fabsf(d_accel);
        float steer_cost = p.q_steer * (u.steer * u.steer);

        // 5. Lateral G / Slip Cost
        float lat_g_cost = 0.0f;
        float ay_abs = fabsf(s.ay);
        if (ay_abs >= 9.5f) {
            float excess = ay_abs - 9.5f;
            lat_g_cost = p.q_lat_g * (__expf(-3.0f * excess));
        }
        
        // 6. Boundary Collision Cost
        float boundary_cost = 0.0f;
        float safe_dist = p.collision_radius + 0.4f;

        if (min_bnd_dist < safe_dist) {
            float penetration = safe_dist - min_bnd_dist;
            float soft_cost = 1000.0f * (penetration * penetration); 

            float hard_cost = 0.0f;
            if (min_bnd_dist < p.collision_radius * 1.5f) {
                float diff = min_bnd_dist - p.collision_radius;
                float capped = fminf(diff, 1.0e-5f);
                hard_cost = p.q_collision * logf(1.0f + __expf(-40.0f * capped));
            }

            boundary_cost = soft_cost + hard_cost;
        }

        // 7. Obstacle Cost
        float obs_cost = 0.0f;
        for (int i = 0; i < p.num_obstacles; ++i) {
            float dx = s.x - p.obs_x[i];
            float dy = s.y - p.obs_y[i];
            float dist = sqrtf(dx * dx + dy * dy);
            float safe_margin = 1.5f; 
            if (dist < safe_margin) {
                obs_cost += p.q_obs / (dist - p.car_radius + 1e-3f); 
            }
        }

        return p.q_dist * dist_error + vel_cost + overspeed_cost + steer_rate_cost + accel_rate_cost + steer_cost + lat_g_cost + boundary_cost + obs_cost;
    }
    
    __device__ float compute_min_boundary_distance(
        const State &s,
        const float *left_xs, const float *left_ys,
        const float *right_xs, const float *right_ys,
        int bnd_len,
        int current_path_idx) 
    {
        if (left_xs == nullptr || right_xs == nullptr || bnd_len <= 0) return 1e9f;

        float min_dist_sq = 1e9f;
        
        int search_window = 30; 
        int start_search = current_path_idx - 5;
        
        if (start_search < 0) start_search += bnd_len; 

        for (int offset = 0; offset < search_window; ++offset)
        {
            int i = start_search + offset;
            if (i >= bnd_len) i -= bnd_len;

            float dx_l = s.x - __ldg(&left_xs[i]);
            float dy_l = s.y - __ldg(&left_ys[i]);
            float dist_sq_l = dx_l * dx_l + dy_l * dy_l;

            float dx_r = s.x - __ldg(&right_xs[i]);
            float dy_r = s.y - __ldg(&right_ys[i]);
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
        int initial_path_idx = start_path_idx; 
        bool is_fault = false;

        for (int t = 0; t < T; ++t)
        {
            int idx = k * T + t;

            Control u_mean_curr = prev_controls[t];
            Control u_mean_prev = (t == 0) ? prev_controls[0] : prev_controls[t-1];
            
            float mean_delta_steer = u_mean_curr.steer - u_mean_prev.steer;  
            float mean_delta_accel = u_mean_curr.accel - u_mean_prev.accel;  

            float noise_delta_steer = curand_normal(&rng_states[idx]) * p.noise_steer_std * p.dt;   
            float noise_delta_accel = curand_normal(&rng_states[idx]) * p.noise_accel_std * p.dt;   

            current_action.steer += fminf(fmaxf(mean_delta_steer + noise_delta_steer, -p.max_steer_rate * p.dt), p.max_steer_rate * p.dt);    
            current_action.accel += fminf(fmaxf(mean_delta_accel + noise_delta_accel, -p.max_accel_rate * p.dt), p.max_accel_rate * p.dt);

            Control u_clamped = current_action;
            u_clamped.steer = fminf(fmaxf(u_clamped.steer, -p.max_steer), p.max_steer);
            
            float v_next = x.v + u_clamped.accel * p.dt;
            if (v_next >= p.max_speed && u_clamped.accel > 0.0f) u_clamped.accel = 0.0;
            else if (v_next <= p.min_speed + 0.1f && u_clamped.accel < 0.0f) u_clamped.accel = 0.0;
            else u_clamped.accel = fminf(fmaxf(u_clamped.accel, p.min_accel), p.max_accel);

            current_action = u_clamped; 

            x = update_dynamics(x, u_clamped, p);
            states[idx] = x;
            controls[idx] = u_clamped; 

            if(fabsf(x.ay) > 9.8f){
                is_fault = true;
            }

            float min_dist = compute_min_boundary_distance(
                x, left_bnd_xs, left_bnd_ys, right_bnd_xs, right_bnd_ys, bnd_len, local_path_idx);
            
            if (min_dist < p.collision_radius) {
                is_fault = true;
            }

            // 🚨 핵심 수정부 1: 생존 비례 보상 (수학적 붕괴 방지)
            if (is_fault) {
                // 기본 패널티를 10000으로 낮추고, 오래 버틸수록 패널티를 50씩 대폭 깎아줍니다.
                // 이로 인해 어차피 박을 상황이면 풀브레이킹+조향으로 1틱이라도 더 버티는 샘플의 가중치가 높아집니다.
                total_cost += 10000.0f - (float)t * 50.0f; 
                
                // 충돌했더라도, 그때까지 더 멀리 전진했다면 보상을 줍니다.
                if (path_len > 0) {
                    int progress = local_path_idx - initial_path_idx;
                    if (progress < -path_len / 2) progress += path_len; 
                    int max_possible_progress = T + 10; 
                    progress = max(0, min(progress, max_possible_progress));
                    total_cost -= p.q_v * (float)progress * 5.0f; 
                }

                const Control zero_control = {0.0f, 0.0f};
                for (int fill_t = t + 1; fill_t < T; ++fill_t) {
                    states[k * T + fill_t] = x;
                    controls[k * T + fill_t] = zero_control;
                }
                break;
            }

            if (path_len > 0)
            {
                total_cost += compute_cost_cuda(
                    x, ref_xs, ref_ys, ref_yaws, ref_vs, path_len,
                    u_clamped, last_u, p, min_dist, &local_path_idx); 
            }

            // 🚨 핵심 수정부 2: 종점 진행도 강력 보상 (속도 향상 유도)
            if (t == T - 1 && path_len > 0) {
                int progress = local_path_idx - initial_path_idx;
                if (progress < -path_len / 2) progress += path_len; 
                int max_possible_progress = T + 10; 
                progress = max(0, min(progress, max_possible_progress));
                // 무사히 완주한 경우 진행 칸수에 비례해 엄청난 보상을 주어 공격적 가속을 유도
                total_cost -= p.q_v * (float)progress * 10.0f; 
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
        if (ref_path_len_ > 10000) ref_path_len_ = 10000; 
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
        if (bnd_len_ > 10000) bnd_len_ = 10000; 
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

        return compute_optimal_control(current_state);
    }

    Control MPPISolver::compute_optimal_control(const State &current_state) {
        auto min_it = std::min_element(h_costs_.begin(), h_costs_.end());
        float min_cost = *min_it;
        best_k_ = static_cast<int>(std::distance(h_costs_.begin(), min_it));

        if (std::isinf(min_cost) || min_cost >= 1.0e8f) { 
            Control stop_control = {0.0f, -5.0f};
            std::fill(h_prev_controls_.begin(), h_prev_controls_.end(), stop_control);
            return stop_control;
        }

        float lambda = params_.lambda; 
        float sum_weights = 0.0f;
        for (int k = 0; k < K_; ++k) {
            if (std::isinf(h_costs_[k])) {
                h_weights_[k] = 0.0f; 
            } else {
                h_weights_[k] = fast_exp(-(h_costs_[k] - min_cost) / lambda);
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

        best_trajectory_.resize(T_);
        State sim_state = current_state; 

        for (int t = 0; t < T_; ++t) {
            sim_state = update_dynamics(sim_state, weighted_controls[t], params_);
            best_trajectory_[t] = sim_state;
        }
        return output;
    }

    const std::vector<State> &MPPISolver::get_generated_trajectories() const { return h_states_; }
    const std::vector<State> &MPPISolver::get_best_trajectory() const { return best_trajectory_; }
    int MPPISolver::get_best_k() const { return best_k_; }
    const std::vector<Control>& MPPISolver::get_optimal_controls() const { return optimal_controls_; }
    const std::vector<float>& MPPISolver::get_costs() const { return h_costs_; }
    int MPPISolver::get_K() const { return K_; }
    int MPPISolver::get_T() const { return T_; }
}