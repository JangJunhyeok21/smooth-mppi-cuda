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

    // [핵심] Dynamic Bicycle Model (Pacejka) 구현
    __device__ State update_dynamics_cuda(const State &s, const Control &u, const Params &p)
    {
        float px = s.x;
        float py = s.y;
        float yaw = s.yaw;
        float vx = s.v;
        float vy = s.vy;
        float omega = s.omega;
        
        // 저속 주행 시 특이점 방지 (Kinematic 근사)
        if (vx < 0.1f) {
            State next_s;
            float beta = atanf(p.l_r * tanf(u.steer) / (p.l_f + p.l_r));
            next_s.x = px + vx * cosf(yaw + beta) * p.dt;
            next_s.y = py + vx * sinf(yaw + beta) * p.dt;
            next_s.yaw = angle_normalize_cuda(yaw + (vx / p.l_r) * sinf(beta) * p.dt);
            next_s.v = vx + u.accel * p.dt;
            next_s.vy = 0.0f;
            next_s.omega = 0.0f;
            return next_s;
        }

        // 1. Slip Angles (alpha)
        // alpha_f = delta - atan2(vy + lf*omega, vx)
        float alpha_f = u.steer - atan2f(vy + p.l_f * omega, vx);
        // alpha_r = - atan2(vy - lr*omega, vx)
        float alpha_r = -atan2f(vy - p.l_r * omega, vx);

        // 2. Lateral Forces (Pacejka Magic Formula)
        // F = D * sin(C * atan(B * alpha))
        float F_fy = p.D_f * sinf(p.C_f * atanf(p.B_f * alpha_f));
        float F_ry = p.D_r * sinf(p.C_r * atanf(p.B_r * alpha_r));

        // 3. Equations of Motion
        // F_rx는 질량 * 가속도 명령으로 가정
        float F_rx = p.mass * u.accel; 

        // dot_vx = (F_rx - F_fy * sin(delta) + m * vy * omega) / m
        float dot_vx = (F_rx - F_fy * sinf(u.steer) + p.mass * vy * omega) / p.mass;

        // dot_vy = (F_ry + F_fy * cos(delta) - m * vx * omega) / m
        float dot_vy = (F_ry + F_fy * cosf(u.steer) - p.mass * vx * omega) / p.mass;

        // dot_omega = (F_fy * lf * cos(delta) - F_ry * lr) / Iz
        float dot_omega = (F_fy * p.l_f * cosf(u.steer) - F_ry * p.l_r) / p.I_z;

        // 4. Integration (Euler)
        // Global Frame Update
        float dot_x = vx * cosf(yaw) - vy * sinf(yaw);
        float dot_y = vx * sinf(yaw) + vy * cosf(yaw);

        State next_s;
        next_s.x = px + dot_x * p.dt;
        next_s.y = py + dot_y * p.dt;
        next_s.yaw = angle_normalize_cuda(yaw + omega * p.dt);
        next_s.v = vx + dot_vx * p.dt;
        next_s.vy = vy + dot_vy * p.dt;
        next_s.omega = omega + dot_omega * p.dt;

        return next_s;
    }

    __device__ float compute_cost_cuda(
        const State &s,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws, const float *ref_vs, int path_len,
        const Control &u, const Control &u_prev,
        const Params &p,
        float min_obstacle_dist)  // 외부에서 계산된 최소 장애물 거리
    {
        float min_dist_sq = 1e9f;
        int nearest_idx = -1;

        for (int i = 0; i < path_len; ++i)
        {
            float dx = s.x - ref_xs[i];
            float dy = s.y - ref_ys[i];
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq > 100.0f) continue; // 너무 먼 점 무시

            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }

        if (nearest_idx == -1) return 1.0e9f;

        float dist_error = min_dist_sq;
        float ref_v = ref_vs[nearest_idx];
        float v_error = (s.v - ref_v) * (s.v - ref_v);

        float path_yaw = ref_yaws[nearest_idx];
        float yaw_diff = angle_normalize_cuda(s.yaw - path_yaw);
        
        float heading_cost = 0.0f;
        if (abs(yaw_diff) > 1.047f) { //60도
            heading_cost = 1000.0f;
        } else {
            heading_cost = p.q_heading * (yaw_diff * yaw_diff);
        }

        float input_cost = p.q_u * (u.steer * u.steer + u.accel * u.accel);

        // a_t - a_{t-1}
        float d_steer = u.steer - u_prev.steer;
        float d_accel = u.accel - u_prev.accel;
        float rate_cost = p.q_du * (d_steer * d_steer + d_accel * d_accel);

        // 횡G 비용
        float current_lat_g = s.v * s.omega; 
        float g_limit = 9.8f; 
        float safe_ratio = 0.85f;
        float lat_cost = 0.0f;
        
        if (abs(current_lat_g) > g_limit * safe_ratio) {
            float violation = abs(current_lat_g) - (g_limit * safe_ratio);
            lat_cost = p.q_lat * (violation * violation); 
        }

        // 벽 근접 비용 (기하급수적 증가)
        float obstacle_cost = 0.0f;
        float danger_threshold = p.collision_radius * 1.5f;
        if (min_obstacle_dist < danger_threshold) {
            float scale = p.collision_radius * 0.5f;
            float proximity = (danger_threshold - min_obstacle_dist) / scale;
            obstacle_cost = fminf(p.q_collision * expf(proximity), 1.0e8f);
        }
        
        return p.q_dist * dist_error + p.q_v * v_error + heading_cost + input_cost + rate_cost + lat_cost + obstacle_cost;
    }

    // 장애물까지의 최소 거리를 반환 (없으면 1e9f)
    __device__ float compute_min_obstacle_distance(
        const State &s,
        const float *scan_ranges, int scan_len,
        float scan_angle_min, float scan_angle_inc,
        const State &robot_pose)
    {
        if (scan_ranges == nullptr || scan_len <= 0) return 1e9f;

        float dx = s.x - robot_pose.x;
        float dy = s.y - robot_pose.y;
        float local_x = dx * cosf(robot_pose.yaw) + dy * sinf(robot_pose.yaw);
        float local_y = -dx * sinf(robot_pose.yaw) + dy * cosf(robot_pose.yaw);

        float min_dist = 1e9f;
        for (int i = 0; i < scan_len; ++i)
        {
            float r = scan_ranges[i];
            if (isinf(r) || isnan(r)) continue;

            float ang = scan_angle_min + i * scan_angle_inc;
            float obs_x = r * cosf(ang);
            float obs_y = r * sinf(ang);

            float dx_obs = local_x - obs_x;
            float dy_obs = local_y - obs_y;
            float dist = sqrtf(dx_obs * dx_obs + dy_obs * dy_obs);

            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        return min_dist;
    }

    __global__ void init_rng_kernel(curandState *states, long seed, int K, int T)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < K * T) {
            curand_init(seed, idx, 0, &states[idx]);
        }
    }

    __global__ void rollout_kernel(
        State *states,
        Control *controls,
        float *costs,
        curandState *rng_states,
        const State start_state,
        const Control *prev_controls,
        const Params p,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws, const float *ref_vs, int path_len,
        const float *scan_ranges, int scan_len, float scan_angle_min, float scan_angle_inc,
        int K, int T)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= K) return;

        State x = start_state;
        float total_cost = 0.0f;
        
        Control current_action = prev_controls[0]; 
        Control last_u = current_action;

        for (int t = 0; t < T; ++t)
        {
            int idx = k * T + t;

            Control u_mean_curr = prev_controls[t];
            Control u_mean_prev = (t == 0) ? prev_controls[0] : prev_controls[t-1];
            
            float mean_steer_rate = u_mean_curr.steer - u_mean_prev.steer;
            float mean_accel_rate = u_mean_curr.accel - u_mean_prev.accel;

            float noise_steer_rate = curand_normal(&rng_states[idx]) * p.noise_steer_std; 
            float noise_accel_rate = curand_normal(&rng_states[idx]) * p.noise_accel_std;

            current_action.steer += (mean_steer_rate + fminf(fmaxf(noise_steer_rate, -p.max_steer_rate), p.max_steer_rate));
            current_action.accel += (mean_accel_rate + fminf(fmaxf(noise_accel_rate, -p.max_accel_rate), p.max_accel_rate));

            Control u_clamped = current_action;
            u_clamped.steer = fminf(fmaxf(u_clamped.steer, -p.max_steer), p.max_steer);
            u_clamped.accel = fminf(fmaxf(u_clamped.accel, p.min_accel), p.max_accel);

            float v_next = x.v + u_clamped.accel * p.dt;
            if (v_next > p.max_speed) u_clamped.accel = (p.max_speed - x.v) / p.dt;
            else if (v_next < p.min_speed) u_clamped.accel = (p.min_speed - x.v) / p.dt;

            current_action = u_clamped; 

            x = update_dynamics_cuda(x, u_clamped, p);
            states[idx] = x;
            controls[idx] = u_clamped; 

            // 최소 장애물 거리 계산 (한 번만)
            float min_dist = compute_min_obstacle_distance(
                x, scan_ranges, scan_len, scan_angle_min, scan_angle_inc, start_state);
            
            // Hard constraint: collision_radius보다 가까우면 즉시 큰 비용
            if (min_dist < p.collision_radius)
            {
                total_cost = 1.0e9f;
                break;
            }

            if (path_len > 0)
            {
                // Soft cost: min_dist를 cost 계산에 전달
                total_cost += compute_cost_cuda(
                    x, ref_xs, ref_ys, ref_yaws, ref_vs, path_len,
                    u_clamped, last_u, p, min_dist);
            }
            last_u = u_clamped;
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

    MPPISolver::~MPPISolver() { cleanup_cuda_memory(); }

    void MPPISolver::allocate_cuda_memory()
    {
        CUDA_CHECK(cudaMalloc(&d_states_, K_ * T_ * sizeof(State)));
        CUDA_CHECK(cudaMalloc(&d_controls_, K_ * T_ * sizeof(Control)));
        CUDA_CHECK(cudaMalloc(&d_prev_controls_, T_ * sizeof(Control)));
        CUDA_CHECK(cudaMalloc(&d_costs_, K_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_states_, K_ * T_ * sizeof(curandState)));

        int max_path = 2000;
        CUDA_CHECK(cudaMalloc(&d_ref_xs_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_ys_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_yaws_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_vs_, max_path * sizeof(float)));

        d_scan_ranges_ = nullptr;

        int threads = 256;
        int blocks = (K_ * T_ + threads - 1) / threads;
        init_rng_kernel<<<blocks, threads>>>((curandState *)d_rng_states_, 1234UL, K_, T_);
        CUDA_CHECK(cudaDeviceSynchronize());
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
        cudaFree(d_ref_yaws_);
        cudaFree(d_ref_vs_);
        if (d_scan_ranges_ != nullptr) cudaFree(d_scan_ranges_);
    }

    void MPPISolver::update_params(Params p) { params_ = p; }

    void MPPISolver::set_reference_path(const std::vector<float> &xs, const std::vector<float> &ys,
                                        const std::vector<float> &yaws, const std::vector<float> &vs)
    {
        ref_path_len_ = xs.size();
        if (ref_path_len_ > 2000) ref_path_len_ = 2000;

        if (ref_path_len_ > 0)
        {
            CUDA_CHECK(cudaMemcpy(d_ref_xs_, xs.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_ys_, ys.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_yaws_, yaws.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_vs_, vs.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    void MPPISolver::set_scan_data(const std::vector<float> &ranges, float angle_min, float angle_inc)
    {
        scan_angle_min_ = angle_min;
        scan_angle_inc_ = angle_inc;

        int new_len = static_cast<int>(ranges.size());
        if (new_len <= 0) { scan_len_ = 0; return; }

        if (new_len != scan_len_)
        {
            if (d_scan_ranges_ != nullptr) cudaFree(d_scan_ranges_);
            cudaError_t err = cudaMalloc(&d_scan_ranges_, new_len * sizeof(float));
            if (err != cudaSuccess) { scan_len_ = 0; return; }
            scan_len_ = new_len;
        }
        CUDA_CHECK(cudaMemcpy(d_scan_ranges_, ranges.data(), scan_len_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    Control MPPISolver::solve(const State &current_state)
    {
        CUDA_CHECK(cudaMemcpy(d_prev_controls_, h_prev_controls_.data(), T_ * sizeof(Control), cudaMemcpyHostToDevice));

        int threadsPerBlock = 128;
        int blocksPerGrid = (K_ + threadsPerBlock - 1) / threadsPerBlock;

        rollout_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_states_, d_controls_, d_costs_, (curandState *)d_rng_states_,
            current_state, d_prev_controls_, params_,
            d_ref_xs_, d_ref_ys_, d_ref_yaws_, d_ref_vs_, ref_path_len_,
            d_scan_ranges_, scan_len_, scan_angle_min_, scan_angle_inc_,
            K_, T_);
        
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_costs_.data(), d_costs_, K_ * sizeof(float), cudaMemcpyDeviceToHost));
        
        if (params_.visualize_candidates) {
            CUDA_CHECK(cudaMemcpy(h_states_.data(), d_states_, K_ * T_ * sizeof(State), cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaMemcpy(h_controls_.data(), d_controls_, K_ * T_ * sizeof(Control), cudaMemcpyDeviceToHost));

        return compute_optimal_control();
    }

    Control MPPISolver::compute_optimal_control()
    {
        auto min_it = std::min_element(h_costs_.begin(), h_costs_.end());
        float min_cost = *min_it;
        best_k_ = static_cast<int>(std::distance(h_costs_.begin(), min_it));

        if (min_cost >= 1.0e8f) {
            Control stop_control;
            stop_control.steer = 0.0f;
            stop_control.accel = params_.min_accel/2.0f;
            for (int t = 0; t < T_; ++t) {
                h_prev_controls_[t].steer = 0.0f;
                h_prev_controls_[t].accel = params_.min_accel/2.0f;
            }
            return stop_control;
        }

        float lambda = params_.lambda; 
        float sum_weights = 0.0f;

        for (int k = 0; k < K_; ++k) {
            h_weights_[k] = expf(-(h_costs_[k] - min_cost) / lambda);
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

        // [추가] 계산된 최적 제어 입력을 멤버 변수에 저장 (디버깅용)
        optimal_controls_ = weighted_controls;

        Control output = weighted_controls[0];

        for (int t = 0; t < T_ - 1; ++t) {
            h_prev_controls_[t] = weighted_controls[t + 1];
        }
        h_prev_controls_[T_ - 1] = weighted_controls[T_ - 1];

        if (params_.visualize_candidates) {
            best_trajectory_.resize(T_);
            int base = best_k_ * T_;
            for (int t = 0; t < T_; ++t) {
                best_trajectory_[t] = h_states_[base + t];
            }
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