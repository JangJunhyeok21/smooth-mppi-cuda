#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace mppi
{
    __host__ __device__ inline float fast_cos(float x)
    {
#ifdef __CUDA_ARCH__
        return __cosf(x);
#else
        return cosf(x);
#endif
    }

    __host__ __device__ inline float fast_sin(float x)
    {
#ifdef __CUDA_ARCH__
        return __sinf(x);
#else
        return sinf(x);
#endif
    }

    __host__ __device__ inline float fast_exp(float x)
    {
#ifdef __CUDA_ARCH__
        return __expf(x);
#else
        return expf(x);
#endif
    }

    __host__ __device__ float angle_normalize(float angle)
    {
        while (angle > M_PI)
            angle -= 2.0f * M_PI;
        while (angle < -M_PI)
            angle += 2.0f * M_PI;
        return angle;
    }

    __host__ __device__ State update_dynamics(const State &s, const Control &u, const Params &p)
    {
        float px = s.x;
        float py = s.y;
        float yaw = s.yaw;
        float vel = s.v;
        float omega = s.omega;
        float slip_angle = s.slip_angle;

        // 1. 저속 구간: 순수 운동학 모델 (Kinematic Model)
        // 분모에 vel이 들어가는 동역학의 특이점(Division by zero)을 방지
        if (fabsf(vel) < 0.5f)
        {
            State next_s;
            float wheelbase = p.l_f + p.l_r;

            // 기존 MPC Kinematic 수식 적용
            float dot_x = vel * fast_cos(yaw);
            float dot_y = vel * fast_sin(yaw);
            float dot_yaw = vel * tanf(u.steer) / wheelbase;
            float dot_vel = u.accel;

            next_s.x = px + dot_x * p.dt;
            next_s.y = py + dot_y * p.dt;
            next_s.yaw = angle_normalize(yaw + dot_yaw * p.dt);
            next_s.v = vel + dot_vel * p.dt;

            // 기존 MPC 기준 미사용 변수 초기화 (omega는 제어 연속성을 위해 dot_yaw 인가)
            next_s.omega = dot_yaw;
            next_s.slip_angle = 0.0f;

            // MPPI 비용 함수용 보조 변수
            next_s.vy = 0.0f;
            next_s.ay = 0.0f;

            return next_s;
        }

        // 2. 고속 구간: 파세이카 동역학 모델 (Pacejka Dynamic Model)

        // 슬립각(beta)으로부터 차량 좌표계 vx, vy 역산 (타이어 슬립각 alpha 연산용)
        float vx = vel * fast_cos(slip_angle);
        float vy = vel * fast_sin(slip_angle);

        // 전/후륜 타이어 슬립각 계산
        float alpha_f = u.steer - atan2f(vy + p.l_f * omega, vx);
        float alpha_r = -atan2f(vy - p.l_r * omega, vx);

        // 타이어 횡력 연산 (Newton 단위 - Df, Dr 파라미터가 40.0 같은 힘의 크기일 때 사용)
        float F_fy = p.D_f * fast_sin(p.C_f * atanf(p.B_f * alpha_f));
        float F_ry = p.D_r * fast_sin(p.C_r * atanf(p.B_r * alpha_r));

        // 기존 MPC 동역학 수식 완벽 적용 (단위 및 로직 동일)
        float dot_x = vel * fast_cos(yaw + slip_angle);
        float dot_y = vel * fast_sin(yaw + slip_angle);
        float dot_yaw = omega;
        float dot_vel = u.accel * (1.0f - p.Cm0 * vel);                              // 모터 감쇠 계수 적용
        float dot_omega = (p.l_f * F_fy * fast_cos(u.steer) - p.l_r * F_ry) / p.I_z; // 토크 / 관성모멘트
        float dot_slip = ((F_fy + F_ry) / (p.mass * vel)) - omega;                   // 횡력의 합 / (질량 * 속도)

        // 상태 업데이트 (Euler Integration)
        State next_s;
        next_s.x = px + dot_x * p.dt;
        next_s.y = py + dot_y * p.dt;
        next_s.yaw = angle_normalize(yaw + dot_yaw * p.dt);
        next_s.v = vel + dot_vel * p.dt;
        next_s.omega = omega + dot_omega * p.dt;
        next_s.slip_angle = slip_angle + dot_slip * p.dt;

        // MPPI 비용 함수에서 사용하는 보조 변수 도출
        next_s.vy = next_s.v * fast_sin(next_s.slip_angle);     // 슬립각 기반 vy
        next_s.ay = (F_fy * fast_cos(u.steer) + F_ry) / p.mass; // 횡가속도 a_y = F_y / m

        return next_s;
    }

    __device__ float compute_cost_cuda(
        const State &s,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws, const float *ref_vs, int path_len,
        const Control &u, const Control &u_prev,
        const Params &p,
        float min_bnd_dist,
        int *last_idx,
        int t)
    {
        float min_dist_sq = 1e9f;
        int nearest_idx = -1;

        int start_search = *last_idx;
        int search_window = 30;

        if (start_search >= path_len)
            start_search %= path_len;
        if (start_search < 0)
            start_search = 0;

        for (int offset = 0; offset < search_window; ++offset)
        {
            int i = start_search + offset;
            if (i >= path_len)
                i -= path_len;

            float dx = s.x - ref_xs[i];
            float dy = s.y - ref_ys[i];
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }

        if (nearest_idx == -1)
            nearest_idx = start_search;
        *last_idx = nearest_idx;

        // 1. Reference Tracking Cost
        float dist_error = min_dist_sq;

        // 2. 속도 보상
        float vel_cost = -p.q_v * (s.v * fast_cos(s.yaw - ref_yaws[nearest_idx])); // 전체적인 직진성 유도

        // 4. Control Input Cost
        float d_steer = u.steer - u_prev.steer;
        float d_accel = u.accel - u_prev.accel;
        float rate_cost = p.q_du * (d_steer * d_steer + d_accel * d_accel);
        float steer_cost = p.q_steer * (u.steer * u.steer);

        // 6. Boundary Collision Cost
        float boundary_cost = 0.0f;
        float safe_dist = p.collision_radius + 0.1f;

        if (min_bnd_dist < safe_dist)
        {
            float penetration = safe_dist - min_bnd_dist;
            float soft_cost = 50.0f * (penetration * penetration);

            float hard_cost = 0.0f;
            if (min_bnd_dist < p.collision_radius * 1.5f)
            {
                float diff = min_bnd_dist - p.collision_radius;
                float capped = fminf(diff, 1.0e-5f);
                hard_cost = p.q_collision * logf(1.0f + __expf(-30.0f * capped));
            }

            boundary_cost = soft_cost + hard_cost;
        }

        // 7. Obstacle Cost (정적 벽 취급 + 동적 예측 및 ACC)
        float obs_cost = 0.0f;

        for (int i = 0; i < p.num_obstacles; ++i)
        {
            if (!p.obstacles[i].is_dynamic)
            {
                // ----------------------------------------------------
                // [1] 정적 장애물: 벽과 동일하게 취급
                // ----------------------------------------------------
                float dx = s.x - p.obstacles[i].x;
                float dy = s.y - p.obstacles[i].y;
                float dist = sqrtf(dx * dx + dy * dy);
                // 범퍼 간 거리 (두 원의 중심 거리 - 각자의 반지름)
                float bumper_dist = dist - p.car_radius - p.obstacles[i].radius;

                if (bumper_dist < 0.15f)
                { // 15cm 이내 위험 구역
                    float penetration = 0.15f - bumper_dist;
                    obs_cost += 300.0f * (penetration * penetration); // 거리에 따른 Soft Cost

                    if (bumper_dist < 0.0f)
                    {
                        obs_cost += 10000.0f; // 충돌 시 데스 패널티 (벽과 동일 취급)
                    }
                }
            }
            else
            {
                // ----------------------------------------------------
                // [2] 동적 장애물: 예측 및 추월/ACC 처리
                // ----------------------------------------------------
                // (1) 등속도 모델(CV) 기반 시간 t초 후의 장애물 예상 위치 역산
                float pred_dt = (float)t * p.dt;
                float pred_x = p.obstacles[i].x + p.obstacles[i].v * fast_cos(p.obstacles[i].yaw) * pred_dt;
                float pred_y = p.obstacles[i].y + p.obstacles[i].v * fast_sin(p.obstacles[i].yaw) * pred_dt;

                float dx = s.x - pred_x;
                float dy = s.y - pred_y;
                float dist = sqrtf(dx * dx + dy * dy);
                float bumper_dist = dist - p.car_radius - p.obstacles[i].radius;

                float target_gap = 0.3f; // 30cm 유지 목표

                // (2) 충돌 회피 (우회 공간이 있으면 자연스럽게 추월 궤적이 선택됨)
                if (bumper_dist <= target_gap)
                {
                    // 30cm 이내 진입 시 강력한 충돌 패널티.
                    // 양옆이 벽으로 막혀있다면, 속도를 줄여서 이 영역에 안 들어가는 샘플만 살아남음.
                    obs_cost += 10000.0f;
                }
                // (3) ACC 구간: 30cm ~ 1.0m 사이
                else if (bumper_dist < target_gap + 0.7f)
                {
                    // 논리 오류 방지: 내가 상대방을 추월하기 위해 옆 차선에 있거나 이미 앞서가고 있다면 감속하면 안 됨.
                    // 내적(Dot Product)을 통해 내 차량이 상대 차량의 '뒤'에 있는지 판단.
                    // 상대 차량의 직진 벡터와, 상대차량 -> 내 차량으로 향하는 벡터의 내적
                    float dot = dx * fast_cos(p.obstacles[i].yaw) + dy * fast_sin(p.obstacles[i].yaw);

                    if (dot < 0.0f)
                    { // 내적이 음수면 내 차량이 상대의 후방에 위치함을 의미
                        float speed_diff = s.v - p.obstacles[i].v;

                        if (speed_diff > 0.0f)
                        {
                            // 내가 더 빨라서 거리가 좁혀지고 있을 때만 패널티 부여
                            // 목표 거리(30cm)에 가까워질수록 패널티를 증폭시켜 부드러운 제동 유도
                            obs_cost += p.q_acc * (speed_diff * speed_diff) / (bumper_dist - target_gap + 1e-4f);
                        }
                    }
                }
            }
        }

        return p.q_dist * dist_error + vel_cost + steer_cost + rate_cost + boundary_cost + obs_cost;
    }

    // [수정된 함수] O(N) 바운더리 탐색을 대체하는 O(1) 횡방향 오차 기반 거리 연산
    __device__ float compute_min_boundary_distance(
        const State &s,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws,
        const float *left_xs, const float *left_ys,
        const float *right_xs, const float *right_ys,
        int path_len, int current_path_idx, int *nearest_idx_out)
    {
        if (ref_xs == nullptr || left_xs == nullptr || path_len <= 0)
            return 1e9f;

        // 1. 이전 인덱스 근처에서 가장 가까운 중심점(Reference) 탐색
        float min_dist_sq = 1e9f;
        int nearest_idx = current_path_idx;
        int search_window = 30;
        int start_search = current_path_idx;

        if (start_search >= path_len)
            start_search %= path_len;
        if (start_search < 0)
            start_search = 0;

        for (int offset = 0; offset < search_window; ++offset)
        {
            int i = start_search + offset;
            if (i >= path_len)
                i -= path_len;

            float dx = s.x - ref_xs[i];
            float dy = s.y - ref_ys[i];
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }

        // 찾은 인덱스를 외부로 반환하여 compute_cost_cuda의 탐색 속도도 높임
        if (nearest_idx_out != nullptr)
            *nearest_idx_out = nearest_idx;

        // 2. 중심선 기준 법선 벡터를 통한 횡방향 편차(e_y) 도출
        float dx = s.x - ref_xs[nearest_idx];
        float dy = s.y - ref_ys[nearest_idx];
        float ref_yaw = ref_yaws[nearest_idx];

        float nx = -fast_sin(ref_yaw);
        float ny = fast_cos(ref_yaw);
        float e_y = dx * nx + dy * ny;

        // 3. 해당 지점의 실제 좌우 도로 폭 연산
        float dx_l = left_xs[nearest_idx] - ref_xs[nearest_idx];
        float dy_l = left_ys[nearest_idx] - ref_ys[nearest_idx];
        float w_left = sqrtf(dx_l * dx_l + dy_l * dy_l);

        float dx_r = right_xs[nearest_idx] - ref_xs[nearest_idx];
        float dy_r = right_ys[nearest_idx] - ref_ys[nearest_idx];
        float w_right = sqrtf(dx_r * dx_r + dy_r * dy_r);

        // 4. 차량에서 양쪽 바운더리까지의 최단 거리 반환
        return fminf(w_left - e_y, w_right + e_y);
    }

    __global__ void init_rng_kernel(curandState *states, long seed, int K, int T)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < K * T)
            curand_init(seed, idx, 0, &states[idx]);
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
        if (k >= K)
            return;

        State x = start_state;
        float total_cost = 0.0f;
        Control current_action = prev_controls[0];
        Control last_u = current_action;
        int local_path_idx = start_path_idx;
        int initial_path_idx = start_path_idx;
        bool is_fault = false;

        float x_steer_prev1 = 0.0f, x_steer_prev2 = 0.0f;
        float y_steer_prev1 = 0.0f, y_steer_prev2 = 0.0f;
        float x_accel_prev1 = 0.0f, x_accel_prev2 = 0.0f;
        float y_accel_prev1 = 0.0f, y_accel_prev2 = 0.0f;

        for (int t = 0; t < T; ++t)
        {
            int idx = k * T + t;

            Control u_mean_curr = prev_controls[t];
            Control u_mean_prev = (t == 0) ? prev_controls[0] : prev_controls[t - 1];

            float mean_delta_steer = u_mean_curr.steer - u_mean_prev.steer;
            float mean_delta_accel = u_mean_curr.accel - u_mean_prev.accel;

            // 1. Raw 백색 잡음 생성 (dt를 곱하지 않음)
            float raw_steer_noise = curand_normal(&rng_states[idx]) * p.noise_steer_std;
            float raw_accel_noise = curand_normal(&rng_states[idx]) * p.noise_accel_std;

            // 2. 2차 버터워스 필터링 (IIR 차분 방정식)
            float filtered_steer = p.filter_coeffs.b0 * raw_steer_noise + p.filter_coeffs.b1 * x_steer_prev1 + p.filter_coeffs.b2 * x_steer_prev2 - p.filter_coeffs.a1 * y_steer_prev1 - p.filter_coeffs.a2 * y_steer_prev2;

            float filtered_accel = p.filter_coeffs.b0 * raw_accel_noise + p.filter_coeffs.b1 * x_accel_prev1 + p.filter_coeffs.b2 * x_accel_prev2 - p.filter_coeffs.a1 * y_accel_prev1 - p.filter_coeffs.a2 * y_accel_prev2;

            // 3. 레지스터 상태 한 칸씩 시프트 (다음 루프를 위함)
            x_steer_prev2 = x_steer_prev1;
            x_steer_prev1 = raw_steer_noise;
            y_steer_prev2 = y_steer_prev1;
            y_steer_prev1 = filtered_steer;

            x_accel_prev2 = x_accel_prev1;
            x_accel_prev1 = raw_accel_noise;
            y_accel_prev2 = y_accel_prev1;
            y_accel_prev1 = filtered_accel;

            // 4. 부드러워진 노이즈에 dt를 곱해 최종 변화량 산출
            float noise_delta_steer = filtered_steer * p.dt;
            float noise_delta_accel = filtered_accel * p.dt;

            current_action.steer += fminf(fmaxf(mean_delta_steer + noise_delta_steer, -p.max_steer_rate * p.dt), p.max_steer_rate * p.dt);
            current_action.accel += fminf(fmaxf(mean_delta_accel + noise_delta_accel, -p.max_accel_rate * p.dt), p.max_accel_rate * p.dt);

            Control u_clamped = current_action;
            u_clamped.steer = fminf(fmaxf(u_clamped.steer, -p.max_steer), p.max_steer);

            float v_next = x.v + u_clamped.accel * p.dt;
            if (v_next >= p.max_speed && u_clamped.accel > 0.0f)
                u_clamped.accel = 0.0;
            else if (v_next <= p.min_speed + 0.1f && u_clamped.accel < 0.0f)
                u_clamped.accel = 0.0;
            else
                u_clamped.accel = fminf(fmaxf(u_clamped.accel, p.min_accel), p.max_accel);

            current_action = u_clamped;

            x = update_dynamics(x, u_clamped, p);
            states[idx] = x;
            controls[idx] = u_clamped;

            if (fabsf(x.ay) > 12.74f)
            { // 1.3g
                is_fault = true;
            }

            float min_dist = compute_min_boundary_distance(
                x, ref_xs, ref_ys, ref_yaws, left_bnd_xs, left_bnd_ys, right_bnd_xs, right_bnd_ys, path_len, local_path_idx, &local_path_idx);

            if (min_dist < p.collision_radius)
            {
                is_fault = true;
            }

            if (is_fault)
            {
                // 기본 패널티를 10000으로 낮추고, 오래 버틸수록 패널티를 50씩 대폭 깎아줍니다.
                // 이로 인해 어차피 박을 상황이면 풀브레이킹+조향으로 1틱이라도 더 버티는 샘플의 가중치가 높아집니다.
                total_cost += 10000.0f - (float)t * 50.0f;

                // 충돌했더라도, 그때까지 더 멀리 전진했다면 보상을 줍니다.
                if (path_len > 0)
                {
                    int progress = local_path_idx - initial_path_idx;
                    if (progress < -path_len / 2)
                        progress += path_len;
                    int max_possible_progress = T + 10;
                    progress = max(0, min(progress, max_possible_progress));
                    total_cost -= p.q_progress * (float)progress;
                }

                const Control zero_control = {0.0f, 0.0f};
                for (int fill_t = t + 1; fill_t < T; ++fill_t)
                {
                    states[k * T + fill_t] = x;
                    controls[k * T + fill_t] = zero_control;
                }
                break;
            }

            if (path_len > 0)
            {
                total_cost += compute_cost_cuda(
                    x, ref_xs, ref_ys, ref_yaws, ref_vs, path_len,
                    u_clamped, last_u, p, min_dist, &local_path_idx, t);
            }

            // 종점 진행도 보상
            if (t == T - 1 && path_len > 0)
            {
                int progress = local_path_idx - initial_path_idx;
                if (progress < -path_len / 2)
                    progress += path_len;
                int max_possible_progress = T + 10;
                progress = max(0, min(progress, max_possible_progress));

                // 1. 기존 로직: 무사히 완주한 경우 진행 칸수에 비례해 보상
                total_cost -= p.q_progress * (float)progress;

                // 2. 탈출 속도 강력 보상 (Out-In-Out 유도)
                // 속도의 제곱을 사용하여 코너 탈출 시 고속을 유지하는 궤적에 압도적인 가산점을 줍니다.
                total_cost -= p.q_escape_vel * (x.v * x.v) * 5.0f;
            }

            last_u = u_clamped;
        }
        costs[k] = total_cost;
    }

    // --- Host Functions ---
    __host__ ButterworthCoeffs compute_butterworth_coeffs(float fc, float dt)
    {
        ButterworthCoeffs coeffs;
        float fs = 1.0f / dt;
        float w0 = tanf(M_PI * fc / fs);
        float K = sqrtf(2.0f) * w0;
        float D = 1.0f + K + w0 * w0;

        coeffs.b0 = (w0 * w0) / D;
        coeffs.b1 = 2.0f * coeffs.b0;
        coeffs.b2 = coeffs.b0;
        coeffs.a1 = 2.0f * (w0 * w0 - 1.0f) / D;
        coeffs.a2 = (1.0f - K + w0 * w0) / D;
        return coeffs;
    }

    MPPISolver::MPPISolver(int K, int T, Params params) : K_(K), T_(T), params_(params)
    {
        params_.filter_coeffs = compute_butterworth_coeffs(3.0f, params_.dt);
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

        int max_path = 1000;
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
        cudaFree(d_left_bnd_xs_);
        cudaFree(d_left_bnd_ys_);
        cudaFree(d_right_bnd_xs_);
        cudaFree(d_right_bnd_ys_);
    }

    void MPPISolver::update_params(Params p)
    {
        params_ = p;
        params_.filter_coeffs = compute_butterworth_coeffs(3.0f, params_.dt);
    }

    void MPPISolver::set_reference_path(const std::vector<float> &xs, const std::vector<float> &ys,
                                        const std::vector<float> &yaws, const std::vector<float> &vs)
    {
        h_ref_xs_ = xs;
        h_ref_ys_ = ys;
        ref_path_len_ = xs.size();
        if (ref_path_len_ > 1000)
            ref_path_len_ = 1000;
        if (ref_path_len_ > 0)
        {
            CUDA_CHECK(cudaMemcpy(d_ref_xs_, xs.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_ys_, ys.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_yaws_, yaws.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_vs_, vs.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    void MPPISolver::set_boundaries(const std::vector<float> &left_xs, const std::vector<float> &left_ys,
                                    const std::vector<float> &right_xs, const std::vector<float> &right_ys)
    {
        bnd_len_ = left_xs.size();
        if (bnd_len_ > 1000)
            bnd_len_ = 1000;
        if (bnd_len_ > 0)
        {
            CUDA_CHECK(cudaMemcpy(d_left_bnd_xs_, left_xs.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_left_bnd_ys_, left_ys.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_right_bnd_xs_, right_xs.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_right_bnd_ys_, right_ys.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    Control MPPISolver::solve(const State &current_state)
    {
        CUDA_CHECK(cudaMemcpy(d_prev_controls_, h_prev_controls_.data(), T_ * sizeof(Control), cudaMemcpyHostToDevice));

        int start_path_idx = 0;
        if (!h_ref_xs_.empty())
        {
            float min_dist_sq = 1e9f;
            for (int i = 0; i < (int)h_ref_xs_.size(); ++i)
            {
                float dx = current_state.x - h_ref_xs_[i];
                float dy = current_state.y - h_ref_ys_[i];
                float d_sq = dx * dx + dy * dy;
                if (d_sq < min_dist_sq)
                {
                    min_dist_sq = d_sq;
                    start_path_idx = i;
                }
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

        if (params_.visualize_candidates)
        {
            CUDA_CHECK(cudaMemcpy(h_states_.data(), d_states_, K_ * T_ * sizeof(State), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaMemcpy(h_controls_.data(), d_controls_, K_ * T_ * sizeof(Control), cudaMemcpyDeviceToHost));

        return compute_optimal_control(current_state);
    }

    Control MPPISolver::compute_optimal_control(const State &current_state)
    {
        auto min_it = std::min_element(h_costs_.begin(), h_costs_.end());
        float min_cost = *min_it;
        best_k_ = static_cast<int>(std::distance(h_costs_.begin(), min_it));

        if (std::isinf(min_cost) || min_cost >= 1.0e8f)
        {
            Control stop_control = {0.0f, -5.0f};
            std::fill(h_prev_controls_.begin(), h_prev_controls_.end(), stop_control);
            return stop_control;
        }
        for (int k = 0; k < K_; ++k)
        {
            if (std::isnan(h_costs_[k]))
            {
                h_costs_[k] = 1.0e8f; // NaN 발생 시 최악의 비용으로 간주하여 도태시킴
            }
        }

        float lambda = params_.lambda;
        float sum_weights = 0.0f;
        for (int k = 0; k < K_; ++k)
        {
            if (std::isinf(h_costs_[k]))
            {
                h_weights_[k] = 0.0f;
            }
            else
            {
                h_weights_[k] = fast_exp(-(h_costs_[k] - min_cost) / lambda);
            }
            sum_weights += h_weights_[k];
        }
        if (sum_weights < 1e-6)
            sum_weights = 1e-6;

        std::vector<Control> weighted_controls(T_, {0.0f, 0.0f});
        for (int k = 0; k < K_; ++k)
        {
            float w = h_weights_[k] / sum_weights;
            for (int t = 0; t < T_; ++t)
            {
                Control u_k = h_controls_[k * T_ + t];
                weighted_controls[t].steer += w * u_k.steer;
                weighted_controls[t].accel += w * u_k.accel;
            }
        }

        optimal_controls_ = weighted_controls;
        Control output = weighted_controls[0];

        for (int t = 0; t < T_ - 1; ++t)
            h_prev_controls_[t] = weighted_controls[t + 1];
        h_prev_controls_[T_ - 1] = weighted_controls[T_ - 1];

        best_trajectory_.resize(T_);
        State sim_state = current_state;

        for (int t = 0; t < T_; ++t)
        {
            sim_state = update_dynamics(sim_state, weighted_controls[t], params_);
            best_trajectory_[t] = sim_state;
        }
        return output;
    }

    const std::vector<State> &MPPISolver::get_generated_trajectories() const { return h_states_; }
    const std::vector<State> &MPPISolver::get_best_trajectory() const { return best_trajectory_; }
    int MPPISolver::get_best_k() const { return best_k_; }
    const std::vector<Control> &MPPISolver::get_optimal_controls() const { return optimal_controls_; }
    const std::vector<float> &MPPISolver::get_costs() const { return h_costs_; }
    int MPPISolver::get_K() const { return K_; }
    int MPPISolver::get_T() const { return T_; }
}