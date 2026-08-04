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

    // 저속(kinematic) ↔ 고속(Pacejka dynamic) 모델을 연속적으로 섞는 가중치.
    // v가 center보다 한참 작으면 0(순수 kinematic), 한참 크면 1(순수 dynamic)에
    // 수렴하고 그 사이(±width 근방)에서 매끄럽게 전이한다. 기존의
    // `if (fabsf(vel) < 0.5f)` 하드 분기가 만들던 궤적 불연속(저속 코너
    // 진입/탈출 시 후보 궤적이 튀는 현상)을 없앤다.
    __host__ __device__ float blend_sigma(float v, float center, float width)
    {
        return 0.5f * (1.0f + tanhf((v - center) / width));
    }

    __host__ __device__ State update_dynamics(const State &s, const Control &u, const Params &p)
    {
        float px = s.x; float py = s.y; float yaw = s.yaw;
        float vel = s.v; float omega = s.omega; float slip_angle = s.slip_angle;
        float wheelbase = p.l_f + p.l_r;

        // ── 1. 운동학 모델 (Kinematic Model) ──────────────────────────
        State next_kin;
        {
            float dot_x = vel * fast_cos(yaw);
            float dot_y = vel * fast_sin(yaw);
            float dot_yaw = vel * tanf(u.steer) / wheelbase;
            float dot_vel = u.accel;

            next_kin.x = px + dot_x * p.dt;
            next_kin.y = py + dot_y * p.dt;
            next_kin.yaw = angle_normalize(yaw + dot_yaw * p.dt);
            next_kin.v = vel + dot_vel * p.dt;

            // 기존 MPC 기준 미사용 변수 초기화 (omega는 제어 연속성을 위해 dot_yaw 인가)
            next_kin.omega = dot_yaw;
            next_kin.slip_angle = 0.0f;
            next_kin.vy = 0.0f;
            next_kin.ay = 0.0f;
        }

        // ── 2. 파세이카 동역학 모델 (Pacejka Dynamic Model) ────────────
        //    이제 저속에서도 항상 계산되므로(가중치만 낮음), 기존에 하드
        //    분기가 암묵적으로 막아주던 division-by-zero(질량*속도)를
        //    여기서 별도로 가드한다.
        State next_dyn;
        {
            // 슬립각(beta)으로부터 차량 좌표계 vx, vy 역산 (타이어 슬립각 alpha 연산용)
            float vx = vel * fast_cos(slip_angle);
            float vy = vel * fast_sin(slip_angle);

            // 전/후륜 타이어 슬립각 계산
            float alpha_f = u.steer - atan2f(vy + p.l_f * omega, vx);
            float alpha_r = -atan2f(vy - p.l_r * omega, vx);

            // 타이어 횡력 연산 — ForzaETH On-Track-SysID 와 동일한 4-파라미터 매직 포뮬러.
            //   F_y = F_z * D * sin( C * atan( B*a - E*(B*a - atan(B*a)) ) )
            // D 는 무차원(마찰계수), 힘의 크기는 정하중 F_z 가 만든다.
            // E=0 이면 안쪽 괄호가 B*a 로 환원되어 이전 3-파라미터 수식과 완전히 동일해진다.
            float bf_a = p.B_f * alpha_f;
            float br_a = p.B_r * alpha_r;
            float F_fy = p.F_zf * p.D_f * fast_sin(p.C_f * atanf(bf_a - p.E_f * (bf_a - atanf(bf_a))));
            float F_ry = p.F_zr * p.D_r * fast_sin(p.C_r * atanf(br_a - p.E_r * (br_a - atanf(br_a))));

            // 기존 MPC 동역학 수식 완벽 적용 (단위 및 로직 동일)
            float dot_x = vel * fast_cos(yaw + slip_angle);
            float dot_y = vel * fast_sin(yaw + slip_angle);
            float dot_yaw = omega;
            float dot_vel = u.accel * (1.0f - p.Cm0 * vel); // 모터 감쇠 계수 적용
            float dot_omega = (p.l_f * F_fy * fast_cos(u.steer) - p.l_r * F_ry) / p.I_z; // 토크 / 관성모멘트

            // vel→0 근방에서 1/(mass*vel)이 발산하지 않도록 부호를 보존한 안전항을 둔다
            // (블렌딩 가중치 sigma가 이미 이 구간의 기여도를 낮춰주지만, 계산 자체가
            //  NaN/Inf가 되면 블렌딩으로 걸러지지 않으므로 별도 가드가 필요하다).
            float vel_safe = (vel >= 0.0f) ? fmaxf(vel, 1.0e-3f) : fminf(vel, -1.0e-3f);
            float dot_slip = ((F_fy + F_ry) / (p.mass * vel_safe)) - omega; // 횡력의 합 / (질량 * 속도)

            next_dyn.x = px + dot_x * p.dt;
            next_dyn.y = py + dot_y * p.dt;
            next_dyn.yaw = angle_normalize(yaw + dot_yaw * p.dt);
            next_dyn.v = vel + dot_vel * p.dt;
            next_dyn.omega = omega + dot_omega * p.dt;
            next_dyn.slip_angle = slip_angle + dot_slip * p.dt;

            // MPPI 비용 함수에서 사용하는 보조 변수 도출
            next_dyn.vy = next_dyn.v * fast_sin(next_dyn.slip_angle); // 슬립각 기반 vy
            next_dyn.ay = (F_fy * fast_cos(u.steer) + F_ry) / p.mass; // 횡가속도 a_y = F_y / m
        }

        // ── 3. tanh 연속 블렌딩 ─────────────────────────────────────
        float sigma = blend_sigma(vel, p.v_blend_center, p.v_blend_width);

        State next_s;
        next_s.x          = sigma * next_dyn.x          + (1.0f - sigma) * next_kin.x;
        next_s.y          = sigma * next_dyn.y          + (1.0f - sigma) * next_kin.y;

        // yaw는 각도이므로 단순 선형 블렌딩은 wraparound에서 틀린 결과를 낸다
        // (예: 179°와 -179°의 산술평균은 0°가 아니라 실제로는 180°여야 함).
        // kin.yaw를 기준으로 dyn과의 최단 각도차를 구해 그 방향으로 sigma만큼만 이동시킨다.
        float dyaw = angle_normalize(next_dyn.yaw - next_kin.yaw);
        next_s.yaw         = angle_normalize(next_kin.yaw + sigma * dyaw);

        next_s.v           = sigma * next_dyn.v           + (1.0f - sigma) * next_kin.v;
        next_s.omega       = sigma * next_dyn.omega       + (1.0f - sigma) * next_kin.omega;
        next_s.slip_angle  = sigma * next_dyn.slip_angle  + (1.0f - sigma) * next_kin.slip_angle;
        next_s.vy          = sigma * next_dyn.vy          + (1.0f - sigma) * next_kin.vy;
        next_s.ay          = sigma * next_dyn.ay          + (1.0f - sigma) * next_kin.ay;

        return next_s;
    }

    __device__ float compute_cost_cuda(
        const State &s,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws, int path_len,
        const Control &u, const Control &u_prev,
        const Params &p,
        float min_bnd_dist,
        int* last_idx)
    {
        float min_dist_sq = 1e9f;
        int nearest_idx = -1;

        int start_search = *last_idx; 
        int search_window = 30; 
        
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

        // 2. 속도 보상
        float vel_cost = - (p.q_v * 0.2f) * (s.v * fast_cos(s.yaw - ref_yaws[nearest_idx]));    //전체적인 직진성 유도

        // 4. Control Input Cost
        float d_steer = u.steer - u_prev.steer;
        float d_accel = u.accel - u_prev.accel;
        float rate_cost = p.q_du * (d_steer * d_steer + d_accel * d_accel);
        float steer_cost = p.q_steer * (u.steer * u.steer);
        
        // 6. Boundary Collision Cost
        float boundary_cost = 0.0f;
        float safe_dist = p.collision_radius + 0.35f;

        if (min_bnd_dist < safe_dist) {
            float penetration = safe_dist - min_bnd_dist;
            float soft_cost = 70.0f * (penetration * penetration); 

            float hard_cost = 0.0f;
            if (min_bnd_dist < p.collision_radius * 1.5f) {
                float diff = min_bnd_dist - p.collision_radius;
                float capped = fmaxf(diff, 1.0e-5f);
                hard_cost = p.q_collision * logf(1.0f + __expf(-30.0f * capped));
            }

            boundary_cost = soft_cost + hard_cost;
        }

        // 7. Obstacle Cost
        float obs_cost = 0.0f;
        for (int i = 0; i < p.num_obstacles; ++i) {
            float dx = s.x - p.obs_x[i];
            float dy = s.y - p.obs_y[i];
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < 1.5f) {
                obs_cost += p.q_obs / (dist - p.car_radius + 1e-3f);
            }
        }

        // 8. 횡가속도(그립 한계) 안전 비용
        //    현재 타이어 파라미터의 실제 최대 횡가속도는 약 7.48 m/s²
        //    (F_zf*D_f + F_zr*D_r)/mass) 이므로, 그 아래인 p.lat_g_threshold부터
        //    이차함수로 증가하는 비용을 부과해 MPPI가 그립 한계를 넘는 궤적을
        //    스스로 피하게 한다 (임계값을 넘을수록 페널티가 커져야 하므로
        //    boundary_cost의 soft_cost와 동일하게 제곱 증가 형태를 사용 —
        //    감쇠하는 지수함수를 쓰면 한계에 다가갈수록 오히려 비용이 줄어들어
        //    MPPI가 그립 한계 쪽으로 더 몰리는 역효과가 난다).
        float ay_abs = fabsf(s.ay);
        float lat_g_over = ay_abs - p.lat_g_threshold;
        float lat_g_cost = (lat_g_over > 0.f) ? p.q_lat_g * lat_g_over * lat_g_over : 0.f;

        return p.q_dist * dist_error + vel_cost + steer_cost + rate_cost + boundary_cost + obs_cost + lat_g_cost;
    }
    
    // [수정된 함수] O(N) 바운더리 탐색을 대체하는 O(1) 횡방향 오차 기반 거리 연산
    __device__ float compute_min_boundary_distance(
        const State &s,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws,
        const float *left_xs, const float *left_ys,
        const float *right_xs, const float *right_ys,
        int path_len, int current_path_idx, int *nearest_idx_out) 
    {
        if (ref_xs == nullptr || left_xs == nullptr || path_len <= 0) return 1e9f;

        // 1. 이전 인덱스 근처에서 가장 가까운 중심점(Reference) 탐색
        float min_dist_sq = 1e9f;
        int nearest_idx = current_path_idx;
        int search_window = 30;
        int start_search = current_path_idx;

        if (start_search >= path_len) start_search %= path_len;
        if (start_search < 0) start_search = 0;

        for (int offset = 0; offset < search_window; ++offset) {
            int i = start_search + offset;
            if (i >= path_len) i -= path_len;
            
            float dx = s.x - ref_xs[i];
            float dy = s.y - ref_ys[i];
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }
        
        // 찾은 인덱스를 외부로 반환하여 compute_cost_cuda의 탐색 속도도 높임
        if (nearest_idx_out != nullptr) *nearest_idx_out = nearest_idx;

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
        if (idx < K * T) curand_init(seed, idx, 0, &states[idx]);
    }

    __global__ void rollout_kernel(
        State *states, Control *controls, float *costs, curandState *rng_states,
        const State start_state, const Control *prev_controls, const Params p,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws, int path_len,
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
        
        float x_steer_prev1 = 0.0f, x_steer_prev2 = 0.0f;
        float y_steer_prev1 = 0.0f, y_steer_prev2 = 0.0f;
        float x_accel_prev1 = 0.0f, x_accel_prev2 = 0.0f;
        float y_accel_prev1 = 0.0f, y_accel_prev2 = 0.0f;

        for (int t = 0; t < T; ++t)
        {
            int idx = k * T + t;

            Control u_mean_curr = prev_controls[t];
            Control u_mean_prev = (t == 0) ? prev_controls[0] : prev_controls[t-1];
            
            float mean_delta_steer = u_mean_curr.steer - u_mean_prev.steer;  
            float mean_delta_accel = u_mean_curr.accel - u_mean_prev.accel;  

            // 1. Raw 백색 잡음 생성 (dt를 곱하지 않음)
            float raw_steer_noise = curand_normal(&rng_states[idx]) * p.noise_steer_std;
            float raw_accel_noise = curand_normal(&rng_states[idx]) * p.noise_accel_std;

            // 2. 2차 버터워스 필터링 (IIR 차분 방정식)
            float filtered_steer = p.filter_coeffs.b0 * raw_steer_noise 
                                + p.filter_coeffs.b1 * x_steer_prev1 + p.filter_coeffs.b2 * x_steer_prev2
                                - p.filter_coeffs.a1 * y_steer_prev1 - p.filter_coeffs.a2 * y_steer_prev2;
            
            float filtered_accel = p.filter_coeffs.b0 * raw_accel_noise 
                                + p.filter_coeffs.b1 * x_accel_prev1 + p.filter_coeffs.b2 * x_accel_prev2
                                - p.filter_coeffs.a1 * y_accel_prev1 - p.filter_coeffs.a2 * y_accel_prev2;

            // 3. 레지스터 상태 한 칸씩 시프트 (다음 루프를 위함)
            x_steer_prev2 = x_steer_prev1; x_steer_prev1 = raw_steer_noise;
            y_steer_prev2 = y_steer_prev1; y_steer_prev1 = filtered_steer;

            x_accel_prev2 = x_accel_prev1; x_accel_prev1 = raw_accel_noise;
            y_accel_prev2 = y_accel_prev1; y_accel_prev1 = filtered_accel;

            // 4. 부드러워진 노이즈에 dt를 곱해 최종 변화량 산출
            float noise_delta_steer = filtered_steer * p.dt;   
            float noise_delta_accel = filtered_accel * p.dt;  

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

            if(fabsf(x.ay) > p.lat_g_fault_threshold){   // 실제 최대 그립(~7.48 m/s²)보다 여유를 둔 하드 페일 임계값
                is_fault = true;
            }

            float min_dist = compute_min_boundary_distance(
                x, ref_xs, ref_ys, ref_yaws, left_bnd_xs, left_bnd_ys, right_bnd_xs, right_bnd_ys, path_len, local_path_idx, &local_path_idx);
            
            if (min_dist < p.collision_radius) {
                is_fault = true;
            }

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

                // 현재 틱(t)에서 사용한 제어 입력(u)의 조향각을 가져옵니다.
                float survival_steer = controls[k * T + t].steer * 0.1; 
                // 속도는 0.0f (또는 마찰원 한계 내의 급제동 -1.0f)로 설정하고 조향은 살립니다.
                Control safe_control = {survival_steer, -2.0f};

                for (int fill_t = t + 1; fill_t < T; ++fill_t) {
                    states[k * T + fill_t] = x;
                    controls[k * T + fill_t] = safe_control;
                }
                break;
            }

            if (path_len > 0)
            {
                total_cost += compute_cost_cuda(
                    x,
                    ref_xs, ref_ys, ref_yaws, path_len,
                    u_clamped, last_u, p, min_dist, &local_path_idx);
            }

            // 종점 진행도 보상
            if (t == T - 1 && path_len > 0) {
                int progress = local_path_idx - initial_path_idx;
                if (progress < -path_len / 2) progress += path_len; 
                int max_possible_progress = T + 10; 
                progress = max(0, min(progress, max_possible_progress));
                
                // 1. 기존 로직: 무사히 완주한 경우 진행 칸수에 비례해 보상
                total_cost -= p.q_progress * (float)progress; 

                // 2. 탈출 속도 강력 보상 (Out-In-Out 유도)
                // 속도의 제곱을 사용하여 코너 탈출 시 고속을 유지하는 궤적에 압도적인 가산점을 줍니다.
                total_cost -= p.q_escape_vel * (x.v * x.v); 
            }

            last_u = u_clamped;
        }
        costs[k] = total_cost;
    }

    // --- Host Functions ---
    __host__ ButterworthCoeffs compute_butterworth_coeffs(float fc, float dt) {
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

    MPPISolver::MPPISolver(int K, int T, Params params) : K_(K), T_(T), params_(params) {
        params_.filter_coeffs = compute_butterworth_coeffs(3.0f, params_.dt);
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
        
        int max_path = 1000;
        CUDA_CHECK(cudaMalloc(&d_ref_xs_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_ys_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_yaws_, max_path * sizeof(float)));

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
        cudaFree(d_ref_xs_); cudaFree(d_ref_ys_); cudaFree(d_ref_yaws_);
        cudaFree(d_left_bnd_xs_); cudaFree(d_left_bnd_ys_);
        cudaFree(d_right_bnd_xs_); cudaFree(d_right_bnd_ys_);
    }

    void MPPISolver::update_params(Params p) { 
        params_ = p; 
        params_.filter_coeffs = compute_butterworth_coeffs(3.0f, params_.dt);
    }   

    void MPPISolver::set_reference_path(const std::vector<float> &xs, const std::vector<float> &ys,
                                        const std::vector<float> &yaws) {
        h_ref_xs_ = xs; h_ref_ys_ = ys;
        ref_path_len_ = xs.size();
        if (ref_path_len_ > 1000) ref_path_len_ = 1000;
        if (ref_path_len_ > 0) {
            CUDA_CHECK(cudaMemcpy(d_ref_xs_, xs.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_ys_, ys.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_yaws_, yaws.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    void MPPISolver::set_boundaries(const std::vector<float>& left_xs, const std::vector<float>& left_ys,
                                    const std::vector<float>& right_xs, const std::vector<float>& right_ys) {
        bnd_len_ = left_xs.size();
        if (bnd_len_ > 1000) bnd_len_ = 1000; 
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
            d_ref_xs_, d_ref_ys_, d_ref_yaws_, ref_path_len_,
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
        for (int k = 0; k < K_; ++k) {
            if (std::isnan(h_costs_[k])) {
                h_costs_[k] = 1.0e8f; // NaN 발생 시 최악의 비용으로 간주하여 도태시킴
            }
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