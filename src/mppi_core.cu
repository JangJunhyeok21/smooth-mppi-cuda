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
        float vel = s.v; float omega = s.omega; float slip_angle = s.slip_angle;
        
        // 1. 저속 구간: 순수 운동학 모델 (Kinematic Model)
        // 분모에 vel이 들어가는 동역학의 특이점(Division by zero)을 방지
        if (fabsf(vel) < 0.5f) {
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
        float dot_slip = ((F_fy + F_ry) / (p.mass * vel)) - omega; // 횡력의 합 / (질량 * 속도)

        // 상태 업데이트 (Euler Integration)
        State next_s;
        next_s.x = px + dot_x * p.dt;
        next_s.y = py + dot_y * p.dt;
        next_s.yaw = angle_normalize(yaw + dot_yaw * p.dt);
        next_s.v = vel + dot_vel * p.dt;
        next_s.omega = omega + dot_omega * p.dt;
        next_s.slip_angle = slip_angle + dot_slip * p.dt;

        // MPPI 비용 함수에서 사용하는 보조 변수 도출
        next_s.vy = next_s.v * fast_sin(next_s.slip_angle); // 슬립각 기반 vy 
        next_s.ay = (F_fy * fast_cos(u.steer) + F_ry) / p.mass; // 횡가속도 a_y = F_y / m 

        return next_s;
    }

    __device__ float compute_cost_cuda(
        const State &s,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws, int path_len,
        const Control &u, const Control &u_prev,
        const Params &p,
        float min_bnd_dist,
        float e_norm,
        const float *ref_kappa,
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
        // 소프트 페널티 시작 거리. 예전엔 +0.35 하드코딩이었는데, map1 기준
        // safe_dist=0.65 면 **트랙의 24% 구간에서 남는 주행폭이 0 이하**가 된다
        // (양쪽 마진이 트랙폭을 다 먹음). 그러면 그 구간에선 어디 있든 항상 페널티라
        // 횡방향 gradient 가 사라지고, 동시에 out-in-out 으로 반경을 키울 여지도 없어져
        // 최대조향각으로도 코너 탈출이 불가능해진다. docs D''''' 참고.
        float safe_dist = p.collision_radius + p.boundary_margin;

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
        //    [3-B] use_curvature_grip 이면 "달성값(s.ay, 타이어 모델 상한 7.38에서 포화)"
        //    대신 "요구값 v^2*kappa"(포화하지 않음)에 페널티를 건다. 포화하는 지표에
        //    페널티를 걸면 그립 한계를 넘어선 뒤로는 속도가 공짜가 되어 코너 진입
        //    감속이 걸리지 않는다. docs/smppi-diagnosis-2026-08.md A-3 참고.
        float lat_g_cost = 0.0f;
        if (p.use_curvature_grip && ref_kappa != nullptr) {
            //  요구 횡가속도 v^2*k 를 그대로 제곱 페널티에 넣으면 map1 처럼 곡률이 큰
            //  트랙(최소반경 0.70m)에서 v=4 일 때 요구값이 22 m/s^2 까지 올라가
            //  비용이 폭발한다. 대신 **곡률이 정하는 속도 한계**로 환산해 유계로 만든다:
            //      v_lim = sqrt(lat_g_threshold / |kappa|)
            //  이게 곧 "코너 앞에서 감속하라"는, 지금까지 비용함수에 없던 항이다.
            float k = fabsf(ref_kappa[nearest_idx]);
            if (k > 1e-3f) {
                float v_lim = sqrtf(p.lat_g_threshold / k);
                float over = s.v - v_lim;
                if (over > 0.f) lat_g_cost = p.q_lat_g * over * over;
            }
        } else {
            //  기존 방식: 모델이 "실제로 낸" s.ay. 타이어 상한(약 7.38)에서 포화하므로
            //  그 위로는 속도가 공짜가 된다. docs/smppi-diagnosis-2026-08.md A-3
            float lat_g_over = fabsf(s.ay) - p.lat_g_threshold;
            if (lat_g_over > 0.f) lat_g_cost = p.q_lat_g * lat_g_over * lat_g_over;
        }

        //    [3-C] 트랙 전폭 barrier. q_dist=0 이면 횡방향으로 비용이 평평해서
        //    최적해가 "벽 페널티가 막 켜지는 지점"에 딱 붙고 robustness margin 이 0 이 된다.
        //    (2*e_y/w)^6 은 |e_norm|<0.8 에서 거의 0(라인 자유도 유지), 경계 근처에서만
        //    급격히 커져 최적해를 제약면에서 떼어놓는다. docs A-1(b) 참고.
        float barrier_cost = 0.0f;
        if (p.q_barrier > 0.0f) {
            float en2 = e_norm * e_norm;
            barrier_cost = p.q_barrier * en2 * en2 * en2;   // e_norm^6
        }

        return p.q_dist * dist_error + vel_cost + steer_cost + rate_cost
             + boundary_cost + obs_cost + lat_g_cost + barrier_cost;
    }
    
    // [수정된 함수] O(N) 바운더리 탐색을 대체하는 O(1) 횡방향 오차 기반 거리 연산
    __device__ float compute_min_boundary_distance(
        const State &s,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws,
        const float *left_xs, const float *left_ys,
        const float *right_xs, const float *right_ys,
        int path_len, int current_path_idx, int *nearest_idx_out,
        float *e_norm_out)
    {
        if (e_norm_out != nullptr) *e_norm_out = 0.0f;
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

        // 4-a. 트랙 폭으로 정규화한 횡편차 (barrier 비용용).
        //      0=센터, ±1=좌/우 경계. 좌우 폭이 다르므로 해당 방향 폭으로 나눈다.
        if (e_norm_out != nullptr) {
            float half = (e_y >= 0.0f) ? w_left : w_right;
            *e_norm_out = (half > 1e-6f) ? (e_y / half) : 0.0f;
        }

        // 4-b. 차량에서 양쪽 바운더리까지의 최단 거리 반환
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
        const float *ref_xs, const float *ref_ys, const float *ref_yaws,
        const float *ref_kappa, int path_len,
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

        // 버터워스 IIR 상태를 정상상태 분포로 워밍업한다.
        // 0에서 시작하면 b0=0.073 탓에 t=0~2 구간의 노이즈가 10배 이상 감쇠되어,
        // 정작 차에 나가는 첫 스텝의 조향 탐색 폭이 0.06deg 로 죽는다 (8000 샘플이
        // 전부 이전 계획의 미세 변형이 되어 회피 기동이 후보 집합에 존재하지 않게 됨).
        // 필터 시정수가 ~1.5스텝이므로 8스텝이면 정상상태에 충분히 수렴한다.
        for (int w = 0; w < 8; ++w) {
            float rs = curand_normal(&rng_states[k * T]) * p.noise_steer_std;
            float ra = curand_normal(&rng_states[k * T]) * p.noise_accel_std;

            float ws = p.filter_coeffs.b0 * rs
                     + p.filter_coeffs.b1 * x_steer_prev1 + p.filter_coeffs.b2 * x_steer_prev2
                     - p.filter_coeffs.a1 * y_steer_prev1 - p.filter_coeffs.a2 * y_steer_prev2;
            float wa = p.filter_coeffs.b0 * ra
                     + p.filter_coeffs.b1 * x_accel_prev1 + p.filter_coeffs.b2 * x_accel_prev2
                     - p.filter_coeffs.a1 * y_accel_prev1 - p.filter_coeffs.a2 * y_accel_prev2;

            x_steer_prev2 = x_steer_prev1; x_steer_prev1 = rs;
            y_steer_prev2 = y_steer_prev1; y_steer_prev1 = ws;
            x_accel_prev2 = x_accel_prev1; x_accel_prev1 = ra;
            y_accel_prev2 = y_accel_prev1; y_accel_prev1 = wa;
        }

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

            float e_norm = 0.0f;
            float min_dist = compute_min_boundary_distance(
                x, ref_xs, ref_ys, ref_yaws, left_bnd_xs, left_bnd_ys, right_bnd_xs, right_bnd_ys,
                path_len, local_path_idx, &local_path_idx, &e_norm);
            
            if (min_dist < p.collision_radius) {
                is_fault = true;
            }

            if (is_fault) {
                // 전 샘플이 fault 나면(코너에서 상시 발생) 아래 항들만으로 순위가 정해지므로,
                // 여기서 무엇을 보상하느냐가 곧 비상시 정책이 된다.
                //   - 오래 버틸수록 이득 (-50t)      : 회피/감속 유도
                //   - 느리게 박을수록 이득 (+q_impact*v^2) : 충돌 에너지 최소화
                // 예전에 있던 progress 보상은 "더 멀리 가서 박기"를 유도해 코너 안쪽으로
                // 파고드는 원인이 되므로 제거했다 (진행 보상은 무사 완주 경로에만 준다).
                total_cost += 10000.0f - (float)t * 50.0f + p.q_impact * x.v * x.v;

                // 감속 정책(fault_accel)은 의도된 동작이라 유지하되, 조향은 죽이지 않는다.
                // 예전 코드는 steer*0.1 로 조향을 1/10 로 깎아 벽에서 벗어날 방법을 없앴다.
                Control safe_control = {u_clamped.steer, p.fault_accel};

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
                    u_clamped, last_u, p, min_dist, e_norm, ref_kappa, &local_path_idx);
            }

            // 종점 진행도 보상
            if (t == T - 1 && path_len > 0) {
                int progress = local_path_idx - initial_path_idx;
                if (progress < -path_len / 2) progress += path_len;
                // 랩어라운드 방어용 상한. T+10(=60칸=5.9m)이면 3.39 m/s 에서 이미 포화해
                // 모든 샘플이 같은 값으로 잘리고 progress 항이 샘플을 구분하지 못했다
                // (max_speed=4.0 이므로 정작 빠른 구간에서 gradient 가 0). 2T 로 올려
                // 5.6 m/s 까지 살려둔다.
                // ponytail: 칸 수 대신 CSV 누적 arc-length(m)를 올려 쓰는 게 제대로 된 수정이다.
                //           지금은 웨이포인트 간격(map1=0.099m)에 q_progress 가 암묵적으로 묶여 있다.
                int max_possible_progress = 2 * T;
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
        CUDA_CHECK(cudaMalloc(&d_ref_kappa_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_ref_kappa_, 0, max_path * sizeof(float)));

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
        cudaFree(d_ref_xs_); cudaFree(d_ref_ys_); cudaFree(d_ref_yaws_); cudaFree(d_ref_kappa_);
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

            // [3-B] 곡률 kappa 계산 후 업로드. 폐곡선 가정(랩어라운드).
            //   kappa = d(psi)/ds  — 이웃 3점의 헤딩 변화를 호길이로 나눈다.
            //   요구 횡가속도 v^2*kappa 를 그립 페널티 지표로 쓰기 위함.
            std::vector<float> kappa(ref_path_len_, 0.0f);
            const int n = ref_path_len_;
            for (int i = 0; i < n; ++i) {
                int im = (i - 1 + n) % n, ip = (i + 1) % n;
                float dpsi = yaws[ip] - yaws[im];
                while (dpsi >  M_PI) dpsi -= 2.0f * M_PI;
                while (dpsi < -M_PI) dpsi += 2.0f * M_PI;
                float ds = std::hypot(xs[ip] - xs[im], ys[ip] - ys[im]);
                kappa[i] = (ds > 1e-6f) ? (dpsi / ds) : 0.0f;
            }
            // 이산 미분 노이즈 억제용 이동평균 (반경 2)
            std::vector<float> ks(kappa);
            for (int i = 0; i < n; ++i) {
                float acc = 0.0f;
                for (int d = -2; d <= 2; ++d) acc += kappa[(i + d + n) % n];
                ks[i] = acc / 5.0f;
            }
            CUDA_CHECK(cudaMemcpy(d_ref_kappa_, ks.data(), n * sizeof(float), cudaMemcpyHostToDevice));
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
            d_ref_xs_, d_ref_ys_, d_ref_yaws_, d_ref_kappa_, ref_path_len_,
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