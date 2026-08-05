#ifndef STATE_ESTIMATOR_HPP_
#define STATE_ESTIMATOR_HPP_

#include <cmath>
#include <algorithm>

namespace mppi {

// 운동학 슬립각(kinematic slip angle) 기반 vy / slip_angle / omega 추정기.
//
// 이전 버전은 IMU 선가속도 적분(vy_hat_ += ay*dt)과 위치미분을 상보필터로
// 섞었으나, 실차 bag + mocap GT 검증 결과 이 IMU로는 가속도 적분으로 vy를
// 얻을 수 없다(SNR<1 — 적분 오차 RMSE 0.34~0.45 m/s가 추정 대상
// |vy|≈0.08~0.22 m/s보다 2~5배 크다). 대신 자이로(정상·정확)와 휠 오도메트리
// (v, 정상)만 쓰는 운동학 관계 beta=atan(l_r*omega/v)가 훨씬 정확하다
// (슬립각 RMSE 5.61°→2.50°, 9.21°→5.60°, corr +0.89/+0.95).
// 가속도계는 쓰지 않는다 — 축이 뒤집혀 있고 적분도 불가능하다.
// 근거: ekf_pose/docs/ekf-pose-analysis-2026-08.md §2, §4, §5.
//
// 콜백(on_imu)에서는 최신 자이로 스냅샷만 저장하고, 실제 연산은 step()에서
// 컨트롤러 타이머 주기(35ms)에 맞춰 한 번만 수행한다 — 콜백 주기가 서로 다른
// 여러 소스를 뒤섞어 계산하면 레이스 컨디션이 생기므로, "타이머 콜백 시작
// 시점에 최신 값을 스냅샷"하는 구조를 유지한다.
class ComplementaryStateEstimator {
public:
    struct Result {
        float vy;
        float slip_angle;
        float omega;
    };

    void set_l_r(float l_r) { l_r_ = l_r; }
    void set_slip_min_speed(float v_min) { slip_min_speed_ = v_min; }
    void set_lowpass_cutoff_hz(float fc_hz) { fc_hz_ = fc_hz; }

    // IMU 콜백에서 호출 — 자이로 z 스냅샷만 저장 (연산 없음).
    void on_imu(float omega_z) {
        imu_omega_z_ = omega_z;
        imu_valid_   = true;
    }

    bool imu_active() const { return imu_valid_; }

    // 타이머 콜백 시작 시 1회 호출. dt는 omega 저역통과용, vx는 현재 종방향
    // 속도(휠 오도메트리) 추정치.
    Result step(float dt, float vx) {
        // 자이로 z를 1차 IIR 저역통과 (cutoff fc_hz_, 기본 ~2Hz)로 스무딩.
        if (imu_valid_ && dt > 1e-4f) {
            float rc   = 1.0f / (2.0f * static_cast<float>(M_PI) * fc_hz_);
            float lp_a = dt / (rc + dt);
            omega_lp_ += lp_a * (imu_omega_z_ - omega_lp_);
        }

        float v_safe = std::max(std::fabs(vx), slip_min_speed_);
        float slip = std::atan(l_r_ * omega_lp_ / v_safe);
        float vy   = vx * std::sin(slip);

        return { vy, slip, omega_lp_ };
    }

private:
    float l_r_            = 0.162f;
    float slip_min_speed_ = 0.5f;
    float fc_hz_          = 2.0f;

    bool  imu_valid_   = false;
    float imu_omega_z_ = 0.0f;
    float omega_lp_    = 0.0f;
};

} // namespace mppi
#endif
