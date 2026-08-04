#ifndef STATE_ESTIMATOR_HPP_
#define STATE_ESTIMATOR_HPP_

#include <cmath>

namespace mppi {

// 상보필터(complementary filter) 기반 vy / slip_angle / omega 추정기.
//
// 풀 EKF(전체 상태) 대신, IMU 고주파 성분(선가속도 적분)과 위치미분 기반
// 저주파 성분(연속 프레임 위치차 → body frame 속도)만 결합하는 경량 구현.
// H 행렬 설계·자코비안 유도·공분산 튜닝이 필요 없고, vy observability
// 문제도 IMU 적분으로 우회한다.
//
// 콜백(on_imu/on_position)에서는 최신 원시값 스냅샷만 저장하고, 실제 융합
// 연산은 step()에서 컨트롤러 타이머 주기(35ms)에 맞춰 한 번만 수행한다.
// 콜백 주기가 서로 다른 여러 소스를 뒤섞어 계산하면 레이스 컨디션이 생기므로,
// "타이머 콜백 시작 시점에 최신 값을 스냅샷"하는 구조를 유지한다.
class ComplementaryStateEstimator {
public:
    struct Result {
        float vy;
        float slip_angle;
        float omega;
    };

    void set_alpha(float alpha) { alpha_ = alpha; }
    void set_lowpass_cutoff_hz(float fc_hz) { fc_hz_ = fc_hz; }

    // IMU 콜백에서 호출 — 스냅샷 저장만 수행 (연산 없음)
    void on_imu(float ay_body, float omega_z) {
        imu_ay_      = ay_body;
        imu_omega_z_ = omega_z;
        imu_valid_   = true;
    }

    // mcl_pose(또는 odom) 콜백에서 호출 — 스냅샷 저장만 수행 (연산 없음)
    void on_position(float x, float y, float yaw) {
        pos_x_ = x; pos_y_ = y; pos_yaw_ = yaw;
        pos_valid_ = true;
    }

    bool imu_active() const { return imu_valid_; }

    // 타이머 콜백 시작 시 1회 호출. dt는 컨트롤러 주기, vx는 현재 종방향
    // 속도 추정치(slip_angle 계산용).
    Result step(float dt, float vx) {
        // ── 저주파 성분: 위치 차분(world frame) → body frame 회전 → 저역통과 ──
        float vy_pos_diff = 0.0f;
        if (pos_valid_ && has_prev_pos_ && dt > 1e-4f) {
            float wx = (pos_x_ - prev_x_) / dt;
            float wy = (pos_y_ - prev_y_) / dt;
            float cy = std::cos(pos_yaw_), sy = std::sin(pos_yaw_);
            // world → body 회전 (vx_body는 종방향 상태에서 이미 별도 추정하므로 vy만 사용)
            vy_pos_diff = -sy * wx + cy * wy;
        }
        if (pos_valid_) { prev_x_ = pos_x_; prev_y_ = pos_y_; has_prev_pos_ = true; }

        // 1차 IIR 저역통과: cutoff fc_hz_ (기본 ~2Hz)
        float rc   = 1.0f / (2.0f * static_cast<float>(M_PI) * fc_hz_);
        float lp_a = dt / (rc + dt);
        vy_pos_diff_lp_ += lp_a * (vy_pos_diff - vy_pos_diff_lp_);

        // ── 고주파 성분: IMU 적분. 매 스텝 직전 융합값(vy_hat_)을 기준으로
        //    예측하므로, 재귀적으로 결합하면 IMU 경로에 자연스러운 하이패스
        //    효과가 생겨(alpha만큼 유지, (1-alpha)만큼 저주파 성분으로 보정)
        //    적분 드리프트가 무한정 누적되지 않는다. ──
        float ay_for_integration = imu_valid_ ? imu_ay_ : 0.0f;
        float vy_imu_pred = vy_hat_ + ay_for_integration * dt;

        vy_hat_ = alpha_ * vy_imu_pred + (1.0f - alpha_) * vy_pos_diff_lp_;

        // omega: IMU 우선(고주파 노이즈는 동일한 저역통과로 완화), IMU 없으면
        // 호출부가 기존 휠 오도메트리 omega를 그대로 유지하도록 imu_active()로 판단.
        if (imu_valid_) {
            omega_lp_ += lp_a * (imu_omega_z_ - omega_lp_);
        }

        float slip = std::atan2(vy_hat_, std::fabs(vx) + 1e-5f);
        return { vy_hat_, slip, omega_lp_ };
    }

private:
    float alpha_ = 0.95f;
    float fc_hz_ = 2.0f;

    bool  imu_valid_   = false;
    float imu_ay_      = 0.0f;
    float imu_omega_z_ = 0.0f;

    bool  pos_valid_    = false;
    bool  has_prev_pos_ = false;
    float pos_x_ = 0.0f, pos_y_ = 0.0f, pos_yaw_ = 0.0f;
    float prev_x_ = 0.0f, prev_y_ = 0.0f;

    float vy_pos_diff_lp_ = 0.0f;
    float vy_hat_         = 0.0f;
    float omega_lp_       = 0.0f;
};

} // namespace mppi
#endif
