#!/usr/bin/env python3
"""Task 4 1단계 — 마찰원(friction circle) 커플링 적용 여부를 데이터로 판단.

|a_x|(종방향 가속)가 큰 구간과 작은 구간에서, 현재 mppi_core.cu의
update_dynamics(Pacejka 모델)가 예측하는 a_y와 IMU 실측 a_y의 오차 분포가
유의미하게 다른지 확인한다. 판단 기준(태스크 스펙과 동일):
    두 그룹의 평균절대오차(MAE) 차이가 20% 이상, 또는
    |a_x|와 예측오차의 상관계수 |r| > 0.3
이면 "유의미"로 보고 2단계(F_fy/F_ry 스케일링) 구현으로 진행한다.

Task 2(identify_actuator_tau.py)의 bag 읽기 인프라를 재사용한다.

사용 예:
    python3 analyze_friction_circle.py /home/a/bags/rosbag2_2026_06_22-15_24_36 \
        --cmd-topic /ackermann_cmd --odom-topic /odom --imu-topic /imu/data
"""
import argparse
import sys

import numpy as np
import yaml

from identify_actuator_tau import read_series, pick_cmd_topic

# smppi_cuda_controller/config/params.yaml 과 동일한 기본값
# (실차에 배포되는 Pacejka 파라미터 — 여기서 어긋나면 판단 자체가 무의미해진다)
DEFAULT_PARAMS = dict(
    mass=3.74, l_f=0.163, l_r=0.162, Cm0=0.04,
    B_f=6.0926, C_f=1.2447, D_f=0.7955, E_f=0.7815,
    B_r=6.6457, C_r=2.2129, D_r=0.7317, E_r=0.0597,
)


def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def pacejka_ay(v, omega, slip_angle, steer, p):
    """mppi_core.cu update_dynamics 고속(dynamic) 블록의 ay 계산을 그대로 복제."""
    vx = v * np.cos(slip_angle)
    vy = v * np.sin(slip_angle)
    vx_safe = np.where(np.abs(vx) < 1e-3, np.sign(vx) * 1e-3 + (vx == 0) * 1e-3, vx)

    alpha_f = steer - np.arctan2(vy + p["l_f"] * omega, vx_safe)
    alpha_r = -np.arctan2(vy - p["l_r"] * omega, vx_safe)

    bf_a = p["B_f"] * alpha_f
    br_a = p["B_r"] * alpha_r
    F_fy = p["F_zf"] * p["D_f"] * np.sin(p["C_f"] * np.arctan(bf_a - p["E_f"] * (bf_a - np.arctan(bf_a))))
    F_ry = p["F_zr"] * p["D_r"] * np.sin(p["C_r"] * np.arctan(br_a - p["E_r"] * (br_a - np.arctan(br_a))))

    ay = (F_fy * np.cos(steer) + F_ry) / p["mass"]
    return ay


def complementary_slip_angle(t, ay_imu, x, y, yaw, v, alpha=0.95, fc_hz=2.0):
    """state_estimator.hpp의 상보필터를 오프라인(NumPy)으로 재현.
    t 그리드(대개 IMU 샘플 시각) 위에서 순차적으로 vy_hat을 적분/융합한다.
    """
    n = len(t)
    vy_hat = np.zeros(n)
    vy_pos_lp = np.zeros(n)
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 1e-4:
            vy_hat[i] = vy_hat[i - 1]
            vy_pos_lp[i] = vy_pos_lp[i - 1]
            continue
        wx = (x[i] - x[i - 1]) / dt
        wy = (y[i] - y[i - 1]) / dt
        cy, sy = np.cos(yaw[i]), np.sin(yaw[i])
        vy_pos_diff = -sy * wx + cy * wy

        rc = 1.0 / (2.0 * np.pi * fc_hz)
        lp_a = dt / (rc + dt)
        vy_pos_lp[i] = vy_pos_lp[i - 1] + lp_a * (vy_pos_diff - vy_pos_lp[i - 1])

        vy_imu_pred = vy_hat[i - 1] + ay_imu[i] * dt
        vy_hat[i] = alpha * vy_imu_pred + (1.0 - alpha) * vy_pos_lp[i]

    slip_angle = np.arctan2(vy_hat, np.abs(v) + 1e-5)
    return slip_angle, vy_hat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_path")
    ap.add_argument("--cmd-topic", default=None)
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--high-percentile", type=float, default=80.0,
                     help="|a_x| 상위 percentile 기준 (기본 80 = 상위 20%%)")
    ap.add_argument("--diff-pct-thresh", type=float, default=20.0)
    ap.add_argument("--corr-thresh", type=float, default=0.3)
    ap.add_argument("--out-yaml", default="friction_circle_judgement.yaml")
    args = ap.parse_args()

    cmd_topic = pick_cmd_topic(args.bag_path, args.cmd_topic)
    print(f"명령 토픽: {cmd_topic}\n오도메트리 토픽: {args.odom_topic}\nIMU 토픽: {args.imu_topic}")

    raw = read_series(args.bag_path, [cmd_topic, args.odom_topic, args.imu_topic])
    for topic in (cmd_topic, args.odom_topic, args.imu_topic):
        if not raw[topic]:
            print(f"[ERROR] {topic}에 메시지가 없습니다.", file=sys.stderr)
            sys.exit(1)

    t0 = min(raw[cmd_topic][0][0], raw[args.odom_topic][0][0], raw[args.imu_topic][0][0])

    t_cmd = np.array([t for t, _ in raw[cmd_topic]]) - t0
    steer_cmd = np.array([m.drive.steering_angle for _, m in raw[cmd_topic]])

    t_odom = np.array([t for t, _ in raw[args.odom_topic]]) - t0
    v_odom = np.array([m.twist.twist.linear.x for _, m in raw[args.odom_topic]])
    x_odom = np.array([m.pose.pose.position.x for _, m in raw[args.odom_topic]])
    y_odom = np.array([m.pose.pose.position.y for _, m in raw[args.odom_topic]])
    ori = [m.pose.pose.orientation for _, m in raw[args.odom_topic]]
    yaw_odom = np.array([quat_to_yaw(o.x, o.y, o.z, o.w) for o in ori])
    yaw_odom_unwrapped = np.unwrap(yaw_odom)

    t_imu = np.array([t for t, _ in raw[args.imu_topic]]) - t0
    ax_imu = np.array([m.linear_acceleration.x for _, m in raw[args.imu_topic]])
    ay_imu = np.array([m.linear_acceleration.y for _, m in raw[args.imu_topic]])
    omega_imu = np.array([m.angular_velocity.z for _, m in raw[args.imu_topic]])

    # 모든 신호를 IMU 그리드(가장 고주파)로 정렬
    v_i = np.interp(t_imu, t_odom, v_odom)
    x_i = np.interp(t_imu, t_odom, x_odom)
    y_i = np.interp(t_imu, t_odom, y_odom)
    yaw_i = np.interp(t_imu, t_odom, yaw_odom_unwrapped)
    yaw_i = np.arctan2(np.sin(yaw_i), np.cos(yaw_i))
    # 조향 명령은 ZOH(zero-order hold): 각 IMU 시각 직전의 최신 명령값 사용
    steer_i = steer_cmd[np.clip(np.searchsorted(t_cmd, t_imu, side="right") - 1, 0, len(t_cmd) - 1)]

    slip_angle_i, vy_hat_i = complementary_slip_angle(t_imu, ay_imu, x_i, y_i, yaw_i, v_i)

    p = dict(DEFAULT_PARAMS)
    l_wb = p["l_f"] + p["l_r"]
    p["F_zf"] = p["mass"] * 9.81 * p["l_r"] / l_wb
    p["F_zr"] = p["mass"] * 9.81 * p["l_f"] / l_wb

    ay_pred = pacejka_ay(v_i, omega_imu, slip_angle_i, steer_i, p)
    err = ay_pred - ay_imu
    abs_err = np.abs(err)

    # 정지/저속 구간은 모델도 실측도 의미가 없으므로 v > 0.3 m/s만 사용
    valid = v_i > 0.3
    ax_v, abs_err_v = np.abs(ax_imu[valid]), abs_err[valid]

    thresh = np.percentile(ax_v, args.high_percentile)
    high_mask = ax_v >= thresh
    low_mask = ~high_mask

    mae_high = float(np.mean(abs_err_v[high_mask]))
    mae_low = float(np.mean(abs_err_v[low_mask]))
    diff_pct = float(abs(mae_high - mae_low) / max(mae_low, 1e-6) * 100.0)
    corr = float(np.corrcoef(ax_v, abs_err_v)[0, 1]) if len(ax_v) > 2 else 0.0

    significant = (diff_pct >= args.diff_pct_thresh) or (abs(corr) > args.corr_thresh)

    print("\n=== 마찰원 커플링 판단 ===")
    print(f"유효 샘플 수: {valid.sum()} / {len(v_i)} (v > 0.3 m/s)")
    print(f"|a_x| 상위 {100 - args.high_percentile:.0f}% 임계값: {thresh:.3f} m/s^2 "
          f"(n_high={high_mask.sum()}, n_low={low_mask.sum()})")
    print(f"MAE(|a_x| 상위 그룹)  = {mae_high:.4f} m/s^2")
    print(f"MAE(|a_x| 나머지 그룹) = {mae_low:.4f} m/s^2")
    print(f"MAE 차이 = {diff_pct:.1f}%  (기준: {args.diff_pct_thresh}%)")
    print(f"corr(|a_x|, |오차|) = {corr:.3f}  (기준: |r| > {args.corr_thresh})")
    print(f"\n판단: {'유의미 — 2단계(스케일링) 구현 진행' if significant else '유의미하지 않음 — 스킵'}")

    out = {
        "n_valid_samples": int(valid.sum()),
        "n_total_samples": int(len(v_i)),
        "high_pct_threshold_ax": float(thresh),
        "n_high": int(high_mask.sum()),
        "n_low": int(low_mask.sum()),
        "mae_high": mae_high,
        "mae_low": mae_low,
        "mae_diff_pct": diff_pct,
        "corr_ax_abs_error": corr,
        "diff_pct_thresh": args.diff_pct_thresh,
        "corr_thresh": args.corr_thresh,
        "significant": bool(significant),
        "source_bag": args.bag_path,
    }
    with open(args.out_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"\n결과 저장: {args.out_yaml}")


if __name__ == "__main__":
    main()
