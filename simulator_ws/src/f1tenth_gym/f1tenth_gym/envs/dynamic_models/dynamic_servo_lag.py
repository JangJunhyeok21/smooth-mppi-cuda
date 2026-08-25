"""Discrete Pacejka plant matching CUDA MPPI's DYNAMIC_SERVO_LAG model."""

import numpy as np


def step(state, steer_command, speed_command, actuator_steer,
         actuator_speed_reference, *, dt, lf, lr, mass, min_speed, max_speed,
         min_accel, max_accel, speed_servo_kp, speed_accel_tau,
         speed_brake_tau, max_speed_reference_rate, steer_scale, steer_bias,
         steer_time_constant, max_steer, max_steer_rate, Bf, Cf, Df, Ef, Br, Cr, Dr,
         Er, Iz):
    """Advance one MPPI knot and return state, actuator states, and FLU IMU.

    State layout: [x, y, steer_cmd, vx, yaw, yaw_rate, beta, vy].
    """
    current = np.asarray(state, dtype=np.float32)
    dt = np.float32(dt)
    steer_command = np.float32(steer_command)
    speed_command = np.float32(np.clip(speed_command, min_speed, max_speed))

    target_steer = np.float32(np.clip(
        steer_scale*steer_command+steer_bias, -max_steer, max_steer))
    steer_rate = np.float32(np.clip(
        (target_steer - actuator_steer) /
        max(steer_time_constant, 1.0e-3),
        -max_steer_rate, max_steer_rate))
    applied_steer = np.float32(np.clip(
        actuator_steer + steer_rate * dt, -max_steer, max_steer))

    speed_tau = (speed_accel_tau if speed_command >= actuator_speed_reference
                 else speed_brake_tau)
    reference_rate = np.float32(np.clip(
        (speed_command - actuator_speed_reference) / max(speed_tau, 1.0e-3),
        -max_speed_reference_rate, max_speed_reference_rate))
    speed_reference = np.float32(
        actuator_speed_reference + reference_rate * dt)

    vx, vy, yaw_rate = map(np.float32, (current[3], current[7], current[5]))
    ax = np.float32(np.clip(
        speed_servo_kp * (speed_reference - vx), min_accel, max_accel))
    safe_vx = np.float32(max(abs(float(vx)), 0.5))
    alpha_f = np.float32(
        applied_steer - np.arctan2(vy + lf * yaw_rate, safe_vx))
    alpha_r = np.float32(-np.arctan2(vy - lr * yaw_rate, safe_vx))
    fzf = np.float32(mass * 9.81 * lr / (lf + lr))
    fzr = np.float32(mass * 9.81 * lf / (lf + lr))
    bf_alpha, br_alpha = np.float32(Bf * alpha_f), np.float32(Br * alpha_r)
    front_inner = np.float32(
        bf_alpha - Ef * (bf_alpha - np.arctan(bf_alpha)))
    rear_inner = np.float32(
        br_alpha - Er * (br_alpha - np.arctan(br_alpha)))
    fyf = np.float32(fzf * Df * np.sin(Cf * np.arctan(front_inner)))
    fyr = np.float32(fzr * Dr * np.sin(Cr * np.arctan(rear_inner)))
    dynamic_ay = np.float32(
        (fyf * np.cos(applied_steer) + fyr) / mass)
    dynamic_yaw_accel = np.float32(
        (lf * fyf * np.cos(applied_steer) - lr * fyr) / Iz)

    # Same smooth 0.2--0.5 m/s kinematic/dynamic blend as mppi_core.cu.
    blend_input = np.float32(np.clip((abs(float(vx)) - 0.2) / 0.3, 0.0, 1.0))
    dynamic_blend = np.float32(
        blend_input * blend_input * (3.0 - 2.0 * blend_input))
    low_speed_tau = np.float32(0.1)
    kinematic_yaw_rate = np.float32(
        vx * np.tan(applied_steer) / max(lf + lr, 1.0e-6))
    kinematic_ay = np.float32(vx * yaw_rate - vy / low_speed_tau)
    kinematic_yaw_accel = np.float32(
        (kinematic_yaw_rate - yaw_rate) / low_speed_tau)
    ay = np.float32(dynamic_blend * dynamic_ay
                    + (1.0 - dynamic_blend) * kinematic_ay)
    yaw_accel = np.float32(dynamic_blend * dynamic_yaw_accel
                           + (1.0 - dynamic_blend) * kinematic_yaw_accel)

    next_vx = np.float32(vx + (ax + vy * yaw_rate) * dt)
    next_vy = np.float32(vy + (ay - vx * yaw_rate) * dt)
    next_yaw_rate = np.float32(yaw_rate + yaw_accel * dt)
    next_speed = np.float32(np.hypot(next_vx, next_vy))
    next_beta = np.float32(np.arctan2(next_vy, next_vx))

    output = current.copy()
    output[0] = current[0] + next_speed * np.cos(current[4] + next_beta) * dt
    output[1] = current[1] + next_speed * np.sin(current[4] + next_beta) * dt
    output[2] = steer_command
    output[3] = next_vx
    output[4] = np.arctan2(
        np.sin(current[4] + next_yaw_rate * dt),
        np.cos(current[4] + next_yaw_rate * dt))
    output[5] = next_yaw_rate
    output[6] = next_beta
    output[7] = next_vy
    imu = np.asarray((next_yaw_rate, ax, ay), dtype=np.float32)
    return output, applied_steer, speed_reference, imu
