"""Discrete kinematic bicycle + IMU-free MLP residual simulator plant.

State layout: [x, y, steering, vx, yaw, yaw_rate, slip_angle, vy].
The update deliberately mirrors update_kinematic_mlp_residual() in mppi_core.cu.
"""
from pathlib import Path

import numpy as np


FEATURE_MEAN = np.asarray([
    1.695870161, .0369881652, -.989558935, -.0632561445,
    1.987617731, 1.987617731, 0., -.802030325,
], dtype=np.float32)
FEATURE_STD = np.asarray([
    2.035665035, .0986970961, 2.410519123, .24627991,
    1.846325278, 1.846325278, 1e-5, 1.945653796,
], dtype=np.float32)
NORM_INDEX = np.asarray([0, 1, 2, 3, 4, 5, 6, 7,
                         3, 4, 3, 4, 3, 4, 3, 4, 3, 4])


def load_weights(path):
    flat = np.fromfile(Path(path), dtype=np.float32)
    if flat.size != 3395:
        raise ValueError(f"expected 3395 float32 MLP values, got {flat.size}: {path}")
    offset = 0
    def take(count, shape):
        nonlocal offset
        value = flat[offset:offset + count].reshape(shape)
        offset += count
        return value
    return (take(64 * 18, (64, 18)), take(64, (64,)),
            take(32 * 64, (32, 64)), take(32, (32,)),
            take(3 * 32, (3, 32)), take(3, (3,)))


def silu(x):
    return x / (np.float32(1.0) + np.exp(-x, dtype=np.float32))


def step(state, steer, acceleration, imu, command_history, weights, dt=.02,
         lf=.163, lr=.161, min_speed=.5, max_speed=9.0,
         steer_scale=1.1058064699, steer_bias=-0.0300696939,
         no_slip=True):
    """Return (next_state, next_history, lateral_acceleration)."""
    s = np.asarray(state, dtype=np.float32)
    history = np.asarray(command_history, dtype=np.float32).reshape(10).copy()
    steer_command = np.float32(steer)
    steer = np.float32(np.clip(
        np.float32(steer_scale) * steer_command + np.float32(steer_bias),
        -.55, .55))
    dt = np.float32(dt)
    # Match CUDA MPPI Control semantics exactly: u.accel is acceleration,
    # and the network-visible speed command is derived from the rollout state.
    speed_cmd = np.float32(np.clip(
        s[3] + np.float32(acceleration) * dt, min_speed, max_speed))
    wheelbase = np.float32(lf + lr)
    beta = (np.float32(0.0) if no_slip else
            np.arctan(np.float32(lr) / wheelbase * np.tan(steer)).astype(np.float32))
    yaw_rate = np.float32(s[3] * np.cos(beta) * np.tan(steer) / wheelbase)

    classic_v = speed_cmd
    classic_vy = np.float32(classic_v * np.sin(beta))
    # `imu` is accepted for RaceCar API compatibility but intentionally not
    # used. This prevents model-output -> synthetic-IMU -> model feedback.
    raw = np.concatenate((
        s[[3, 7, 5]],
        np.asarray([steer_command, speed_cmd, classic_v, classic_vy, yaw_rate], np.float32),
        history,
    )).astype(np.float32)
    z = ((raw - FEATURE_MEAN[NORM_INDEX]) / FEATURE_STD[NORM_INDEX]).astype(np.float32)
    w1, b1, w2, b2, w3, b3 = weights
    h1 = silu((w1 @ z + b1).astype(np.float32))
    h2 = silu((w2 @ h1 + b2).astype(np.float32))
    corr = (np.tanh((w3 @ h2 + b3).astype(np.float32))
            * np.asarray([8., 8., 30.], np.float32)).astype(np.float32)

    nvx = np.float32(classic_v + corr[0] * dt)
    nvy = np.float32(classic_vy + corr[1] * dt)
    nomega = np.float32(yaw_rate + corr[2] * dt)
    c, sn = np.cos(s[4]).astype(np.float32), np.sin(s[4]).astype(np.float32)
    result = np.zeros(8, dtype=np.float32)
    result[0] = s[0] + (nvx * c - nvy * sn) * dt
    result[1] = s[1] + (nvx * sn + nvy * c) * dt
    result[2] = steer_command
    result[3] = nvx
    result[4] = np.arctan2(np.sin(s[4] + nomega * dt),
                           np.cos(s[4] + nomega * dt)).astype(np.float32)
    result[5] = nomega
    result[6] = np.arctan2(nvy, np.abs(nvx) + np.float32(1e-5)).astype(np.float32)
    result[7] = nvy
    ay = np.float32((nvy - s[7]) / dt + nvx * nomega)
    history[:-2] = history[2:]
    history[-2:] = (steer_command, speed_cmd)
    return result, history, ay


def get_standardized_state_kmlp(x):
    return {"x": x[0], "y": x[1], "delta": x[2], "v_x": x[3],
            "v_y": x[7], "yaw": x[4], "yaw_rate": x[5], "slip": x[6]}
