"""No-IMU direct-speed kinematic MLP matching CUDA MPPI."""
from pathlib import Path

import numpy as np

FEATURE_MEAN = np.asarray([1.24055886, -.29015705, -.0524405465,
    1.30056524, 1.24055886, -.29015705, -.0524403863, 1.30050373,
    -.0524403863, 1.30050373, -.0524403863, 1.30050373, -.0524403863,
    1.30050373, -.0524403863, 1.30050373], np.float32)
FEATURE_STD = np.asarray([.443839431, .862938941, .219339266,
    .366840065, .443839431, .862938941, .219339624, .36683923,
    .219339624, .36683923, .219339624, .36683923, .219339624, .36683923,
    .219339624, .36683923], np.float32)


def load_weights(path):
    flat = np.fromfile(Path(path), dtype=np.float32)
    if flat.size not in (3267, 3299):
        raise ValueError(f"expected 3267 or 3299 float32 values, got {flat.size}: {path}")
    offset = 0
    def take(count, shape):
        nonlocal offset
        value = flat[offset:offset + count].reshape(shape)
        offset += count
        return value
    weights = (take(64 * 16, (64, 16)), take(64, (64,)),
            take(32 * 64, (32, 64)), take(32, (32,)),
            take(3 * 32, (3, 32)), take(3, (3,)))
    mean, std = FEATURE_MEAN, FEATURE_STD
    if flat.size == 3299:
        mean, std = take(16, (16,)), take(16, (16,))
    return weights, mean, std


def step(state, steer_command, speed_command, history, weights, mean, std,
         dt=.02, lf=.163, lr=.161, min_speed=-10., max_speed=10.,
         min_accel=-8., max_accel=8.5, speed_servo_kp=8.,
         steer_scale=.8954927921, steer_bias=-.0036726743):
    s = np.asarray(state, dtype=np.float32)
    dt = np.float32(dt)
    command = np.float32(steer_command)
    speed_command = np.float32(np.clip(speed_command, min_speed, max_speed))
    steer = np.float32(np.clip(steer_scale * command + steer_bias, -.55, .55))
    base_ax = np.float32(np.clip(speed_servo_kp * (speed_command - s[3]),
                                 min_accel, max_accel))
    base_v = np.float32(np.clip(s[3] + base_ax * dt, min_speed, max_speed))
    base_w = np.float32(s[3] * np.tan(steer) / np.float32(lf + lr))
    hist = np.asarray(history, dtype=np.float32).reshape(10).copy()
    raw = np.concatenate((s[[3, 5]], [command, speed_command, base_v, base_w], hist)).astype(np.float32)
    z = ((raw - mean) / std).astype(np.float32)
    w1, b1, w2, b2, w3, b3 = weights
    h1 = (w1 @ z + b1).astype(np.float32); h1 /= 1. + np.exp(-h1)
    h2 = (w2 @ h1 + b2).astype(np.float32); h2 /= 1. + np.exp(-h2)
    corr = np.tanh((w3 @ h2 + b3).astype(np.float32)) * np.asarray([8., 8., 30.], np.float32)
    nv = np.float32(base_v + corr[0] * dt)
    nw = np.float32(base_w + corr[2] * dt)
    result = np.zeros(8, np.float32)
    result[0] = s[0] + nv * np.cos(s[4]) * dt
    result[1] = s[1] + nv * np.sin(s[4]) * dt
    result[2] = command; result[3] = nv
    result[4] = np.arctan2(np.sin(s[4] + nw * dt), np.cos(s[4] + nw * dt))
    result[5] = nw; result[6] = 0.; result[7] = 0.
    hist[:-2] = hist[2:]; hist[-2:] = (command, speed_command)
    return result, hist
