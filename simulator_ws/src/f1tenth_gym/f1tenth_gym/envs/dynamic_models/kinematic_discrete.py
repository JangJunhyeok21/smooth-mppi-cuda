"""Discrete kinematic bicycle model matching the CUDA MPPI plant update."""

import numpy as np


def step(state, steer, acceleration, dt=.02, lf=.163, lr=.161,
         min_speed=.5, max_speed=9.0, steer_scale=1.1058064699,
         steer_bias=-.0300696939, no_slip=True):
    """Advance ``state`` using MPPI ``Control{steer, acceleration}`` semantics."""
    s = np.asarray(state, dtype=np.float32)
    dt = np.float32(dt)
    steer_command = np.float32(steer)
    wheel_steer = np.float32(np.clip(
        np.float32(steer_scale) * steer_command + np.float32(steer_bias),
        -.55, .55,
    ))
    speed = np.float32(np.clip(
        s[3] + np.float32(acceleration) * dt, min_speed, max_speed,
    ))
    wheelbase = np.float32(lf + lr)
    beta = (np.float32(0.0) if no_slip else
            np.arctan(np.float32(lr) / wheelbase * np.tan(wheel_steer)).astype(np.float32))
    # Match CUDA update_kinematic(): pose/yaw use the current velocity and the
    # acceleration result is stored as the next state's velocity.
    current_speed = np.float32(s[3])
    lateral_speed = np.float32(speed * np.sin(beta))
    yaw_rate = np.float32(
        current_speed * np.cos(beta) * np.tan(wheel_steer) / wheelbase
    )

    result = np.zeros(8, dtype=np.float32)
    result[0] = s[0] + current_speed * np.cos(s[4] + beta).astype(np.float32) * dt
    result[1] = s[1] + current_speed * np.sin(s[4] + beta).astype(np.float32) * dt
    result[2] = steer_command # delta
    result[3] = speed # v
    result[4] = np.arctan2(np.sin(s[4] + yaw_rate * dt),
                           np.cos(s[4] + yaw_rate * dt)).astype(np.float32) # yaw = yaw + yaw_rate * dt
    result[5] = yaw_rate # yaw_rate
    result[6] = np.arctan2(lateral_speed,
                           np.abs(speed) + np.float32(1e-5)).astype(np.float32) # slip = arctan(v_y / v_x)
    result[7] = lateral_speed # v_y = v * sin(beta)
    return result


def get_standardized_state_kinematic(x):
    return {"x": x[0], "y": x[1], "delta": x[2], "v_x": x[3],
            "v_y": x[7], "yaw": x[4], "yaw_rate": x[5], "slip": x[6]}
