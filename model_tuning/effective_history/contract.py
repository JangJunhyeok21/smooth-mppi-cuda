"""Contract for the 40 ms command-to-state effective dynamics model.

This model intentionally does not claim to identify a physical steering servo.
It maps commanded steering/speed history and causal vehicle observations to the
state 40 ms later.  Python training/export and CUDA must preserve this order.
"""
from dataclasses import asdict, dataclass
import json
import numpy as np

CONTROL_DT = 0.02
MODEL_DT = 0.04
HISTORY_STEPS = 10

FEATURES = (
    "vx", "vy_est", "yaw_rate", "ax_smooth", "ay_smooth",
    "steer_cmd", "speed_cmd",
    *(f"steer_t-{k}" for k in range(9, -1, -1)),
    *(f"speed_t-{k}" for k in range(9, -1, -1)),
    "steer_delta_1", "steer_delta_3", "speed_delta_1",
    "vx_steer", "vx2_steer", "vx_yaw_rate", "abs_vx_steer",
)
OUTPUTS = ("delta_vx", "delta_vy", "delta_yaw_rate")


@dataclass(frozen=True)
class EffectiveContract:
    control_dt: float = CONTROL_DT
    model_dt: float = MODEL_DT
    history_dt: float = CONTROL_DT
    history_steps: int = HISTORY_STEPS
    wheelbase: float = 0.324
    effective_steer_scale: float = 0.51
    effective_steer_bias: float = 0.01
    effective_yaw_response_tau: float = 0.10
    effective_max_yaw_accel: float = 15.0
    effective_speed_response_gain: float = 0.76
    effective_max_accel: float = 1.0
    effective_vy_decay_tau: float = 0.12
    position_speed_scale: float = 0.8633491306389823
    residual_limits: tuple = (0.12, 0.10, 0.25)
    feature_count: int = len(FEATURES)
    output_count: int = len(OUTPUTS)
    residual_semantics: str = "state correction after one 0.04 s transition"
    model_name: str = "effective_history_state_residual"

    def dump(self, path):
        payload = {**asdict(self), "features": list(FEATURES), "outputs": list(OUTPUTS),
                   "substep_contract": ["command[t] @ 0.02 s", "command[t+1] @ 0.02 s"]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def baseline_body_step(state, command0, command1, c=EffectiveContract()):
    """Two 20 ms command/history substeps and one 40 ms body-state result.

    state is [..., vx, vy, yaw_rate]. Commands are [..., steer, speed].
    The effective response state is yaw_rate itself; no unobserved rack angle is
    represented or reported as ground truth.
    """
    vx, vy, yaw_rate = np.moveaxis(np.asarray(state, dtype=float), -1, 0)
    for command in (command0, command1):
        steer, speed = np.moveaxis(np.asarray(command, dtype=float), -1, 0)
        effective_steer = c.effective_steer_scale * steer + c.effective_steer_bias
        target_yaw_rate = vx / c.wheelbase * np.tan(effective_steer)
        yaw_accel = np.clip((target_yaw_rate-yaw_rate)/c.effective_yaw_response_tau,
                            -c.effective_max_yaw_accel, c.effective_max_yaw_accel)
        accel = np.clip(c.effective_speed_response_gain*(speed-vx),
                        -c.effective_max_accel, c.effective_max_accel)
        vx = vx + accel*c.control_dt
        vy = vy + (-vy/c.effective_vy_decay_tau)*c.control_dt
        yaw_rate = yaw_rate + yaw_accel*c.control_dt
    return np.stack((vx, vy, yaw_rate), axis=-1)


def integrate_pose(pose, next_body, c=EffectiveContract()):
    x, y, yaw = np.moveaxis(np.asarray(pose, dtype=float), -1, 0)
    vx, vy, yaw_rate = np.moveaxis(np.asarray(next_body, dtype=float), -1, 0)
    nx = x + c.position_speed_scale*(vx*np.cos(yaw)-vy*np.sin(yaw))*c.model_dt
    ny = y + c.position_speed_scale*(vx*np.sin(yaw)+vy*np.cos(yaw))*c.model_dt
    return np.stack((nx, ny, wrap_angle(yaw+yaw_rate*c.model_dt)), axis=-1)


def make_features(body, accel, command_history):
    body=np.asarray(body); accel=np.asarray(accel); h=np.asarray(command_history)
    s=h[..., -1, 0]; vcmd=h[..., -1, 1]; vx=body[..., 0]; r=body[..., 2]
    return np.concatenate((body, accel, np.stack((s,vcmd),-1), h[..., :, 0], h[..., :, 1],
      np.stack((s-h[..., -2,0], s-h[..., -4,0], vcmd-h[..., -2,1], vx*s,
                vx*vx*s, vx*r, np.abs(vx)*s),-1)), axis=-1)

