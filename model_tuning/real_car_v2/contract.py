"""Shared real-car classic/residual dynamics contract.

Equations follow ``mppi_core.cu``: actuator hidden states and body velocities
advance first; pose then uses the new velocity and old heading.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

DT = 0.02
IDENTIFIED_PARAMETER_NAMES = (
    "speed_kp", "speed_accel_tau", "speed_brake_tau", "steer_tau",
    "Iz", "B_f", "C_f", "D_f", "E_f",
    "B_r", "C_r", "D_r", "E_r")
FEATURES = ("vx", "vy", "yaw_rate", "steer_cmd", "speed_cmd", "applied_steer",
            "steer_cmd_delta", "base_next_vx", "base_next_vy", "base_next_yaw_rate",
            "steer_t-4", "speed_t-4", "steer_t-3", "speed_t-3", "steer_t-2",
            "speed_t-2", "steer_t-1", "speed_t-1", "steer_t", "speed_t")
IMU_RESIDUAL_FEATURES = FEATURES + ("imu_ax", "imu_ay")
OUTPUTS = ("delta_ax", "delta_ay", "delta_yaw_accel")


@dataclass
class ClassicModelParameters:
    speed_kp: float = 0.7616888694734905
    speed_accel_tau: float = 0.04
    speed_brake_tau: float = 0.02
    v_ref_slew_rate_max: float = 8.0
    ax_min: float = -1.0
    ax_max: float = 1.0
    # Ackermann steering input is already the commanded wheel angle.
    steer_scale: float = 1.0
    steer_bias: float = 0.0
    steer_tau: float = 0.15514851356820727
    max_steer: float = 0.4788
    max_steer_rate: float = 6.544984694978735
    Iz: float = 0.04712
    B_f: float = 5.0
    C_f: float = 1.3
    D_f: float = 1.0
    E_f: float = 0.0
    B_r: float = 5.0
    C_r: float = 1.3
    D_r: float = 1.0
    E_r: float = 0.0

    @classmethod
    def from_mapping(cls, p):
        def get(*keys, default):
            return float(next((p[k] for k in keys if k in p), default))
        return cls(
            speed_kp=get("speed_servo_kp", "speed_kp", default=cls.speed_kp),
            speed_accel_tau=get("speed_reference_accel_time_constant", "speed_accel_tau", default=cls.speed_accel_tau),
            speed_brake_tau=get("speed_reference_brake_time_constant", "speed_brake_tau", default=cls.speed_brake_tau),
            v_ref_slew_rate_max=get("v_ref_slew_rate_max", "actuator_max_speed_reference_rate", default=cls.v_ref_slew_rate_max),
            ax_min=get("ax_min", "min_accel", default=cls.ax_min),
            ax_max=get("ax_max", "max_accel", default=cls.ax_max),
            steer_scale=get("kinematic_steer_scale", "steer_scale", default=cls.steer_scale),
            steer_bias=get("kinematic_steer_bias", "steer_bias", default=cls.steer_bias),
            steer_tau=get("steer_servo_time_constant", "steer_tau", default=cls.steer_tau),
            max_steer=get("max_steer", default=cls.max_steer),
            max_steer_rate=get("actuator_max_steer_rate", "max_steer_rate", default=cls.max_steer_rate),
            Iz=get("dynamic_mlp_I_z", "Iz", default=cls.Iz),
            **{q: get(f"dynamic_mlp_{q}", q, default=getattr(cls, q))
               for q in ("B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r")})

    @classmethod
    def from_yaml(cls, path):
        import yaml
        document = yaml.safe_load(Path(path).read_text())
        return cls.from_mapping(document.get("/**", {}).get("ros__parameters", document))

    def identified_dict(self):
        return {name: getattr(self, name) for name in IDENTIFIED_PARAMETER_NAMES}

    def digest(self):
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def runtime_updates(self):
        return {
            "speed_servo_kp": self.speed_kp,
            "speed_reference_accel_time_constant": self.speed_accel_tau,
            "speed_reference_brake_time_constant": self.speed_brake_tau,
            "actuator_max_speed_reference_rate": self.v_ref_slew_rate_max,
            "min_accel": self.ax_min, "max_accel": self.ax_max,
            "kinematic_steer_scale": self.steer_scale,
            "kinematic_steer_bias": self.steer_bias,
            "steer_servo_time_constant": self.steer_tau,
            "max_steer": self.max_steer,
            "actuator_max_steer_rate": self.max_steer_rate,
            "dynamic_mlp_I_z": self.Iz,
            **{f"dynamic_mlp_{q}": getattr(self, q)
               for q in ("B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r")}}


@dataclass
class Contract:
    dt: float = DT
    steer_scale: float = ClassicModelParameters.steer_scale
    steer_bias: float = ClassicModelParameters.steer_bias
    steer_tau: float = ClassicModelParameters.steer_tau
    max_steer: float = ClassicModelParameters.max_steer
    max_steer_rate: float = ClassicModelParameters.max_steer_rate
    speed_kp: float = ClassicModelParameters.speed_kp
    speed_accel_tau: float = ClassicModelParameters.speed_accel_tau
    speed_brake_tau: float = ClassicModelParameters.speed_brake_tau
    max_speed_reference_rate: float = ClassicModelParameters.v_ref_slew_rate_max
    position_speed_scale: float = 1.0
    drag: float = 0.0
    min_accel: float = ClassicModelParameters.ax_min
    max_accel: float = ClassicModelParameters.ax_max
    max_residual_ax: float = 0.0
    max_residual_ay: float = 8.0
    max_residual_yaw_accel: float = 12.0

    @classmethod
    def from_parameters(cls, p, dt=DT):
        return cls(dt=dt, steer_scale=p.steer_scale, steer_bias=p.steer_bias,
                   steer_tau=p.steer_tau, max_steer=p.max_steer,
                   max_steer_rate=p.max_steer_rate,
                   speed_kp=p.speed_kp, speed_accel_tau=p.speed_accel_tau,
                   speed_brake_tau=p.speed_brake_tau,
                   max_speed_reference_rate=p.v_ref_slew_rate_max,
                   min_accel=p.ax_min, max_accel=p.ax_max)

    def dump(self, path):
        Path(path).write_text(json.dumps({**asdict(self), "features": FEATURES,
                                         "outputs": OUTPUTS}, indent=2) + "\n")


def artifact_metadata(iteration_id, parameters, kf_version, config_paths=()):
    h = hashlib.sha256()
    for path in config_paths:
        h.update(Path(path).read_bytes())
    return {"iteration_id": int(iteration_id), "classic_parameter_hash": parameters.digest(),
            "kf_output_version": str(kf_version), "config_hash": h.hexdigest(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "residual_target_semantics": "temporary KF-transition minus classic-transition pseudo-label; not physical GT"}


def actuator_step(steer, steer_cmd, speed_cmd, vx, c=Contract()):
    target = np.clip(c.steer_scale*steer_cmd+c.steer_bias,
                     -c.max_steer, c.max_steer)
    steer_rate = np.clip((target-steer)/max(c.steer_tau, 1e-3),
                         -c.max_steer_rate, c.max_steer_rate)
    steer2 = np.clip(steer+steer_rate*c.dt, -c.max_steer, c.max_steer)
    ax = np.clip(c.speed_kp*(speed_cmd-vx)-c.drag*vx, c.min_accel, c.max_accel)
    return steer2, ax


def longitudinal_actuator_step(speed_reference, speed_cmd, vx, c=Contract()):
    tau = c.speed_accel_tau if speed_cmd >= speed_reference else c.speed_brake_tau
    rate = np.clip((speed_cmd-speed_reference)/max(tau, 1e-3),
                   -c.max_speed_reference_rate, c.max_speed_reference_rate)
    speed_reference2 = speed_reference+rate*c.dt
    ax = np.clip(c.speed_kp*(speed_reference2-vx)-c.drag*vx, c.min_accel, c.max_accel)
    return speed_reference2, ax


def warmup_speed_reference(commands, initial_vx, c):
    reference = float(initial_vx)
    for command in np.asarray(commands, float):
        reference, _ = longitudinal_actuator_step(reference, command, initial_vx, c)
    return reference


def warmup_applied_steer(commands, c):
    commands = np.asarray(commands, float)
    steer = float(np.clip(commands[0], -c.max_steer, c.max_steer)) if len(commands) else 0.0
    for command in commands[1:]:
        steer, _ = actuator_step(steer, command, 0.0, 0.0, c)
    return steer


def low_speed_gate(vx, c=Contract()):
    u=np.clip((np.abs(vx)-.2)/.3,0.,1.)
    return u*u*(3.-2.*u)


def residual_gates(vx, c=Contract()):
    one = low_speed_gate(vx, c)
    return np.stack((one, one, one), axis=-1) if np.ndim(one) else np.ones(3)


def integrate(state, base_accel, residual, c=Contract()):
    """Semi-implicit CUDA-compatible update; state=[x,y,yaw,vx,vy,r]."""
    x, y, yaw, vx, vy, r = np.asarray(state, float)
    limit = np.asarray((c.max_residual_ax, c.max_residual_ay, c.max_residual_yaw_accel))
    ax, ay, rdot = np.asarray(base_accel)+np.clip(np.asarray(residual), -limit, limit)
    nvx = vx+(ax+vy*r)*c.dt
    nvy = vy+(ay-vx*r)*c.dt
    nr = r+rdot*c.dt
    return np.array([x+c.position_speed_scale*(nvx*np.cos(yaw)-nvy*np.sin(yaw))*c.dt,
                     y+c.position_speed_scale*(nvx*np.sin(yaw)+nvy*np.cos(yaw))*c.dt,
                     (yaw+nr*c.dt+np.pi)%(2*np.pi)-np.pi, nvx, nvy, nr])
