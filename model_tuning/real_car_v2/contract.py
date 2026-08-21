"""Single source of truth for the real-car residual dynamics contract."""
from dataclasses import dataclass, asdict
import json
import numpy as np

DT = 0.02
FEATURES = ("vx","vy","yaw_rate","steer_cmd","speed_cmd","applied_steer",
            "steer_cmd_delta","base_next_vx","base_next_vy","base_next_yaw_rate",
            "steer_t-4","speed_t-4","steer_t-3","speed_t-3","steer_t-2",
            "speed_t-2","steer_t-1","speed_t-1","steer_t","speed_t")
IMU_RESIDUAL_FEATURES = FEATURES + ("imu_ax", "imu_ay")
OUTPUTS = ("delta_ax", "delta_ay", "delta_yaw_accel")

@dataclass
class Contract:
    dt: float = DT
    steer_scale: float = 0.50927964
    steer_bias: float = 0.01015773
    steer_tau: float = 0.15514851356820727
    max_steer_rate: float = 0.8344090950084138
    speed_kp: float = 0.7616888694734905
    speed_accel_tau: float = 0.04
    speed_brake_tau: float = 0.02
    max_speed_reference_rate: float = 8.0
    # Training/evaluation integrates body velocity directly; no fitted S_v.
    position_speed_scale: float = 1.0
    drag: float = 0.0
    min_accel: float = -1.0
    max_accel: float = 1.0
    max_residual_ax: float = 0.0
    max_residual_ay: float = 8.0
    max_residual_yaw_accel: float = 12.0
    def dump(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({**asdict(self), "features": FEATURES, "outputs": OUTPUTS}, f, indent=2)

def actuator_step(steer, steer_cmd, speed_cmd, vx, c=Contract()):
    target=np.clip(c.steer_scale*steer_cmd+c.steer_bias,-.55,.55)
    rate=np.clip((target-steer)/max(c.steer_tau,1e-3),-c.max_steer_rate,c.max_steer_rate)
    steer2=np.clip(steer+rate*c.dt,-.55,.55)
    ax=np.clip(c.speed_kp*(speed_cmd-vx)-c.drag*vx,c.min_accel,c.max_accel)
    return steer2,ax

def longitudinal_actuator_step(speed_reference, speed_cmd, vx, c=Contract()):
    tau=c.speed_accel_tau if speed_cmd>=speed_reference else c.speed_brake_tau
    rate=np.clip((speed_cmd-speed_reference)/max(tau,1e-3),
                 -c.max_speed_reference_rate,c.max_speed_reference_rate)
    speed_reference2=speed_reference+rate*c.dt
    ax=np.clip(c.speed_kp*(speed_reference2-vx)-c.drag*vx,c.min_accel,c.max_accel)
    return speed_reference2,ax

def low_speed_gate(vx, c=Contract()):
    """Compatibility API: residual gating was removed; always return one."""
    return np.ones_like(vx, dtype=float) if np.ndim(vx) else 1.0

def residual_gates(vx, c=Contract()):
    """Compatibility API: no speed-dependent residual gate remains."""
    one=low_speed_gate(vx,c)
    return np.stack((one,one,one),axis=-1) if np.ndim(one) else np.ones(3)

def integrate(state, base_accel, residual, c=Contract()):
    """state=[x,y,yaw,vx,vy,r], accelerations use ISO body axes."""
    x,y,yaw,vx,vy,r=np.asarray(state,float)
    limit=np.asarray((c.max_residual_ax,c.max_residual_ay,c.max_residual_yaw_accel))
    bounded=np.clip(np.asarray(residual),-limit,limit)
    ax,ay,rdot=np.asarray(base_accel)+bounded
    nvx=vx+(ax+vy*r)*c.dt
    nvy=vy+(ay-vx*r)*c.dt
    nr=r+rdot*c.dt
    return np.array([x+c.position_speed_scale*(nvx*np.cos(yaw)-nvy*np.sin(yaw))*c.dt,
      y+c.position_speed_scale*(nvx*np.sin(yaw)+nvy*np.cos(yaw))*c.dt,
      (yaw+nr*c.dt+np.pi)%(2*np.pi)-np.pi,nvx,nvy,nr])
