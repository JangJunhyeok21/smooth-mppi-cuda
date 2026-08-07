#!/usr/bin/env python3
"""Causal, scalar 2-state lateral-velocity KF matching the C++ observer."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class LateralVelocityKFParams:
    cornering_stiffness_front: float = 110.0
    cornering_stiffness_rear: float = 199.0
    mass: float = 3.74
    yaw_inertia: float = 0.04712
    l_f: float = 0.163
    l_r: float = 0.161
    dt: float = 0.02
    min_longitudinal_speed: float = 0.5
    low_speed_threshold: float = 1.5
    max_abs_vy: float = 2.0
    process_var_vy: float = 0.02
    process_var_yaw_rate: float = 0.02
    measurement_var_lateral_accel: float = 0.5
    measurement_var_yaw_rate: float = 0.01
    initial_var_vy: float = 0.25
    initial_var_yaw_rate: float = 0.10
    # IMU linear_acceleration.y must be vehicle body +Y (left).
    imu_lateral_accel_sign: float = 1.0


class LateralVelocityKF:
    """No nonlinear tire calls and no matrix inverse/allocation in update()."""

    def __init__(self, params=LateralVelocityKFParams()):
        self.p = params
        self.initialized = False
        self.reset(0.0)

    def reset(self, measured_yaw_rate=0.0):
        self.vy = 0.0
        self.yaw_rate = measured_yaw_rate if math.isfinite(measured_yaw_rate) else 0.0
        self.p00, self.p01 = max(self.p.initial_var_vy, 1e-8), 0.0
        self.p10, self.p11 = 0.0, max(self.p.initial_var_yaw_rate, 1e-8)
        self.initialized = True

    def _covariance(self, p00, p01, p10, p11):
        off = 0.5 * (p01 + p10)
        self.p00, self.p01 = max(p00, 1e-8), off
        self.p10, self.p11 = off, max(p11, 1e-8)

    def _clamp(self):
        if not math.isfinite(self.vy) or not math.isfinite(self.yaw_rate):
            self.reset(0.0)
        self.vy = float(np.clip(self.vy, -self.p.max_abs_vy, self.p.max_abs_vy))

    def update(self, measured_vx, steering_angle, measured_yaw_rate,
               measured_lateral_accel):
        p = self.p
        abs_vx = abs(measured_vx) if math.isfinite(measured_vx) else 0.0
        yaw_ok, ay_ok = math.isfinite(measured_yaw_rate), math.isfinite(measured_lateral_accel)
        if abs_vx < p.low_speed_threshold:
            self.vy = 0.0
            if yaw_ok:
                self.yaw_rate = measured_yaw_rate
            self.p00 = min(self.p00 + p.process_var_vy, max(p.initial_var_vy, 1e-8))
            self.p01 = self.p10 = 0.0
            self.p11 = max(p.measurement_var_yaw_rate, 1e-8) if yaw_ok else self.p11 + p.process_var_yaw_rate
            return self.vy

        vx = max(abs_vx, p.min_longitudinal_speed)
        cf, cr, m, iz, lf, lr = (p.cornering_stiffness_front,
                                  p.cornering_stiffness_rear, p.mass,
                                  p.yaw_inertia, p.l_f, p.l_r)
        iv = 1.0 / vx
        a00 = -(cf + cr) * iv / m
        a01 = -(vx + (lf * cf - lr * cr) * iv / m)
        a10 = -(lf * cf - lr * cr) * iv / iz
        a11 = -(lf * lf * cf + lr * lr * cr) * iv / iz
        ad00, ad01 = 1 + p.dt*a00, p.dt*a01
        ad10, ad11 = p.dt*a10, 1 + p.dt*a11
        vy0, w0 = self.vy, self.yaw_rate
        self.vy = ad00*vy0 + ad01*w0 + p.dt*cf/m*steering_angle
        self.yaw_rate = ad10*vy0 + ad11*w0 + p.dt*lf*cf/iz*steering_angle

        ap00, ap01 = ad00*self.p00 + ad01*self.p10, ad00*self.p01 + ad01*self.p11
        ap10, ap11 = ad10*self.p00 + ad11*self.p10, ad10*self.p01 + ad11*self.p11
        pp00 = ap00*ad00 + ap01*ad01 + p.process_var_vy
        pp01 = ap00*ad10 + ap01*ad11
        pp10 = ap10*ad00 + ap11*ad01
        pp11 = ap10*ad10 + ap11*ad11 + p.process_var_yaw_rate
        if not (yaw_ok and ay_ok):
            self._covariance(pp00, pp01, pp10, pp11); self._clamp(); return self.vy

        h00 = -(cf + cr)*iv/m
        h01 = (-lf*cf + lr*cr)*iv/m
        r0 = p.imu_lateral_accel_sign*measured_lateral_accel - (h00*self.vy + h01*self.yaw_rate + cf/m*steering_angle)
        r1 = measured_yaw_rate - self.yaw_rate
        hp00, hp01 = h00*pp00 + h01*pp10, h00*pp01 + h01*pp11
        s00 = hp00*h00 + hp01*h01 + p.measurement_var_lateral_accel
        s01 = hp01
        s10 = pp10*h00 + pp11*h01
        s11 = pp11 + p.measurement_var_yaw_rate
        det = s00*s11 - s01*s10
        if not math.isfinite(det) or abs(det) < 1e-10:
            self._covariance(pp00, pp01, pp10, pp11); self._clamp(); return self.vy
        si00, si01, si10, si11 = s11/det, -s01/det, -s10/det, s00/det
        ph00, ph01 = pp00*h00 + pp01*h01, pp01
        ph10, ph11 = pp10*h00 + pp11*h01, pp11
        k00, k01 = ph00*si00 + ph01*si10, ph00*si01 + ph01*si11
        k10, k11 = ph10*si00 + ph11*si10, ph10*si01 + ph11*si11
        self.vy += k00*r0 + k01*r1
        self.yaw_rate += k10*r0 + k11*r1
        ikh00, ikh01 = 1-k00*h00, -(k00*h01+k01)
        ikh10, ikh11 = -k10*h00, 1-k10*h01-k11
        self._covariance(ikh00*pp00+ikh01*pp10, ikh00*pp01+ikh01*pp11,
                         ikh10*pp00+ikh11*pp10, ikh10*pp01+ikh11*pp11)
        self._clamp()
        return self.vy

    def get_vy(self): return self.vy
    def get_yaw_rate(self): return self.yaw_rate


def estimate_dataset(samples, columns, dt, params=None, steer_scale=1.1058064699,
                     steer_bias=-0.0300696939, max_steer=0.4788,
                     imu_ema_alpha=0.25):
    """Estimate causally and reset KF/IMU EMA at every data discontinuity."""
    names = {str(name): i for i, name in enumerate(columns)}
    required = ("t", "vx", "steer", "bag_id", "imu_wz", "imu_ay")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"KF dataset is missing columns: {missing}")
    cfg = params or LateralVelocityKFParams(dt=dt)
    cfg.dt = dt
    kf = LateralVelocityKF(cfg)
    vy = np.zeros(len(samples)); yaw_rate = np.zeros(len(samples))
    previous_bag, previous_t = None, None
    filtered_wz = filtered_ay = 0.0
    for i, row in enumerate(samples):
        bag, stamp = int(row[names["bag_id"]]), row[names["t"]]
        wz, ay = row[names["imu_wz"]], row[names["imu_ay"]]
        reset = (previous_bag != bag or previous_t is None or
                 abs(stamp-previous_t-dt) > 0.5*dt)
        if reset:
            filtered_wz, filtered_ay = wz, ay
            kf.reset(filtered_wz)
        else:
            alpha = float(np.clip(imu_ema_alpha, 0.0, 1.0))
            filtered_wz = alpha*wz + (1.0-alpha)*filtered_wz
            filtered_ay = alpha*ay + (1.0-alpha)*filtered_ay
        delta = np.clip(steer_scale*row[names["steer"]] + steer_bias, -max_steer, max_steer)
        vy[i] = kf.update(row[names["vx"]], delta, filtered_wz, filtered_ay)
        yaw_rate[i] = kf.get_yaw_rate()
        previous_bag, previous_t = bag, stamp
    return vy, yaw_rate
