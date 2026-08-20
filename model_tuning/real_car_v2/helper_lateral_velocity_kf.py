#!/usr/bin/env python3
"""Causal, scalar 2-state lateral-velocity KF matching the C++ observer."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class LateralVelocityKFParams:
    # 과거 선형 KF 비교 스크립트 전용이며 Pacejka EKF 계산에는 사용하지 않는다.
    cornering_stiffness_front: float = 110.0
    cornering_stiffness_rear: float = 199.0
    mass: float = 3.74
    yaw_inertia: float = 0.04712
    l_f: float = 0.163
    l_r: float = 0.161
    pacejka_b_front: float = 2.9844349007584565
    pacejka_c_front: float = 1.3
    pacejka_d_front: float = 0.362611229414815
    pacejka_e_front: float = 0.0
    pacejka_b_rear: float = 0.3173165891873783
    pacejka_c_rear: float = 1.3
    pacejka_d_rear: float = 2.799999941680244
    pacejka_e_rear: float = 0.0
    dt: float = 0.02
    min_longitudinal_speed: float = 0.5
    low_speed_threshold: float = 0.0
    max_abs_vy: float = 2.0
    process_var_vy: float = 0.02
    process_var_yaw_rate: float = 0.02
    measurement_var_lateral_accel: float = 0.5
    measurement_var_yaw_rate: float = 0.01
    initial_var_vy: float = 0.25
    initial_var_yaw_rate: float = 0.10
    # IMU linear_acceleration.y must be vehicle body +Y (left).
    imu_lateral_accel_sign: float = 1.0
    process_var_ay_bias: float = 0.0
    initial_var_ay_bias: float = 1.0e-8
    max_abs_ay_bias: float = 0.0
    measurement_var_pose_vy: float = 0.08
    pose_vy_gate: float = 0.8


class LateralVelocityKF:
    """Two-state EKF using the same Pacejka lateral-force model as MPPI."""

    def __init__(self, params=LateralVelocityKFParams()):
        self.p = params
        self.initialized = False
        self.reset(0.0)

    def reset(self, measured_yaw_rate=0.0):
        self.vy = 0.0
        self.yaw_rate = measured_yaw_rate if math.isfinite(measured_yaw_rate) else 0.0
        self.p00, self.p01 = max(self.p.initial_var_vy, 1e-8), 0.0
        self.p10, self.p11 = 0.0, max(self.p.initial_var_yaw_rate, 1e-8)
        self.ay_bias = 0.0
        self.p_bias = max(self.p.initial_var_ay_bias, 1e-8)
        self.initialized = True

    def _covariance(self, p00, p01, p10, p11):
        off = 0.5 * (p01 + p10)
        self.p00, self.p01 = max(p00, 1e-8), off
        self.p10, self.p11 = off, max(p11, 1e-8)

    def _clamp(self):
        if not math.isfinite(self.vy) or not math.isfinite(self.yaw_rate):
            self.reset(0.0)
        self.vy = float(np.clip(self.vy, -self.p.max_abs_vy, self.p.max_abs_vy))

    def _update_pose_vy(self, measured_pose_vy):
        if not math.isfinite(measured_pose_vy):
            return
        residual = measured_pose_vy-self.vy
        if abs(residual) > self.p.pose_vy_gate:
            return
        gain = self.p00/(self.p00+max(self.p.measurement_var_pose_vy, 1e-8))
        self.vy += gain*residual
        self.p00 = max((1.0-gain)*self.p00, 1e-8)
        self.p01 *= 1.0-gain
        self.p10 = self.p01

    @staticmethod
    def _force(slip, fz, b, c, d, e):
        bs = b*slip
        return fz*d*math.sin(c*math.atan(bs-e*(bs-math.atan(bs))))

    def _dynamics(self, vx, steering, vy, yaw_rate):
        p = self.p
        safe_vx = max(abs(vx), p.min_longitudinal_speed)
        af = steering-math.atan2(vy+p.l_f*yaw_rate, safe_vx)
        ar = -math.atan2(vy-p.l_r*yaw_rate, safe_vx)
        wheelbase = p.l_f+p.l_r
        fzf, fzr = p.mass*9.81*p.l_r/wheelbase, p.mass*9.81*p.l_f/wheelbase
        fyf = self._force(af, fzf, p.pacejka_b_front, p.pacejka_c_front,
                          p.pacejka_d_front, p.pacejka_e_front)
        fyr = self._force(ar, fzr, p.pacejka_b_rear, p.pacejka_c_rear,
                          p.pacejka_d_rear, p.pacejka_e_rear)
        ay = (fyf*math.cos(steering)+fyr)/p.mass
        return ay-vx*yaw_rate, (p.l_f*fyf*math.cos(steering)-p.l_r*fyr)/p.yaw_inertia, ay

    def update(self, measured_vx, steering_angle, measured_yaw_rate,
               measured_lateral_accel, measured_pose_vy=math.nan):
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
        vy0, w0 = self.vy, self.yaw_rate
        dvy, rdot, model_ay = self._dynamics(vx, steering_angle, vy0, w0)
        pacejka_vy_prediction = vy0+p.dt*dvy
        self.yaw_rate = w0+p.dt*rdot
        eps = 1e-3
        dvy_v, rdot_v, _ = self._dynamics(vx, steering_angle, vy0+eps, w0)
        dvy_r, rdot_r, _ = self._dynamics(vx, steering_angle, vy0, w0+eps)
        ad00, ad01 = 1+p.dt*(dvy_v-dvy)/eps, p.dt*(dvy_r-dvy)/eps
        ad10, ad11 = p.dt*(rdot_v-rdot)/eps, 1+p.dt*(rdot_r-rdot)/eps
        self.vy = pacejka_vy_prediction

        ap00, ap01 = ad00*self.p00 + ad01*self.p10, ad00*self.p01 + ad01*self.p11
        ap10, ap11 = ad10*self.p00 + ad11*self.p10, ad10*self.p01 + ad11*self.p11
        pp00 = ap00*ad00 + ap01*ad01 + p.process_var_vy
        pp01 = ap00*ad10 + ap01*ad11
        pp10 = ap10*ad00 + ap11*ad01
        pp11 = ap10*ad10 + ap11*ad11 + p.process_var_yaw_rate
        if not (yaw_ok and ay_ok):
            self._covariance(pp00, pp01, pp10, pp11)
            self._update_pose_vy(measured_pose_vy); self._clamp(); return self.vy

        _, _, predicted_ay = self._dynamics(vx, steering_angle, self.vy, self.yaw_rate)
        _, _, ay_v = self._dynamics(vx, steering_angle, self.vy+eps, self.yaw_rate)
        _, _, ay_r = self._dynamics(vx, steering_angle, self.vy, self.yaw_rate+eps)
        h00, h01 = (ay_v-predicted_ay)/eps, (ay_r-predicted_ay)/eps
        r0 = p.imu_lateral_accel_sign*measured_lateral_accel - (predicted_ay+self.ay_bias)
        r1 = measured_yaw_rate - self.yaw_rate
        hp00, hp01 = h00*pp00 + h01*pp10, h00*pp01 + h01*pp11
        s00 = hp00*h00 + hp01*h01 + p.measurement_var_lateral_accel
        s01 = hp01
        s10 = pp10*h00 + pp11*h01
        s11 = pp11 + p.measurement_var_yaw_rate
        det = s00*s11 - s01*s10
        if not math.isfinite(det) or abs(det) < 1e-10:
            self._covariance(pp00, pp01, pp10, pp11)
            self._update_pose_vy(measured_pose_vy); self._clamp(); return self.vy
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
        self.p_bias += max(p.process_var_ay_bias, 0.0)
        bias_gain = self.p_bias / (self.p_bias + p.measurement_var_lateral_accel)
        self.ay_bias = float(np.clip(self.ay_bias+bias_gain*r0,
                                    -p.max_abs_ay_bias, p.max_abs_ay_bias))
        self.p_bias = max((1.0-bias_gain)*self.p_bias, 1e-8)
        self._update_pose_vy(measured_pose_vy)
        self._clamp()
        return self.vy

    def get_vy(self): return self.vy
    def get_yaw_rate(self): return self.yaw_rate


def estimate_dataset(samples, columns, dt, params=None, steer_scale=1.1058064699,
                     steer_bias=-0.0300696939, max_steer=0.4788,
                     imu_ema_alpha=0.25, imu_wz_sign=1.0, imu_ay_sign=1.0,
                     use_pose_vy=False, pose_window_s=0.12):
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
    pose_history = []
    for i, row in enumerate(samples):
        bag, stamp = int(row[names["bag_id"]]), row[names["t"]]
        wz = imu_wz_sign * row[names["imu_wz"]]
        ay = imu_ay_sign * row[names["imu_ay"]]
        reset = (previous_bag != bag or previous_t is None or
                 abs(stamp-previous_t-dt) > 0.5*dt)
        if reset:
            filtered_wz, filtered_ay = wz, ay
            kf.reset(filtered_wz)
            pose_history = []
        else:
            alpha = float(np.clip(imu_ema_alpha, 0.0, 1.0))
            filtered_wz = alpha*wz + (1.0-alpha)*filtered_wz
            filtered_ay = alpha*ay + (1.0-alpha)*filtered_ay
        delta = np.clip(steer_scale*row[names["steer"]] + steer_bias, -max_steer, max_steer)
        pose_vy = math.nan
        if use_pose_vy and all(name in names for name in ("x", "y", "yaw")):
            pose_history.append((float(stamp),float(row[names["x"]]),float(row[names["y"]])))
            while len(pose_history)>2 and stamp-pose_history[0][0]>pose_window_s:
                pose_history.pop(0)
            if len(pose_history)>=3:
                ph=np.asarray(pose_history,float); tt=ph[:,0]-ph[:,0].mean(); denom=float(tt@tt)
                if denom>1e-8:
                    world_vx=float(tt@(ph[:,1]-ph[:,1].mean())/denom)
                    world_vy=float(tt@(ph[:,2]-ph[:,2].mean())/denom)
                    yaw=float(row[names["yaw"]])
                    pose_vy=-math.sin(yaw)*world_vx+math.cos(yaw)*world_vy
        vy[i] = kf.update(row[names["vx"]], delta, filtered_wz, filtered_ay, pose_vy)
        yaw_rate[i] = kf.get_yaw_rate()
        previous_bag, previous_t = bag, stamp
    return vy, yaw_rate


def estimate_dataset_pose_only(samples, columns, dt, params=None,
                               steer_scale=1.1058064699, steer_bias=-0.0300696939,
                               max_steer=0.4788, imu_ema_alpha=0.25,
                               imu_wz_sign=1.0, use_pose_vy=True,
                               pose_window_s=0.12, **_unused):
    """Scalar runtime candidate: one Pacejka call + causal pose-vy update."""
    names = {str(name): i for i, name in enumerate(columns)}
    cfg = params or LateralVelocityKFParams(dt=dt)
    vy = np.zeros(len(samples)); yaw_rate = np.zeros(len(samples))
    estimate = 0.0; covariance = max(cfg.initial_var_vy, 1e-8)
    previous_bag = previous_t = None; filtered_wz = 0.0; pose_history = []
    dynamics = LateralVelocityKF(cfg)
    for i, row in enumerate(samples):
        bag, stamp = int(row[names["bag_id"]]), row[names["t"]]
        wz = imu_wz_sign*row[names["imu_wz"]]
        reset = previous_bag != bag or previous_t is None or abs(stamp-previous_t-dt) > .5*dt
        if reset:
            estimate=0.0; covariance=max(cfg.initial_var_vy,1e-8)
            filtered_wz=wz; pose_history=[]
        else:
            alpha=float(np.clip(imu_ema_alpha,0.,1.))
            filtered_wz=alpha*wz+(1.-alpha)*filtered_wz
        abs_vx=abs(row[names["vx"]])
        if abs_vx < cfg.low_speed_threshold:
            estimate=0.0
            covariance=min(covariance+cfg.process_var_vy,max(cfg.initial_var_vy,1e-8))
        else:
            delta=np.clip(steer_scale*row[names["steer"]]+steer_bias,-max_steer,max_steer)
            dvy,_,_=dynamics._dynamics(max(abs_vx,cfg.min_longitudinal_speed),delta,estimate,filtered_wz)
            estimate += dt*dvy
            covariance=max(covariance+cfg.process_var_vy,1e-8)
            measured_pose_vy=math.nan
            if use_pose_vy and all(name in names for name in ("x","y","yaw")):
                pose_history.append((float(stamp),float(row[names["x"]]),float(row[names["y"]])))
                while len(pose_history)>2 and stamp-pose_history[0][0]>pose_window_s:
                    pose_history.pop(0)
                if len(pose_history)>=3:
                    ph=np.asarray(pose_history);tt=ph[:,0]-ph[:,0].mean();denom=float(tt@tt)
                    if denom>1e-8:
                        wx=float(tt@(ph[:,1]-ph[:,1].mean())/denom)
                        wy=float(tt@(ph[:,2]-ph[:,2].mean())/denom);yaw=float(row[names["yaw"]])
                        measured_pose_vy=-math.sin(yaw)*wx+math.cos(yaw)*wy
            innovation=measured_pose_vy-estimate
            if math.isfinite(innovation) and abs(innovation)<=cfg.pose_vy_gate:
                gain=covariance/(covariance+max(cfg.measurement_var_pose_vy,1e-8))
                estimate+=gain*innovation;covariance=max((1.-gain)*covariance,1e-8)
            estimate=float(np.clip(estimate,-cfg.max_abs_vy,cfg.max_abs_vy))
        vy[i]=estimate;yaw_rate[i]=filtered_wz
        previous_bag,previous_t=bag,stamp
    return vy,yaw_rate


def estimate_dataset_inertial_pose(samples, columns, dt, params=None,
                                   imu_ema_alpha=.25, imu_wz_sign=1.,
                                   imu_ax_sign=1., imu_ay_sign=1.,
                                   imu_wz_bias=0.,imu_ax_bias=0.,imu_ay_bias=0., **_unused):
    """C++ parity: 7-state MCL+odom+IMU EKF used by runtime MPPI."""
    names={str(name):i for i,name in enumerate(columns)};n=len(samples)
    output=np.zeros(n);yaw_output=np.zeros(n);previous_bag=previous_t=None
    q=np.array((2e-5,2e-5,2e-5,2e-3,8e-3,2e-6,2e-6),float)
    initial=np.array((2e-2,2e-2,2e-2,1e-1,2.5e-1,1e-1,1e-1),float)
    measurement=np.array((4e-3,4e-3,8e-3,3e-2),float)
    low_speed=params.low_speed_threshold if params else .5
    max_vy=params.max_abs_vy if params else 2.
    state=np.zeros(7);P=np.diag(initial);filtered=np.zeros(3);I=np.eye(7)
    def angle(v):return (v+np.pi)%(2*np.pi)-np.pi
    for k,row in enumerate(samples):
        bag=int(row[names["bag_id"]]);stamp=row[names["t"]]
        raw=np.array((imu_wz_sign*row[names["imu_wz"]]-imu_wz_bias,
                      imu_ax_sign*row[names["imu_ax"]]-imu_ax_bias,
                      imu_ay_sign*row[names["imu_ay"]]-imu_ay_bias))
        reset=previous_bag!=bag or previous_t is None or abs(stamp-previous_t-dt)>.5*dt
        if reset:
            filtered=raw.copy();state[:4]=(row[names["x"]],row[names["y"]],row[names["yaw"]],row[names["vx"]]);state[4:]=0.;P=np.diag(initial)
        else:filtered=imu_ema_alpha*raw+(1.-imu_ema_alpha)*filtered
        yaw,vx,vy,bax,bay=state[2],state[3],state[4],state[5],state[6];r,ax,ay=filtered
        F=I.copy();F[0,2]+=dt*(-vx*np.sin(yaw)-vy*np.cos(yaw));F[0,3]+=dt*np.cos(yaw);F[0,4]-=dt*np.sin(yaw)
        F[1,2]+=dt*(vx*np.cos(yaw)-vy*np.sin(yaw));F[1,3]+=dt*np.sin(yaw);F[1,4]+=dt*np.cos(yaw)
        F[3,4]+=dt*r;F[3,5]-=dt;F[4,3]-=dt*r;F[4,6]-=dt
        state+=dt*np.array((vx*np.cos(yaw)-vy*np.sin(yaw),vx*np.sin(yaw)+vy*np.cos(yaw),r,ax-bax+r*vy,ay-bay-r*vx,0.,0.));state[2]=angle(state[2]);P=F@P@F.T+np.diag(q)
        z=(row[names["x"]],row[names["y"]],row[names["yaw"]],row[names["vx"]])
        for index,value,var in zip((0,1,2,3),z,measurement):
            innovation=angle(value-state[index]) if index==2 else value-state[index]
            gain=P[:,index]/(P[index,index]+var);old=P.copy();state+=gain*innovation;state[2]=angle(state[2]);P=old-np.outer(gain,old[index]);P=.5*(P+P.T);P[np.diag_indices(7)]=np.maximum(np.diag(P),1e-9)
        if abs(row[names["vx"]])<low_speed:state[4]=0.
        state[4]=np.clip(state[4],-max_vy,max_vy);output[k]=state[4];yaw_output[k]=r
        previous_bag,previous_t=bag,stamp
    return output,yaw_output
