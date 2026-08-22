#!/usr/bin/env python3
"""Causal classic MPPI-model EKF used by Step 1 and runtime parity checks."""
import numpy as np


def wrap(v): return (v+np.pi)%(2*np.pi)-np.pi


def _pacejka(slip,fz,b,c,d,e):
    z=b*slip
    return fz*d*np.sin(c*np.arctan(z-e*(z-np.arctan(z))))


def _dynamic_speed_blend(vx):
    """C1 blend: no-slip kinematic below 0.2 m/s, Pacejka above 0.5 m/s."""
    u=np.clip((abs(vx)-.2)/.3,0.,1.)
    return u*u*(3.-2.*u)


def _accelerations(state,applied,speed_reference,cfg):
    _,_,_,vx,vy,r=state;lf=float(cfg["l_f"]);lr=float(cfg["l_r"])
    mass=float(cfg["mass"]);iz=float(cfg["dynamic_mlp_I_z"]);safe=max(abs(vx),.5)
    af=applied-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe)
    fzf=mass*9.81*lr/(lf+lr);fzr=mass*9.81*lf/(lf+lr)
    fyf=_pacejka(af,fzf,*[float(cfg[f"dynamic_mlp_{q}_f"]) for q in "BCDE"])
    fyr=_pacejka(ar,fzr,*[float(cfg[f"dynamic_mlp_{q}_r"]) for q in "BCDE"])
    ax=np.clip(float(cfg["speed_servo_kp"])*(speed_reference-vx),
               float(cfg["min_accel"]),float(cfg["max_accel"]))
    dynamic_ay=(fyf*np.cos(applied)+fyr)/mass
    dynamic_rdot=(lf*fyf*np.cos(applied)-lr*fyr)/iz
    blend=_dynamic_speed_blend(vx)
    # Below the dynamic-model validity range, drive vy to zero and yaw rate to
    # the no-slip bicycle value.  The body identity vy_dot=ay-vx*r makes this
    # ay choice decay vy without leaving a fictitious stationary tire force.
    low_speed_tau=.1
    kinematic_r=vx*np.tan(applied)/max(lf+lr,1e-6)
    kinematic_ay=vx*r-vy/low_speed_tau
    kinematic_rdot=(kinematic_r-r)/low_speed_tau
    ay=blend*dynamic_ay+(1.-blend)*kinematic_ay
    rdot=blend*dynamic_rdot+(1.-blend)*kinematic_rdot
    return np.array((ax,ay,rdot))


def _step(state,applied,speed_reference,dt,cfg):
    x,y,yaw,vx,vy,r=state;ax,ay,rdot=_accelerations(state,applied,speed_reference,cfg)
    nvx=vx+(ax+vy*r)*dt;nvy=vy+(ay-vx*r)*dt;nr=r+rdot*dt
    return np.array((x+(nvx*np.cos(yaw)-nvy*np.sin(yaw))*dt,
        y+(nvx*np.sin(yaw)+nvy*np.cos(yaw))*dt,wrap(yaw+nr*dt),nvx,nvy,nr))


def _jacobian(function,state):
    base=function(state);jac=np.empty((len(base),len(state)))
    for j in range(len(state)):
        eps=1e-5*max(1.,abs(state[j]));other=state.copy();other[j]+=eps
        delta=function(other)-base
        if len(base)>=3:delta[2]=wrap(delta[2])
        jac[:,j]=delta/eps
    return jac


def _actuator_sequences(odom_vx,steer_cmd,speed_cmd,dt,cfg):
    n=len(odom_vx);applied=np.empty(n);reference=np.empty(n)
    applied[0]=np.clip(float(cfg["kinematic_steer_scale"])*steer_cmd[0]
        +float(cfg["kinematic_steer_bias"]),-.55,.55)
    reference[0]=odom_vx[0]
    for k in range(1,n):
        target=np.clip(float(cfg["kinematic_steer_scale"])*steer_cmd[k-1]
            +float(cfg["kinematic_steer_bias"]),-.55,.55)
        rate=np.clip((target-applied[k-1])/max(float(cfg["steer_servo_time_constant"]),1e-3),
            -float(cfg["actuator_max_steer_rate"]),float(cfg["actuator_max_steer_rate"]))
        applied[k]=np.clip(applied[k-1]+rate*dt,-.55,.55)
        tau=float(cfg["speed_reference_accel_time_constant"]
                  if speed_cmd[k-1]>=reference[k-1]
                  else cfg["speed_reference_brake_time_constant"])
        rr=np.clip((speed_cmd[k-1]-reference[k-1])/max(tau,1e-3),
            -float(cfg["actuator_max_speed_reference_rate"]),
            float(cfg["actuator_max_speed_reference_rate"]))
        reference[k]=reference[k-1]+rr*dt
    return applied,reference


def rollout_classic_segment(initial_state,odom_vx,steer_cmd,speed_cmd,dt,cfg):
    """Free open-loop classic rollout with no measurement correction."""
    ovx,steer,speed=(np.asarray(value,float) for value in
                     (odom_vx,steer_cmd,speed_cmd))
    if not len(ovx) or len(steer)!=len(ovx) or len(speed)!=len(ovx):
        raise ValueError("open-loop rollout needs aligned non-empty inputs")
    applied,reference=_actuator_sequences(ovx,steer,speed,dt,cfg)
    states=np.empty((len(ovx),6));states[0]=np.asarray(initial_state,float)
    for k in range(1,len(states)):
        states[k]=_step(states[k-1],applied[k],reference[k],dt,cfg)
    return {"state":states,"applied_steer":applied,"speed_reference":reference}


def filter_classic_segment(mcl_x,mcl_y,mcl_yaw,odom_vx,odom_vy,imu_r,imu_ax,imu_ay,
                           steer_cmd,speed_cmd,dt,cfg):
    """Return causal filtered state/acceleration and covariance diagnostics."""
    arrays=[np.asarray(q,float) for q in (mcl_x,mcl_y,mcl_yaw,odom_vx,odom_vy,
        imu_r,imu_ax,imu_ay,steer_cmd,speed_cmd)];n=len(arrays[0])
    if n<10 or any(len(q)!=n for q in arrays):raise ValueError("classic filter needs >=10 aligned samples")
    x,y,yaw,ovx,ovy,gyro,iax,iay,steer,speed=arrays
    applied,reference=_actuator_sequences(ovx,steer,speed,dt,cfg)
    filtered=np.empty((n,6));predicted=np.empty((n,6));pf=[];pp=[];trans=[]
    initial_var=np.asarray(cfg.get("classic_kf_initial_var",
        (.01,.01,.005,.03,.12,.02)),float)
    process_var=np.asarray(cfg.get("classic_kf_process_var",
        (2e-5,2e-5,2e-5,3e-3,1e-2,3e-3)),float)
    measurement_var=np.asarray(cfg.get("classic_kf_measurement_var",
        (.015**2,.015**2,.01**2,.025**2,.12**2,.02**2,.35**2,.35**2)),float)
    if initial_var.shape!=(6,) or process_var.shape!=(6,) or measurement_var.shape!=(8,):
        raise ValueError("classic KF variance lengths must be P0=6, Q=6, R=8")
    if np.any(initial_var<=0.) or np.any(process_var<=0.) or np.any(measurement_var<=0.):
        raise ValueError("classic KF variances must all be positive")
    state=np.array((x[0],y[0],yaw[0],ovx[0],ovy[0] if np.isfinite(ovy[0]) else 0.,
                    gyro[0] if np.isfinite(gyro[0]) else 0.));P=np.diag(initial_var)
    Q=np.diag(process_var)
    R=np.diag(measurement_var)
    I=np.eye(6)
    for k in range(n):
        if k:
            f=lambda s:_step(s,applied[k],reference[k],dt,cfg);F=_jacobian(f,state)
            state=f(state);P=F@P@F.T+Q;trans.append(F)
        predicted[k]=state;pp.append(P.copy())
        z=np.array((x[k],y[k],yaw[k],ovx[k],ovy[k],gyro[k],iax[k],iay[k]))
        def observe(s):
            acc=_accelerations(s,applied[k],reference[k],cfg)
            return np.r_[s[:6],acc[:2]]
        h=observe(state);H=_jacobian(observe,state);innovation=z-h;innovation[2]=wrap(innovation[2])
        finite=np.isfinite(z);innovation[~finite]=0.
        effective_R=R.copy();effective_R[np.diag_indices_from(effective_R)]=np.where(finite,np.diag(R),1e9)
        S=H@P@H.T+effective_R;K=np.linalg.solve(S,H@P).T
        state=state+K@innovation;state[2]=wrap(state[2])
        P=(I-K@H)@P@(I-K@H).T+K@effective_R@K.T;P=.5*(P+P.T)
        filtered[k]=state;pf.append(P.copy())
    acceleration=np.asarray([_accelerations(filtered[k],applied[k],reference[k],cfg)[:2] for k in range(n)])
    residual=np.c_[filtered[:,0]-x,filtered[:,1]-y,wrap(filtered[:,2]-yaw),filtered[:,3]-ovx,filtered[:,4]-ovy,filtered[:,5]-gyro,acceleration[:,0]-iax,acceleration[:,1]-iay]
    return {"state":filtered,"acceleration":acceleration,"applied_steer":applied,
            "filtered_state":filtered,"predicted_state":predicted,
            "speed_reference":reference,"valid_mask":np.isfinite(filtered).all(1),
            "residual":residual,"covariance_diagonal":np.asarray([np.diag(q) for q in pf]),
            "metrics":{"method":"classic_mppi_forward_ekf","rmse":np.sqrt(np.mean(residual**2,axis=0)).tolist(),
            "measurement_order":["mcl_x","mcl_y","mcl_yaw","odom_vx","odom_vy","imu_r","imu_ax","imu_ay"],
            "pacejka_source":"config dynamic_mlp B,C,D,E front/rear"}}
