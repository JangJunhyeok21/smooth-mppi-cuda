#!/usr/bin/env python3
"""Offline EKF/RTS smoother driven by the deployed classic MPPI model."""
import numpy as np


def wrap(v): return (v+np.pi)%(2*np.pi)-np.pi


def _pacejka(slip,fz,b,c,d,e):
    z=b*slip
    return fz*d*np.sin(c*np.arctan(z-e*(z-np.arctan(z))))


def _accelerations(state,applied,speed_reference,cfg):
    _,_,_,vx,vy,r=state;lf=float(cfg["l_f"]);lr=float(cfg["l_r"])
    mass=float(cfg["mass"]);iz=float(cfg["dynamic_mlp_I_z"]);safe=max(abs(vx),.5)
    af=applied-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe)
    fzf=mass*9.81*lr/(lf+lr);fzr=mass*9.81*lf/(lf+lr)
    fyf=_pacejka(af,fzf,*[float(cfg[f"dynamic_mlp_{q}_f"]) for q in "BCDE"])
    fyr=_pacejka(ar,fzr,*[float(cfg[f"dynamic_mlp_{q}_r"]) for q in "BCDE"])
    ax=np.clip(float(cfg["speed_servo_kp"])*(speed_reference-vx),
               float(cfg["min_accel"]),float(cfg["max_accel"]))
    ay=(fyf*np.cos(applied)+fyr)/mass
    rdot=(lf*fyf*np.cos(applied)-lr*fyr)/iz
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


def smooth_classic_segment(mcl_x,mcl_y,mcl_yaw,odom_vx,odom_vy,imu_r,imu_ax,imu_ay,
                           steer_cmd,speed_cmd,dt,cfg,causal_only=False):
    """Return smoothed state/acceleration and covariance diagnostics."""
    arrays=[np.asarray(q,float) for q in (mcl_x,mcl_y,mcl_yaw,odom_vx,odom_vy,
        imu_r,imu_ax,imu_ay,steer_cmd,speed_cmd)];n=len(arrays[0])
    if n<10 or any(len(q)!=n for q in arrays):raise ValueError("classic smoother needs >=10 aligned samples")
    x,y,yaw,ovx,ovy,gyro,iax,iay,steer,speed=arrays
    applied=np.empty(n);reference=np.empty(n);applied[0]=np.clip(float(cfg["kinematic_steer_scale"])*steer[0]+float(cfg["kinematic_steer_bias"]),-.55,.55);reference[0]=ovx[0]
    for k in range(1,n):
        target=np.clip(float(cfg["kinematic_steer_scale"])*steer[k-1]+float(cfg["kinematic_steer_bias"]),-.55,.55)
        rate=np.clip((target-applied[k-1])/max(float(cfg["steer_servo_time_constant"]),1e-3),-float(cfg["actuator_max_steer_rate"]),float(cfg["actuator_max_steer_rate"]))
        applied[k]=np.clip(applied[k-1]+rate*dt,-.55,.55)
        tau=float(cfg["speed_reference_accel_time_constant"] if speed[k-1]>=reference[k-1] else cfg["speed_reference_brake_time_constant"])
        rr=np.clip((speed[k-1]-reference[k-1])/max(tau,1e-3),-float(cfg["actuator_max_speed_reference_rate"]),float(cfg["actuator_max_speed_reference_rate"]))
        reference[k]=reference[k-1]+rr*dt
    filtered=np.empty((n,6));predicted=np.empty((n,6));pf=[];pp=[];trans=[]
    state=np.array((x[0],y[0],yaw[0],ovx[0],ovy[0] if np.isfinite(ovy[0]) else 0.,
                    gyro[0] if np.isfinite(gyro[0]) else 0.));P=np.diag((.01,.01,.005,.03,.12,.02))
    Q=np.diag((2e-5,2e-5,2e-5,3e-3,1e-2,3e-3))
    R=np.diag((.015**2,.015**2,.01**2,.025**2,.12**2,.02**2,.35**2,.35**2))
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
    smoothed=filtered.copy();ps=[q.copy() for q in pf]
    if not causal_only:
        for k in range(n-2,-1,-1):
            G=pf[k]@trans[k].T@np.linalg.pinv(pp[k+1])
            delta=smoothed[k+1]-predicted[k+1];delta[2]=wrap(delta[2])
            smoothed[k]=filtered[k]+G@delta;smoothed[k,2]=wrap(smoothed[k,2])
            ps[k]=pf[k]+G@(ps[k+1]-pp[k+1])@G.T
    acceleration=np.asarray([_accelerations(smoothed[k],applied[k],reference[k],cfg)[:2] for k in range(n)])
    residual=np.c_[smoothed[:,0]-x,smoothed[:,1]-y,wrap(smoothed[:,2]-yaw),smoothed[:,3]-ovx,smoothed[:,4]-ovy,smoothed[:,5]-gyro,acceleration[:,0]-iax,acceleration[:,1]-iay]
    return {"state":smoothed,"acceleration":acceleration,"applied_steer":applied,
            "filtered_state":filtered,"predicted_state":predicted,
            "speed_reference":reference,"valid_mask":np.isfinite(smoothed).all(1),
            "residual":residual,"covariance_diagonal":np.asarray([np.diag(q) for q in ps]),
            "metrics":{"method":"classic_mppi_forward_ekf" if causal_only else "classic_mppi_ekf_rts","rmse":np.sqrt(np.mean(residual**2,axis=0)).tolist(),
            "measurement_order":["mcl_x","mcl_y","mcl_yaw","odom_vx","odom_vy","imu_r","imu_ax","imu_ay"],
            "pacejka_source":"config dynamic_mlp B,C,D,E front/rear"}}


def filter_classic_segment(mcl_x,mcl_y,mcl_yaw,odom_vx,odom_vy,imu_r,imu_ax,imu_ay,
                           steer_cmd,speed_cmd,dt,cfg):
    """Run the causal MPPI-classic-model EKF without a backward pass."""
    return smooth_classic_segment(mcl_x,mcl_y,mcl_yaw,odom_vx,odom_vy,imu_r,imu_ax,
        imu_ay,steer_cmd,speed_cmd,dt,cfg,causal_only=True)
