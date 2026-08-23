#!/usr/bin/env python3
"""Classic-model regression implementation used by numbered Step 3.

This script never uses reconstructed diagnostic CSV data.  It preserves the
bag-level train/validation/test split and selects one *global* eight-parameter
classic model by validation open-loop rollout error.
"""
from pathlib import Path
import json, os
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import differential_evolution, least_squares, minimize, minimize_scalar
from scipy.signal import savgol_filter
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DYNAMIC_SOURCE_DATA", ROOT /
    "model_tuning/data/ifac0810_0819_autonomous_physics_clean"))
OUT = Path(os.environ.get("DYNAMIC_REGRESSION_OUT", ROOT / "model_tuning/results/dynamic_40ms_regression"))
SEED = 31
HORIZON = 25                    # 1.0 s at one 40 ms MPPI knot
# The archive now contains more than 200 discontinuity-safe segments. Keep the
# optimizer balanced per bag, but cap redundant windows so a full rerun remains
# practical after adding a day of data.
MAX_PER_BAG = 80
WARMUP_SAMPLES = 40             # 0.8 s at source 50 Hz
ADAM_RESTARTS = int(os.environ.get("CLASSIC_ADAM_RESTARTS","3"))
ADAM_STEPS = int(os.environ.get("CLASSIC_ADAM_STEPS","600"))
SURROGATE_SAMPLES = int(os.environ.get("CLASSIC_SURROGATE_SAMPLES","400"))
SURROGATE_PROPOSALS = int(os.environ.get("CLASSIC_SURROGATE_PROPOSALS","40000"))
DE_POPSIZE = int(os.environ.get("CLASSIC_DE_POPSIZE","6"))
DE_MAXITER = int(os.environ.get("CLASSIC_DE_MAXITER","35"))

NAMES = ("B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r")
BOUNDS = np.asarray((
    (.2, 30.), (.5, 2.5), (.05, 3.5), (-1., 1.),
    (.2, 30.), (.5, 2.5), (.05, 3.5), (-1., 1.)), dtype=np.float64)
REFERENCE = np.asarray((6., 1.3, 1.0, 0., 6., 1.3, 1.0, 0.))
FROZEN_MLP_BIN = os.environ.get("FROZEN_MLP_BIN")
CLASSIC_RESIDUAL_PENALTY = float(os.environ.get("CLASSIC_RESIDUAL_PENALTY", "0.0001"))
I_Z_MIN = 0.005
I_Z_MAX = 0.5
SHOW_PLOTS = True
INTERACTIVE_BAG_INSPECTOR = True
TRAJECTORY_TIME_LABEL_INTERVAL_S = 1.0
EVALUATE_ONLY = False
EVALUATION_PARAMS_PATH = OUT / "params.json"
APPLY_ACCEPTED_PARAMS_TO_YAML = False
USE_VALIDATION_TEST_SPLIT = False
TRAIN_EVALUATION_BAG_INDEX = -1
MAX_POSITION_STEP_20MS = 0.15
MAX_YAW_STEP_20MS = 0.12
AUTO_FIT_POSITION_SPEED_SCALE = True
POSITION_SPEED_SCALE_BOUNDS = (0.70, 1.20)
VX_LOSS_WEIGHT = 1.0
VY_LOSS_WEIGHT = 1.0
YAW_RATE_LOSS_WEIGHT = 1.0
POSITION_LOSS_WEIGHT = 0.8
YAW_TRAJECTORY_LOSS_WEIGHT = 0.5
# Additional teacher-forced first-transition term.  Zero preserves the
# historical recursive-only objective; joint Step 3/4 configures it explicitly.
ONE_STEP_LOSS_WEIGHT = 0.0
# Pose error across every recursive knot.  Endpoint and one-step terms remain
# separate so callers can emphasize local, full-trajectory, and terminal fit.
FULL_TRAJECTORY_POSE_LOSS_WEIGHT = 0.0
# CVaR-style endpoint penalty.  It prevents improvements on easy windows from
# hiding a worse high-error tail, which is the safety-critical MPPI regime.
ENDPOINT_TAIL_LOSS_WEIGHT = 0.0
ENDPOINT_TAIL_QUANTILE = 0.90
GT_CONSISTENCY_MODE = "adjust_states_to_pose"
VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S = 0.20
# Longitudinal load transfer. Zero preserves the static axle loads.
LOAD_TRANSFER_H_CG_M = 0.0


class RegressionData(dict):
    """Dict with the small ``NpzFile.files`` compatibility surface."""
    @property
    def files(self):
        return list(self.keys())


def pose_derived_states(samples,names):
    """Return smooth body vx/vy/yaw-rate derived only from raw MCL pose."""
    result=np.full((len(samples),3),np.nan,dtype=np.float64)
    segments=(samples[:,names["bag_id"]].astype(int)
              if "bag_id" in names else np.zeros(len(samples),int))
    for segment in np.unique(segments):
        ii=np.flatnonzero(segments==segment)
        if len(ii)<5:continue
        t=samples[ii,names["t"]];median_dt=float(np.median(np.diff(t)))
        window=max(5,int(round(VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S/median_dt))|1)
        window=min(window,len(ii) if len(ii)%2 else len(ii)-1)
        yaw=np.unwrap(samples[ii,names["yaw"]])
        x=savgol_filter(samples[ii,names["x"]],window,min(3,window-2),mode="interp")
        y=savgol_filter(samples[ii,names["y"]],window,min(3,window-2),mode="interp")
        smooth_yaw=savgol_filter(yaw,window,min(3,window-2),mode="interp")
        dx=np.gradient(x,t,edge_order=2);dy=np.gradient(y,t,edge_order=2)
        result[ii,0]=dx*np.cos(smooth_yaw)+dy*np.sin(smooth_yaw)
        result[ii,1]=-dx*np.sin(smooth_yaw)+dy*np.cos(smooth_yaw)
        result[ii,2]=np.gradient(smooth_yaw,t,edge_order=2)
    return result


def plot_vy_reference_point_diagnostic(data,config):
    """Compare rear-axle MCL vy with its CG conversion against raw KF vy."""
    pose=np.asarray(data["mcl_pose"],float);bag=np.asarray(data["bag_id"],int)
    raw_state=np.asarray(data["features"][:,:3],float)
    derived=np.full((len(pose),3),np.nan,float)
    for bag_id in np.unique(bag):
        ii=np.flatnonzero(bag==bag_id)
        if len(ii)<5:continue
        window=max(5,int(round(VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S/.02))|1)
        window=min(window,len(ii) if len(ii)%2 else len(ii)-1)
        yaw=np.unwrap(pose[ii,2])
        x=savgol_filter(pose[ii,0],window,min(3,window-2),mode="interp")
        y=savgol_filter(pose[ii,1],window,min(3,window-2),mode="interp")
        smooth_yaw=savgol_filter(yaw,window,min(3,window-2),mode="interp")
        dx=np.gradient(x,.02,edge_order=2);dy=np.gradient(y,.02,edge_order=2)
        derived[ii,0]=dx*np.cos(smooth_yaw)+dy*np.sin(smooth_yaw)
        derived[ii,1]=-dx*np.sin(smooth_yaw)+dy*np.cos(smooth_yaw)
        derived[ii,2]=np.gradient(smooth_yaw,.02,edge_order=2)
    l_r=float(config["l_r"])
    rear_vy=derived[:,1]
    cg_vy=rear_vy+l_r*derived[:,2]
    opposite_vy=rear_vy-l_r*derived[:,2]
    valid=np.asarray(data["valid"],bool)&np.isfinite(derived).all(1)&(np.abs(raw_state[:,0])>.5)
    reports=[]
    for bag_id in np.unique(bag[valid]):
        mask=valid&(bag==bag_id)
        rear_rmse=float(np.sqrt(np.mean((rear_vy[mask]-raw_state[mask,1])**2)))
        cg_rmse=float(np.sqrt(np.mean((cg_vy[mask]-raw_state[mask,1])**2)))
        opposite_rmse=float(np.sqrt(np.mean(
            (opposite_vy[mask]-raw_state[mask,1])**2)))
        source=(Path(str(data["source_paths"][bag_id])).name
                if "source_paths" in data.files and bag_id<len(data["source_paths"])
                else f"bag_id={bag_id}")
        reports.append({"bag_id":int(bag_id),"source":source,"samples":int(mask.sum()),
            "rear_axle_vy_rmse_mps":rear_rmse,"cg_corrected_vy_rmse_mps":cg_rmse,
            "opposite_sign_vy_rmse_mps":opposite_rmse,
            "rmse_reduction_mps":rear_rmse-cg_rmse})
    if not reports:return None
    mask=valid
    rear_all=float(np.sqrt(np.mean((rear_vy[mask]-raw_state[mask,1])**2)))
    cg_all=float(np.sqrt(np.mean((cg_vy[mask]-raw_state[mask,1])**2)))
    opposite_all=float(np.sqrt(np.mean(
        (opposite_vy[mask]-raw_state[mask,1])**2)))
    representative=max(reports,key=lambda item:abs(item["rmse_reduction_mps"]))
    ii=np.flatnonzero(valid&(bag==representative["bag_id"]));time=.02*np.arange(len(ii))
    fig,axes=plt.subplots(2,2,figsize=(16,10),constrained_layout=True)
    axes[0,0].plot(time,raw_state[ii,1],"k-",lw=1.8,label="raw KF vy (model state)")
    axes[0,0].plot(time,rear_vy[ii],color="tab:blue",alpha=.8,
                   label="MCL derivative vy at base_link/rear axle")
    axes[0,0].plot(time,cg_vy[ii],color="tab:red",alpha=.85,
                   label="MCL vy + l_r·yaw_rate (CG conversion)")
    axes[0,0].plot(time,opposite_vy[ii],color="tab:green",alpha=.8,
                   label="MCL vy - l_r·yaw_rate (sign/reference check)")
    axes[0,0].set(title=f"Representative {representative['source']}",xlabel="time [s]",
                  ylabel="vy [m/s]");axes[0,0].grid(alpha=.3);axes[0,0].legend()
    take=np.flatnonzero(valid)[::max(1,int(valid.sum()/5000))]
    axes[0,1].scatter(raw_state[take,1],rear_vy[take],s=5,alpha=.25,label="rear axle")
    axes[0,1].scatter(raw_state[take,1],cg_vy[take],s=5,alpha=.25,label="CG corrected")
    axes[0,1].scatter(raw_state[take,1],opposite_vy[take],s=5,alpha=.25,
                      label="opposite sign")
    limits=np.nanquantile(np.r_[raw_state[take,1],rear_vy[take],cg_vy[take],
                                opposite_vy[take]],[.01,.99])
    axes[0,1].plot(limits,limits,"k--",lw=1,label="identity")
    axes[0,1].set(xlabel="raw KF vy [m/s]",ylabel="MCL-derived vy [m/s]",
                  title="All valid moving samples");axes[0,1].grid(alpha=.3);axes[0,1].legend()
    labels=[str(item["bag_id"]) for item in reports];x=np.arange(len(reports));width=.27
    axes[1,0].bar(x-width,[item["rear_axle_vy_rmse_mps"] for item in reports],width,
                  label="rear axle MCL vy")
    axes[1,0].bar(x,[item["cg_corrected_vy_rmse_mps"] for item in reports],width,
                  label="+ l_r·yaw_rate")
    axes[1,0].bar(x+width,[item["opposite_sign_vy_rmse_mps"] for item in reports],width,
                  label="- l_r·yaw_rate")
    axes[1,0].set(xticks=x,xticklabels=labels,xlabel="bag id",ylabel="RMSE [m/s]",
                  title="Per-bag comparison");axes[1,0].grid(alpha=.3,axis="y");axes[1,0].legend()
    correction=l_r*derived[ii,2]
    axes[1,1].plot(time,derived[ii,2],label="MCL-derived yaw rate [rad/s]")
    axes[1,1].plot(time,correction,label="l_r·yaw_rate correction [m/s]")
    axes[1,1].set(xlabel="time [s]",title=f"Lever-arm correction, l_r={l_r:g} m")
    axes[1,1].grid(alpha=.3);axes[1,1].legend()
    fig.suptitle(f"MCL vy reference-point diagnostic | RMSE rear/+l_r*r/-l_r*r = "
                 f"{rear_all:.4f}/{cg_all:.4f}/{opposite_all:.4f} m/s",fontsize=15)
    output=OUT/"vy_rear_axle_vs_cg_correction.png";fig.savefig(output,dpi=180)
    if SHOW_PLOTS:fig.show();fig.canvas.draw_idle()
    else:plt.close(fig)
    report={"l_r_m":l_r,"overall_rear_axle_vy_rmse_mps":rear_all,
            "overall_cg_corrected_vy_rmse_mps":cg_all,
            "overall_opposite_sign_vy_rmse_mps":opposite_all,
            "overall_rmse_reduction_mps":rear_all-cg_all,"bags":reports}
    (OUT/"vy_rear_axle_vs_cg_correction.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"vy reference-point diagnostic: rear/+l_r*r/-l_r*r RMSE "
          f"{rear_all:.6f}/{cg_all:.6f}/{opposite_all:.6f} m/s; {output}")
    return output


def states_integrated_pose(samples,names,states):
    """Integrate fixed body states while retaining each MCL segment's origin."""
    result=np.full((len(samples),3),np.nan,dtype=np.float64)
    segments=(samples[:,names["bag_id"]].astype(int)
              if "bag_id" in names else np.zeros(len(samples),int))
    for segment in np.unique(segments):
        ii=np.flatnonzero(segments==segment)
        if not len(ii):continue
        result[ii[0]]=samples[ii[0],[names["x"],names["y"],names["yaw"]]]
        t=samples[ii,names["t"]]
        for local in range(1,len(ii)):
            previous=ii[local-1];current=ii[local];dt=t[local]-t[local-1]
            velocity=.5*(states[previous,:2]+states[current,:2])
            dyaw=.5*(states[previous,2]+states[current,2])*dt
            yaw_mid=result[previous,2]+.5*dyaw
            result[current,0]=result[previous,0]+(
                velocity[0]*np.cos(yaw_mid)-velocity[1]*np.sin(yaw_mid))*dt
            result[current,1]=result[previous,1]+(
                velocity[0]*np.sin(yaw_mid)+velocity[1]*np.cos(yaw_mid))*dt
            result[current,2]=result[previous,2]+dyaw
    return result


def load_regression_data(path,config):
    """Accept either the merged feature archive or one direct Step-1 NPZ."""
    path=Path(path).expanduser().resolve()
    if path.is_dir():
        paths=[]
        for candidate in sorted(path.glob("*.npz")):
            with np.load(candidate) as probe:
                if {"samples","columns","dt"}.issubset(probe.files):
                    paths.append(candidate)
                else:
                    print(f"Skipping non-Step-1 NPZ: {candidate}")
        if len(paths)<3:
            raise RuntimeError(f"{path}: need at least three Step-1 NPZ files for "
                               "bag-disjoint train/validation/test; found {len(paths)}")
        converted=[]
        for bag_index,candidate in enumerate(paths):
            item,_=load_regression_data(candidate,config)
            item["bag_id"][:]=bag_index
            converted.append(item)
        # Deterministic chronological bag split: reserve at least one complete
        # bag each for validation and test. No segment from a held-out bag is
        # visible to the optimizer.
        count=len(converted);train_count=max(1,int(np.floor(.6*count)))
        validation_count=max(1,int(np.floor(.2*count)))
        if train_count+validation_count>=count:
            train_count=count-2;validation_count=1
        for index,item in enumerate(converted):
            split_id=0 if index<train_count else 1 if index<train_count+validation_count else 2
            item["split"][:]=split_id
        fields=("features","observations","teacher_state","target_pose","mcl_pose",
                "bag_id","split","valid")
        merged=RegressionData({field:np.concatenate([item[field] for item in converted])
                               for field in fields})
        merged["source_paths"]=np.asarray([str(candidate) for candidate in paths])
        merged["source_contract"]=np.array(
            "multiple direct Step-1 NPZ; bag-disjoint chronological 60/20/20 split")
        return merged,(f"{len(paths)} direct Step-1 NPZ files; bag-disjoint "
                       f"train={train_count}, validation={validation_count}, "
                       f"test={count-train_count-validation_count}")
    archive=np.load(path)
    if "features" in archive.files:
        return archive,"merged feature archive"
    if not {"samples","columns","dt"}.issubset(archive.files):
        raise RuntimeError(f"{path}: expected features or Step-1 samples/columns/dt")
    samples=np.asarray(archive["samples"],float)
    names={str(value):index for index,value in enumerate(archive["columns"])}
    required=("t","x","y","yaw","steer","speed_cmd","imu_wz","imu_ax","imu_ay",
              "kf_x","kf_y","kf_yaw","kf_vx","kf_vy","kf_yaw_rate")
    missing=[name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"{path}: rerun Step 1; missing fields {missing}")
    count=len(samples);feature=np.zeros((count,20),np.float32)
    feature[:,:3]=samples[:,[names["kf_vx"],names["kf_vy"],names["kf_yaw_rate"]]]
    feature[:,3]=samples[:,names["steer"]]
    feature[:,4]=samples[:,names["speed_cmd"]]
    # These fields preserve the established 20-D contract. Classic rollouts
    # reconstruct actuator state candidate-by-candidate from the warm-up
    # commands, so feature[:,5] is diagnostic rather than an optimizer input.
    max_steer=float(config["max_steer"])
    steer_scale=float(config["kinematic_steer_scale"])
    steer_bias=float(config["kinematic_steer_bias"])
    feature[0,5]=np.clip(steer_scale*feature[0,3]+steer_bias,
                         -max_steer,max_steer)
    times=samples[:,names["t"]]
    steer_tau=max(float(config["steer_servo_time_constant"]),1e-3)
    steer_rate=float(config["actuator_max_steer_rate"])
    for index in range(1,count):
        dt=max(0.,times[index]-times[index-1])
        target=np.clip(steer_scale*feature[index-1,3]+steer_bias,
                       -max_steer,max_steer)
        rate=np.clip((target-feature[index-1,5])/steer_tau,
                     -steer_rate,steer_rate)
        feature[index,5]=np.clip(
            feature[index-1,5]+rate*dt,-max_steer,max_steer)
    feature[:,6]=feature[:,3]-np.r_[feature[0,3],feature[:-1,3]]
    for index in range(count):
        for history_index in range(5):
            source=max(0,index-4+history_index)
            feature[index,10+2*history_index:12+2*history_index]=feature[source,3:5]
    signs=(np.asarray(archive["imu_axis_signs"],float)
           if "imu_axis_signs" in archive.files else np.ones(3))
    observations=np.c_[
        signs[1]*samples[:,names["imu_ax"]]-float(config.get("imu_ax_bias",0.)),
        signs[2]*samples[:,names["imu_ay"]]-float(config.get("imu_ay_bias",0.)),
        signs[0]*samples[:,names["imu_wz"]]-float(config.get("imu_wz_bias",0.))].astype(np.float32)
    # One direct bag still needs honest held-out windows. Use contiguous 60/20/20
    # time blocks; starts() additionally rejects horizons crossing a boundary.
    first=int(.6*count);second=int(.8*count)
    split=np.zeros(count,np.int8);split[first:second]=1;split[second:]=2
    raw_state=feature[:,:3].astype(np.float64)
    raw_pose=samples[:,[names["x"],names["y"],names["yaw"]]].astype(np.float64)
    if GT_CONSISTENCY_MODE=="adjust_states_to_pose":
        teacher_state=pose_derived_states(samples,names);target_pose=raw_pose.copy()
    elif GT_CONSISTENCY_MODE=="adjust_pose_to_states":
        teacher_state=raw_state.copy()
        target_pose=states_integrated_pose(samples,names,teacher_state)
    elif GT_CONSISTENCY_MODE=="none":
        teacher_state=raw_state.copy();target_pose=raw_pose.copy()
    else:
        raise ValueError(f"invalid GT_CONSISTENCY_MODE={GT_CONSISTENCY_MODE!r}; expected "
                         "'adjust_states_to_pose', 'adjust_pose_to_states', or 'none'")
    valid=(np.isfinite(feature).all(1)&np.isfinite(observations).all(1)
           &np.isfinite(teacher_state).all(1)&np.isfinite(target_pose).all(1))
    state_rmse=np.sqrt(np.mean((teacher_state[valid]-raw_state[valid])**2,axis=0))
    print(f"{Path(path).name}: mode={GT_CONSISTENCY_MODE}; target-vs-KF state RMSE "
          f"vx/vy/r={state_rmse[0]:.4f}/{state_rmse[1]:.4f}/{state_rmse[2]:.4f}")
    data=RegressionData(features=feature,observations=observations,
        teacher_state=teacher_state.astype(np.float32),target_pose=target_pose,
        mcl_pose=raw_pose,
        bag_id=np.zeros(count,np.int32),split=split,valid=valid,
        source_path=np.array(str(Path(path).resolve())),
        source_contract=np.array(
            f"direct Step-1 NPZ; consistency mode={GT_CONSISTENCY_MODE}"))
    archive.close()
    return data,"direct Step-1 NPZ with temporal 60/20/20 split"


def load_frozen_mlp(path):
    if not path:
        return None
    raw=np.fromfile(path,dtype='<f4');offset=0;layers=[]
    for output_dim,input_dim in ((64,22),(32,64),(3,32)):
        count=output_dim*input_dim;weight=raw[offset:offset+count].reshape(output_dim,input_dim);offset+=count
        bias=raw[offset:offset+output_dim];offset+=output_dim;layers.append((weight,bias))
    mean=raw[offset:offset+22];std=raw[offset+22:offset+44]
    if len(raw)!=3695:raise ValueError(f"invalid frozen MLP binary: {len(raw)} floats")
    return layers,mean,std


FROZEN_MLP = load_frozen_mlp(FROZEN_MLP_BIN)


def frozen_mlp_forward(feature):
    if FROZEN_MLP is None:return np.zeros((len(feature),3))
    layers,mean,std=FROZEN_MLP;value=(feature-mean)/std
    for index,(weight,bias) in enumerate(layers):
        value=value@weight.T+bias
        if index<2:value=np.maximum(value,0.)
    return value


def starts(data, split):
    features, bag, splits, valid = (data[k] for k in
                                    ("features", "bag_id", "split", "valid"))
    result = []
    for bag_id in np.unique(bag[splits == split]):
        candidate = np.asarray([
            index for index in range(WARMUP_SAMPLES, len(features)-2*HORIZON)
            if bag[index] == bag_id and splits[index] == split
            and np.all(bag[index-WARMUP_SAMPLES:index+1] == bag_id)
            and valid[index:index+2*HORIZON+1].all()
            and np.all(splits[index:index+2*HORIZON+1] == split)
            and np.all(bag[index:index+2*HORIZON+1] == bag_id)
            and ("mcl_pose" not in data.files or np.max(np.linalg.norm(
                np.diff(data["mcl_pose"][index:index+2*HORIZON+1,:2],axis=0),axis=1))
                <= MAX_POSITION_STEP_20MS)
            and ("mcl_pose" not in data.files or np.max(np.abs(
                (np.diff(data["mcl_pose"][index:index+2*HORIZON+1,2])+np.pi)
                %(2*np.pi)-np.pi)) <= MAX_YAW_STEP_20MS)
            and np.mean(np.abs(features[index:index+2*HORIZON, 0])) > .5], int)
        if len(candidate) > MAX_PER_BAG:
            candidate = candidate[np.linspace(
                0, len(candidate)-1, MAX_PER_BAG).astype(int)]
        result.extend(candidate[::3])
    return np.asarray(result, int)


def rollout_numpy(parameters, data, window_starts, config, return_residual=False,
                  return_acceleration=False):
    feature = data["features"]
    max_steer=float(config["max_steer"])
    state = feature[window_starts, :3].astype(np.float64).copy()
    if "teacher_state" in data.files:
        state[:]=data["teacher_state"][window_starts]
    applied_steer = np.clip(
        feature[window_starts-WARMUP_SAMPLES, 3], -max_steer, max_steer)
    # ``speed_reference`` is a causal hidden actuator state.  Reconstruct it
    # forward from the beginning of the warm-up interval.  Initializing it
    # from state[:, 0] used vx at the *rollout start* and then applied past
    # commands, which mixed two different times and biased the first rollout
    # acceleration.  Use the original causal KF vx at t-WARMUP instead of a
    # consistency-adjusted teacher vx: this state belongs to the real
    # longitudinal actuator, not to the pose-derived training target.
    speed_reference = feature[window_starts-WARMUP_SAMPLES, 0].astype(
        np.float64).copy()
    # Candidate-dependent actuator warm-up. Never initialize from feature[:,5],
    # because that field was generated with the previous steering parameters.
    for offset in range(-WARMUP_SAMPLES, 0):
        warm_command = feature[window_starts+offset, 3:5]
        warm_target = np.clip(float(config["kinematic_steer_scale"])*
            warm_command[:,0]+float(config["kinematic_steer_bias"]),
            -max_steer, max_steer)
        warm_rate = np.clip((warm_target-applied_steer)/max(
            float(config["steer_servo_time_constant"]),1e-3),
            -float(config["actuator_max_steer_rate"]),
            float(config["actuator_max_steer_rate"]))
        applied_steer = np.clip(
            applied_steer+warm_rate*.02, -max_steer, max_steer)
        warm_tau = np.where(warm_command[:,1] >= speed_reference,
            float(config["speed_reference_accel_time_constant"]),
            float(config["speed_reference_brake_time_constant"]))
        speed_reference += np.clip((warm_command[:,1]-speed_reference)/warm_tau,
            -float(config["actuator_max_speed_reference_rate"]),
            float(config["actuator_max_speed_reference_rate"]))*0.02
    prediction, ground_truth, residual_trace, acceleration_trace = [], [], [], []
    history=feature[window_starts,10:20].astype(np.float64).reshape(-1,5,2).copy()
    acceleration=(data["observations"][window_starts,:2].astype(np.float64).copy()
                  if "observations" in data.files else np.zeros((len(window_starts),2)))
    Bf, Cf, Df, Ef, Br, Cr, Dr, Er = parameters
    lf, lr, mass, iz = [float(config[key]) for key in
                        ("l_f", "l_r", "mass", "dynamic_mlp_I_z")]
    wheelbase = lf + lr
    static_front_load = mass*9.81*lr/wheelbase
    static_rear_load = mass*9.81*lf/wheelbase
    h_cg=float(config.get("load_transfer_h_cg_m",LOAD_TRANSFER_H_CG_M))
    dt = .04
    for step in range(HORIZON):
        row = window_starts + 2*step
        command = feature[row, 3:5]
        if step:history=np.concatenate((history[:,1:],command[:,None]),axis=1)
        previous_command=history[:,-2,0]
        current_state=state.copy()
        steer_target = np.clip(float(config["kinematic_steer_scale"])*
            command[:,0]+float(config["kinematic_steer_bias"]),
            -max_steer, max_steer)
        steer_rate = np.clip((steer_target-applied_steer)/max(
            float(config["steer_servo_time_constant"]),1e-3),
            -float(config["actuator_max_steer_rate"]),
            float(config["actuator_max_steer_rate"]))
        applied_steer = np.clip(
            applied_steer + steer_rate*dt, -max_steer, max_steer)
        speed_command = np.clip(command[:, 1], float(config["min_speed"]), 4.)
        tau = np.where(speed_command >= speed_reference,
                       float(config["speed_reference_accel_time_constant"]),
                       float(config["speed_reference_brake_time_constant"]))
        speed_reference += np.clip(
            (speed_command-speed_reference)/tau,
            -float(config["actuator_max_speed_reference_rate"]),
            float(config["actuator_max_speed_reference_rate"]))*dt
        vx, vy, yaw_rate = state.T
        ax = np.clip(float(config["speed_servo_kp"])
                     *(speed_reference-vx),
                     float(config["min_accel"]), float(config["max_accel"]))
        safe_vx = np.maximum(np.abs(vx), .5)
        alpha_front = applied_steer-np.arctan2(vy+lf*yaw_rate, safe_vx)
        alpha_rear = -np.arctan2(vy-lr*yaw_rate, safe_vx)
        front_term = Bf*alpha_front
        rear_term = Br*alpha_rear
        front_load=np.maximum(.05*mass*9.81,
            static_front_load-mass*ax*h_cg/wheelbase)
        rear_load=np.maximum(.05*mass*9.81,
            static_rear_load+mass*ax*h_cg/wheelbase)
        fy_front = front_load*Df*np.sin(Cf*np.arctan(
            front_term-Ef*(front_term-np.arctan(front_term))))
        fy_rear = rear_load*Dr*np.sin(Cr*np.arctan(
            rear_term-Er*(rear_term-np.arctan(rear_term))))
        dynamic_ay = (fy_front*np.cos(applied_steer)+fy_rear)/mass
        dynamic_yaw_accel = (lf*fy_front*np.cos(applied_steer)-lr*fy_rear)/iz
        blend_input=np.clip((np.abs(vx)-.2)/.3,0.,1.)
        dynamic_blend=blend_input**2*(3.-2.*blend_input)
        kinematic_yaw_rate=vx*np.tan(applied_steer)/max(wheelbase,1e-6)
        ay=dynamic_blend*dynamic_ay+(1.-dynamic_blend)*(vx*yaw_rate-vy/.1)
        yaw_accel=(dynamic_blend*dynamic_yaw_accel
                   +(1.-dynamic_blend)*(kinematic_yaw_rate-yaw_rate)/.1)
        state = np.column_stack((
            vx+(ax+vy*yaw_rate)*dt,
            vy+(ay-vx*yaw_rate)*dt,
            yaw_rate+yaw_accel*dt))
        mlp_feature=np.concatenate((current_state,command,applied_steer[:,None],
            (command[:,0]-previous_command)[:,None],state,history.reshape(len(state),-1),
            acceleration),axis=1)
        residual=frozen_mlp_forward(mlp_feature)
        state=state+residual*dt
        acceleration=np.column_stack((ax+residual[:,0],ay+residual[:,1]))
        acceleration_trace.append(acceleration.copy())
        residual_trace.append(residual)
        prediction.append(state.copy())
        truth=feature[window_starts+2*(step+1), :3].copy()
        if "teacher_state" in data.files:
            truth[:]=data["teacher_state"][window_starts+2*(step+1)]
        ground_truth.append(truth)
    result=(np.stack(prediction,1),np.stack(ground_truth,1))
    extras=[]
    if return_residual:extras.append(np.stack(residual_trace,1))
    if return_acceleration:
        gt_acceleration=np.stack([data["observations"][window_starts+2*(step+1),:2]
                                  for step in range(HORIZON)],axis=1)
        extras.extend((np.stack(acceleration_trace,1),gt_acceleration))
    return (*result,*extras)


def relative_pose(states, scale):
    pose = np.zeros((len(states), states.shape[1], 3)); dt = .04
    for step in range(states.shape[1]):
        previous = pose[:, step-1] if step else np.zeros((len(states), 3))
        vx, vy, yaw_rate = states[:, step].T
        pose[:, step, 0] = previous[:, 0] + scale*(
            vx*np.cos(previous[:, 2])-vy*np.sin(previous[:, 2]))*dt
        pose[:, step, 1] = previous[:, 1] + scale*(
            vx*np.sin(previous[:, 2])+vy*np.cos(previous[:, 2]))*dt
        pose[:, step, 2] = previous[:, 2] + yaw_rate*dt
    return pose


def integrate_measured_kf_trace(states, dt=.04):
    """Integrate sampled KF [body vx, body vy, yaw-rate] without a model rollout.

    Trapezoidal velocity/yaw-rate and midpoint heading reduce integration-method
    error, so the remaining difference primarily exposes KF/MCL inconsistency.
    """
    states=np.asarray(states,dtype=np.float64)
    pose=np.zeros(states.shape[:-1]+(3,),dtype=np.float64)
    for step in range(1,states.shape[1]):
        velocity=.5*(states[:,step-1,:2]+states[:,step,:2])
        yaw_increment=.5*(states[:,step-1,2]+states[:,step,2])*dt
        midpoint_yaw=pose[:,step-1,2]+.5*yaw_increment
        pose[:,step,0]=pose[:,step-1,0]+(
            velocity[:,0]*np.cos(midpoint_yaw)-velocity[:,1]*np.sin(midpoint_yaw))*dt
        pose[:,step,1]=pose[:,step-1,1]+(
            velocity[:,0]*np.sin(midpoint_yaw)+velocity[:,1]*np.cos(midpoint_yaw))*dt
        pose[:,step,2]=pose[:,step-1,2]+yaw_increment
    return pose


def state_derived_body_acceleration(states,dt):
    """Return [dvx/dt-r*vy, dvy/dt+r*vx] from a state time series."""
    states=np.asarray(states,dtype=np.float64)
    edge=2 if states.shape[-2]>=3 else 1
    dvx=np.gradient(states[...,0],dt,axis=-1,edge_order=edge)
    dvy=np.gradient(states[...,1],dt,axis=-1,edge_order=edge)
    return np.stack((dvx-states[...,2]*states[...,1],
                     dvy+states[...,2]*states[...,0]),axis=-1)


def sampled_relative_pose(pose,window_starts):
    """Sample absolute poses at MPPI knots in each initial local frame."""
    pose=np.asarray(pose,dtype=np.float64);result=[]
    for step in range(1,HORIZON+1):
        initial=pose[window_starts]; current=pose[window_starts+2*step]
        dx=current[:,0]-initial[:,0];dy=current[:,1]-initial[:,1];yaw=initial[:,2]
        result.append(np.c_[dx*np.cos(yaw)+dy*np.sin(yaw),
                            -dx*np.sin(yaw)+dy*np.cos(yaw),
                            (current[:,2]-yaw+np.pi)%(2*np.pi)-np.pi])
    return np.stack(result,1)


def mcl_relative_pose(data, window_starts):
    """Configured trajectory target in each rollout's initial local frame."""
    return sampled_relative_pose(data.get("target_pose",data["mcl_pose"]),window_starts)


def estimate_position_speed_scale(data, window_starts):
    """Robust bag-median scale aligning integrated KF velocity to MCL pose."""
    if not ({"target_pose","mcl_pose"}&set(data.files)) or not len(window_starts):
        return 1.0
    state_source=data.get("teacher_state",data["features"][:,:3])
    truth=np.stack([state_source[window_starts+2*(step+1)]
                    for step in range(HORIZON)],axis=1)
    integrated=relative_pose(truth,1.0)[:,:,:2]
    measured=mcl_relative_pose(data,window_starts)[:,:,:2]
    bag_scales=[]
    for bag_id in np.unique(data["bag_id"][window_starts]):
        mask=data["bag_id"][window_starts]==bag_id
        denominator=float(np.sum(integrated[mask]**2))
        if denominator>1e-8:
            bag_scales.append(float(np.sum(integrated[mask]*measured[mask])/denominator))
    scale=float(np.median(bag_scales)) if bag_scales else 1.0
    return float(np.clip(scale,*POSITION_SPEED_SCALE_BOUNDS))


def objective(parameters, data, window_starts, config, regularize=True):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if FROZEN_MLP is None:
            prediction,truth=rollout_numpy(parameters,data,window_starts,config);residual_trace=None
        else:
            prediction,truth,residual_trace=rollout_numpy(parameters,data,window_starts,config,True)
    # Broad Pacejka sampling can create numerically unstable open-loop states.
    # Treat those candidates as invalid instead of allowing inf/NaN to poison
    # differential evolution or the float32 surrogate targets.
    if (not np.isfinite(prediction).all() or not np.isfinite(truth).all()
            or np.max(np.abs(prediction)) > 1e4):
        return 1e12
    time_weight = np.linspace(.25, 1., HORIZON)[None, :, None]
    state_error = (prediction-truth)*np.asarray((.4, 2., 1.5))[None, None, :]
    huber = np.where(np.abs(state_error) < .3,
                     .5*state_error**2, .3*(np.abs(state_error)-.15))
    state_weights=np.asarray((VX_LOSS_WEIGHT,VY_LOSS_WEIGHT,
                              YAW_RATE_LOSS_WEIGHT))[None,None,:]
    loss = float(np.sum(huber*time_weight*state_weights)
                 / max(len(prediction)*np.sum(time_weight)*np.sum(state_weights),1e-12))
    position_scale=float(config.get("kinematic_position_speed_scale",1.0))
    trajectory_truth=(mcl_relative_pose(data,window_starts) if "target_pose" in data.files or "mcl_pose" in data.files
                      else relative_pose(truth,position_scale))
    predicted_trajectory=relative_pose(prediction,position_scale)
    trajectory_position_error=(predicted_trajectory[:,:,:2]
                               -trajectory_truth[:,:,:2])
    trajectory_yaw_error=((predicted_trajectory[:,:,2]-trajectory_truth[:,:,2]
                           +np.pi)%(2*np.pi)-np.pi)
    if FULL_TRAJECTORY_POSE_LOSS_WEIGHT>0.:
        pose_time_weight=time_weight[:,:,0]
        pose_denominator=max(len(prediction)*float(np.sum(pose_time_weight)),1e-12)
        loss += FULL_TRAJECTORY_POSE_LOSS_WEIGHT*(
            POSITION_LOSS_WEIGHT*float(np.sum(
                np.sum(trajectory_position_error**2,axis=2)*pose_time_weight)
                /pose_denominator)
            +YAW_TRAJECTORY_LOSS_WEIGHT*float(np.sum(
                trajectory_yaw_error**2*pose_time_weight)/pose_denominator))
    position_error=predicted_trajectory[:,-1,:2]-trajectory_truth[:,-1,:2]
    yaw_error=(predicted_trajectory[:,-1,2]-trajectory_truth[:,-1,2]+np.pi)%(2*np.pi)-np.pi
    loss += POSITION_LOSS_WEIGHT*float(np.mean(np.sum(position_error**2,axis=1)))
    loss += YAW_TRAJECTORY_LOSS_WEIGHT*float(np.mean(yaw_error**2))
    if ENDPOINT_TAIL_LOSS_WEIGHT>0.:
        if not 0.<ENDPOINT_TAIL_QUANTILE<1.:
            raise ValueError("ENDPOINT_TAIL_QUANTILE must lie in (0, 1)")
        endpoint_state_error=(prediction[:,-1]-truth[:,-1])*np.asarray((.4,2.,1.5))
        endpoint_state_huber=np.where(
            np.abs(endpoint_state_error)<.3,.5*endpoint_state_error**2,
            .3*(np.abs(endpoint_state_error)-.15))
        endpoint_state_cost=np.sum(endpoint_state_huber*state_weights[0],axis=1) \
            / max(float(np.sum(state_weights)),1e-12)
        position_cost=np.sum(position_error**2,axis=1)
        yaw_cost=yaw_error**2
        def cvar(values):
            threshold=np.quantile(values,ENDPOINT_TAIL_QUANTILE)
            tail=values[values>=threshold]
            return float(np.mean(tail))
        loss += ENDPOINT_TAIL_LOSS_WEIGHT*(
            .2*cvar(endpoint_state_cost)
            +POSITION_LOSS_WEIGHT*cvar(position_cost)
            +YAW_TRAJECTORY_LOSS_WEIGHT*cvar(yaw_cost))
    if ONE_STEP_LOSS_WEIGHT>0.:
        # The recursive term already contains step 1 with the smallest time
        # weight.  This explicit term prevents long-horizon compensation from
        # sacrificing the local 40 ms dynamics used by MPPI.
        first_state_error=(prediction[:,0]-truth[:,0])*np.asarray((.4,2.,1.5))
        first_state_huber=np.where(
            np.abs(first_state_error)<.3,
            .5*first_state_error**2,
            .3*(np.abs(first_state_error)-.15))
        first_state_loss=float(np.sum(first_state_huber*state_weights[0])
                               / max(len(prediction)*np.sum(state_weights),1e-12))
        first_position_error=(predicted_trajectory[:,0,:2]
                              -trajectory_truth[:,0,:2])
        first_yaw_error=((predicted_trajectory[:,0,2]-trajectory_truth[:,0,2]
                          +np.pi)%(2*np.pi)-np.pi)
        first_loss=(first_state_loss
                    +POSITION_LOSS_WEIGHT*float(np.mean(np.sum(
                        first_position_error**2,axis=1)))
                    +YAW_TRAJECTORY_LOSS_WEIGHT*float(np.mean(
                        first_yaw_error**2)))
        loss += ONE_STEP_LOSS_WEIGHT*first_loss
    if FROZEN_MLP is not None:
        # Decomposition regularizer: favor classic parameters for systematic
        # dynamics without overwhelming prediction quality.
        loss += CLASSIC_RESIDUAL_PENALTY*float(np.mean(residual_trace**2))
    if regularize:
        span = BOUNDS[:, 1]-BOUNDS[:, 0]
        loss += 2e-4*float(np.mean(((parameters-REFERENCE)/span)**2))
        # Keep front/rear small-slip gains in the same physical order without
        # forcing identical tires under unequal load/observability.
        gains = np.asarray((parameters[0]*parameters[1]*parameters[2],
                            parameters[4]*parameters[5]*parameters[6]))
        loss += 1e-4*float((np.log((gains[0]+1e-4)/(gains[1]+1e-4)))**2)
    return loss if np.isfinite(loss) else 1e12


def objective_breakdown(parameters,data,window_starts,config):
    """Return interpretable components whose sum matches ``objective``."""
    prediction,truth=rollout_numpy(parameters,data,window_starts,config)
    time_weight=np.linspace(.25,1.,HORIZON)[None,:,None]
    scales=np.asarray((.4,2.,1.5));weights=np.asarray(
        (VX_LOSS_WEIGHT,VY_LOSS_WEIGHT,YAW_RATE_LOSS_WEIGHT))
    error=(prediction-truth)*scales[None,None,:]
    huber=np.where(np.abs(error)<.3,.5*error**2,.3*(np.abs(error)-.15))
    denominator=max(len(prediction)*float(np.sum(time_weight))*float(np.sum(weights)),1e-12)
    result={f"recursive_{name}":float(np.sum(
        huber[:,:,column]*time_weight[:,:,0])*weights[column]/denominator)
        for column,name in enumerate(("vx","vy","yaw_rate"))}
    position_scale=float(config.get("kinematic_position_speed_scale",1.0))
    trajectory_truth=(mcl_relative_pose(data,window_starts)
        if "target_pose" in data.files or "mcl_pose" in data.files
        else relative_pose(truth,position_scale))
    predicted_trajectory=relative_pose(prediction,position_scale)
    trajectory_position_error=predicted_trajectory[:,:,:2]-trajectory_truth[:,:,:2]
    trajectory_yaw_error=((predicted_trajectory[:,:,2]-trajectory_truth[:,:,2]
                           +np.pi)%(2*np.pi)-np.pi)
    if FULL_TRAJECTORY_POSE_LOSS_WEIGHT>0.:
        pose_time_weight=time_weight[:,:,0]
        pose_denominator=max(len(prediction)*float(np.sum(pose_time_weight)),1e-12)
        result["full_trajectory_position_xy"]=(
            FULL_TRAJECTORY_POSE_LOSS_WEIGHT*POSITION_LOSS_WEIGHT*float(np.sum(
                np.sum(trajectory_position_error**2,axis=2)*pose_time_weight)
                /pose_denominator))
        result["full_trajectory_yaw"]=(
            FULL_TRAJECTORY_POSE_LOSS_WEIGHT*YAW_TRAJECTORY_LOSS_WEIGHT*float(
                np.sum(trajectory_yaw_error**2*pose_time_weight)/pose_denominator))
    position_error=predicted_trajectory[:,-1,:2]-trajectory_truth[:,-1,:2]
    yaw_error=(predicted_trajectory[:,-1,2]-trajectory_truth[:,-1,2]+np.pi)%(2*np.pi)-np.pi
    result["position_xy"]=POSITION_LOSS_WEIGHT*float(np.mean(np.sum(position_error**2,axis=1)))
    result["trajectory_yaw"]=YAW_TRAJECTORY_LOSS_WEIGHT*float(np.mean(yaw_error**2))
    one_step_total=0.
    if ONE_STEP_LOSS_WEIGHT>0.:
        first_error=(prediction[:,0]-truth[:,0])*scales
        first_huber=np.where(np.abs(first_error)<.3,.5*first_error**2,
                             .3*(np.abs(first_error)-.15))
        for column,name in enumerate(("vx","vy","yaw_rate")):
            value=(ONE_STEP_LOSS_WEIGHT*float(np.sum(first_huber[:,column]))
                   *weights[column]/max(len(prediction)*float(np.sum(weights)),1e-12))
            result[f"one_step_{name}"]=value;one_step_total+=value
        first_position=predicted_trajectory[:,0,:2]-trajectory_truth[:,0,:2]
        first_yaw=(predicted_trajectory[:,0,2]-trajectory_truth[:,0,2]+np.pi)%(2*np.pi)-np.pi
        result["one_step_position_xy"]=(ONE_STEP_LOSS_WEIGHT*POSITION_LOSS_WEIGHT
            *float(np.mean(np.sum(first_position**2,axis=1))))
        result["one_step_yaw"]=(ONE_STEP_LOSS_WEIGHT*YAW_TRAJECTORY_LOSS_WEIGHT
            *float(np.mean(first_yaw**2)))
        one_step_total+=result["one_step_position_xy"]+result["one_step_yaw"]
    tail_total=0.
    if ENDPOINT_TAIL_LOSS_WEIGHT>0.:
        endpoint_error=(prediction[:,-1]-truth[:,-1])*scales
        endpoint_huber=np.where(np.abs(endpoint_error)<.3,.5*endpoint_error**2,
                                .3*(np.abs(endpoint_error)-.15))
        endpoint_cost=np.sum(endpoint_huber*weights,axis=1)/max(float(np.sum(weights)),1e-12)
        def cvar(values):
            threshold=np.quantile(values,ENDPOINT_TAIL_QUANTILE)
            return float(np.mean(values[values>=threshold]))
        tail_total=ENDPOINT_TAIL_LOSS_WEIGHT*(
            .2*cvar(endpoint_cost)
            +POSITION_LOSS_WEIGHT*cvar(np.sum(position_error**2,axis=1))
            +YAW_TRAJECTORY_LOSS_WEIGHT*cvar(yaw_error**2))
    result["endpoint_tail"]=tail_total
    subtotal=sum(result.values())
    total=float(objective(parameters,data,window_starts,config,regularize=True))
    result["regularization_or_residual"]=total-subtotal
    result["total"]=total
    return result


def metrics(parameters, data, window_starts, config):
    prediction, truth = rollout_numpy(parameters, data, window_starts, config)
    state_error = np.abs(prediction[:, -1]-truth[:, -1])
    first_state_error=np.abs(prediction[:,0]-truth[:,0])
    position_scale=float(config.get("kinematic_position_speed_scale",1.0))
    trajectory_truth=(mcl_relative_pose(data,window_starts) if "target_pose" in data.files or "mcl_pose" in data.files
                      else relative_pose(truth,position_scale))
    predicted_trajectory=relative_pose(prediction,position_scale)
    xy_error=predicted_trajectory[:,-1,:2]-trajectory_truth[:,-1,:2]
    position_error=np.linalg.norm(xy_error,axis=1)
    yaw_error=np.abs((predicted_trajectory[:,-1,2]-trajectory_truth[:,-1,2]+np.pi)
                     %(2*np.pi)-np.pi)
    first_xy_error=predicted_trajectory[:,0,:2]-trajectory_truth[:,0,:2]
    first_position_error=np.linalg.norm(first_xy_error,axis=1)
    first_yaw_error=np.abs((predicted_trajectory[:,0,2]-trajectory_truth[:,0,2]
                            +np.pi)%(2*np.pi)-np.pi)
    return {"windows": len(window_starts),
            "one_step_state_mae":first_state_error.mean(0).tolist(),
            "one_step_position_mean_m":float(first_position_error.mean()),
            "one_step_yaw_mean_rad":float(first_yaw_error.mean()),
            "state_mae": state_error.mean(0).tolist(),
            "state_p95": np.quantile(state_error, .95, axis=0).tolist(),
            "trajectory_x_rmse_m":float(np.sqrt(np.mean(xy_error[:,0]**2))),
            "trajectory_y_rmse_m":float(np.sqrt(np.mean(xy_error[:,1]**2))),
            "trajectory_mean_m": float(position_error.mean()),
            "trajectory_p95_m": float(np.quantile(position_error, .95)),
            "trajectory_yaw_mean_rad":float(yaw_error.mean()),
            "trajectory_yaw_p95_rad":float(np.quantile(yaw_error,.95))}


class Surrogate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 1))
    def forward(self, value): return self.net(value).squeeze(-1)


def surrogate_search(data, train_starts, config, rng):
    subset = train_starts[np.linspace(
        0, len(train_starts)-1, min(240, len(train_starts))).astype(int)]
    unit = (np.arange(SURROGATE_SAMPLES)[:, None]
            + rng.random((SURROGATE_SAMPLES, 8)))/SURROGATE_SAMPLES
    for column in range(8): rng.shuffle(unit[:, column])
    samples = BOUNDS[:, 0]+unit*(BOUNDS[:, 1]-BOUNDS[:, 0])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_targets = np.asarray(
            [objective(p, data, subset, config) for p in samples],
            dtype=np.float64)
    finite = np.isfinite(raw_targets) & (raw_targets < 1e11)
    if np.count_nonzero(finite) < 32:
        # The surrogate has too few stable examples to learn a useful global
        # ranking. Return the best physically evaluated stable sample instead.
        stable_targets = np.where(np.isfinite(raw_targets), raw_targets, 1e12)
        return samples[int(np.argmin(stable_targets))]
    unit = unit[finite]
    samples = samples[finite]
    raw_targets = raw_targets[finite]
    # Extreme but finite unstable candidates need not dominate normalization.
    target_cap = float(np.quantile(raw_targets, .95))
    targets = np.minimum(raw_targets, target_cap).astype(np.float32)
    x = torch.as_tensor(unit, dtype=torch.float32)
    target_mean, target_std = float(targets.mean()), max(float(targets.std()), 1e-6)
    y = torch.as_tensor((targets-target_mean)/target_std)
    torch.manual_seed(SEED); network = Surrogate(); optimizer = torch.optim.AdamW(
        network.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(1200):
        index = torch.randint(len(x), (min(256, len(x)),))
        loss = torch.nn.functional.smooth_l1_loss(network(x[index]), y[index])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    proposals = rng.random((SURROGATE_PROPOSALS, 8)).astype(np.float32)
    with torch.no_grad(): predicted = network(torch.from_numpy(proposals)).numpy()
    best = proposals[np.argpartition(predicted, 32)[:32]]
    physical = BOUNDS[:, 0]+best*(BOUNDS[:, 1]-BOUNDS[:, 0])
    return min(physical, key=lambda p: objective(p, data, train_starts, config))


def torch_rollout_loss(raw_parameters, data, window_starts, config, device):
    lower = torch.tensor(BOUNDS[:, 0], device=device, dtype=torch.float64)
    upper = torch.tensor(BOUNDS[:, 1], device=device, dtype=torch.float64)
    parameters = lower+(upper-lower)*torch.sigmoid(raw_parameters)
    feature = torch.as_tensor(data["features"], device=device, dtype=torch.float64)
    starts_tensor = torch.as_tensor(window_starts, device=device, dtype=torch.long)
    state = feature[starts_tensor, :3].clone()
    teacher_state=(torch.as_tensor(data["teacher_state"],device=device,dtype=torch.float64)
                   if "teacher_state" in data.files else None)
    if teacher_state is not None:state=teacher_state[starts_tensor].clone()
    max_steer=float(config["max_steer"])
    applied=torch.clamp(
        feature[starts_tensor-WARMUP_SAMPLES,3],-max_steer,max_steer)
    # Match rollout_numpy exactly: initialize the hidden longitudinal state at
    # the past warm-up boundary, never from the future rollout-start vx.
    speed_reference = feature[starts_tensor-WARMUP_SAMPLES, 0].clone()
    predictions=[]; truths=[]
    for offset in range(-WARMUP_SAMPLES,0):
        warm=feature[starts_tensor+offset,3:5]
        target=torch.clamp(float(config["kinematic_steer_scale"])*warm[:,0]+
            float(config["kinematic_steer_bias"]),-max_steer,max_steer)
        rate=torch.clamp((target-applied)/max(
            float(config["steer_servo_time_constant"]),1e-3),
            -float(config["actuator_max_steer_rate"]),
            float(config["actuator_max_steer_rate"]))
        applied=torch.clamp(applied+rate*.02,-max_steer,max_steer)
        tau=torch.where(warm[:,1]>=speed_reference,
            torch.full_like(speed_reference,float(config["speed_reference_accel_time_constant"])),
            torch.full_like(speed_reference,float(config["speed_reference_brake_time_constant"])))
        speed_reference=speed_reference+torch.clamp((warm[:,1]-speed_reference)/tau,
            -float(config["actuator_max_speed_reference_rate"]),float(config["actuator_max_speed_reference_rate"]))*0.02
    Bf,Cf,Df,Ef,Br,Cr,Dr,Er=parameters
    lf,lr,mass,iz=[float(config[k]) for k in ("l_f","l_r","mass","dynamic_mlp_I_z")]
    wheelbase=lf+lr
    static_front_load=mass*9.81*lr/wheelbase
    static_rear_load=mass*9.81*lf/wheelbase
    h_cg=float(config.get("load_transfer_h_cg_m",LOAD_TRANSFER_H_CG_M));dt=.04
    for step in range(HORIZON):
        row=starts_tensor+2*step; command=feature[row,3:5]
        target=torch.clamp(float(config["kinematic_steer_scale"])*command[:,0]+
            float(config["kinematic_steer_bias"]),-max_steer,max_steer)
        rate=torch.clamp((target-applied)/max(
            float(config["steer_servo_time_constant"]),1e-3),
            -float(config["actuator_max_steer_rate"]),
            float(config["actuator_max_steer_rate"]))
        applied=torch.clamp(applied+rate*dt,-max_steer,max_steer)
        speed=torch.clamp(command[:,1],float(config["min_speed"]),4.)
        tau=torch.where(speed>=speed_reference,
            torch.full_like(speed,float(config["speed_reference_accel_time_constant"])),
            torch.full_like(speed,float(config["speed_reference_brake_time_constant"])))
        speed_reference=speed_reference+torch.clamp((speed-speed_reference)/tau,
            -float(config["actuator_max_speed_reference_rate"]),
            float(config["actuator_max_speed_reference_rate"]))*dt
        vx,vy,yaw_rate=state.unbind(1)
        ax=torch.clamp(float(config["speed_servo_kp"])*(speed_reference-vx),
                       float(config["min_accel"]),float(config["max_accel"]))
        safe=torch.clamp(torch.abs(vx),min=.5)
        af=applied-torch.atan2(vy+lf*yaw_rate,safe); ar=-torch.atan2(vy-lr*yaw_rate,safe)
        bf=Bf*af;br=Br*ar
        front_load=torch.clamp(static_front_load-mass*ax*h_cg/wheelbase,
                               min=.05*mass*9.81)
        rear_load=torch.clamp(static_rear_load+mass*ax*h_cg/wheelbase,
                              min=.05*mass*9.81)
        fyf=front_load*Df*torch.sin(Cf*torch.atan(bf-Ef*(bf-torch.atan(bf))))
        fyr=rear_load*Dr*torch.sin(Cr*torch.atan(br-Er*(br-torch.atan(br))))
        dynamic_ay=(fyf*torch.cos(applied)+fyr)/mass
        dynamic_yaw_accel=(lf*fyf*torch.cos(applied)-lr*fyr)/iz
        blend_input=torch.clamp((torch.abs(vx)-.2)/.3,0.,1.)
        dynamic_blend=blend_input*blend_input*(3.-2.*blend_input)
        kinematic_yaw_rate=vx*torch.tan(applied)/max(wheelbase,1e-6)
        ay=dynamic_blend*dynamic_ay+(1.-dynamic_blend)*(vx*yaw_rate-vy/.1)
        yaw_accel=(dynamic_blend*dynamic_yaw_accel
                   +(1.-dynamic_blend)*(kinematic_yaw_rate-yaw_rate)/.1)
        state=torch.stack((vx+(ax+vy*yaw_rate)*dt,
                           vy+(ay-vx*yaw_rate)*dt,yaw_rate+yaw_accel*dt),1)
        predictions.append(state)
        truth=feature[starts_tensor+2*(step+1),:3].clone()
        if teacher_state is not None:
            truth=teacher_state[starts_tensor+2*(step+1)].clone()
        truths.append(truth)
    prediction=torch.stack(predictions,1);truth=torch.stack(truths,1)
    error=(prediction-truth)*torch.tensor((.4,2.,1.5),device=device)
    state_weights=torch.tensor((VX_LOSS_WEIGHT,VY_LOSS_WEIGHT,YAW_RATE_LOSS_WEIGHT),
                               device=device,dtype=torch.float64)
    element_loss=torch.nn.functional.smooth_l1_loss(
        error,torch.zeros_like(error),beta=.3,reduction="none")
    loss=torch.sum(element_loss*state_weights)/(
        prediction.shape[0]*prediction.shape[1]*torch.sum(state_weights))
    # Match the black-box objective: recursively integrate body velocities so
    # Adam cannot improve vy/r while silently worsening the actual trajectory.
    predicted_pose=torch.zeros((len(window_starts),3),device=device,dtype=torch.float64)
    truth_pose=torch.zeros_like(predicted_pose)
    for step in range(HORIZON):
        pvx,pvy,pr=prediction[:,step].unbind(1)
        tvx,tvy,tr=truth[:,step].unbind(1)
        predicted_pose=torch.stack((
            predicted_pose[:,0]+(pvx*torch.cos(predicted_pose[:,2])-pvy*torch.sin(predicted_pose[:,2]))*.04,
            predicted_pose[:,1]+(pvx*torch.sin(predicted_pose[:,2])+pvy*torch.cos(predicted_pose[:,2]))*.04,
            predicted_pose[:,2]+pr*.04),1)
        truth_pose=torch.stack((truth_pose[:,0]+(tvx*torch.cos(truth_pose[:,2])-tvy*torch.sin(truth_pose[:,2]))*.04,
            truth_pose[:,1]+(tvx*torch.sin(truth_pose[:,2])+tvy*torch.cos(truth_pose[:,2]))*.04,truth_pose[:,2]+tr*.04),1)
    if "target_pose" in data.files or "mcl_pose" in data.files:
        pose_source=data.get("target_pose",data["mcl_pose"])
        target_pose=torch.as_tensor(pose_source,device=device,dtype=torch.float64)
        initial=target_pose[starts_tensor]; final=target_pose[starts_tensor+2*HORIZON]
        dx=final[:,0]-initial[:,0];dy=final[:,1]-initial[:,1];heading=initial[:,2]
        truth_pose=torch.stack((dx*torch.cos(heading)+dy*torch.sin(heading),
                                -dx*torch.sin(heading)+dy*torch.cos(heading),
                                final[:,2]-heading),1)
    loss=loss+POSITION_LOSS_WEIGHT*torch.mean(torch.sum(
        (predicted_pose[:,:2]-truth_pose[:,:2])**2,dim=1))
    yaw_pose_error=torch.atan2(torch.sin(predicted_pose[:,2]-truth_pose[:,2]),
                               torch.cos(predicted_pose[:,2]-truth_pose[:,2]))
    loss=loss+YAW_TRAJECTORY_LOSS_WEIGHT*torch.mean(yaw_pose_error**2)
    # Tail-risk term: classic yaw error in a few hard corners dominates MPPI
    # open-loop heading even when mean state loss is small.
    endpoint_yaw_error=torch.abs(prediction[:,-1,2]-truth[:,-1,2])
    # A static classic parameter vector cannot selectively repair a few
    # bag-specific yaw outliers.  A large CVaR term made the complete test
    # distribution worse, so tail correction is left to the residual model.
    # The classic fit still receives dense state and integrated-position loss.
    YAW_ENDPOINT_CVAR_WEIGHT=0.0
    if YAW_ENDPOINT_CVAR_WEIGHT>0:
        tail_count=max(1,int(.10*len(endpoint_yaw_error)))
        loss=loss+YAW_ENDPOINT_CVAR_WEIGHT*torch.topk(endpoint_yaw_error,tail_count).values.mean()
    reference=torch.tensor(REFERENCE,device=device,dtype=torch.float64)
    loss=loss+2e-4*torch.mean(((parameters-reference)/(upper-lower))**2)
    return loss,parameters


def adam_search(data, train_starts, config):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset=train_starts[np.linspace(0,len(train_starts)-1,
        min(700,len(train_starts))).astype(int)]
    best=None
    for restart in range(ADAM_RESTARTS):
        generator=torch.Generator().manual_seed(SEED+restart)
        raw=torch.nn.Parameter(torch.randn(
            8,generator=generator,dtype=torch.float64).to(device))
        optimizer=torch.optim.AdamW([raw],lr=.035,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,ADAM_STEPS,eta_min=2e-4)
        for _ in range(ADAM_STEPS):
            loss,parameters=torch_rollout_loss(raw,data,subset,config,device)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([raw],5.);optimizer.step();scheduler.step()
        candidate=parameters.detach().cpu().numpy()
        score=objective(candidate,data,train_starts,config)
        if best is None or score<best[0]:best=(score,candidate)
    return best[1]


def validation_score(metric):
    state_scale=np.asarray((.4,2.,1.5))
    state_weight=np.asarray((VX_LOSS_WEIGHT,VY_LOSS_WEIGHT,YAW_RATE_LOSS_WEIGHT))
    state_mean=float(np.sum(state_scale*state_weight*np.asarray(metric["state_mae"]))
                     / max(np.sum(state_weight),1e-12))
    state_p95=float(np.sum(state_scale*state_weight*np.asarray(metric["state_p95"]))
                    / max(np.sum(state_weight),1e-12))
    score=(POSITION_LOSS_WEIGHT*(metric["trajectory_mean_m"]
                                +.5*metric["trajectory_p95_m"])
           +YAW_TRAJECTORY_LOSS_WEIGHT*(metric["trajectory_yaw_mean_rad"]
                                        +.5*metric["trajectory_yaw_p95_rad"])
           +.2*state_mean+.1*state_p95)
    if ONE_STEP_LOSS_WEIGHT>0. and "one_step_state_mae" in metric:
        first_state=float(np.sum(
            state_scale*state_weight*np.asarray(metric["one_step_state_mae"]))
            / max(np.sum(state_weight),1e-12))
        score += ONE_STEP_LOSS_WEIGHT*(
            .2*first_state
            +POSITION_LOSS_WEIGHT*metric["one_step_position_mean_m"]
            +YAW_TRAJECTORY_LOSS_WEIGHT*metric["one_step_yaw_mean_rad"])
    return score


def print_pose_metric_change(title,previous,fitted):
    """Print endpoint pose errors in a compact previous -> fitted form."""
    fields=(("x RMSE","trajectory_x_rmse_m","m"),
            ("y RMSE","trajectory_y_rmse_m","m"),
            ("position mean","trajectory_mean_m","m"),
            ("position P95","trajectory_p95_m","m"),
            ("yaw mean","trajectory_yaw_mean_rad","rad"),
            ("yaw P95","trajectory_yaw_p95_rad","rad"))
    print(f"\n{title} pose open-loop metrics:")
    for label,key,unit in fields:
        old=float(previous[key]);new=float(fitted[key])
        reduction=100.*(old-new)/old if abs(old)>1e-12 else float("nan")
        print(f"  {label}: {old:.6g} -> {new:.6g} {unit} "
              f"(reduction {reduction:+.2f}%)")


def plot_tire_force_curves(data, previous, fitted, config):
    """Compare normalized Pacejka force curves and mark observed slip ranges."""
    import matplotlib.pyplot as plt
    state=(np.asarray(data["teacher_state"],float)
           if "teacher_state" in data.files else np.asarray(data["features"][:,:3],float))
    applied=np.asarray(data["features"][:,5],float)
    valid=np.asarray(data["valid"],bool)&np.isfinite(state).all(1)&np.isfinite(applied)
    vx,vy,yaw_rate=state[valid].T
    safe=np.maximum(np.abs(vx),.5)
    front_observed=applied[valid]-np.arctan2(vy+float(config["l_f"])*yaw_rate,safe)
    rear_observed=-np.arctan2(vy-float(config["l_r"])*yaw_rate,safe)
    observed=(front_observed,rear_observed)
    max_observed=max(float(np.quantile(np.abs(values),.995)) for values in observed)
    alpha_limit=np.clip(max(np.deg2rad(20.),1.25*max_observed),np.deg2rad(20.),np.deg2rad(60.))
    alpha=np.linspace(-alpha_limit,alpha_limit,1001)

    def normalized_force(values,offset):
        b,c,d,e=np.asarray(values[offset:offset+4],float)
        z=b*alpha
        return d*np.sin(c*np.arctan(z-e*(z-np.arctan(z))))

    fig,axes=plt.subplots(1,2,figsize=(14,5.5),sharey=True)
    for axis,title,offset,values in zip(
            axes,("Front tire","Rear tire"),(0,4),observed):
        axis.plot(np.rad2deg(alpha),normalized_force(previous,offset),lw=2,
                  color="tab:orange",label="baseline (pre-fit)")
        axis.plot(np.rad2deg(alpha),normalized_force(fitted,offset),lw=2,
                  color="tab:blue",label="regressed candidate")
        q01,q99=np.quantile(values,(.01,.99));qmin,qmax=np.min(values),np.max(values)
        axis.axvspan(np.rad2deg(q01),np.rad2deg(q99),color="tab:green",alpha=.14,
                    label="observed slip 1–99%")
        axis.axvline(np.rad2deg(qmin),color="tab:green",ls=":",alpha=.7,
                     label="observed min/max")
        axis.axvline(np.rad2deg(qmax),color="tab:green",ls=":",alpha=.7)
        axis.axhline(0.,color="black",lw=.7);axis.axvline(0.,color="black",lw=.7)
        axis.set_title(f"{title}: normalized lateral force")
        axis.set_xlabel("slip angle α [deg]");axis.grid(alpha=.25);axis.legend(fontsize=8)
    axes[0].set_ylabel("Fy / Fz")
    fig.suptitle("Pacejka Fy–α curves: baseline vs Step 3 candidate")
    fig.tight_layout()
    output=OUT/"pacejka_fy_vs_alpha.png";fig.savefig(output,dpi=180)
    if SHOW_PLOTS:plt.show()
    plt.close(fig)

    # E is poorly identifiable when B/C/D are optimized simultaneously.  Show
    # its effect with B/C/D frozen at the fitted values so a boundary solution
    # (for example E=-100) cannot be mistaken for an ordinary tire curve.
    e_values=(1.0,0.0,-1.0,-2.0,-10.0,-100.0)
    e_fig,e_axes=plt.subplots(1,2,figsize=(14,5.5),sharey=True)
    for axis,title,offset,observed_values in zip(
            e_axes,("Front tire","Rear tire"),(0,4),observed):
        b,c,d,_=np.asarray(fitted[offset:offset+4],float)
        z=b*alpha
        for e in e_values:
            force=d*np.sin(c*np.arctan(z-e*(z-np.arctan(z))))
            axis.plot(np.rad2deg(alpha),force,lw=1.8,label=f"E={e:g}")
        q01,q99=np.quantile(observed_values,(.01,.99))
        axis.axvspan(np.rad2deg(q01),np.rad2deg(q99),color="tab:green",alpha=.12,
                    label="observed slip 1–99%")
        axis.axhline(0.,color="black",lw=.7);axis.axvline(0.,color="black",lw=.7)
        axis.set(title=f"{title}: fixed fitted B/C/D",xlabel="slip angle α [deg]")
        axis.grid(alpha=.25);axis.legend(fontsize=8,ncol=2)
    e_axes[0].set_ylabel("Fy / Fz")
    e_fig.suptitle("Pacejka curvature-factor sensitivity: only E changes")
    e_fig.tight_layout()
    e_output=OUT/"pacejka_E_sensitivity.png";e_fig.savefig(e_output,dpi=180)
    if SHOW_PLOTS:plt.show()
    plt.close(e_fig)
    print(f"Pacejka E-sensitivity plot: {e_output}")
    return output


def alternating_pacejka_inertia(initial_tire, data, train, config):
    """Block-coordinate Pacejka <-> Iz fit, then narrow joint polish."""
    tire = np.asarray(initial_tire, float)
    inertia = float(config["dynamic_mlp_I_z"])
    subset = train[np.linspace(0, len(train)-1, min(500, len(train))).astype(int)]
    trace = []
    for round_index in range(2):
        local_bounds = np.column_stack((np.maximum(BOUNDS[:, 0], tire*.8),
                                        np.minimum(BOUNDS[:, 1], tire*1.2)))
        # E may be zero/negative, so multiplicative bounds are not meaningful.
        for index in (3, 7):
            local_bounds[index] = (max(BOUNDS[index, 0], tire[index]-.2),
                                   min(BOUNDS[index, 1], tire[index]+.2))
        local_config = dict(config); local_config["dynamic_mlp_I_z"] = inertia
        pacejka = differential_evolution(
            lambda p: objective(p, data, subset, local_config), local_bounds,
            seed=SEED+100+round_index, popsize=5, maxiter=12, polish=True)
        tire = pacejka.x
        iz_bounds = (max(I_Z_MIN, .7*inertia), min(I_Z_MAX, 1.3*inertia))
        iz_fit = minimize_scalar(lambda value: objective(
            tire, data, subset, {**config, "dynamic_mlp_I_z": value}),
            bounds=iz_bounds, method="bounded")
        inertia = float(iz_fit.x)
        trace.append({"round": round_index+1, "I_z": inertia,
                      "pacejka": dict(zip(NAMES, tire.tolist()))})

    center = np.r_[tire, inertia]
    lower = np.r_[np.maximum(BOUNDS[:,0], tire*.9), max(I_Z_MIN, inertia*.9)]
    upper = np.r_[np.minimum(BOUNDS[:,1], tire*1.1), min(I_Z_MAX, inertia*1.1)]
    for index in (3, 7):
        lower[index] = max(BOUNDS[index,0], tire[index]-.1)
        upper[index] = min(BOUNDS[index,1], tire[index]+.1)
    joint = minimize(lambda p: objective(p[:8], data, subset,
                     {**config, "dynamic_mlp_I_z": p[8]}), center,
                     method="Nelder-Mead", options={"maxiter": 350})
    final = np.clip(joint.x, lower, upper)
    return final[:8], float(final[8]), trace


def plot_clicked_classic_rollout(data,start,previous_tire,fitted_tire,
                                 previous_config,fitted_config,bag_name):
    """Open a 3x3 state/pose/acceleration comparison from one clicked row."""
    if TRAJECTORY_TIME_LABEL_INTERVAL_S<=0:
        raise ValueError("TRAJECTORY_TIME_LABEL_INTERVAL_S must be positive")
    starts_array=np.asarray([start],int)
    previous,truth,previous_accel,gt_accel=rollout_numpy(
        previous_tire,data,starts_array,previous_config,return_acceleration=True)
    fitted,fitted_truth,fitted_accel,fitted_gt_accel=rollout_numpy(
        fitted_tire,data,starts_array,fitted_config,return_acceleration=True)
    if not np.allclose(truth,fitted_truth) or not np.allclose(gt_accel,fitted_gt_accel):
        raise RuntimeError("clicked previous/fitted rollouts do not share GT")
    initial=data["teacher_state"][start].astype(float)
    previous_trace=np.r_[initial[None],previous[0]]
    fitted_trace=np.r_[initial[None],fitted[0]]
    truth_trace=np.r_[initial[None],truth[0]]
    raw_state_trace=np.stack([data["features"][start+2*step,:3]
                              for step in range(HORIZON+1)]).astype(float)
    zero=np.zeros((1,1,3))
    previous_pose=np.concatenate((zero,relative_pose(previous,float(
        previous_config.get("kinematic_position_speed_scale",1.0)))),axis=1)[0]
    fitted_pose=np.concatenate((zero,relative_pose(fitted,float(
        fitted_config.get("kinematic_position_speed_scale",1.0)))),axis=1)[0]
    gt_pose=np.concatenate((zero,mcl_relative_pose(data,starts_array)),axis=1)[0]
    raw_pose=np.concatenate((zero,sampled_relative_pose(
        data["mcl_pose"],starts_array)),axis=1)[0]
    initial_accel=data["observations"][start,:2].astype(float)
    previous_accel_trace=np.r_[initial_accel[None],previous_accel[0]]
    fitted_accel_trace=np.r_[initial_accel[None],fitted_accel[0]]
    gt_accel_trace=np.r_[initial_accel[None],gt_accel[0]]
    state_accel_trace=state_derived_body_acceleration(truth_trace,.04)
    imu_yaw_rate=np.r_[data["observations"][start,2],
        [data["observations"][start+2*(step+1),2] for step in range(HORIZON)]]
    yaw_imu_rmse=float(np.sqrt(np.mean((truth_trace[:,2]-imu_yaw_rate)**2)))
    time=.04*np.arange(HORIZON+1)
    fig,axes=plt.subplots(3,3,figsize=(20,15),constrained_layout=True)
    axis=axes[0,0]
    axis.plot(gt_pose[:,0],gt_pose[:,1],"k-",lw=2.4,label="configured pose GT")
    axis.plot(raw_pose[:,0],raw_pose[:,1],":",color="0.4",lw=2,
              label="raw MCL pose GT")
    axis.plot(previous_pose[:,0],previous_pose[:,1],"--",color="tab:blue",lw=2,
              label="current parameters")
    axis.plot(fitted_pose[:,0],fitted_pose[:,1],color="tab:red",lw=2,
              label="fitted parameters")
    axis.scatter([0],[0],color="tab:green",s=45,zorder=5,label="clicked start")
    label_step=max(1,int(round(TRAJECTORY_TIME_LABEL_INTERVAL_S/.04)))
    for index in range(0,len(time),label_step):
        axis.annotate(f"+{time[index]:.1f}s",gt_pose[index,:2],xytext=(4,4),
                      textcoords="offset points",fontsize=8,color="black",
                      bbox={"boxstyle":"round,pad=.15","fc":"white","ec":"none","alpha":.72})
    axis.set(xlabel="relative x [m]",ylabel="relative y [m]",title="open-loop trajectory")
    axis.axis("equal");axis.grid(alpha=.3);axis.legend()
    state_specs=((0,"vx","m/s"),(1,"vy","m/s"),(2,"yaw rate","rad/s"))
    for axis,(column,title,unit) in zip(
            (axes[0,1],axes[0,2],axes[1,0]),state_specs):
        axis.plot(time,truth_trace[:,column],"k-",lw=2.3,label="GT")
        axis.plot(time,raw_state_trace[:,column],":",color="0.4",lw=1.8,
                  label=f"raw KF {title}")
        axis.plot(time,previous_trace[:,column],"--",color="tab:blue",lw=2,label="current")
        axis.plot(time,fitted_trace[:,column],color="tab:red",lw=2,label="fitted")
        if column==2:
            axis.plot(time,imu_yaw_rate,color="tab:green",ls=":",lw=1.9,
                      label="signed raw IMU yaw rate")
        shown_title=(f"{title} | configured GT vs IMU RMSE={yaw_imu_rmse:.3f} rad/s"
                     if column==2 else title)
        axis.set(title=shown_title,xlabel="rollout time [s]",ylabel=unit)
        axis.grid(alpha=.3);axis.legend()
    pose_specs=((0,"x","m"),(1,"y","m"),(2,"yaw","rad"))
    for axis,(column,title,unit) in zip(
            (axes[1,1],axes[1,2],axes[2,0]),pose_specs):
        axis.plot(time,gt_pose[:,column],"k-",lw=2.3,label="GT")
        axis.plot(time,raw_pose[:,column],":",color="0.4",lw=1.8,
                  label=f"raw MCL {title}")
        axis.plot(time,previous_pose[:,column],"--",color="tab:blue",lw=2,label="current")
        axis.plot(time,fitted_pose[:,column],color="tab:red",lw=2,label="fitted")
        axis.set(title=title,xlabel="rollout time [s]",ylabel=unit)
        axis.grid(alpha=.3);axis.legend()
    for axis,column,title in ((axes[2,1],0,"longitudinal acceleration ax"),
                              (axes[2,2],1,"lateral acceleration ay")):
        state_imu_rmse=float(np.sqrt(np.mean(
            (state_accel_trace[:,column]-gt_accel_trace[:,column])**2)))
        fitted_imu_rmse=float(np.sqrt(np.mean(
            (fitted_accel_trace[:,column]-gt_accel_trace[:,column])**2)))
        axis.plot(time,gt_accel_trace[:,column],"k-",lw=2.2,label="signed IMU GT")
        axis.plot(time,state_accel_trace[:,column],color="tab:purple",ls=":",lw=2,
                  label="configured-state derivative")
        axis.plot(time,previous_accel_trace[:,column],"--",color="tab:blue",lw=2,
                  label="current model")
        axis.plot(time,fitted_accel_trace[:,column],color="tab:red",lw=2,
                  label="fitted model")
        axis.set(title=(f"{title} | state/model vs IMU RMSE="
                        f"{state_imu_rmse:.2f}/{fitted_imu_rmse:.2f} m/s²"),
                 xlabel="rollout time [s]",ylabel="m/s²")
        axis.grid(alpha=.3);axis.legend()
    local_time=(start-int(np.flatnonzero(data["bag_id"]==data["bag_id"][start])[0]))*.02
    fig.suptitle(f"Clicked Step 3 open loop | {bag_name} | start={local_time:.3f} s | "
                 f"mode={GT_CONSISTENCY_MODE}",fontsize=15)
    safe_name=Path(bag_name).stem.replace("/","_")
    output=OUT/f"clicked_classic_open_loop_{safe_name}_{local_time:.3f}s.png"
    fig.savefig(output,dpi=180);fig.show();fig.canvas.draw_idle()
    print(f"clicked Step 3 open-loop plot: {output}")


def plot_interactive_bag_inspector(data,usable_starts,previous_tire,fitted_tire,
                                   previous_config,fitted_config):
    """Show one complete evaluation bag; p + click opens a detailed rollout."""
    if TRAJECTORY_TIME_LABEL_INTERVAL_S<=0:
        raise ValueError("TRAJECTORY_TIME_LABEL_INTERVAL_S must be positive")
    usable_starts=np.unique(np.asarray(usable_starts,int))
    if not len(usable_starts):return None
    bag_id=int(data["bag_id"][usable_starts[0]])
    rows=np.flatnonzero(data["bag_id"]==bag_id);first=int(rows[0]);time=(rows-first)*.02
    # Training starts are deliberately sparse (quality filters, per-bag cap,
    # then candidate[::3]).  They must not quantize an interactive click to the
    # same training window.  For inspection, allow every 20 ms start that has
    # actuator warm-up and a complete, valid recursive horizon in this bag.
    last=int(rows[-1]);valid=data["valid"]
    selectable_starts=np.asarray([
        start for start in range(first+WARMUP_SAMPLES,last-2*HORIZON+1)
        if valid[start:start+2*HORIZON+1].all()
        and np.all(data["bag_id"][start-WARMUP_SAMPLES:start+2*HORIZON+1]
                   ==bag_id)],int)
    if not len(selectable_starts):
        print(f"{bag_id}: no complete valid horizon available for interactive clicks")
        return None
    bag_name=(Path(str(data["source_paths"][bag_id])).name
              if "source_paths" in data.files else f"bag_id={bag_id}")
    target_state=data["teacher_state"][rows];raw_state=data["features"][rows,:3]
    target_pose=data["target_pose"][rows];raw_pose=data["mcl_pose"][rows]
    target_acceleration=state_derived_body_acceleration(target_state,.02)
    imu=data["observations"][rows]
    bag_ax_rmse=float(np.sqrt(np.mean((target_acceleration[:,0]-imu[:,0])**2)))
    bag_ay_rmse=float(np.sqrt(np.mean((target_acceleration[:,1]-imu[:,1])**2)))
    bag_yaw_rate_rmse=float(np.sqrt(np.mean((target_state[:,2]-imu[:,2])**2)))
    fig,axes=plt.subplots(4,2,figsize=(17,18),constrained_layout=True)
    axis=axes[0,0]
    axis.plot(raw_pose[:,0],raw_pose[:,1],color="0.55",lw=1.4,label="raw MCL")
    axis.plot(target_pose[:,0],target_pose[:,1],"k-",lw=2,label="configured pose GT")
    label_bucket=np.floor(time/TRAJECTORY_TIME_LABEL_INTERVAL_S).astype(int)
    label_indices=np.flatnonzero(np.r_[True,np.diff(label_bucket)>0])
    axis.scatter(target_pose[label_indices,0],target_pose[label_indices,1],
                 s=18,color="tab:purple",zorder=5,
                 label=f"position every {TRAJECTORY_TIME_LABEL_INTERVAL_S:g} s")
    for index in label_indices:
        axis.annotate(f"{time[index]:.0f}s",target_pose[index,:2],xytext=(4,4),
                      textcoords="offset points",fontsize=7,color="tab:purple",
                      bbox={"boxstyle":"round,pad=.15","fc":"white","ec":"none","alpha":.72})
    axis.set(title="complete bag trajectory",xlabel="x [m]",ylabel="y [m]")
    axis.axis("equal");axis.grid(alpha=.3);axis.legend()
    panels=((axes[0,1],0,"vx","m/s"),(axes[1,0],1,"vy","m/s"),
            (axes[1,1],2,"yaw rate","rad/s"))
    for axis,column,title,unit in panels:
        axis.plot(time,raw_state[:,column],color="0.55",ls=":",label="original KF")
        axis.plot(time,target_state[:,column],"k-",lw=1.8,label="configured GT")
        if column==2:
            axis.plot(time,data["observations"][rows,2],color="tab:green",ls=":",
                      lw=1.7,label="signed raw IMU yaw rate")
        axis.set(title=title,xlabel="bag time [s]",ylabel=unit);axis.grid(alpha=.3);axis.legend()
    for axis,column,title in ((axes[2,0],0,"signed IMU ax"),(axes[2,1],1,"signed IMU ay")):
        axis.plot(time,data["observations"][rows,column],"k-",lw=1.6,
                  label=f"signed raw IMU {'ax' if column==0 else 'ay'}")
        axis.plot(time,target_acceleration[:,column],color="tab:purple",ls=":",lw=1.7,
                  label="configured-state derivative")
        axis.set(title=title,xlabel="bag time [s]",ylabel="m/s²");axis.grid(alpha=.3)
        axis.legend()
    axes[3,0].plot(time,data["features"][rows,3],label="steer command")
    axes[3,0].plot(time,data["features"][rows,4],label="speed command")
    axes[3,0].set(title="commands",xlabel="bag time [s]");axes[3,0].grid(alpha=.3);axes[3,0].legend()
    raw_yaw=np.unwrap(raw_pose[:,2]);target_yaw=np.unwrap(target_pose[:,2])
    axes[3,1].plot(time,raw_yaw,color="0.55",ls=":",label="raw MCL yaw")
    axes[3,1].plot(time,target_yaw,"k-",label="configured yaw GT")
    axes[3,1].set(title="yaw",xlabel="bag time [s]",ylabel="rad");axes[3,1].grid(alpha=.3);axes[3,1].legend()
    time_axes=set(axes.flat[1:]);armed={"value":False}
    manager=getattr(fig.canvas,"manager",None);handler=getattr(manager,"key_press_handler_id",None)
    if handler is not None:fig.canvas.mpl_disconnect(handler)
    def on_key(event):
        if event.key and event.key.lower()=="p":
            armed["value"]=True
            fig.suptitle(f"{bag_name} | PREDICTION ARMED: click a time panel",color="tab:red")
            fig.canvas.draw_idle();print("Step 3 prediction armed: click a time-series panel.")
    def on_click(event):
        if not armed["value"] or event.inaxes not in time_axes or event.xdata is None:return
        armed["value"]=False
        requested=first+int(round(float(event.xdata)/.02))
        start=int(selectable_starts[np.argmin(np.abs(selectable_starts-requested))])
        selected_time=(start-first)*.02
        fig.suptitle(f"{bag_name} | selected click start={selected_time:.3f} s",color="black")
        fig.canvas.draw_idle()
        plot_clicked_classic_rollout(data,start,previous_tire,fitted_tire,
                                     previous_config,fitted_config,bag_name)
    fig.canvas.mpl_connect("key_press_event",on_key)
    fig.canvas.mpl_connect("button_press_event",on_click)
    fig.suptitle(f"Step 3 bag inspector | {bag_name} | press p, then click a time panel\n"
                 f"configured state vs IMU RMSE: ax={bag_ax_rmse:.2f} m/s², "
                 f"ay={bag_ay_rmse:.2f} m/s², yaw-rate={bag_yaw_rate_rmse:.3f} rad/s")
    print(f"{bag_name}: configured state vs IMU RMSE ax/ay/yaw-rate="
          f"{bag_ax_rmse:.3f}/{bag_ay_rmse:.3f}/{bag_yaw_rate_rmse:.4f}")
    output=OUT/"interactive_bag_inspector.png";fig.savefig(output,dpi=180)
    if SHOW_PLOTS:plt.show()
    plt.close(fig);return output


def plot_open_loop_evaluation(data, previous_tire, fitted_tire, previous_config,
                              fitted_config, validation, test):
    """Visualize evaluation GT/previous/fitted free rollouts for all states."""
    heldout=np.unique(np.concatenate((validation,test)))
    if not len(heldout):
        raise RuntimeError("no validation/test classic rollout is available for plotting")
    previous_prediction,truth=rollout_numpy(previous_tire,data,heldout,previous_config)
    fitted_prediction,fitted_truth=rollout_numpy(fitted_tire,data,heldout,fitted_config)
    if not np.allclose(truth,fitted_truth,equal_nan=True):
        raise RuntimeError("previous/fitted classic rollouts do not share identical GT")
    normalized=(previous_prediction[:,:,1:3]-truth[:,:,1:3])/np.array((.5,1.))
    score=np.sqrt(np.mean(normalized**2,axis=(1,2)));order=np.argsort(score)
    selected=(order[0],order[int(round(.95*(len(order)-1)))],order[-1])
    labels=("best","p95","worst")
    initial=data["features"][heldout,:3].astype(float).copy()
    if "teacher_state" in data.files:initial[:]=data["teacher_state"][heldout]
    previous_trace=np.concatenate((initial[:,None,:],previous_prediction),axis=1)
    fitted_trace=np.concatenate((initial[:,None,:],fitted_prediction),axis=1)
    truth_trace=np.concatenate((initial[:,None,:],truth),axis=1)
    raw_kf_trace=np.stack([data["features"][heldout+2*step,:3]
                           for step in range(HORIZON+1)],axis=1).astype(float)
    imu_yaw_rate_trace=np.stack([data["observations"][heldout+2*step,2]
                                 for step in range(HORIZON+1)],axis=1).astype(float)
    zero=np.zeros((len(heldout),1,3))
    previous_scale=float(previous_config.get("kinematic_position_speed_scale",1.0))
    fitted_scale=float(fitted_config.get("kinematic_position_speed_scale",1.0))
    previous_pose=np.concatenate((zero,relative_pose(previous_prediction,previous_scale)),axis=1)
    fitted_pose=np.concatenate((zero,relative_pose(fitted_prediction,fitted_scale)),axis=1)
    future_truth_pose=(mcl_relative_pose(data,heldout) if "target_pose" in data.files or "mcl_pose" in data.files
                       else relative_pose(truth,fitted_scale))
    truth_pose=np.concatenate((zero,future_truth_pose),axis=1)
    raw_future_pose=sampled_relative_pose(data["mcl_pose"],heldout)
    raw_pose=np.concatenate((zero,raw_future_pose),axis=1)
    time=.04*np.arange(HORIZON+1)
    fig,axes=plt.subplots(3,5,figsize=(29,14),constrained_layout=True);cases={}
    for row,(label,index) in enumerate(zip(labels,selected)):
        axis=axes[row,0]
        axis.plot(truth_pose[index,:,0],truth_pose[index,:,1],"k-",lw=2.3,label="GT")
        axis.plot(raw_pose[index,:,0],raw_pose[index,:,1],":",color="0.4",lw=1.8,
                  label="raw MCL pose GT")
        axis.plot(previous_pose[index,:,0],previous_pose[index,:,1],"--",lw=2,
                  color="tab:blue",label="previous classic")
        axis.plot(fitted_pose[index,:,0],fitted_pose[index,:,1],color="tab:red",lw=1.9,
                  label="fitted classic")
        axis.set(title=f"{label.upper()} open-loop trajectory",xlabel="relative x [m]",
                 ylabel="relative y [m]");axis.axis("equal");axis.grid(alpha=.3);axis.legend()
        axis=axes[row,1]
        axis.plot(time,truth_trace[index,:,0],"k-",lw=2.3,label="GT vx")
        axis.plot(time,raw_kf_trace[index,:,0],":",color="0.45",lw=1.7,
                  label="raw KF vx")
        axis.plot(time,previous_trace[index,:,0],"--",color="tab:blue",lw=2,
                  label="previous classic")
        axis.plot(time,fitted_trace[index,:,0],color="tab:red",lw=1.9,
                  label="fitted classic")
        axis.set(title=f"{label.upper()} longitudinal velocity",xlabel="rollout time [s]",
                 ylabel="vx [m/s]");axis.grid(alpha=.3);axis.legend()
        axis=axes[row,2]
        axis.plot(time,truth_trace[index,:,1],"k-",lw=2.3,
                  label=f"configured state GT vy ({GT_CONSISTENCY_MODE})")
        axis.plot(time,raw_kf_trace[index,:,1],":",color="0.45",lw=1.7,
                  label="original classic-KF vy (not GT)")
        axis.plot(time,previous_trace[index,:,1],"--",color="tab:blue",lw=2,
                  label="previous classic")
        axis.plot(time,fitted_trace[index,:,1],color="tab:red",lw=1.9,label="fitted classic")
        axis.set(title=f"{label.upper()} lateral velocity",xlabel="rollout time [s]",
                 ylabel="vy [m/s]");axis.grid(alpha=.3);axis.legend()
        axis=axes[row,3]
        axis.plot(time,truth_trace[index,:,2],"k-",lw=2.3,
                  label=f"configured state GT yaw rate ({GT_CONSISTENCY_MODE})")
        axis.plot(time,raw_kf_trace[index,:,2],":",color="0.45",lw=1.7,
                  label="raw KF yaw rate")
        axis.plot(time,imu_yaw_rate_trace[index],":",color="tab:green",lw=1.8,
                  label="signed raw IMU yaw rate")
        axis.plot(time,previous_trace[index,:,2],"--",color="tab:blue",lw=2,
                  label="previous classic")
        axis.plot(time,fitted_trace[index,:,2],color="tab:red",lw=1.9,label="fitted classic")
        axis.set(title=f"{label.upper()} yaw rate",xlabel="rollout time [s]",
                 ylabel="yaw rate [rad/s]");axis.grid(alpha=.3);axis.legend()
        axis=axes[row,4]
        axis.plot(time,truth_pose[index,:,2],"k-",lw=2.3,label="GT relative yaw")
        axis.plot(time,raw_pose[index,:,2],":",color="0.4",lw=1.7,
                  label="raw MCL relative yaw")
        axis.plot(time,previous_pose[index,:,2],"--",color="tab:blue",lw=2,
                  label="previous classic")
        axis.plot(time,fitted_pose[index,:,2],color="tab:red",lw=1.9,
                  label="fitted classic")
        axis.set(title=f"{label.upper()} yaw",xlabel="rollout time [s]",
                 ylabel="relative yaw [rad]");axis.grid(alpha=.3);axis.legend()
        previous_rmse=np.sqrt(np.mean((previous_prediction[index,:,1:3]-truth[index,:,1:3])**2,axis=0))
        fitted_rmse=np.sqrt(np.mean((fitted_prediction[index,:,1:3]-truth[index,:,1:3])**2,axis=0))
        cases[label]={"source_row":int(heldout[index]),
            "previous_vy_rmse_mps":float(previous_rmse[0]),
            "previous_yaw_rate_rmse_radps":float(previous_rmse[1]),
            "fitted_vy_rmse_mps":float(fitted_rmse[0]),
            "fitted_yaw_rate_rmse_radps":float(fitted_rmse[1])}
    mode=("held-out" if USE_VALIDATION_TEST_SPLIT else "selected train-bag")
    fig.suptitle(f"Classic Pacejka + I_z {mode} open-loop evaluation ({HORIZON*.04:.1f} s)",
                 fontsize=16)
    output=OUT/"open_loop_comparison.png";fig.savefig(output,dpi=180)

    # This is not a classic-model prediction. It directly integrates the
    # recorded KF states and the replacement consistency-vy state against MCL.
    integration_steps=min(HORIZON,int(round(1.0/.04)))
    raw_kf_integrated=integrate_measured_kf_trace(
        raw_kf_trace[:,:integration_steps+1])
    consistency_integrated=integrate_measured_kf_trace(
        truth_trace[:,:integration_steps+1])
    mcl_one_second=truth_pose[:,:integration_steps+1]
    integration_time=.04*np.arange(integration_steps+1)
    integration_fig,integration_axes=plt.subplots(
        3,3,figsize=(18,14),constrained_layout=True)
    for row,(label,index) in enumerate(zip(labels,selected)):
        raw_position_error=np.linalg.norm(
            raw_kf_integrated[index,:,:2]-mcl_one_second[index,:,:2],axis=1)
        consistency_position_error=np.linalg.norm(
            consistency_integrated[index,:,:2]-mcl_one_second[index,:,:2],axis=1)
        yaw_error=(consistency_integrated[index,:,2]-mcl_one_second[index,:,2]+np.pi)%(2*np.pi)-np.pi
        axis=integration_axes[row,0]
        axis.plot(mcl_one_second[index,:,0],mcl_one_second[index,:,1],"k-o",
                  ms=2.5,lw=2.2,label=f"configured pose GT ({GT_CONSISTENCY_MODE})")
        axis.plot(raw_kf_integrated[index,:,0],raw_kf_integrated[index,:,1],
                  color="0.5",ls=":",lw=2,label="integrated original KF vx/vy/r")
        axis.plot(consistency_integrated[index,:,0],consistency_integrated[index,:,1],
                  color="tab:purple",marker=".",lw=2,
                  label="integrated KF vx/r + consistency vy")
        axis.scatter([0.],[0.],s=45,color="tab:green",zorder=5,label="common start")
        axis.set(title=f"{label.upper()}: recorded KF state integration",
                 xlabel="relative x [m]",ylabel="relative y [m]")
        axis.axis("equal");axis.grid(alpha=.3);axis.legend()
        axis=integration_axes[row,1]
        axis.plot(integration_time,raw_position_error,color="0.5",ls=":",lw=2,
                  label="original KF")
        axis.plot(integration_time,consistency_position_error,color="tab:red",lw=2,
                  label="consistency vy")
        axis.set(title=f"position error; consistency end={consistency_position_error[-1]:.3f} m",
                 xlabel="integration time [s]",ylabel="||integrated KF - MCL|| [m]")
        axis.grid(alpha=.3);axis.legend()
        axis=integration_axes[row,2]
        axis.plot(integration_time,yaw_error,color="tab:orange",lw=2)
        axis.set(title=f"yaw error; end={yaw_error[-1]:+.3f} rad",
                 xlabel="integration time [s]",ylabel="integrated KF yaw - MCL yaw [rad]")
        axis.grid(alpha=.3)
        cases[label].update({
            "kf_state_integration_horizon_s":float(integration_steps*.04),
            "original_kf_integration_end_position_error_m":float(raw_position_error[-1]),
            "consistency_vy_integration_end_position_error_m":float(consistency_position_error[-1]),
            "consistency_vy_integration_position_rmse_m":float(np.sqrt(np.mean(consistency_position_error**2))),
            "kf_state_integration_end_yaw_error_rad":float(yaw_error[-1]),
            "kf_state_integration_yaw_rmse_rad":float(np.sqrt(np.mean(yaw_error**2)))})
    integration_fig.suptitle(
        f"State integration against configured pose GT: mode={GT_CONSISTENCY_MODE}\n"
        "(same initial pose; no Pacejka prediction and no position-speed scale)",fontsize=15)
    integration_output=OUT/"kf_state_integration_vs_mcl_1s.png"
    integration_fig.savefig(integration_output,dpi=180)
    (OUT/"representative_open_loop_rollouts.json").write_text(json.dumps({
        "selection":"best/p95/worst by previous normalized vy/yaw-rate rollout RMSE",
        "horizon_s":HORIZON*.04,"heldout_windows":int(len(heldout)),"cases":cases},indent=2)+"\n")
    if SHOW_PLOTS and not INTERACTIVE_BAG_INSPECTOR:plt.show()
    plt.close(fig);plt.close(integration_fig)
    print(f"KF-state integration diagnostic: {integration_output}")
    for label in labels:
        item=cases[label]
        print(f"  {label}: 1 s position end/RMSE="
              f"{item['consistency_vy_integration_end_position_error_m']:.4f}/"
              f"{item['consistency_vy_integration_position_rmse_m']:.4f} m, yaw end/RMSE="
              f"{item['kf_state_integration_end_yaw_error_rad']:+.4f}/"
              f"{item['kf_state_integration_yaw_rmse_rad']:.4f} rad")
    plot_vy_reference_point_diagnostic(data,fitted_config)
    if INTERACTIVE_BAG_INSPECTOR:
        inspector=plot_interactive_bag_inspector(
            data,heldout,previous_tire,fitted_tire,previous_config,fitted_config)
        print(f"interactive Step 3 bag inspector: {inspector}")
    return output


def main():
    global BOUNDS, REFERENCE
    OUT.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    config["load_transfer_h_cg_m"]=float(LOAD_TRANSFER_H_CG_M)
    data,data_contract=load_regression_data(DATA,config)
    local_fraction=float(os.environ.get("CLASSIC_LOCAL_FRACTION","0"))
    if local_fraction>0:
        center=np.asarray([config[f"dynamic_mlp_{name}"] for name in NAMES],float)
        original=BOUNDS.copy();BOUNDS=np.column_stack((center*(1-local_fraction),center*(1+local_fraction)))
        for index in (3,7):BOUNDS[index]=(center[index]-.2,center[index]+.2)
        BOUNDS[:,0]=np.maximum(BOUNDS[:,0],original[:,0]);BOUNDS[:,1]=np.minimum(BOUNDS[:,1],original[:,1])
        REFERENCE=center
    split_starts = tuple(starts(data, index) for index in range(3))
    if USE_VALIDATION_TEST_SPLIT:
        train, validation, test = split_starts
        evaluation_contract = "bag-disjoint validation/test"
    else:
        nonempty = [indices for indices in split_starts if len(indices)]
        train = np.concatenate(nonempty) if nonempty else np.empty(0, dtype=int)
        usable_bags = np.unique(data["bag_id"][train]) if len(train) else np.empty(0, int)
        if not len(usable_bags):
            raise RuntimeError("no usable train bag is available for evaluation")
        evaluation_bag = usable_bags[TRAIN_EVALUATION_BAG_INDEX]
        evaluation = train[data["bag_id"][train] == evaluation_bag]
        validation = evaluation.copy()
        test = evaluation.copy()
        evaluation_contract = (
            f"in-sample train-bag diagnostic: bag_id={int(evaluation_bag)}")
    rng=np.random.default_rng(SEED)
    print(f"Step 3 evaluation mode: {evaluation_contract}")
    if min(map(len,(train,validation,test)))==0:
        raise RuntimeError(f"{DATA}: no usable 1.0 s rollout in split sizes "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}")
    previous_config=dict(config)
    current=np.asarray([config[f"dynamic_mlp_{name}"] for name in NAMES],float)
    if EVALUATE_ONLY:
        parameter_path=Path(EVALUATION_PARAMS_PATH).expanduser().resolve()
        if not parameter_path.is_file():
            raise FileNotFoundError(f"evaluation-only parameter file does not exist: {parameter_path}")
        saved=json.loads(parameter_path.read_text())
        fitted_values=saved.get("expanded_fitted",{})
        missing=[name for name in (*NAMES,"I_z") if name not in fitted_values]
        if missing:
            raise KeyError(f"{parameter_path}: missing fitted parameters {missing}")
        selected=np.asarray([fitted_values[name] for name in NAMES],float)
        fitted_config=dict(config)
        fitted_config["dynamic_mlp_I_z"]=float(fitted_values["I_z"])
        fitted_config["kinematic_position_speed_scale"]=float(
            fitted_values.get("kinematic_position_speed_scale",
                              config.get("kinematic_position_speed_scale",1.0)))
        evaluation_report={
            "mode":"evaluation_only_no_regression","parameter_path":str(parameter_path),
            "gt_consistency_mode":GT_CONSISTENCY_MODE,
            "current_parameters":dict(zip(NAMES,current.tolist())),
            "saved_fitted_parameters":dict(zip(NAMES,selected.tolist())),
            "saved_fitted_I_z":float(fitted_config["dynamic_mlp_I_z"]),
            "saved_position_speed_scale":float(
                fitted_config["kinematic_position_speed_scale"]),
            "current_metrics":metrics(current,data,validation,previous_config),
            "saved_fitted_metrics":metrics(selected,data,validation,fitted_config)}
        (OUT/"evaluation_only_report.json").write_text(
            json.dumps(evaluation_report,indent=2)+"\n")
        plot_path=plot_open_loop_evaluation(
            data,current,selected,previous_config,fitted_config,validation,test)
        tire_plot_path=plot_tire_force_curves(data,current,selected,fitted_config)
        print(json.dumps(evaluation_report,indent=2))
        print_pose_metric_change("Step 3 evaluation-only validation",
            evaluation_report["current_metrics"],
            evaluation_report["saved_fitted_metrics"])
        print("Step 3 evaluation-only mode: optimizer was not executed.")
        print(f"open-loop plot: {plot_path}")
        print(f"Pacejka Fy-vs-alpha plot: {tire_plot_path}")
        return
    previous_position_scale=float(config.get("kinematic_position_speed_scale",1.0))
    fitted_position_scale=(estimate_position_speed_scale(data,train)
                           if AUTO_FIT_POSITION_SPEED_SCALE
                           else previous_position_scale)
    config=dict(config)
    config["kinematic_position_speed_scale"]=fitted_position_scale
    print(f"Step 3 position speed scale: {previous_position_scale:.6f} -> "
          f"{fitted_position_scale:.6f}")
    # Existing robust optimizer retained strictly as a comparison baseline.
    de=differential_evolution(lambda p:objective(p,data,train,config),BOUNDS,
        seed=SEED,popsize=DE_POPSIZE,maxiter=DE_MAXITER,tol=8e-4,polish=False,workers=1)
    ls=least_squares(lambda p:(rollout_numpy(p,data,train,config)[0]
        -rollout_numpy(p,data,train,config)[1]).ravel(),de.x,
        bounds=(BOUNDS[:,0],BOUNDS[:,1]),loss="soft_l1",f_scale=.3,max_nfev=100)
    candidates={"current":current,"de_robust_ls":ls.x,
                "adam_differentiable":adam_search(data,train,config),
                "mlp_surrogate":surrogate_search(data,train,config,rng)}
    comparison={}
    for name,parameters in candidates.items():
        comparison[name]={"parameters":dict(zip(NAMES,parameters.tolist())),
            "train":metrics(parameters,data,train,config),
            "validation":metrics(parameters,data,validation,config),
            "test":metrics(parameters,data,test,config)}
        comparison[name]["validation_score"]=validation_score(comparison[name]["validation"])
    winner=min(comparison,key=lambda name:comparison[name]["validation_score"])
    base_selected=np.asarray([comparison[winner]["parameters"][name] for name in NAMES])
    base_iz=float(config["dynamic_mlp_I_z"])
    alternating_selected, alternating_iz, alternating_trace = alternating_pacejka_inertia(
        base_selected, data, train, config)
    alternating_config={**config,"dynamic_mlp_I_z":alternating_iz}
    alternating_validation=metrics(
        alternating_selected,data,validation,alternating_config)
    alternating_accepted=(validation_score(alternating_validation)
                          < comparison[winner]["validation_score"])
    if alternating_accepted:
        selected,selected_iz=alternating_selected,alternating_iz
        selected_method=winner+"+alternating_pacejka_Iz"
    else:
        selected,selected_iz=base_selected,base_iz
        selected_method=winner
    fitted_config={**config,"dynamic_mlp_I_z":selected_iz}
    runtime_previous_metrics={
        "validation":metrics(current,data,validation,previous_config),
        "test":metrics(current,data,test,previous_config)}
    selected_metrics={split:metrics(selected,data,indices,fitted_config)
                      for split,indices in (("train",train),("validation",validation),("test",test))}
    tolerance=.01*(BOUNDS[:,1]-BOUNDS[:,0])
    boundary={name:bool(abs(value-low)<=tol or abs(high-value)<=tol)
              for name,value,(low,high),tol in zip(NAMES,selected,BOUNDS,tolerance)}
    previous_validation=comparison["current"]["validation"]
    fitted_validation=selected_metrics["validation"]
    previous_p95_score=(POSITION_LOSS_WEIGHT*previous_validation["trajectory_p95_m"]
        +YAW_TRAJECTORY_LOSS_WEIGHT*previous_validation["trajectory_yaw_p95_rad"])
    fitted_p95_score=(POSITION_LOSS_WEIGHT*fitted_validation["trajectory_p95_m"]
        +YAW_TRAJECTORY_LOSS_WEIGHT*fitted_validation["trajectory_yaw_p95_rad"])
    score_improved=(validation_score(fitted_validation)
                    < validation_score(previous_validation))
    p95_improved=fitted_p95_score < previous_p95_score
    deployment_gate_passed=bool(score_improved and p95_improved)
    report={"model_dt":.04,"integration":"semi-implicit Euler: next body state advances pose at 0.04 s",
        "input_path":str(DATA.resolve()),"input_contract":data_contract,
        "evaluation_contract":evaluation_contract,
        "use_validation_test_split":USE_VALIDATION_TEST_SPLIT,
        "gt_consistency_mode":GT_CONSISTENCY_MODE,
        "load_transfer_h_cg_m":float(LOAD_TRANSFER_H_CG_M),
        "state_target":("MCL-pose-derived [vx,vy,yaw_rate]"
                        if GT_CONSISTENCY_MODE=="adjust_states_to_pose"
                        else "original classic-KF [vx,vy,yaw_rate]"),
        "trajectory_target":("state-integrated [x,y,yaw]"
                              if GT_CONSISTENCY_MODE=="adjust_pose_to_states"
                              else "raw MCL [x,y,yaw]"),
        "loss_weights":{"vx":VX_LOSS_WEIGHT,"vy":VY_LOSS_WEIGHT,
            "yaw_rate":YAW_RATE_LOSS_WEIGHT,"position_xy":POSITION_LOSS_WEIGHT,
            "trajectory_yaw":YAW_TRAJECTORY_LOSS_WEIGHT},
        "window_filter":{"max_position_step_20ms_m":MAX_POSITION_STEP_20MS,
            "max_yaw_step_20ms_rad":MAX_YAW_STEP_20MS,
            "collision_filter":"none; Step 1 manual review owns collision removal"},
        "previous_position_speed_scale":previous_position_scale,
        "position_speed_scale":fitted_position_scale,
        "parameter_names":list(NAMES)+["I_z","kinematic_position_speed_scale"],
        "selection":"lowest evaluation open-loop score",
        "selected_method":selected_method,"expanded_fitted":{**dict(zip(NAMES,selected.tolist())),
            "I_z":selected_iz,
            "kinematic_position_speed_scale":fitted_position_scale},
        "boundary_solution":boundary,
        "boundary_solution_is_diagnostic_only":True,
        "deployment_gate":{
            "policy":"overall validation and weighted pose/yaw P95 must improve; bounds are diagnostic",
            "previous_validation_score":validation_score(previous_validation),
            "fitted_validation_score":validation_score(fitted_validation),
            "previous_weighted_p95_score":previous_p95_score,
            "fitted_weighted_p95_score":fitted_p95_score,
            "overall_score_improved":bool(score_improved),
            "weighted_p95_improved":bool(p95_improved)},
        "deployment_gate_passed":deployment_gate_passed,
        "fitted_I_z":selected_iz,
        "alternating_candidate_accepted":bool(alternating_accepted),
        "alternating_candidate_validation":alternating_validation,
        "pacejka_inertia_alternating_trace":alternating_trace,
        "metrics_previous":runtime_previous_metrics,
        "metrics_fitted":{"validation":selected_metrics["validation"],
                          "test":selected_metrics["test"]},
        "fixed_parameters":{"mass":float(config["mass"]),
            "l_f":float(config["l_f"]),"l_r":float(config["l_r"])},
        "fixed_actuator_mapping":{"kinematic_steer_scale":float(config["kinematic_steer_scale"]),
            "kinematic_steer_bias":float(config["kinematic_steer_bias"])},
        "methods":comparison}
    (OUT/"advanced_params.json").write_text(json.dumps(report,indent=2)+"\n")
    # Canonical downstream dataset/deployer consumes params.json.
    (OUT/"params.json").write_text(json.dumps(report,indent=2)+"\n")
    if APPLY_ACCEPTED_PARAMS_TO_YAML:
        if report["deployment_gate_passed"]:
            from deploy_residual_model import replace_scalar
            yaml_path=ROOT/"config/params.yaml"
            yaml_text=yaml_path.read_text()
            for name,value in zip(NAMES,selected):
                yaml_text=replace_scalar(yaml_text,f"dynamic_mlp_{name}",float(value))
            yaml_text=replace_scalar(yaml_text,"dynamic_mlp_I_z",float(selected_iz))
            if AUTO_FIT_POSITION_SPEED_SCALE:
                yaml_text=replace_scalar(yaml_text,"kinematic_position_speed_scale",
                                         float(fitted_position_scale))
            yaml_path.write_text(yaml_text)
            print(f"accepted Step 3 parameters applied to {yaml_path}")
        else:
            print("Step 3 parameters were not applied: deployment gate failed")
    plot_path=plot_open_loop_evaluation(
        data,current,selected,previous_config,fitted_config,validation,test)
    tire_plot_path=plot_tire_force_curves(data,current,selected,fitted_config)
    print(json.dumps(report,indent=2))
    print("\nTuned classic-model parameters "
          f"(selected method: {selected_method}):")
    initial_parameters = np.r_[current, float(config["dynamic_mlp_I_z"])]
    tuned_parameters = np.r_[selected, selected_iz]
    for name, initial, tuned in zip(
            (*NAMES, "I_z"), initial_parameters, tuned_parameters):
        print(f"  {name}: {initial:.9g} -> {tuned:.9g}")
    print(f"  kinematic_position_speed_scale: {previous_position_scale:.9g} "
          f"-> {fitted_position_scale:.9g}")
    print_pose_metric_change("Step 3 validation",
        runtime_previous_metrics["validation"],selected_metrics["validation"])
    print(f"open-loop plot: {plot_path}")
    print(f"Pacejka Fy-vs-alpha plot: {tire_plot_path}")


if __name__=="__main__":main()
