"""Load Step-1 callback archives with online MPPI-model EKF states directly."""
import hashlib
from pathlib import Path
import numpy as np


VAL_NAMES={"rosbag2_2026_08_19-19_53_54.npz","rosbag2_2026_08_19-20_02_26.npz"}
TEST_NAMES={"rosbag2_2026_08_19-20_23_43.npz"}
STATE_NAMES=("kf_x","kf_y","kf_yaw","kf_vx","kf_vy","kf_yaw_rate")


def _bag_group(name):
    """Strip a Step-7 '__segment_NN' suffix so a source bag's segments stay
    on one side of the split (no leakage from near-duplicate windows)."""
    return name.split("__segment_")[0]


def _fallback_split(paths):
    """Deterministic bag-disjoint ~70/15/15 split, used only when a dataset
    contains none of the legacy hardcoded VAL_NAMES/TEST_NAMES (i.e. it is
    not the original 2026-08-19 dataset those names were pinned to)."""
    groups=sorted({_bag_group(p.name) for p in paths})
    val,test=set(),set()
    for group in groups:
        bucket=int(hashlib.sha1(group.encode()).hexdigest(),16)%100
        if bucket<15:test.add(group)
        elif bucket<30:val.add(group)
    return val,test
HISTORY_NAMES=tuple(value for k in range(4,-1,-1) for value in
    ((f"steer_t-{k}",f"speed_t-{k}") if k else ("steer_t","speed_t")))


def _interpolate(t,values,query):
    result=np.empty((*query.shape,values.shape[1]),np.float64)
    flat=query.ravel()
    for column in range(values.shape[1]):
        source=np.unwrap(values[:,column]) if column==2 else values[:,column]
        result[...,column]=np.interp(flat,t,source,left=np.nan,right=np.nan).reshape(query.shape)
    return result


def load_callback_archives(directory,model_dt=.04,horizon=30):
    """Return in-memory arrays; never creates an intermediate training NPZ."""
    records=[]
    paths=sorted(Path(directory).glob("*.npz"))
    if any(p.name in VAL_NAMES or p.name in TEST_NAMES for p in paths):
        fallback_val,fallback_test=set(),set()
    else:
        fallback_val,fallback_test=_fallback_split(paths)
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            required=("samples","columns","callback_inputs","callback_input_columns",
                      "callback_future_commands","callback_future_offsets_s")
            missing=[name for name in required if name not in data.files]
            if missing:raise RuntimeError(f"{path}: rerun Step 1; missing {missing}")
            columns={str(name):i for i,name in enumerate(data["columns"])}
            callback_columns={str(name):i for i,name in enumerate(data["callback_input_columns"])}
            absent=[name for name in STATE_NAMES if name not in columns]
            if absent:raise RuntimeError(f"{path}: missing online MPPI-model EKF fields {absent}")
            samples=np.asarray(data["samples"],np.float64)
            callbacks=np.asarray(data["callback_inputs"],np.float64)
            future_commands=np.asarray(data["callback_future_commands"],np.float64)
            offsets=np.asarray(data["callback_future_offsets_s"],np.float64)
        target_offsets=model_dt*np.arange(1,horizon+1)
        target_indices=np.asarray([np.argmin(abs(offsets-value)) for value in target_offsets])
        if np.max(abs(offsets[target_indices]-target_offsets))>1e-7:
            raise RuntimeError(f"{path}: no {model_dt:g} s callback target grid")
        order=np.argsort(samples[:,columns["t"]]);sample_t=samples[order,columns["t"]]
        state=samples[order][:,[columns[name] for name in STATE_NAMES]]
        keep=np.r_[True,np.diff(sample_t)>1e-9];sample_t,state=sample_t[keep],state[keep]
        anchor_t=callbacks[:,callback_columns["t"]]
        initial=_interpolate(sample_t,state,anchor_t)
        target=_interpolate(sample_t,state,anchor_t[:,None]+target_offsets[None])
        commands=np.empty((len(callbacks),horizon,2),np.float64)
        commands[:,0]=callbacks[:,[callback_columns["steer_cmd"],callback_columns["speed_cmd"]]]
        if horizon>1:commands[:,1:]=future_commands[:,target_indices[:-1]]
        history=callbacks[:,[callback_columns[name] for name in HISTORY_NAMES]]
        imu=callbacks[:,[callback_columns["imu_ax"],callback_columns["imu_ay"]]]
        actuator=callbacks[:,[callback_columns["applied_steer"],callback_columns["speed_reference"]]]
        finite=(np.isfinite(initial).all(1)&np.isfinite(target).all((1,2))&
                np.isfinite(commands).all((1,2))&np.isfinite(history).all(1)&
                np.isfinite(imu).all(1)&np.isfinite(actuator).all(1))
        group=_bag_group(path.name)
        split=(2 if path.name in TEST_NAMES or group in fallback_test else
               1 if path.name in VAL_NAMES or group in fallback_val else 0)
        records.append(dict(anchor_time=anchor_t[finite],
            initial_pose=initial[finite,:3],initial_state=initial[finite,3:],
            target_pose=target[finite,:,:3],target_state=target[finite,:,3:],
            commands=commands[finite],history=history[finite],imu=imu[finite],
            actuator=actuator[finite],split=np.full(finite.sum(),split,np.int8),
            bag_name=np.full(finite.sum(),path.name)))
    if not records:raise RuntimeError(f"{directory}: no Step-1 callback archives")
    keys=records[0]
    return {key:np.concatenate([record[key] for record in records]) for key in keys}
