#!/usr/bin/env python3
"""Recompute causal classic-model KF fields for existing Step-1 NPZ files.

Source archives are never overwritten.  This is the fast outer-loop equivalent
of rereading the same bags when only classic parameters have changed.
"""
from pathlib import Path
import argparse
import datetime as dtlib
import json
import re
import sys

import numpy as np
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from classic_model_kalman_filter import filter_classic_segment
from contract import ClassicModelParameters
from step_1_extract_data import causal_ema,causal_mcl_body_vy

IMU_CONVENTION_CUTOFF=dtlib.date(2026,8,17)
LEGACY_IMU_SIGNS=np.array((-1.0,1.0,-1.0))
CURRENT_IMU_SIGNS=np.array((1.0,1.0,1.0))


def source_date(path,archive=None):
    if archive is not None and "recording_date" in archive.files:
        return dtlib.date.fromisoformat(str(archive["recording_date"]))
    match=re.search(r"(?:rosbag2_)?(20\d{2})[_-](\d{2})[_-](\d{2})",Path(path).name)
    return dtlib.date(*(int(value) for value in match.groups())) if match else None


def imu_signs_for_source(path,archive):
    date=source_date(path,archive)
    if date is None:raise RuntimeError(f"{path}: cannot determine IMU convention date")
    expected=CURRENT_IMU_SIGNS if date>=IMU_CONVENTION_CUTOFF else LEGACY_IMU_SIGNS
    if "imu_axis_signs" in archive.files and not np.array_equal(archive["imu_axis_signs"],expected):
        raise RuntimeError(f"{path}: stored IMU signs conflict with recording date")
    return expected.copy(),date

DEFAULT_SOURCE=ROOT/"model_tuning/data/ifac0810_0819_autonomous_physics_clean"
DEFAULT_OUTPUT=ROOT/"model_tuning/data/ifac0817_0820_classic_kf"
KF_COLUMNS=("kf_x","kf_y","kf_yaw","kf_vx","kf_vy","kf_yaw_rate","kf_ax","kf_ay")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--date-from",default="2026-08-17")
    parser.add_argument("--date-to",default="2026-08-20")
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    digest=ClassicModelParameters.from_mapping(cfg).digest();manifest=[]
    for path in sorted(args.source.glob("*.npz")):
        source=np.load(path);date=source_date(path,source)
        if date is None or not (args.date_from<=date.isoformat()<=args.date_to):continue
        if "samples" not in source.files or "columns" not in source.files:continue
        columns=list(map(str,source["columns"]));names={name:i for i,name in enumerate(columns)}
        required=("t","x","y","yaw","vx","steer","speed_cmd","bag_id","imu_wz","imu_ax","imu_ay")
        if any(name not in names for name in required):continue
        samples=np.asarray(source["samples"],float);base_indices=[i for i,name in enumerate(columns) if name not in KF_COLUMNS]
        samples=samples[:,base_indices];columns=[columns[i] for i in base_indices];names={name:i for i,name in enumerate(columns)}
        signs,_=imu_signs_for_source(path,source);kf=np.full((len(samples),8),np.nan);dt=float(source["dt"])
        for segment in np.unique(samples[:,names["bag_id"]].astype(int)):
            idx=np.flatnonzero(samples[:,names["bag_id"]].astype(int)==segment);part=samples[idx]
            if len(part)<10:continue
            mcl_vy=causal_mcl_body_vy(part[:,names["t"]],part[:,names["x"]],
                part[:,names["y"]],part[:,names["yaw"]],
                float(cfg.get("kf_pose_vy_window_s",.12)))
            alpha=float(cfg.get("imu_ema_alpha",.25))
            gyro=causal_ema(signs[0]*part[:,names["imu_wz"]]-float(cfg.get("imu_wz_bias",0.)),alpha)
            ax=causal_ema(signs[1]*part[:,names["imu_ax"]]-float(cfg.get("imu_ax_bias",0.)),alpha)
            ay=causal_ema(signs[2]*part[:,names["imu_ay"]]-float(cfg.get("imu_ay_bias",0.)),alpha)
            result=filter_classic_segment(part[:,names["x"]],part[:,names["y"]],part[:,names["yaw"]],part[:,names["vx"]],mcl_vy,gyro,ax,ay,part[:,names["steer"]],part[:,names["speed_cmd"]],dt,cfg)
            kf[idx,:6]=result["state"];kf[idx,6:]=result["acceleration"]
        payload={key:source[key] for key in source.files if key not in ("samples","columns","kf_parameter_hash")}
        snapshot={**ClassicModelParameters.from_mapping(cfg).runtime_updates(),
            "classic_kf_process_var":list(map(float,cfg["classic_kf_process_var"])),
            "classic_kf_measurement_var":list(map(float,cfg["classic_kf_measurement_var"])),
            "classic_kf_initial_var":list(map(float,cfg["classic_kf_initial_var"])),
            "kf_pose_vy_window_s":float(cfg.get("kf_pose_vy_window_s",.12))}
        payload.update(samples=np.c_[samples,kf],columns=np.asarray(columns+list(KF_COLUMNS)),
            kf_parameter_hash=np.array(digest),
            kf_config_snapshot_json=np.array(json.dumps(snapshot,sort_keys=True)))
        output=args.out/path.name;np.savez_compressed(output,**payload)
        manifest.append({"source":str(path),"output":str(output),"date":date.isoformat(),"samples":len(samples)})
    report={"classic_parameter_hash":digest,"files":len(manifest),"samples":sum(q["samples"] for q in manifest),"sources":manifest}
    (args.out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
