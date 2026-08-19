#!/usr/bin/env python3
"""Deep timestamp and signal-path audit for the IFAC bag inventory."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

ROOT=Path(__file__).resolve().parents[1]
INVENTORY=ROOT/"model_tuning/results/real_car_v2_audit/bag_inventory.json"
OUT=INVENTORY.parent/"timestamp_signal_audit.json"
SELECT=("/newmcl_pose","/odom","/estimate_odom","/imu/data","/imu/data_raw","/lpf_imu",
        "/ackermann_cmd","/drive","/commands/motor/speed","/commands/servo/position",
        "/sensors/servo_position_command","/teleop","/joy","/initialpose","/tf_static")

def stamp(msg):
    h=getattr(msg,"header",None); s=getattr(h,"stamp",None)
    if s is None or (s.sec==0 and s.nanosec==0): return None
    return float(s.sec)+float(s.nanosec)*1e-9

def pose_xyyaw(msg):
    p=getattr(msg,"pose",None)
    if hasattr(p,"pose"): p=p.pose
    if not hasattr(p,"position"): return None
    q=p.orientation
    yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
    return p.position.x,p.position.y,yaw

def command(msg):
    d=getattr(msg,"drive",None)
    return (d.steering_angle,d.speed,d.acceleration) if d is not None else None

def summarize_time(records):
    rec=np.array([x[0] for x in records]); hdr=np.array([x[1] for x in records if x[1] is not None])
    use=hdr if len(hdr)>=2 else rec
    dif=np.diff(use); pos=dif[dif>0]
    paired=np.array([r-h for r,h,*_ in records if h is not None])
    return {"count":len(records),"header_count":len(hdr),"clock":"header" if len(hdr)>=2 else "record",
            "monotonic_violations":int(np.sum(dif<0)),"duplicate_timestamps":int(np.sum(abs(dif)<=1e-9)),
            "median_hz":float(1/np.median(pos)) if len(pos) else 0.0,
            "min_instantaneous_hz":float(1/np.max(pos)) if len(pos) else 0.0,
            "max_gap_s":float(np.max(pos)) if len(pos) else 0.0,
            "record_minus_header_s":({"median":float(np.median(paired)),"p05":float(np.quantile(paired,.05)),
              "p95":float(np.quantile(paired,.95)),"min":float(paired.min()),"max":float(paired.max())} if len(paired) else None)}

def main():
    inv=json.loads(INVENTORY.read_text()); reports=[]
    for si,row in enumerate(inv["sessions"],1):
        path=Path(row["session"]); report={"session":str(path),"date_group":row["date_group"],"read_error":None}
        try:
            reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(path),storage_id=row["storage"] or ""),rosbag2_py.ConverterOptions("",""))
            types={x.name:x.type for x in reader.get_all_topics_and_types()}; wanted=set(types)&set(SELECT)
            records={t:[] for t in wanted}; frames={t:set() for t in wanted}
            while reader.has_next():
                topic,data,record_ns=reader.read_next()
                if topic not in wanted: continue
                msg=deserialize_message(data,get_message(types[topic])); hs=stamp(msg); extra=None
                if topic=="/newmcl_pose": extra=pose_xyyaw(msg)
                elif topic in ("/ackermann_cmd","/drive","/teleop"): extra=command(msg)
                elif topic in ("/odom","/estimate_odom"):
                    tw=msg.twist.twist; extra=(tw.linear.x,tw.linear.y,tw.angular.z)
                elif "imu" in topic:
                    extra=(msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z,
                           msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z)
                elif hasattr(msg,"data"): extra=(float(msg.data),)
                frame=getattr(getattr(msg,"header",None),"frame_id","")
                if frame:frames[topic].add(frame)
                records[topic].append((record_ns*1e-9,hs,extra))
            stats={t:summarize_time(v) for t,v in records.items()};flags=[]
            for t,s in stats.items():
                s["frames"]=sorted(frames[t])
                if s["monotonic_violations"]:flags.append(f"{t}: non-monotonic header/record time")
                if s["duplicate_timestamps"]:flags.append(f"{t}: duplicate timestamps")
                if s["max_gap_s"]>.2:flags.append(f"{t}: gap {s['max_gap_s']:.3f}s")
                lag=s["record_minus_header_s"]
                if lag and lag["p95"]-lag["p05"]>.05:flags.append(f"{t}: variable record-header delay")
            # Localization discontinuities using header time when available.
            pose=records.get("/newmcl_pose",[]); jumps=[]
            for a,b in zip(pose,pose[1:]):
                if a[2] is None or b[2] is None:continue
                dt=(b[1] if b[1] is not None else b[0])-(a[1] if a[1] is not None else a[0])
                dist=math.hypot(b[2][0]-a[2][0],b[2][1]-a[2][1]);dyaw=abs(math.atan2(math.sin(b[2][2]-a[2][2]),math.cos(b[2][2]-a[2][2])))
                if dt>0 and (dist>.30 or dyaw>.5):jumps.append({"time":b[1] or b[0],"dt":dt,"distance_m":dist,"yaw_rad":dyaw})
            if jumps: flags.append(f"localization jumps: {len(jumps)}")
            # Exact autonomous equality at nearest command timestamps.
            ack=records.get("/ackermann_cmd",[]); drive=records.get("/drive",[]); auto=None
            if ack and drive:
                dtimes=np.array([x[1] if x[1] is not None else x[0] for x in drive]);matches=[]
                for x in ack:
                    t=x[1] if x[1] is not None else x[0]; j=int(np.argmin(abs(dtimes-t)))
                    ac,dc=x[2],drive[j][2]
                    matches.append(bool(ac and dc and abs(ac[0]-dc[0])<1e-4 and abs(ac[1]-dc[1])<1e-4 and abs(dtimes[j]-t)<.1))
                auto={"ack_samples":len(ack),"matching_samples":int(sum(matches)),"matching_fraction":float(np.mean(matches))}
                if auto["matching_fraction"]<.8:flags.append("ackermann/drive mismatch or manual mux intervals")
            # Conservative collision clues, not automatic exclusion by themselves.
            imu=records.get("/imu/data",[]); impact=[]
            for x in imu:
                if x[2] and math.sqrt(sum(v*v for v in x[2][3:]))>20:impact.append(x[1] or x[0])
            odom=records.get("/odom",[]); reverse=[x[1] or x[0] for x in odom if x[2] and x[2][0]<-.15]
            if impact:flags.append(f"IMU impact candidates: {len(impact)}")
            if reverse:flags.append(f"reverse/recovery samples: {len(reverse)}")
            report.update({"topics":stats,"available_topics":sorted(types),"localization_jumps":jumps[:100],
                           "autonomous_command_match":auto,"impact_candidate_count":len(impact),
                           "reverse_sample_count":len(reverse),"quality_flags":flags})
        except Exception as exc:
            report["read_error"]=f"{type(exc).__name__}: {exc}";report["quality_flags"]=["unreadable bag"]
        reports.append(report);print(f"[{si}/{len(inv['sessions'])}] {path.name}: {report['quality_flags']}",flush=True)
    OUT.write_text(json.dumps({"sessions":reports},indent=2,allow_nan=False)+"\n")
    print(OUT)
if __name__=="__main__":main()
