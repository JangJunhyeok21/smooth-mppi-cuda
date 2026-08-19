#!/usr/bin/env python3
"""Inventory IFAC real-car bags before any residual-model training.

This stage intentionally reads metadata only.  Message/header timestamp and
signal-quality audits are performed by the extractor so that metadata clocks
are never mistaken for sensor clocks.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path("/mnt/nas_custom/F1tenth/2026 IFAC")
DATES = {"0807", "0808", "0809", "0810", "0811", "0812"}
OUT = Path(__file__).resolve().parent / "results/real_car_v2_audit"
SIGNALS = {
    "pose": ("pose", "mcl", "mocap", "localization"),
    "odom": ("odom", "velocity", "wheel"),
    "imu": ("imu",),
    "planner_command": ("ackermann_cmd",),
    "applied_command": ("/drive", "motor/speed", "servo/position", "servo_position_command"),
    "actuator_feedback": ("vesc", "servo", "wheel", "motor"),
    "mode_safety": ("teleop", "joy", "estop", "mode", "safety"),
}


def date_group(path: Path):
    for part in path.parts:
        if part in DATES:
            return part
    return None


def main():
    rows=[]
    for meta_path in sorted(ROOT.rglob("metadata.yaml")):
        group=date_group(meta_path)
        if group not in DATES:
            continue
        raw=yaml.safe_load(meta_path.read_text())
        info=raw.get("rosbag2_bagfile_information", raw)
        duration_ns=int((info.get("duration") or {}).get("nanoseconds",0))
        duration=duration_ns*1e-9
        start_ns=int((info.get("starting_time") or {}).get("nanoseconds_since_epoch",0))
        topics=[]
        for entry in info.get("topics_with_message_count",[]):
            md=entry.get("topic_metadata",{})
            count=int(entry.get("message_count",0)); rate=count/duration if duration>0 else 0.0
            topics.append({"name":md.get("name"),"type":md.get("type"),
                           "serialization":md.get("serialization_format"),
                           "count":count,"metadata_hz":rate})
        nonzero=[t["metadata_hz"] for t in topics if t["count"]>0]
        available={key:[t["name"] for t in topics if any(token in (t["name"] or "").lower()
                    for token in tokens)] for key,tokens in SIGNALS.items()}
        storage=info.get("storage_identifier")
        files=list(info.get("relative_file_paths",[]) or [])
        if not files:
            files=[p.name for p in sorted(meta_path.parent.glob("*.db3"))]
            files += [p.name for p in sorted(meta_path.parent.glob("*.mcap"))]
        rows.append({
            "date_group":group,"session":str(meta_path.parent),"storage":storage,
            "files":files,"start_epoch_s":start_ns*1e-9,
            "start_utc":datetime.fromtimestamp(start_ns*1e-9,tz=timezone.utc).isoformat() if start_ns else None,
            "end_epoch_s":start_ns*1e-9+duration,"duration_s":duration,
            "message_count":int(info.get("message_count",sum(t["count"] for t in topics))),
            "topic_count":len(topics),
            "mean_nonzero_topic_hz":sum(nonzero)/len(nonzero) if nonzero else 0.0,
            "min_nonzero_topic_hz":min(nonzero) if nonzero else 0.0,
            "available":available,"topics":topics,
            "quality_flags":[],"training_eligible":None,
            "exclusion_reason":"pending header-time/signal/manual/collision audit",
        })
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"bag_inventory.json").write_text(json.dumps({"root":str(ROOT),"sessions":rows},indent=2)+"\n")
    lines=["# IFAC 0807–0812 rosbag metadata inventory","",
           "Rates below are metadata count/duration estimates, not header-time validation.","",
           "|date|session|format|duration s|messages|topics|mean Hz|min Hz|pose|odom|IMU|planner cmd|applied/actuator|status|",
           "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|"]
    for r in rows:
        av=r["available"]
        brief=lambda xs:", ".join(xs) if xs else "—"
        lines.append("|{date_group}|{name}|{storage}|{duration_s:.2f}|{message_count}|{topic_count}|{mean_nonzero_topic_hz:.2f}|{min_nonzero_topic_hz:.3f}|{pose}|{odom}|{imu}|{pc}|{ap}|pending deep audit|".format(
            **r,name=Path(r["session"]).name,pose=brief(av["pose"]),odom=brief(av["odom"]),imu=brief(av["imu"]),pc=brief(av["planner_command"]),ap=brief(av["applied_command"]+av["actuator_feedback"])))
    missing=sorted(DATES-{r["date_group"] for r in rows})
    lines += ["",f"Missing date groups: {', '.join(missing) if missing else 'none'}"]
    (OUT/"bag_inventory.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"sessions":len(rows),"by_date":{d:sum(r['date_group']==d for r in rows) for d in sorted(DATES)},"output":str(OUT)},indent=2))


if __name__ == "__main__":
    main()
