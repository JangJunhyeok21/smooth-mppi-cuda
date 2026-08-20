#!/usr/bin/env python3
"""Cold-start Map1 lap-time A/B benchmark for a small set of MPPI costs."""
import json, os, signal, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SIM_ROOT=ROOT/"f1tenth_gym_ros"
OUT=ROOT/"model_tuning/results/map1_simulator_gru_controller_confirmation"
VARIANTS=(
    ("safe25",{"objective_mode":"mpcc","max_speed":"2.5","q_progress":"10.0",
               "q_escape_vel":"0.0","q_rear_slip":"1000.0","rear_slip_soft_limit_deg":"5.0",
               "q_lat_g":"60.0","q_boundary_slack":"10000.0"}),
    ("safe30_progress15",{"objective_mode":"mpcc","max_speed":"3.0","q_progress":"15.0",
               "q_escape_vel":"2.0","q_rear_slip":"1800.0","rear_slip_soft_limit_deg":"5.0",
               "q_lat_g":"90.0","q_heading":"4.0","q_boundary_slack":"15000.0"}),
)

def start(command,cwd,log):
    stream=log.open("w")
    process=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,
                             start_new_session=True,text=True)
    return process,stream

def stop(item):
    if item is None:return
    process,stream=item
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGINT)
        try:process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=4)
    stream.close()

def main():
    OUT.mkdir(parents=True,exist_ok=True);results=[]
    existing_nodes=subprocess.run(["ros2","node","list"],capture_output=True,text=True,check=True).stdout
    reuse_simulator="/bridge" in existing_nodes.splitlines()
    for name,overrides in VARIANTS:
        directory=OUT/name;directory.mkdir(parents=True,exist_ok=True)
        for filename in ("summary.txt","map1_lap_data.npz","map1_mppi_prediction_vs_simulator.png"):
            path=directory/filename
            if path.exists():path.unlink()
        simulator=recorder=controller=None
        try:
            if not reuse_simulator:
                simulator=start(["ros2","launch","f1tenth_gym_ros","gym_bridge_launch.py"],
                                SIM_ROOT,directory/"sim.log")
                time.sleep(3.0)
            subprocess.run(["ros2","topic","pub","--once","/drive",
                "ackermann_msgs/msg/AckermannDriveStamped","{drive: {speed: 0.0, steering_angle: 0.0}}"],
                check=True,stdout=subprocess.DEVNULL)
            subprocess.run(["ros2","topic","pub","--once","/initialpose",
                "geometry_msgs/msg/PoseWithCovarianceStamped",
                "{header: {frame_id: map}, pose: {pose: {position: {x: -1.796, y: -5.478}, orientation: {z: 0.6965, w: 0.7176}}}}"],
                check=True,stdout=subprocess.DEVNULL)
            time.sleep(1.0)
            recorder=start(["/usr/bin/python3","scripts/record_map1_lap.py","--laps","2",
                            "--timeout","60","--output",str(directory)],ROOT,directory/"recorder.log")
            time.sleep(.4)
            command=["ros2","run","smppi_cuda_controller","smppi_node","--ros-args",
                     "--params-file",str(ROOT/"config/params.yaml"),
                     "-p","is_simulation:=true","-p","obstacle_avoidance_enabled:=false"]
            for key,value in overrides.items():command.extend(("-p",f"{key}:={value}"))
            controller=start(command,ROOT,directory/"controller.log")
            deadline=time.monotonic()+65
            while not (directory/"summary.txt").exists() and time.monotonic()<deadline:
                time.sleep(.1)
            if not (directory/"summary.txt").exists():raise TimeoutError("recorder timeout")
            fields=dict(line.strip().split("=",1) for line in
                        (directory/"summary.txt").read_text().splitlines() if "=" in line)
            row={"variant":name,"overrides":overrides,"status":fields["status"],
                 "duration_s":float(fields["duration_s"]),"lap_ratio":float(fields["lap_ratio"])}
            row["seconds_per_lap"]=row["duration_s"]/row["lap_ratio"] if row["lap_ratio"] else 999.
        except Exception as error:
            row={"variant":name,"overrides":overrides,"status":"harness_error",
                 "seconds_per_lap":999.,"error":repr(error)}
        finally:
            stop(controller);stop(recorder);stop(simulator)
        results.append(row);print(json.dumps(row),flush=True)
        (OUT/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(sorted(results,key=lambda row:row["seconds_per_lap"]),indent=2))

if __name__=="__main__":main()
