#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SMPPI 최적 파라미터 다중 맵 반복 검증.

smppi_auto_tuner 의 인프라(스택 관리·에피소드 실행·메트릭 측정)를 재사용해,
고정된 파라미터로 맵마다 N 에피소드(에피소드당 1랩+추월 시나리오)를 반복 실행하고
랩타임 평균/표준편차·완주율·추월율·충돌 횟수를 집계한다.

실행:
  source install/setup.bash
  python3 src/control/smppi_cuda_controller/scripts/smppi_validate.py \
      --maps iccas2025,icra2025,map1 --episodes 10 \
      --params-from /home/user/capstone_ws/tuning_results/full_run
"""

import argparse
import csv
import math
import os
import statistics
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smppi_auto_tuner as tuner  # noqa: E402

# trials.csv 에서 파라미터가 아닌 메트릭 컬럼 (나머지 전부를 파라미터로 간주)
METRIC_COLS = {"trial", "status", "objective", "lap_time", "lap_times",
               "collisions", "progress", "min_opp_dist", "fsm_states",
               "lap_time_ep2", "status_ep2"}
# ROS 파라미터 타입 불일치 방지: int 로 선언된 파라미터만 int 로 전달
INT_PARAMS = {"num_samples", "fsm_side_confirm_ticks",
              "static_obs_min_hits", "static_obs_miss_limit"}


def load_best_params(results_dir, trial_no=None):
    """튜닝 결과 trials.csv 에서 파라미터를 읽는다.

    trial_no 지정 시 해당 trial, 미지정 시 최단 랩 완주 trial.
    파라미터 컬럼은 헤더에서 동적 도출 (솔로/추월 탐색 공간 모두 지원).
    """
    path = os.path.join(results_dir, "trials.csv")
    rows = [r for r in csv.DictReader(open(path))
            if r["status"] == "finished" and r["lap_time"]]
    if not rows:
        sys.exit(f"완주 trial 이 없습니다: {path}")
    if trial_no is not None:
        cand = [r for r in rows if int(r["trial"]) == trial_no]
        if not cand:
            sys.exit(f"trial #{trial_no} 이 완주 목록에 없습니다: {path}")
        best = cand[0]
    else:
        best = min(rows, key=lambda r: float(r["lap_time"]))
    params = {}
    for k, v in best.items():
        if k in METRIC_COLS or v is None or v == "":
            continue
        params[k] = int(float(v)) if k in INT_PARAMS else float(v)
    return params, int(best["trial"]), float(best["lap_time"])


def wait_stack_ready(node, timeout=45.0, solo=False):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s = node.snapshot()
        now = time.monotonic()
        if now - s["ego_stamp"] < 0.5 and (solo or now - s["opp_stamp"] < 0.5):
            return True
        time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser(description="SMPPI 최적 파라미터 다중 맵 검증")
    ap.add_argument("--maps", type=str, default="iccas2025,icra2025,map1")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--solo", action="store_true",
                    help="상대차 없이 단독 주행 검증 (use_car1:=false)")
    ap.add_argument("--laps", type=int, default=None,
                    help="에피소드당 랩 수, 평균 랩타임 평가 (기본: solo 3 / 추월 1)")
    ap.add_argument("--opp-gap", type=float, default=6.0)
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="에피소드 제한시간 하한 [s] (맵 길이에 따라 자동 증가)")
    ap.add_argument("--params-from", type=str,
                    default=os.path.join(tuner.WS, "tuning_results", "full_run"))
    ap.add_argument("--trial", type=int, default=None,
                    help="검증할 trial 번호 (기본: 최단 랩 완주 trial)")
    ap.add_argument("--extra-param", action="append", default=[],
                    help="추가 파라미터 오버라이드 key=value (반복 지정 가능)")
    ap.add_argument("--results-dir", type=str, default=None)
    args = ap.parse_args()
    if args.laps is None:
        args.laps = 3 if args.solo else 1

    if "ROS_DISTRO" not in os.environ:
        sys.exit("ROS 환경이 없습니다. 먼저 `source install/setup.bash` 후 실행하세요.")

    import subprocess
    chk = subprocess.run(["pgrep", "-f", "racecar_simulator.*simulator|simulator.launch"],
                         capture_output=True, text=True)
    if chk.stdout.strip():
        sys.exit("racecar_simulator 가 이미 실행 중입니다. 종료 후 다시 실행하세요.")

    results_dir = args.results_dir or os.path.join(
        tuner.WS, "tuning_results", datetime.now().strftime("validate_%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)

    def log(msg):
        print(f"[validate {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    params, best_trial, best_lap = load_best_params(args.params_from, args.trial)
    for kv in args.extra_param:
        k, _, v = kv.partition("=")
        try:
            params[k] = int(v) if k in INT_PARAMS else float(v)
        except ValueError:
            params[k] = v
    log(f"검증 파라미터 (trial #{best_trial}, 튜닝 랩 {best_lap:.2f}s):")
    for k, v in params.items():
        log(f"  {k} = {v}")

    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    for m in maps:
        csv_p = os.path.join(tuner.SIM_MAPS_DIR, m, f"{m}_centerline.csv")
        if not os.path.isfile(csv_p):
            sys.exit(f"centerline 없음: {csv_p}")

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    rclpy.init()
    node = tuner.make_tuner_node(tuner.Centerline(
        os.path.join(tuner.SIM_MAPS_DIR, maps[0], f"{maps[0]}_centerline.csv")),
        solo=args.solo)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    ep_fields = ["map", "episode", "status", "lap_time", "lap_times",
                 "collisions", "progress", "min_opp_dist", "fsm_states"]
    ep_csv = os.path.join(results_dir, "episodes.csv")
    with open(ep_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=ep_fields).writeheader()

    summary = []
    stack = None
    runner = None
    try:
        for m in maps:
            map_dir = os.path.join(results_dir, m)
            os.makedirs(map_dir, exist_ok=True)
            cl = tuner.Centerline(
                os.path.join(tuner.SIM_MAPS_DIR, m, f"{m}_centerline.csv"))
            # 맵 길이에 맞춰 제한시간 조정 (상대차 페이스 2 m/s 로 한 바퀴 + 여유)
            ep_timeout = max(args.timeout, 1.3 * cl.L / 2.0)
            log(f"=== 맵 {m}: L={cl.L:.1f} m, 에피소드 {args.episodes}회, "
                f"timeout {ep_timeout:.0f}s ===")

            with node.lock:
                node.cl = cl
            sim_launch = tuner.write_sim_launch(map_dir, m, solo=args.solo)
            stack = tuner.StackManager(map_dir, log, sim_launch,
                                       os.path.join(tuner.SIM_MAPS_DIR, m,
                                                    f"{m}_centerline.csv"),
                                       solo=args.solo)
            stack.start()
            if not wait_stack_ready(node, solo=args.solo):
                log(f"맵 {m}: 스택 기동 실패 — 건너뜀")
                stack.stop()
                stack = None
                continue

            run_args = SimpleNamespace(opp_gap=args.opp_gap, timeout=ep_timeout,
                                       solo=args.solo, laps=args.laps)
            runner = tuner.TrialRunner(node, stack, cl, run_args, map_dir, log)

            eps = []
            for ep in range(args.episodes):
                metrics = runner.run(params, ep)
                eps.append(metrics)
                lap = f"{metrics['lap_time']:.2f}s" if metrics["lap_time"] else "-"
                lap_detail = metrics.get("lap_times", "")
                overtook = "OVERTAKE" in metrics.get("fsm_states", "")
                log(f"[{m}] ep {ep}: {metrics['status']}, 평균lap={lap}"
                    + (f" (랩별 {lap_detail})" if lap_detail else "")
                    + f", 충돌={metrics['collisions']}, 진행률={metrics['progress']:.2f}"
                    + ("" if args.solo else f", 추월={'O' if overtook else 'X'}"))
                with open(ep_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=ep_fields).writerow(
                        {"map": m, "episode": ep, **{k: metrics.get(k) for k in
                         ["status", "lap_time", "lap_times", "collisions",
                          "progress", "min_opp_dist", "fsm_states"]}})

            stack.stop()
            stack = None
            time.sleep(2.0)

            fin = [e for e in eps if e["status"] == "finished"]
            laps = [e["lap_time"] for e in fin]
            ovt = [e for e in fin if "OVERTAKE" in e.get("fsm_states", "")]
            coll = sum(e["collisions"] for e in eps)
            summary.append({
                "map": m, "episodes": len(eps), "finished": len(fin),
                "collision_eps": sum(1 for e in eps if e["collisions"] > 0),
                "total_collisions": coll, "overtake_eps": len(ovt),
                "lap_mean": round(statistics.mean(laps), 2) if laps else None,
                "lap_std": round(statistics.stdev(laps), 2) if len(laps) > 1 else 0.0,
                "lap_best": round(min(laps), 2) if laps else None,
                "lap_worst": round(max(laps), 2) if laps else None,
            })

        sum_csv = os.path.join(results_dir, "summary.csv")
        with open(sum_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)

        log("========== 검증 요약 ==========")
        for s in summary:
            log(f"{s['map']:>10}: 완주 {s['finished']}/{s['episodes']}, "
                f"충돌 에피소드 {s['collision_eps']}, 추월 {s['overtake_eps']}, "
                f"lap {s['lap_mean']}±{s['lap_std']}s "
                f"(best {s['lap_best']} / worst {s['lap_worst']})")
        log(f"에피소드 기록: {ep_csv}")
        log(f"요약: {sum_csv}")
    except KeyboardInterrupt:
        log("사용자 중단")
    finally:
        if runner is not None:
            runner._stop_controller()
        if stack is not None:
            stack.stop()
        try:
            executor.shutdown(timeout_sec=2.0)
            spin_thread.join(timeout=3.0)
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
