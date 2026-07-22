#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SMPPI 자동 파라미터 튜너 — Optuna 기반 랩타임 최적화.

파이프라인 (매 trial 반복):
  1) racecar_simulator 스택(시뮬레이터·map_publisher·상대차 pure pursuit·race_stat)
     + f1_perception + path_publisher 를 백그라운드 상시 실행
  2) ego 를 /initialpose 로 SF 라인 1 m 전방에 리셋
  3) smppi_node_fsm 을 trial 파라미터(-p 오버라이드)로 subprocess 실행
  4) ego 출발 감지 → 상대차를 /goal_pose 로 ego 전방 opp_gap(m) 지점에 텔레포트
  5) SF 통과 → 재통과 시간 = Lap Time, /collision0 상승 엣지 = 충돌(벽·상대차 공통)
  6) 노드 종료·정지 후 Optuna 에 목적값 보고

목적 함수:
  완주(무충돌)      → lap_time [s]
  충돌              → 1000 - 500·(진행률)   (>= 500, 사실상 무한대 페널티)
  타임아웃/정체     → 600  - 300·(진행률)
  제어기 비정상 종료 → 1200

설계 노트:
  * /opponent_odom 은 opponent_tracker 대신 본 스크립트가 시뮬레이터
    ground-truth(/odom1, map 좌표)에 LiDAR FOV(270°)·거리(0.3~8 m) 제한을
    적용해 발행한다. opponent_tracker 는 laser 프레임 상대좌표를 그대로
    내보내는데 FSM 은 map 좌표를 기대하고, 폭 1.1 m 트랙에서는 최근접
    클러스터가 벽이 되기 쉬워 튜닝 신호를 오염시키기 때문.
  * horizon(T=50)·dt(0.035 s)·Butterworth 차단주파수는 소스 하드코딩이라
    리빌드 없이는 튜닝 불가 → 탐색 공간에서 제외.
  * num_samples 는 노드 내부에서 int16 에 저장되므로 상한 16000 로 제한.

실행:
  source install/setup.bash
  # 추월 시나리오 튜닝
  python3 src/control/smppi_cuda_controller/scripts/smppi_auto_tuner.py \
      --trials 60 --timeout 60 --opp-gap 6.0
  # 단독 주행(SOLO) 랩타임 튜닝 — 상대차 없음(use_car1:=false), 탐색 공간이 다름
  python3 src/control/smppi_cuda_controller/scripts/smppi_auto_tuner.py \
      --solo --trials 80 --episodes-per-trial 2 --map map1

솔로 모드 노트:
  * 속도는 목표 추종이 아니라 비용 트레이드오프(q_v 보상 vs 경계 fault)로 결정되고
    max_speed 가 유일한 하드 캡. target_speed 는 SOLO 에서 max_vel(약한 감속 천장)
    로만 쓰여 매 trial max_speed 와 동일하게 묶는다.
  * trial 당 --episodes-per-trial 에피소드를 돌려 worst 를 채택 (낙관 편향 방지).
"""

import argparse
import csv
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

WS = "/home/user/capstone_ws"
BASE_YAML = os.path.join(WS, "src/control/smppi_cuda_controller/config/params.yaml")
SIM_MAPS_DIR = os.path.join(
    WS, "src/racecar_simulator/src/racecar_simulator/maps/f1tenth_racetracks")

# simulator.launch.py 는 map1 이 하드코딩되어 있어, 맵을 주입할 수 있는
# 런치 파일을 결과 디렉토리에 생성해 사용한다 (rviz 포함, 구성은 원본과 동일).
SIM_LAUNCH_TEMPLATE = '''\
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory("racecar_simulator")
    rviz_config_file = os.path.join(pkg_dir, "params", "simulator.rviz")
    simulation_config_file = os.path.join(pkg_dir, "params", "simulation.yaml")
    race_stat_config_file = os.path.join(pkg_dir, "params", "race_stats.yaml")
    car0_xacro_file = os.path.join(pkg_dir, "params", "racecar0.xacro")
    car1_xacro_file = os.path.join(pkg_dir, "params", "racecar1.xacro")

    return LaunchDescription([
        Node(package="racecar_simulator", executable="simulator",
             name="racecar_simulator", output="screen",
             parameters=[simulation_config_file, {{"use_car1": {use_car1}}}]),
        Node(package="racecar_simulator", executable="map_publisher",
             name="map_publisher", output="screen",
             parameters=[{{"map_img_file_path": "{map_img}"}},
                         {{"map_yaml_file_path": "{map_yaml}"}},
                         {{"race_line_file_path": "{map_center}"}}]),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             name="robot_state_publisher", namespace="racecar0", output="screen",
             parameters=[{{"robot_description": ParameterValue(
                 Command(["xacro ", str(car0_xacro_file), " prefix:=0"]),
                 value_type=str)}}]),
{opponent_nodes}        Node(package="racecar_simulator", executable="race_stat",
             name="race_stat", output="screen",
             parameters=[race_stat_config_file]),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", rviz_config_file], output="screen"),
    ])
'''

# 추월 모드에서만 포함되는 상대차 노드 (format 값으로 삽입되므로 중괄호는 단일)
OPPONENT_NODES = '''\
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             name="robot_state_publisher", namespace="racecar1", output="screen",
             parameters=[{"robot_description": ParameterValue(
                 Command(["xacro ", str(car1_xacro_file), " prefix:=1"]),
                 value_type=str)}]),
        Node(package="racecar_simulator", executable="pure_pursuit_car1.py",
             name="pure_pursuit_car1", output="screen"),
'''


def write_sim_launch(results_dir, map_name, solo=False):
    map_dir = os.path.join(SIM_MAPS_DIR, map_name)
    path = os.path.join(results_dir, "tuner_sim.launch.py")
    with open(path, "w") as f:
        f.write(SIM_LAUNCH_TEMPLATE.format(
            map_img=os.path.join(map_dir, f"{map_name}_map.pgm"),
            map_yaml=os.path.join(map_dir, f"{map_name}_map.yaml"),
            map_center=os.path.join(map_dir, f"{map_name}_centerline.csv"),
            use_car1="False" if solo else "True",
            opponent_nodes="" if solo else OPPONENT_NODES,
        ))
    return path

# smppi_node_fsm 에 항상 적용할 고정 오버라이드
FIXED_PARAMS = {
    "odom_topic": "/ekf_odom",
    "drive_topic": "/drive",
    "visualize_candidates": False,   # 후보 궤적 시각화 생략 (GPU/대역폭 절약)
    "max_speed": 8.0,                # 실차용 안전 상한(YAML 3.0) 해제 — FSM 속도가 실효 상한
}

# 솔로 모드: max_speed 는 탐색 대상이므로 고정 목록에서 제외.
# q_obs_gauss 는 검증된 trial#42 값으로 고정 (솔로에선 벽/인지 오탐에만 관여).
# min_speed 0.1: YAML 0.5 는 명령 클램프가 항상 v>=0.5 를 강제해 벽 앞
# 전-샘플-fault 상황에서도 멈추지 못하고 크리핑으로 벽에 닿음 → 사실상 정지 허용.
SOLO_FIXED_PARAMS = {
    **{k: v for k, v in FIXED_PARAMS.items() if k != "max_speed"},
    "q_obs_gauss": 197.33,
    "min_speed": 0.1,
}

# trials.csv 파라미터 컬럼 (모드별 탐색 공간 순서)
OVERTAKE_PARAM_KEYS = ["num_samples", "lambda", "noise_steer_std", "noise_accel_std",
                       "q_du", "q_obs_gauss", "fsm_overtake_speed", "fsm_follow_speed"]
SOLO_PARAM_KEYS = ["max_speed", "target_speed", "q_v", "q_progress", "q_escape_vel",
                   "lambda", "noise_steer_std", "noise_accel_std", "q_du", "q_steer",
                   "num_samples", "collision_radius", "q_dist"]

# LiDAR 흉내 (ground-truth 상대차 검출 게이트)
DETECT_FOV_HALF = math.radians(135.0)   # 270° 스캔
DETECT_RANGE = (0.3, 8.0)               # opponent_tracker 와 동일


def fmt_param(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def kill_group(proc, name, log, sig_first=signal.SIGINT, grace=6.0):
    """프로세스 그룹에 SIGINT → 미종료 시 SIGKILL."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig_first)
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        log(f"{name}: SIGINT 무응답 → SIGKILL")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    except ProcessLookupError:
        pass


class Centerline:
    """시뮬레이터 centerline CSV 기반 호 길이(s) 좌표계."""

    def __init__(self, csv_path):
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        self.xs = data[:, 0].copy()
        self.ys = data[:, 1].copy()
        dx = np.diff(np.append(self.xs, self.xs[0]))
        dy = np.diff(np.append(self.ys, self.ys[0]))
        seg = np.hypot(dx, dy)
        self.s = np.concatenate(([0.0], np.cumsum(seg)))[:-1]  # 각 점의 호 길이
        self.L = float(np.sum(seg))
        self.seg_yaw = np.arctan2(dy, dx)

    def nearest_s(self, x, y):
        i = int(np.argmin((self.xs - x) ** 2 + (self.ys - y) ** 2))
        return float(self.s[i]), i

    def nearest_idx(self, x, y, prev_idx=None, window=10):
        """연속성 기반 최근접점: 이전 인덱스 주변만 탐색해 인접 구간 퇴화 방지."""
        n = len(self.xs)
        if prev_idx is None:
            return int(np.argmin((self.xs - x) ** 2 + (self.ys - y) ** 2))
        idxs = np.arange(prev_idx - window, prev_idx + window + 1) % n
        d2 = (self.xs[idxs] - x) ** 2 + (self.ys[idxs] - y) ** 2
        return int(idxs[np.argmin(d2)])

    def pose_at_s(self, s):
        s = s % self.L
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        i = max(0, min(i, len(self.xs) - 1))
        return float(self.xs[i]), float(self.ys[i]), float(self.seg_yaw[i])


def make_planner_centerline(centerline_csv, out_dir, spacing=0.05,
                            max_ray=3.5, slope=0.3, cell_margin=0.05):
    """플래너(path_publisher) 전용 센터라인 CSV 생성.

    원본 CSV 의 w_tr_*_m 폭이 얇은 내벽(5 cm)을 관통해 측정된 구간이 있어
    (map1 s≈10.5~12 알코브: 실벽 0.6 m 인데 CSV 1.2 m), 경계 모델이 유령
    공간으로 라인을 그리다 벽에 충돌한다. 실제 맵 점유(pgm) 레이캐스트로
    좌/우 폭을 재계산하고, 폭 변화율 제한(벽 구멍·급변 보정) 및 5 cm
    다운샘플(GPU 최근접 탐색 윈도 30점 = 1.5 m 확보)을 적용한다.
    시뮬레이터/랩 측정은 원본 CSV 를 계속 사용한다.
    """
    map_dir = os.path.dirname(centerline_csv)
    name = os.path.basename(centerline_csv).replace("_centerline.csv", "")
    pgm_path = os.path.join(map_dir, f"{name}_map.pgm")
    yaml_path = os.path.join(map_dir, f"{name}_map.yaml")
    if not (os.path.isfile(pgm_path) and os.path.isfile(yaml_path)):
        return centerline_csv

    import yaml as _yaml
    from PIL import Image
    meta = _yaml.safe_load(open(yaml_path))
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    img = np.array(Image.open(pgm_path))
    H, W = img.shape
    occ = img < 100  # 점유(벽)
    # 2셀(0.1 m) 팽창: 얇은 내벽의 1~3셀 구멍(라이다는 통과, 차는 불가)을
    # 메워 레이캐스트 누수를 막고 디지털화 오차를 보수화
    dil = occ.copy()
    for di in range(-2, 3):
        for dj in range(-2, 3):
            dil |= np.roll(np.roll(occ, di, axis=0), dj, axis=1)
    occ = dil

    d = np.genfromtxt(centerline_csv, delimiter=",", skip_header=1)
    xs, ys = d[:, 0], d[:, 1]
    dx = np.diff(np.append(xs, xs[0]))
    dy = np.diff(np.append(ys, ys[0]))
    seg = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(seg)))[:-1]
    L = float(seg.sum())

    n = max(int(L / spacing), 8)
    s_out = np.arange(n) * (L / n)
    xi = np.interp(s_out, s, xs)
    yi = np.interp(s_out, s, ys)
    tx = np.gradient(np.concatenate((xi, xi[:1])))[:-1]
    ty = np.gradient(np.concatenate((yi, yi[:1])))[:-1]
    norm = np.hypot(tx, ty)
    tx, ty = tx / norm, ty / norm
    nx, ny = -ty, tx  # 좌측 법선

    def ray(px, py, ux, uy):
        t = 0.0
        while t < max_ray:
            gx = int((px + ux * t - ox) / res)
            gy = H - 1 - int((py + uy * t - oy) / res)
            if gx < 0 or gy < 0 or gx >= W or gy >= H or occ[gy, gx]:
                return t
            t += 0.02
        return max_ray

    wl = np.array([ray(xi[i], yi[i], nx[i], ny[i]) for i in range(n)])
    wr = np.array([ray(xi[i], yi[i], -nx[i], -ny[i]) for i in range(n)])
    wl = np.maximum(wl - cell_margin, 0.1)
    wr = np.maximum(wr - cell_margin, 0.1)

    # 폭 변화율 제한 (순환, 전/후진 패스): 벽 구멍으로 새어나간 스파이크 제거
    step = slope * (L / n)
    for arr in (wl, wr):
        for _ in range(2):
            for i in range(1, 2 * n):
                arr[i % n] = min(arr[i % n], arr[(i - 1) % n] + step)
            for i in range(2 * n - 1, -1, -1):
                arr[i % n] = min(arr[i % n], arr[(i + 1) % n] + step)

    out_path = os.path.join(out_dir, f"{name}_centerline_planner.csv")
    with open(out_path, "w") as f:
        f.write("x_m,y_m,w_tr_left_m,w_tr_right_m\n")
        for i in range(n):
            f.write(f"{xi[i]:.4f},{yi[i]:.4f},{wl[i]:.3f},{wr[i]:.3f}\n")
    return out_path


class StackManager:
    """시뮬레이터/인지/경로 발행 등 상시 프로세스 관리.

    solo=True 면 perception(+laser TF 브리지)을 생략한다 — 상대차가 없어
    장애물 인지가 불필요하고, 벽 오탐 가능성만 남기 때문.
    """

    def __init__(self, results_dir, log, sim_launch, centerline_csv, solo=False):
        self.results_dir = results_dir
        self.log = log
        self.procs = {}
        self.logfiles = {}
        planner_csv = make_planner_centerline(centerline_csv, results_dir)
        if planner_csv != centerline_csv:
            log(f"플래너용 센터라인 생성(레이캐스트 폭 보정): {planner_csv}")
        centerline_csv = planner_csv
        self.defs = [
            ("sim", ["ros2", "launch", sim_launch]),
            # publish_rate 20 Hz: 제어기 경로 구독이 volatile QoS 라 latched 미수신 —
            # 저주기(1 Hz)면 재시작 직후 ~2 s 를 경로 없이 min_speed 크리핑으로 주행
            # (스폰 직후 벽 충돌의 주원인). 고주기로 블라인드 구간 최소화.
            ("path_pub", ["ros2", "run", "smppi_cuda_controller", "path_publisher",
                          "--ros-args", "--params-file", BASE_YAML,
                          "-p", f"csv_file_path:={centerline_csv}",
                          "-p", "frame_id:=map",
                          "-p", "publish_rate:=20.0"]),
        ]
        if not solo:
            self.defs[1:1] = [
                # 시뮬레이터 laser 프레임(laser_model0) → perception 이 기대하는 laser 프레임 브리지
                ("tf_laser", ["ros2", "run", "tf2_ros", "static_transform_publisher",
                              "--frame-id", "laser_model0", "--child-frame-id", "laser"]),
                ("perception", ["ros2", "launch", "f1_perception", "f1_perception.launch.py"]),
            ]

    def start(self):
        for name, cmd in self.defs:
            lf = open(os.path.join(self.results_dir, f"stack_{name}.log"), "a")
            self.logfiles[name] = lf
            self.procs[name] = subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            self.log(f"스택 시작: {name} (pid {self.procs[name].pid})")
            time.sleep(1.0)  # 시뮬레이터 → 나머지 순차 기동

    def all_alive(self):
        return all(p.poll() is None for p in self.procs.values())

    def dead_members(self):
        return [n for n, p in self.procs.items() if p.poll() is not None]

    def stop(self):
        for name, proc in self.procs.items():
            kill_group(proc, name, self.log)
        for lf in self.logfiles.values():
            lf.close()
        self.procs.clear()
        self.logfiles.clear()

    def restart(self):
        self.log(f"스택 재시작 (사망 노드: {self.dead_members()})")
        self.stop()
        time.sleep(2.0)
        self.start()


def make_tuner_node(centerline, solo=False):
    """rclpy 노드 생성 (모니터링 + 리셋/스폰 + ground-truth 상대차 발행).

    solo=True 면 /opponent_odom 발행을 생략한다 (FSM 은 영구 SOLO 유지).
    """
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool, String
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
    from ackermann_msgs.msg import AckermannDriveStamped

    class TunerNode(Node):
        def __init__(self):
            super().__init__("smppi_auto_tuner")
            self.cl = centerline
            self.lock = threading.Lock()

            # ego / opp 최신 상태
            self.ego_x = self.ego_y = self.ego_yaw = 0.0
            self.ego_v = 0.0
            self.ego_stamp = 0.0
            self.opp_x = self.opp_y = self.opp_yaw = 0.0
            self.opp_vx = self.opp_vy = 0.0
            self.opp_stamp = 0.0

            # 에피소드 상태
            self.tracking = False
            self.s_prev = None
            self.idx_prev = None
            self.cum_progress = 0.0
            self.sf_times = []       # SF 통과 시각 (monotonic)
            self.sf_progress = []    # SF 통과 시점의 누적 진행량
            self.last_sf_t = -1e9
            self.collision_armed = False
            self.collision_count = 0
            self._coll_prev = False
            self.min_opp_dist = float("inf")
            self.fsm_states = set()
            self.fsm_last = ""

            self.create_subscription(Odometry, "/ekf_odom", self._ego_cb, 10)
            self.create_subscription(Odometry, "/odom1", self._opp_cb, 10)
            self.create_subscription(Bool, "/collision0", self._coll_cb, 10)
            self.create_subscription(String, "/fsm/state", self._fsm_cb, 10)

            self.init_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
            self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
            self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

            if not solo:
                self.opp_pub = self.create_publisher(Odometry, "/opponent_odom", 10)
                # ground-truth 기반 FOV 제한 상대차 발행 (opponent_tracker 대체)
                self.create_timer(0.05, self._publish_opponent)

        # ── 콜백 ─────────────────────────────────────────────────────
        @staticmethod
        def _yaw_of(q):
            return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        def _ego_cb(self, msg):
            p = msg.pose.pose.position
            v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
            with self.lock:
                self.ego_x, self.ego_y = p.x, p.y
                self.ego_yaw = self._yaw_of(msg.pose.pose.orientation)
                self.ego_v = v
                self.ego_stamp = time.monotonic()
                if not self.tracking:
                    return
                idx = self.cl.nearest_idx(p.x, p.y, self.idx_prev)
                s = float(self.cl.s[idx])
                if self.s_prev is not None:
                    raw_ds = s - self.s_prev
                    ds = raw_ds
                    if ds < -self.cl.L / 2:
                        ds += self.cl.L
                    elif ds > self.cl.L / 2:
                        ds -= self.cl.L
                    if abs(ds) < 5.0:  # 점 간격 최대 ~2.3 m 허용
                        self.cum_progress += ds
                    # SF 통과 = s 가 0 을 지나 래핑 (전진 중). 점 간격이 불균일해도
                    # 래핑 자체는 놓치지 않는다 (윈도우 방식은 SF 부근 스킵에 취약).
                    if raw_ds < -self.cl.L / 2 and v > 0.5:
                        t = time.monotonic()
                        if t - self.last_sf_t > 2.0:
                            self.sf_times.append(t)
                            self.sf_progress.append(self.cum_progress)
                            self.last_sf_t = t
                self.s_prev = s
                self.idx_prev = idx
                if self.opp_stamp > 0:
                    d = math.hypot(p.x - self.opp_x, p.y - self.opp_y)
                    self.min_opp_dist = min(self.min_opp_dist, d)

        def _opp_cb(self, msg):
            p = msg.pose.pose.position
            with self.lock:
                self.opp_x, self.opp_y = p.x, p.y
                self.opp_yaw = self._yaw_of(msg.pose.pose.orientation)
                self.opp_vx = msg.twist.twist.linear.x
                self.opp_vy = msg.twist.twist.linear.y
                self.opp_stamp = time.monotonic()

        def _coll_cb(self, msg):
            with self.lock:
                rising = msg.data and not self._coll_prev
                self._coll_prev = msg.data
                if rising and self.collision_armed:
                    self.collision_count += 1

        def _fsm_cb(self, msg):
            with self.lock:
                if self.tracking:
                    self.fsm_states.add(msg.data)
                self.fsm_last = msg.data

        def _publish_opponent(self):
            from nav_msgs.msg import Odometry
            with self.lock:
                now = time.monotonic()
                if now - self.ego_stamp > 0.5 or now - self.opp_stamp > 0.5:
                    return
                dx, dy = self.opp_x - self.ego_x, self.opp_y - self.ego_y
                rng = math.hypot(dx, dy)
                bearing = math.atan2(dy, dx) - self.ego_yaw
                bearing = math.atan2(math.sin(bearing), math.cos(bearing))
                visible = (DETECT_RANGE[0] < rng < DETECT_RANGE[1]
                           and abs(bearing) <= DETECT_FOV_HALF)
                ox, oy, oyaw = self.opp_x, self.opp_y, self.opp_yaw
                ovx, ovy = self.opp_vx, self.opp_vy
            if not visible:
                return
            od = Odometry()
            od.header.stamp = self.get_clock().now().to_msg()
            od.header.frame_id = "map"
            od.child_frame_id = "opponent"
            od.pose.pose.position.x = ox
            od.pose.pose.position.y = oy
            od.pose.pose.orientation.z = math.sin(oyaw * 0.5)
            od.pose.pose.orientation.w = math.cos(oyaw * 0.5)
            od.twist.twist.linear.x = ovx
            od.twist.twist.linear.y = ovy
            self.opp_pub.publish(od)

        # ── 조작 ─────────────────────────────────────────────────────
        def reset_ego(self, x, y, yaw, repeat=3):
            from geometry_msgs.msg import PoseWithCovarianceStamped
            for _ in range(repeat):
                m = PoseWithCovarianceStamped()
                m.header.frame_id = "map"
                m.header.stamp = self.get_clock().now().to_msg()
                m.pose.pose.position.x = x
                m.pose.pose.position.y = y
                m.pose.pose.orientation.z = math.sin(yaw * 0.5)
                m.pose.pose.orientation.w = math.cos(yaw * 0.5)
                self.init_pub.publish(m)
                time.sleep(0.05)

        def place_opponent(self, x, y, yaw, repeat=3):
            from geometry_msgs.msg import PoseStamped
            for _ in range(repeat):
                m = PoseStamped()
                m.header.frame_id = "map"
                m.header.stamp = self.get_clock().now().to_msg()
                m.pose.position.x = x
                m.pose.position.y = y
                m.pose.orientation.z = math.sin(yaw * 0.5)
                m.pose.orientation.w = math.cos(yaw * 0.5)
                self.goal_pub.publish(m)
                time.sleep(0.05)

        def publish_stop(self, repeat=3):
            from ackermann_msgs.msg import AckermannDriveStamped
            for _ in range(repeat):
                m = AckermannDriveStamped()
                m.drive.speed = 0.0
                m.drive.steering_angle = 0.0
                self.drive_pub.publish(m)
                time.sleep(0.05)

        def begin_tracking(self):
            with self.lock:
                self.tracking = True
                self.s_prev = None
                self.idx_prev = None
                self.cum_progress = 0.0
                self.sf_times = []
                self.sf_progress = []
                self.last_sf_t = -1e9
                self.collision_armed = False
                self.collision_count = 0
                self._coll_prev = False
                self.min_opp_dist = float("inf")
                self.fsm_states = set()

        def arm_collisions(self):
            with self.lock:
                self.collision_armed = True

        def end_tracking(self):
            with self.lock:
                self.tracking = False
                self.collision_armed = False

        def snapshot(self):
            with self.lock:
                return {
                    "ego_x": self.ego_x, "ego_y": self.ego_y, "ego_v": self.ego_v,
                    "ego_stamp": self.ego_stamp, "opp_stamp": self.opp_stamp,
                    "s_prev": self.s_prev, "cum_progress": self.cum_progress,
                    "sf_times": list(self.sf_times),
                    "sf_progress": list(self.sf_progress),
                    "collision_count": self.collision_count,
                    "min_opp_dist": self.min_opp_dist,
                    "fsm_states": sorted(self.fsm_states),
                    "fsm_last": self.fsm_last,
                    "opp_x": self.opp_x, "opp_y": self.opp_y,
                }

    return TunerNode()


class TrialRunner:
    def __init__(self, node, stack, centerline, args, results_dir, log):
        self.node = node
        self.stack = stack
        self.cl = centerline
        self.args = args
        self.results_dir = results_dir
        self.log = log
        self.current_proc = None
        self.current_logf = None
        self.solo = getattr(args, "solo", False)
        self.fixed_params = SOLO_FIXED_PARAMS if self.solo else FIXED_PARAMS

    def _wait(self, pred, timeout, poll=0.05):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if pred():
                return True
            time.sleep(poll)
        return False

    def _start_controller(self, params, tag):
        cmd = ["ros2", "run", "smppi_cuda_controller", "smppi_node_fsm",
               "--ros-args", "--params-file", BASE_YAML]
        for k, v in {**self.fixed_params, **params}.items():
            cmd += ["-p", f"{k}:={fmt_param(v)}"]
        logf = open(os.path.join(self.results_dir, f"trial_{tag}_smppi.log"), "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        return proc, logf

    def _stop_controller(self):
        if self.current_proc is not None:
            kill_group(self.current_proc, "smppi_node_fsm", self.log, grace=3.0)
            self.current_proc = None
        if self.current_logf is not None:
            self.current_logf.close()
            self.current_logf = None
        self.node.publish_stop()

    def run(self, params, tag):
        node, cl, args = self.node, self.cl, self.args
        if not isinstance(tag, str):
            tag = f"{tag:03d}"
        # 다중 랩: 정지 출발이 낀 첫 랩의 영향을 희석하기 위해 laps 바퀴를 돌고
        # 평균 랩타임을 보고한다 (SF 통과 laps+1 회 필요).
        laps = max(1, int(getattr(args, "laps", 1)))
        ep_limit = args.timeout * (1.0 + 0.5 * (laps - 1))

        infra = {"status": "infra", "lap_time": None, "lap_times": "",
                 "progress": 0.0, "collisions": 0, "min_opp_dist": None,
                 "fsm_states": ""}

        if not self.stack.all_alive():
            self.stack.restart()
            if not self._wait(lambda: time.monotonic() - node.snapshot()["ego_stamp"] < 0.5, 40):
                return infra

        # 1. 정지 명령 (+ 추월 모드면 상대차를 반 바퀴 앞으로 치움) + ego 리셋
        node.publish_stop()
        sx, sy, syaw = cl.pose_at_s(cl.L - 1.0)
        if not self.solo:
            px, py, pyaw = cl.pose_at_s(cl.L - 1.0 + 0.5 * cl.L)
            node.place_opponent(px, py, pyaw)
        ok = False
        for _ in range(3):
            node.reset_ego(sx, sy, syaw)
            ok = self._wait(
                lambda: (math.hypot(node.snapshot()["ego_x"] - sx,
                                    node.snapshot()["ego_y"] - sy) < 1.0
                         and node.snapshot()["ego_v"] < 0.6), 3.0)
            if ok:
                break
        if not ok:
            self.log("ego 리셋 실패 (infra)")
            return infra

        # 2. 제어기 시작 (CUDA 초기화 대기)
        node.begin_tracking()
        self.current_proc, self.current_logf = self._start_controller(params, tag)
        moved = self._wait(
            lambda: node.snapshot()["ego_v"] > 0.4 or self.current_proc.poll() is not None, 25.0)
        if self.current_proc.poll() is not None or not moved:
            status = "proc_died" if self.current_proc.poll() is not None else "no_start"
            self._stop_controller()
            node.end_tracking()
            return {**infra, "status": status}

        # 3. (추월 모드) 상대차를 ego 현재 호 위치 + gap 앞에 스폰
        if not self.solo:
            snap = node.snapshot()
            s_now, _ = cl.nearest_s(snap["ego_x"], snap["ego_y"])
            ox, oy, oyaw = cl.pose_at_s(s_now + args.opp_gap)
            node.place_opponent(ox, oy, oyaw)
        time.sleep(0.3)
        node.arm_collisions()

        # 4. 모니터링 루프
        t0 = time.monotonic()
        low_v_since = None
        status, lap_time, lap_times = "timeout", None, []
        while True:
            time.sleep(0.05)
            s = node.snapshot()
            now = time.monotonic()
            if s["collision_count"] > 0:
                status = "crashed"
                break
            if len(s["sf_times"]) >= laps + 1:
                status = "finished"
                lap_times = [s["sf_times"][i + 1] - s["sf_times"][i]
                             for i in range(laps)]
                lap_time = sum(lap_times) / laps
                break
            if self.current_proc.poll() is not None:
                status = "proc_died"
                break
            if now - t0 > ep_limit:
                status = "timeout"
                break
            if s["ego_v"] < 0.15:
                low_v_since = low_v_since or now
                if now - low_v_since > 8.0:
                    status = "stuck"
                    break
            else:
                low_v_since = None

        final = node.snapshot()
        self._stop_controller()
        node.end_tracking()

        # 진행률: 첫 SF 통과 이후 누적 진행량 / (트랙 길이 × 랩 수)
        if final["sf_progress"]:
            prog = (final["cum_progress"] - final["sf_progress"][0]) / (cl.L * laps)
        else:
            prog = final["cum_progress"] / (cl.L * laps)
        prog = max(0.0, min(1.0, prog))
        mod = final["min_opp_dist"]
        return {"status": status, "lap_time": lap_time,
                "lap_times": "|".join(f"{lt:.2f}" for lt in lap_times),
                "progress": prog,
                "collisions": final["collision_count"],
                "min_opp_dist": None if math.isinf(mod) else round(mod, 3),
                "fsm_states": "|".join(final["fsm_states"])}


def suggest_params(trial):
    return {
        "num_samples": trial.suggest_int("num_samples", 2000, 16000, step=1000),
        "lambda": trial.suggest_float("lambda", 3.0, 40.0, log=True),
        "noise_steer_std": trial.suggest_float("noise_steer_std", 0.15, 0.8),
        "noise_accel_std": trial.suggest_float("noise_accel_std", 0.8, 4.0),
        "q_du": trial.suggest_float("q_du", 0.05, 1.0, log=True),
        "q_obs_gauss": trial.suggest_float("q_obs_gauss", 80.0, 400.0),
        "fsm_overtake_speed": trial.suggest_float("fsm_overtake_speed", 5.0, 8.0),
        "fsm_follow_speed": trial.suggest_float("fsm_follow_speed", 3.0, 5.5),
    }


def suggest_params_solo(trial, refine=False):
    """단독 주행 랩타임 탐색 공간.

    속도는 목표 추종이 아니라 비용 트레이드오프(q_v 속도 보상 vs 경계 fault)로
    결정되고, max_speed 가 유일한 하드 캡이다. target_speed 는 SOLO 에서
    max_vel(-0.5 m/s² 약한 감속 천장)로만 쓰이므로 max_speed 와 동일하게 묶어
    천장을 무력화한다.

    refine=True: 1차 탐색 챔피언 주변으로 좁힌 정밀 탐색 (신뢰성 확보 목적 —
    1차 상위 후보들이 10회 검증에서 30~70% 충돌해 worst-of-N 에피소드와 함께 사용).
    """
    if refine:
        p = {
            "max_speed": trial.suggest_float("max_speed", 4.0, 5.2),
            "q_v": trial.suggest_float("q_v", 0.8, 4.5, log=True),
            "q_progress": trial.suggest_float("q_progress", 12.0, 40.0, log=True),
            "q_escape_vel": trial.suggest_float("q_escape_vel", 15.0, 75.0, log=True),
            "lambda": trial.suggest_float("lambda", 5.0, 15.0, log=True),
            "noise_steer_std": trial.suggest_float("noise_steer_std", 0.2, 0.4),
            "noise_accel_std": trial.suggest_float("noise_accel_std", 1.0, 2.5),
            "q_du": trial.suggest_float("q_du", 0.05, 0.3, log=True),
            "q_steer": trial.suggest_float("q_steer", 2.0, 8.0, log=True),
            "num_samples": trial.suggest_int("num_samples", 6000, 12000, step=2000),
            "collision_radius": trial.suggest_float("collision_radius", 0.18, 0.26),
            "q_dist": trial.suggest_float("q_dist", 0.0, 60.0),
        }
        p["target_speed"] = p["max_speed"]
        return p
    p = {
        "max_speed": trial.suggest_float("max_speed", 4.0, 10.0),
        "q_v": trial.suggest_float("q_v", 0.5, 8.0, log=True),
        "q_progress": trial.suggest_float("q_progress", 10.0, 100.0, log=True),
        "q_escape_vel": trial.suggest_float("q_escape_vel", 8.0, 80.0, log=True),
        "lambda": trial.suggest_float("lambda", 3.0, 40.0, log=True),
        "noise_steer_std": trial.suggest_float("noise_steer_std", 0.15, 0.9),
        "noise_accel_std": trial.suggest_float("noise_accel_std", 1.0, 5.0),
        "q_du": trial.suggest_float("q_du", 0.05, 1.0, log=True),
        "q_steer": trial.suggest_float("q_steer", 0.5, 8.0, log=True),
        "num_samples": trial.suggest_int("num_samples", 2000, 12000, step=2000),
        # 벽 안전 마진: 플래너 경계가 레이캐스트+팽창으로 이미 실벽-0.15m 라
        # 과도한 값은 협폭 구간 회랑을 봉쇄 → 0.18~0.32 탐색
        "collision_radius": trial.suggest_float("collision_radius", 0.18, 0.32),
        # 센터라인 유지 비용 (YAML 0.0 — 협폭 트랙에선 센터라인이 안전선)
        "q_dist": trial.suggest_float("q_dist", 0.0, 200.0),
    }
    p["target_speed"] = p["max_speed"]
    return p


def objective_value(metrics):
    st, prog = metrics["status"], metrics["progress"]
    if st == "finished" and metrics["collisions"] == 0:
        return metrics["lap_time"]
    if st == "crashed":
        return 1000.0 - 500.0 * prog
    if st in ("timeout", "stuck", "no_start"):
        return 600.0 - 300.0 * prog
    return 1200.0  # proc_died / infra


def main():
    ap = argparse.ArgumentParser(description="SMPPI Optuna 자동 튜너")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=float, default=60.0, help="에피소드 제한시간 [s]")
    ap.add_argument("--opp-gap", type=float, default=6.0, help="상대차 스폰 전방 거리 [m]")
    ap.add_argument("--map", type=str, default="iccas2025",
                    help="시뮬레이터 맵 이름 (기본 iccas2025 — 폭 2.6m 로 추월 가능. "
                         "map1 은 폭 1.1m 라 FSM 추월 판정 불가, 솔로 튜닝은 가능)")
    ap.add_argument("--solo", action="store_true",
                    help="상대차 없이 단독 주행 랩타임 최적화 (use_car1:=false, "
                         "솔로 전용 탐색 공간: max_speed·비용 가중치)")
    ap.add_argument("--episodes-per-trial", type=int, default=None,
                    help="trial 당 에피소드 수, worst 채택 (기본: solo 다중랩이면 1, 아니면 solo 2 / 추월 1)")
    ap.add_argument("--laps", type=int, default=None,
                    help="에피소드당 랩 수 — 평균 랩타임으로 평가해 정지 출발 영향 희석 "
                         "(기본: solo 3 / 추월 1)")
    ap.add_argument("--refine", action="store_true",
                    help="솔로 정밀 탐색: 1차 챔피언 주변으로 좁힌 범위 사용")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", type=str, default=None)
    ap.add_argument("--study-name", type=str, default="smppi_auto_tune")
    args = ap.parse_args()
    if args.laps is None:
        args.laps = 3 if args.solo else 1
    if args.episodes_per_trial is None:
        args.episodes_per_trial = 1 if (args.solo and args.laps >= 2) else (2 if args.solo else 1)

    if "ROS_DISTRO" not in os.environ:
        sys.exit("ROS 환경이 없습니다. 먼저 `source install/setup.bash` 후 실행하세요.")

    # 이미 떠 있는 시뮬레이터와 충돌 방지
    chk = subprocess.run(["pgrep", "-f", "racecar_simulator.*simulator|simulator.launch"],
                         capture_output=True, text=True)
    if chk.stdout.strip():
        sys.exit(f"racecar_simulator 가 이미 실행 중입니다 (pid: {chk.stdout.split()}). "
                 "종료 후 다시 실행하세요.")

    results_dir = args.results_dir or os.path.join(
        WS, "tuning_results", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)

    def log(msg):
        print(f"[tuner {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    centerline_csv = os.path.join(SIM_MAPS_DIR, args.map, f"{args.map}_centerline.csv")
    if not os.path.isfile(centerline_csv):
        sys.exit(f"centerline CSV 없음: {centerline_csv}")
    sim_launch = write_sim_launch(results_dir, args.map, solo=args.solo)

    cl = Centerline(centerline_csv)
    log(f"맵 {args.map}: 트랙 길이 L = {cl.L:.2f} m ({len(cl.xs)}점), "
        f"모드: {'솔로' if args.solo else '추월'}, 결과 디렉토리: {results_dir}")

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    rclpy.init()
    node = make_tuner_node(cl, solo=args.solo)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    stack = StackManager(results_dir, log, sim_launch, centerline_csv, solo=args.solo)
    runner = TrialRunner(node, stack, cl, args, results_dir, log)

    csv_path = os.path.join(results_dir, "trials.csv")
    param_keys = SOLO_PARAM_KEYS if args.solo else OVERTAKE_PARAM_KEYS
    csv_fields = ["trial", "status", "objective", "lap_time", "lap_times",
                  "collisions", "progress", "min_opp_dist", "fsm_states",
                  "lap_time_ep2", "status_ep2"] + param_keys
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{os.path.join(results_dir, 'study.db')}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
    )

    if args.solo and args.refine and len(study.trials) == 0:
        # 정밀 탐색 웜스타트: 1차 탐색 챔피언(#12, #26)과 검증 최고 완주율 변형(8/10)
        study.enqueue_trial({"max_speed": 4.45, "q_v": 3.52, "q_progress": 18.0,
                             "q_escape_vel": 24.0, "lambda": 9.6,
                             "noise_steer_std": 0.29, "noise_accel_std": 1.11,
                             "q_du": 0.05, "q_steer": 5.22, "num_samples": 6000,
                             "collision_radius": 0.185, "q_dist": 1.0})
        study.enqueue_trial({"max_speed": 5.15, "q_v": 2.11, "q_progress": 32.0,
                             "q_escape_vel": 71.0, "lambda": 7.1,
                             "noise_steer_std": 0.25, "noise_accel_std": 1.05,
                             "q_du": 0.09, "q_steer": 3.75, "num_samples": 10000,
                             "collision_radius": 0.208, "q_dist": 20.0})
        study.enqueue_trial({"max_speed": 4.5, "q_v": 2.11, "q_progress": 32.0,
                             "q_escape_vel": 71.0, "lambda": 7.1,
                             "noise_steer_std": 0.25, "noise_accel_std": 1.05,
                             "q_du": 0.09, "q_steer": 3.75, "num_samples": 10000,
                             "collision_radius": 0.25, "q_dist": 60.0})
    elif args.solo and len(study.trials) == 0:
        # 웜스타트: 보수적 YAML 근사 + 벽 마진/센터라인 유지 확보, max_speed·q_v 변주
        seed_base = {"num_samples": 8000, "lambda": 15.0, "noise_steer_std": 0.4,
                     "noise_accel_std": 2.0, "q_du": 0.2,
                     "q_v": 1.0, "q_steer": 3.0, "q_progress": 36.0, "q_escape_vel": 32.0,
                     "collision_radius": 0.28, "q_dist": 20.0}
        study.enqueue_trial({**seed_base, "max_speed": 5.0})
        study.enqueue_trial({**seed_base, "max_speed": 7.0, "q_v": 3.0})
        study.enqueue_trial({**seed_base, "max_speed": 9.0, "q_v": 3.0,
                             "noise_accel_std": 3.5,
                             "collision_radius": 0.22, "q_dist": 5.0})

    infra_fails = {"consecutive": 0}

    def objective(trial):
        params = (suggest_params_solo(trial, refine=args.refine)
                  if args.solo else suggest_params(trial))
        log(f"trial {trial.number}: {', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in params.items())}")

        # 단일 에피소드 낙관 편향 방지: N 에피소드 실행 후 worst 채택.
        # 실패 에피소드가 나오면 그 값이 worst 로 확정이므로 즉시 중단.
        ep_metrics = []
        for ep in range(args.episodes_per_trial):
            tag = (f"{trial.number:03d}" if args.episodes_per_trial == 1
                   else f"{trial.number:03d}_ep{ep}")
            m = runner.run(params, tag)
            if m["status"] == "infra":
                infra_fails["consecutive"] += 1
                if infra_fails["consecutive"] >= 3:
                    raise RuntimeError("인프라 실패 3연속 — 스택 로그를 확인하세요: " + results_dir)
                raise optuna.TrialPruned()
            infra_fails["consecutive"] = 0
            ep_metrics.append(m)
            if not (m["status"] == "finished" and m["collisions"] == 0):
                break

        values = [objective_value(m) for m in ep_metrics]
        worst = max(range(len(values)), key=lambda i: values[i])
        metrics, value = ep_metrics[worst], values[worst]

        if args.solo and metrics.get("fsm_states", "") not in ("", "SOLO"):
            log(f"경고: 솔로 모드인데 FSM 상태 {metrics['fsm_states']} — 상대차 잔재 의심")

        for k, v in metrics.items():
            trial.set_user_attr(k, v if v is None or isinstance(v, (int, float, str)) else str(v))
        trial.set_user_attr("episodes", len(ep_metrics))
        eps_all = "/".join(f"{m['lap_time']:.2f}" if m["lap_time"] else m["status"]
                           for m in ep_metrics)
        lap_detail = metrics.get("lap_times", "")
        overtook = "OVERTAKE" in metrics.get("fsm_states", "")
        log(f"trial {trial.number} 결과: {metrics['status']}, 평균lap=[{eps_all}]"
            + (f" (랩별 {lap_detail})" if lap_detail else "")
            + f", 충돌={metrics['collisions']}, 진행률={metrics['progress']:.2f}, "
            + ("" if args.solo else f"추월={'O' if overtook else 'X'}, ")
            + f"목적값={value:.2f}")
        ep2 = ep_metrics[1] if len(ep_metrics) > 1 else None
        with open(csv_path, "a", newline="") as f:
            row = {"trial": trial.number, "status": metrics["status"],
                   "objective": round(value, 3),
                   "lap_time": metrics["lap_time"],
                   "lap_times": lap_detail,
                   "collisions": metrics["collisions"],
                   "progress": round(metrics["progress"], 3),
                   "min_opp_dist": metrics["min_opp_dist"],
                   "fsm_states": metrics.get("fsm_states", ""),
                   "lap_time_ep2": ep2["lap_time"] if ep2 else None,
                   "status_ep2": ep2["status"] if ep2 else None,
                   **params}
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
        return value

    def cleanup(*_):
        runner._stop_controller()
        stack.stop()

    signal.signal(signal.SIGTERM, lambda *a: (cleanup(), sys.exit(1)))

    try:
        stack.start()
        log("스택 기동 대기 중 (ego odom 수신%s)..." % ("" if args.solo else " + 상대차 odom"))
        t0 = time.monotonic()
        ready = False
        while time.monotonic() - t0 < 45:
            s = node.snapshot()
            now = time.monotonic()
            if now - s["ego_stamp"] < 0.5 and (args.solo or now - s["opp_stamp"] < 0.5):
                ready = True
                break
            time.sleep(0.5)
        if not ready:
            raise RuntimeError("시뮬레이터 odom 미수신 — stack_sim.log 확인 필요")
        if not args.solo and node.count_publishers("/f1/perception/object/obstacles/arr") == 0:
            log("경고: perception 장애물 토픽 발행자 없음 — stack_perception.log 확인")
        log(f"스택 준비 완료. {args.trials} trial 시작 (timeout {args.timeout}s, "
            + (f"랩/에피소드 {args.laps}, 에피소드/trial {args.episodes_per_trial}"
               if args.solo else f"gap {args.opp_gap}m")
            + ")")

        study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

        done = [t for t in study.trials if t.value is not None]
        finished = [t for t in done if t.user_attrs.get("status") == "finished"]
        log(f"완료: {len(done)} trials (완주 {len(finished)}회)")
        if finished:
            best = study.best_trial
            log(f"최적 trial #{best.number}: 목적값 {best.value:.2f}s")
            for k, v in best.params.items():
                log(f"  {k} = {v}")
            best_params = dict(best.params)
            if args.solo:
                best_params["target_speed"] = best_params["max_speed"]
            fixed = SOLO_FIXED_PARAMS if args.solo else FIXED_PARAMS
            best_yaml = os.path.join(
                results_dir, "best_params_solo.yaml" if args.solo else "best_params.yaml")
            import yaml as _yaml
            with open(BASE_YAML) as f:
                base = _yaml.safe_load(f)
            base["/**"]["ros__parameters"].update({**fixed, **best_params})
            with open(best_yaml, "w") as f:
                if args.solo:
                    f.write("# SIM-ONLY 솔로 튜닝 파라미터 — max_speed 가 실차 안전 상한(3.0)을 초과.\n"
                            "# 실차 배포 및 config/params.yaml 병합 금지.\n")
                _yaml.safe_dump(base, f, sort_keys=False, allow_unicode=True)
            log(f"최적 파라미터 저장: {best_yaml}")
        else:
            log("완주한 trial 이 없습니다 — 탐색 범위/timeout 조정 필요")
        log(f"trial 별 기록: {csv_path}")
    except KeyboardInterrupt:
        log("사용자 중단")
    finally:
        cleanup()
        # 종료 순서: executor 정지 → 노드 파괴 → rclpy 종료 (역순이면 SIGABRT)
        try:
            executor.shutdown(timeout_sec=2.0)
            spin_thread.join(timeout=3.0)
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
