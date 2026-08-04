#!/usr/bin/env python3
"""rosbag2에서 조향/속도 명령 대비 실제 응답의 1차 지연 시정수(tau)를 추출한다.

    y(t) = y_inf + (y0 - y_inf) * exp(-t / tau)

조향 채널: 명령 steering_angle 스텝 → 실제 각속도(omega, odom 또는 IMU) 응답
속도 채널: 명령 speed 스텝        → 실제 종방향 속도(odom twist.linear.x) 응답

사용 예:
    python3 identify_actuator_tau.py /home/a/bags/rosbag2_2026_06_22-15_24_36 \
        --cmd-topic /ackermann_cmd --odom-topic /odom --out-dir .

여러 bag의 스텝 구간을 모아 평균 tau를 낼 수도 있다 — bag 경로를 여러 개 나열하거나,
bag들이 들어있는 상위 디렉토리 하나만 지정하면 그 아래 bag들을 전부 찾아 처리한다:
    python3 identify_actuator_tau.py /home/a/bags/0730 --out-dir .
    python3 identify_actuator_tau.py bagA bagB bagC --out-dir .

자동 스텝 검출이 실패하는 구간은 --manual-steer-window / --manual-motor-window로
(start end) 시간 구간을 직접 지정해 보강할 수 있다 (여러 번 지정 가능, 단일 bag에서만).
"""
import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

CMD_TOPIC_CANDIDATES = ["/drive", "/ackermann_cmd", "/ackermann_cmd0"]


# ════════════════════════════════════════════════════════════════════
#  bag 읽기
# ════════════════════════════════════════════════════════════════════
def open_reader(bag_path, topics):
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in type_map]
    if missing:
        raise RuntimeError(f"토픽을 bag에서 찾을 수 없음: {missing} (가용 토픽: {sorted(type_map)})")
    reader.set_filter(StorageFilter(topics=topics))
    return reader, type_map


def read_series(bag_path, topics):
    """{topic: (t_array, dict_of_field_arrays)} 형태로 필요한 필드만 뽑아온다."""
    reader, type_map = open_reader(bag_path, topics)
    msg_classes = {t: get_message(type_map[t]) for t in topics}
    raw = {t: [] for t in topics}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in raw:
            continue
        msg = deserialize_message(data, msg_classes[topic])
        raw[topic].append((t_ns * 1e-9, msg))
    return raw


def pick_cmd_topic(bag_path, override):
    if override:
        return override
    _, type_map = open_reader(bag_path, [])
    for cand in CMD_TOPIC_CANDIDATES:
        if cand in type_map:
            return cand
    raise RuntimeError(
        f"명령 토픽을 자동으로 찾지 못했습니다. --cmd-topic으로 직접 지정하세요. "
        f"(후보: {CMD_TOPIC_CANDIDATES}, 가용 토픽: {sorted(type_map)})")


def discover_bags(paths):
    """입력 경로들을 실제 rosbag2 디렉토리 목록으로 확장한다.

    각 경로가 그 자체로 bag(metadata.yaml 보유)이면 그대로 쓰고, 아니라면 바로 아래
    하위 디렉토리들 중 bag인 것들을 전부 찾아 추가한다 (한 단계만 탐색).
    """
    bags = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "metadata.yaml")):
            bags.append(os.path.normpath(p))
        elif os.path.isdir(p):
            found = sorted(
                entry.path for entry in os.scandir(p)
                if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "metadata.yaml")))
            if not found:
                raise RuntimeError(f"'{p}' 는 bag도 아니고 bag을 담은 디렉토리도 아닙니다.")
            bags.extend(os.path.normpath(b) for b in found)
        else:
            raise RuntimeError(f"경로를 찾을 수 없습니다: {p}")
    seen = set()
    unique_bags = []
    for b in bags:
        if b not in seen:
            seen.add(b)
            unique_bags.append(b)
    return unique_bags


# ════════════════════════════════════════════════════════════════════
#  스텝 구간 자동 검출
# ════════════════════════════════════════════════════════════════════
@dataclass
class Step:
    t0: float
    t1: float
    channel: str  # "steer" | "motor"


def detect_steps(t, u, thresh, min_window, max_window, t_end):
    """명령 신호 u(t)의 미분이 thresh를 넘는 지점을 스텝 시작으로 탐지."""
    steps = []
    if len(t) < 2:
        return steps
    du = np.diff(u)
    dt = np.diff(t)
    du_dt = np.divide(du, dt, out=np.zeros_like(du), where=dt > 1e-6)
    for i in range(len(du_dt)):
        if abs(du_dt[i]) * min(dt[i], 0.2) < thresh:
            continue
        t0 = t[i + 1]
        # 다음 스텝(같은 채널) 전까지, 혹은 max_window 중 짧은 쪽을 응답 구간으로 사용
        t1 = min(t0 + max_window, t_end)
        if steps and t0 - steps[-1].t0 < min_window:
            continue  # 직전 스텝과 너무 붙어있으면 스킵 (응답이 안 끝남)
        steps.append(Step(t0=t0, t1=t1, channel=""))
    return steps


# ════════════════════════════════════════════════════════════════════
#  1차 지연 피팅
# ════════════════════════════════════════════════════════════════════
def first_order_model(t, y_inf, y0, tau):
    tau = max(tau, 1e-3)
    return y_inf + (y0 - y_inf) * np.exp(-t / tau)


def fit_step(t_resp, y_resp, tau_max):
    """window (t_resp[0]..) 내 응답을 1차 지연으로 피팅. 실패 시 None.

    curve_fit이 tau 상한에 붙어버리는(=수렴 실패, 사실상 선형에 가까운 구간을
    거대한 tau로 눈속임 피팅) 경우는 R^2가 높게 나올 수 있어 R^2만으로는
    걸러지지 않는다. tau가 상한의 98% 이상이면 별도로 발산 처리한다.
    """
    if len(t_resp) < 5:
        return None
    t_rel = t_resp - t_resp[0]
    y0_guess = y_resp[0]
    y_inf_guess = np.median(y_resp[-max(3, len(y_resp) // 5):])
    if abs(y_inf_guess - y0_guess) < 1e-3:
        return None  # 사실상 스텝이 아님 (노이즈로 오검출)
    tau_guess = max(t_rel[-1] / 3.0, 0.02)

    try:
        popt, _ = curve_fit(
            first_order_model, t_rel, y_resp,
            p0=[y_inf_guess, y0_guess, tau_guess],
            bounds=([-1e3, -1e3, 1e-3], [1e3, 1e3, tau_max]),
            maxfev=5000)
    except Exception:
        return None

    y_fit = first_order_model(t_rel, *popt)
    ss_res = np.sum((y_resp - y_fit) ** 2)
    ss_tot = np.sum((y_resp - np.mean(y_resp)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    y_inf, y0, tau = popt
    diverged = tau >= 0.98 * tau_max
    return {"tau": float(tau), "y_inf": float(y_inf), "y0": float(y0),
            "r2": float(r2), "t_rel": t_rel, "y": y_resp, "y_fit": y_fit,
            "t0": t_resp[0], "diverged": diverged}


# ════════════════════════════════════════════════════════════════════
#  메인 파이프라인
# ════════════════════════════════════════════════════════════════════
def extract_windows_manual(t, y, windows):
    out = []
    for (start, end) in windows:
        mask = (t >= start) & (t <= end)
        if mask.sum() >= 5:
            out.append((t[mask], y[mask]))
    return out


def run_channel(name, t_cmd, u_cmd, t_resp, y_resp, thresh, min_window, max_window,
                 manual_windows, r2_min, tau_max):
    results = []
    t_end = t_resp[-1] if len(t_resp) else 0.0

    windows = []
    if manual_windows:
        windows = [(w[0], w[1]) for w in manual_windows]
    else:
        steps = detect_steps(t_cmd, u_cmd, thresh, min_window, max_window, t_end)
        windows = [(s.t0, s.t1) for s in steps]

    print(f"[{name}] 검출된 스텝 후보: {len(windows)}개")

    for (t0, t1) in windows:
        mask = (t_resp >= t0) & (t_resp <= t1)
        if mask.sum() < 5:
            continue
        fit = fit_step(t_resp[mask], y_resp[mask], tau_max)
        if fit is None:
            continue
        fit["window"] = (t0, t1)
        results.append(fit)

    accepted = [r for r in results if r["r2"] >= r2_min and not r["diverged"]]
    rejected = [r for r in results if r["r2"] < r2_min or r["diverged"]]

    for r in rejected:
        reason = []
        if r["diverged"]:
            reason.append(f"tau가 상한({tau_max}s)에 근접 — 피팅 발산")
        if r["r2"] < r2_min:
            reason.append(f"R^2={r['r2']:.3f} < {r2_min}")
        print(f"  [WARN] {name} 스텝 @ t={r['t0']:.2f}s : {', '.join(reason)} "
              f"— 평균 계산에서 제외")
    for r in accepted:
        print(f"  {name} 스텝 @ t={r['t0']:.2f}s : tau={r['tau']:.4f}s, R^2={r['r2']:.3f}")

    return accepted, rejected


def make_plot(steer_fits, motor_fits, out_path):
    all_fits = [("steer", f) for f in steer_fits] + [("motor", f) for f in motor_fits]
    n = max(len(all_fits), 1)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, (label, f) in enumerate(all_fits):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot(f["t_rel"], f["y"], ".", ms=3, label="measured", color="tab:blue")
        ax.plot(f["t_rel"], f["y_fit"], "-", lw=2, label="fit", color="tab:red")
        bag_tag = f" [{f['bag']}]" if f.get("bag") else ""
        ax.set_title(f"{label}{bag_tag} @ t0={f['t0']:.1f}s\ntau={f['tau']:.3f}s R2={f['r2']:.2f}",
                     fontsize=9)
        ax.set_xlabel("t [s]")
        ax.legend(fontsize=7)

    for idx in range(len(all_fits), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"플롯 저장: {out_path}")


def process_bag(bag_path, args, manual_steer, manual_motor):
    """단일 bag을 읽어 (steer_fits, motor_fits)를 반환. 처리 불가하면 None."""
    bag_name = os.path.basename(os.path.normpath(bag_path))
    try:
        cmd_topic = pick_cmd_topic(bag_path, args.cmd_topic)
        raw = read_series(bag_path, [cmd_topic, args.odom_topic])
    except RuntimeError as e:
        print(f"[WARN] '{bag_name}' 스킵: {e}")
        return None

    print(f"  명령 토픽: {cmd_topic} / 응답 토픽: {args.odom_topic}")

    if not raw[cmd_topic]:
        print(f"[WARN] '{bag_name}' 스킵: {cmd_topic}에 메시지가 없습니다.")
        return None
    if not raw[args.odom_topic]:
        print(f"[WARN] '{bag_name}' 스킵: {args.odom_topic}에 메시지가 없습니다.")
        return None

    t_cmd = np.array([t for t, _ in raw[cmd_topic]])
    steer_cmd = np.array([m.drive.steering_angle for _, m in raw[cmd_topic]])
    speed_cmd = np.array([m.drive.speed for _, m in raw[cmd_topic]])

    t_odom = np.array([t for t, _ in raw[args.odom_topic]])
    omega_resp = np.array([m.twist.twist.angular.z for _, m in raw[args.odom_topic]])
    v_resp = np.array([m.twist.twist.linear.x for _, m in raw[args.odom_topic]])

    # bag 타임스탬프를 0 기준 상대시간으로 정렬
    t0_global = min(t_cmd[0], t_odom[0])
    t_cmd = t_cmd - t0_global
    t_odom = t_odom - t0_global

    steer_fits, _ = run_channel(
        "steer", t_cmd, steer_cmd, t_odom, omega_resp,
        args.steer_step_thresh, args.min_window, args.max_window, manual_steer,
        args.r2_min, args.tau_max)
    motor_fits, _ = run_channel(
        "motor", t_cmd, speed_cmd, t_odom, v_resp,
        args.speed_step_thresh, args.min_window, args.max_window, manual_motor,
        args.r2_min, args.tau_max)

    for f in steer_fits + motor_fits:
        f["bag"] = bag_name

    return steer_fits, motor_fits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_paths", nargs="+",
                     help="rosbag2 디렉토리 경로 (여러 개 나열 가능, 또는 bag들을 담은 상위 디렉토리 하나)")
    ap.add_argument("--cmd-topic", default=None,
                     help=f"명령 토픽 (기본: 자동탐색 {CMD_TOPIC_CANDIDATES})")
    ap.add_argument("--odom-topic", default="/odom", help="응답(omega, v) 소스 토픽")
    ap.add_argument("--steer-step-thresh", type=float, default=0.05,
                     help="조향 스텝 검출 임계값 [rad] (기본 0.05)")
    ap.add_argument("--speed-step-thresh", type=float, default=0.5,
                     help="속도 스텝 검출 임계값 [m/s] (기본 0.5)")
    ap.add_argument("--min-window", type=float, default=0.15, help="응답 구간 최소 길이 [s]")
    ap.add_argument("--max-window", type=float, default=1.5, help="응답 구간 최대 길이 [s]")
    ap.add_argument("--r2-min", type=float, default=0.7,
                     help="이 값 미만인 구간은 평균 계산에서 제외 (기본 0.7)")
    ap.add_argument("--tau-max", type=float, default=2.0,
                     help="tau 탐색 상한 [s] — 여기에 붙는 피팅은 발산으로 간주해 제외 (기본 2.0)")
    ap.add_argument("--start", type=float, default=None,
                     help="자동 검출 실패 시 수동 지정할 단일 조향 구간의 시작 시각 [s]")
    ap.add_argument("--end", type=float, default=None,
                     help="자동 검출 실패 시 수동 지정할 단일 조향 구간의 종료 시각 [s]")
    ap.add_argument("--manual-steer-window", nargs=2, type=float, action="append",
                     metavar=("START", "END"),
                     help="조향 응답 구간 수동 지정 (여러 번 지정 가능)")
    ap.add_argument("--manual-motor-window", nargs=2, type=float, action="append",
                     metavar=("START", "END"),
                     help="속도 응답 구간 수동 지정 (여러 번 지정 가능)")
    ap.add_argument("--out-dir", default=".", help="tau_params.yaml / tau_fit_result.png 출력 위치")
    args = ap.parse_args()

    manual_steer = list(args.manual_steer_window or [])
    manual_motor = list(args.manual_motor_window or [])
    if args.start is not None and args.end is not None:
        manual_steer.append((args.start, args.end))

    bag_paths = discover_bags(args.bag_paths)
    if len(bag_paths) > 1 and (manual_steer or manual_motor):
        print("[ERROR] --manual-steer-window/--manual-motor-window/--start/--end는 "
              "bag이 1개일 때만 지정할 수 있습니다 (bag마다 상대시간 의미가 달라집니다).",
              file=sys.stderr)
        sys.exit(1)
    print(f"처리할 bag {len(bag_paths)}개: {[os.path.basename(b) for b in bag_paths]}")

    steer_fits, motor_fits = [], []
    per_bag = []
    skipped = 0
    for bag_path in bag_paths:
        print(f"\n=== bag: {os.path.basename(bag_path)} ===")
        result = process_bag(bag_path, args, manual_steer, manual_motor)
        if result is None:
            skipped += 1
            continue
        bag_steer, bag_motor = result
        steer_fits.extend(bag_steer)
        motor_fits.extend(bag_motor)
        per_bag.append({"bag": os.path.basename(bag_path),
                         "n_steer_fits": len(bag_steer), "n_motor_fits": len(bag_motor)})

    print(f"\n처리 완료: {len(bag_paths) - skipped}개 사용, {skipped}개 스킵")

    def summarize(name, fits, min_count=5):
        if len(fits) == 0:
            print(f"[WARN] {name}: 유효한 스텝 구간이 0개 — tau 추정 불가")
            return None, None, 0
        if len(fits) < min_count:
            print(f"[WARN] {name}: 유효한 스텝 구간이 {len(fits)}개뿐 (권장 최소 {min_count}개)")
        taus = np.array([f["tau"] for f in fits])
        return float(np.mean(taus)), float(np.std(taus)), len(fits)

    tau_steer_mean, tau_steer_std, n_steer = summarize("steer", steer_fits)
    tau_motor_mean, tau_motor_std, n_motor = summarize("motor", motor_fits)

    print("\n=== 결과 요약 ===")
    if tau_steer_mean is not None:
        print(f"tau_steer = {tau_steer_mean:.4f} ± {tau_steer_std:.4f} s  (n={n_steer})")
    if tau_motor_mean is not None:
        print(f"tau_motor = {tau_motor_mean:.4f} ± {tau_motor_std:.4f} s  (n={n_motor})")

    out = {
        "tau_steer": tau_steer_mean,
        "tau_motor": tau_motor_mean,
        "tau_steer_std": tau_steer_std,
        "tau_motor_std": tau_motor_std,
        "n_steer_fits": n_steer,
        "n_motor_fits": n_motor,
        "source_bags": bag_paths,
        "per_bag": per_bag,
    }
    out_yaml = f"{args.out_dir}/tau_params.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"\n결과 저장: {out_yaml}")

    out_png = f"{args.out_dir}/tau_fit_result.png"
    make_plot(steer_fits, motor_fits, out_png)


if __name__ == "__main__":
    main()
