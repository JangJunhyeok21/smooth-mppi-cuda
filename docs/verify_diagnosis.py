#!/usr/bin/env python3
"""smppi-diagnosis-2026-08.md 의 모든 수치를 재현한다.

config/params.yaml 값을 그대로 하드코딩했다 (numpy 외 의존성 없음).
파라미터를 바꾼 뒤 이 스크립트를 다시 돌려 진단이 아직 유효한지 확인할 것.

    python3 docs/verify_diagnosis.py
"""
import numpy as np

# ── config/params.yaml (2026-08-03 기준) ────────────────────────────────
DT, T = 0.035, 50                    # smppi_node.cpp:25, :437
MASS, LF, LR, IZ = 3.74, 0.163, 0.161, 0.04712
CM0 = 0.04
BF, CF, DF, EF = 6.0926, 1.2447, 0.7955, 0.7815
BR, CR, DR, ER = 6.6457, 2.2129, 0.7317, 0.0597
MAX_STEER, MAX_SPEED = 0.4788, 4.0
NOISE_STEER_STD, MAX_STEER_RATE = 0.4, 8.0
Q_PROGRESS, Q_ESCAPE_VEL, Q_COLLISION = 36.0, 32.0, 300.0
Q_LAT_G, LAT_G_TH, LAT_G_FAULT_TH = 500.0, 5.5, 7.0
Q_IMPACT = 300.0
COLLISION_RADIUS = 0.3
SPACING = 0.099                      # map1_centerline.csv 평균 웨이포인트 간격
MAX_PROGRESS_IDX = 2 * T             # mppi_core.cu 의 max_possible_progress
FILTER_WARMUP = 8                    # mppi_core.cu 롤아웃 시작 전 IIR 워밍업 스텝 수

FZF = MASS * 9.81 * LR / (LF + LR)
FZR = MASS * 9.81 * LF / (LF + LR)

FAIL = []


def check(name, ok, detail):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAIL.append(name)


def magic_formula(a, B, C, D, E, Fz):
    """mppi_core.cu:88-91 과 동일한 4-파라미터 매직 포뮬러."""
    ba = B * a
    return Fz * D * np.sin(C * np.arctan(ba - E * (ba - np.arctan(ba))))


def step(s, steer, accel, dt=DT):
    """mppi_core.cu:41-115 update_dynamics() 포팅."""
    x, y, yaw, v, om, beta = s
    if abs(v) < 0.5:                                   # 운동학 분기
        L = LF + LR
        return (x + v * np.cos(yaw) * dt, y + v * np.sin(yaw) * dt,
                yaw + v * np.tan(steer) / L * dt, v + accel * dt,
                v * np.tan(steer) / L, 0.0)
    vx, vy = v * np.cos(beta), v * np.sin(beta)
    af = steer - np.arctan2(vy + LF * om, vx)
    ar = -np.arctan2(vy - LR * om, vx)
    Ff = magic_formula(af, BF, CF, DF, EF, FZF)
    Fr = magic_formula(ar, BR, CR, DR, ER, FZR)
    dom = (LF * Ff * np.cos(steer) - LR * Fr) / IZ
    dbeta = (Ff + Fr) / (MASS * v) - om
    return (x + v * np.cos(yaw + beta) * dt, y + v * np.sin(yaw + beta) * dt,
            yaw + om * dt, v + accel * (1 - CM0 * v) * dt,
            om + dom * dt, beta + dbeta * dt)


# ══════════════════════════════════════════════════════════════════════
print("\n== A-3. 그립 페널티 포화 ==")
a = np.linspace(0, 0.6, 20001)
ay_max = (magic_formula(a, BF, CF, DF, EF, FZF).max()
          + magic_formula(a, BR, CR, DR, ER, FZR).max()) / MASS
check("모델 최대 횡가속", abs(ay_max - 7.38) < 0.1, f"{ay_max:.2f} m/s^2")
check("lat_g 하드 페일 도달 가능", ay_max >= LAT_G_FAULT_TH,
      f"threshold={LAT_G_FAULT_TH} > ay_max={ay_max:.2f} -> dead code")
print(f"        soft 페널티 포화값 = {Q_LAT_G * (ay_max - LAT_G_TH)**2:.0f} "
      f"(이 위로는 속도가 공짜)")

# ══════════════════════════════════════════════════════════════════════
print("\n== A-4. progress 보상 포화 ==")
v_sat = MAX_PROGRESS_IDX * SPACING / (T * DT)
check("progress 보상이 max_speed 까지 살아있음", v_sat >= MAX_SPEED,
      f"{v_sat:.2f} m/s 에서 포화, max_speed={MAX_SPEED}")

# ══════════════════════════════════════════════════════════════════════
print("\n== A-2. 코너컷 이득 vs 벽 페널티 ==")


def boundary_cost(d, r=COLLISION_RADIUS):
    """mppi_core.cu:166-181."""
    safe = r + 0.35
    if d >= safe:
        return 0.0
    c = 70.0 * (safe - d) ** 2
    if d < r * 1.5:
        c += Q_COLLISION * np.log(1 + np.exp(-30 * max(d - r, 1e-5)))
    return c


gain = Q_PROGRESS * 2 + Q_ESCAPE_VEL * (4.5**2 - 4.0**2)
wall = 6 * boundary_cost(0.35)
check("벽 페널티가 코너컷 이득보다 한 자릿수 위", wall > 10 * gain,
      f"이득 {gain:.0f} vs 6스텝 벽비용 {wall:.0f} (비 {wall/gain:.1f}x)")
for d in (0.65, 0.55, 0.45, 0.40, 0.35, 0.30):
    print(f"        d={d:.2f} m -> step당 {boundary_cost(d):7.1f}")

# ══════════════════════════════════════════════════════════════════════
print("\n== C-1. 샘플러 조향 탐색 폭 ==")
fs, fc = 1 / DT, 3.0
w0 = np.tan(np.pi * fc / fs)
Kb = np.sqrt(2) * w0
Db = 1 + Kb + w0 * w0
b0 = w0 * w0 / Db
b1, b2 = 2 * b0, b0
a1 = 2 * (w0 * w0 - 1) / Db
a2 = (1 - Kb + w0 * w0) / Db

N = 20000
rng = np.random.default_rng(0)
raw = rng.normal(0, NOISE_STEER_STD, (N, FILTER_WARMUP + T))
x1 = x2 = y1 = y2 = np.zeros(N)
acc = np.zeros(N)
steer_std = []
for t in range(FILTER_WARMUP + T):
    f = b0 * raw[:, t] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
    x2, x1 = x1, raw[:, t]
    y2, y1 = y1, f
    if t < FILTER_WARMUP:          # 워밍업 구간은 조향에 누적하지 않는다
        continue
    acc = acc + np.clip(f * DT, -MAX_STEER_RATE * DT, MAX_STEER_RATE * DT)
    steer_std.append(acc.std())
print(f"        IIR 워밍업 {FILTER_WARMUP} 스텝 적용 상태로 측정")
check("t=0 탐색 폭이 max_steer 의 5% 이상", steer_std[0] > 0.05 * MAX_STEER,
      f"t=0 {steer_std[0]:.4f} rad ({np.degrees(steer_std[0]):.2f} deg), "
      f"max_steer={MAX_STEER}")
check("horizon 끝 탐색 폭이 max_steer 의 절반 이상",
      steer_std[-1] > 0.5 * MAX_STEER,
      f"t={T-1} {steer_std[-1]:.4f} rad ({np.degrees(steer_std[-1]):.2f} deg)")

# ══════════════════════════════════════════════════════════════════════
print("\n== C-3. 오일러 적분 안정성 ==")
Cf_lin = FZF * DF * CF * BF      # 코너링 강성 (원점 기울기)
Cr_lin = FZR * DR * CR * BR
print(f"        코너링 강성 Cf={Cf_lin:.1f} Cr={Cr_lin:.1f} N/rad")
worst = 0.0
for v in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
    A = np.array([[-(Cf_lin + Cr_lin) / (MASS * v),
                   -1 + (LR * Cr_lin - LF * Cf_lin) / (MASS * v * v)],
                  [(LR * Cr_lin - LF * Cf_lin) / IZ,
                   -(LF * LF * Cf_lin + LR * LR * Cr_lin) / (IZ * v)]])
    zmax = max(abs(np.linalg.eigvals(np.eye(2) + DT * A)))
    worst = max(worst, zmax) if v >= 1.0 else worst
    print(f"        v={v:4.1f} m/s -> |z|max={zmax:6.3f} "
          f"{'UNSTABLE' if zmax > 1 else ''}")
check("전 속도 구간에서 오일러 안정", worst <= 1.0,
      f"최악 |z|={worst:.2f} (1.0 이하여야 함), 안정 한계 dt<{2/57*1000:.0f}ms")

print("\n        스텝 조향(0.38 rad) 응답 — omega [rad/s]")
for v0 in (1.5, 2.0, 4.0):
    se = (0., 0., 0., v0, 0., 0.)
    sr = (0., 0., 0., v0, 0., 0.)
    oe, orf = [], []
    for _ in range(8):
        se = step(se, 0.38, 0.0)
        for _ in range(16):
            sr = step(sr, 0.38, 0.0, DT / 16)
        oe.append(se[4])
        orf.append(sr[4])
    print(f"        v={v0}  Euler : " + " ".join(f"{x:6.2f}" for x in oe))
    print(f"        {'':6} 참값  : " + " ".join(f"{x:6.2f}" for x in orf))

# ══════════════════════════════════════════════════════════════════════
print("\n== C-4. 후륜 타이어 피크 후 붕괴 ==")
peak_a = a[np.argmax(magic_formula(a, BR, CR, DR, ER, FZR))]
peak_f = magic_formula(peak_a, BR, CR, DR, ER, FZR)
drop = magic_formula(0.5, BR, CR, DR, ER, FZR) / peak_f
check("피크 후 감소가 실제 타이어 수준(20% 이내)", drop > 0.8,
      f"alpha=0.5 에서 피크의 {drop*100:.0f}% (C_r={CR}, 통상 1.3~1.8)")

# ══════════════════════════════════════════════════════════════════════
print("\n== A-1(c). fault 순위 ==")
print(f"        cost = 10000 - 50t + {Q_IMPACT}*v^2   (progress 항 제거됨)")
costs = {}
for v in (4.0, 2.0, 0.6):
    t_imp = int(min(T - 1, 1.0 / (v * DT)))       # 1 m 앞 벽
    costs[v] = 10000 - 50 * t_imp + Q_IMPACT * v * v
    print(f"        v={v} m/s -> t_impact={t_imp:2d}, cost={costs[v]:8.0f}")
check("느리게 박는 쪽이 확실히 유리", costs[0.6] < costs[2.0] < costs[4.0],
      f"0.6 m/s({costs[0.6]:.0f}) < 2.0({costs[2.0]:.0f}) < 4.0({costs[4.0]:.0f})")

# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
if FAIL:
    print(f"미해결 항목 {len(FAIL)}건:")
    for f in FAIL:
        print(f"  - {f}")
    print("\n자세한 내용은 docs/smppi-diagnosis-2026-08.md 참고")
else:
    print("모든 항목 통과 — 진단 문서의 지적사항이 해소되었다.")
