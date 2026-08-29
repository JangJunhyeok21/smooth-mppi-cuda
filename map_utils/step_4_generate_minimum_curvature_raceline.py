#!/usr/bin/env python3
"""alpha-RACER/TUM minimum-curvature optimizer로 MPPI raceline CSV를 만든다."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAP_NAME = "ifac2026"
DEFAULT_INPUT = ROOT / f"data/{MAP_NAME}/{MAP_NAME}_mppi_track.csv"
DEFAULT_OUTPUT = ROOT / f"data/{MAP_NAME}/{MAP_NAME}_mppi_track_optimal.csv"


def intersect_boundary_along_normals(
        race: np.ndarray, yaw: np.ndarray, boundaries: tuple[np.ndarray, ...],
        expected_side: float, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Intersect each raceline normal with a closed physical boundary.

    ``expected_side`` is +1 for the left boundary and -1 for the right.  This
    preserves the geometric MPPI contract after a minimum-curvature line cuts
    corners; matching the two paths by normalized arclength does not.
    """
    # At tight bends a normal ray can hit the boundary whose historical CSV
    # label is opposite to the local raceline side.  Both polylines together
    # define the physical corridor; search both without adding a spurious
    # connecting segment between them.
    q = np.concatenate(boundaries, axis=0)
    segment = np.concatenate([
        np.roll(boundary, -1, axis=0) - boundary for boundary in boundaries],
        axis=0)
    intersections = np.empty_like(race)
    signed_widths = np.empty(len(race), dtype=float)
    eps = 1.0e-10
    for index, (point, heading) in enumerate(zip(race, yaw)):
        normal = np.array([-np.sin(heading), np.cos(heading)])
        delta = q - point
        denominator = normal[0] * segment[:, 1] - normal[1] * segment[:, 0]
        usable = np.abs(denominator) > eps
        t = np.full(len(segment), np.nan)
        u = np.full(len(segment), np.nan)
        t[usable] = ((delta[usable, 0] * segment[usable, 1]
                      - delta[usable, 1] * segment[usable, 0])
                     / denominator[usable])
        u[usable] = ((delta[usable, 0] * normal[1]
                      - delta[usable, 1] * normal[0])
                     / denominator[usable])
        valid = usable & (u >= -1.0e-8) & (u <= 1.0 + 1.0e-8)
        valid &= expected_side * t > 1.0e-6
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            raise RuntimeError(
                f"raceline point {index} has no {label} boundary intersection "
                "along its local normal; check track orientation/boundaries")
        chosen = candidates[np.argmin(np.abs(t[candidates]))]
        signed_widths[index] = t[chosen]
        intersections[index] = point + t[chosen] * normal
    return intersections, np.abs(signed_widths)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=DEFAULT_INPUT,
                        help="STEP3 MPPI-contract CSV containing center, widths, and explicit boundaries")
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path,
                        default=ROOT / f"model_tuning/results/{MAP_NAME}_raceline.png")
    parser.add_argument("--alpha-racer-root", type=Path, default=Path(
        "/home/a/alpha-RACER/global_racetrajectory_optimization"))
    parser.add_argument("--vehicle-width", type=float, default=1.25,
                        help="경계 최적화에 사용하는 차량 폭 [m]")
    parser.add_argument("--curvature-limit", type=float, default=8.0,
                        help="raceline 곡률 상한 [1/m]")
    parser.add_argument("--step", type=float, default=0.05,
                        help="출력 raceline 간격 [m]")
    parser.add_argument("--optimizer-step", type=float, default=0.40,
                        help="IQP 입력 centerline 간격 [m]. 출력 간격과 독립적이다.")
    parser.add_argument("--iqp-curvature-tolerance", type=float, default=0.25,
                        help="IQP 선형화 곡률 오차 종료 기준 [1/m]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.alpha_racer_root))
    import trajectory_planning_helpers as tph

    raw = np.genfromtxt(args.input, delimiter=",", names=True)
    required = {
        "x_m", "y_m", "w_tr_left_m", "w_tr_right_m",
        "left_x_m", "left_y_m", "right_x_m", "right_y_m",
    }
    available = set(raw.dtype.names or ())
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"{args.input} is not a STEP3 MPPI track CSV; missing columns "
            f"{missing}. Use {DEFAULT_INPUT} or pass --input explicitly.")
    # alpha-RACER contract: x, y, right width, left width. STEP3 stores a
    # dense 1 cm path; feeding all of it to IQP makes the QP unnecessarily
    # large. Periodically resample the complete contract at the requested
    # optimizer/output spacing while retaining the original CSV below for
    # physical-boundary projection.
    dense_reftrack = np.column_stack((raw["x_m"], raw["y_m"],
                                      raw["w_tr_right_m"], raw["w_tr_left_m"]))
    dense_xy_closed = np.vstack((dense_reftrack[:, :2], dense_reftrack[0, :2]))
    dense_segments = np.linalg.norm(np.diff(dense_xy_closed, axis=0), axis=1)
    dense_s = np.r_[0.0, np.cumsum(dense_segments)]
    if args.step <= 0.0 or args.optimizer_step <= 0.0:
        raise ValueError("--step and --optimizer-step must be positive")
    query_s = np.arange(0.0, dense_s[-1], args.optimizer_step)
    closed_contract = np.vstack((dense_reftrack, dense_reftrack[0]))
    reftrack = np.column_stack([
        np.interp(query_s, dense_s, closed_contract[:, column])
        for column in range(closed_contract.shape[1])])
    print(f"IQP reference resampled: {len(dense_reftrack)} -> {len(reftrack)} "
          f"points ({args.optimizer_step:.3f} m); output step={args.step:.3f} m")
    # The map utility already produces a smooth, equally sampled closed line.
    # Build the same spline/normal inputs that alpha-RACER's prep_track hands
    # to TUM's minimum-curvature QP. This also avoids its legacy SciPy fmin
    # compatibility path, which is unnecessary for this pre-refined map.
    prepared = reftrack
    closed_xy = np.vstack((prepared[:, :2], prepared[0, :2]))
    coeffs_x, coeffs_y, a_matrix, normals = tph.calc_splines.calc_splines(path=closed_xy)
    spline_len = tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=coeffs_x, coeffs_y=coeffs_y)
    psi, kappa, dkappa = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=coeffs_x, coeffs_y=coeffs_y,
        ind_spls=np.arange(coeffs_x.shape[0]), t_spls=np.zeros(coeffs_x.shape[0]),
        calc_curv=True, calc_dcurv=True)
    alpha, prepared, normals, *_ = tph.iqp_handler.iqp_handler(
        reftrack=prepared, normvectors=normals, A=a_matrix,
        spline_len=spline_len, psi=psi, kappa=kappa, dkappa=dkappa,
        kappa_bound=args.curvature_limit, w_veh=args.vehicle_width,
        print_debug=True, plot_debug=False,
        stepsize_interp=args.optimizer_step,
        iters_min=5, curv_error_allowed=args.iqp_curvature_tolerance)
    race, _, coeff_x, coeff_y, spline_idx, t_vals, *_ = (
        tph.create_raceline.create_raceline(
            refline=prepared[:, :2], normvectors=normals, alpha=alpha,
            stepsize_interp=args.step))
    yaw, curvature = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=coeff_x, coeffs_y=coeff_y, ind_spls=spline_idx,
        t_spls=t_vals)
    # trajectory_planning_helpers returns the normal-direction convention for
    # this create_raceline output. MPPI requires the geometric path tangent.
    # Convert explicitly and verify below rather than publishing a 90-degree
    # heading error in the track contract.
    yaw = (yaw + 0.5 * np.pi + np.pi) % (2.0 * np.pi) - np.pi
    geometric_yaw = np.arctan2(np.roll(race[:, 1], -1) - race[:, 1],
                               np.roll(race[:, 0], -1) - race[:, 0])
    heading_error = np.arctan2(np.sin(yaw - geometric_yaw),
                               np.cos(yaw - geometric_yaw))
    if np.percentile(np.abs(heading_error), 95) > 0.15:
        raise RuntimeError("optimized psi_rad is inconsistent with raceline tangent: "
                           f"p95={np.percentile(np.abs(heading_error), 95):.3f} rad")

    # IQP changes its working reference line and widths. Never reconstruct the
    # physical boundaries from that result.  More importantly, do not pair the
    # raceline and centerline by normalized lap progress: corner cutting makes
    # that boundary vector strongly tangential and overstates the usable road
    # width.  Intersect the local raceline normal with the original physical
    # boundary polylines instead.
    original_center = np.column_stack((raw["x_m"], raw["y_m"]))
    left_original = np.column_stack((raw["left_x_m"], raw["left_y_m"]))
    right_original = np.column_stack((raw["right_x_m"], raw["right_y_m"]))
    left_xy, left_width = intersect_boundary_along_normals(
        race, yaw, (left_original, right_original), +1.0, "left")
    right_xy, right_width = intersect_boundary_along_normals(
        race, yaw, (left_original, right_original), -1.0, "right")
    # Diagnostic-only reference point: midpoint of the two normal
    # intersections. MPPI consumes the explicit boundaries, not this column.
    center_xy = 0.5 * (left_xy + right_xy)
    left_tangent_error = np.abs(
        (left_xy[:, 0] - race[:, 0]) * np.cos(yaw)
        + (left_xy[:, 1] - race[:, 1]) * np.sin(yaw))
    right_tangent_error = np.abs(
        (right_xy[:, 0] - race[:, 0]) * np.cos(yaw)
        + (right_xy[:, 1] - race[:, 1]) * np.sin(yaw))
    print("normal-boundary validation: "
          f"left tangent p95={np.percentile(left_tangent_error, 95):.3e} m, "
          f"right tangent p95={np.percentile(right_tangent_error, 95):.3e} m, "
          f"width range=({left_width.min():.3f}, {left_width.max():.3f})/"
          f"({right_width.min():.3f}, {right_width.max():.3f}) m")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "y_m", "psi_rad", "kappa_radpm",
                         "w_tr_left_m", "w_tr_right_m", "w_total_m",
                         "boundary_ref_x_m", "boundary_ref_y_m",
                         "left_x_m", "left_y_m", "right_x_m", "right_y_m"])
        for i in range(len(race)):
            writer.writerow([race[i, 0], race[i, 1], yaw[i], curvature[i],
                             left_width[i], right_width[i],
                             left_width[i] + right_width[i],
                             center_xy[i, 0], center_xy[i, 1],
                             left_xy[i, 0], left_xy[i, 1],
                             right_xy[i, 0], right_xy[i, 1]])

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(left_xy[:, 0], left_xy[:, 1], "k-", lw=1, label="track boundary")
    ax.plot(right_xy[:, 0], right_xy[:, 1], "k-", lw=1)
    ax.plot(original_center[:, 0], original_center[:, 1], "--",
            color="0.65", label="centerline")
    points = ax.scatter(race[:, 0], race[:, 1], c=np.abs(curvature), s=8,
                        cmap="turbo", label="minimum-curvature raceline")
    fig.colorbar(points, ax=ax, label="|curvature| [1/m]")
    ax.axis("equal"); ax.grid(True); ax.legend(); ax.set_title(f"{MAP_NAME} optimized raceline")
    fig.tight_layout(); fig.savefig(args.plot, dpi=180)
    print(f"wrote {len(race)} points: {args.output}")
    print(f"alpha range [{alpha.min():.3f}, {alpha.max():.3f}] m, "
          f"max |kappa|={np.max(np.abs(curvature)):.3f} 1/m")
    print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
