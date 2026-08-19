from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAP_UTILS_DIR = Path(__file__).resolve().parent
MPPI_DATA_DIR = MAP_UTILS_DIR.parent / "data"


def _load_local_cubic_spline_2d():
    package_name = "map_utils_runtime_centerline_refinement"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MAP_UTILS_DIR)]
    sys.modules[package_name] = package
    for module_name in ("CubicSpline1D", "CubicSpline2D"):
        full_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(full_name, MAP_UTILS_DIR / f"{module_name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load local {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.CubicSpline2D"].CubicSpline2D


CubicSpline2D = _load_local_cubic_spline_2d()

MAP_NAME = "berlin"
DEFAULT_CENTERLINE = Path(f"/home/a/RL-RACER/simulators/map_paths/{MAP_NAME}/centerline.csv")
DEFAULT_WIDTH_PROFILE = Path(f"/home/a/RL-RACER/simulators/map_paths/{MAP_NAME}/width_profile.csv")
DEFAULT_CENTERLINE_OUTPUT = Path(f"/home/a/RL-RACER/simulators/map_paths/{MAP_NAME}/centerline_equal.csv")
DEFAULT_WIDTH_OUTPUT = Path(f"/home/a/RL-RACER/simulators/map_paths/{MAP_NAME}/width_profile_equal.csv")

USE_SPLINE = True

WIDTH_SCRIPT_PATH = Path(__file__).resolve().with_name("compute_track_width_profile.py")


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


WIDTH_MODULE = _load_module(WIDTH_SCRIPT_PATH, "compute_track_width_profile_module_for_centerline_equalization")
TrackMap = WIDTH_MODULE.TrackMap
sample_centerline_pose = WIDTH_MODULE.sample_centerline_pose
save_width_profile = WIDTH_MODULE.save_width_profile
save_sample_points = WIDTH_MODULE.save_sample_points
save_mppi_track_csv = WIDTH_MODULE.save_mppi_track_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the existing left/right boundaries fixed and shift the centerline "
            "to their midpoint so the resulting left/right widths are equal."
        ),
    )
    parser.add_argument("--centerline-csv", type=Path, default=DEFAULT_CENTERLINE, help="Input centerline CSV")
    parser.add_argument("--width-profile-csv", type=Path, default=DEFAULT_WIDTH_PROFILE, help="Input asymmetric width profile")
    parser.add_argument("--centerline-output", type=Path, default=DEFAULT_CENTERLINE_OUTPUT, help="Output midpoint centerline CSV")
    parser.add_argument("--width-output", type=Path, default=DEFAULT_WIDTH_OUTPUT, help="Output equal-width profile CSV")
    parser.add_argument("--xy-output", type=Path, default=None, help="Output refined boundary-point CSV (default: <width-output-stem>_xy.csv)")
    parser.add_argument("--mppi-output", type=Path, default=None, help="MPPI centerline+lane CSV (default: ../data/<map>_centerline.csv)")
    parser.add_argument(
        "--tangent-window",
        type=float,
        default=1.0,
        help="Old-centerline arc-length window used to estimate the left normal",
    )
    parser.add_argument(
        "--spline-refine",
        default=USE_SPLINE,        
        help=(
            "Smooth the boundary-midpoint samples as a closed path with CubicSpline2D. "
            "This trades exact pointwise boundary preservation for continuous heading/curvature."
        ),
    )
    parser.add_argument(
        "--spline-prefilter-window",
        type=float,
        default=0.50,
        help="Circular moving-average window [m] applied before fitting spline knots (default: 0.50)",
    )
    parser.add_argument(
        "--spline-knot-spacing",
        type=float,
        default=0.35,
        help="Approximate closed-spline control-point spacing [m] (default: 0.35)",
    )
    parser.add_argument(
        "--spline-sample-step",
        type=float,
        default=0.01,
        help="Output centerline point spacing [m] when spline refinement is enabled (default: 0.01)",
    )
    return parser.parse_args()


def load_width_profile(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        values = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    except ValueError:
        values = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, skiprows=1)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Width profile must have columns (s,left_width,right_width): {csv_path}")
    if values.shape[0] < 2:
        raise ValueError(f"Width profile must contain at least two rows: {csv_path}")
    if not bool(np.isfinite(values[:, :3]).all()):
        raise ValueError(f"Width profile contains non-finite values: {csv_path}")
    if bool((values[:, 1:3] < 0.0).any()):
        raise ValueError(f"Width profile contains negative widths: {csv_path}")
    if bool((np.diff(values[:, 0]) <= 0.0).any()):
        raise ValueError(f"Width profile s values must be strictly increasing: {csv_path}")
    return values[:, 0], values[:, 1], values[:, 2]


def infer_map_name(centerline_csv: Path) -> str:
    return centerline_csv.parent.name if centerline_csv.stem in {"centerline", "centerline_equal"} else centerline_csv.stem


def equalize_centerline_samples(
    old_centers: np.ndarray,
    left_normals: np.ndarray,
    left_width: np.ndarray,
    right_width: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary-preserving midpoint centers and equal half-widths.

    With L=C+n*w_left and R=C-n*w_right, the midpoint is
    C_new=(L+R)/2=C+n*(w_left-w_right)/2 and both new half-widths are
    (w_left+w_right)/2. Thus the original lane boundaries and total width are
    unchanged at every input sample.
    """
    centers = np.asarray(old_centers, dtype=np.float64)
    normals = np.asarray(left_normals, dtype=np.float64)
    left = np.asarray(left_width, dtype=np.float64).reshape(-1)
    right = np.asarray(right_width, dtype=np.float64).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] != 2 or normals.shape != centers.shape:
        raise ValueError("old_centers and left_normals must both have shape [N, 2]")
    if not (centers.shape[0] == left.size == right.size):
        raise ValueError(
            "center, left-width, and right-width counts must match, "
            f"got {centers.shape[0]}, {left.size}, and {right.size}"
        )
    if not bool(np.isfinite(centers).all() and np.isfinite(normals).all()):
        raise ValueError("centerline samples and normals must be finite")
    normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
    if bool((normal_length <= 1e-8).any()):
        raise ValueError("left normals must have nonzero length")
    normals = normals / normal_length
    center_shift = 0.5 * (left - right)
    new_centers = centers + normals * center_shift[:, None]
    equal_width = 0.5 * (left + right)
    return new_centers.astype(np.float32), equal_width.astype(np.float32)


def sample_old_centerline_geometry(
    track_map: TrackMap,
    s_values: np.ndarray,
    tangent_window: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for s_value in s_values:
        center, normal = sample_centerline_pose(
            track_map,
            float(s_value),
            tangent_window=float(tangent_window),
        )
        centers.append(center)
        normals.append(normal)
    return np.asarray(centers, dtype=np.float32), np.asarray(normals, dtype=np.float32)


def centerline_arc_lengths(points: np.ndarray) -> np.ndarray:
    """Return each vertex's s coordinate along the new closed centerline."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError("centerline points must have shape [N, 2] with N >= 2")
    segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if bool((segment_lengths <= 1e-10).any()):
        raise ValueError("new centerline contains duplicate consecutive points")
    return np.concatenate(([0.0], np.cumsum(segment_lengths, dtype=np.float64))).astype(np.float32)


def closed_path_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum())


def resample_closed_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Sample a closed polyline at near-uniform arc-length intervals."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("closed polyline must have shape [N, 2] with N >= 3")
    if spacing <= 0.0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    closed = np.vstack((pts, pts[0]))
    segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if bool((segments <= 1e-10).any()):
        raise ValueError("closed polyline contains duplicate consecutive points")
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total = float(cumulative[-1])
    count = max(3, int(np.ceil(total / spacing)))
    query = np.linspace(0.0, total, count, endpoint=False)
    return np.column_stack(
        (np.interp(query, cumulative, closed[:, 0]), np.interp(query, cumulative, closed[:, 1]))
    ).astype(np.float32)


def circular_moving_average(points: np.ndarray, window_points: int) -> np.ndarray:
    """Smooth closed-path coordinates without introducing an endpoint seam."""
    pts = np.asarray(points, dtype=np.float64)
    window = max(1, int(window_points))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return pts.astype(np.float32)
    if window >= pts.shape[0]:
        raise ValueError("spline prefilter window must be smaller than the centerline")
    radius = window // 2
    result = sum(np.roll(pts, shift, axis=0) for shift in range(-radius, radius + 1)) / window
    return result.astype(np.float32)


@dataclass(frozen=True)
class SplinePoint:
    x: float
    y: float


def refine_closed_centerline_with_spline(
    midpoint_samples: np.ndarray,
    *,
    prefilter_window: float,
    knot_spacing: float,
    sample_step: float,
) -> np.ndarray:
    """Create a seam-safe, C2 midpoint path using the repository CubicSpline2D.

    CubicSpline2D uses natural endpoint conditions. Two wrapped knots before and
    three after the requested lap move those artificial endpoints away from the
    lap seam, so the returned central lap has continuous first/second derivatives.
    """
    for name, value in (
        ("prefilter_window", prefilter_window),
        ("knot_spacing", knot_spacing),
        ("sample_step", sample_step),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")

    raw = np.asarray(midpoint_samples, dtype=np.float64)
    median_step = float(np.median(np.linalg.norm(np.diff(raw, axis=0), axis=1)))
    window_points = max(1, int(round(prefilter_window / median_step)))
    filtered = circular_moving_average(raw, window_points)
    knots = resample_closed_polyline(filtered, knot_spacing)
    if knots.shape[0] < 6:
        raise ValueError("spline refinement requires at least six control points")

    padded = np.vstack((knots[-2:], knots, knots[:3]))
    spline = CubicSpline2D([SplinePoint(float(x), float(y)) for x, y in padded])
    start_s = float(spline.s[2])
    end_s = float(spline.s[2 + len(knots)])
    dense_step = min(sample_step * 0.5, knot_spacing / 10.0)
    query = np.arange(start_s, end_s, dense_step, dtype=np.float64)
    dense = np.asarray([[spline.calc_x(s), spline.calc_y(s)] for s in query], dtype=np.float64)
    if not bool(np.isfinite(dense).all()):
        raise ValueError("CubicSpline2D produced non-finite centerline coordinates")
    return resample_closed_polyline(dense, sample_step)


def resample_periodic_values(values: np.ndarray, output_count: int) -> np.ndarray:
    """Resample values by normalized lap phase, including across the lap seam."""
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    source_phase = np.arange(source.size + 1, dtype=np.float64) / source.size
    closed_values = np.concatenate((source, source[:1]))
    output_phase = np.arange(output_count, dtype=np.float64) / output_count
    return np.interp(output_phase, source_phase, closed_values).astype(np.float32)


def curvature_summary(points: np.ndarray) -> tuple[float, float, float]:
    """Return max segment, p95 |curvature|, and p95 adjacent curvature change."""
    pts = np.asarray(points, dtype=np.float64)
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    a = pts - prev
    b = nxt - pts
    c = nxt - prev
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    curvature = np.divide(2.0 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), denom, out=np.zeros_like(denom), where=denom > 1e-12)
    delta = np.abs(curvature - np.roll(curvature, 1))
    max_segment = float(np.linalg.norm(nxt - pts, axis=1).max())
    return max_segment, float(np.percentile(np.abs(curvature), 95.0)), float(np.percentile(delta, 95.0))


def save_centerline(csv_path: Path, points: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        for point in np.asarray(points, dtype=np.float32):
            writer.writerow([f"{float(point[0]):.8f}", f"{float(point[1]):.8f}"])


def build_width_rows(s_values: np.ndarray, equal_width: np.ndarray) -> list[dict[str, float]]:
    if len(s_values) != len(equal_width):
        raise ValueError(
            f"s_values and equal_width must have the same length, got {len(s_values)} and {len(equal_width)}"
        )
    return [
        {
            "s": float(s_value),
            "left_width": float(width),
            "right_width": float(width),
        }
        for s_value, width in zip(s_values, equal_width)
    ]


def build_boundary_rows(
    centerline: np.ndarray,
    s_values: np.ndarray,
    equal_width: np.ndarray,
    tangent_window: float,
) -> list[dict[str, float]]:
    track = TrackMap(centerline=np.asarray(centerline, dtype=np.float32), track_width=1.0, name="refined")
    rows: list[dict[str, float]] = []
    for s_value, width in zip(s_values, equal_width):
        center, normal = sample_centerline_pose(track, float(s_value), tangent_window=tangent_window)
        rows.append({
            "s": float(s_value),
            "left_width": float(width),
            "right_width": float(width),
            "left_x": float(center[0] + normal[0] * width),
            "left_y": float(center[1] + normal[1] * width),
            "right_x": float(center[0] - normal[0] * width),
            "right_y": float(center[1] - normal[1] * width),
        })
    return rows


def main() -> None:
    args = parse_args()
    if args.tangent_window <= 0.0:
        raise ValueError(f"--tangent-window must be positive, got {args.tangent_window}")

    old_s, left_width, right_width = load_width_profile(args.width_profile_csv)
    old_track = TrackMap.from_centerline_csv(
        str(args.centerline_csv),
        track_width=1.0,
        name=args.centerline_csv.stem,
    )
    old_centers, left_normals = sample_old_centerline_geometry(
        old_track,
        old_s,
        tangent_window=float(args.tangent_window),
    )
    new_centerline, equal_width = equalize_centerline_samples(
        old_centers,
        left_normals,
        left_width,
        right_width,
    )
    raw_centerline = new_centerline
    if args.spline_refine:
        new_centerline = refine_closed_centerline_with_spline(
            raw_centerline,
            prefilter_window=float(args.spline_prefilter_window),
            knot_spacing=float(args.spline_knot_spacing),
            sample_step=float(args.spline_sample_step),
        )
        equal_width = resample_periodic_values(equal_width, len(new_centerline))

    new_s = centerline_arc_lengths(new_centerline)
    width_rows = build_width_rows(new_s, equal_width)
    boundary_rows = build_boundary_rows(
        new_centerline, new_s, equal_width, tangent_window=float(args.tangent_window)
    )

    save_centerline(args.centerline_output, new_centerline)
    save_width_profile(args.width_output, width_rows)
    xy_output = (
        args.width_output.with_name(f"{args.width_output.stem}_xy.csv")
        if args.xy_output is None else args.xy_output
    )
    mppi_output = (
        MPPI_DATA_DIR / f"{infer_map_name(args.centerline_csv)}_centerline.csv"
        if args.mppi_output is None else args.mppi_output
    )
    save_sample_points(xy_output, boundary_rows)
    save_mppi_track_csv(mppi_output, boundary_rows)
    print(f"Saved {len(new_centerline)} midpoint centerline points to {args.centerline_output}")
    print(f"Saved {len(width_rows)} equal-width rows to {args.width_output}")
    print(f"Saved {len(boundary_rows)} refined boundary rows to {xy_output}")
    print(f"Saved {len(boundary_rows)} MPPI track rows to {mppi_output}")
    if args.spline_refine:
        raw_metrics = curvature_summary(raw_centerline)
        refined_metrics = curvature_summary(new_centerline)
        print(
            "Spline refinement metrics "
            f"(max segment, p95 |curvature|, p95 |delta curvature|): "
            f"raw={raw_metrics}, refined={refined_metrics}"
        )


if __name__ == "__main__":
    main()
