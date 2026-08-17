from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

DEFAULT_CENTERLINE = Path("/home/a/RL-RACER/simulators/map_paths/berlin/centerline.csv")
DEFAULT_WIDTH_PROFILE = Path("/home/a/RL-RACER/simulators/map_paths/berlin/width_profile.csv")
DEFAULT_OUTPUT = Path("/home/a/RL-RACER/simulators/map_paths/berlin/width_profile_equal.csv")

REPO_ROOT = Path(__file__).resolve().parents[2]
WIDTH_SCRIPT_PATH = Path(__file__).resolve().with_name("compute_track_width_profile.py")


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


WIDTH_MODULE = _load_module(WIDTH_SCRIPT_PATH, "compute_track_width_profile_module_for_equal_width")
TrackMap = WIDTH_MODULE.TrackMap
save_width_profile = WIDTH_MODULE.save_width_profile
save_sample_points = WIDTH_MODULE.save_sample_points
sample_centerline_pose = WIDTH_MODULE.sample_centerline_pose

def default_xy_output_path(width_output: Path) -> Path:
    return width_output.with_name(f"{width_output.stem}_xy.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine a width profile so left/right widths become equal from the current centerline, using the smaller side and smooth interpolation over s.",
    )
    parser.add_argument("--centerline-csv", type=Path, default=DEFAULT_CENTERLINE, help="Centerline CSV path")
    parser.add_argument("--width-profile-csv", type=Path, default=DEFAULT_WIDTH_PROFILE, help="Input width_profile.csv path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output equalized width_profile.csv path")
    parser.add_argument("--xy-output", type=Path, default=None, help="Optional output equalized width_profile_xy.csv path")
    parser.add_argument("--tangent-window", type=float, default=1.0, help="Frenet arc-length window used to estimate local tangent for left/right normals")
    parser.add_argument("--interp-multiplier", type=int, default=8, help="Dense interpolation multiplier applied before smoothing")
    parser.add_argument("--smooth-window", type=int, default=31, help="Odd closed-loop moving-average window on the dense equal-width profile")
    parser.add_argument("--min-width", type=float, default=0.0, help="Clamp the equalized width to at least this value")
    return parser.parse_args()


def load_width_profile(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        values = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    except ValueError:
        values = np.loadtxt(csv_path, delimiter=",", dtype=np.float32, skiprows=1)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Width profile must have at least 3 columns (s,left_width,right_width): {csv_path}")
    return values[:, 0], values[:, 1], values[:, 2]


def smooth_closed_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or window <= 1:
        return arr.astype(np.float32)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    acc = np.zeros_like(arr, dtype=np.float64)
    for offset in range(-radius, radius + 1):
        acc += np.roll(arr, offset)
    return (acc / float(window)).astype(np.float32)


def periodic_interp(sample_s: np.ndarray, sample_values: np.ndarray, query_s: np.ndarray, total_length: float) -> np.ndarray:
    base_s = np.asarray(sample_s, dtype=np.float64)
    base_v = np.asarray(sample_values, dtype=np.float64)
    query = np.asarray(query_s, dtype=np.float64)
    if total_length <= 0.0:
        return np.full_like(query, fill_value=float(base_v[0]) if base_v.size else 0.0, dtype=np.float64)
    wrapped_s = np.mod(base_s, total_length)
    order = np.argsort(wrapped_s)
    wrapped_s = wrapped_s[order]
    base_v = base_v[order]
    extended_s = np.concatenate([wrapped_s - total_length, wrapped_s, wrapped_s + total_length])
    extended_v = np.concatenate([base_v, base_v, base_v])
    return np.interp(np.mod(query, total_length), extended_s, extended_v)


def equalize_widths(s_values: np.ndarray, left_width: np.ndarray, right_width: np.ndarray, interp_multiplier: int, smooth_window: int, min_width: float) -> np.ndarray:
    equal_raw = np.minimum(np.asarray(left_width, dtype=np.float32), np.asarray(right_width, dtype=np.float32))
    total_length = float(s_values[-1] + (s_values[1] - s_values[0])) if s_values.size > 1 else float(max(s_values[0], 1.0))
    dense_count = max(int(s_values.size * max(interp_multiplier, 1)), int(s_values.size))
    dense_s = np.linspace(0.0, total_length, num=max(dense_count, 1), endpoint=False, dtype=np.float64)
    dense_equal = periodic_interp(s_values, equal_raw, dense_s, total_length).astype(np.float32)
    dense_equal = smooth_closed_1d(dense_equal, smooth_window)
    equal_smooth = periodic_interp(dense_s, dense_equal, s_values, total_length).astype(np.float32)
    equal_smooth = smooth_closed_1d(equal_smooth, max(3, smooth_window // max(interp_multiplier, 1)))
    return np.maximum(equal_smooth, float(min_width)).astype(np.float32)


def build_equalized_rows(track_map: TrackMap, s_values: np.ndarray, equal_width: np.ndarray, tangent_window: float) -> list[dict[str, float]]:
    if len(s_values) != len(equal_width):
        raise ValueError(
            f"s_values and equal_width must have the same length, got {len(s_values)} and {len(equal_width)}"
        )
    rows: list[dict[str, float]] = []
    for s_value, width in zip(s_values, equal_width):
        center, normal = sample_centerline_pose(track_map, float(s_value), tangent_window=float(tangent_window))
        rows.append(
            {
                "s": float(s_value),
                "left_x": float(center[0] + normal[0] * float(width)),
                "left_y": float(center[1] + normal[1] * float(width)),
                "right_x": float(center[0] - normal[0] * float(width)),
                "right_y": float(center[1] - normal[1] * float(width)),
                "left_width": float(width),
                "right_width": float(width),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.interp_multiplier <= 0:
        raise ValueError(f"--interp-multiplier must be positive, got {args.interp_multiplier}")
    if args.smooth_window <= 0:
        raise ValueError(f"--smooth-window must be positive, got {args.smooth_window}")
    if args.tangent_window <= 0.0:
        raise ValueError(f"--tangent-window must be positive, got {args.tangent_window}")

    s_values, left_width, right_width = load_width_profile(args.width_profile_csv)
    equal_width = equalize_widths(
        s_values=s_values,
        left_width=left_width,
        right_width=right_width,
        interp_multiplier=int(args.interp_multiplier),
        smooth_window=int(args.smooth_window),
        min_width=float(args.min_width),
    )
    track_map = TrackMap.from_centerline_csv(str(args.centerline_csv), track_width=1.0, name=args.centerline_csv.stem)
    rows = build_equalized_rows(track_map, s_values, equal_width, tangent_window=float(args.tangent_window))
    save_width_profile(args.output, rows)
    xy_output = default_xy_output_path(args.output) if args.xy_output is None else args.xy_output
    save_sample_points(xy_output, rows)
    print(f"Saved {len(rows)} equal-width rows to {args.output}")
    print(f"Saved {len(rows)} equal-width boundary rows to {xy_output}")


if __name__ == "__main__":
    main()
