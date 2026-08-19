from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


_TRACK_MAP_CANDIDATES = [
    Path(__file__).resolve().parents[1] / "multi_car_rl" / "track_map.py",
    Path("/home/a/RL-RACER/multi_car_rl/track_map.py"),
]
TRACK_MAP_PATH = next((path for path in _TRACK_MAP_CANDIDATES if path.is_file()), None)
if TRACK_MAP_PATH is None:
    raise FileNotFoundError(
        "Could not find multi_car_rl/track_map.py in: "
        + ", ".join(str(path) for path in _TRACK_MAP_CANDIDATES)
    )
TRACK_MAP_SPEC = importlib.util.spec_from_file_location("track_map_module", TRACK_MAP_PATH)
if TRACK_MAP_SPEC is None or TRACK_MAP_SPEC.loader is None:
    raise ImportError(f"Could not load TrackMap module from {TRACK_MAP_PATH}")
TRACK_MAP_MODULE = importlib.util.module_from_spec(TRACK_MAP_SPEC)
sys.modules[TRACK_MAP_SPEC.name] = TRACK_MAP_MODULE
TRACK_MAP_SPEC.loader.exec_module(TRACK_MAP_MODULE)
TrackMap = TRACK_MAP_MODULE.TrackMap


DEFAULT_MAP_YAML = Path("/home/a/RL-RACER/simulators/maps/map1.yaml")
DEFAULT_CENTERLINE = Path("/home/a/RL-RACER/simulators/map_paths/map1/centerline.csv")
DEFAULT_OUTPUT = Path("/home/a/RL-RACER/simulators/map_paths/map1/width_profile.csv")


def default_xy_output_path(width_output: Path) -> Path:
    return width_output.with_name(f"{width_output.stem}_xy.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute left/right track widths along Frenet s from a centerline CSV and occupancy map.",
    )
    parser.add_argument("--map-yaml", type=Path, default=DEFAULT_MAP_YAML, help="ROS map YAML path")
    parser.add_argument("--centerline-csv", type=Path, default=DEFAULT_CENTERLINE, help="Centerline CSV used as the Frenet reference path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--xy-output", type=Path, default=None, help="Optional output CSV path for sampled s,x,y points (default: <output_stem>_xy.csv)")
    parser.add_argument("--s-step", type=float, default=0.01, help="Spacing in meters between sampled Frenet s values")
    parser.add_argument("--ray-step", type=float, default=0.01, help="Step size in meters while tracing toward the map boundary")
    parser.add_argument("--max-width", type=float, default=20.0, help="Maximum half-width to search on each side in meters")
    parser.add_argument("--fallback-hit-distance", type=float, default=3.0, help="Boundary-hit search distance; misses are filled from neighboring valid hits around the closed track")
    parser.add_argument("--occupied-margin", type=float, default=0.0, help="Optional safety margin subtracted from each measured half-width")
    parser.add_argument("--tangent-window", type=float, default=1.0, help="Frenet arc-length window used to estimate the local tangent for left/right normals")
    return parser.parse_args()


def load_map_yaml(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    required = {"image", "resolution", "origin", "negate", "occupied_thresh", "free_thresh"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Missing map YAML keys in {yaml_path}: {', '.join(missing)}")
    return data


def _read_pgm_token(fp) -> bytes:
    while True:
        token = fp.readline()
        if token == b"":
            raise ValueError("Unexpected EOF while reading PGM header")
        token = token.strip()
        if token and not token.startswith(b"#"):
            return token


def load_pgm(pgm_path: Path) -> np.ndarray:
    with pgm_path.open("rb") as fp:
        magic = _read_pgm_token(fp)
        if magic not in {b"P2", b"P5"}:
            raise ValueError(f"Unsupported PGM format {magic!r} in {pgm_path}")

        dims = _read_pgm_token(fp).split()
        if len(dims) != 2:
            raise ValueError(f"Expected width/height in {pgm_path}")
        width, height = (int(dims[0]), int(dims[1]))
        max_value = int(_read_pgm_token(fp))
        if max_value <= 0:
            raise ValueError(f"Invalid max value {max_value} in {pgm_path}")

        if magic == b"P5":
            dtype = np.uint8 if max_value < 256 else ">u2"
            image = np.frombuffer(fp.read(), dtype=dtype)
        else:
            payload = fp.read().split()
            image = np.asarray(payload, dtype=np.uint16 if max_value >= 256 else np.uint8)

    expected = width * height
    if image.size != expected:
        raise ValueError(f"Expected {expected} pixels in {pgm_path}, found {image.size}")
    return image.reshape(height, width).astype(np.float32)


def build_traversable_mask(image: np.ndarray, negate: int, occupied_thresh: float, free_thresh: float) -> np.ndarray:
    normalized = image / 255.0
    occupancy = normalized if negate else 1.0 - normalized
    return occupancy < float(free_thresh)


def world_to_grid(position: np.ndarray, origin: np.ndarray, resolution: float, image_shape: tuple[int, int]) -> tuple[int, int] | None:
    col = int(math.floor((float(position[0]) - float(origin[0])) / resolution))
    row_from_bottom = int(math.floor((float(position[1]) - float(origin[1])) / resolution))
    height, width = image_shape
    row = height - 1 - row_from_bottom
    if row < 0 or row >= height or col < 0 or col >= width:
        return None
    return row, col


def is_traversable(position: np.ndarray, traversable_mask: np.ndarray, origin: np.ndarray, resolution: float) -> bool:
    grid = world_to_grid(position, origin, resolution, traversable_mask.shape)
    if grid is None:
        return False
    row, col = grid
    return bool(traversable_mask[row, col])


def build_s_samples(total_length: float, spacing: float) -> np.ndarray:
    if spacing <= 0.0:
        raise ValueError(f"s-step must be positive, got {spacing}")
    if total_length <= 0.0:
        return np.zeros((1,), dtype=np.float32)
    samples = np.arange(0.0, total_length, spacing, dtype=np.float64)
    if samples.size == 0:
        samples = np.array([0.0], dtype=np.float64)
    return samples.astype(np.float32)


def sample_centerline_point(track_map: TrackMap, s_value: float) -> np.ndarray:
    if track_map.total_length <= 0.0:
        return track_map.centerline[0].astype(np.float32)

    s_wrapped = float(s_value % track_map.total_length)
    cumulative = track_map.cumulative_lengths
    idx = int(np.searchsorted(cumulative, s_wrapped, side="right") - 1)
    idx = track_map.wrapped_index(idx)
    next_idx = track_map.wrapped_index(idx + 1)
    start = track_map.centerline[idx].astype(np.float32)
    end = track_map.centerline[next_idx].astype(np.float32)
    segment = end - start
    seg_len = float(np.linalg.norm(segment))

    if seg_len < 1e-8:
        return start

    along = (s_wrapped - float(cumulative[idx])) / seg_len
    along = float(np.clip(along, 0.0, 1.0))
    point = start + segment * along
    return point.astype(np.float32)


def sample_centerline_pose(track_map: TrackMap, s_value: float, tangent_window: float) -> tuple[np.ndarray, np.ndarray]:
    point = sample_centerline_point(track_map, s_value)
    if track_map.total_length <= 0.0:
        return point, np.array([0.0, 1.0], dtype=np.float32)

    step = max(float(tangent_window), 1e-3)
    prev_point = sample_centerline_point(track_map, s_value - step)
    next_point = sample_centerline_point(track_map, s_value + step)
    tangent = next_point - prev_point
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-8:
        idx = track_map.index_at_s(float(s_value % track_map.total_length))
        tangent = track_map.tangent_at(idx)
        tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-8:
        return point, np.array([0.0, 1.0], dtype=np.float32)
    tangent = tangent / tangent_norm
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    return point.astype(np.float32), normal.astype(np.float32)


def trace_half_width(
    center: np.ndarray,
    normal: np.ndarray,
    direction: float,
    traversable_mask: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    ray_step: float,
    max_width: float,
    fallback_hit_distance: float,
) -> tuple[float, bool]:
    last_free = 0.0
    distance = ray_step
    # Search the full user-requested range. Boundary fallback is handled after
    # all samples are traced, using the previous valid boundary point.
    search_limit = max_width
    while distance <= search_limit:
        probe = center + normal * (direction * distance)
        if not is_traversable(probe, traversable_mask, origin, resolution):
            low = last_free
            high = distance
            for _ in range(12):
                mid = 0.5 * (low + high)
                midpoint = center + normal * (direction * mid)
                if is_traversable(midpoint, traversable_mask, origin, resolution):
                    low = mid
                else:
                    high = mid
            return float(low), True
        last_free = distance
        distance += ray_step
    return float(last_free), False


def interpolate_missing_widths_circular(
    s_values: np.ndarray,
    widths: np.ndarray,
    hit_mask: np.ndarray,
    total_length: float,
) -> np.ndarray:
    """Fill missed boundary rays from neighboring real hits on a closed track.

    A missed ray must not be treated as a measured maximum lane width.  Linear
    interpolation from valid hits on both sides gives a continuous fallback;
    ``period`` also prevents a jump where the CSV wraps from its last row back
    to its first row.
    """
    samples = np.asarray(s_values, dtype=np.float64)
    values = np.asarray(widths, dtype=np.float64)
    valid = np.asarray(hit_mask, dtype=bool) & np.isfinite(values)
    if samples.shape != values.shape or samples.shape != valid.shape:
        raise ValueError("s_values, widths, and hit_mask must have identical shapes")
    if values.size == 0 or valid.all() or not valid.any():
        return values.astype(np.float32)
    if int(valid.sum()) == 1:
        return np.full(values.shape, values[valid][0], dtype=np.float32)
    period = float(total_length)
    if period <= 0.0:
        filled = np.interp(samples, samples[valid], values[valid])
    else:
        filled = np.interp(samples, samples[valid], values[valid], period=period)
    return np.where(valid, values, filled).astype(np.float32)


def traversable_boundary_points(
    traversable_mask: np.ndarray,
    origin: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """World coordinates of free grid cells touching a non-free cell."""
    free = np.asarray(traversable_mask, dtype=bool)
    interior = free.copy()
    interior[1:, :] &= free[:-1, :]
    interior[:-1, :] &= free[1:, :]
    interior[:, 1:] &= free[:, :-1]
    interior[:, :-1] &= free[:, 1:]
    rows, cols = np.nonzero(free & ~interior)
    height = free.shape[0]
    xs = float(origin[0]) + (cols.astype(np.float64) + 0.5) * resolution
    ys = float(origin[1]) + (height - 1 - rows.astype(np.float64) + 0.5) * resolution
    return np.column_stack((xs, ys)).astype(np.float32)


def compute_width_profile(
    track_map: TrackMap,
    traversable_mask: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    s_step: float,
    ray_step: float,
    max_width: float,
    fallback_hit_distance: float,
    occupied_margin: float,
    tangent_window: float,
    missing_width_fallback: str = "previous_boundary",
) -> list[dict[str, float]]:
    if missing_width_fallback not in {"previous_boundary", "nearest_boundary", "max_width"}:
        raise ValueError(
            "missing_width_fallback must be 'previous_boundary', "
            "'nearest_boundary', or 'max_width', "
            f"got {missing_width_fallback!r}"
        )
    rows: list[dict[str, float]] = []
    centers: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    left_hits: list[bool] = []
    right_hits: list[bool] = []
    for s_value in build_s_samples(track_map.total_length, s_step):
        center, normal = sample_centerline_pose(track_map, float(s_value), tangent_window=tangent_window)
        left_width, left_hit = trace_half_width(
            center,
            normal,
            1.0,
            traversable_mask,
            origin,
            resolution,
            ray_step,
            max_width,
            fallback_hit_distance,
        )
        right_width, right_hit = trace_half_width(
            center,
            normal,
            -1.0,
            traversable_mask,
            origin,
            resolution,
            ray_step,
            max_width,
            fallback_hit_distance,
        )
        if missing_width_fallback == "max_width":
            if not left_hit:
                left_width = max_width
            if not right_hit:
                right_width = max_width
        left_width = max(0.0, left_width - occupied_margin)
        right_width = max(0.0, right_width - occupied_margin)
        centers.append(center)
        normals.append(normal)
        left_hits.append(left_hit)
        right_hits.append(right_hit)
        rows.append(
            {
                "s": float(s_value),
                "left_width": float(left_width),
                "right_width": float(right_width),
                "left_hit": float(left_hit),
                "right_hit": float(right_hit),
            }
        )

    if not rows:
        return rows
    for idx, row in enumerate(rows):
        center = centers[idx]
        normal = normals[idx]
        left_width = float(row["left_width"])
        right_width = float(row["right_width"])
        row["left_width"] = left_width
        row["right_width"] = right_width
        row["left_x"] = float(center[0] + normal[0] * left_width)
        row["left_y"] = float(center[1] + normal[1] * left_width)
        row["right_x"] = float(center[0] - normal[0] * right_width)
        row["right_y"] = float(center[1] - normal[1] * right_width)

    # If a ray finds no boundary within max_width, retain the previous valid
    # boundary x,y on that side and measure the width from the current center
    # to that retained point. Start from a real hit and wrap once, so misses at
    # the CSV seam correctly inherit the last valid point of the lap.
    map_boundary_points = (
        traversable_boundary_points(traversable_mask, origin, resolution)
        if missing_width_fallback == "nearest_boundary" else None
    )
    fallback_sides = (
        (("left", left_hits), ("right", right_hits))
        if missing_width_fallback in {"previous_boundary", "nearest_boundary"} else ()
    )
    for side, hit_mask in fallback_sides:
        valid_indices = [idx for idx, hit in enumerate(hit_mask) if hit]
        if not valid_indices:
            continue
        start = valid_indices[0]
        previous = np.asarray(
            [rows[start][f"{side}_x"], rows[start][f"{side}_y"]], dtype=np.float64
        )
        for offset in range(1, len(rows)):
            idx = (start + offset) % len(rows)
            if hit_mask[idx]:
                previous = np.asarray(
                    [rows[idx][f"{side}_x"], rows[idx][f"{side}_y"]], dtype=np.float64
                )
                continue
            center = np.asarray(centers[idx], dtype=np.float64)
            if map_boundary_points is not None and map_boundary_points.size:
                candidates = map_boundary_points.astype(np.float64, copy=False)
                near_previous = np.linalg.norm(candidates - previous, axis=1) <= fallback_hit_distance
                direction = 1.0 if side == "left" else -1.0
                lateral = (candidates - center) @ np.asarray(normals[idx], dtype=np.float64)
                valid = near_previous & (lateral * direction > 0.0)
                if bool(valid.any()):
                    local_candidates = candidates[valid]
                    distances = np.linalg.norm(local_candidates - center, axis=1)
                    previous = local_candidates[int(np.argmin(distances))]
            rows[idx][f"{side}_x"] = float(previous[0])
            rows[idx][f"{side}_y"] = float(previous[1])
            rows[idx][f"{side}_width"] = float(np.linalg.norm(previous - center))
    return rows


def save_width_profile(csv_path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["s", "left_width", "right_width"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{row[key]:.8f}" for key in fieldnames})


def save_sample_points(csv_path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["s", "left_x", "left_y", "right_x", "right_y"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{row[key]:.8f}" for key in fieldnames})


def save_mppi_track_csv(csv_path: Path, rows: list[dict[str, float]]) -> None:
    """Save the single-file centerline/lane format consumed by PathPublisher.

    The center point is reconstructed from the two boundary points and their
    asymmetric widths, so the output remains consistent with the exact
    boundary samples produced by the map extraction/refinement steps.
    """
    fieldnames = [
        "x_m", "y_m", "w_tr_left_m", "w_tr_right_m", "w_total_m",
        "left_x_m", "left_y_m", "right_x_m", "right_y_m",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            left_width = float(row["left_width"])
            right_width = float(row["right_width"])
            total_width = left_width + right_width
            left_x, left_y = float(row["left_x"]), float(row["left_y"])
            right_x, right_y = float(row["right_x"]), float(row["right_y"])
            if total_width > 1e-9:
                center_x = (right_width * left_x + left_width * right_x) / total_width
                center_y = (right_width * left_y + left_width * right_y) / total_width
            else:
                center_x = 0.5 * (left_x + right_x)
                center_y = 0.5 * (left_y + right_y)
            output = {
                "x_m": center_x,
                "y_m": center_y,
                "w_tr_left_m": left_width,
                "w_tr_right_m": right_width,
                "w_total_m": total_width,
                "left_x_m": left_x,
                "left_y_m": left_y,
                "right_x_m": right_x,
                "right_y_m": right_y,
            }
            writer.writerow({key: f"{output[key]:.8f}" for key in fieldnames})


def main() -> None:
    args = parse_args()
    if args.s_step <= 0.0:
        raise ValueError(f"--s-step must be positive, got {args.s_step}")
    if args.ray_step <= 0.0:
        raise ValueError(f"--ray-step must be positive, got {args.ray_step}")
    if args.max_width <= 0.0:
        raise ValueError(f"--max-width must be positive, got {args.max_width}")
    if args.fallback_hit_distance <= 0.0:
        raise ValueError(f"--fallback-hit-distance must be positive, got {args.fallback_hit_distance}")
    if args.tangent_window <= 0.0:
        raise ValueError(f"--tangent-window must be positive, got {args.tangent_window}")

    map_cfg = load_map_yaml(args.map_yaml)
    image_path = args.map_yaml.parent / Path(map_cfg["image"])
    image = load_pgm(image_path)
    traversable_mask = build_traversable_mask(
        image=image,
        negate=int(map_cfg["negate"]),
        occupied_thresh=float(map_cfg["occupied_thresh"]),
        free_thresh=float(map_cfg["free_thresh"]),
    )
    track_map = TrackMap.from_centerline_csv(str(args.centerline_csv), track_width=1.0, name=args.centerline_csv.stem)
    rows = compute_width_profile(
        track_map=track_map,
        traversable_mask=traversable_mask,
        origin=np.asarray(map_cfg["origin"][:2], dtype=np.float32),
        resolution=float(map_cfg["resolution"]),
        s_step=float(args.s_step),
        ray_step=float(args.ray_step),
        max_width=float(args.max_width),
        fallback_hit_distance=float(args.fallback_hit_distance),
        occupied_margin=float(args.occupied_margin),
        tangent_window=float(args.tangent_window),
    )
    save_width_profile(args.output, rows)
    xy_output = default_xy_output_path(args.output) if args.xy_output is None else args.xy_output
    save_sample_points(xy_output, rows)
    print(f"Saved {len(rows)} width samples to {args.output}")
    print(f"Saved {len(rows)} sampled points to {xy_output}")


if __name__ == "__main__":
    main()
