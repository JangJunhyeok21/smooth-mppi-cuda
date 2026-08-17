from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("/home/a/RL-RACER/simulators/map_paths/map1/width_profile_xy.csv")
DEFAULT_CENTERLINE_OUTPUT = Path("/home/a/RL-RACER/simulators/map_paths/map1/centerline.csv")
DEFAULT_WIDTH_OUTPUT = Path("/home/a/RL-RACER/simulators/map_paths/map1/width_profile.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild centerline.csv and width_profile.csv from refined width boundary XY samples.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input refined_width_profile_xy.csv path")
    parser.add_argument("--centerline-output", type=Path, default=DEFAULT_CENTERLINE_OUTPUT, help="Output centerline.csv path")
    parser.add_argument("--width-output", type=Path, default=DEFAULT_WIDTH_OUTPUT, help="Output width_profile.csv path")
    parser.add_argument("--smooth-window", type=int, default=7, help="Odd moving-average window for midpoint smoothing")
    parser.add_argument("--chaikin-iters", type=int, default=3, help="Number of Chaikin corner-cutting iterations for smooth closed-loop reconstruction")
    return parser.parse_args()


def load_boundary_points(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    required = {"s", "left_x", "left_y", "right_x", "right_y"}
    if not rows:
        raise ValueError(f"Empty CSV file: {csv_path}")
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        raise ValueError(f"CSV must contain columns {sorted(required)}: {csv_path}")
    s_values = np.asarray([float(row["s"]) for row in rows], dtype=np.float32)
    left = np.asarray([[float(row["left_x"]), float(row["left_y"])] for row in rows], dtype=np.float32)
    right = np.asarray([[float(row["right_x"]), float(row["right_y"])] for row in rows], dtype=np.float32)
    return s_values, left, right


def smooth_closed(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return points.astype(np.float32)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    acc = np.zeros_like(points, dtype=np.float64)
    for offset in range(-radius, radius + 1):
        acc += np.roll(points, offset, axis=0)
    return (acc / float(window)).astype(np.float32)


def resample_closed(points: np.ndarray, target_count: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")
    if pts.shape[0] == 1:
        return np.repeat(pts, target_count, axis=0).astype(np.float32)

    segments = np.roll(pts, -1, axis=0) - pts
    seg_lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.zeros(pts.shape[0] + 1, dtype=np.float64)
    cumulative[1:] = np.cumsum(seg_lengths, dtype=np.float64)
    total_length = float(cumulative[-1])
    if total_length <= 1e-8:
        return np.repeat(pts[:1], target_count, axis=0).astype(np.float32)

    sample_s = np.linspace(0.0, total_length, num=target_count, endpoint=False, dtype=np.float64)
    closed = np.vstack([pts, pts[:1]])
    x = np.interp(sample_s, cumulative, closed[:, 0])
    y = np.interp(sample_s, cumulative, closed[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def chaikin_closed(points: np.ndarray, iterations: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    for _ in range(max(int(iterations), 0)):
        next_pts = np.roll(pts, -1, axis=0)
        q = 0.75 * pts + 0.25 * next_pts
        r = 0.25 * pts + 0.75 * next_pts
        refined = np.empty((pts.shape[0] * 2, 2), dtype=np.float32)
        refined[0::2] = q
        refined[1::2] = r
        pts = refined
    return pts.astype(np.float32)


def midpoint_centerline(left_points: np.ndarray, right_points: np.ndarray, smooth_window: int, chaikin_iters: int) -> np.ndarray:
    midpoints = 0.5 * (left_points + right_points)
    smoothed = smooth_closed(midpoints, smooth_window)
    smoothed = chaikin_closed(smoothed, chaikin_iters)
    smoothed = resample_closed(smoothed, target_count=midpoints.shape[0])
    return smooth_closed(smoothed, smooth_window)


def normal_vectors(points: np.ndarray) -> np.ndarray:
    prev_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    tangents = next_points - prev_points
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    tangents = tangents / norms
    return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1).astype(np.float32)


def width_rows_from_boundaries(s_values: np.ndarray, centerline: np.ndarray, left_points: np.ndarray, right_points: np.ndarray) -> list[dict[str, float]]:
    normals = normal_vectors(centerline)
    left_delta = left_points - centerline
    right_delta = centerline - right_points
    left_width = np.maximum(np.sum(left_delta * normals, axis=1), 0.0)
    right_width = np.maximum(np.sum(right_delta * normals, axis=1), 0.0)
    rows: list[dict[str, float]] = []
    if not (len(s_values) == len(left_width) == len(right_width)):
        raise ValueError(
            "s_values, left_width, and right_width must have the same length, "
            f"got {len(s_values)}, {len(left_width)}, and {len(right_width)}"
        )
    for s_value, lw, rw in zip(s_values, left_width, right_width):
        rows.append({
            "s": float(s_value),
            "left_width": float(lw),
            "right_width": float(rw),
        })
    return rows


def save_centerline(csv_path: Path, points: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        for point in np.asarray(points, dtype=np.float32):
            writer.writerow([f"{float(point[0]):.8f}", f"{float(point[1]):.8f}"])


def save_width_profile(csv_path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["s", "left_width", "right_width"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{row[key]:.8f}" for key in fieldnames})


def main() -> None:
    args = parse_args()
    s_values, left_points, right_points = load_boundary_points(args.input)
    centerline = midpoint_centerline(
        left_points,
        right_points,
        smooth_window=int(args.smooth_window),
        chaikin_iters=int(args.chaikin_iters),
    )
    width_rows = width_rows_from_boundaries(s_values, centerline, left_points, right_points)
    save_centerline(args.centerline_output, centerline)
    save_width_profile(args.width_output, width_rows)
    print(f"Saved {centerline.shape[0]} centerline points to {args.centerline_output}")
    print(f"Saved {len(width_rows)} width rows to {args.width_output}")


if __name__ == "__main__":
    main()
