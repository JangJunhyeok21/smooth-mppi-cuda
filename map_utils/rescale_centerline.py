from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("/home/a/RL-RACER/simulators/map_paths/berlin_2018/centerline.csv")
DEFAULT_SCALE = 1.5
DEFAULT_OUTPUT = Path("/home/a/RL-RACER/simulators/map_paths/berlin_2018/centerline_scaled_4p5.csv")


def load_points(csv_path: Path) -> list[tuple[float, float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV file: {csv_path}")

        points: list[tuple[float, float]] = []
        for row_idx, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"Expected at least 2 columns at row {row_idx} in {csv_path}")
            points.append((float(row[0]), float(row[1])))
    return points


def save_points(csv_path: Path, points: list[tuple[float, float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["x", "y"])
        for x, y in points:
            writer.writerow([f"{x:.8f}", f"{y:.8f}"])


def rescale_points(points: list[tuple[float, float]], scale: float) -> list[tuple[float, float]]:
    return [(x * scale, y * scale) for x, y in points]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescale centerline CSV coordinates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input centerline CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output centerline CSV path")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Scale factor for x and y coordinates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.input)
    scaled_points = rescale_points(points, args.scale)
    save_points(args.output, scaled_points)
    print(f"Saved {len(scaled_points)} points to {args.output}")


if __name__ == "__main__":
    main()
