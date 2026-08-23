#!/usr/bin/env python3
"""Create a reverse-direction copy of a map/track directory."""

import argparse
import csv
import math
import shutil
from pathlib import Path

import numpy as np
import yaml


def cumulative_s(points: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def reverse_plain_centerline(source: Path, target: Path) -> None:
    points = np.loadtxt(source, delimiter=",", dtype=float)
    np.savetxt(target, points[::-1], delimiter=",", fmt="%.8f")


def reverse_dict_csv(source: Path, target: Path) -> None:
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)[::-1]
    if not fields:
        raise ValueError(f"Missing CSV header: {source}")

    swap_pairs = (
        ("left_width", "right_width"),
        ("w_tr_left_m", "w_tr_right_m"),
        ("left_x", "right_x"),
        ("left_y", "right_y"),
        ("left_x_m", "right_x_m"),
        ("left_y_m", "right_y_m"),
    )
    for row in rows:
        for left, right in swap_pairs:
            if left in row and right in row:
                row[left], row[right] = row[right], row[left]

        # Reversing traversal changes the tangent by pi and changes signed
        # curvature.  Keeping the forward values creates a geometrically
        # reversed CSV whose vehicle heading/dynamics still point clockwise.
        if "psi_rad" in row and row["psi_rad"]:
            yaw = float(row["psi_rad"]) + math.pi
            row["psi_rad"] = f"{math.atan2(math.sin(yaw), math.cos(yaw)):.12g}"
        if "kappa_radpm" in row and row["kappa_radpm"]:
            row["kappa_radpm"] = f"{-float(row['kappa_radpm']):.12g}"

    if "s" in fields and rows:
        if "x_m" in fields and "y_m" in fields:
            points = np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in rows])
        elif "left_x" in fields and "right_x" in fields:
            points = np.asarray([[(float(row["left_x"]) + float(row["right_x"])) * 0.5,
                                  (float(row["left_y"]) + float(row["right_y"])) * 0.5]
                                 for row in rows])
        else:
            old_s = np.asarray([float(row["s"]) for row in rows[::-1]])
            new_s = old_s[-1] - old_s[::-1]
            points = None
        distances = cumulative_s(points) if points is not None else new_s
        for row, distance in zip(rows, distances):
            row["s"] = f"{distance:.8f}"

    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    source_name, target_name = source.name, target.name
    for path in source.iterdir():
        if not path.is_file():
            continue
        output_name = path.name.replace(source_name, target_name)
        output = target / output_name
        if path.suffix == ".csv":
            first_line = path.open().readline().strip()
            if first_line.lower().startswith("x_m,") or "left_" in first_line or first_line.startswith("s,"):
                reverse_dict_csv(path, output)
            else:
                reverse_plain_centerline(path, output)
        elif path.suffix == ".pgm":
            shutil.copy2(path, output)
        elif path.suffix in (".yaml", ".yml"):
            metadata = yaml.safe_load(path.read_text())
            metadata["image"] = str(metadata["image"]).replace(source_name, target_name)
            output.write_text(yaml.safe_dump(metadata, sort_keys=False))

    print(f"Created reverse track: {source} -> {target}")


if __name__ == "__main__":
    main()
