from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.backend_bases import Event, KeyEvent, MouseEvent
from matplotlib.lines import Line2D



MAP_NAME = "ifac2026"
ROOT = Path(__file__).resolve().parents[1]

USE_EQUAL = True
if USE_EQUAL:
    DEFAULT_MAP_YAML = Path(f"{ROOT}/data/{MAP_NAME}/{MAP_NAME}.yaml")
    DEFAULT_INPUT = Path(f"{ROOT}/data/{MAP_NAME}/width_profile_xy.csv")
    DEFAULT_CENTERLINE = Path(f"{ROOT}/data/{MAP_NAME}/centerline_equal.csv")
    DEFAULT_OUTPUT = Path(f"{ROOT}/data/{MAP_NAME}/refined_width_profile_xy.csv")
else:
    DEFAULT_MAP_YAML = Path(f"{ROOT}/data/{MAP_NAME}/{MAP_NAME}.yaml")
    DEFAULT_INPUT = Path(f"{ROOT}/data/{MAP_NAME}/width_profile_xy.csv")
    DEFAULT_CENTERLINE = Path(f"{ROOT}/data/{MAP_NAME}/centerline.csv")
    DEFAULT_OUTPUT = Path(f"{ROOT}/data/{MAP_NAME}/refined_width_profile_xy.csv")

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


WIDTH_MODULE = _load_module(WIDTH_SCRIPT_PATH, "compute_track_width_profile_module_for_gui")
load_map_yaml = WIDTH_MODULE.load_map_yaml
load_pgm = WIDTH_MODULE.load_pgm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive GUI for refining left/right width boundary points on a map.")
    parser.add_argument("--map-yaml", type=Path, default=DEFAULT_MAP_YAML, help="ROS map YAML path")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input width_profile_xy CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output refined width_profile_xy CSV path")
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE, help="Optional centerline CSV path to overlay")
    parser.add_argument("--show-matched-lanes", dest="show_matched_lanes", action="store_true", default=True, help="Show line segments from each centerline point to its matched left/right lane points")
    parser.add_argument("--no-show-matched-lanes", dest="show_matched_lanes", action="store_false", help="Hide line segments from each centerline point to its matched left/right lane points")
    parser.add_argument("--point-size", type=float, default=12.0, help="Boundary point marker size")
    parser.add_argument("--pick-radius", type=float, default=0.35, help="Selection radius in world meters")
    return parser.parse_args()


def load_boundary_points(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    required = {"s", "left_x", "left_y", "right_x", "right_y"}
    if not rows:
        raise ValueError(f"Empty CSV file: {csv_path}")
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        raise ValueError(f"CSV must contain columns {sorted(required)}: {csv_path}")
    return {
        "s": np.asarray([float(row["s"]) for row in rows], dtype=np.float32),
        "left": np.asarray([[float(row["left_x"]), float(row["left_y"])] for row in rows], dtype=np.float32),
        "right": np.asarray([[float(row["right_x"]), float(row["right_y"])] for row in rows], dtype=np.float32),
    }


def load_centerline(csv_path: Path) -> np.ndarray:
    try:
        return np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    except ValueError:
        return np.loadtxt(csv_path, delimiter=",", dtype=np.float32, skiprows=1)


def save_boundary_points(csv_path: Path, s_values: np.ndarray, left_points: np.ndarray, right_points: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["s", "left_x", "left_y", "right_x", "right_y"])
        if not (len(s_values) == len(left_points) == len(right_points)):
            raise ValueError(
                "s_values, left_points, and right_points must have the same length, "
                f"got {len(s_values)}, {len(left_points)}, and {len(right_points)}"
            )
        for s_value, left, right in zip(s_values, left_points, right_points):
            writer.writerow(
                [
                    f"{float(s_value):.8f}",
                    f"{float(left[0]):.8f}",
                    f"{float(left[1]):.8f}",
                    f"{float(right[0]):.8f}",
                    f"{float(right[1]):.8f}",
                ]
            )


def map_extent(map_cfg: dict, image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    resolution = float(map_cfg["resolution"])
    origin = np.asarray(map_cfg["origin"][:2], dtype=np.float32)
    height, width = image_shape
    x0 = float(origin[0])
    x1 = x0 + float(width) * resolution
    y0 = float(origin[1])
    y1 = y0 + float(height) * resolution
    return (x0, x1, y0, y1)


def matched_boundary_points(centerline: np.ndarray, boundary_points: np.ndarray) -> np.ndarray:
    centerline_arr = np.asarray(centerline, dtype=np.float32)
    boundary_arr = np.asarray(boundary_points, dtype=np.float32)
    if centerline_arr.shape[0] == 0 or boundary_arr.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    if centerline_arr.shape[0] == 1:
        return boundary_arr[:1].astype(np.float32)
    indices = np.rint(np.linspace(0, boundary_arr.shape[0] - 1, num=centerline_arr.shape[0], endpoint=True)).astype(np.int32)
    return boundary_arr[indices].astype(np.float32)


def lane_match_segments(centerline: np.ndarray, boundary_points: np.ndarray) -> np.ndarray:
    centerline_arr = np.asarray(centerline, dtype=np.float32)
    matched_boundary = matched_boundary_points(centerline_arr, boundary_points)
    if centerline_arr.shape[0] == 0 or matched_boundary.shape[0] == 0:
        return np.empty((0, 2, 2), dtype=np.float32)
    return np.stack([centerline_arr, matched_boundary], axis=1).astype(np.float32)


def lane_match_segment_list(centerline: np.ndarray, boundary_points: np.ndarray) -> list[np.ndarray]:
    return [segment for segment in lane_match_segments(centerline, boundary_points)]


class WidthProfileEditor:
    def __init__(
        self,
        map_image: np.ndarray,
        map_cfg: dict,
        s_values: np.ndarray,
        left_points: np.ndarray,
        right_points: np.ndarray,
        centerline: np.ndarray | None,
        show_matched_lanes: bool,
        output_path: Path,
        point_size: float,
        pick_radius: float,
    ) -> None:
        self.map_image = np.flipud(map_image)
        self.map_cfg = map_cfg
        self.s_values = s_values
        self.left_points = left_points.copy()
        self.right_points = right_points.copy()
        self.centerline = None if centerline is None else np.asarray(centerline, dtype=np.float32)
        self.show_matched_lanes = bool(show_matched_lanes)
        self.output_path = output_path
        self.pick_radius = float(pick_radius)

        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Width Profile Refiner")
        self.ax.imshow(
            self.map_image,
            cmap="gray",
            origin="lower",
            extent=map_extent(self.map_cfg, self.map_image.shape),
            interpolation="nearest",
        )
        self.left_line = Line2D(self.left_points[:, 0], self.left_points[:, 1], color="tab:blue", linewidth=1.0, alpha=0.9)
        self.right_line = Line2D(self.right_points[:, 0], self.right_points[:, 1], color="tab:red", linewidth=1.0, alpha=0.9)
        self.centerline_line = None
        if self.centerline is not None and self.centerline.size > 0:
            centerline_arr = self.centerline
            self.centerline_line = Line2D(centerline_arr[:, 0], centerline_arr[:, 1], color="tab:green", linewidth=1.0, alpha=0.9, label="centerline")
            self.ax.add_line(self.centerline_line)
        self.left_lane_matches = None
        self.right_lane_matches = None
        if self.show_matched_lanes and self.centerline is not None and self.centerline.size > 0:
            self.left_lane_matches = LineCollection(lane_match_segment_list(self.centerline, self.left_points), colors="deepskyblue", linewidths=0.4, alpha=0.35)
            self.right_lane_matches = LineCollection(lane_match_segment_list(self.centerline, self.right_points), colors="salmon", linewidths=0.4, alpha=0.35)
            self.ax.add_collection(self.left_lane_matches)
            self.ax.add_collection(self.right_lane_matches)
        self.ax.add_line(self.left_line)
        self.ax.add_line(self.right_line)
        self.left_scatter = self.ax.scatter(self.left_points[:, 0], self.left_points[:, 1], s=point_size, c="tab:blue", label="left")
        self.right_scatter = self.ax.scatter(self.right_points[:, 0], self.right_points[:, 1], s=point_size, c="tab:red", label="right")
        self.active_side: str | None = None
        self.active_idx: int | None = None
        self.dragging = False
        self.drag_start_point: np.ndarray | None = None
        self.undo_stack: list[tuple[str, int, np.ndarray]] = []
        self.status = self.ax.text(0.01, 0.99, self._status_text(), transform=self.ax.transAxes, va="top", ha="left", color="yellow", bbox={"facecolor": "black", "alpha": 0.6, "pad": 4})
        self.ax.set_title("Drag left/right boundary points. Keys: s=save, l/r=select side, esc=clear selection")
        self.ax.legend(loc="upper right")
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _status_text(self) -> str:
        side = self.active_side or "none"
        idx = -1 if self.active_idx is None else self.active_idx
        return f"selected_side={side} selected_idx={idx} output={self.output_path.name}"

    def _update_artists(self) -> None:
        self.left_line.set_data(self.left_points[:, 0], self.left_points[:, 1])
        self.right_line.set_data(self.right_points[:, 0], self.right_points[:, 1])
        self.left_scatter.set_offsets(self.left_points)
        self.right_scatter.set_offsets(self.right_points)
        if self.left_lane_matches is not None and self.centerline is not None:
            self.left_lane_matches.set_segments(lane_match_segment_list(self.centerline, self.left_points))
        if self.right_lane_matches is not None and self.centerline is not None:
            self.right_lane_matches.set_segments(lane_match_segment_list(self.centerline, self.right_points))
        self.status.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def _nearest_point(self, x: float, y: float) -> tuple[str, int] | tuple[None, None]:
        target = np.asarray([x, y], dtype=np.float32)
        left_dist = np.linalg.norm(self.left_points - target, axis=1)
        right_dist = np.linalg.norm(self.right_points - target, axis=1)
        best_left_idx = int(np.argmin(left_dist))
        best_right_idx = int(np.argmin(right_dist))
        best_left = float(left_dist[best_left_idx])
        best_right = float(right_dist[best_right_idx])
        if best_left <= best_right and best_left <= self.pick_radius:
            return "left", best_left_idx
        if best_right < best_left and best_right <= self.pick_radius:
            return "right", best_right_idx
        return None, None

    def _active_points(self) -> np.ndarray:
        if self.active_side == "left":
            return self.left_points
        if self.active_side == "right":
            return self.right_points
        raise RuntimeError("No active side selected")

    def on_press(self, event: Event) -> None:
        if not isinstance(event, MouseEvent):
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        side, idx = self._nearest_point(float(event.xdata), float(event.ydata))
        if side is None:
            return
        self.active_side = side
        self.active_idx = idx
        self.dragging = True
        self.drag_start_point = self._active_points()[idx].copy()
        self._update_artists()

    def on_release(self, event: Event) -> None:
        if self.dragging and self.active_side is not None and self.active_idx is not None and self.drag_start_point is not None:
            current_point = self._active_points()[self.active_idx]
            if not np.allclose(current_point, self.drag_start_point):
                self.undo_stack.append((self.active_side, self.active_idx, self.drag_start_point.copy()))
        self.dragging = False
        self.drag_start_point = None

    def on_motion(self, event: Event) -> None:
        if not isinstance(event, MouseEvent):
            return
        if not self.dragging or self.active_idx is None or self.active_side is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        points = self._active_points()
        points[self.active_idx, 0] = float(event.xdata)
        points[self.active_idx, 1] = float(event.ydata)
        self._update_artists()

    def on_key(self, event: Event) -> None:
        if not isinstance(event, KeyEvent):
            return
        if event.key == "s":
            save_boundary_points(self.output_path, self.s_values, self.left_points, self.right_points)
            print(f"Saved refined boundary points to {self.output_path}")
            return
        if event.key in {"ctrl+z", "cmd+z"}:
            if self.undo_stack:
                side, idx, point = self.undo_stack.pop()
                points = self.left_points if side == "left" else self.right_points
                points[idx] = point
                self.active_side = side
                self.active_idx = idx
                self.dragging = False
                self.drag_start_point = None
                self._update_artists()
            return
        if event.key == "escape":
            self.active_side = None
            self.active_idx = None
            self.dragging = False
            self.drag_start_point = None
            self._update_artists()

    def show(self) -> None:
        plt.show()


def main() -> None:
    args = parse_args()
    map_cfg = load_map_yaml(args.map_yaml)
    image = load_pgm(args.map_yaml.parent / Path(map_cfg["image"]))
    data = load_boundary_points(args.input)
    centerline = load_centerline(args.centerline) if args.centerline.exists() else None
    editor = WidthProfileEditor(
        map_image=image,
        map_cfg=map_cfg,
        s_values=data["s"],
        left_points=data["left"],
        right_points=data["right"],
        centerline=centerline,
        show_matched_lanes=bool(args.show_matched_lanes),
        output_path=args.output,
        point_size=float(args.point_size),
        pick_radius=float(args.pick_radius),
    )
    editor.show()


if __name__ == "__main__":
    main()
