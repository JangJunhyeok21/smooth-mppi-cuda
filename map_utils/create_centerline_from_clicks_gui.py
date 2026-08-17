from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import Event, KeyEvent, MouseEvent
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_UTILS_DIR = Path(__file__).resolve().parent
WIDTH_SCRIPT_PATH = MAP_UTILS_DIR / "step_2_compute_track_width_profile.py"
SPLINE1D_PATH = MAP_UTILS_DIR / "CubicSpline1D.py"
SPLINE2D_PATH = MAP_UTILS_DIR / "CubicSpline2D.py"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_cubic_spline_2d():
    package_name = "map_utils_runtime"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(MAP_UTILS_DIR)]
        sys.modules[package_name] = package
    _load_module(SPLINE1D_PATH, f"{package_name}.CubicSpline1D")
    spline2d_module = _load_module(SPLINE2D_PATH, f"{package_name}.CubicSpline2D")
    return spline2d_module.CubicSpline2D


WIDTH_MODULE = _load_module(WIDTH_SCRIPT_PATH, "step_2_step_2_compute_track_width_profile_module_for_click_gui")
CubicSpline2D = _load_cubic_spline_2d()
load_map_yaml = WIDTH_MODULE.load_map_yaml
load_pgm = WIDTH_MODULE.load_pgm


DEFAULT_MAP_YAML = Path("/home/a/RL-RACER/simulators/maps/map1.yaml")
DEFAULT_OUTPUT = Path("/home/a/RL-RACER/simulators/map_paths/map1/centerline.csv")


@dataclass
class PathPoint:
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click representative centerline points and save a smoothed centerline CSV using CubicSpline2D.")
    parser.add_argument("--map-yaml", type=Path, default=DEFAULT_MAP_YAML, help="ROS map YAML path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output centerline.csv path")
    parser.add_argument("--sample-step", type=float, default=0.01, help="Arc-length spacing for sampled spline output")
    parser.add_argument("--point-size", type=float, default=18.0, help="Representative clicked point marker size")
    return parser.parse_args()


def map_extent(map_cfg: dict, image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    resolution = float(map_cfg["resolution"])
    origin = np.asarray(map_cfg["origin"][:2], dtype=np.float32)
    height, width = image_shape
    x0 = float(origin[0])
    x1 = x0 + float(width) * resolution
    y0 = float(origin[1])
    y1 = y0 + float(height) * resolution
    return (x0, x1, y0, y1)


def save_centerline(csv_path: Path, points: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        for point in np.asarray(points, dtype=np.float32):
            writer.writerow([f"{float(point[0]):.8f}", f"{float(point[1]):.8f}"])


def build_closed_spline_points(clicked_points: np.ndarray) -> np.ndarray:
    pts = np.asarray(clicked_points, dtype=np.float32)
    if pts.shape[0] < 4:
        raise ValueError("At least 4 representative points are required for closed-loop smoothing")
    return np.vstack([pts[-2:], pts, pts[:3]]).astype(np.float32)


def spline_centerline(clicked_points: np.ndarray, sample_step: float) -> np.ndarray:
    if sample_step <= 0.0:
        raise ValueError(f"sample_step must be positive, got {sample_step}")
    closed_points = build_closed_spline_points(clicked_points)
    spline_input = [PathPoint(float(x), float(y)) for x, y in closed_points]
    spline = CubicSpline2D(spline_input)
    point_count = clicked_points.shape[0]
    start_s = float(spline.s[2])
    end_s = float(spline.s[2 + point_count])
    sample_s = np.arange(start_s, end_s, float(sample_step), dtype=np.float64)
    if sample_s.size == 0:
        sample_s = np.linspace(start_s, max(start_s + sample_step, end_s), num=max(point_count * 10, 2), endpoint=False, dtype=np.float64)
    coords = np.asarray([[spline.calc_x(float(s)), spline.calc_y(float(s))] for s in sample_s], dtype=np.float32)
    finite_mask = np.isfinite(coords).all(axis=1)
    coords = coords[finite_mask]
    if coords.shape[0] == 0:
        raise ValueError("Spline sampling produced no finite centerline points")
    return coords.astype(np.float32)


class ClickCenterlineEditor:
    def __init__(self, map_image: np.ndarray, map_cfg: dict, output_path: Path, sample_step: float, point_size: float) -> None:
        self.map_image = np.flipud(map_image)
        self.map_cfg = map_cfg
        self.output_path = output_path
        self.sample_step = float(sample_step)
        self.clicked_points = np.empty((0, 2), dtype=np.float32)
        self.current_centerline = np.empty((0, 2), dtype=np.float32)
        self.undo_stack: list[np.ndarray] = []

        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Centerline Click Builder")
        self.ax.imshow(
            self.map_image,
            cmap="gray",
            origin="lower",
            extent=map_extent(self.map_cfg, self.map_image.shape),
            interpolation="nearest",
        )
        self.clicked_scatter = self.ax.scatter([], [], s=point_size, c="tab:orange", label="clicked points")
        self.polyline = Line2D([], [], color="tab:orange", linewidth=1.0, alpha=0.8)
        self.centerline_line = Line2D([], [], color="tab:green", linewidth=1.5, alpha=0.9, label="spline centerline")
        self.ax.add_line(self.polyline)
        self.ax.add_line(self.centerline_line)
        self.status = self.ax.text(
            0.01,
            0.99,
            self._status_text(),
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            color="yellow",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 4},
        )
        self.ax.set_title("Left click: add point | shift+click: snap x/y to previous point | ctrl+z: undo | s: save | c: clear")
        self.ax.legend(loc="upper right")
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _status_text(self) -> str:
        return f"clicked_points={self.clicked_points.shape[0]} sampled_centerline={self.current_centerline.shape[0]} output={self.output_path.name}"

    def _rebuild_spline_preview(self) -> None:
        if self.clicked_points.shape[0] < 4:
            self.current_centerline = np.empty((0, 2), dtype=np.float32)
            return
        try:
            self.current_centerline = spline_centerline(self.clicked_points, sample_step=self.sample_step)
        except Exception:
            self.current_centerline = np.empty((0, 2), dtype=np.float32)

    def _update_artists(self) -> None:
        self.clicked_scatter.set_offsets(self.clicked_points if self.clicked_points.size else np.empty((0, 2), dtype=np.float32))
        self.polyline.set_data(self.clicked_points[:, 0], self.clicked_points[:, 1])
        self.centerline_line.set_data(self.current_centerline[:, 0], self.current_centerline[:, 1])
        self.status.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def on_press(self, event: Event) -> None:
        if not isinstance(event, MouseEvent):
            return
        if event.button != 1:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.undo_stack.append(self.clicked_points.copy())
        x_value = float(event.xdata)
        y_value = float(event.ydata)
        if event.key == "shift" and self.clicked_points.shape[0] > 0:
            prev_x, prev_y = self.clicked_points[-1]
            if abs(x_value - float(prev_x)) <= abs(y_value - float(prev_y)):
                x_value = float(prev_x)
            else:
                y_value = float(prev_y)
        new_point = np.asarray([[x_value, y_value]], dtype=np.float32)
        self.clicked_points = np.vstack([self.clicked_points, new_point])
        self._rebuild_spline_preview()
        self._update_artists()

    def on_key(self, event: Event) -> None:
        if not isinstance(event, KeyEvent):
            return
        if event.key in {"ctrl+z", "cmd+z"}:
            if self.undo_stack:
                self.clicked_points = self.undo_stack.pop()
                self._rebuild_spline_preview()
                self._update_artists()
            return
        if event.key == "c":
            if self.clicked_points.shape[0] > 0:
                self.undo_stack.append(self.clicked_points.copy())
            self.clicked_points = np.empty((0, 2), dtype=np.float32)
            self.current_centerline = np.empty((0, 2), dtype=np.float32)
            self._update_artists()
            return
        if event.key == "s":
            if self.current_centerline.shape[0] == 0:
                print("Need at least 4 representative points before saving centerline")
                return
            save_centerline(self.output_path, self.current_centerline)
            print(f"Saved {self.current_centerline.shape[0]} centerline points to {self.output_path}")

    def show(self) -> None:
        plt.show()


def main() -> None:
    args = parse_args()
    map_cfg = load_map_yaml(args.map_yaml)
    image = load_pgm(args.map_yaml.parent / Path(map_cfg["image"]))
    editor = ClickCenterlineEditor(
        map_image=image,
        map_cfg=map_cfg,
        output_path=args.output,
        sample_step=float(args.sample_step),
        point_size=float(args.point_size),
    )
    editor.show()


if __name__ == "__main__":
    main()
