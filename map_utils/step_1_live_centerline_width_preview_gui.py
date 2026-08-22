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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

MAP_NAME = "map2"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_YAML = Path(f"{ROOT}/data/{MAP_NAME}/{MAP_NAME}.yaml")
DEFAULT_CENTERLINE_OUTPUT = Path(f"{ROOT}/data/{MAP_NAME}/centerline.csv")
DEFAULT_WIDTH_OUTPUT = Path(f"{ROOT}/data/{MAP_NAME}/width_profile.csv")


REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_UTILS_DIR = Path(__file__).resolve().parent
MPPI_DATA_DIR = MAP_UTILS_DIR.parent / "data"
WIDTH_SCRIPT_PATH = MAP_UTILS_DIR / "compute_track_width_profile.py"
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
    package_name = "map_utils_runtime_live_preview"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(MAP_UTILS_DIR)]
        sys.modules[package_name] = package
    _load_module(SPLINE1D_PATH, f"{package_name}.CubicSpline1D")
    spline2d_module = _load_module(SPLINE2D_PATH, f"{package_name}.CubicSpline2D")
    return spline2d_module.CubicSpline2D


WIDTH_MODULE = _load_module(WIDTH_SCRIPT_PATH, "compute_track_width_profile_module_for_live_preview")
CubicSpline2D = _load_cubic_spline_2d()
TrackMap = WIDTH_MODULE.TrackMap
load_map_yaml = WIDTH_MODULE.load_map_yaml
load_pgm = WIDTH_MODULE.load_pgm
build_traversable_mask = WIDTH_MODULE.build_traversable_mask
compute_width_profile = WIDTH_MODULE.compute_width_profile
save_width_profile = WIDTH_MODULE.save_width_profile
save_sample_points = WIDTH_MODULE.save_sample_points
save_mppi_track_csv = WIDTH_MODULE.save_mppi_track_csv

@dataclass
class PathPoint:
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click representative centerline points and preview live lane widths while building a smoothed centerline.")
    parser.add_argument("--map-yaml", type=Path, default=DEFAULT_MAP_YAML, help="ROS map YAML path")
    parser.add_argument(
        "--centerline-input",
        type=Path,
        default=None,
        help="Existing centerline CSV to load (default: centerline-output when it exists)",
    )
    parser.add_argument(
        "--no-load-centerline",
        action="store_true",
        help="Start with an empty click editor even when centerline-output exists",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Load centerline-input, compute and save width outputs, then exit without opening the GUI",
    )
    parser.add_argument("--centerline-output", type=Path, default=DEFAULT_CENTERLINE_OUTPUT, help="Output centerline.csv path")
    parser.add_argument("--width-output", type=Path, default=DEFAULT_WIDTH_OUTPUT, help="Output width_profile.csv path")
    parser.add_argument("--xy-output", type=Path, default=None, help="Optional output width_profile_xy.csv path (default: <width_output_stem>_xy.csv)")
    parser.add_argument("--mppi-output", type=Path, default=None, help="MPPI centerline+lane CSV (default: ../data/<map-name>_centerline.csv)")
    parser.add_argument("--sample-step", type=float, default=0.01, help="Arc-length spacing for sampled spline output and width preview")
    parser.add_argument("--ray-step", type=float, default=0.01, help="Ray marching step for width extraction preview")
    parser.add_argument("--preview-sample-step", type=float, default=0.05, help="Coarser centerline/width spacing used only while clicking")
    parser.add_argument("--preview-ray-step", type=float, default=0.04, help="Coarser boundary ray step used only while clicking")
    parser.add_argument("--max-width", type=float, default=3.0, help="Maximum half-width to search on each side")
    parser.add_argument(
        "--missing-width-fallback",
        choices=("previous_boundary", "nearest_boundary", "max_width"),
        default="max_width",
        help="On a missed ray: reuse previous x,y, find a nearby map boundary, or use max-width",
    )
    parser.add_argument("--fallback-hit-distance", type=float, default=3.0, help="Search radius around the previous point for nearest_boundary fallback")
    parser.add_argument("--occupied-margin", type=float, default=0.0, help="Optional margin subtracted from measured widths")
    parser.add_argument("--tangent-window", type=float, default=1.0, help="Window for estimating local tangent during width extraction")
    parser.add_argument("--point-size", type=float, default=18.0, help="Representative clicked point marker size")
    parser.add_argument("--lane-preview-stride", type=int, default=25, help="Draw one width connector every N preview samples")
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


def load_centerline(csv_path: Path) -> np.ndarray:
    try:
        points = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    except ValueError:
        points = np.loadtxt(csv_path, delimiter=",", dtype=np.float32, skiprows=1)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] < 4:
        raise ValueError(f"Centerline must have at least four x,y rows: {csv_path}")
    points = points[:, :2]
    if not bool(np.isfinite(points).all()):
        raise ValueError(f"Centerline contains non-finite coordinates: {csv_path}")
    if bool((np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-8).any()):
        raise ValueError(f"Centerline contains duplicate consecutive points: {csv_path}")
    return points


def default_xy_output_path(width_output: Path) -> Path:
    return width_output.with_name(f"{width_output.stem}_xy.csv")


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
    coords = coords[np.isfinite(coords).all(axis=1)]
    if coords.shape[0] == 0:
        raise ValueError("Spline sampling produced no finite centerline points")
    return coords.astype(np.float32)


def rows_to_boundary_arrays(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    left = np.asarray([[float(row["left_x"]), float(row["left_y"])] for row in rows], dtype=np.float32)
    right = np.asarray([[float(row["right_x"]), float(row["right_y"])] for row in rows], dtype=np.float32)
    return left, right


def connector_segments(centerline: np.ndarray, boundary: np.ndarray, stride: int) -> list[np.ndarray]:
    if centerline.shape[0] == 0 or boundary.shape[0] == 0:
        return []
    step = max(int(stride), 1)
    end = min(centerline.shape[0], boundary.shape[0])
    return [np.asarray([centerline[idx], boundary[idx]], dtype=np.float32) for idx in range(0, end, step)]


class LiveWidthPreviewEditor:
    def __init__(
        self,
        map_cfg: dict,
        map_image: np.ndarray,
        traversable_mask: np.ndarray,
        centerline_output: Path,
        width_output: Path,
        xy_output: Path,
        mppi_output: Path,
        args: argparse.Namespace,
        centerline_input: Path | None = None,
    ) -> None:
        self.map_cfg = map_cfg
        self.map_image = np.flipud(map_image)
        self.traversable_mask = traversable_mask
        self.centerline_output = centerline_output
        self.width_output = width_output
        self.xy_output = xy_output
        self.mppi_output = mppi_output
        self.sample_step = float(args.sample_step)
        self.ray_step = float(args.ray_step)
        self.preview_sample_step = max(float(args.preview_sample_step), self.sample_step)
        self.preview_ray_step = max(float(args.preview_ray_step), self.ray_step)
        self.max_width = float(args.max_width)
        self.missing_width_fallback = str(args.missing_width_fallback)
        self.fallback_hit_distance = float(args.fallback_hit_distance)
        self.occupied_margin = float(args.occupied_margin)
        self.tangent_window = float(args.tangent_window)
        self.lane_preview_stride = int(args.lane_preview_stride)
        self.centerline_input = centerline_input
        self.loaded_centerline = (
            load_centerline(centerline_input)
            if centerline_input is not None
            else np.empty((0, 2), dtype=np.float32)
        )
        self.clicked_points = np.empty((0, 2), dtype=np.float32)
        self.current_centerline = self.loaded_centerline.copy()
        self.current_rows: list[dict[str, float]] = []
        self.left_boundary = np.empty((0, 2), dtype=np.float32)
        self.right_boundary = np.empty((0, 2), dtype=np.float32)
        self.undo_stack: list[np.ndarray] = []

        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Live Centerline + Width Preview")
        self.ax.imshow(
            self.map_image,
            cmap="gray",
            origin="lower",
            extent=map_extent(self.map_cfg, self.map_image.shape),
            interpolation="nearest",
        )
        self.clicked_scatter = self.ax.scatter([], [], s=float(args.point_size), c="tab:orange", label="clicked points")
        self.polyline = Line2D([], [], color="tab:orange", linewidth=1.0, alpha=0.8)
        self.centerline_line = Line2D([], [], color="tab:green", linewidth=1.5, alpha=0.9, label="live centerline")
        self.left_line = Line2D([], [], color="tab:blue", linewidth=1.0, alpha=0.9, label="left boundary")
        self.right_line = Line2D([], [], color="tab:red", linewidth=1.0, alpha=0.9, label="right boundary")
        self.left_connectors = LineCollection([], colors="deepskyblue", linewidths=0.4, alpha=0.35)
        self.right_connectors = LineCollection([], colors="salmon", linewidths=0.4, alpha=0.35)
        for artist in [self.polyline, self.centerline_line, self.left_line, self.right_line]:
            self.ax.add_line(artist)
        self.ax.add_collection(self.left_connectors)
        self.ax.add_collection(self.right_connectors)
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
        self.ax.set_title(
            "Loaded centerline is preserved | left click: start replacement spline | "
            "l: reload | ctrl+z: undo | s: save centerline+width | c: clear"
        )
        self.ax.legend(loc="upper right")
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        if self.current_centerline.shape[0] > 0:
            self._compute_width_preview(self.current_centerline)
            self._update_artists()

    def _status_text(self) -> str:
        left_misses = sum(not bool(row.get("left_hit", 1.0)) for row in self.current_rows)
        right_misses = sum(not bool(row.get("right_hit", 1.0)) for row in self.current_rows)
        return (
            f"source={'loaded' if self.clicked_points.shape[0] < 4 and self.loaded_centerline.size else 'clicked'} "
            f"clicked={self.clicked_points.shape[0]} centerline_pts={self.current_centerline.shape[0]} "
            f"width_rows={len(self.current_rows)} previous-boundary fallbacks(L/R)={left_misses}/{right_misses} "
            f"centerline_out={self.centerline_output.name} width_out={self.width_output.name}"
        )

    def _compute_width_preview(self, centerline: np.ndarray, *, precise: bool = False) -> None:
        self.current_centerline = np.asarray(centerline, dtype=np.float32).copy()
        track_map = TrackMap(centerline=self.current_centerline, track_width=1.0, name="live_preview")
        self.current_rows = compute_width_profile(
            track_map=track_map,
            traversable_mask=self.traversable_mask,
            origin=np.asarray(self.map_cfg["origin"][:2], dtype=np.float32),
            resolution=float(self.map_cfg["resolution"]),
            s_step=self.sample_step if precise else self.preview_sample_step,
            ray_step=self.ray_step if precise else self.preview_ray_step,
            max_width=self.max_width,
            fallback_hit_distance=self.fallback_hit_distance,
            occupied_margin=self.occupied_margin,
            tangent_window=self.tangent_window,
            missing_width_fallback=self.missing_width_fallback,
        )
        self.left_boundary, self.right_boundary = rows_to_boundary_arrays(self.current_rows)

    def _rebuild_preview(self) -> None:
        if self.clicked_points.shape[0] < 4:
            if self.loaded_centerline.size:
                # The loaded-track preview was already computed at startup.
                # Do not repeat the full width scan for clicks 1--3.
                if not np.array_equal(self.current_centerline, self.loaded_centerline):
                    self._compute_width_preview(self.loaded_centerline)
            else:
                self.current_centerline = np.empty((0, 2), dtype=np.float32)
                self.current_rows = []
                self.left_boundary = np.empty((0, 2), dtype=np.float32)
                self.right_boundary = np.empty((0, 2), dtype=np.float32)
            return
        try:
            self._compute_width_preview(
                spline_centerline(self.clicked_points, sample_step=self.preview_sample_step)
            )
        except Exception as exc:
            print(f"Could not rebuild preview: {exc}")
            self.current_centerline = np.empty((0, 2), dtype=np.float32)
            self.current_rows = []
            self.left_boundary = np.empty((0, 2), dtype=np.float32)
            self.right_boundary = np.empty((0, 2), dtype=np.float32)

    def _update_artists(self) -> None:
        self.clicked_scatter.set_offsets(self.clicked_points if self.clicked_points.size else np.empty((0, 2), dtype=np.float32))
        self.polyline.set_data(self.clicked_points[:, 0], self.clicked_points[:, 1])
        self.centerline_line.set_data(self.current_centerline[:, 0], self.current_centerline[:, 1])
        self.left_line.set_data(self.left_boundary[:, 0], self.left_boundary[:, 1])
        self.right_line.set_data(self.right_boundary[:, 0], self.right_boundary[:, 1])
        self.left_connectors.set_segments(connector_segments(self.current_centerline, self.left_boundary, self.lane_preview_stride))
        self.right_connectors.set_segments(connector_segments(self.current_centerline, self.right_boundary, self.lane_preview_stride))
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
        self.clicked_points = np.vstack([self.clicked_points, np.asarray([[x_value, y_value]], dtype=np.float32)])
        self._rebuild_preview()
        self._update_artists()

    def on_key(self, event: Event) -> None:
        if not isinstance(event, KeyEvent):
            return
        if event.key in {"ctrl+z", "cmd+z"}:
            if self.undo_stack:
                self.clicked_points = self.undo_stack.pop()
                self._rebuild_preview()
                self._update_artists()
            return
        if event.key == "c":
            if self.clicked_points.shape[0] > 0:
                self.undo_stack.append(self.clicked_points.copy())
            self.clicked_points = np.empty((0, 2), dtype=np.float32)
            self.loaded_centerline = np.empty((0, 2), dtype=np.float32)
            self._rebuild_preview()
            self._update_artists()
            return
        if event.key == "l":
            if self.centerline_input is None:
                print("No centerline input was configured")
                return
            self.loaded_centerline = load_centerline(self.centerline_input)
            self.clicked_points = np.empty((0, 2), dtype=np.float32)
            self._rebuild_preview()
            self._update_artists()
            print(f"Reloaded {self.loaded_centerline.shape[0]} points from {self.centerline_input}")
            return
        if event.key == "s":
            if self.current_centerline.shape[0] == 0 or not self.current_rows:
                print("Need at least 4 representative points before saving live preview outputs")
                return
            preserving_loaded_centerline = (
                self.loaded_centerline.size > 0 and self.clicked_points.shape[0] < 4
            )
            # Clicking uses a coarse preview for responsiveness. Rebuild once
            # at the requested output resolution immediately before saving.
            precise_centerline = (
                self.loaded_centerline
                if preserving_loaded_centerline
                else spline_centerline(self.clicked_points, sample_step=self.sample_step)
            )
            self._compute_width_preview(precise_centerline, precise=True)
            if not preserving_loaded_centerline:
                save_centerline(self.centerline_output, self.current_centerline)
            save_width_profile(self.width_output, self.current_rows)
            save_sample_points(self.xy_output, self.current_rows)
            save_mppi_track_csv(self.mppi_output, self.current_rows)
            if preserving_loaded_centerline:
                print(f"Preserved loaded centerline without rewriting {self.centerline_output}")
            else:
                print(f"Saved {self.current_centerline.shape[0]} centerline points to {self.centerline_output}")
            print(f"Saved {len(self.current_rows)} width rows to {self.width_output}")
            print(f"Saved {len(self.current_rows)} boundary rows to {self.xy_output}")
            print(f"Saved {len(self.current_rows)} MPPI track rows to {self.mppi_output}")

    def show(self) -> None:
        plt.show()


def main() -> None:
    args = parse_args()
    map_cfg = load_map_yaml(args.map_yaml)
    image = load_pgm(args.map_yaml.parent / Path(map_cfg["image"]))
    traversable_mask = build_traversable_mask(
        image=image,
        negate=int(map_cfg["negate"]),
        occupied_thresh=float(map_cfg["occupied_thresh"]),
        free_thresh=float(map_cfg["free_thresh"]),
    )
    xy_output = default_xy_output_path(args.width_output) if args.xy_output is None else args.xy_output
    mppi_output = (
        MPPI_DATA_DIR / f"{args.map_yaml.stem}_centerline.csv"
        if args.mppi_output is None else args.mppi_output
    )
    centerline_input = None
    if not args.no_load_centerline:
        candidate = args.centerline_output if args.centerline_input is None else args.centerline_input
        if candidate.is_file():
            centerline_input = candidate
            print(f"Loading existing centerline from {centerline_input}")
    if args.batch:
        if args.no_load_centerline:
            raise ValueError("--batch cannot be combined with --no-load-centerline")
        if centerline_input is None:
            requested = args.centerline_output if args.centerline_input is None else args.centerline_input
            raise FileNotFoundError(f"--batch requires an existing centerline CSV: {requested}")
        centerline = load_centerline(centerline_input)
        track_map = TrackMap(centerline=centerline, track_width=1.0, name=args.map_yaml.stem)
        rows = compute_width_profile(
            track_map=track_map,
            traversable_mask=traversable_mask,
            origin=np.asarray(map_cfg["origin"][:2], dtype=np.float32),
            resolution=float(map_cfg["resolution"]),
            s_step=float(args.sample_step),
            ray_step=float(args.ray_step),
            max_width=float(args.max_width),
            fallback_hit_distance=float(args.fallback_hit_distance),
            occupied_margin=float(args.occupied_margin),
            tangent_window=float(args.tangent_window),
            missing_width_fallback=str(args.missing_width_fallback),
        )
        save_width_profile(args.width_output, rows)
        save_sample_points(xy_output, rows)
        save_mppi_track_csv(mppi_output, rows)
        print(f"Preserved loaded centerline without rewriting {centerline_input}")
        print(f"Saved {len(rows)} width rows to {args.width_output}")
        print(f"Saved {len(rows)} boundary rows to {xy_output}")
        print(f"Saved {len(rows)} MPPI track rows to {mppi_output}")
        return
    editor = LiveWidthPreviewEditor(
        map_cfg=map_cfg,
        map_image=image,
        traversable_mask=traversable_mask,
        centerline_output=args.centerline_output,
        width_output=args.width_output,
        xy_output=xy_output,
        mppi_output=mppi_output,
        args=args,
        centerline_input=centerline_input,
    )
    editor.show()


if __name__ == "__main__":
    main()
