#!/usr/bin/env python3
"""Step 7: clone manually trimmed Step-1 archives and rebuild current-KF GT.

No rosbag is read and no trim decision is recomputed.  Every selected NPZ and
its JSON metadata are copied first; only the KF/Pacejka-derived columns in the
copy are refreshed from config/params.yaml.  Step 6 then interpolates these
refreshed ``samples`` columns at callback timestamps to construct recursive GT.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from helper_filter_collision_recovery import (
    collision_recovery_mask, physical_inconsistency_mask)
from step_1_extract_data import collision_review_mask, refresh_saved_kf_from_yaml


ROOT = Path(__file__).resolve().parents[2]

# User-configurable defaults.  The source is never modified.
SOURCE_DATA_PATH = ROOT / "model_tuning/data/ifac2026"
OUTPUT_DATA_PATH = ROOT / "model_tuning/data/ifac2026_collision_refined_current_kf_gt"
OVERWRITE_OUTPUT = False
REFINE_COLLISIONS_AND_SPLIT = True
# A retained piece must support actuator warm-up plus a full 1.6 s rollout.
MINIMUM_SEGMENT_DURATION_S = 2.5


def archive_signature(path: Path) -> dict:
    """Return invariants proving that a manual trim was preserved."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"samples", "columns", "alignment_start_epoch_s"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"{path}: not a Step-1 archive; missing {missing}")
        samples = np.asarray(archive["samples"])
        columns = {str(name): i for i, name in enumerate(archive["columns"])}
        if "t" not in columns or not len(samples):
            raise RuntimeError(f"{path}: empty archive or missing t column")
        return {
            "rows": int(len(samples)),
            "first_t_s": float(samples[0, columns["t"]]),
            "last_t_s": float(samples[-1, columns["t"]]),
            "alignment_start_epoch_s": float(archive["alignment_start_epoch_s"]),
            "callback_anchors": int(len(archive["callback_inputs"]))
                if "callback_inputs" in archive.files else 0,
        }


def _continuous_clean_runs(samples: np.ndarray, columns: np.ndarray, dt: float):
    """Return collision-free continuous index runs and filter diagnostics."""
    names = {str(name): i for i, name in enumerate(columns)}
    base = samples[:, [names[name] for name in
        ("t", "x", "y", "yaw", "vx", "vy", "omega", "steer", "accel", "speed_cmd")]]
    physical, physical_events = physical_inconsistency_mask(base, dt)
    recovery, recovery_events = collision_recovery_mask(base, dt)
    review, _ = collision_review_mask(samples, columns)
    bad = physical | recovery | review
    kept = np.flatnonzero(~bad)
    if not len(kept):
        return [], bad, physical_events, recovery_events
    bag_id = (samples[:, names["bag_id"]].astype(int)
              if "bag_id" in names else np.zeros(len(samples), int))
    breaks = np.flatnonzero(
        (np.diff(kept) > 1) |
        (np.diff(bag_id[kept]) != 0)) + 1
    minimum = max(10, int(np.ceil(MINIMUM_SEGMENT_DURATION_S / dt)))
    runs = [run for run in np.split(kept, breaks) if len(run) >= minimum]
    return runs, bad, physical_events, recovery_events


def _write_segment(source: Path, output: Path, payload: dict,
                   run: np.ndarray, segment_index: int) -> tuple[Path, dict]:
    columns = np.asarray(payload["columns"])
    names = {str(name): i for i, name in enumerate(columns)}
    original = np.asarray(payload["samples"])
    start_t = float(original[run[0], names["t"]])
    end_t = float(original[run[-1], names["t"]])
    segment = original[run].copy()
    segment[:, names["t"]] -= start_t
    if "bag_id" in names:
        segment[:, names["bag_id"]] = 0
    result = {key: np.asarray(value) for key, value in payload.items()}
    result["samples"] = segment
    result["alignment_start_epoch_s"] = np.asarray(
        float(payload["alignment_start_epoch_s"]) + start_t, np.float64)
    callbacks_kept = 0
    if "callback_inputs" in result:
        callbacks = np.asarray(result["callback_inputs"]).copy()
        callback_names = {str(name): i for i, name in
                          enumerate(result["callback_input_columns"])}
        callback_t = callbacks[:, callback_names["t"]]
        future_horizon = (float(np.max(result["callback_future_offsets_s"]))
                          if len(result["callback_future_offsets_s"]) else 0.)
        keep = ((callback_t >= start_t - 1e-9) &
                (callback_t + future_horizon <= end_t + 1e-9))
        callbacks = callbacks[keep]
        callbacks[:, callback_names["t"]] -= start_t
        if "bag_id" in callback_names:
            callbacks[:, callback_names["bag_id"]] = 0
        result["callback_inputs"] = callbacks
        for field in ("callback_future_states", "callback_future_commands"):
            if field in result:
                result[field] = np.asarray(result[field])[keep]
        callbacks_kept = int(keep.sum())
    destination = output / f"{source.stem}__segment_{segment_index:02d}.npz"
    np.savez_compressed(destination, **result)
    metadata_source = source.with_suffix(".json")
    metadata = (json.loads(metadata_source.read_text())
                if metadata_source.is_file() else {})
    metadata.update({
        "source_trimmed_npz": str(source.resolve()),
        "collision_refined_segment": segment_index,
        "segment_source_start_s": start_t,
        "segment_source_end_s": end_t,
        "output_samples": int(len(segment)),
        "callback_anchors": callbacks_kept,
        "discontinuity_policy": "each retained continuous interval is a separate NPZ",
    })
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination, {"source_start_s": start_t, "source_end_s": end_t,
                         "rows": int(len(segment)), "callback_anchors": callbacks_kept}


def build(source: Path, output: Path, overwrite: bool = False) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        raise RuntimeError("SOURCE_DATA_PATH and OUTPUT_DATA_PATH must differ")
    paths = sorted(source.glob("*.npz"))
    if not paths:
        raise RuntimeError(f"no trimmed Step-1 NPZ files found in {source}")
    output.mkdir(parents=True, exist_ok=True)
    existing = sorted(output.glob("*.npz"))
    if existing and not overwrite:
        raise RuntimeError(
            f"{output} already contains {len(existing)} NPZ files; set "
            "OVERWRITE_OUTPUT=True or pass --overwrite")

    before = {}
    refinement_reports = []
    copied_names = set()
    for number, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
        if REFINE_COLLISIONS_AND_SPLIT:
            runs, bad, physical_events, recovery_events = _continuous_clean_runs(
                np.asarray(payload["samples"]), np.asarray(payload["columns"]),
                float(payload["dt"]))
            for segment_index, run in enumerate(runs):
                destination, segment_info = _write_segment(
                    path, output, payload, run, segment_index)
                before[destination.name] = archive_signature(destination)
                copied_names.add(destination.name)
            retained_rows = int(sum(len(run) for run in runs))
            refinement_reports.append({
                "source_file": path.name,
                "input_rows": int(len(payload["samples"])),
                "suspected_collision_rows_removed": int(bad.sum()),
                "short_clean_fragment_rows_discarded": int(
                    len(payload["samples"]) - bad.sum() - retained_rows),
                "physical_events": physical_events,
                "reverse_recovery_events": recovery_events,
                "output_segments": len(runs),
                "retained_rows": retained_rows,
            })
            print(f"[{number}/{len(paths)}] refined {path.name}: removed={int(bad.sum())}, "
                  f"physical_events={len(physical_events)}, recovery_events="
                  f"{len(recovery_events)}, retained_segments={len(runs)}")
        else:
            signature = archive_signature(path)
            destination = output / path.name
            shutil.copy2(path, destination)
            metadata = path.with_suffix(".json")
            if metadata.is_file():
                shutil.copy2(metadata, output / metadata.name)
            before[path.name] = signature
            copied_names.add(path.name)
            print(f"[{number}/{len(paths)}] cloned retained archive: {path.name}; "
                  f"rows={signature['rows']}, t={signature['first_t_s']:.3f}.."
                  f"{signature['last_t_s']:.3f} s")

    # When overwrite is requested, remove stale output archives that are not
    # present in the selected source set so Step 6 cannot train on old bags.
    if overwrite:
        for stale in output.glob("*.npz"):
            if stale.name not in copied_names:
                stale.unlink()
                stale.with_suffix(".json").unlink(missing_ok=True)

    refresh_saved_kf_from_yaml(output)

    records = []
    for name, expected in before.items():
        destination = output / name
        actual = archive_signature(destination)
        if actual != expected:
            raise RuntimeError(
                f"{name}: trim/callback invariant changed: {expected} -> {actual}")
        with np.load(destination, allow_pickle=False) as archive:
            parameter_hash = str(np.asarray(archive["kf_parameter_hash"]).item())
        records.append({"file": name, **actual, "kf_parameter_hash": parameter_hash})

    manifest = {
        "purpose": "current params.yaml causal-KF GT rebuilt from manually trimmed Step-1 NPZs",
        "source_directory": str(source),
        "output_directory": str(output),
        "source_npz_count": len(paths),
        "trim_policy": ("original manual trim retained; suspected collision intervals removed; "
                        "each remaining continuous interval stored separately"),
        "step6_gt_contract": (
            "Step 6 interpolates refreshed samples[kf_x,kf_y,kf_yaw,kf_vx,kf_vy,kf_yaw_rate] "
            "at callback timestamps"),
        "archives": records,
        "collision_refinement": refinement_reports,
    }
    manifest_path = output / "step_7_current_kf_gt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Step 7 complete: {len(records)} trimmed archives -> {output}")
    print(f"manifest: {manifest_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DATA_PATH)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_OUTPUT)
    args = parser.parse_args()
    build(args.source, args.output, args.overwrite)


if __name__ == "__main__":
    main()
