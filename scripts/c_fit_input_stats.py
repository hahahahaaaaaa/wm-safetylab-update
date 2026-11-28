#!/usr/bin/env python3
"""Fit clean S1 image/vector input statistics from raw steps.jsonl observations.

Only completed, explicitly clean runs are accepted. The script refuses to fit
statistics from a run with plugins, scheduled perturbations, target moves, or
S1 fault/attack events in its trace. It never synthesizes observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def episode_dirs(run_dir):
    run_dir = Path(run_dir)
    for root in (run_dir / "output" / "episodes", run_dir / "episodes", run_dir):
        if root.is_dir():
            found = sorted(path for path in root.glob("ep_*") if path.is_dir())
            if found:
                return found
    return []


def obs_of(row):
    for key in ("obs", "observation", "raw_obs", "observation_raw", "input_obs"):
        if row.get(key) is not None:
            return row[key]
    if any(key in row for key in ("rgb", "vector", "observation_vector")):
        return row
    transition = row.get("transition")
    if isinstance(transition, dict):
        return obs_of(transition)
    return None


def _walk_events(value):
    if isinstance(value, dict):
        if {"plugin", "data", "t"}.issubset(value):
            yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _walk_events(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                yield from _walk_events(child)


def _assert_clean_run(run_dir):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    result_path = run_dir / "output" / "result.json"
    if not config_path.is_file():
        raise ValueError(f"{run_dir}: missing config.json; cannot verify clean-run provenance")
    if not result_path.is_file():
        raise ValueError(f"{run_dir}: missing output/result.json; cannot verify completed run")

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"{run_dir}: run status is {result.get('status')!r}, not 'completed'")

    params = config.get("params") or {}
    plugins = params.get("plugins", config.get("plugins")) or []
    perturbations = params.get("perturbations", config.get("perturbations")) or []
    target_moves = params.get("target_moves", config.get("target_moves")) or []
    if plugins:
        raise ValueError(f"{run_dir}: params.plugins is non-empty; fit only from a plugin-free clean baseline")
    if perturbations:
        raise ValueError(f"{run_dir}: params.perturbations is non-empty; this is not a clean baseline")
    if target_moves:
        raise ValueError(f"{run_dir}: params.target_moves is non-empty; this is not a clean baseline")

    for episode_dir in episode_dirs(run_dir):
        trace_path = episode_dir / "trace.jsonl"
        if not trace_path.is_file():
            continue
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for event in _walk_events(row):
                    role = event.get("role")
                    stage = event.get("stage")
                    data = event.get("data") or {}
                    if (stage == "S1" and role in {"fault", "attack"}) or data.get("injected") is True:
                        raise ValueError(
                            f"{trace_path}:{line_number}: detected an S1 fault/attack event; refusing dirty data"
                        )

    return {
        "path": str(run_dir.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "seed": config.get("seed"),
        "seeds": params.get("seeds"),
        "episodes_requested": params.get("episodes"),
        "model": config.get("model"),
        "environment": config.get("environment"),
    }


def _image_features(image, previous):
    difference = 0.0 if previous is None else float(np.mean(np.abs(image - previous)))
    return np.concatenate((
        [difference],
        image.mean(axis=(0, 1)),
        image.std(axis=(0, 1)),
    ))


def _read_runs(run_dirs, camera, kind):
    image_rows, vector_rows = [], []
    provenance = []
    for run_dir in run_dirs:
        metadata = _assert_clean_run(run_dir)
        provenance.append(metadata)
        episodes = episode_dirs(run_dir)
        if not episodes:
            raise SystemExit(f"No ep_* directories found under clean run: {run_dir}")

        for episode_dir in episodes:
            steps_path = episode_dir / "steps.jsonl"
            if not steps_path.is_file():
                raise SystemExit(f"Missing raw steps file for episode: {steps_path}")
            previous = None
            step_count = 0
            with steps_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    obs = obs_of(row)
                    if obs is None:
                        raise ValueError(f"{steps_path}:{line_number}: no raw observation field found")
                    step_count += 1

                    if isinstance(obs, dict):
                        rgb = obs.get("rgb")
                        if kind in ("both", "image"):
                            if not isinstance(rgb, dict) or not rgb:
                                raise ValueError(f"{steps_path}:{line_number}: RGB observation missing")
                            if camera not in rgb:
                                raise ValueError(
                                    f"{steps_path}:{line_number}: camera {camera!r} not found; available={list(rgb)}"
                                )
                            raw_image = np.asarray(rgb[camera])
                            if (raw_image.ndim != 3 or raw_image.shape[-1] != 3
                                    or not np.issubdtype(raw_image.dtype, np.number)
                                    or not np.isfinite(raw_image).all()):
                                raise ValueError(f"{steps_path}:{line_number}: invalid RGB image array")
                            if np.any(raw_image < 0) or np.any(raw_image > 255):
                                raise ValueError(f"{steps_path}:{line_number}: RGB values outside [0, 255]")
                            if not np.array_equal(raw_image, np.rint(raw_image)):
                                raise ValueError(f"{steps_path}:{line_number}: RGB values are not integer-valued")
                            image = raw_image.astype(np.float64, copy=False)
                            if previous is not None and previous.shape != image.shape:
                                raise ValueError(f"{steps_path}:{line_number}: RGB image shape changed within episode")
                            image_rows.append(_image_features(image, previous))
                            previous = image.copy()

                        if kind in ("both", "vector"):
                            vector = obs.get("observation_vector", obs.get("vector"))
                            if vector is not None:
                                values = np.asarray(vector, dtype=np.float64).reshape(-1)
                                if not values.size or not np.isfinite(values).all():
                                    raise ValueError(f"{steps_path}:{line_number}: invalid observation vector")
                                vector_rows.append(values)
                            elif kind == "vector":
                                raise ValueError(f"{steps_path}:{line_number}: observation vector missing")
                    elif kind in ("both", "vector") and isinstance(obs, (list, tuple)):
                        values = np.asarray(obs, dtype=np.float64).reshape(-1)
                        if not values.size or not np.isfinite(values).all():
                            raise ValueError(f"{steps_path}:{line_number}: invalid observation vector")
                        vector_rows.append(values)
                    elif kind == "image":
                        raise ValueError(f"{steps_path}:{line_number}: expected dict RGB observation")
            if step_count == 0:
                raise ValueError(f"{steps_path}: no observations found")

    return image_rows, vector_rows, provenance


def _build_payloads(run_dirs, camera, kind, image_rows, vector_rows, provenance):
    payloads = {}
    if kind in ("both", "image"):
        if len(image_rows) < 2:
            raise SystemExit(
                "Fewer than 2 raw RGB observations found. Inspect steps.jsonl; do not fit from summary-only data."
            )
        matrix = np.asarray(image_rows, dtype=np.float64)
        covariance = np.atleast_2d(np.cov(matrix, rowvar=False))
        if not np.isfinite(matrix).all() or not np.isfinite(covariance).all():
            raise SystemExit("Non-finite image feature statistics; refusing to write output")
        payloads["a-input-image.json"] = {
            "schema": "s1-input-features-v1",
            "runs": [str(path) for path in run_dirs],
            "provenance": provenance,
            "image": {
                camera: {
                    "feature_names": [
                        "frame_diff", "mean_r", "mean_g", "mean_b",
                        "std_r", "std_g", "std_b",
                    ],
                    "feature_mean": matrix.mean(axis=0).tolist(),
                    "feature_covariance": covariance.tolist(),
                    "feature_std": matrix.std(axis=0).tolist(),
                    "n": int(matrix.shape[0]),
                    "pixel_units": "raw 0..255",
                }
            },
        }
    if kind in ("both", "vector"):
        if len(vector_rows) < 2:
            raise SystemExit(
                "Fewer than 2 raw vector observations found. Inspect steps.jsonl; do not fit from summary-only data."
            )
        dimensions = {row.size for row in vector_rows}
        if len(dimensions) != 1:
            raise SystemExit(f"Vector dimensions differ across clean observations: {sorted(dimensions)}")
        matrix = np.asarray(vector_rows, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise SystemExit("Non-finite vector statistics; refusing to write output")
        payloads["a-input-vector.json"] = {
            "schema": "s1-input-features-v1",
            "runs": [str(path) for path in run_dirs],
            "provenance": provenance,
            "observation_vector": {
                "mean": matrix.mean(axis=0).tolist(),
                "std": np.maximum(matrix.std(axis=0), 1e-8).tolist(),
                "n": int(matrix.shape[0]),
            },
        }
    return payloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="completed clean baseline run directories only")
    parser.add_argument("--camera", default="3rd_view_camera")
    parser.add_argument("--output-dir", default="native/plugins/stats")
    parser.add_argument("--kind", choices=("both", "image", "vector"), default="both")
    parser.add_argument("--overwrite", action="store_true", help="replace existing a-input statistics files")
    args = parser.parse_args()

    run_dirs = [Path(path).expanduser().resolve() for path in args.runs]
    if len(run_dirs) != len(set(run_dirs)):
        raise SystemExit("Duplicate run directory supplied; provide each clean run only once")
    image_rows, vector_rows, provenance = _read_runs(run_dirs, args.camera, args.kind)
    payloads = _build_payloads(run_dirs, args.camera, args.kind, image_rows, vector_rows, provenance)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        target = output_dir / filename
        if target.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing statistics file: {target} (pass --overwrite explicitly)")
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    print("Wrote clean input statistics:")
    for filename, payload in payloads.items():
        node = payload.get("image", {}).get(args.camera) or payload.get("observation_vector")
        print(f"  {output_dir / filename} (n={node['n']})")


if __name__ == "__main__":
    main()
