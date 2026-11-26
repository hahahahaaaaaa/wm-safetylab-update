"""Statistics of clean runs that plugins use as their units and thresholds.

    python -m wms.episodes.clean_stats RUN_DIR [RUN_DIR ...] --output stats.json

A plugin's strength and a detector's threshold should come from what the model and the
environment do when nothing is wrong, not from numbers written into the plugin. This reads
clean episode-kernel runs (no fault, no attack; monitors are fine) and writes one JSON:

  observation_vector  per dimension: mean, std, min, max (vector observations, steps.jsonl obs_seen)
  action              per dimension: mean, std, p99 of |a| and of |a_t - a_t-1|; quantiles of the
                      action's L2 norm (executed actions, steps.jsonl)
  image               per camera: per-channel mean and std of the pixels (0..255), and quantiles of
                      the mean absolute difference between consecutive frames (frames_<cam>.npz)
  latent              per dimension: mean and std of the model latent, from monitor-latent-moments
  events              for each (plugin, field) logged by a monitor with a numeric value (e.g.
                      monitor-surprise's "surprise"): n, mean, std, p50, p90, p99, p99.9, max

Plugins take this file through their config (e.g. {"stats": "stats.json"} or the numbers
copied in); quantiles are what a detector's 1% false-positive threshold is read from.
Needs numpy. Quantiles of steps pool all episodes of all runs given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

QUANTILES = (0.5, 0.9, 0.99, 0.999)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n": 0}
    out = {"n": int(values.size), "mean": float(values.mean()), "std": float(values.std()), "max": float(values.max())}
    out.update({f"p{q * 100:g}": float(np.quantile(values, q)) for q in QUANTILES})
    return out


def _per_dim(rows):
    import numpy as np
    array = np.asarray(rows, dtype=float)
    return {"dims": int(array.shape[1]), "n": int(array.shape[0]), "mean": array.mean(0).tolist(),
            "std": array.std(0).tolist(), "min": array.min(0).tolist(), "max": array.max(0).tolist()}


def collect(run_dirs, max_frames=20000):
    import numpy as np
    vectors, actions, deltas, norms = [], [], [], []
    events, images, latent = {}, {}, {"n": 0, "sum": None, "sumsq": None}
    episodes = 0
    for run_dir in map(Path, run_dirs):
        summary = json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))
        for row in summary["episodes"]:
            episodes += 1
            folder = run_dir / "episodes" / f"ep_{int(row['episode']):03d}"
            previous = None
            for step in _lines(folder / "steps.jsonl"):
                obs = step.get("obs_seen")
                if isinstance(obs, list) and obs and all(isinstance(v, (int, float)) for v in obs):
                    vectors.append(obs)
                action = step.get("action")
                if action is not None:
                    action = [float(v) for v in action]
                    actions.append(action)
                    norms.append(float(np.linalg.norm(action)))
                    if previous is not None and len(previous) == len(action):
                        deltas.append([abs(a - b) for a, b in zip(action, previous)])
                    previous = action
            trace = folder / "trace.jsonl"
            for line in _lines(trace) if trace.is_file() else []:
                for event in line.get("events", []):
                    if event.get("role") != "monitor":
                        continue
                    name = event["plugin"].partition(":")[2]
                    for field, value in (event.get("data") or {}).items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            events.setdefault(f"{name}.{field}", []).append(float(value))
            for report in (row.get("plugin_reports") or {}).values():
                if report.get("latent_n"):
                    s, q = np.asarray(report["latent_sum"]), np.asarray(report["latent_sumsq"])
                    latent["sum"] = s if latent["sum"] is None else latent["sum"] + s
                    latent["sumsq"] = q if latent["sumsq"] is None else latent["sumsq"] + q
                    latent["n"] += int(report["latent_n"])
            for frames_path in sorted(folder.glob("frames_*.npz")):
                camera = frames_path.stem[len("frames_"):]
                store = images.setdefault(camera, {"pixels": [], "diffs": [], "frames": 0})
                if store["frames"] >= max_frames:
                    continue
                frames = np.load(frames_path)["seen"].astype(np.float32)
                store["frames"] += len(frames)
                store["pixels"].append(frames.reshape(len(frames), -1, frames.shape[-1]).mean(1))
                store["pixels_sq"] = store.get("pixels_sq", []) + [(frames ** 2).reshape(len(frames), -1, frames.shape[-1]).mean(1)]
                if len(frames) > 1:
                    store["diffs"] += np.abs(np.diff(frames, axis=0)).reshape(len(frames) - 1, -1).mean(1).tolist()
    out = {"episodes": episodes, "runs": [str(Path(r)) for r in run_dirs]}
    if vectors:
        out["observation_vector"] = _per_dim(vectors)
    if actions:
        array = np.asarray(actions)
        out["action"] = {**_per_dim(actions), "abs_p99": np.quantile(np.abs(array), 0.99, axis=0).tolist(),
                         "norm": _summary(norms)}
        if deltas:
            out["action"]["delta_p99"] = np.quantile(np.asarray(deltas), 0.99, axis=0).tolist()
    for camera, store in images.items():
        mean = np.concatenate(store["pixels"]).mean(0)
        square = np.concatenate(store["pixels_sq"]).mean(0)
        out.setdefault("image", {})[camera] = {"frames": store["frames"], "channel_mean": mean.tolist(),
                                               "channel_std": np.sqrt(np.maximum(square - mean ** 2, 0)).tolist(),
                                               "frame_diff": _summary(store["diffs"])}
    if latent["n"]:
        mean = latent["sum"] / latent["n"]
        std = np.sqrt(np.maximum(latent["sumsq"] / latent["n"] - mean ** 2, 0))
        out["latent"] = {"n": latent["n"], "dims": int(mean.size), "mean": mean.tolist(), "std": std.tolist()}
    if events:
        out["events"] = {key: _summary(values) for key, values in sorted(events.items())}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Statistics of clean runs, for plugin units and thresholds.")
    parser.add_argument("runs", nargs="+", help="run output directories (with episodes.json)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    stats = collect(args.runs)
    Path(args.output).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    brief = {key: (value if not isinstance(value, dict) else sorted(value)) for key, value in stats.items()}
    print(json.dumps(brief, ensure_ascii=False))


if __name__ == "__main__":
    main()
