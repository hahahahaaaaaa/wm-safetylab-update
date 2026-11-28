#!/usr/bin/env python3
"""Sparse-RS-style black-box search for an S1 fixed image patch.

Each candidate is evaluated only through the run's episode summaries. No
simulator truth or model internals are read. The 40-query budget applies to
training candidates; the final best-vs-random comparison uses the disjoint
held-out seeds specified by the task card.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
import uuid
from pathlib import Path


def parse_seeds(value):
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("seed list must not be empty")
    if len(result) != len(set(result)):
        raise ValueError("seed list contains duplicates")
    return result


def _nested_get(row, keys):
    if not isinstance(row, dict):
        return None
    for key in keys:
        if key in row:
            return row[key]
    for parent in ("summary", "metrics", "result", "data", "episode", "info", "metadata"):
        child = row.get(parent)
        if isinstance(child, dict):
            value = _nested_get(child, keys)
            if value is not None:
                return value
    return None


def _episode_rows(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = raw.get("episodes", raw.get("results", raw.get("episode_results")))
        if isinstance(rows, dict):
            rows = list(rows.values())
    else:
        rows = None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Cannot find episode summary rows in {path}")
    return rows


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"Unrecognized success value: {value!r}")


def read_metrics(run_id, expected_seeds):
    run_dir = Path(".workspace/runs") / run_id
    config_path = run_dir / "config.json"
    result_path = run_dir / "output" / "result.json"
    episodes_path = run_dir / "output" / "episodes.json"

    if not config_path.is_file():
        raise ValueError(f"{run_id}: missing resolved config {config_path}")
    if not result_path.is_file():
        raise ValueError(f"{run_id}: missing result status {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"{run_id}: run status is {result.get('status')!r}, not 'completed'")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_seeds = (config.get("params") or {}).get("seeds")
    if isinstance(configured_seeds, str):
        configured_seeds = json.loads(configured_seeds)
    if configured_seeds is not None:
        configured_seeds = [int(seed) for seed in configured_seeds]
        if configured_seeds != list(expected_seeds):
            raise ValueError(
                f"{run_id}: config seeds {configured_seeds} do not match requested {list(expected_seeds)}"
            )

    rows = _episode_rows(episodes_path)
    by_seed = {}
    for row in rows:
        recorded_seed = _nested_get(row, ("seed", "episode_seed", "random_seed"))
        if recorded_seed is None:
            raise ValueError(
                f"{run_id}: episode summary has no recorded seed; refusing to pair by row order"
            )
        seed = int(recorded_seed)
        if seed in by_seed:
            raise ValueError(f"{run_id}: duplicate episode seed {seed} in {episodes_path}")
        by_seed[seed] = row

    expected = list(expected_seeds)
    if set(by_seed) != set(expected):
        raise ValueError(
            f"{run_id}: episode seeds {sorted(by_seed)} do not exactly match requested {sorted(expected)}"
        )

    successes, distances = [], []
    for seed in expected:
        row = by_seed[seed]
        success = _nested_get(row, ("success", "is_success"))
        if success is None:
            raise ValueError(f"{run_id}: missing success field for seed {seed} in {episodes_path}")
        successes.append(_as_bool(success))
        distance = _nested_get(row, ("final_distance", "distance", "goal_distance"))
        if distance is None:
            raise ValueError(f"{run_id}: missing final distance for seed {seed} in {episodes_path}")
        distance = float(distance)
        if not math.isfinite(distance):
            raise ValueError(f"{run_id}: non-finite final distance for seed {seed} in {episodes_path}")
        distances.append(distance)

    return {
        "episodes": len(successes),
        "seeds": expected,
        "successes": successes,
        "success_rate": sum(successes) / len(successes),
        "mean_final_distance": sum(distances) / len(distances),
    }


def patch_box(center_x, center_y, area):
    side = math.sqrt(area)
    half = side / 2.0
    center_x = min(1.0 - half, max(half, center_x))
    center_y = min(1.0 - half, max(half, center_y))
    return [center_x - half, center_y - half, center_x + half, center_y + half]


def _run_candidate(args, box, seeds, label):
    run_id = f"{args.prefix}_{label}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    plugins = [{
        "plugin": "a-patch-attack",
        "config": {"area": args.area, "box": box, "camera": args.camera, "value": 0},
    }]
    command = [
        "./dev.sh", "run", args.recipe, run_id,
        f"params.episodes={len(seeds)}",
        f"params.seeds={json.dumps(seeds, separators=(',', ':'))}",
        f"params.plugins={json.dumps(plugins, separators=(',', ':'))}",
        "--registry", args.registry,
    ]
    subprocess.run(command, check=True)
    return run_id, read_metrics(run_id, seeds)


def _objective(metrics):
    distance = metrics["mean_final_distance"]
    return metrics["success_rate"], -(distance if distance is not None else 0.0)


def _bootstrap_paired_ci(best, random_baseline, seed, samples=10000):
    if len(best) != len(random_baseline):
        raise ValueError("paired bootstrap inputs must have equal lengths")
    differences = [int(a) - int(b) for a, b in zip(best, random_baseline)]
    if not differences:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [differences[rng.randrange(len(differences))] for _ in differences]
        means.append(sum(draw) / len(draw))
    means.sort()
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, help="exact MemoryVLA carrot recipe ID")
    parser.add_argument("--registry", default="native/plugins_c/recipes.json")
    parser.add_argument("--prefix", default="c_s1_patch")
    parser.add_argument("--camera", default="3rd_view_camera")
    parser.add_argument("--area", type=float, default=0.04)
    parser.add_argument("--train-seeds", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--eval-seeds", default="12,13,14,15,16,17,18,19,20,21,22,23")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--out", default=".workspace/s1_patch_search")
    args = parser.parse_args()

    if not 0.0 < args.area <= 0.04:
        raise SystemExit("--area must be greater than 0 and no larger than the declared 0.04 threat budget")
    if args.candidates < 1 or args.rounds < 1 or args.candidates * args.rounds > 40:
        raise SystemExit("candidate count and rounds must be positive; total training candidates must be <= 40")
    if not math.isfinite(args.radius) or args.radius <= 0:
        raise SystemExit("--radius must be finite and positive")

    try:
        train_seeds = parse_seeds(args.train_seeds)
        eval_seeds = parse_seeds(args.eval_seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if set(train_seeds) & set(eval_seeds):
        raise SystemExit("training and held-out seed sets must be disjoint")

    rng = random.Random(args.seed)
    parent = Path(args.out)
    parent.mkdir(parents=True, exist_ok=True)
    output_dir = parent / f"search_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "search_log.jsonl"

    best_center = None
    best_metrics = None
    radius = args.radius
    queries = 0

    with log_path.open("w", encoding="utf-8") as log:
        for round_index in range(args.rounds):
            for candidate_index in range(args.candidates):
                if best_center is None:
                    center_x, center_y = rng.random(), rng.random()
                else:
                    center_x = best_center[0] + rng.uniform(-radius, radius)
                    center_y = best_center[1] + rng.uniform(-radius, radius)
                box = patch_box(center_x, center_y, args.area)
                run_id, metrics = _run_candidate(
                    args, box, train_seeds, f"r{round_index + 1}_q{candidate_index + 1}"
                )
                queries += 1
                record = {
                    "round": round_index + 1,
                    "candidate": candidate_index + 1,
                    "query": queries,
                    "run_id": run_id,
                    "box": box,
                    "metrics": {key: value for key, value in metrics.items() if key != "successes"},
                }
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                if best_metrics is None or _objective(metrics) < _objective(best_metrics):
                    best_center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                    best_metrics = metrics
            radius *= 0.5

    best_box = patch_box(best_center[0], best_center[1], args.area)
    random_box = patch_box(rng.random(), rng.random(), args.area)
    best_run, best_eval = _run_candidate(args, best_box, eval_seeds, "heldout_best")
    random_run, random_eval = _run_candidate(args, random_box, eval_seeds, "heldout_random")
    difference = best_eval["success_rate"] - random_eval["success_rate"]
    confidence_interval = _bootstrap_paired_ci(
        best_eval["successes"], random_eval["successes"], args.seed
    )

    summary = {
        "recipe": args.recipe,
        "search_seed": args.seed,
        "candidate_count_per_round": args.candidates,
        "rounds": args.rounds,
        "initial_radius": args.radius,
        "camera": args.camera,
        "patch_area": args.area,
        "training_seeds": train_seeds,
        "heldout_seeds": eval_seeds,
        "training_queries": queries,
        "configured_training_query_budget": args.candidates * args.rounds,
        "hard_training_query_limit": 40,
        "training_episodes_total": queries * len(train_seeds),
        "heldout_evaluation_runs": 2,
        "paired_seed_order": eval_seeds,
        "total_heldout_episodes": 2 * len(eval_seeds),
        "best_box": best_box,
        "training_metrics": {key: value for key, value in best_metrics.items() if key != "successes"},
        "heldout_best": {
            "run_id": best_run,
            **{key: value for key, value in best_eval.items() if key != "successes"},
        },
        "heldout_random_same_area": {
            "run_id": random_run,
            "box": random_box,
            **{key: value for key, value in random_eval.items() if key != "successes"},
        },
        "success_rate_difference_best_minus_random": difference,
        "paired_bootstrap_95_percent_ci": confidence_interval,
        "interpretation": "Negative best-minus-random difference means the searched patch reduced success more than the same-area random patch.",
        "search_log": str(log_path),
        "note": "Training search uses only training seeds. The final best-vs-random comparison uses disjoint held-out seeds. No simulator truth or model internals are read.",
    }
    summary_path = output_dir / "search_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
