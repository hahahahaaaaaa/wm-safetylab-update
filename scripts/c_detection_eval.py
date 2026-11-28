#!/usr/bin/env python3
"""Evaluate an S1/S6 monitor from trace.jsonl using clean p99 and injection labels.

The evaluator is deliberately fail-closed: every recorded step must have a
monitor score, injected steps must have explicit injection labels, and missed
episodes remain in the detection-rate denominator. Mean latency is reported only
among detected episodes and is named accordingly; undetected episodes are not
silently discarded.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


DEFAULT_INJECTIONS = (
    "a-obs-corruption",
    "a-obs-noise-vector",
    "a-camera-jitter",
)


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


def _plugin_matches(actual, expected):
    if not isinstance(actual, str):
        return False
    return actual == expected or actual.rsplit(":", 1)[-1] == expected


def _episode_dirs(run_dir):
    run_dir = Path(run_dir)
    for root in (run_dir / "output" / "episodes", run_dir / "episodes", run_dir):
        if root.is_dir():
            found = sorted(path for path in root.glob("ep_*") if path.is_dir())
            if found:
                return found
    return []


def _records(run_dir, monitor_id, injection_ids):
    run_dir = Path(run_dir)
    episodes = _episode_dirs(run_dir)
    if not episodes:
        raise ValueError(f"No ep_* directories found under {run_dir}")

    records = {}
    for episode_dir in episodes:
        trace_path = episode_dir / "trace.jsonl"
        if not trace_path.is_file():
            raise ValueError(f"Missing trace.jsonl: {trace_path}")
        episode_records = {}
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_t = row.get("t") if isinstance(row, dict) else None
                if row_t is not None:
                    episode_records.setdefault(int(row_t), {
                        "scores": [], "injected": False, "uncalibrated_scores": 0,
                        "stats_sources": set(),
                    })
                events = list(_walk_events(row))
                for event in events:
                    t = int(event.get("t", row_t if row_t is not None else -1))
                    if t < 0:
                        raise ValueError(f"{trace_path}:{line_number}: event has no valid step index")
                    slot = episode_records.setdefault(t, {
                        "scores": [], "injected": False, "uncalibrated_scores": 0,
                        "stats_sources": set(),
                    })
                    plugin_id = event.get("plugin")
                    data = event.get("data") or {}
                    if _plugin_matches(plugin_id, monitor_id) and data.get("score") is not None:
                        score = float(data["score"])
                        if not math.isfinite(score):
                            raise ValueError(f"non-finite monitor score in {trace_path} at t={t}")
                        slot["scores"].append(score)
                        if data.get("calibrated") is not True:
                            slot["uncalibrated_scores"] += 1
                        if data.get("stats_source"):
                            slot["stats_sources"].add(str(data["stats_source"]))
                    if any(_plugin_matches(plugin_id, item) for item in injection_ids):
                        if data.get("injected") is True:
                            slot["injected"] = True
        if not episode_records:
            raise ValueError(f"No step/event records found in {trace_path}")
        records[episode_dir.name] = episode_records
    return records


def _quantile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("no clean monitor scores found")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def _auc(labels, scores):
    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(bool(label) for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def evaluate(clean_runs, test_runs, monitor_id, injection_ids, allow_uncalibrated=False):
    clean_scores = []
    uncalibrated_count = 0
    stats_sources = set()
    for run_dir in clean_runs:
        clean_records = _records(run_dir, monitor_id, injection_ids)
        for episode, rows in clean_records.items():
            for t, slot in rows.items():
                if slot["injected"]:
                    raise ValueError(f"Clean run {run_dir}/{episode} contains an injection event at t={t}")
                if not slot["scores"]:
                    raise ValueError(
                        f"Clean run {run_dir}/{episode} step {t} has no monitor score"
                    )
                clean_scores.append(max(slot["scores"]))
                uncalibrated_count += slot["uncalibrated_scores"]
                stats_sources.update(slot["stats_sources"])

    threshold = _quantile(clean_scores, 0.99)
    labels, scores = [], []
    per_episode = []
    missing_score_steps = []

    for run_dir in test_runs:
        records = _records(run_dir, monitor_id, injection_ids)
        for episode, rows in records.items():
            injected_steps = [t for t, slot in rows.items() if slot["injected"]]
            for t, slot in rows.items():
                uncalibrated_count += slot["uncalibrated_scores"]
                stats_sources.update(slot["stats_sources"])
                if not slot["scores"]:
                    missing_score_steps.append({
                        "run": str(run_dir), "episode": episode, "step": t,
                        "injected": slot["injected"], "score_count": 0,
                    })
                    continue
                labels.append(bool(slot["injected"]))
                scores.append(max(slot["scores"]))

            if not injected_steps:
                continue
            first_injected = min(injected_steps)
            detected_steps = [
                t for t, slot in rows.items()
                if t >= first_injected and slot["scores"] and max(slot["scores"]) > threshold
            ]
            detected = bool(detected_steps)
            last_step = max(rows)
            delay = min(detected_steps) - first_injected if detected else None
            per_episode.append({
                "run": str(run_dir),
                "episode": episode,
                "first_injected_step": first_injected,
                "detected": detected,
                "detection_delay_steps": delay,
                "observation_horizon_after_injection_steps": max(0, last_step - first_injected),
            })

    if missing_score_steps:
        sample = missing_score_steps[:5]
        raise ValueError(
            f"{len(missing_score_steps)} trace steps have no monitor score; "
            f"refusing biased evaluation. Examples: {sample}"
        )
    if not allow_uncalibrated and uncalibrated_count:
        raise ValueError(
            f"Found {uncalibrated_count} uncalibrated monitor scores; fit clean input statistics before final evaluation"
        )
    if not scores:
        raise ValueError(f"No scores for monitor ID {monitor_id!r} found in test traces")
    if not any(labels):
        raise ValueError(f"No explicit injection events for IDs {list(injection_ids)!r} found in test traces")

    predictions = [score > threshold for score in scores]
    tp = sum(pred and label for pred, label in zip(predictions, labels))
    fp = sum(pred and not label for pred, label in zip(predictions, labels))
    tn = sum((not pred) and (not label) for pred, label in zip(predictions, labels))
    fn = sum((not pred) and label for pred, label in zip(predictions, labels))
    detected_delays = [item["detection_delay_steps"] for item in per_episode
                       if item["detection_delay_steps"] is not None]
    detected_episode_count = len(detected_delays)
    injected_episode_count = len(per_episode)

    return {
        "monitor_id": monitor_id,
        "injection_ids": list(injection_ids),
        "clean_score_count": len(clean_scores),
        "clean_p99_threshold": threshold,
        "stats_sources": sorted(stats_sources),
        "uncalibrated_score_count": uncalibrated_count,
        "test_step_count": len(scores),
        "injected_step_count": sum(labels),
        "auroc": _auc(labels, scores),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "injected_episode_count": injected_episode_count,
        "detected_episode_count": detected_episode_count,
        "undetected_episode_count": injected_episode_count - detected_episode_count,
        "episode_detection_rate": detected_episode_count / injected_episode_count if injected_episode_count else None,
        "mean_detection_delay_steps_detected_only": statistics.mean(detected_delays) if detected_delays else None,
        "median_detection_delay_steps_detected_only": statistics.median(detected_delays) if detected_delays else None,
        "episodes": per_episode,
        "note": (
            "AUROC and threshold metrics use the maximum monitor score recorded at each step. "
            "Undetected episodes remain in the detection-rate denominator; latency averages are conditional on detection."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", nargs="+", required=True, help="clean runs with the monitor enabled")
    parser.add_argument("--test", nargs="+", required=True, help="runs with the monitor and injection plugin")
    parser.add_argument("--monitor-id", default="a-input-monitor", help="one registered monitor ID per invocation")
    parser.add_argument(
        "--injection-id", action="append", default=None,
        help="registered injection ID; repeat for multiple IDs (defaults to S1 #1/#2 IDs)",
    )
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="exploratory only: allow scores without calibrated=true (not valid for final S1 results)")
    parser.add_argument("--output", default="s1_detection_eval.json")
    args = parser.parse_args()

    injection_ids = tuple(args.injection_id or DEFAULT_INJECTIONS)
    result = evaluate(
        args.clean, args.test, args.monitor_id, injection_ids,
        allow_uncalibrated=args.allow_uncalibrated,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
