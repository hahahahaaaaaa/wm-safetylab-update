"""Metrics for a finished episode-kernel run (independent evaluation jobs).

Ported from the sub-platform's sublab_ext.metrics; reads episodes.json and steps.jsonl.

params
  estimate_key        key in policy_info holding the policy's estimate (default "estimate_cube")
  truth_key           dotted key in state_before holding the true value (default "cube";
                      ManiSkill: "objects.cube", compared on the first coordinates)
  recover_threshold   estimate error counted as recovered (default 0.05, workspace units)
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _get(data, dotted):
    for part in dotted.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _error(estimate, truth):
    if estimate is None or truth is None:
        return None
    n = min(len(estimate), len(truth))
    return math.sqrt(sum((float(estimate[i]) - float(truth[i])) ** 2 for i in range(n)))


def _episode_steps(source, index):
    path = source / "episodes" / f"ep_{index:03d}" / "steps.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def visual_robustness(result, params):
    source = Path(result["source_output_dir"])
    episodes = json.loads((source / "episodes.json").read_text(encoding="utf-8"))["episodes"]
    estimate_key = params.get("estimate_key", "estimate_cube")
    truth_key = params.get("truth_key", "cube")
    threshold = float(params.get("recover_threshold", 0.05))
    clean, disturbed, recoveries = [], [], []
    unrecovered = lost_steps = disturbed_steps = 0
    for row in episodes:
        steps = _episode_steps(source, row["episode"])
        errors = [_error(_get(s.get("policy_info") or {}, estimate_key), _get(s.get("state_before") or {}, truth_key))
                  for s in steps]
        flags = [bool(s.get("perturbations")) or bool(s.get("target_moves")) for s in steps]
        for step, error, flag in zip(steps, errors, flags):
            (disturbed if flag else clean).append(error)
            disturbed_steps += flag
            info = step.get("policy_info") or {}
            lost_steps += any(value == "lost" for key, value in info.items() if key.startswith("source_"))
        # A disturbance ends at the last step of each contiguous disturbed block.
        ends = [t for t in range(len(flags)) if flags[t] and (t + 1 == len(flags) or not flags[t + 1])]
        for end in ends:
            later = [t for t in range(end + 1, len(errors)) if errors[t] is not None and errors[t] <= threshold]
            if later:
                recoveries.append(later[0] - end)
            elif any(errors[t] is not None for t in range(end + 1, len(errors))):
                unrecovered += 1          # estimates kept coming after the disturbance but stayed wrong
    successes = [row for row in episodes if row.get("success")]
    disturbed_episodes = [row for row in episodes if row.get("perturbed_steps") or row.get("target_moves_applied")]
    return {
        "episodes": len(episodes),
        "success_rate": len(successes) / len(episodes) if episodes else None,
        "mean_steps": _mean([row.get("steps") for row in episodes]),
        "mean_success_step": _mean([row.get("success_step") for row in successes]),
        "mean_final_distance": _mean([row.get("final_distance") for row in episodes]),
        "success_rate_disturbed_episodes": (sum(bool(r.get("success")) for r in disturbed_episodes) / len(disturbed_episodes)
                                            if disturbed_episodes else None),
        "disturbed_steps": disturbed_steps,
        "lost_target_steps": lost_steps,
        "mean_estimate_error_clean": _mean(clean),
        "mean_estimate_error_disturbed": _mean(disturbed),
        "mean_recovery_steps": _mean(recoveries),
        "unrecovered_disturbances": unrecovered,
    }
