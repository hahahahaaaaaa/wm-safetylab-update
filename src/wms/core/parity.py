"""A/B parity: does the episode entry (path B) reproduce the official entry (path A)?

Native task; reads two finished runs and writes parity.json:

  inputs  official_run   run ID of the official evaluation
          episode_run    run ID of the episode kernel (episodes.json + episodes/ep_*/steps.jsonl)
  params  official_format  "memoryvla" (default): episodes.jsonl, plus actions.jsonl when the run used
                           params.log_actions; episodes paired by layout (episode_id == episode seed).
                           "tdmpc2": evaluation.json + trajectory_*.npz of policy_evaluate (needs numpy);
                           episodes paired by seed, success = success at any step (success_once).
          task           memoryvla only: official task name to compare (default: the only one in the run)
          action_tol     largest allowed |difference| in the first action (default 1e-4)
          min_agreement  share of episodes whose success must agree (default 22/24)

The verdict passes when every paired first action is within action_tol (if the official run has
actions) and at least min_agreement of the outcomes agree. The report also gives the largest
action difference over all steps and any episode-length mismatch, per episode.
"""
from __future__ import annotations

import json
from pathlib import Path
import re


def _run_output(output_dir, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"run_[0-9a-f]+", run_id):
        raise ValueError(f"Use an existing platform run ID (got {run_id!r})")
    path = Path(output_dir).resolve().parents[1] / run_id / "output"
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def official_memoryvla(source, task=None):
    """(task, {episode_id: {"success", "actions"}}) from wms.engines.memoryvla.simpler.evaluate."""
    rows = _jsonl(source / "episodes.jsonl")
    tasks = sorted({row["task"] for row in rows})
    if task is None:
        if len(tasks) != 1:
            raise ValueError(f"The official run has tasks {tasks}; set params.task")
        task = tasks[0]
    actions = {}
    for row in _jsonl(source / "actions.jsonl"):
        if row.get("task") == task:
            actions.setdefault(int(row["episode_id"]), []).append(row["action"])
    return task, {int(row["episode_id"]): {"success": bool(row["success"]),
                                           "actions": actions.get(int(row["episode_id"]))}
                  for row in rows if row["task"] == task}


def official_tdmpc2(source):
    """(env_id, {seed: {"success", "actions"}}) from wms.engines.tdmpc2.native.policy_evaluate."""
    import numpy as np
    report = json.loads((source / "evaluation.json").read_text(encoding="utf-8"))
    out = {}
    for row in report["episodes"]:
        with np.load(source / row["trajectory"]) as trajectory:
            actions = trajectory["actions"].tolist()
        out[int(row["seed"])] = {"success": bool(row["success_once"]), "actions": actions}
    return report.get("env_id"), out


def episode_episodes(source):
    """{seed: {"success", "actions", "steps"}} from an episode-kernel run."""
    summary = json.loads((source / "episodes.json").read_text(encoding="utf-8"))
    out = {}
    for row in summary["episodes"]:
        steps = _jsonl(source / "episodes" / f"ep_{int(row['episode']):03d}" / "steps.jsonl")
        out[int(row["seed"])] = {"success": bool(row["success"]), "steps": row.get("steps"),
                                 "actions": [step["action"] for step in steps]}
    return out


def _max_diff(a, b):
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def compare(official, episode, action_tol=1e-4, min_agreement=22 / 24):
    paired = sorted(set(official) & set(episode))
    rows, agree, worst_first, worst_all = [], 0, 0.0, 0.0
    actions_checked = bool(paired) and all(official[k].get("actions") for k in paired)
    for key in paired:
        a, b = official[key], episode[key]
        same = a["success"] == b["success"]
        agree += same
        row = {"layout": key, "official_success": a["success"], "episode_success": b["success"],
               "outcome_agrees": same, "first_action_max_abs_diff": None, "all_steps_max_abs_diff": None,
               "official_steps": None, "episode_steps": len(b.get("actions") or [])}
        if a.get("actions") and b.get("actions"):
            row["first_action_max_abs_diff"] = _max_diff(a["actions"][0], b["actions"][0])
            row["all_steps_max_abs_diff"] = max(_max_diff(x, y) for x, y in zip(a["actions"], b["actions"]))
            row["official_steps"] = len(a["actions"])
            worst_first = max(worst_first, row["first_action_max_abs_diff"])
            worst_all = max(worst_all, row["all_steps_max_abs_diff"])
        rows.append(row)
    share = agree / len(paired) if paired else 0.0
    checks = {"paired_episodes": len(paired), "outcome_agreement": share,
              "outcome_ok": bool(paired) and share >= min_agreement - 1e-12,
              "first_action_checked": actions_checked,
              "first_action_max_abs_diff": worst_first if actions_checked else None,
              "all_steps_max_abs_diff": worst_all if actions_checked else None,
              "length_mismatches": [r["layout"] for r in rows
                                    if r["official_steps"] is not None and r["official_steps"] != r["episode_steps"]],
              "first_action_ok": (worst_first <= action_tol) if actions_checked else None,
              "unpaired_official": sorted(set(official) - set(episode)),
              "unpaired_episode": sorted(set(episode) - set(official))}
    checks["passed"] = checks["outcome_ok"] and checks["first_action_ok"] is not False
    return checks, rows


def run(config, output_dir, context):
    inputs, params = dict(config.get("inputs", {})), dict(config.get("params", {}))
    official_dir = _run_output(output_dir, inputs.get("official_run"))
    episode_dir = _run_output(output_dir, inputs.get("episode_run"))
    form = params.get("official_format", "memoryvla")
    if form == "memoryvla":
        task, official = official_memoryvla(official_dir, params.get("task"))
    elif form == "tdmpc2":
        task, official = official_tdmpc2(official_dir)
    else:
        raise ValueError("params.official_format must be 'memoryvla' or 'tdmpc2'")
    episode = episode_episodes(episode_dir)
    tol = float(params.get("action_tol", 1e-4))
    share = float(params.get("min_agreement", 22 / 24))
    checks, rows = compare(official, episode, tol, share)
    report = {"task": task, "official_format": form, "official_run": inputs["official_run"],
              "episode_run": inputs["episode_run"], "criteria": {"action_tol": tol, "min_agreement": share},
              "official_successes": sum(r["official_success"] for r in rows),
              "episode_successes": sum(r["episode_success"] for r in rows), **checks, "episodes": rows}
    context.write_json("parity.json", report)
    context.event("progress", {"phase": "parity", "passed": checks["passed"],
                               "outcome_agreement": checks["outcome_agreement"],
                               "first_action_max_abs_diff": checks["first_action_max_abs_diff"]})
    data = {key: report[key] for key in ("task", "passed", "paired_episodes", "outcome_agreement",
                                         "first_action_checked", "first_action_max_abs_diff", "all_steps_max_abs_diff",
                                         "length_mismatches", "official_successes", "episode_successes")}
    return {"data": data, "outputs": [{"path": "parity.json", "kind": "report", "media_type": "application/json"}]}
