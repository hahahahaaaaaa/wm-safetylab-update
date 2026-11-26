"""Calibrate wms.scenarios.hazards.SimpleHazards thresholds on clean runs.

    python -m wms.scenarios.hazard_calibration RUN_DIR [RUN_DIR ...] [--false-positive 0.01] [--output FILE]

A hazard predicate is useful only if it rarely fires when nothing is wrong. This reads the
per-step true states (steps.jsonl: tcp_xyz, objects) of clean episode-kernel runs (no fault,
no attack), recomputes the quantity behind each SimpleHazards predicate on every step, and
proposes the threshold that clean steps exceed at most at the given rate (default 1%):

  workspace_half_xy   per step max(|tcp_x - centre_x|, |tcp_y - centre_y|), the centre being the
                      objects' initial xy centre (as SimpleHazards); ceiling, floor_margin from
                      tcp height above / below the lowest object's initial height
  drop_height         how far each object is below its initial height
  fast_speed          tcp speed (m/s) on steps that end within contact_radius of an object

It prints, per quantity, the clean-step quantiles and how often the current default would fire,
and writes {"hazards": {...}} ready for an environment's config. A default that already fires on
at most the given rate of clean steps is kept (tightening it to the policy's own quantile would
make the predicate fire on noise, e.g. drop_height when nothing ever falls); one that fires more
often is raised to that quantile times a margin. Report the calibration runs with any hazard result.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .hazards import DEFAULTS


def _quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _episodes(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))
    safety = (summary.get("env_spec") or {}).get("safety") or {}
    dt = float(safety.get("dt_s", 0.1))
    for row in summary["episodes"]:
        path = run_dir / "episodes" / f"ep_{int(row['episode']):03d}" / "steps.jsonl"
        steps = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        yield row.get("initial_state") or (steps[0]["state_before"] if steps else None), steps, dt, safety


def measure(run_dirs, contact_radius=None):
    """Per-step quantities behind each SimpleHazards predicate, over all clean episodes."""
    out = {"half_xy": [], "above": [], "below": [], "drop": [], "near_speed": [], "steps": 0, "episodes": 0}
    for run_dir in run_dirs:
        for initial, steps, dt, safety in _episodes(run_dir):
            if not initial or "tcp_xyz" not in initial or not steps:
                raise ValueError(f"{run_dir}: steps.jsonl has no tcp_xyz / objects (not a SimpleHazards environment)")
            radius = float(contact_radius or safety.get("contact_radius", DEFAULTS["contact_radius"]))
            start = {name: list(map(float, pose[:3])) for name, pose in initial["objects"].items()}
            cx = sum(p[0] for p in start.values()) / len(start)
            cy = sum(p[1] for p in start.values()) / len(start)
            low = min(p[2] for p in start.values())
            last = list(map(float, initial["tcp_xyz"][:3]))
            out["episodes"] += 1
            for step in steps:
                state = step.get("state_after") or {}
                if "tcp_xyz" not in state:
                    continue
                tcp = list(map(float, state["tcp_xyz"][:3]))
                objects = {name: list(map(float, pose[:3])) for name, pose in state["objects"].items()}
                out["steps"] += 1
                out["half_xy"].append(max(abs(tcp[0] - cx), abs(tcp[1] - cy)))
                out["above"].append(tcp[2] - low)
                out["below"].append(low - tcp[2])
                out["drop"] += [start[name][2] - xyz[2] for name, xyz in objects.items() if name in start]
                distance = min(math.dist(tcp, xyz) for xyz in objects.values())
                if distance < radius:
                    out["near_speed"].append(math.dist(tcp, last) / dt)
                last = tcp
    return out


def propose(measured, false_positive=0.01, margin=1.05):
    q = 1.0 - false_positive

    def summary(values, current):
        return {"n": len(values), "p50": _quantile(values, 0.5), "p90": _quantile(values, 0.9),
                "p99": _quantile(values, 0.99), "max": max(values) if values else None, "current": current,
                "current_fires": (sum(v > current for v in values) / len(values)) if values else None}

    stats = {"workspace_half_xy": summary(measured["half_xy"], DEFAULTS["workspace_half_xy"]),
             "ceiling": summary(measured["above"], DEFAULTS["ceiling"]),
             "floor_margin": summary(measured["below"], DEFAULTS["floor_margin"]),
             "drop_height": summary(measured["drop"], DEFAULTS["drop_height"]),
             "fast_speed": summary(measured["near_speed"], DEFAULTS["fast_speed"])}
    sources = {"workspace_half_xy": "half_xy", "ceiling": "above", "floor_margin": "below",
               "drop_height": "drop", "fast_speed": "near_speed"}
    hazards, changed = {}, []
    for key, source in sources.items():
        value, fires = _quantile(measured[source], q), stats[key]["current_fires"]
        if value is None or fires is None or fires <= false_positive:
            hazards[key] = DEFAULTS[key]
        else:
            hazards[key] = round(value * margin, 4)
            changed.append(key)
    return {"changed": changed, "false_positive_target": false_positive, "margin": margin, "episodes": measured["episodes"],
            "steps": measured["steps"], "hazards": hazards, "stats": stats}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Propose SimpleHazards thresholds from clean runs.")
    parser.add_argument("runs", nargs="+", help="run output directories (with episodes.json)")
    parser.add_argument("--false-positive", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.05, help="multiply each quantile by this")
    parser.add_argument("--contact-radius", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    result = propose(measure(args.runs, args.contact_radius), args.false_positive, args.margin)
    result["runs"] = [str(Path(r)) for r in args.runs]
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
