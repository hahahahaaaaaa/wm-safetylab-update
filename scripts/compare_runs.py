"""Compare episode-kernel runs pairwise, step by step: actions, lengths, outcomes, hazards, takeovers.

    python scripts/compare_runs.py RUN_A RUN_B [RUN_A2 RUN_B2 ...] [--runs-dir runs]

A and B are run IDs under --runs-dir (or output directories). For the same seeds, a run with
no plugin, or only monitors, must give max_abs_action_diff 0 against its reference: that is
the gate every platform change and every monitor passes. For a fault, attack or defense the
same table is the paired comparison: per-episode success in A and B, and the success
difference with a 95% interval over the paired episodes. Needs only the standard library.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _output(runs_dir, run):
    path = Path(run)
    if (path / "episodes.json").exists():
        return path
    return Path(runs_dir) / run / "output"


def _steps(output, episode):
    path = output / "episodes" / f"ep_{episode:03d}" / "steps.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_difference(a, b):
    """Mean of b - a over paired episodes and a normal-approximation 95% interval."""
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    n = len(diffs)
    if not n:
        return None
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "ci95": [mean - half, mean + half]}


def compare(runs_dir, a, b):
    out_a, out_b = _output(runs_dir, a), _output(runs_dir, b)
    rows_a = json.loads((out_a / "episodes.json").read_text(encoding="utf-8"))["episodes"]
    rows_b = json.loads((out_b / "episodes.json").read_text(encoding="utf-8"))["episodes"]
    worst, lengths, outcomes, seeds = 0.0, [], [], []
    for ra, rb in zip(rows_a, rows_b):
        if ra.get("seed") != rb.get("seed"):
            seeds.append(ra["episode"])
        sa, sb = _steps(out_a, ra["episode"]), _steps(out_b, rb["episode"])
        if len(sa) != len(sb):
            lengths.append(ra["episode"])
        for x, y in zip(sa, sb):
            if x["action"] is not None and y["action"] is not None:
                worst = max(worst, max(abs(float(p) - float(q)) for p, q in zip(x["action"], y["action"])))
        if ra["success"] != rb["success"]:
            outcomes.append(ra["episode"])
    hazards = {}
    for label, rows in (("a", rows_a), ("b", rows_b)):
        totals = {}
        for row in rows:
            for key, count in (row.get("hazard_steps") or {}).items():
                totals[key] = totals.get(key, 0) + count
        hazards[label] = totals
    return {"a": a, "b": b, "episodes": [len(rows_a), len(rows_b)], "seed_mismatch": seeds,
            "max_abs_action_diff": worst, "length_mismatch": lengths, "outcome_mismatch": outcomes,
            "success": [sum(r["success"] for r in rows_a), sum(r["success"] for r in rows_b)],
            "success_difference": paired_difference([r["success"] for r in rows_a], [r["success"] for r in rows_b]),
            "hazard_steps": hazards, "takeovers_b": sum(r.get("takeovers", 0) for r in rows_b)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare episode-kernel runs pairwise.")
    parser.add_argument("runs", nargs="+", help="RUN_A RUN_B [RUN_A2 RUN_B2 ...]")
    parser.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[1] / "runs"))
    args = parser.parse_args(argv)
    if len(args.runs) % 2:
        raise SystemExit("give runs in pairs")
    for a, b in zip(args.runs[::2], args.runs[1::2]):
        print(json.dumps(compare(args.runs_dir, a, b), ensure_ascii=False))


if __name__ == "__main__":
    main()
