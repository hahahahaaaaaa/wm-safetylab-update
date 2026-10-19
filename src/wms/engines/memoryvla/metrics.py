"""Measure a saved MemoryVLA SimplerEnv evaluation without importing models or simulators."""
import json
import math
from pathlib import Path

VERSION = "memoryvla-simpler/1"


def _rate(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be a percentage")
    return float(value)


def measure(result, params):
    """Recount success from episodes.jsonl (not the summary) and compare with the published numbers."""
    source = Path(result["source_output_dir"]).resolve()
    output = Path(result["output_dir"]).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Evaluation output must be outside the source directory")
    summary = json.loads((source / "evaluation.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (source / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("episodes.jsonl has no episodes")
    tasks = {}
    for row in rows:
        if type(row.get("success")) is not bool:
            raise ValueError("every episode needs a boolean success")
        tasks.setdefault(row["task"], []).append(row["success"])
    metrics = {f"memoryvla_{name}_success_pct": 100.0 * sum(values) / len(values) for name, values in tasks.items()}
    metrics["memoryvla_average_success_pct"] = sum(metrics.values()) / len(metrics)
    metrics["memoryvla_episodes"] = len(rows)
    reference = params.get("reference") or {}
    for name in tasks:
        if name in reference:
            metrics[f"memoryvla_{name}_minus_official_pct"] = (
                metrics[f"memoryvla_{name}_success_pct"] - _rate(reference[name], name))
    if "average" in reference and len(tasks) == 4:
        metrics["memoryvla_average_minus_official_pct"] = (
            metrics["memoryvla_average_success_pct"] - _rate(reference["average"], "average"))
    recorded = summary.get("average_success_pct")
    if recorded is None or abs(recorded - metrics["memoryvla_average_success_pct"]) > 1e-9:
        raise ValueError("evaluation.json average does not match episodes.jsonl")
    report = {"metric_version": VERSION, "source_output_dir": str(source), "metrics": metrics,
              "differences_from_upstream_script": summary.get("differences_from_upstream_script"),
              "safety_status": "not_measured",
              "conclusion_limits": ["24 episodes per task: one episode is 4.2 percentage points.",
                                    "Upstream reports its best of several checkpoints; this is the one released checkpoint.",
                                    "Task success does not measure safety."]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "memoryvla_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
