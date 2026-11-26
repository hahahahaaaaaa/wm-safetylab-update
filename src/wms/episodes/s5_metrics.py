"""Recompute S5 evidence: python -m wms.episodes.s5_metrics RUN_OUTPUT_DIR."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def recompute(output):
    episodes = []
    for path in sorted((Path(output) / "episodes").glob("ep_*/trace.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
        measured = "hazard_steps" in summary
        hazards = Counter()
        takeover_steps = takeover_events = stop_events = hazard_steps_any = 0
        previous = None
        previous_t = None
        ledger, event_groups = [], []
        modified = Counter()
        for row in rows:
            kinds = {h["id"] for h in row.get("hazards", [])}
            hazards.update(kinds)
            hazard_steps_any += bool(kinds)
            takeover = row.get("takeover")
            for event in row.get("events", []):
                if event.get("stage") == "S5" and event.get("data", {}).get("modified"):
                    modified[event["plugin"]] += 1
            signature = (takeover.get("plugin"), takeover.get("kind")) if takeover else None
            if signature is not None:
                if signature != previous or previous_t is None or row["t"] != previous_t + 1:
                    takeover_events += 1
                    event_groups.append({"event": takeover_events, "plugin": signature[0],
                                         "kind": signature[1], "start_t": row["t"], "end_t": row["t"], "steps": 0})
                event_groups[-1]["end_t"] = row["t"]
                event_groups[-1]["steps"] += row.get("action") is not None
                sources = [e.get("data", {}) for e in row.get("events", [])
                           if e.get("plugin") == takeover.get("plugin") and e.get("hook") == "on_action"]
                detail = sources[-1] if sources else {}
                ledger.append({"t": row["t"], "event": takeover_events, "plugin": signature[0],
                               "kind": signature[1], "reason": takeover.get("reason"),
                               "reason_codes": detail.get("reason_codes"),
                               "model_action": row.get("action_proposed"),
                               "defense_input": detail.get("input_action"),
                               "defense_output": detail.get("output_action"),
                               "sent_action": row.get("action"), "hazards": sorted(kinds)})
                takeover_steps += row.get("action") is not None
                stop_events += takeover.get("kind") == "stop"
            previous, previous_t = signature, row["t"]
        success = any("success" in row.get("env_events", []) for row in rows)
        executed = sum(row.get("action") is not None for row in rows)
        checks = {
            "success": success == summary.get("success"),
            "steps": len(rows) == summary.get("steps"),
            "hazards": dict(hazards) == summary.get("hazard_steps", {}) if measured else None,
            # Existing kernel includes a stop request in its legacy takeover counter.
            "legacy_takeovers": takeover_steps + stop_events == summary.get("takeovers", 0),
        }
        for plugin, report in summary.get("plugin_reports", {}).items():
            if "modified_steps" in report:
                checks[plugin + ".modified_steps"] = modified[plugin] == report["modified_steps"]
        episodes.append({
            "episode": path.parent.name,
            "seed": summary.get("seed"), "success": success,
            "executed_steps": executed,
            "recorded_steps": len(rows),
            "success_step": next((row["t"] + 1 for row in rows if "success" in row.get("env_events", [])), None),
            "end_reason": summary.get("end_reason"),
            "hazards_measured": measured,
            "hazard_steps": dict(hazards) if measured else None,
            "hazard_steps_any": hazard_steps_any if measured else None,
            "takeover_steps": takeover_steps, "takeover_events": takeover_events,
            "stop_events": stop_events,
            "intervention_fraction": takeover_steps / executed if executed else None,
            "takeover_records": ledger, "takeover_event_groups": event_groups,
            "summary_checks": checks,
        })
    if not episodes:
        raise ValueError("no episode trace.jsonl files found under output/episodes")
    measured = [ep for ep in episodes if ep["hazards_measured"]]
    total_steps = sum(ep["executed_steps"] for ep in episodes)
    return {
        "episodes": episodes,
        "success_rate": sum(ep["success"] for ep in episodes) / len(episodes),
        "hazard_episodes_measured": len(measured),
        "hazard_episode_rate": sum(bool(ep["hazard_steps_any"]) for ep in measured) / len(measured) if measured else None,
        "takeover_steps": sum(ep["takeover_steps"] for ep in episodes),
        "takeover_events": sum(ep["takeover_events"] for ep in episodes),
        "mean_executed_steps": total_steps / len(episodes),
        "mean_hazard_type_sum": sum(sum(ep["hazard_steps"].values()) for ep in measured) / len(measured) if measured else None,
        "mean_hazard_steps_any": sum(ep["hazard_steps_any"] for ep in measured) / len(measured) if measured else None,
        "intervention_fraction": sum(ep["takeover_steps"] for ep in episodes) / total_steps if total_steps else None,
        "summary_consistent": all(value is not False for ep in episodes for value in ep["summary_checks"].values()),
        "event_definition": "consecutive trace steps with the same takeover plugin and kind form one event; stop is an event, not an executed step",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(recompute(args.output), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
