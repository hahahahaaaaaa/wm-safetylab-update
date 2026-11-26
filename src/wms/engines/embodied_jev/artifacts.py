"""Archive native embodied-jev exports without importing the upstream runtime."""
import copy
import html
import json
import math
from pathlib import Path

from .metrics import VERSION, executed, records, summarize


def _decision(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("decision must be an object or null")
    result = copy.deepcopy(value)
    probabilities = result.get("probabilities")
    if probabilities is not None:
        if not isinstance(probabilities, dict):
            raise ValueError("probabilities must be an object or null")
        for probability in probabilities.values():
            if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Reported probabilities must be finite values in [0, 1]")
    result["probabilities"] = probabilities or None
    selected = result.get("selected_probability")
    if selected is not None and (
            type(selected) not in (int, float) or not math.isfinite(selected) or not 0 <= selected <= 1):
        raise ValueError("selected_probability must be in [0, 1] or null")
    result["selected_probability"] = selected
    if "channel_decisions" in result:
        channels = result["channel_decisions"]
        if not isinstance(channels, dict):
            raise ValueError("channel_decisions must be an object")
        result["channel_decisions"] = {key: _decision(row) for key, row in channels.items()}
    return result


def _text(value):
    return html.escape("unknown" if value is None else str(value), quote=True)


def _page(episode, summary, decisions):
    rows = []
    for row in decisions["cycles"]:
        decision = row["decision"] or {}
        rows.append("<tr>" + "".join("<td>" + _text(value) + "</td>" for value in (
            row["cycle"], row["phase"], row["executed"], decision.get("choice"),
            decision.get("selected_probability"), row["rejected_count"])) + "</tr>")
    event_rows = []
    for event in records(episode, "events"):
        event_rows.append("<li>" + _text(event.get("time")) + " · " +
                          _text(event.get("event")) + " — " + _text(event.get("message")) + "</li>")
    # No dynamic HTML, scripts, remote URLs, or unescaped JSON are embedded.
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>embodied-jev episode evidence</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17212b}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:1rem}
.note{padding:1rem;background:#fff5dc}a{color:#174c96}.scroll{overflow-x:auto}</style></head><body>
<h1>embodied-jev decision timeline</h1>
<p>Episode """ + _text(episode.get("id")) + " · Task " + _text(episode.get("task")) +         " · Status " + _text(summary["episode_status"]) +         " · Task success " + _text(summary["task_success"]) + """</p>
<p class="note">This is an archived decision timeline, not simulation video.
Steps are recorded decision cycles. Candidate rejections include configured feasibility checks;
they are not actual contacts. Missing probabilities and token usage are unknown.
Configured forbidden-contact observations do not establish comprehensive safety.</p>
<p><a href="episode.raw.json">Raw episode</a> · <a href="trajectory.json">Trajectory</a> ·
<a href="decisions.json">Decisions</a> · <a href="events.json">Events</a> ·
<a href="summary.json">Summary</a></p>
<div class="scroll"><table><thead><tr><th>Cycle</th><th>Phase</th><th>Executed</th>
<th>Choice</th><th>Reported probability</th><th>Rejected candidates</th></tr></thead><tbody>""" +         "".join(rows) + """</tbody></table></div>
<h2>Evidence summary</h2><pre>""" + html.escape(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), quote=True) +         "</pre><h2>Events</h2><ol>" + "".join(event_rows) + "</ol></body></html>"


def write_artifacts(output: Path, episode: dict, metadata: dict) -> list[dict]:
    """Write lossless raw JSON and derived evidence; metadata must be secret-free.

    Raw preservation means JSON value equivalence to the supplied export, not
    original whitespace preservation. No simulator, policy, or remote API runs.
    """
    summary = summarize(episode, metadata)
    history, frames = records(episode, "history"), records(episode, "frames")
    decisions = {"schema_version": VERSION, "cycles": [],
                 "last_decision": _decision(episode.get("last_decision")),
                 "last_intent": _decision(episode.get("last_intent")),
                 "last_decision_inputs": copy.deepcopy(episode.get("last_decision_inputs")),
                 "note": "Last decisions may be unexecuted or awaiting intervention."}
    steps = []
    for index, row in enumerate(history):
        did_execute = executed(row, episode.get("control_mode"))
        decisions["cycles"].append({
            "index": index, "cycle": row.get("cycle"), "phase": row.get("phase"),
            "intent": _decision(row.get("intent")), "decision": _decision(row.get("decision")),
            "candidates": copy.deepcopy(row.get("candidates")),
            "decision_inputs": copy.deepcopy(row.get("decision_inputs")),
            "executed": did_execute, "rejected_count": row.get("rejected_count"),
            "rejection": row.get("rejection"),
        })
        after = row.get("after")
        contact = after.get("forbidden_contact") if isinstance(after, dict) else None
        steps.append({
            "t": index, "cycle": row.get("cycle"), "obs": copy.deepcopy(row.get("before")),
            "action": copy.deepcopy(row.get("action")), "next_obs": copy.deepcopy(after),
            "reward": None, "cost": None, "terminated": None, "truncated": None,
            "events": ["configured_forbidden_contact_observed"] if contact is True else [],
            "info": {"phase": row.get("phase"), "label": row.get("label"), "executed": did_execute,
                     "configured_contact_observed": contact,
                     "event_observation_known": type(contact) is bool,
                     "raw_history_index": index},
        })
    final_obs = frames[-1].get("observation") if frames else (history[-1].get("after") if history else None)
    trajectory = {"schema_version": VERSION, "step_unit": "recorded_decision_cycles",
                  "seed": episode.get("seed"), "task": episode.get("task"), "steps": steps,
                  "final_obs": copy.deepcopy(final_obs), "frames": copy.deepcopy(frames),
                  "summary": {key: summary[key] for key in (
                      "episode_status", "task_success", "steps", "step_unit", "frame_count")},
                  "note": "Action segments may be rejected. Physics substeps are not reconstructed. Rewards and costs are unmeasured."}
    documents = {
        "episode.raw.json": episode,
        "trajectory.json": trajectory,
        "decisions.json": decisions,
        "events.json": {"schema_version": VERSION, "events": copy.deepcopy(records(episode, "events"))},
        "run_metadata.json": metadata,
        "summary.json": summary,
    }
    # Validate every JSON value and render external text before writing anything.
    serialized = {name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                  for name, value in documents.items()}
    page = _page(episode, summary, decisions)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, content in {**serialized, "visualization.html": page}.items():
        target = output / name
        if target.resolve().parent != output:
            raise ValueError("Artifact paths must stay in their output directory")
        target.write_text(content, encoding="utf-8")
    return [{"path": name, "kind": "trajectory" if name == "trajectory.json" else "evidence",
             "media_type": "application/json", "label": name} for name in documents] + [
        {"path": "visualization.html", "kind": "visualization", "media_type": "text/html",
         "label": "Decision timeline"}]

