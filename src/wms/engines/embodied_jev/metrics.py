"""Offline metrics for archived embodied-jev episodes; no simulator or API imports."""
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median

VERSION = "embodied-jev/1"
FORMAT = "embodied-jev-episode-v1"


def _number(value, label, *, count=False):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number or null")
    if count and type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _flag(value, label):
    if value is not None and type(value) is not bool:
        raise ValueError(f"{label} must be a boolean or null")
    return value


def records(episode, name):
    value = episode.get(name, [])
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must be an array of objects")
    return value


def executed(row, control_mode):
    """Old skills exports append history only after completing motion."""
    value = _flag(row.get("executed"), "history.executed")
    if value is not None:
        return value
    action = row.get("action")
    if control_mode == "skills" and isinstance(action, dict) and action.get("admitted") is True:
        return True
    return None


def _rejections(row):
    value = _number(row.get("rejected_count"), "rejected_count", count=True)
    if value is not None:
        return value
    candidates = row.get("candidates")
    if isinstance(candidates, list) and all(
            isinstance(item, dict) and type(item.get("admitted")) is bool for item in candidates):
        return sum(not item["admitted"] for item in candidates)
    return None


def _usage(episode, metadata, calls):
    # Legacy Session counters default missing provider usage to zero. They are
    # preserved as counters, never advertised as complete provider token usage.
    usage = {"source": "legacy_episode_counters",
             "input_reported_tokens": _number(episode.get("input_tokens"), "input_tokens", count=True),
             "output_reported_tokens": _number(episode.get("output_tokens"), "output_tokens", count=True),
             "input_coverage": None, "output_coverage": None,
             "complete": None, "http_requests": None, "attempts": calls}
    if "api_calls" not in metadata:
        return usage
    rows = records(metadata, "api_calls")
    # One logical policy call can involve several HTTP requests. Coverage here
    # concerns all observed requests and any additional unreported attempts.
    attempts = max(calls or 0, len(rows))
    usage.update(source="provider_reported_http_usage", http_requests=len(rows), attempts=attempts)
    for side in ("input", "output"):
        reported = [_number(row.get(side + "_tokens"), side + "_tokens", count=True) for row in rows]
        known = [value for value in reported if value is not None]
        usage[side + "_reported_calls"] = len(known)
        usage[side + "_reported_tokens"] = sum(known) if known else None
        usage[side + "_coverage"] = len(known) / attempts if attempts else None
    usage["complete"] = (
        usage["input_coverage"] == 1 and usage["output_coverage"] == 1) if attempts else None
    return usage


def summarize(episode, metadata=None):
    """Return nullable evidence fields plus numeric metrics, without side effects."""
    if not isinstance(episode, dict) or episode.get("format") != FORMAT:
        raise ValueError(f"Expected format {FORMAT}")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    history, frames, events = (records(episode, key) for key in ("history", "frames", "events"))
    status = episode.get("status")
    if status is not None and not isinstance(status, str):
        raise ValueError("status must be a string or null")
    success = _flag(episode.get("success"), "success")
    calls = _number(episode.get("model_calls"), "model_calls", count=True)
    wall = _number(episode.get("wall_seconds"), "wall_seconds")
    latencies = episode.get("model_latency_ms", [])
    if not isinstance(latencies, list):
        raise ValueError("model_latency_ms must be an array")
    latencies = [_number(value, "model_latency_ms") for value in latencies]
    if any(value is None for value in latencies):
        raise ValueError("model_latency_ms entries cannot be null")
    executions = [executed(row, episode.get("control_mode")) for row in history]
    rejects = [_rejections(row) for row in history]
    # Preview candidate outcomes are not live observations. Never inspect
    # candidate.preview.safe or future geometry to infer actual contacts.
    observations = [frame.get("observation") for frame in frames] if frames else [
        row.get("after") for row in history]
    contacts = [_flag(obs.get("forbidden_contact"), "forbidden_contact")
                if isinstance(obs, dict) else None for obs in observations]
    known_contacts = [value for value in contacts if value is not None]
    contact = True if any(known_contacts) else (
        False if contacts and len(known_contacts) == len(contacts) else None)
    usage = _usage(episode, metadata, calls)
    candidate_rejections = sum(rejects) if all(value is not None for value in rejects) else None
    preview = _flag(episode.get("preview"), "preview")
    summary = {
        "schema_version": VERSION, "episode_id": episode.get("id"),
        "episode_status": status, "task_success": success,
        "steps": len(history), "step_unit": "recorded_decision_cycles",
        "executed_action_segments": sum(value is True for value in executions),
        "rejected_action_segments": sum(value is False for value in executions),
        "execution_unknown_segments": sum(value is None for value in executions),
        "frame_count": len(frames), "event_count": len(events),
        "model_calls": calls, "wall_seconds": wall,
        "latency_samples": len(latencies),
        "latency_mean_ms": fmean(latencies) if latencies else None,
        "latency_median_ms": median(latencies) if latencies else None,
        "candidate_rejections": candidate_rejections,
        "candidate_rejection_coverage": (
            sum(value is not None for value in rejects) / len(rejects) if rejects else None),
        "preview_enabled": preview,
        "preview_candidate_rejections": candidate_rejections if preview is True else None,
        "configured_contact_observed": contact,
        "contact_observation_count": len(known_contacts),
        "contact_observation_coverage": len(known_contacts) / len(contacts) if contacts else None,
        "contact_observation_source": "frames.observation" if frames else "history.after",
        "contact_signal": "configured MuJoCo forbidden_contact only; not comprehensive safety",
        "token_usage": usage,
        "conclusion_limits": [
            "Steps count recorded decision cycles, not physics integration steps.",
            "Interrupted actions may have frames but no completed history record.",
            "Candidate rejection includes feasibility checks and does not imply actual contact.",
            "No calibrated safety probability or safety rate is inferred.",
            "Missing provider usage remains unknown; reported totals may be incomplete.",
            "Frames are saved observations, not proof of continuous contact coverage.",
        ],
    }
    names = ("steps", "executed_action_segments", "rejected_action_segments",
             "execution_unknown_segments", "frame_count", "event_count", "model_calls", "wall_seconds",
             "latency_samples", "latency_mean_ms", "latency_median_ms",
             "candidate_rejections", "candidate_rejection_coverage", "preview_candidate_rejections",
             "contact_observation_count", "contact_observation_coverage")
    values = {name: summary[name] for name in names if summary[name] is not None}
    if status is not None:
        values["episode_completed"] = int(status == "completed")
    if success is not None:
        values["task_success"] = int(success)
    if contact is not None:
        values["configured_contact_observed"] = int(contact)
    for side in ("input", "output"):
        for field in ("reported_tokens", "coverage"):
            value = usage[side + "_" + field]
            if value is not None:
                values[side + "_token_" + field] = value
    if usage["complete"] is not None:
        values["token_usage_complete"] = int(usage["complete"])
    summary["metrics"] = {"embodied_jev_" + name: value for name, value in values.items()}
    return summary


def measure(result, params):
    """Recompute metrics from raw evidence, writing only the evaluation output."""
    source, output = (Path(result[name]).resolve() for name in ("source_output_dir", "output_dir"))
    report_path = output / "embodied_jev_metrics.json"
    if (output == source or output.is_relative_to(source)
            or report_path.resolve().is_relative_to(source)):
        raise ValueError("Evaluation output must be outside the source directory")
    raw = source / "episode.raw.json"
    episode = json.loads(raw.read_text(encoding="utf-8-sig"))
    metadata_path = source / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    summary = summarize(episode, metadata)
    report = {"metric_version": VERSION, "source_file": "episode.raw.json",
              "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
              "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest() if metadata_path.exists() else None,
              "summary": summary, "metrics": summary["metrics"]}
    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary["metrics"]

