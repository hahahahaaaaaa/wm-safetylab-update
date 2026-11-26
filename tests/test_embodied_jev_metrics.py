"""Evidence/metric contracts only: no MuJoCo, renderer, model or remote API."""
import copy
import hashlib
import json

import pytest

from wms.engines.embodied_jev.artifacts import write_artifacts
from wms.engines.embodied_jev.metrics import VERSION, measure, summarize


def episode():
    return {
        "format": "embodied-jev-episode-v1", "id": "fixture", "task": "transfer", "seed": 2,
        "provider": "chat", "control_mode": "hierarchical", "preview": True,
        "status": "stalled", "success": False, "model_calls": 3,
        "input_tokens": 0, "output_tokens": 0, "wall_seconds": 2.5,
        "model_latency_ms": [100, 300],
        "history": [
            {"cycle": 1, "phase": "move", "executed": True,
             "action": {"id": "left", "admitted": True},
             "decision": {"choice": "left", "probabilities": {}, "selected_probability": None},
             "before": {"object": [0, 0, 0]}, "after": {"object": [1, 0, 0]},
             "rejected_count": 2},
            {"cycle": 2, "phase": "move", "executed": False, "rejection": "preview refused",
             "action": {"id": "down", "admitted": False, "preview": {"safe": False}},
             "decision": {"choice": "down", "selected_probability": None},
             "before": {"object": [1, 0, 0]}, "after": {"object": [1, 0, 0]},
             "rejected_count": 1},
        ],
        "frames": [{"time": 0, "observation": {"forbidden_contact": False}},
                   {"time": 1, "observation": {"forbidden_contact": False}}],
        "events": [{"event": "stalled", "message": "repeated command", "cycle": 2}],
    }


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hashes(path):
    return {file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob("*") if file.is_file()}


def evaluate(tmp_path, data=None, metadata=None):
    source = tmp_path / "source"
    write_artifacts(source, data or episode(), metadata or {})
    return measure({"source_output_dir": str(source), "output_dir": str(tmp_path / "evaluation")}, {})


def test_archives_raw_data_and_distinguishes_decisions_execution_and_contacts(tmp_path):
    data = episode()
    original = copy.deepcopy(data)
    outputs = write_artifacts(tmp_path / "source", data, {"source_revision": "fixture"})
    assert data == original == read(tmp_path / "source" / "episode.raw.json")
    assert {item["path"] for item in outputs} == {
        "episode.raw.json", "trajectory.json", "decisions.json", "events.json",
        "run_metadata.json", "summary.json", "visualization.html"}
    trajectory = read(tmp_path / "source" / "trajectory.json")
    assert trajectory["step_unit"] == "recorded_decision_cycles"
    assert len(trajectory["steps"]) == 2
    assert trajectory["steps"][1]["info"]["executed"] is False
    assert all(row["reward"] is None and row["cost"] is None for row in trajectory["steps"])
    decision = read(tmp_path / "source" / "decisions.json")["cycles"][0]["decision"]
    assert decision["probabilities"] is None and decision["selected_probability"] is None
    values = evaluate(tmp_path, data)
    assert values["embodied_jev_steps"] == 2
    assert values["embodied_jev_executed_action_segments"] == 1
    assert values["embodied_jev_rejected_action_segments"] == 1
    assert values["embodied_jev_preview_candidate_rejections"] == 3
    assert values["embodied_jev_configured_contact_observed"] == 0
    assert values["embodied_jev_episode_completed"] == 0
    assert values["embodied_jev_task_success"] == 0
    assert values["embodied_jev_latency_mean_ms"] == 200
    assert not any("safety_rate" in key or "collision_count" in key for key in values)


def test_unknown_safety_tokens_and_probabilities_are_not_fabricated(tmp_path):
    data = episode()
    data["frames"] = [{"observation": {}}]
    data["model_latency_ms"] = []
    values = evaluate(tmp_path, data)
    assert "embodied_jev_configured_contact_observed" not in values
    assert "embodied_jev_input_token_coverage" not in values
    assert "embodied_jev_token_usage_complete" not in values
    assert "embodied_jev_latency_mean_ms" not in values
    summary = read(tmp_path / "source" / "summary.json")
    assert summary["configured_contact_observed"] is None
    assert summary["token_usage"]["input_coverage"] is None
    assert summary["token_usage"]["complete"] is None


def test_provider_usage_has_explicit_partial_coverage(tmp_path):
    metadata = {"api_calls": [
        {"input_tokens": 20, "output_tokens": 3},
        {"input_tokens": 10, "output_tokens": None}]}
    values = evaluate(tmp_path, metadata=metadata)
    assert values["embodied_jev_input_token_reported_tokens"] == 30
    assert values["embodied_jev_output_token_reported_tokens"] == 3
    assert values["embodied_jev_input_token_coverage"] == pytest.approx(2 / 3)
    assert values["embodied_jev_output_token_coverage"] == pytest.approx(1 / 3)
    assert values["embodied_jev_token_usage_complete"] == 0


def test_missing_usage_for_all_requests_is_unknown_total(tmp_path):
    values = evaluate(tmp_path, metadata={"api_calls": [{}, {}, {}]})
    assert values["embodied_jev_input_token_coverage"] == 0
    assert "embodied_jev_input_token_reported_tokens" not in values


def test_actual_contact_is_not_hidden_by_preview(tmp_path):
    data = episode()
    data["frames"][1]["observation"]["forbidden_contact"] = True
    values = evaluate(tmp_path, data)
    assert values["embodied_jev_configured_contact_observed"] == 1
    assert values["embodied_jev_preview_candidate_rejections"] == 3


def test_incomplete_contact_observations_remain_unknown_unless_positive():
    data = episode()
    data["frames"][1]["observation"] = {}
    assert summarize(data)["configured_contact_observed"] is None
    data["frames"][0]["observation"]["forbidden_contact"] = True
    assert summarize(data)["configured_contact_observed"] is True


def test_old_skills_export_infers_completed_segments_without_model_calls():
    data = episode()
    data.update(control_mode="skills", model_calls=0, model_latency_ms=[])
    data["history"] = [data["history"][0]]
    del data["history"][0]["executed"]
    summary = summarize(data, {"api_calls": []})
    assert summary["executed_action_segments"] == 1
    assert summary["token_usage"]["input_coverage"] is None
    assert summary["token_usage"]["complete"] is None


def test_metric_recomputation_ignores_derived_summary_and_preserves_sources(tmp_path):
    source, output = tmp_path / "source", tmp_path / "evaluation"
    write_artifacts(source, episode(), {})
    (source / "summary.json").write_text('{"metrics":{"fake":999}}', encoding="utf-8")
    before = hashes(source)
    request = {"source_output_dir": str(source), "output_dir": str(output)}
    first = measure(request, {})
    assert first == measure(request, {})
    assert hashes(source) == before
    assert "fake" not in first
    report = read(output / "embodied_jev_metrics.json")
    assert report["metric_version"] == VERSION
    assert report["source_sha256"] == before["episode.raw.json"]


@pytest.mark.parametrize("relative", [".", "child"])
def test_evaluation_refuses_source_or_child_directory(tmp_path, relative):
    source = tmp_path / "source"
    write_artifacts(source, episode(), {})
    before = hashes(source)
    with pytest.raises(ValueError, match="outside"):
        measure({"source_output_dir": str(source), "output_dir": str(source / relative)}, {})
    assert hashes(source) == before


def test_missing_raw_episode_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        measure({"source_output_dir": str(tmp_path / "empty"), "output_dir": str(tmp_path / "eval")}, {})


@pytest.mark.parametrize("field,value", [
    ("success", 1), ("model_calls", -1), ("wall_seconds", float("nan")),
    ("model_latency_ms", [float("inf")]), ("history", {}), ("format", "unknown")])
def test_malformed_evidence_is_rejected_before_artifact_write(tmp_path, field, value):
    data = episode()
    data[field] = value
    with pytest.raises(ValueError):
        write_artifacts(tmp_path / "source", data, {})
    assert not (tmp_path / "source").exists()


def test_html_escapes_external_text_and_preserves_raw(tmp_path):
    data = episode()
    attack = '<script>alert("x")</script><img src=x onerror=alert(1)>'
    data["id"] = attack
    data["history"][0]["decision"]["choice"] = attack
    data["events"][0]["message"] = attack
    write_artifacts(tmp_path, data, {})
    page = (tmp_path / "visualization.html").read_text(encoding="utf-8")
    assert attack not in page and "<script>" not in page
    assert "&lt;script&gt;" in page and "Content-Security-Policy" in page
    assert read(tmp_path / "episode.raw.json")["id"] == attack


def test_nested_channel_probabilities_are_null_when_missing(tmp_path):
    data = episode()
    data["last_decision"] = {"channel_decisions": {"x": {"choice": "hold"}}}
    write_artifacts(tmp_path, data, {})
    channel = read(tmp_path / "decisions.json")["last_decision"]["channel_decisions"]["x"]
    assert channel["probabilities"] is None and channel["selected_probability"] is None



def test_unknown_task_success_and_metadata_type_are_not_coerced():
    data = episode()
    data["success"] = None
    assert "embodied_jev_task_success" not in summarize(data)["metrics"]
    with pytest.raises(ValueError, match="metadata"):
        summarize(data, [])
