"""Cancelled native runs retain only verified, explicitly returned evidence."""
import json
from pathlib import Path

import pytest

from wms.runners.unified import UnifiedWorker


def _cancel_with_artifact(config, output, context):
    context.write_json("episode.raw.json", {"status": "cancelled", "frames": [{"time": 0.5}]})
    (output / "cancel.flag").touch()
    return {"status": "cancelled", "outputs": [{"kind": "evidence", "path": "episode.raw.json"}],
            "data": {"episode_status": "cancelled", "worker_joined": True}}


def _raise_cancelled(config, output, context):
    context.write_json("partial.json", {"steps": 1})
    (output / "cancel.flag").touch()
    context.check_cancelled()


def _completed_with_cancel_flag(config, output, context):
    context.write_json("partial.json", {"steps": 1})
    (output / "cancel.flag").touch()
    return {"status": "completed", "outputs": [{"path": "partial.json"}]}


def _invalid_cancelled_artifact(config, output, context):
    external = output.parent / "outside.json"
    external.write_text("{}", encoding="utf-8")
    (output / "cancel.flag").touch()
    return {"status": "cancelled", "outputs": [{"path": config["artifact"]}]}


def _run(tmp_path, entry, **extra):
    output = tmp_path / "output"
    worker = UnifiedWorker({"task": "native", "entrypoint": f"{__name__}:{entry}", **extra}, output)
    code = worker.run()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    return code, result, output


def test_native_cancelled_return_keeps_evidence_and_cancelled_exit(tmp_path):
    code, result, output = _run(tmp_path, "_cancel_with_artifact")
    assert code == 130
    assert result["status"] == "cancelled"
    assert result["data"]["episode_status"] == "cancelled"
    assert result["outputs"] == [{"kind": "evidence", "path": "episode.raw.json", "media_type": "application/json"}]
    assert json.loads((output / result["outputs"][0]["path"]).read_text())["frames"] == [{"time": 0.5}]


def test_raised_cancellation_still_uses_existing_empty_result(tmp_path):
    code, result, output = _run(tmp_path, "_raise_cancelled")
    assert code == 130 and result["status"] == "cancelled"
    assert result["outputs"] == []
    assert result["error"] == "Run cancelled"
    assert (output / "partial.json").exists()


def test_cancel_flag_still_overrides_normal_completion(tmp_path):
    code, result, _ = _run(tmp_path, "_completed_with_cancel_flag")
    assert code == 130 and result["status"] == "cancelled"
    assert result["outputs"] == []


@pytest.mark.parametrize("artifact", ["missing.json", "../outside.json"])
def test_cancelled_native_does_not_bypass_artifact_validation(tmp_path, artifact):
    code, result, _ = _run(tmp_path, "_invalid_cancelled_artifact", artifact=artifact)
    assert code == 1 and result["status"] == "failed"
    assert result["outputs"] == []
    assert "FileNotFoundError" in result["error"] or "ValueError" in result["error"]