"""Episode model/environment must select the same real checkpoint on a new store."""
import json

import pytest

from wms.engines.tdmpc2.native import episode_checkpoint


def test_runtime_default_is_relative_to_runtime_and_shared_between_components(tmp_path):
    machine = tmp_path / "machine"
    machine.mkdir()
    checkpoint = machine / "model.pt"
    checkpoint.write_bytes(b"path fixture, not weights")
    runtime = machine / "runtime.json"
    runtime.write_text(json.dumps({"episode_checkpoint": "model.pt"}))
    settings = {"runtime_config": str(runtime), "source_run": "", "output_dir": str(tmp_path / "new-store")}
    assert episode_checkpoint(settings) == checkpoint
    assert episode_checkpoint(settings, {"episode_checkpoint": "model.pt"}) == checkpoint


def test_explicit_selection_wins_and_missing_explicit_file_does_not_fall_back(tmp_path):
    runtime_default = tmp_path / "default.pt"
    runtime_default.touch()
    selected = tmp_path / "explicit.pt"
    selected.touch()
    runtime = {"episode_checkpoint": str(runtime_default)}
    assert episode_checkpoint({"checkpoint": str(selected)}, runtime) == selected
    with pytest.raises(FileNotFoundError):
        episode_checkpoint({"checkpoint": str(tmp_path / "missing.pt")}, runtime)
    old_run = tmp_path / "runs/run_aa/output"
    old_run.mkdir(parents=True)
    (old_run / "model.pt").touch()
    assert episode_checkpoint({"source_run": "run_aa", "output_dir": str(tmp_path / "runs/run_bb/output")}, runtime) == old_run / "model.pt"


def test_missing_selection_remains_an_error():
    with pytest.raises(ValueError, match="checkpoint"):
        episode_checkpoint({}, {})
