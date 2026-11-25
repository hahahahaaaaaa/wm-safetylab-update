import json
from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient
import pytest

from wms.app.api import create_app
from wms.app import model_library
from wms.core.registry import Registry


def test_catalog_never_exposes_runtime_credentials(tmp_path):
    machine = tmp_path / "runtime.json"
    machine.write_text(json.dumps({"python": sys.executable, "env": {"SECRET_TOKEN": "sensitive-marker"}}))
    registry = Registry(native_configs={"cosmos": str(machine)})
    items = model_library.catalog(registry)
    assert "sensitive-marker" not in json.dumps(items)
    assert "runtime" not in next(i for i in items if i["id"] == "native-cosmos")["template"]
    assert next(i for i in items if i["id"] == "native-cosmos")["stages"] == []
    assert next(i for i in items if i["id"] == "toy-episode-occlusion")["stages"] == ['S1', 'S2', 'S5', 'S6']
    assert next(i for i in items if i["id"] == "tdmpc2-pushcube-episode")["template"]["seed"] == 1000
    assert next(i for i in items if i["id"] == "demo-predict")["latest_run"] is None


def test_missing_nested_resources_are_not_marked_ready(tmp_path):
    child = tmp_path / "adapter.json"
    child.write_text(json.dumps({"checkpoint_dir": "missing-checkpoint"}))
    parent = tmp_path / "runtime.json"
    parent.write_text(json.dumps({"config": "adapter.json"}))
    config = {"runtime": {"python": sys.executable, "config_file": str(parent)}}
    assert any("checkpoint_dir" in message for message in model_library.resource_issues(config))
    child.write_text(json.dumps({"config": "runtime.json"}))
    assert not model_library.resource_issues(config)  # cycles are bounded
    child.write_text('[]')
    assert any('无法读取' in message for message in model_library.resource_issues(config))


def test_credentials_checked_using_worker_environment(tmp_path):
    parent = tmp_path / "runtime.json"
    parent.write_text(json.dumps({"profiles": {"chat": {"model": "test", "url": "https://example.com/v1",
                                                      "api_key_env": "TEST_MODEL_API_KEY"}}}))
    config = {"runtime": {"python": sys.executable, "config_file": str(parent),
                          "env": {"TEST_MODEL_API_KEY": "fake-test-credential"}}, "params": {"provider": "chat"}}
    assert not model_library.resource_issues(config)
    config["runtime"]["env"] = {}
    assert model_library.resource_issues(config)


def test_connection_note_clears_after_a_successful_retest(tmp_path, monkeypatch):
    registry = Registry()
    (tmp_path / 'native').mkdir()
    metadata = (model_library.PROJECT_ROOT / 'native/model-library.json').read_text()
    (tmp_path / 'native/model-library.json').write_text(metadata)
    (tmp_path / 'model-library.local.json').write_text(json.dumps({
        'demo-predict': {'availability_note': 'Previous provider quota error'}}))
    monkeypatch.setattr(model_library, 'PROJECT_ROOT', tmp_path)
    run = {'id': 'run_test', 'state': 'failed', 'config': {'workbench_recipe': 'demo-predict'}}
    item = next(i for i in model_library.catalog(registry, [run]) if i['id'] == 'demo-predict')
    assert item['availability_note'] == 'Previous provider quota error'
    run['state'] = 'completed'
    item = next(i for i in model_library.catalog(registry, [run]) if i['id'] == 'demo-predict')
    assert item['availability_note'] == ''


def test_submit_runs_actual_worker_and_keeps_recipe_identity(tmp_path):
    with TestClient(create_app(storage_root=tmp_path / "store", worker_python=sys.executable)) as client:
        assert client.post('/api/v1/model-library/demo-predict/runs', json={"runtime": {}}).status_code == 400
        assert client.post('/api/v1/model-library/demo-predict/runs', json={"params": []}).status_code == 400
        assert client.post('/api/v1/model-library/demo-predict/runs', json={"seed": True}).status_code == 400
        assert client.post('/api/v1/model-library/no-such-model/runs', json={}).status_code == 404
        response = client.post('/api/v1/model-library/demo-predict/runs', json={"seed": 7})
        assert response.status_code == 200, response.text
        run = response.json()
        for _ in range(100):
            run = client.get('/api/v1/runs/' + run['id']).json()
            if run['state'] in {'completed', 'failed', 'cancelled'}: break
            time.sleep(.05)
        assert run['state'] == 'completed', run
        assert run['config']['workbench_recipe'] == 'demo-predict'
        catalog = client.get('/api/v1/model-library').json()['items']
        latest = next(i for i in catalog if i['id'] == 'demo-predict')['latest_run']
        assert latest['id'] == run['id'] and latest['state'] == 'completed'
        public = client.get('/api/v1/model-library/runs').json()['items']
        assert public[0]['id'] == run['id']
        assert public[0]['config']['runtime'] == {}
        for output in run['result']['outputs']:
            assert client.get('/api/v1/runs/' + run['id'] + '/files/' + output['path']).status_code == 200


@pytest.mark.parametrize('value', [True, 0, -1, 1.5, '20', 101])
def test_cosmos_bad_step_budget_rejected(value):
    from wms.engines.cosmos.native import _integer
    with pytest.raises(ValueError): _integer(value, 'steps', 1, 100)
