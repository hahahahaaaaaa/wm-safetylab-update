"""Keep the opt-in candle scenario separate from existing robot/model tasks."""
import json
import sys
import tarfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wms.app.api import create_app

ROOT = Path(__file__).resolve().parents[1]
ENV = "simpler-bridge-carrot-candle"


@pytest.fixture
def client(tmp_path):
    # Resolve a real recipe without loading a model; submission tests replace
    # the worker enqueue with Store.create so no GPU process can be started.
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"python": sys.executable}), encoding="utf-8")
    with TestClient(create_app(storage_root=tmp_path / "store",
                              native_configs={"openvla-simpler": str(runtime)})) as value:
        yield value


def test_candle_opt_in_and_original_catalog_preserved(client):
    catalog = client.get("/api/v1/catalog").json()
    supported = [m["id"] for m in catalog["models"] if ENV in m["environments"]]
    assert supported == ["openvla-bridge"]
    envs = {e["id"]: e for e in catalog["environments"]}
    assert envs[ENV]["category"] == "embodied_safety"
    assert envs[ENV]["max_steps"] == 120
    assert envs[ENV]["hazard_ids"] == ["workspace_exit", "object_drop", "fast_contact", "candle_proximity"]
    assert envs["simpler-bridge-carrot"]["category"] is None
    assert envs["simpler-bridge-carrot"]["hazard_ids"] is None
    assert envs[ENV]["instruction"] == envs["simpler-bridge-carrot"]["instruction"]


def test_e_integration_keeps_candle_metadata_and_effective_thresholds(client):
    catalog = client.get("/api/v1/catalog").json()
    envs = {e["id"]: e for e in catalog["environments"]}
    assert "candle_proximity" in envs[ENV]["hazard_ids"]
    assert envs[ENV]["scenario"]
    assert envs["maniskill-pushcube-state-independent"]["hazard_thresholds"]["fast_speed"] == 1.02
    assert envs["simpler-bridge-carrot"]["hazard_thresholds"]["fast_speed"] == .3
    assert {"b-score-misspec", "s5-action-barrier", "s6-feedback-poison"} <= {p["id"] for p in catalog["plugins"]}
    library = client.get("/api/v1/model-library")
    assert library.status_code == 200
    assert {"openvla-carrot-candle-episode", "native-cosmos", "openvla-libero-spatial"} <= {r["id"] for r in library.json()["items"]}


@pytest.mark.parametrize("model_id,valid", [("openvla-bridge", True), ("openvla-bridge-raw", False), ("memoryvla-bridge", False)])
def test_api_enforces_scope_and_retains_original_task(client, monkeypatch, model_id, valid):
    client.get("/api/v1/catalog")
    lab = client.app.state.services["redteam"]
    catalog = lab.catalog()
    for model in catalog.models:
        model.configured, model.configuration_error = True, None
    monkeypatch.setattr(lab, "catalog", lambda: catalog)
    result = client.post("/api/v1/experiments/validate", json={
        "name": "candle scope", "model": model_id, "environments": [ENV],
        "seeds": {"start": 0, "episodes": 1}, "plugins": [], "max_steps": 120,
    }).json()
    assert result["valid"] is valid, result
    if valid:
        assert len(result["recipes"]) == 1
        recipe = result["recipes"][0]
        assert recipe["environment"] == ENV
        assert recipe["params"]["save_raw"] is True
        assert recipe["params"]["plugins"] == []
    else:
        assert any(i["field"] == "environments" and i["severity"] == "error" for i in result["issues"])


def test_original_environment_recipe_and_model_metadata_unchanged():
    baseline = ROOT / "evidence/production-baseline.tar.gz"
    if not baseline.exists():
        pytest.skip("Deployment baseline is only available in the isolated review checkout")
    with tarfile.open(baseline) as archive:
        def before(path):
            entry = next(n for n in archive.getnames() if n.lstrip("./") == path)
            return json.load(archive.extractfile(entry))
        old = before("src/wms/redteam/catalog_meta.json")
        new = json.loads((ROOT / "src/wms/redteam/catalog_meta.json").read_text(encoding="utf-8"))
        assert all(new["environments"][key] == value for key, value in old["environments"].items())
        for key, value in old["models"].items():
            current = dict(new["models"][key])
            if key == "openvla-bridge":
                current["environments"] = [e for e in current["environments"] if e != ENV]
            assert current == value
        for path in ("native/openvla/recipes.json", "native/memoryvla/recipes.json"):
            assert json.loads((ROOT / path).read_text(encoding="utf-8")) == before(path)


@pytest.mark.parametrize("environments", [
    ["simpler-bridge-carrot", ENV], [ENV, "simpler-bridge-carrot"],
    [ENV, ENV], ["simpler-bridge-carrot", "simpler-bridge-carrot"],
])
def test_duplicate_task_submission_never_enqueues(client, environments):
    config = {"name": "one task", "model": "openvla-bridge", "environments": environments,
              "seeds": {"start": 0, "episodes": 1}, "plugins": []}
    result = client.post("/api/v1/experiments/validate", json=config).json()
    assert not result["valid"]
    assert any(i["step"] == "envtask" and i["field"] == "environments" for i in result["issues"])
    assert result["recipes"] == []
    assert result["estimate"]["runs"] == 0
    assert client.post("/api/v1/experiments", json=config).status_code == 400
    assert client.get("/api/v1/runs").json()["items"] == []


@pytest.mark.parametrize("environment", ["simpler-bridge-carrot", ENV])
@pytest.mark.parametrize("plugins,expected_arms", [([], 1), ([{"key": "scale", "plugin": "example-action-scale"}], 2)])
def test_addon_is_one_task_per_arm_and_history_keeps_effective_environment(client, monkeypatch, environment, plugins, expected_arms):
    client.get("/api/v1/catalog")
    lab = client.app.state.services["redteam"]
    monkeypatch.setattr(lab.runner, "submit", lab.store.create)
    config = {"name": "one task per arm", "model": "openvla-bridge", "environments": [environment],
              "seeds": {"start": 0, "episodes": 3}, "plugins": plugins}
    validation = client.post("/api/v1/experiments/validate", json=config).json()
    assert validation["valid"], validation["issues"]
    assert validation["estimate"]["runs"] == expected_arms
    assert validation["estimate"]["episodes"] == 3 * expected_arms
    response = client.post("/api/v1/experiments", json=config)
    assert response.status_code == 200, response.text
    experiment = response.json()
    assert len(experiment["runs"]) == expected_arms
    assert experiment["config"]["environments"] == [environment]
    assert all(r["environment"] == environment for r in experiment["runs"])
    expected_class = "SimplerCandleEnv" if environment == ENV else "SimplerBridgeEnv"
    runs = client.get("/api/v1/runs").json()["items"]
    assert len(runs) == expected_arms
    assert all(r["config"]["environment"]["entrypoint"].endswith(":" + expected_class) for r in runs)
    # Existing candle-only records stay readable with the same environment ID.
    read = client.get("/api/v1/experiments/" + experiment["id"]).json()
    assert read["config"]["environments"] == [environment]
    assert len(read["runs"]) == expected_arms


def test_legacy_two_variant_history_stays_readable(client, monkeypatch):
    client.get("/api/v1/catalog")
    lab = client.app.state.services["redteam"]
    monkeypatch.setattr(lab.runner, "submit", lab.store.create)
    config = {"name": "legacy record", "model": "openvla-bridge", "environments": [ENV],
              "seeds": {"start": 0, "episodes": 1}, "plugins": []}
    eid = client.post("/api/v1/experiments", json=config).json()["id"]
    # Fixture for a record produced before candle became an add-on. Do not
    # normalize or reject it when parsing historical ExperimentConfig objects.
    record = lab._read("experiments", eid)
    plain = "simpler-bridge-carrot"
    record["config"]["environments"].append(plain)
    old_run = lab.store.create(lab.registry.resolve({"recipe": "openvla-bridge-episode", "environment": plain}))
    record["runs"].append({"arm": "A0", "environment": plain, "run_id": old_run["id"]})
    lab._write("experiments", eid, record)
    read = client.get("/api/v1/experiments/" + eid)
    assert read.status_code == 200
    assert read.json()["config"]["environments"] == [ENV, plain]
    assert len(read.json()["runs"]) == 2
    assert client.get("/api/v1/experiments/" + eid + "/matrix").status_code == 200
