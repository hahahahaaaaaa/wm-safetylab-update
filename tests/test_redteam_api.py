import json
from pathlib import Path
import sys
import time

import pytest
from fastapi.testclient import TestClient

from wms.app.api import create_app
from wms.redteam import stats

TOY_PYTHON = Path("K:/world-model/envs/wms-onboarding/Scripts/python.exe")
ATTACK, GATE = "redteam_fixtures:NudgeAttack", "redteam_fixtures:TruthGate"


def config(**overrides):
    value = {"name": "rt-test", "model": "toy-planner", "environments": ["toy-pick"], "seeds": {"start": 0, "episodes": 10},
             "plugins": [{"key": "m", "plugin": "monitor-plan-summary"},
                         {"key": "f", "plugin": "example-action-scale"}]}
    value.update(overrides)
    return value


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(storage_root=tmp_path / "store", worker_python=sys.executable)) as value:
        yield value


def validate(client, **overrides):
    return client.post("/api/v1/experiments/validate", json=config(**overrides)).json()


def test_catalog_comes_from_the_registry(client):
    catalog = client.get("/api/v1/catalog").json()
    models = {m["id"]: m for m in catalog["models"]}
    assert models["toy-planner"]["probes"] == ["state", "predict", "plan", "embed"]
    assert models["openvla-bridge"]["probes"] == []
    assert models["maniskill-planner"]["kind"] == "planner"
    envs = {e["id"]: e for e in catalog["environments"]}
    assert "image_box" in envs["simpler-bridge-carrot"]["capabilities"]
    assert envs["maniskill-pushcube-state"]["observation"] == "vector"
    plugins = {p["id"]: p for p in catalog["plugins"]}
    assert (plugins["monitor-plan-summary"]["stage"], plugins["monitor-plan-summary"]["role"]) == ("S4", "monitor")
    assert plugins["monitor-surprise"]["stage"] == "S6"
    assert [s["id"] for s in catalog["stages"]] == ["ENV", "S1", "S2", "S3", "S4", "S5", "S6"]
    assert {p["id"] for p in catalog["planned"]} == {"T1", "T2", "T3", "T4"}


def test_arms_baseline_singles_combined(client):
    result = validate(client, plugins=[{"key": "m", "plugin": "monitor-plan-summary"},
                                       {"key": "f", "plugin": "example-action-scale"},
                                       {"key": "a", "entrypoint": ATTACK, "threat": {"budget": "0.1 per step"}},
                                       {"key": "g", "entrypoint": GATE, "config": {"oracle": True}}])
    assert result["valid"], result["issues"]
    arms = [(a["id"], a["kind"], a["plugins"], a["oracle"]) for a in result["arms"]]
    assert arms == [("A0", "baseline", ["m"], False), ("A1", "single", ["f", "m"], False),
                    ("A2", "single", ["a", "m"], False), ("A3", "combined", ["f", "a", "m"], False),
                    ("A4", "defended", ["f", "a", "g", "m"], True), ("A5", "defense_only", ["g", "m"], True)]
    assert result["estimate"] == {"runs": 6, "episodes": 60, "minutes": 1.0}
    recipe = result["recipes"][4]
    assert recipe["model"] == "toy-planner" and recipe["seed"] == 0 and recipe["params"]["episodes"] == 10
    assert [p["id"] for p in recipe["params"]["plugins"]] == ["f", "a", "g", "m"]
    assert recipe["params"]["plugins"][1]["config"]["threat"]["budget"] == "0.1 per step"


def test_missing_probe_and_ability_grey_out_plugins(client):
    result = validate(client, model="toy-centroid")
    check = {c["key"]: c for c in result["plugins"]}
    assert not check["m"]["available"] and "需要探针 plan" in check["m"]["reasons"][0]
    moves = {"key": "t", "plugin": "target-moves", "config": {"moves": [{"step": 3, "goal": {"delta": [0.1, 0]}}]}}
    result = validate(client, model="tdmpc2-pushcube", environments=["maniskill-pushcube-state"],
                      plugins=[moves, {"key": "i", "plugin": "image-perturbation",
                                       "config": {"schedule": [{"type": "occlude", "start": 1, "end": 3, "box": "object:cube"}]}}])
    check = {c["key"]: c for c in result["plugins"]}
    assert check["t"]["available"] and check["t"]["requires"] == ["set_goal"]
    assert check["i"]["privileged"] and not check["i"]["available"]
    assert any("image_box" in r for r in check["i"]["reasons"])
    assert any(i["step"] == "model" for i in result["issues"])        # no TD-MPC2 machine configuration here


def test_threat_card_and_oracle_are_required(client):
    result = validate(client, plugins=[{"key": "a", "entrypoint": ATTACK}, {"key": "g", "entrypoint": GATE}])
    threat = [i for i in result["issues"] if i["step"] == "threat"]
    assert not result["valid"]
    assert {i["plugin"] for i in threat} == {"a", "g"}
    assert any("budget" in i["message"] for i in threat if i["plugin"] == "a")


def test_drafts_round_trip_and_validation_errors_block_submit(client):
    saved = client.put("/api/v1/experiments/drafts/d1", json=config()).json()
    assert client.get("/api/v1/experiments/drafts/d1").json()["config"] == saved["config"]
    assert client.get("/api/v1/experiments/drafts/missing").status_code == 404
    response = client.post("/api/v1/experiments", json=config(name=""))
    assert response.status_code == 400 and "实验名称" in response.json()["detail"]


def test_wilson_and_paired_bootstrap():
    rate, low, high = stats.wilson(8, 10)
    assert rate == 0.8 and 0.49 < low < 0.5 and 0.94 < high < 0.95
    mean, low, high = stats.paired_bootstrap([-1.0] * 5 + [0.0] * 5)
    assert mean == -0.5 and low < mean < high
    assert stats.paired_bootstrap([]) is None


@pytest.mark.skipif(not TOY_PYTHON.is_file(), reason="needs the toy runtime (native/toy/runtime.json)")
def test_toy_experiment_end_to_end(client):
    experiment = client.post("/api/v1/experiments", json=config(seeds={"start": 0, "episodes": 3})).json()
    deadline = time.time() + 300
    while experiment["state"] in ("queued", "running") and time.time() < deadline:
        time.sleep(0.5)
        experiment = client.get(f"/api/v1/experiments/{experiment['id']}").json()
    assert experiment["state"] == "completed", experiment["runs"]
    assert experiment["episodes_done"] == experiment["episodes_total"] == 6
    base = f"/api/v1/experiments/{experiment['id']}"
    report = client.get(base + "/report").json()
    assert [a["arm"] for a in report["arms"]] == ["A0", "A1"] and report["complete"]
    assert report["arms"][1]["delta_success"]["pairs"] == 3
    page = client.get(base + "/episodes", params={"arm": "A1"}).json()
    assert page["total"] == 3
    detail = client.get(f"{base}/episodes/{page['items'][0]['id']}").json()
    assert detail["baseline"] and detail["baseline"].endswith(":0")
    trace = client.get(f"{base}/episodes/{page['items'][0]['id']}/trace", params={"start": 0, "end": 2}).json()
    assert len(trace["steps"]) == 2 and trace["steps"][0]["plugins"]["S5"] == ["f"]
    frame = client.get(f"{base}/episodes/{page['items'][0]['id']}/frames/top/0.png")
    assert frame.status_code == 200 and frame.content.startswith(b"\x89PNG")
    matrix = client.get(base + "/matrix").json()
    assert matrix["rows"][1]["cells"]["S5"]["plugins"] == ["example-action-scale"]
    events = client.get(base + "/events").text
    assert "event: state" in events and json.loads(events.split("data: ")[1].split("\n")[0])["data"]["state"] == "completed"
