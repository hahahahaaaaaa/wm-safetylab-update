"""Episode kernel, perturbations and A/B parity on the CPU toy environment."""
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from wms.core.parity import compare, episode_episodes
from wms.core.registry import Registry
from wms.perturb import Perturbations, blur, occlude
from wms.runners.unified import UnifiedWorker

TOY_ENV = {"entrypoint": "wms.scenarios.toy.env:ToyPickEnv", "config": {"image_size": 64, "max_steps": 60}}
TOY_POLICY = {"entrypoint": "wms.scenarios.toy.policy:CentroidPickPolicy", "config": {"memory": True}}
OCCLUDE = {"type": "occlude", "start": 4, "end": 12, "box": "object:cube", "margin": 0.04}
MOVE = {"step": 6, "object": "cube", "delta": [0.15, 0.0]}


def _run(tmp_path, name="output", **params):
    output = tmp_path / name
    config = {"task": "native", "entrypoint": "wms.episodes.kernel:run", "seed": 0,
              "model": TOY_POLICY, "environment": TOY_ENV,
              "params": {"episodes": 2, "video": False, **params}}
    code = UnifiedWorker(config, output).run()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    return code, result, output


def _steps(output, episode=0):
    path = output / "episodes" / f"ep_{episode:03d}" / "steps.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_perturbations_change_only_the_scheduled_frames():
    image = np.full((8, 8, 3), 200, dtype=np.uint8)
    hidden = occlude(image, [0.25, 0.25, 0.75, 0.75])
    assert hidden[4, 4].tolist() == [0, 0, 0] and hidden[0, 0].tolist() == [200, 200, 200]
    assert blur(image, 2.0).shape == image.shape
    schedule = Perturbations([{"type": "drop", "start": 2, "end": 3}, {"type": "noise", "start": 0, "end": 9, "std": 5}],
                             seed=7)
    obs = {"rgb": {"cam": image}}
    seen, records = schedule.apply(obs, 2)
    assert [r["type"] for r in records] == ["drop", "noise"] and seen["rgb"]["cam"].max() <= 15
    assert obs["rgb"]["cam"] is image and image.min() == 200          # the input is never modified
    schedule.reset(7)
    first = schedule.apply(obs, 0)[0]["rgb"]["cam"]
    schedule.reset(7)
    assert np.array_equal(first, schedule.apply(obs, 0)[0]["rgb"]["cam"])
    with pytest.raises(ValueError, match="type must be one of"):
        Perturbations([{"type": "smudge"}])


def test_kernel_logs_aligned_steps_perturbations_and_moves(tmp_path):
    code, result, output = _run(tmp_path, perturbations=[OCCLUDE], target_moves=[MOVE])
    assert code == 0, result.get("error")
    data = result["data"]
    assert data["path"] == "episode" and data["episodes"] == 2 and data["seeds"] == [0, 1]
    summary = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    for row in summary["episodes"]:
        steps = _steps(output, row["episode"])
        frames = np.load(output / "episodes" / f"ep_{row['episode']:03d}" / "frames_top.npz")["seen"]
        assert len(frames) == len(steps) + 1 == row["steps"] + 1
        occluded = [s["t"] for s in steps if s["perturbations"]]
        assert occluded == [t for t in range(4, 12) if t < len(steps)]
        assert [s["t"] for s in steps if s["target_moves"]] == [6]
        moved = steps[6]["target_moves"][0]
        assert moved["state_after"]["cube"][0] == pytest.approx(min(1.0, moved["state_before"]["cube"][0] + 0.15))
        assert "estimate_cube" in steps[0]["policy_info"]
    assert any(item["path"] == "run_manifest.json" for item in result["outputs"])


def test_kernel_is_deterministic_for_a_seed(tmp_path):
    _, _, first = _run(tmp_path, "a", perturbations=[OCCLUDE, {"type": "noise", "start": 0, "end": 60, "std": 4}])
    _, _, second = _run(tmp_path, "b", perturbations=[OCCLUDE, {"type": "noise", "start": 0, "end": 60, "std": 4}])
    assert (first / "episodes/ep_001/steps.jsonl").read_bytes() == (second / "episodes/ep_001/steps.jsonl").read_bytes()


class NoStateEnv:
    def __init__(self, config): ...
    def reset(self, seed, inputs): return {}
    def step(self, action): return {}
    def close(self): ...


class TerminatingPolicy:
    def __init__(self, config):
        self.info = {}

    def reset(self, seed):
        self.t = 0

    def act(self, observation, instruction=None):
        self.t += 1
        self.info = {"terminate_episode": self.t >= 3}
        return [0.0, 0.0, -1.0]


def test_kernel_rejects_components_without_the_episode_protocol(tmp_path):
    output = tmp_path / "output"
    config = {"task": "native", "entrypoint": "wms.episodes.kernel:run", "model": TOY_POLICY,
              "environment": {"entrypoint": f"{__name__}:NoStateEnv", "config": {}}, "params": {}}
    assert UnifiedWorker(config, output).run() == 1
    error = str(json.loads((output / "result.json").read_text(encoding="utf-8"))["error"])
    assert "lacks ['spec', 'observe', 'get_state']" in error


def test_policy_termination_ends_the_episode_after_that_step(tmp_path):
    output = tmp_path / "output"
    config = {"task": "native", "entrypoint": "wms.episodes.kernel:run",
              "model": {"entrypoint": f"{__name__}:TerminatingPolicy", "config": {}}, "environment": TOY_ENV,
              "params": {"episodes": 1, "video": False}}
    assert UnifiedWorker(config, output).run() == 0
    row = json.loads((output / "episodes.json").read_text(encoding="utf-8"))["episodes"][0]
    assert row["steps"] == 3 and row["end_reason"] == "policy_terminated"
    no_stop = tmp_path / "no_stop"
    config["params"]["stop_on_terminate"] = False
    assert UnifiedWorker(config, no_stop).run() == 0
    assert json.loads((no_stop / "episodes.json").read_text(encoding="utf-8"))["episodes"][0]["steps"] == 60


CALLS = []


class OrderEnv:
    def __init__(self, config): ...
    def reset(self, seed, inputs): CALLS.append("env"); return {}
    def step(self, action): return {"obs": {}, "truncated": True}
    def close(self): ...
    def spec(self): return {"max_steps": 1}
    def observe(self): return {}
    def get_state(self): return {}


class OrderPolicy:
    def __init__(self, config):
        self.info = {}

    def reset(self, seed):
        CALLS.append("policy")

    def act(self, observation, instruction=None):
        return [0.0]


@pytest.mark.parametrize("order, expected", [("env_first", ["env", "policy"]), ("policy_first", ["policy", "env"])])
def test_reset_order(tmp_path, order, expected):
    CALLS.clear()
    config = {"task": "native", "entrypoint": "wms.episodes.kernel:run",
              "model": {"entrypoint": f"{__name__}:OrderPolicy", "config": {}},
              "environment": {"entrypoint": f"{__name__}:OrderEnv", "config": {}},
              "params": {"episodes": 1, "video": False, "reset_order": order}}
    assert UnifiedWorker(config, tmp_path / "output").run() == 0
    assert CALLS == expected


def test_parity_pairs_layouts_and_applies_both_criteria(tmp_path):
    _, _, output = _run(tmp_path, episodes=3)
    episode = episode_episodes(output)
    assert sorted(episode) == [0, 1, 2]
    official = {seed: {"success": row["success"], "actions": [list(a) for a in row["actions"]]}
                for seed, row in episode.items()}
    checks, rows = compare(official, episode, action_tol=1e-4, min_agreement=1.0)
    assert checks["passed"] and checks["first_action_max_abs_diff"] == 0.0 and len(rows) == 3
    assert checks["all_steps_max_abs_diff"] == 0.0 and checks["length_mismatches"] == []
    official[1]["actions"][-1][0] += 0.5                      # later steps are reported, not gated
    checks, _ = compare(official, episode)
    assert checks["passed"] and checks["all_steps_max_abs_diff"] == pytest.approx(0.5)
    official[1]["actions"][0][0] += 1e-3
    assert not compare(official, episode)[0]["passed"]
    official[1]["actions"][0][0] -= 1e-3
    official[0]["actions"] = official[0]["actions"][:-1]
    assert compare(official, episode)[0]["length_mismatches"] == [0]
    official[2]["success"] = not official[2]["success"]
    checks, _ = compare(official, episode, min_agreement=0.9)
    assert not checks["outcome_ok"] and checks["outcome_agreement"] == pytest.approx(2 / 3)
    for row in official.values():
        row["actions"] = None
    checks, _ = compare(official, episode, min_agreement=0.5)
    assert checks["passed"] and checks["first_action_checked"] is False


def test_new_recipes_resolve_to_episode_components():
    registry = Registry()
    recipes = registry.items["recipes"]
    for name in ("memoryvla-bridge-episode", "memoryvla-bridge-episode-occlusion", "memoryvla-bridge-official-carrot",
                 "memoryvla-bridge-parity", "simpler-bridge-replay", "toy-episode-occlusion"):
        assert name in recipes
    config = registry.resolve({"recipe": "memoryvla-bridge-episode"}, allow_unconfigured=True)
    assert config["model"]["entrypoint"] == "wms.engines.memoryvla.adapter:MemoryVLAPolicy"
    assert config["environment"]["entrypoint"] == "wms.scenarios.simpler.env:SimplerBridgeEnv"
    original = json.loads(Path(__file__).parents[1].joinpath("native/memoryvla/recipes.json").read_text(encoding="utf-8"))
    assert original["recipes"]["memoryvla-simpler-bridge-eval"]["entrypoint"] == "wms.engines.memoryvla.native:simpler_evaluate"
