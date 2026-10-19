"""S1-S6 plugin framework (wms.plugins) on the CPU toy: order, takeover, checks, probes, conformance."""
import json

import pytest

np = pytest.importorskip("numpy")

from wms.core.registry import Registry
from wms.plugins.base import Plugin
from wms.plugins.conformance import check as conformance
from wms.runners.unified import UnifiedWorker

TOY_ENV = {"entrypoint": "wms.scenarios.toy.env:ToyPickEnv", "config": {"image_size": 64, "max_steps": 40}}
CENTROID = {"entrypoint": "wms.scenarios.toy.policy:CentroidPickPolicy", "config": {"memory": True}}
PLANNER = {"entrypoint": "wms.scenarios.toy.planner:ToyPlannerPolicy", "config": {"samples": 16, "horizon": 3}}
OCCLUDE = {"type": "occlude", "start": 4, "end": 12, "box": "object:cube", "margin": 0.04}
MOVE = {"step": 6, "object": "cube", "delta": [0.15, 0.0]}
HERE = __name__


def run(tmp_path, name="out", model=CENTROID, episodes=2, **params):
    output = tmp_path / name
    config = {"task": "native", "entrypoint": "wms.episodes.kernel:run", "seed": 0, "model": model,
              "environment": TOY_ENV, "params": {"episodes": episodes, "video": False, **params}}
    code = UnifiedWorker(config, output).run()
    return code, json.loads((output / "result.json").read_text(encoding="utf-8")), output


def lines(output, name="steps.jsonl", episode=0):
    path = output / "episodes" / f"ep_{episode:03d}" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def plugin(entry, **config):
    return {"entrypoint": f"{HERE}:{entry}", "config": config}


# ------------------------------------------------------------------ test plugins
CALLS = []


class AddOne(Plugin):
    stage, role = "S5", "fault"

    def on_action(self, action, ctx):
        CALLS.append(("fault", list(action)))
        return [action[0] + 1.0, *action[1:]]


class Clip(Plugin):
    stage, role = "S5", "defense"

    def on_action(self, action, ctx):
        CALLS.append(("defense", list(action)))
        return [max(-1.0, min(1.0, v)) for v in action]


class Watch(Plugin):
    stage, role = "S5", "monitor"

    def on_action(self, action, ctx):
        CALLS.append(("monitor", list(action)))
        ctx.log(seen=list(action))
        return None                                   # ignored: monitors cannot change anything


class StopAt(Plugin):
    stage, role = "S5", "defense"

    def on_action(self, action, ctx):
        if ctx.t == int(self.config.get("t", 3)):
            ctx.intervene(self.config.get("kind", "stop"), action=self.config.get("action"), reason="test")
        return action


class InPlace(Plugin):
    stage, role = "S1", "fault"

    def on_observation(self, obs, ctx):
        obs["rgb"]["top"][0, 0] = 0
        return obs


class PeekTruth(Plugin):
    stage, role = "S5", "defense"

    def on_action(self, action, ctx):
        ctx.truth()
        return action


class PrivilegedGate(PeekTruth):
    privileged = True


class NoCard(Plugin):
    stage, role = "S1", "attack"

    def on_observation(self, obs, ctx):
        return obs


class Feedback(Plugin):
    stage, role = "S6", "fault"

    def on_feedback(self, transition, ctx):
        return transition


class Update(Plugin):
    stage, role, requires = "S6", "fault", ("learn",)

    def on_update(self, batch, ctx):
        return batch


class ForgetMemory(Plugin):
    """S2 fault: clear the toy memory before every read from step `start` on."""
    stage, role, side, requires = "S2", "fault", "model", ("state",)

    def on_state(self, state, ctx):
        if ctx.phase == "read" and ctx.t >= int(self.config.get("start", 0)):
            return {"cube": None, "goal": None, "age": {"cube": 0, "goal": 0}}
        return state


class PreferLeft(Plugin):
    """S4 defense-shaped re-ranking: only candidates whose first move goes left may win."""
    stage, role, side, requires = "S4", "defense", "model", ("plan",)

    def on_plan(self, plan, ctx):
        scores = np.where(plan["candidates"][:, 0, 0] < 0, plan["scores"], -np.inf)
        ctx.log(allowed=int(np.isfinite(scores).sum()))
        return {**plan, "scores": scores}


class PredictionNoise(Plugin):
    stage, role, side, requires = "S3", "fault", "model", ("predict",)

    def on_prediction(self, prediction, ctx):
        if ctx.meta.get("kind") != "rollout":
            return prediction
        return prediction + ctx.rng.normal(0, 0.05, size=np.shape(prediction))


class ResidualMonitor(Plugin):
    """S3 monitor: predicted gripper position for the executed move vs where the gripper really went."""
    stage, role, side, requires = "S3", "monitor", "model", ("predict",)

    def reset(self, episode, ctx):
        self.pending, self.residuals = None, []

    def on_prediction(self, prediction, ctx):
        if ctx.meta.get("kind") == "chosen_next":
            self.pending = np.asarray(prediction)[:2]
        return prediction

    def after_step(self, step, ctx):
        if self.pending is not None:
            self.residuals.append(float(np.linalg.norm(self.pending - np.asarray(ctx.truth()["gripper"]))))
            self.pending = None

    def end_episode(self, ctx):
        return {"steps": len(self.residuals), "max_residual": max(self.residuals, default=None)}


# ------------------------------------------------------------------ tests
def test_legacy_shorthand_equals_the_builtin_plugins(tmp_path):
    _, _, legacy = run(tmp_path, "legacy", perturbations=[OCCLUDE], target_moves=[MOVE])
    plugins = [{"entrypoint": "wms.plugins.builtin:TargetMoves", "config": {"moves": [MOVE]}},
               {"entrypoint": "wms.plugins.builtin:ImagePerturbation", "config": {"schedule": [OCCLUDE]}}]
    code, result, explicit = run(tmp_path, "explicit", plugins=plugins)
    assert code == 0, result.get("error")
    for episode in (0, 1):
        for name in ("steps.jsonl", "frames_top.npz"):
            a = (legacy / "episodes" / f"ep_{episode:03d}" / name).read_bytes()
            b = (explicit / "episodes" / f"ep_{episode:03d}" / name).read_bytes()
            assert a == b, name
    assert lines(legacy)[6]["target_moves"] and lines(legacy)[5]["perturbations"]


def test_hooks_run_fault_then_defense_then_monitor(tmp_path):
    CALLS.clear()
    code, result, output = run(tmp_path, episodes=1, max_steps=2,
                               plugins=[plugin("Watch"), plugin("Clip"), plugin("AddOne")])
    assert code == 0, result.get("error")
    assert [kind for kind, _ in CALLS[:3]] == ["fault", "defense", "monitor"]
    proposed = CALLS[0][1]
    assert CALLS[1][1][0] == pytest.approx(proposed[0] + 1)        # the defense sees the attacked action
    trace = lines(output, "trace.jsonl")[0]
    assert trace["action_proposed"] == proposed
    assert trace["action"] == lines(output)[0]["action"] == [min(1.0, proposed[0] + 1), *proposed[1:]]
    assert trace["plugins"]["S5"] == ["2:" + f"{HERE}:AddOne", "1:" + f"{HERE}:Clip", "0:" + f"{HERE}:Watch"]
    assert trace["events"][0]["data"] == {"seen": trace["action"]}


def test_stop_takeover_ends_the_episode_without_running_the_action(tmp_path):
    code, result, output = run(tmp_path, episodes=1, plugins=[plugin("StopAt", t=3)])
    assert code == 0, result.get("error")
    row = json.loads((output / "episodes.json").read_text(encoding="utf-8"))["episodes"][0]
    assert row["end_reason"] == "plugin_stop" and row["steps"] == 4 and row["takeovers"] == 1
    steps = lines(output)
    assert steps[3]["action"] is None and steps[3]["events"] == ["plugin_stop"]
    assert steps[3]["state_after"] == steps[3]["state_before"]
    assert len(np.load(output / "episodes/ep_000/frames_top.npz")["seen"]) == row["steps"] + 1
    assert lines(output, "trace.jsonl")[3]["takeover"]["kind"] == "stop"


def test_fallback_takeover_runs_the_given_action(tmp_path):
    code, result, output = run(tmp_path, episodes=1, max_steps=5,
                               plugins=[plugin("StopAt", t=2, kind="fallback", action=[0.0, 0.0, -1.0])])
    assert code == 0, result.get("error")
    trace = lines(output, "trace.jsonl")[2]
    assert trace["action"] == [0.0, 0.0, -1.0] != trace["action_proposed"]
    assert lines(output)[2]["action"] == [0.0, 0.0, -1.0]


def test_strict_mode_catches_in_place_changes(tmp_path):
    code, result, _ = run(tmp_path, episodes=1, plugin_strict=True, plugins=[plugin("InPlace")])
    assert code == 1 and "changed its input in place" in result["error"]


@pytest.mark.parametrize("entry, message", [
    ("PeekTruth", "declare privileged"),
    ("PrivilegedGate", "oracle"),
    ("NoCard", "threat card"),
    ("Feedback", "must list 'learn'"),
    ("Update", "round 2"),
])
def test_contract_violations_are_refused(tmp_path, entry, message):
    code, result, _ = run(tmp_path, episodes=1, max_steps=3, plugins=[plugin(entry)])
    assert code == 1 and message in result["error"]


def test_oracle_defense_and_carded_attack_are_accepted(tmp_path):
    card = {"knowledge": "black_box", "access": "S1 image", "budget": "none", "goal": "test"}
    code, result, _ = run(tmp_path, episodes=1, max_steps=3,
                          plugins=[plugin("PrivilegedGate", oracle=True), plugin("NoCard", threat=card)])
    assert code == 0, result.get("error")


def test_model_probe_requirements_are_checked(tmp_path):
    openvla_like = {"entrypoint": f"{HERE}:NoProbePolicy", "config": {}}
    code, result, _ = run(tmp_path, model=openvla_like, episodes=1, plugins=[plugin("ForgetMemory")])
    assert code == 1 and "needs model probe 'state'" in result["error"]


class NoProbePolicy:
    def __init__(self, config):
        self.info = {}

    def reset(self, seed):
        pass

    def act(self, observation, instruction=None):
        return [0.0, 0.0, -1.0]


def test_s2_probe_on_the_toy_memory(tmp_path):
    _, _, kept = run(tmp_path, "kept", episodes=1, perturbations=[OCCLUDE])
    code, result, forgot = run(tmp_path, "forgot", episodes=1, perturbations=[OCCLUDE],
                               plugins=[plugin("ForgetMemory", start=4)])
    assert code == 0, result.get("error")
    assert lines(kept)[6]["policy_info"]["source_cube"] == "memory"
    assert lines(forgot)[6]["policy_info"]["source_cube"] == "lost"


def test_s3_s4_probes_on_the_toy_planner(tmp_path):
    code, result, base = run(tmp_path, "base", model=PLANNER, episodes=1)
    assert code == 0, result.get("error")
    code, result, left = run(tmp_path, "left", model=PLANNER, episodes=1, max_steps=3, plugins=[plugin("PreferLeft")])
    assert code == 0, result.get("error")
    assert all(step["action"][0] < 0 for step in lines(left))
    assert lines(left, "trace.jsonl")[0]["events"][0]["data"]["allowed"] >= 1
    code, result, noisy = run(tmp_path, "noisy", model=PLANNER, episodes=1,
                              plugins=[plugin("PredictionNoise"), plugin("ResidualMonitor")])
    assert code == 0, result.get("error")
    assert [s["action"] for s in lines(noisy)] != [s["action"] for s in lines(base)]
    row = json.loads((noisy / "episodes.json").read_text(encoding="utf-8"))["episodes"][0]
    report = row["plugin_reports"][f"1:{HERE}:ResidualMonitor"]
    assert report["steps"] == row["steps"] and report["max_residual"] < 1e-9     # the toy model is exact


def test_toy_hazards_reach_the_trace_and_summary(tmp_path):
    code, result, output = run(tmp_path, episodes=1, plugins=[{"entrypoint": "wms.plugins.examples:ActionScale",
                                                               "config": {"scale": 1.0}}])
    assert code == 0, result.get("error")
    row = json.loads((output / "episodes.json").read_text(encoding="utf-8"))["episodes"][0]
    assert "hazard_steps" in row
    hazards = [h["id"] for line in lines(output, "trace.jsonl") for h in line["hazards"]]
    assert set(hazards) <= {"workspace_exit", "object_drop", "fast_contact"}
    assert sum(row["hazard_steps"].values()) == len(hazards)


class PlannerEnv:
    """A fake 'planner' environment: plans five +1 actions, the S5 hooks may change them."""

    def __init__(self, config):
        self.executed = []

    def reset(self, seed, inputs=None):
        self.t = 0
        return {"t": 0}

    def step(self, action):
        self.t += 1
        self.executed.append(list(action))
        return {"obs": {"t": self.t}, "reward": 0.0, "terminated": False, "truncated": False}

    def close(self): ...
    def spec(self): return {"max_steps": 10, "action_dim": 3}
    def observe(self): return {"t": self.t}
    def get_state(self): return {"t": self.t, "executed": list(self.executed)}

    def run_planner(self, seed, hooks, max_steps):
        hooks.on_reset(self.reset(seed))
        try:
            for t in range(5):
                action = [1.0, 1.0, 1.0]
                action = hooks.before_step(t, action) or action
                transition = self.step(action)
                hooks.after_step(t, action, transition)
        except Exception as exc:
            if not hasattr(exc, "episode_status"):
                raise
            return exc.episode_status
        return "planner_completed"


def test_planner_executor_runs_s5_plugins(tmp_path):
    output = tmp_path / "planner"
    config = {"task": "native", "entrypoint": "wms.episodes.kernel:run", "seed": 0,
              "environment": {"entrypoint": f"{HERE}:PlannerEnv", "config": {}},
              "params": {"executor": "planner", "episodes": 1, "video": False,
                         "plugins": [plugin("Clip"), plugin("AddOne"), plugin("StopAt", t=3)]}}
    assert UnifiedWorker(config, output).run() == 0, (output / "result.json").read_text()
    row = json.loads((output / "episodes.json").read_text(encoding="utf-8"))["episodes"][0]
    assert row["end_reason"] == "plugin_stop" and row["steps"] == 4
    assert [s["action"] for s in lines(output)][:3] == [[1.0, 1.0, 1.0]] * 3   # 2.0 clipped back to 1.0
    assert lines(output, "trace.jsonl")[0]["action_proposed"] == [1.0, 1.0, 1.0]


def test_registry_checks_plugins_at_submission(tmp_path):
    registry = Registry()
    assert "example-action-scale" in registry.items["plugins"]
    base = {"recipe": "toy-episode-occlusion"}
    config = registry.resolve({**base, "params": {"plugins": [{"plugin": "example-action-scale"}]}},
                              allow_unconfigured=True)
    assert config["params"]["plugins"][0]["entrypoint"] == "wms.plugins.examples:ActionScale"
    assert "Plugin check failed" not in (config.get("configuration_error") or "")
    bad = registry.resolve({**base, "params": {"plugins": [{"plugin": "example-memory-age"}]},
                            "model": "toy-centroid"}, allow_unconfigured=True)
    assert "Plugin check failed" not in (bad.get("configuration_error") or "")     # toy-centroid has a state probe
    no_probe = registry.resolve({**base, "params": {"plugins": [{"plugin": "example-memory-age"}]},
                                 "model": {"entrypoint": f"{HERE}:NoProbePolicy", "config": {}}},
                                allow_unconfigured=True)
    assert "needs model probe 'state'" in no_probe["configuration_error"]
    unknown = registry.resolve({**base, "params": {"plugins": [{"plugin": "nope"}]}}, allow_unconfigured=True)
    assert "unknown plugin 'nope'" in unknown["configuration_error"]


def test_declared_metadata_checks_plugins_the_service_cannot_import(tmp_path):
    path = tmp_path / "recipes.json"
    path.write_text(json.dumps({"plugins": {
        "far-away": {"entrypoint": "not_installed_here.plugins:Gate", "stage": "S5", "role": "defense"},
        "far-away-attack": {"entrypoint": "not_installed_here.plugins:Patch", "stage": "S1", "role": "attack"},
        "undeclared": {"entrypoint": "not_installed_here.plugins:Thing"}}}), encoding="utf-8")
    registry = Registry(paths=[path])
    resolve = lambda name: registry.resolve({"recipe": "toy-episode-occlusion",
                                             "params": {"plugins": [{"plugin": name}]}}, allow_unconfigured=True)
    assert "Plugin check failed" not in (resolve("far-away").get("configuration_error") or "")
    assert "threat card" in resolve("far-away-attack")["configuration_error"]
    assert "declare stage and role" in resolve("undeclared")["configuration_error"]


@pytest.mark.parametrize("name", ["example-action-scale", "example-memory-age"])
def test_conformance_passes_the_examples(tmp_path, name):
    entry = Registry().items["plugins"][name]
    spec = {"id": name, "entrypoint": entry["entrypoint"], "config": entry["config"]}
    report = conformance(spec, tmp_path)
    assert not [row for row in report if row[0] == "FAIL"], report


def test_conformance_fails_a_monitor_that_changes_actions(tmp_path):
    report = conformance({"id": "x", "entrypoint": f"{HERE}:SneakyMonitor", "config": {}}, tmp_path)
    assert ("FAIL", "runs") in [(status, name) for status, name, _ in report]          # strict mode


class SneakyMonitor(Plugin):
    stage, role = "S5", "monitor"

    def on_action(self, action, ctx):
        action[0] = 0.0                    # mutates in place: strict mode refuses it
        return action
