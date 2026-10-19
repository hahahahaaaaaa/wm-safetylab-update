"""S6 feedback, transitions, alarms, observation kinds and returned-value checks (week 0 of phase 1)."""
import json

import pytest

np = pytest.importorskip("numpy")

from wms.plugins.base import Plugin
from wms.plugins.conformance import check as conformance
from wms.plugins.tree import check_like, map_floats
from wms.runners.unified import UnifiedWorker

TOY_ENV = {"entrypoint": "wms.scenarios.toy.env:ToyPickEnv", "config": {"image_size": 64, "max_steps": 40}}
VECTOR_ENV = {"entrypoint": "wms.scenarios.toy.env:ToyPickEnv",
              "config": {"image_size": 64, "max_steps": 40, "observation": "vector"}}
PLANNER = {"entrypoint": "wms.scenarios.toy.planner:ToyPlannerPolicy", "config": {"samples": 16, "horizon": 3}}
ONLINE = {"entrypoint": "wms.scenarios.toy.planner:ToyOnlinePlannerPolicy",
          "config": {"samples": 16, "horizon": 3, "initial_gain": 0.5}}
HERE = __name__


def run(tmp_path, name="out", model=PLANNER, env=TOY_ENV, episodes=1, **params):
    output = tmp_path / name
    config = {"task": "native", "entrypoint": "wms.episodes.kernel:run", "seed": 0, "model": model,
              "environment": env, "params": {"episodes": episodes, "video": False, **params}}
    code = UnifiedWorker(config, output).run()
    return code, json.loads((output / "result.json").read_text(encoding="utf-8")), output


def lines(output, name="steps.jsonl", episode=0):
    path = output / "episodes" / f"ep_{episode:03d}" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def summary(output, episode=0):
    return json.loads((output / "episodes" / f"ep_{episode:03d}" / "summary.json").read_text(encoding="utf-8"))


def plugin(entry, **config):
    return {"entrypoint": f"{HERE}:{entry}", "config": config}


# ------------------------------------------------------------------ test plugins
class Fields(Plugin):
    """S6 monitor that declared predict and embed: it sees both fields; reports the gripper residual."""
    stage, role, side, requires = "S6", "monitor", "model", ("predict", "embed")

    def reset(self, episode, ctx):
        self.seen, self.residual = [], []

    def on_feedback(self, transition, ctx):
        self.seen.append((transition["predicted_next"] is not None, transition["next_state"] is not None))
        if transition["predicted_next"] is not None and not transition["taken_over"]:
            self.residual.append(float(np.abs(np.asarray(transition["predicted_next"])[:2]
                                              - np.asarray(transition["next_state"])[:2]).max()))
        return transition

    def end_episode(self, ctx):
        return {"seen": [list(map(bool, x)) for x in set(self.seen)], "max_residual": max(self.residual, default=None)}


class Blind(Fields):
    """The same monitor without declaring the probes: the fields are hidden (None)."""
    requires = ()


class AfterStepReader(Plugin):
    stage, role = "S5", "monitor"

    def on_action(self, action, ctx):
        return action

    def after_step(self, step, ctx):
        transition = step["transition"]
        ctx.log(keys=sorted(transition), reward=transition["reward"], next_state=transition["next_state"])


class SurpriseAlarm(Plugin):
    """S6 defense: alarm at step `at`, asking the kernel for a fallback takeover of `steps` actions."""
    stage, role, side, requires = "S6", "defense", "model", ("predict", "embed")

    def on_feedback(self, transition, ctx):
        if transition["t"] == int(self.config.get("at", 2)):
            ctx.alarm("test surprise", score=1.5, takeover=self.config.get("takeover", "fallback"),
                      steps=int(self.config.get("steps", 2)), action=[0.0, 0.0, -1.0])
        return transition


class AlarmReader(Plugin):
    stage, role = "S5", "defense"

    def on_action(self, action, ctx):
        if ctx.alarms:
            ctx.log(alarms=[(a["plugin"], a["t"], a["reason"]) for a in ctx.alarms])
        return action


class MonitorAlarm(Plugin):
    stage, role = "S6", "monitor"

    def on_feedback(self, transition, ctx):
        ctx.alarm("monitors only log")
        return transition


class ShrinkFeedback(Plugin):
    """S6 fault: report only `factor` of the gripper's real motion to a learning model."""
    stage, role, side, requires = "S6", "fault", "model", ("learn",)
    neutral = {"factor": 1.0}

    def on_feedback(self, transition, ctx):
        factor = float(self.config.get("factor", 0.3))
        if factor == 1.0:
            return transition
        before, after = transition["obs"]["proprio"], transition["next_obs"]["proprio"]
        moved = [b + factor * (a - b) for a, b in zip(after[:2], before[:2])]
        ctx.log(factor=factor)
        return {**transition, "next_obs": {**transition["next_obs"], "proprio": [*moved, *after[2:]]}}


class FloatImage(Plugin):
    stage, role = "S1", "fault"
    neutral = {}

    def on_observation(self, obs, ctx):
        return {**obs, "rgb": {cam: img.astype(float) for cam, img in obs["rgb"].items()}}


class NeedsRgb(Plugin):
    stage, role, requires = "S5", "monitor", ("rgb",)

    def on_action(self, action, ctx):
        return action


class NeedsVector(NeedsRgb):
    requires = ("vector",)


class NoNeutral(Plugin):
    stage, role = "S5", "fault"

    def on_action(self, action, ctx):
        return [0.5 * v for v in action]


# ------------------------------------------------------------------ tests
def test_transition_fields_follow_each_plugins_requires(tmp_path):
    code, result, out = run(tmp_path, max_steps=15, plugins=[plugin("Fields"), plugin("Blind")])
    assert code == 0, result.get("error")
    reports = summary(out)["plugin_reports"]
    assert reports["0:test_plugins_feedback:Fields"]["seen"] == [[True, True]]
    assert reports["1:test_plugins_feedback:Blind"]["seen"] == [[False, False]]
    # the toy planner's model is exact for the gripper (away from the walls)
    assert reports["0:test_plugins_feedback:Fields"]["max_residual"] < 1e-9


def test_after_step_sees_the_transition_and_its_logs_reach_the_trace(tmp_path):
    code, result, out = run(tmp_path, max_steps=4, plugins=[plugin("AfterStepReader")])
    assert code == 0, result.get("error")
    events = [e for row in lines(out, "trace.jsonl") for e in row["events"] if e["hook"] == "after_step"]
    assert len(events) == 4
    assert events[0]["data"]["keys"] == sorted(["t", "obs", "action", "next_obs", "reward", "done", "taken_over",
                                                "predicted_next", "next_state"])
    assert events[0]["data"]["next_state"] is None                     # the monitor did not require embed


def test_s6_alarm_takes_over_the_next_actions(tmp_path):
    code, result, out = run(tmp_path, max_steps=8, plugins=[plugin("SurpriseAlarm", at=2, steps=2),
                                                            plugin("AlarmReader")])
    assert code == 0, result.get("error")
    trace, steps = lines(out, "trace.jsonl"), lines(out)
    assert trace[2]["alarms"][0]["reason"] == "test surprise" and trace[2]["takeover"] is None
    for t in (3, 4):
        assert trace[t]["takeover"]["via"] == "alarm" and trace[t]["takeover"]["plugin"] == "0:test_plugins_feedback:SurpriseAlarm"
        assert steps[t]["action"] == [0.0, 0.0, -1.0]
    assert trace[5]["takeover"] is None
    seen = [e["data"]["alarms"] for row in trace for e in row["events"] if e["plugin"].endswith("AlarmReader")]
    assert seen == [[["0:test_plugins_feedback:SurpriseAlarm", 2, "test surprise"]]]   # S5 saw it at step 3
    assert summary(out)["takeovers"] == 2


def test_alarm_stop_ends_the_episode(tmp_path):
    code, result, out = run(tmp_path, max_steps=8, plugins=[plugin("SurpriseAlarm", at=1, takeover="stop")])
    assert code == 0, result.get("error")
    assert summary(out)["end_reason"] == "plugin_stop" and len(lines(out)) == 3


def test_only_defenses_raise_alarms(tmp_path):
    code, result, _ = run(tmp_path, max_steps=3, plugins=[plugin("MonitorAlarm")])
    assert code == 1 and "only defenses" in result["error"]


def test_online_planner_learns_and_s6_feedback_changes_what_it_learns(tmp_path):
    code, result, honest = run(tmp_path, "honest", model=ONLINE, max_steps=30)
    assert code == 0, result.get("error")
    code, result, shrunk = run(tmp_path, "shrunk", model=ONLINE, max_steps=30,
                               plugins=[plugin("ShrinkFeedback", factor=0.3)])
    assert code == 0, result.get("error")
    gain = lambda out: lines(out)[-1]["policy_info"]["move_gain"][0]
    assert lines(honest)[0]["policy_info"]["move_gain"][0] == pytest.approx(0.03)        # 0.5 x step 0.06
    assert gain(honest) == pytest.approx(0.06, rel=0.1)
    assert gain(shrunk) < 0.03


def test_s6_fault_must_not_target_a_model_that_does_not_learn(tmp_path):
    code, result, _ = run(tmp_path, max_steps=3, plugins=[plugin("ShrinkFeedback")])
    assert code == 1 and "needs model probe 'learn'" in result["error"]


def test_vector_observations_and_observation_kinds(tmp_path):
    code, result, _ = run(tmp_path, "vector", env=VECTOR_ENV, episodes=2)
    assert code == 0 and result["data"]["success_rate"] > 0, result.get("error")
    code, result, _ = run(tmp_path, "rgb_on_vector", env=VECTOR_ENV, max_steps=3, plugins=[plugin("NeedsRgb")])
    assert code == 1 and "needs camera images" in result["error"]
    code, result, _ = run(tmp_path, "vector_on_rgb", max_steps=3, plugins=[plugin("NeedsVector")])
    assert code == 1 and "needs vector observations" in result["error"]


def test_a_hook_must_return_a_value_of_its_inputs_form(tmp_path):
    code, result, _ = run(tmp_path, max_steps=3, plugins=[plugin("FloatImage")])
    assert code == 1 and "dtype float64, expected uint8" in result["error"]


def test_conformance_runs_every_matching_toy_and_the_neutral_check(tmp_path):
    spec = {"id": "scale", "entrypoint": "wms.plugins.examples:ActionScale", "config": {"scale": 0.5, "dims": 2}}
    report = conformance(spec, tmp_path / "scale")
    assert not [row for row in report if row[0] == "FAIL"], report
    assert len([row for row in report if row[1] == "neutral" and row[0] == "PASS"]) == 4      # all four toys
    report = conformance({"id": "x", "entrypoint": f"{HERE}:NoNeutral", "config": {}}, tmp_path / "none")
    assert ("FAIL", "neutral") in [(status, name) for status, name, _ in report]
    report = conformance({"id": "shrink", "entrypoint": f"{HERE}:ShrinkFeedback", "config": {}}, tmp_path / "shrink")
    assert not [row for row in report if row[0] == "FAIL"], report
    assert [row[2] for row in report if row[1] == "toys"] == ["toy-planner-online + toy-pick"]


def test_tree_helpers():
    state = {"latent": np.ones(3, dtype=np.float32), "ids": np.arange(3), "memory": [1.0, {"age": 2}], "name": "x"}
    changed = map_floats(state, lambda x: x * 2)
    assert changed["latent"].dtype == np.float32 and changed["latent"].tolist() == [2.0, 2.0, 2.0]
    assert changed["ids"] is state["ids"] and changed["memory"] == [2.0, {"age": 2}]
    assert state["latent"].tolist() == [1.0, 1.0, 1.0]
    assert check_like(changed, state) is None
    assert "shape" in check_like({**state, "latent": np.ones(4, dtype=np.float32)}, state)


class RolloutMeta(Plugin):
    stage, role, side, requires = "S3", "monitor", "model", ("predict",)

    def reset(self, episode, ctx):
        self.seen = set()

    def on_prediction(self, prediction, ctx):
        if ctx.meta.get("kind") == "rollout":
            self.seen.add((ctx.meta["source"], ctx.meta["h"]))
        return prediction

    def end_episode(self, ctx):
        return {"seen": sorted(self.seen)}


class ColdStart(Plugin):
    """S2 fault: clear the warm start (plan_mean None) from step 5 on."""
    stage, role, side, requires = "S2", "fault", "model", ("state",)
    neutral = {"start": 10 ** 9}

    def on_state(self, state, ctx):
        if ctx.t >= int(self.config.get("start", 5)):
            return {**state, "plan_mean": None}
        return state


def test_rollout_meta_and_cold_start_on_the_toy_planner(tmp_path):
    code, result, out = run(tmp_path, "meta", max_steps=6, plugins=[plugin("RolloutMeta")])
    assert code == 0, result.get("error")
    assert summary(out)["plugin_reports"]["0:test_plugins_feedback:RolloutMeta"]["seen"] == [
        ["value", 0], ["value", 1], ["value", 2]]
    _, _, base = run(tmp_path, "base", max_steps=12)
    code, result, cold = run(tmp_path, "cold", max_steps=12, plugins=[plugin("ColdStart", start=5)])
    assert code == 0, result.get("error")
    a, b = lines(base), lines(cold)
    assert [r["action"] for r in a[:5]] == [r["action"] for r in b[:5]]
    assert [r["action"] for r in a[5:]] != [r["action"] for r in b[5:]]


def test_clean_stats_from_a_toy_run(tmp_path):
    from wms.episodes.clean_stats import collect
    monitors = [{"plugin": "monitor-surprise"}, {"plugin": "monitor-latent-moments"}]
    code, result, out = run(tmp_path, episodes=2, max_steps=20, plugins=monitors)
    assert code == 0, result.get("error")
    stats = collect([out])
    assert stats["episodes"] == 2 and stats["action"]["dims"] == 3 and len(stats["action"]["delta_p99"]) == 3
    assert stats["latent"]["dims"] == 7 and stats["latent"]["n"] > 0
    assert stats["events"]["monitor-surprise.surprise"]["n"] > 0
    assert len(stats["image"]["top"]["channel_mean"]) == 3
    _, _, vector = run(tmp_path, "vector", env=VECTOR_ENV, max_steps=10)
    assert collect([vector])["observation_vector"]["dims"] == 7
