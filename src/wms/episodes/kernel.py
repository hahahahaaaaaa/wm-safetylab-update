"""Episode kernel: run a policy in an environment step by step, with S1-S6 plugins.

The platform owns the loop, so an experiment (a fault, an attack, a defense, a monitor)
is written once as a plugin (wms.plugins) and works for every policy/environment pair
that follows wms.core.protocols.EpisodePolicy / EpisodeEnv. Ported from the
sub-platform's sublab_ext.tasks.visual_robustness; logs have the same layout.

Recipe fields (task "native", entrypoint "wms.episodes.kernel:run")
  model        the policy under test (executor "model")
  environment  the environment; image_box() for boxes that follow objects,
               set_object_pose()/set_goal() for target moves, hazards() for hazard predicates
  seed         first episode seed; episode i uses seed + i unless params.seeds is given.
               The same seed goes to environment.reset() and policy.reset().
  inputs       "instruction" goes to model.act(); everything else to environment.reset()
  params
    episodes | seeds   how many episodes, or the exact seed list
    max_steps          step budget per episode (default: environment spec max_steps)
    executor           "model" (default) or "planner" (environment.run_planner, e.g. ManiSkill's
                       official motion planner)
    plugins            [{"plugin": registry name | "entrypoint": "module:Class", "config": {...}}],
                       see wms.plugins.base for stages, roles and hooks
    perturbations      shorthand for the S1 plugin wms.plugins.builtin:ImagePerturbation (wms.perturb)
    target_moves       shorthand for the ENV plugin wms.plugins.builtin:TargetMoves:
                       [{"step": 6, "object": "cube", "delta": [0.2, 0.0]}, {"step": 10, "goal": true, "xyz": [...]}]
    plugin_strict      also check that no plugin changes a hook's input in place (wms plugin test)
    stop_on_terminate  end the episode when the policy sets info["terminate_episode"] (default true,
                       as SimplerEnv's official loop does)
    reset_order        "env_first" (default): environment.reset(seed), then policy.reset(seed);
                       "policy_first": the other way round, for official loops that seed the policy's
                       random streams before resetting the simulator (TD-MPC2's evaluation)
    save_raw           also store unperturbed frames (default true)
    video, fps         mp4 of what the policy saw (default true, 10)

Order inside step t: ENV plugins (target moves) for t -> true state_before -> frame t
rendered, S1 plugins, logged -> policy acts on that frame (S2-S4 plugins run inside, through
the model's probes) -> S5 plugins, then a pending alarm takeover -> simulator step -> hazards,
true state_after -> S6 on_feedback on the step's transition -> model.learn(transition) for
models with the learn probe -> after_step (with the transition, read-only) -> logs.
Every frame k in frames_<cam>.npz matches line t=k of steps.jsonl and trace.jsonl; the extra
last frame is the observation after the final action. With no plugins, every step is exactly
what it was before plugins existed.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import os
import sys

from wms.core.protocols import EPISODE_ENV, EPISODE_POLICY, missing_methods
from wms.episodes.steplog import StepLogger
from wms.plugins.runtime import PluginSet, PluginStop

CLAIM_LIMIT = ("Measures this policy on this environment under the listed perturbations and plugins only. "
               "The toy environment is a pipeline check, not evidence about real manipulation.")
PACKAGES = ("numpy", "torch", "transformers", "sapien", "mani-skill", "gymnasium", "imageio")


def _validate_moves(moves):
    for index, move in enumerate(moves or []):
        if "step" not in move or not ("object" in move or move.get("goal") or move.get("target") == "goal"):
            raise ValueError(f"target_moves[{index}] needs step and object (or goal: true)")
        if move.get("xyz") is None and move.get("delta") is None:
            raise ValueError(f"target_moves[{index}] needs xyz or delta")
    return list(moves or [])


def _normalise_moves(moves):
    """apply_target_moves expects {"goal": {...}} for goal moves."""
    out = []
    for move in moves:
        if move.get("goal") is True or move.get("target") == "goal":
            out.append({"step": move["step"], "goal": {"xyz": move.get("xyz"), "delta": move.get("delta")}})
        else:
            out.append(move)
    return out


def _plugin_specs(params, moves):
    """The legacy shorthands first (as they always ran first), then params.plugins."""
    specs = []
    if moves:
        specs.append({"id": "target_moves", "entrypoint": "wms.plugins.builtin:TargetMoves",
                      "config": {"moves": moves}})
    if params.get("perturbations"):
        specs.append({"id": "perturbations", "entrypoint": "wms.plugins.builtin:ImagePerturbation",
                      "config": {"schedule": params["perturbations"]}})
    items = params.get("plugins") or []
    if any(isinstance(item, str) or "entrypoint" not in item for item in items):
        # a configuration that did not pass through the registry (direct worker, tests)
        from wms.core.registry import Registry
        from wms.plugins.resolve import resolve_specs
        items = resolve_specs(items, Registry().items["plugins"])
    return specs + list(items)


def _call_optional(component, method, *args):
    function = getattr(component, method, None)
    return function(*args) if callable(function) else None


def _require(component, names, role, entrypoint):
    missing = missing_methods(component, names)
    if missing:
        raise TypeError(f"{role} {entrypoint} lacks {missing}; the episode kernel needs "
                        f"{list(names)} (see wms.core.protocols)")


def _manifest(config, **extra):
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": sys.executable, "packages": packages,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "seed": config.get("seed"), "model": config.get("model"), "environment": config.get("environment"),
            "inputs": config.get("inputs"), "params": config.get("params"), **extra}


class _Episode:
    """State shared by both executors for one episode."""

    def __init__(self, env, plugins, logger, context):
        self.env, self.plugins = env, plugins
        self.logger, self.context = logger, context
        self.hazard_source = getattr(env, "hazards", None) if callable(getattr(env, "hazards", None)) else None
        self.raw = None
        self.events, self.success_step = [], None
        self.end_reason = "step_limit"
        self.perturbed_steps, self.moves_applied, self.takeovers = 0, 0, 0
        self.hazard_counts = {}
        self.model, self.learns = None, False
        self.started = False
        self.reset_moves = []          # planner mode: step-0 moves, applied before planning

    def apply_moves(self, t):
        self.plugins.begin_step(t)
        moved = self.plugins.on_env(t) if self.plugins.has("on_env") else []
        if moved:
            self.raw = self.env.observe()
            self.moves_applied += len(moved)
        return moved

    def observe(self, t, applied=None):
        """applied: moves for step t that already happened (planner step 0); else apply them now."""
        self.context.check_cancelled()
        self.plugins.check_observation(self.raw)
        if applied is None:
            moved = self.apply_moves(t)
        else:
            moved = applied
        before = self.env.get_state()
        seen = self.plugins.call("on_observation", self.raw) if self.plugins.has("on_observation") else self.raw
        records = self.plugins.records("S1", "on_observation")
        self.logger.frame(seen, self.raw)
        self.perturbed_steps += bool(records)
        return seen, records, moved, before

    def instruction(self, seen, text):
        """S1 on_instruction: the changed text reaches the policy both ways it may read it."""
        if not self.plugins.has("on_instruction"):
            return seen, text
        original = seen.get("instruction") if isinstance(seen, dict) and seen.get("instruction") else text
        changed = self.plugins.call("on_instruction", original)
        if changed != original:
            if isinstance(seen, dict) and "instruction" in seen:
                seen = {**seen, "instruction": changed}
            text = changed
        return seen, text

    def act_on(self, action):
        """S5, then an alarm's takeover: (action to execute or None to stop, proposed action)."""
        if not self.plugins:
            return action, action
        executed = self.plugins.call("on_action", action) if self.plugins.has("on_action") else action
        takeover = self.plugins.alarm_takeover()
        if takeover:
            self.takeovers += 1
            if takeover["kind"] == "stop":
                return None, action
            executed = takeover["action"]
        return executed, action

    def stopped(self, t, proposed, seen, records, moved, before, policy_info):
        """A defense stopped the episode before this step's action ran: log the step without it."""
        transition = {"obs": self.raw, "reward": None, "terminated": False, "truncated": False,
                      "events": ["plugin_stop"]}
        self.record(t, None, transition, seen, records, moved, before, policy_info, proposed=proposed)
        self.end_reason = "plugin_stop"

    def feedback(self, t, action, transition):
        """The step's transition (wms.plugins.runtime), as the environment returned it."""
        import numpy as np
        out = {"t": t, "obs": self.raw, "action": np.asarray(action, dtype=float).copy(),
               "next_obs": transition["obs"], "reward": transition.get("reward"),
               "done": bool(transition.get("terminated") or transition.get("truncated")),
               "taken_over": bool(self.plugins.takeover), "predicted_next": None, "next_state": None}
        if self.plugins.needs["predicted_next"]:
            out["predicted_next"] = self.plugins.predicted_next
        if self.plugins.needs["next_state"] and self.model is not None:
            out["next_state"] = self.model.embed(transition["obs"])
        return out

    def record(self, t, action, transition, seen, records, moved, before, policy_info, proposed=None):
        hazards = list(self.hazard_source() or []) if self.hazard_source and action is not None else []
        for hazard in hazards:
            self.hazard_counts[hazard["id"]] = self.hazard_counts.get(hazard["id"], 0) + 1
        state_after = self.env.get_state()
        step_transition = None
        if action is not None and (self.plugins or self.learns):
            step_transition = self.feedback(t, action, transition)
            self.plugins.check_reward(step_transition["reward"])
            learned = self.plugins.feedback(step_transition, hazards)
            if self.learns:
                self.model.learn(learned)
        if self.plugins:
            self.plugins.after_step({"t": t, "action_proposed": proposed if proposed is not None else action,
                                     "action": action, "reward": transition.get("reward"),
                                     "events": list(transition.get("events", [])),
                                     "terminated": bool(transition.get("terminated")),
                                     "truncated": bool(transition.get("truncated")),
                                     "info": transition.get("info"), "transition": step_transition}, hazards)
        self.logger.step(t, action=action, transition=transition, state_before=before,
                         state_after=state_after, perturbations=records, moves=moved,
                         policy_info=policy_info, seen_obs=seen)
        self.logger.trace(t, action_proposed=proposed if proposed is not None else action, action=action,
                          hazards=hazards, env_events=transition.get("events", []), **self.plugins.trace())
        self.raw = transition["obs"]
        for event in transition.get("events", []):
            self.events.append({"t": t, "event": event})
            if event == "success" and self.success_step is None:
                self.success_step = t + 1
        if transition.get("terminated") or transition.get("truncated"):
            self.end_reason = "terminated" if transition.get("terminated") else "truncated"
            return True
        return False

    def finish(self, extra=None):
        steps = self.logger.count
        self.plugins.begin_step(steps)
        seen = (self.plugins.call("on_observation", self.raw, final=True)
                if self.plugins.has("on_observation") else self.raw)
        self.logger.frame(seen, self.raw)
        final = self.env.get_state()
        summary = {"success": self.success_step is not None, "success_step": self.success_step,
                   "end_reason": self.end_reason, "final_distance": final.get("distance_to_goal"),
                   "final_state": final, "events": self.events, "perturbed_steps": self.perturbed_steps,
                   "target_moves_applied": self.moves_applied, **(extra or {})}
        if self.hazard_source:
            summary["hazard_steps"] = self.hazard_counts
        if self.plugins:
            summary["takeovers"] = self.takeovers
            reports = self.plugins.end_episode()
            if reports:
                summary["plugin_reports"] = reports
        return summary


def _model_episode(episode, model, seed, inputs, max_steps, stop_on_terminate, policy_first=False):
    env_inputs = {key: value for key, value in inputs.items() if key != "instruction"}
    episode.model = model
    episode.learns = "learn" in (getattr(model, "probes", ()) or ()) and callable(getattr(model, "learn", None))
    if policy_first:                   # the official loop seeds the policy before resetting the simulator
        model.reset(seed)
    episode.raw = episode.env.reset(seed, env_inputs)
    if not policy_first:
        model.reset(seed)
    episode.logger.begin_episode(episode.index, seed, episode.env.get_state())
    for t in range(max_steps):
        seen, records, moved, before = episode.observe(t)
        seen, instruction = episode.instruction(seen, inputs.get("instruction"))
        action = model.act(seen, instruction)
        info = deepcopy(getattr(model, "info", None))
        executed, proposed = episode.act_on(action)
        if executed is None:
            episode.stopped(t, proposed, seen, records, moved, before, info)
            break
        transition = episode.env.step(executed)
        if episode.record(t, executed, transition, seen, records, moved, before, info, proposed=proposed):
            break
        if stop_on_terminate and isinstance(info, dict) and info.get("terminate_episode"):
            episode.end_reason = "policy_terminated"
            break
    return episode.finish()


def _planner_episode(episode, seed, max_steps):
    class Hooks:
        pending = None

        @staticmethod
        def on_reset(obs):
            if episode.started:
                raise RuntimeError("The planner reset the environment twice in one episode")
            episode.started = True
            episode.raw = obs
            episode.logger.begin_episode(episode.index, seed, episode.env.get_state())
            # The planner plans right after reset, before its first step: step-0 moves happen
            # here so that "move at step 0" means "before planning". Line 0 still logs them.
            episode.reset_moves = episode.apply_moves(0)

        @staticmethod
        def before_step(t, action):
            """Returns the action to execute (an S5 plugin may change it) or raises PluginStop."""
            observed = episode.observe(t, applied=episode.reset_moves if t == 0 else None)
            executed, proposed = episode.act_on(action)
            Hooks.pending = (*observed, proposed)
            if executed is None:
                seen, records, moved, before, _ = Hooks.pending
                episode.stopped(t, proposed, seen, records, moved, before, Hooks.info())
                raise PluginStop("plugin_stop")
            return executed

        @staticmethod
        def after_step(t, action, transition):
            seen, records, moved, before, proposed = Hooks.pending
            episode.record(t, action, transition, seen, records, moved, before, Hooks.info(), proposed=proposed)

        @staticmethod
        def info():
            return {"executor": "official_motion_planner", "uses_images": False}

    status = episode.env.run_planner(seed, Hooks, max_steps)
    if not episode.started:
        raise RuntimeError("The planner never reset the environment")
    if status == "plugin_stop":
        episode.end_reason = "plugin_stop"
    elif status == "step_budget":
        episode.end_reason = "step_limit"
    elif status in ("planner_completed", "no_plan") and episode.end_reason == "step_limit":
        # the planner stopped stepping before the episode ended: say so instead of "step_limit"
        episode.end_reason = "planner_finished" if status == "planner_completed" else "no_plan"
    return episode.finish({"planner_status": status})


def run(config, output_dir, context):
    params, inputs = dict(config.get("params", {})), dict(config.get("inputs", {}))
    executor = params.get("executor", "model")
    if executor not in ("model", "planner"):
        raise ValueError("params.executor must be 'model' or 'planner'")
    seed = int(config.get("seed", 0))
    seeds = [int(s) for s in params.get("seeds") or [seed + i for i in range(int(params.get("episodes", 1)))]]
    if not seeds:
        raise ValueError("No episodes: params.episodes must be >= 1 or params.seeds non-empty")
    moves = _normalise_moves(_validate_moves(params.get("target_moves")))
    specs = _plugin_specs(params, moves)
    stop_on_terminate = bool(params.get("stop_on_terminate", True))
    reset_order = params.get("reset_order", "env_first")
    if reset_order not in ("env_first", "policy_first"):
        raise ValueError("params.reset_order must be 'env_first' or 'policy_first'")
    env_spec, model_spec = config.get("environment"), config.get("model")
    if not env_spec:
        raise ValueError("The episode kernel needs an environment")
    if executor == "model" and not model_spec:
        raise ValueError("executor 'model' needs a model")
    from wms.runners.unified import load_entrypoint
    plugins = PluginSet(specs, lambda spec: load_entrypoint(spec["entrypoint"])(dict(spec.get("config") or {})),
                        strict=bool(params.get("plugin_strict", False)))
    context.write_json("run_manifest.json", _manifest(config, executor=executor, seeds=seeds,
                                                      plugins=plugins.describe()))

    env = model = None
    try:
        env = context.component(env_spec)
        _require(env, EPISODE_ENV if executor == "model" else EPISODE_ENV + ("run_planner",),
                 "environment", env_spec.get("entrypoint"))
        if executor == "model":
            model = context.component(model_spec)
            _require(model, EPISODE_POLICY, "model", model_spec.get("entrypoint"))
        plugins.check(env, model, executor)
        spec = env.spec() or {}
        _call_optional(model, "bind_env", spec)
        max_steps = int(params.get("max_steps") or spec.get("max_steps") or 100)
        logger = StepLogger(output_dir, save_raw=bool(params.get("save_raw", True)),
                            video=bool(params.get("video", True)), fps=int(params.get("fps", 10)))
        for index, episode_seed in enumerate(seeds):
            plugins.begin_episode(index, episode_seed, spec)
            episode = _Episode(env, plugins, logger, context)
            episode.index = index
            if executor == "model":
                summary = _model_episode(episode, model, episode_seed, inputs, max_steps, stop_on_terminate,
                                         reset_order == "policy_first")
            else:
                summary = _planner_episode(episode, episode_seed, max_steps)
            row = logger.end_episode(summary)
            context.event("progress", {"phase": "episode", "episode": index + 1, "of": len(seeds),
                                       "seed": episode_seed, "success": row["success"], "steps": row["steps"]})
    finally:
        errors = []
        for component in (plugins, model, env):
            try:
                _call_optional(component, "close")
            except Exception as exc:  # noqa: BLE001 - close everything, report the first failure
                errors.append(exc)
        if errors and sys.exc_info()[0] is None:
            raise errors[0]

    rows = logger.episodes
    data = {"path": "episode", "episodes": len(rows), "executor": executor,
            "success_rate": sum(r["success"] for r in rows) / len(rows),
            "mean_steps": sum(r["steps"] for r in rows) / len(rows),
            "environment": env_spec.get("entrypoint"),
            "model": model_spec.get("entrypoint") if executor == "model" else "official_motion_planner",
            "perturbations": len(params.get("perturbations") or []), "target_moves": len(moves),
            "plugins": [item["id"] for item in plugins.describe()], "seeds": seeds,
            "claim_limit": CLAIM_LIMIT}
    outputs = logger.close({"path": "episode", "executor": executor, "env_spec": spec,
                            "perturbations": params.get("perturbations", []), "target_moves": moves,
                            "plugins": plugins.describe(), "max_steps": max_steps, "claim_limit": CLAIM_LIMIT})
    outputs.append({"path": "run_manifest.json", "kind": "report", "media_type": "application/json"})
    return {"data": data, "outputs": outputs}
