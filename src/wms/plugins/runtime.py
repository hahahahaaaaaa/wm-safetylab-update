"""Run the plugins of one experiment inside the episode kernel.

PluginSet builds the plugins, checks them against the model and environment actually
loaded, calls their hooks in a fixed order and collects what they log:

  per hook: fault / attack first, then defense, then monitor (list order within a role)
  per step: ENV -> S1 (frame t) -> model.act [S2, S3, S4 through the model's probes] -> S5
            (then an alarm's takeover) -> simulator step -> hazards -> S6 on_feedback
            -> model.learn (models with the learn probe) -> after_step

Each plugin gets its own context (ctx): its own random stream seeded from (episode seed,
plugin position), so plugins never draw from the global generators and adding one plugin
does not change another's randomness; ctx.log(...) writes to the step's trace.

What a hook returns is checked before it goes on: the same structure as its input (keys,
lengths, array shapes and dtypes; S2 only the top-level keys, since forgetting may shrink a
memory; S4 the scores' shape), and an S5 action of the same length with finite values. An
action outside the spec's bounds is allowed (a fault may command too much; the environment
clips it, and a later defense may clip it first).

Alarms: a defense of S1-S4 or S6 calls ctx.alarm(reason, score, takeover="stop" | "fallback",
steps=n). The alarm is logged; S5 plugins see it in ctx.alarms (alarms raised during S6 or
after_step of step t are seen at step t+1). With takeover, the kernel takes over at the next
action, after the S5 plugins, for n actions, unless an S5 defense already took over that step.

Transitions: after the simulator step the kernel builds {"t", "obs", "action", "next_obs",
"reward", "done", "taken_over", "predicted_next", "next_state"} (obs and next_obs as the
environment returned them, before S1; action as executed). predicted_next (the model's
prediction for the action it chose) and next_state (model.embed(next_obs)) are filled only
when some plugin requires 'predict' / 'embed', and each plugin sees only the fields whose
probe it declared (the others are None).
"""
from __future__ import annotations

import copy

import numpy as np

from .base import (ALARM_STAGES, ENV_METHODS, HOOKS, OBSERVATION_KINDS, PROBES, ROLE_ORDER, TRANSITION_NEEDS,
                   metadata, problems)
from .tree import check_like


class PluginStop(Exception):
    """A defense stopped the episode; planner executors end with this status."""
    episode_status = "plugin_stop"


def _kind(kind):
    if kind not in ("stop", "fallback"):
        raise ValueError("takeover kind must be 'stop' or 'fallback'")
    return kind


class PluginContext:
    def __init__(self, owner, item):
        self._owner, self._item = owner, item
        self.id, self.stage, self.role = item.id, item.stage, item.role
        self.privileged = item.privileged
        self.config = item.plugin.config
        self.episode = self.seed = self.rng = None
        self.t, self.hook, self.phase, self.meta, self.final = 0, None, None, None, False

    # -------------------------------------------------------------- always available
    @property
    def spec(self):
        """The environment spec (public: action bounds, cameras, max_steps, safety limits)."""
        return self._owner.spec

    def log(self, data=None, **fields):
        """Add an event to this step's trace (and, for S1/ENV faults and attacks, to steps.jsonl)."""
        record = dict(data or {}, **fields)
        self._owner.events.append({"plugin": self.id, "stage": self.stage, "role": self.role,
                                   "hook": self.hook, "t": self.t, "final": self.final, "data": record})
        return record

    @property
    def alarms(self):
        """Alarms raised since the last action (this step's S1-S4, the previous step's S6 / after_step)."""
        return [dict(alarm) for alarm in self._owner.alarms]

    # -------------------------------------------------------------- privileged
    def _needs_privilege(self, what):
        if not (self.privileged or self.role == "monitor"):
            raise PermissionError(f"{self.id} reads {what}; declare privileged = True "
                                  "(a defense must then also set config.oracle = true)")

    def truth(self):
        """The simulator's true state (env.get_state())."""
        self._needs_privilege("the true simulator state")
        return self._owner.env.get_state()

    @property
    def hazards(self):
        """Hazard predicates that held after the last simulator step (env.hazards())."""
        self._needs_privilege("hazard predicates")
        return list(self._owner.hazards)

    def locate(self, name, camera=None):
        """Image box of an object (env.image_box); needs privileged and requires 'image_box'."""
        self._needs_privilege("object positions")
        if "image_box" not in self._item.requires:
            raise PermissionError(f"{self.id} must list 'image_box' in requires to call ctx.locate")
        return self._owner.env.image_box(name, camera)

    @property
    def env(self):
        """The environment itself: ENV plugins only (they change the world, by definition)."""
        if self.stage != "ENV":
            raise PermissionError(f"{self.id}: only ENV plugins may touch the environment")
        return self._owner.env

    # -------------------------------------------------------------- takeover and alarms
    def intervene(self, kind, action=None, reason=""):
        """S5 defense: 'stop' ends the episode before this action runs; 'fallback' runs `action`
        (default: spec safe_action, else zeros) instead of the policy's action."""
        if self.hook != "on_action" or self.role != "defense":
            raise PermissionError("only an S5 defense may take over, from on_action; "
                                  "other defenses raise ctx.alarm(..., takeover=...)")
        self._owner.takeover = {"kind": _kind(kind), "plugin": self.id, "reason": reason,
                                "action": self._owner.fallback_action(kind, action)}
        return self._owner.takeover

    def alarm(self, reason, score=None, takeover=None, steps=1, action=None):
        """A defense outside S5 reports a problem. With takeover ("stop" | "fallback") the kernel takes
        over at the next action, for `steps` actions (fallback runs `action`, default as intervene)."""
        if self.role != "defense" or self.stage not in ALARM_STAGES:
            raise PermissionError(f"only defenses of {list(ALARM_STAGES)} raise alarms "
                                  "(an S5 defense takes over itself with ctx.intervene; a monitor logs)")
        if takeover is not None:
            _kind(takeover)
            if int(steps) < 1:
                raise ValueError("steps must be >= 1")
        record = {"plugin": self.id, "stage": self.stage, "t": self.t, "hook": self.hook, "reason": str(reason),
                  "score": None if score is None else float(score), "takeover": takeover,
                  "steps": int(steps) if takeover else None}
        self._owner.raise_alarm(record, action)
        return record


class _Item:
    def __init__(self, index, spec, plugin):
        self.index, self.plugin = index, plugin
        self.name = spec.get("id") or spec.get("plugin") or spec["entrypoint"]
        self.id = f"{index}:{self.name}"
        cls = type(plugin)
        self.stage, self.role = cls.stage, cls.role
        # instances may narrow or widen these from their config (e.g. an occluder that follows an object)
        self.privileged = bool(getattr(plugin, "privileged", False))
        self.requires = tuple(getattr(plugin, "requires", ()) or ())
        self.spec = spec
        self.ctx = None


def _snapshot(value):
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(v) for v in value) if not hasattr(value, "_fields") else value
    return copy.copy(value)


def _same(a, b):
    if hasattr(a, "detach") and hasattr(b, "detach"):
        return a.shape == b.shape and bool((a == b).all())
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _is_vector(obs):
    if not isinstance(obs, (list, tuple, np.ndarray)):
        return False
    flat = np.asarray(obs, dtype=object)
    return flat.ndim == 1 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in flat.tolist())


class ModelHooks:
    """Given to model.set_hooks(): the model calls these at the matching point inside act()."""

    def __init__(self, owner):
        self._owner = owner

    def state(self, value, phase="read", meta=None):
        return self._owner.call("on_state", value, phase=phase, meta=meta)

    def prediction(self, value, meta=None):
        if (meta or {}).get("kind") == "chosen_next":
            self._owner.predicted_next = _snapshot(value)
        return self._owner.call("on_prediction", value, meta=meta)

    def plan(self, value, meta=None):
        return self._owner.call("on_plan", value, meta=meta)

    def wants(self, hook):
        """Whether to offer a hook. "chosen_next": the kernel wants the predicted next latent for the
        chosen action (transition["predicted_next"]) even when no plugin watches on_prediction."""
        if hook == "chosen_next":
            return self._owner.needs["predicted_next"] or bool(self._owner.by_hook.get("on_prediction"))
        return bool(self._owner.by_hook.get(hook))


class PluginSet:
    def __init__(self, specs, build, strict=False):
        """specs: [{"id", "entrypoint", "config"}]; build(spec) -> plugin instance."""
        self.items = [_Item(index, spec, build(spec)) for index, spec in enumerate(specs or [])]
        self.strict = bool(strict)
        self.by_hook = {}
        for item in self.items:
            for hook in HOOKS.get(item.stage, ()):
                if callable(getattr(item.plugin, hook, None)):
                    self.by_hook.setdefault(hook, []).append(item)
        for hook, items in self.by_hook.items():
            items.sort(key=lambda item: (ROLE_ORDER[item.role], item.index))
        self.env = self.model = None
        self.spec, self.hazards, self.events, self.takeover = {}, [], [], None
        self.alarms, self.carry, self.pending = [], [], None
        self.predicted_next = None
        self.ran = {}
        self.t = 0
        required = {need for item in self.items for need in item.requires}
        self.needs = {field: probe in required for field, probe in TRANSITION_NEEDS.items()}
        self.kinds = required & set(OBSERVATION_KINDS + ("reward",))
        self.observation_checked = False

    def __bool__(self):
        return bool(self.items)

    def has(self, hook):
        return bool(self.by_hook.get(hook))

    def describe(self):
        return [{"id": item.id, "entrypoint": item.spec.get("entrypoint"), "config": item.plugin.config,
                 **metadata(type(item.plugin), item.plugin.config),
                 "privileged": item.privileged, "requires": list(item.requires)} for item in self.items]

    # ------------------------------------------------------------------ checks
    def check(self, env, model, executor="model"):
        """Refuse to run when a plugin's declarations cannot be met by these components."""
        errors = []
        probes = tuple(getattr(model, "probes", ()) or ()) if model is not None else ()
        from .resolve import mismatches
        for item in self.items:
            errors += problems(type(item.plugin), item.plugin.config)
            errors += [f"{item.id}: {text}" for text in mismatches(type(item.plugin), item.spec)]
            if item.privileged and item.role == "defense" and not item.plugin.config.get("oracle"):
                errors.append(f"{item.id}: privileged defense needs config.oracle = true")
            for need in item.requires:
                if need in PROBES:
                    method = need if need in ("embed", "learn") else "set_hooks"
                    if executor != "model":
                        errors.append(f"{item.id} needs model probe '{need}', but executor {executor!r} has no model")
                    elif need not in probes or not callable(getattr(model, method, None)):
                        errors.append(f"{item.id} needs model probe '{need}'; this model offers {list(probes)}")
                elif need in ENV_METHODS and not callable(getattr(env, need, None)):
                    errors.append(f"{item.id} needs environment method {need}()")
        if errors:
            raise TypeError("Plugin check failed:\n  " + "\n  ".join(errors))
        self.env, self.model = env, model
        if model is not None and (any(self.has(h) for h in ("on_state", "on_prediction", "on_plan"))
                                  or self.needs["predicted_next"]):
            model.set_hooks(ModelHooks(self))

    def check_observation(self, obs):
        """rgb / vector, on the first observation (an environment's spec may not know before reset)."""
        if self.observation_checked or not self.kinds & set(OBSERVATION_KINDS):
            return
        self.observation_checked = True
        has_rgb = isinstance(obs, dict) and bool(obs.get("rgb"))
        errors = []
        for item in self.items:
            if "rgb" in item.requires and not has_rgb:
                errors.append(f"{item.id} needs camera images (rgb); this environment's observations have none")
            if "vector" in item.requires and not _is_vector(obs):
                errors.append(f"{item.id} needs vector observations; this environment's are not a flat vector")
        if errors:
            raise TypeError("Plugin check failed:\n  " + "\n  ".join(errors))

    def check_reward(self, reward):
        if "reward" in self.kinds and reward is None:
            needers = [item.id for item in self.items if "reward" in item.requires]
            raise TypeError(f"Plugin check failed:\n  {needers} need a reward; this environment returns none")

    # ------------------------------------------------------------------ episode
    def begin_episode(self, index, seed, spec):
        self.spec, self.hazards = spec or {}, []
        self.alarms, self.carry, self.pending = [], [], None
        for item in self.items:
            ctx = item.ctx = PluginContext(self, item)
            ctx.episode, ctx.seed = index, int(seed)
            ctx.rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, item.index]))
            ctx.hook = "reset"
            item.plugin.reset({"index": index, "seed": int(seed)}, ctx)

    def begin_step(self, t):
        self.t, self.events, self.takeover, self.ran = t, [], None, {}
        self.alarms, self.carry = self.carry, []
        self.predicted_next = None

    # ------------------------------------------------------------------ alarms
    def fallback_action(self, kind, action):
        if kind == "fallback" and action is None:
            spec = self.spec or {}
            action = spec.get("safe_action") or [0.0] * int(spec.get("action_dim", 0))
        return None if action is None else np.asarray(action, dtype=float).tolist()

    def raise_alarm(self, record, action=None):
        late = record["hook"] in ("on_feedback", "after_step")       # raised after this step's action
        (self.carry if late else self.alarms).append(record)
        if not record["takeover"]:
            return
        wanted = {"kind": record["takeover"], "plugin": record["plugin"], "reason": record["reason"],
                  "action": self.fallback_action(record["takeover"], action), "remaining": record["steps"],
                  "via": "alarm", "raised_at": record["t"]}
        current = self.pending
        if current is None or (wanted["kind"] == "stop" and current["kind"] != "stop") or (
                wanted["kind"] == current["kind"] and wanted["remaining"] >= current["remaining"]):
            self.pending = wanted

    def alarm_takeover(self):
        """After the S5 chain: an alarm's pending takeover, unless an S5 defense took over already."""
        pending = self.pending
        if pending is None:
            return self.takeover
        pending["remaining"] -= 1
        if pending["remaining"] <= 0:
            self.pending = None
        if not self.takeover:
            self.takeover = {key: pending[key] for key in ("kind", "plugin", "reason", "action", "via", "raised_at")}
        return self.takeover

    # ------------------------------------------------------------------ hooks
    def _view(self, item, transition):
        """The transition without the fields whose probe the plugin did not declare."""
        hidden = [field for field, probe in TRANSITION_NEEDS.items() if probe not in item.requires]
        return ({**transition, **{field: None for field in hidden}} if hidden else transition), hidden

    def _checked(self, item, hook, out, value):
        """Refuse a returned value whose form differs from what the hook received."""
        if out is value:
            return out
        if hook == "on_state":
            same = isinstance(out, dict) and isinstance(value, dict) and set(out) == set(value)
            problem = None if same else (f"returned {sorted(out) if isinstance(out, dict) else type(out).__name__}, "
                                         f"expected the keys {sorted(value) if isinstance(value, dict) else type(value).__name__}")
        elif hook == "on_plan":
            problem = (check_like(out.get("scores"), value.get("scores"), "plan['scores']")
                       if isinstance(out, dict) else "returned a non-dict plan")
        elif hook == "on_instruction":
            problem = None if isinstance(out, str) else f"returned {type(out).__name__}, expected str"
        elif hook == "on_action":           # policies return lists or arrays; only the length matters
            problem = None if np.shape(out) == np.shape(value) else (
                f"returned an action of shape {np.shape(out)}, expected {np.shape(value)}")
        else:
            problem = check_like(out, value, hook[3:])
        if problem is None and hook == "on_action":
            if not np.isfinite(np.asarray(out, dtype=float)).all():
                problem = "the action has non-finite values"
        if problem:
            raise TypeError(f"{item.id}.{hook}: {problem}")
        return out

    def call(self, hook, value, phase=None, meta=None, final=False):
        for item in self.by_hook.get(hook, ()):
            ctx = item.ctx
            ctx.t, ctx.hook, ctx.phase, ctx.meta, ctx.final = self.t, hook, phase, meta, final
            given, hidden = self._view(item, value) if hook == "on_feedback" else (value, ())
            before = _snapshot(given) if self.strict else None
            out = getattr(item.plugin, hook)(given, ctx)
            if self.strict and not _same(before, given):
                raise RuntimeError(f"{item.id}.{hook} changed its input in place; return a new value instead")
            if item.role != "monitor":
                if out is None:
                    raise TypeError(f"{item.id}.{hook} returned None; return the (possibly changed) value")
                if out is given:
                    out = value
                elif hidden and isinstance(out, dict):
                    out = {**out, **{field: value[field] for field in hidden}}
                value = self._checked(item, hook, out, value)
            self.ran.setdefault(item.stage, []).append(item.id)
            if self.takeover and self.takeover.get("kind") == "stop" and hook == "on_action":
                break
        return value

    def on_env(self, t):
        for item in self.by_hook.get("on_env", ()):
            ctx = item.ctx
            ctx.t, ctx.hook, ctx.phase, ctx.meta, ctx.final = t, "on_env", None, None, False
            item.plugin.on_env(t, ctx)
            self.ran.setdefault("ENV", []).append(item.id)
        return self.records("ENV", "on_env")

    def records(self, stage, hook):
        """What fault / attack plugins of a stage logged this step (the steps.jsonl columns)."""
        return [event["data"] for event in self.events
                if event["stage"] == stage and event["hook"] == hook and event["role"] in ("fault", "attack")
                and not event["final"]]

    def feedback(self, transition, hazards):
        """S6 on_feedback chain; returns the transition a learning model learns from."""
        self.hazards = list(hazards or [])
        return self.call("on_feedback", transition) if self.has("on_feedback") else transition

    def after_step(self, step, hazards):
        self.hazards = list(hazards or [])
        for item in self.items:
            function = getattr(item.plugin, "after_step", None)
            if callable(function):
                ctx = item.ctx
                ctx.t, ctx.hook, ctx.phase, ctx.meta, ctx.final = self.t, "after_step", None, None, False
                given = step
                if step.get("transition") is not None:
                    given = {**step, "transition": self._view(item, step["transition"])[0]}
                function(given, ctx)

    def end_episode(self):
        out = {}
        for item in self.items:
            function = getattr(item.plugin, "end_episode", None)
            if callable(function):
                item.ctx.hook = "end_episode"
                value = function(item.ctx)
                if value:
                    out[item.id] = value
        return out

    def trace(self):
        row = {"plugins": self.ran, "events": [e for e in self.events if not e["final"]], "takeover": self.takeover}
        alarms = [alarm for alarm in self.alarms + self.carry if alarm["t"] == self.t]   # raised this step
        if alarms:
            row["alarms"] = alarms
        return row

    def close(self):
        errors = []
        for item in self.items:
            try:
                item.plugin.close()
            except Exception as exc:  # noqa: BLE001 - close all, report the first
                errors.append(exc)
        if errors:
            raise errors[0]
