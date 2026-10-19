"""wms plugin test <ID>: the checks every plugin must pass before review, on the CPU toys (seconds).

    python -m wms.plugins.conformance <registry ID | module:Class> [--config JSON] [--registry FILE ...]

Needs a Python with numpy (the toy runtime's). `wms plugin test` in the platform's own
environment forwards to it. Checks:

  declaration     stage, role, hooks, requires, privileged, threat card are consistent
                  (wms.plugins.base), and the registry entry agrees with the class
  on every toy whose probes and abilities cover the plugin's requires (a plugin that should
  work on any model must work on all of them):
    runs          two toy episodes with the plugin finish, in strict mode: no hook may change
                  its input in place, and every returned value keeps its input's form
    called        the plugin's hooks were called (trace.jsonl)
    reproducible  the same seeds give byte-identical steps.jsonl and trace.jsonl
    own randomness  the global random / numpy.random state is untouched (use ctx.rng)
    read-only     monitors only: every action equals the run without the plugin
    neutral       with the class's `neutral` config (its strength 0) the run equals the run
                  without the plugin: same steps.jsonl (apart from the plugin's own log
                  columns) and same frames. Faults and attacks must declare `neutral`.
    effect        others: on how many steps the executed action differs from the run without
                  the plugin (information, not pass/fail: a fault with no effect is legal)

Toys (model + environment):
  toy-centroid-memory + toy-pick          probe state; camera images
  toy-planner + toy-pick                  probes state, predict, plan, embed; camera images
  toy-planner + toy-pick-vector           the same planner on vector observations
  toy-planner-online + toy-pick           also learn: fits its dynamics online (S6 faults / attacks)
Passing says the plugin follows the contract; it says nothing about the plugin's effect on a
real model, which its task card's GPU run shows.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
TOY_ENVS = {
    "toy-pick": {"entrypoint": "wms.scenarios.toy.env:ToyPickEnv",
                 "capabilities": ["image_box", "set_object_pose", "set_goal", "hazards", "rgb", "reward"],
                 "config": {"image_size": 64, "max_steps": 60}},
    "toy-pick-vector": {"entrypoint": "wms.scenarios.toy.env:ToyPickEnv",
                        "capabilities": ["set_object_pose", "set_goal", "hazards", "vector", "reward"],
                        "config": {"image_size": 64, "max_steps": 60, "observation": "vector"}},
}
TOY_MODELS = {
    "toy-centroid-memory": {"entrypoint": "wms.scenarios.toy.policy:CentroidPickPolicy", "probes": ["state"],
                            "config": {"memory": True}},
    "toy-planner": {"entrypoint": "wms.scenarios.toy.planner:ToyPlannerPolicy",
                    "probes": ["state", "predict", "plan", "embed"], "config": {"samples": 32, "horizon": 4}},
    "toy-planner-online": {"entrypoint": "wms.scenarios.toy.planner:ToyOnlinePlannerPolicy",
                           "probes": ["state", "predict", "plan", "embed", "learn"],
                           "config": {"samples": 32, "horizon": 4, "initial_gain": 0.5}},
}
TOYS = [("toy-centroid-memory", "toy-pick"), ("toy-planner", "toy-pick"), ("toy-planner", "toy-pick-vector"),
        ("toy-planner-online", "toy-pick")]
OWN_COLUMNS = ("perturbations", "target_moves")          # what a plugin logs into steps.jsonl itself


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run(spec, model, env, output, seed, episodes, with_plugin=True):
    from wms.runners.unified import UnifiedWorker
    params = {"episodes": episodes, "video": False, "save_raw": False, "plugin_strict": True,
              "plugins": [spec] if with_plugin else []}
    config = {"task": "native", "entrypoint": "wms.episodes.kernel:run", "seed": seed,
              "model": model, "environment": env, "params": params}
    with contextlib.redirect_stdout(io.StringIO()):          # progress events
        code = UnifiedWorker(config, output).run()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    return code, result


def _episodes(output, episodes):
    return [(output / "episodes" / f"ep_{i:03d}") for i in range(episodes)]


def toys_for(cls):
    """The toy (model, environment) pairs whose probes and abilities cover the plugin's requires."""
    from .base import PROBES
    requires = set(getattr(cls, "requires", ()) or ())
    return [(m, e) for m, e in TOYS
            if requires & set(PROBES) <= set(TOY_MODELS[m]["probes"])
            and requires - set(PROBES) <= set(TOY_ENVS[e]["capabilities"])]


def _same_run(a, b, episodes):
    """Where run a differs from run b, ignoring the plugin's own log columns (None = identical)."""
    import numpy as np
    for ep_a, ep_b in zip(_episodes(a, episodes), _episodes(b, episodes)):
        rows_a, rows_b = _lines(ep_a / "steps.jsonl"), _lines(ep_b / "steps.jsonl")
        if len(rows_a) != len(rows_b):
            return f"{ep_a.name}: {len(rows_a)} steps, {len(rows_b)} without the plugin"
        for x, y in zip(rows_a, rows_b):
            for key in set(x) | set(y):
                if key not in OWN_COLUMNS and x.get(key) != y.get(key):
                    return f"{ep_a.name} step {x['t']}: {key} differs"
        for frames in sorted(ep_b.glob("frames_*.npz")):
            other = ep_a / frames.name
            if not other.exists() or not np.array_equal(np.load(other)["seen"], np.load(frames)["seen"]):
                return f"{ep_a.name}: {frames.name} differs"
    return None


def _check_toy(spec, cls, toy, output, seed, episodes):
    import numpy as np
    model_name, env_name = toy
    model, env = TOY_MODELS[model_name], TOY_ENVS[env_name]
    label = f"{model_name} + {env_name}"
    report = []
    state = (random.getstate(), np.random.get_state())
    code, result = _run(spec, model, env, output / "a", seed, episodes)
    touched = random.getstate() != state[0] or not all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(np.random.get_state(), state[1]))
    if code != 0:
        return [("FAIL", "runs", f"{label}: {result.get('error')}")]
    report.append(("PASS", "runs", f"{label}: {episodes} episodes, strict mode"))
    report.append(("FAIL" if touched else "PASS", "own randomness",
                   f"{label}: the global random / numpy.random state changed; draw from ctx.rng" if touched
                   else f"{label}: global state untouched"))

    plugin_id = f"0:{spec.get('id') or spec['entrypoint']}"
    calls = sum(plugin_id in ids for ep in _episodes(output / "a", episodes)
                for row in _lines(ep / "trace.jsonl") for ids in row.get("plugins", {}).values())
    report.append(("PASS" if calls else "FAIL", "called", f"{label}: {calls} hook calls" if calls else
                   f"{label}: no hook of this plugin was called (does the toy offer what it waits for?)"))

    code_b, _ = _run(spec, model, env, output / "b", seed, episodes)
    same = code_b == 0 and all((ep_a / name).read_bytes() == (ep_b / name).read_bytes()
                               for ep_a, ep_b in zip(_episodes(output / "a", episodes), _episodes(output / "b", episodes))
                               for name in ("steps.jsonl", "trace.jsonl"))
    report.append(("PASS" if same else "FAIL", "reproducible",
                   f"{label}: identical steps.jsonl and trace.jsonl" if same else
                   f"{label}: a second run with the same seeds differs"))

    code_c, result_c = _run(spec, model, env, output / "baseline", seed, episodes, with_plugin=False)
    if code_c != 0:
        report.append(("FAIL", "baseline", f"{label}: {result_c.get('error')}"))
        return report
    differ = total = 0
    for ep_a, ep_c in zip(_episodes(output / "a", episodes), _episodes(output / "baseline", episodes)):
        a, c = _lines(ep_a / "steps.jsonl"), _lines(ep_c / "steps.jsonl")
        total += max(len(a), len(c))
        differ += sum(x["action"] != y["action"] for x, y in zip(a, c)) + abs(len(a) - len(c))
    if cls.role == "monitor":
        report.append(("PASS" if differ == 0 else "FAIL", "read-only",
                       f"{label}: actions identical to the run without it" if differ == 0 else
                       f"{label}: a monitor changed the actions on {differ}/{total} steps"))
        return report
    report.append(("INFO", "effect", f"{label}: executed action differs on {differ}/{total} steps "
                   f"(success {result['data']['success_rate']:.2f} vs {result_c['data']['success_rate']:.2f} without)"))

    neutral = getattr(cls, "neutral", None)
    if neutral is None:
        if cls.role in ("fault", "attack"):
            report.append(("FAIL", "neutral", f"{label}: a {cls.role} must declare `neutral`, the config "
                           "that makes it a no-op (its strength 0), so a dose-response starts at the baseline"))
        return report
    zero = {**spec, "config": {**spec.get("config", {}), **neutral}}
    code_n, result_n = _run(zero, model, env, output / "neutral", seed, episodes)
    if code_n != 0:
        report.append(("FAIL", "neutral", f"{label}: {result_n.get('error')}"))
        return report
    where = _same_run(output / "neutral", output / "baseline", episodes)
    report.append(("FAIL" if where else "PASS", "neutral",
                   f"{label}: with {json.dumps(neutral)} the run differs from the run without the plugin ({where})"
                   if where else f"{label}: with {json.dumps(neutral)} identical to the run without the plugin"))
    return report


def check(spec, output, seed=0, episodes=2, toys=None):
    """[(status, name, detail)] with status PASS / FAIL / INFO."""
    from wms.runners.unified import load_entrypoint
    from .base import metadata, problems
    from .resolve import mismatches
    report = []
    try:
        cls = load_entrypoint(spec["entrypoint"])
    except Exception as exc:  # noqa: BLE001
        return [("FAIL", "declaration", f"cannot import {spec['entrypoint']}: {type(exc).__name__}: {exc}")]
    issues = problems(cls, spec.get("config")) + mismatches(cls, spec)
    report.append(("FAIL" if issues else "PASS", "declaration", "; ".join(issues) or json.dumps(
        {k: v for k, v in metadata(cls, spec.get("config")).items() if v not in (None, [], False)}, ensure_ascii=False)))
    if issues:
        return report
    pairs = [toy for toy in toys_for(cls) if toys is None or toy in toys]
    if not pairs:
        report.append(("FAIL", "toys", f"no toy offers all of {list(cls.requires)}; a plugin needs at least one "
                       "toy to be checked on (ask for a toy, or narrow requires)"))
        return report
    report.append(("INFO", "toys", "; ".join(f"{m} + {e}" for m, e in pairs)))
    for model_name, env_name in pairs:
        report += _check_toy(spec, cls, (model_name, env_name), output / f"{model_name}__{env_name}", seed, episodes)
    return report


def _spec(target, config, registries):
    from wms.core.registry import Registry
    from .resolve import resolve_specs
    registry = Registry(paths=registries)
    item = {"entrypoint": target} if ":" in target else {"plugin": target}
    return resolve_specs([{**item, "config": config}], registry.items["plugins"])[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check one plugin against the plugin contract on the CPU toys.")
    parser.add_argument("plugin", help="registry ID or module:Class")
    parser.add_argument("--config", default="{}", help="JSON object (or a JSON file) merged over the registry config")
    parser.add_argument("--registry", action="append", default=[], help="extra recipes.json / registry.json")
    parser.add_argument("--output", default=None, help="where the toy runs go (default test-output/plugin-test/<ID>)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args(argv)
    text = args.config.strip()
    if not text.startswith("{") and Path(text).is_file():          # a JSON file (easier to quote)
        text = Path(text).read_text(encoding="utf-8-sig")
    spec = _spec(args.plugin, json.loads(text), args.registry)
    for path in reversed(spec.get("python_paths", [])):
        sys.path.insert(0, path)
    output = Path(args.output or ROOT / "test-output" / "plugin-test" / re.sub(r"[^A-Za-z0-9_.-]+", "_", args.plugin))
    if output.exists() and any(output.iterdir()):
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    report = check(spec, output, args.seed, args.episodes)
    width = max(len(name) for _, name, _ in report)
    for status, name, detail in report:
        print(f"{status:4}  {name:<{width}}  {detail}")
    failed = any(status == "FAIL" for status, _, _ in report)
    (output / "report.json").write_text(json.dumps({"plugin": args.plugin, "spec": spec, "passed": not failed,
                                                    "checks": [dict(zip(("status", "check", "detail"), row)) for row in report]},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(("FAILED" if failed else "PASSED") + f"  (runs in {output})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
