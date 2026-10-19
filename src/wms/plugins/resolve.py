"""Turn a recipe's params.plugins into plugin specs and check them before a run is queued.

A recipe lists plugins by registry name or entrypoint:

  "params": {"plugins": [
      {"plugin": "toy-action-noise", "config": {"std": 0.2}},
      {"entrypoint": "my_pkg.plugins:Gate", "config": {"oracle": true}}]}

Registry entries live in any recipes.json under "plugins":
  "toy-action-noise": {"entrypoint": "wms.plugins.examples:ActionNoise", "config": {"std": 0.1},
                       "stage": "S5", "role": "fault", "description": "..."}

The service checks a plugin by importing its class. The service's Python has no numpy or
model libraries, so when the import fails for a missing module it checks the entry's own
declaration instead (stage, role, side, privileged, requires, threat); the kernel later
refuses to run if the class disagrees with that declaration.

Models and environments may declare in their registry entries what plugins can rely on,
so a mismatch is refused at submission instead of after the GPU job starts:
  model entry        "probes": ["state", "predict", "plan"]
  environment entry  "capabilities": ["image_box", "set_object_pose", "set_goal", "hazards"]
The kernel checks the loaded objects again at run time; that check is authoritative.
"""
from __future__ import annotations

from copy import deepcopy

from .base import ENV_ABILITIES, PROBES, Plugin, problems

DECLARED = ("stage", "role", "side", "privileged", "requires", "threat")


def _load_class(entrypoint, paths=()):
    import sys
    from wms.runners.unified import load_entrypoint
    added = [str(p) for p in paths if str(p) not in sys.path]
    sys.path[:0] = added
    try:
        return load_entrypoint(entrypoint)
    finally:
        for path in added:
            sys.path.remove(path)


def resolve_specs(items, plugins):
    """items: params.plugins; plugins: registry "plugins" group. Returns [{"id", "entrypoint", "config"}]."""
    if items in (None, []):
        return []
    if not isinstance(items, list):
        raise ValueError("params.plugins must be a list")
    out = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            item = {"plugin": item}
        if not isinstance(item, dict):
            raise ValueError(f"params.plugins[{index}] must be a name or an object")
        base = {}
        if item.get("plugin") is not None:
            name = item["plugin"]
            if name not in plugins:
                raise ValueError(f"params.plugins[{index}]: unknown plugin {name!r}; known: {sorted(plugins)}")
            base = deepcopy(plugins[name])
        entrypoint = item.get("entrypoint") or base.get("entrypoint")
        if not entrypoint:
            raise ValueError(f"params.plugins[{index}] needs plugin (a registry name) or entrypoint")
        config = {**base.get("config", {}), **item.get("config", {})}
        spec = {"id": item.get("id") or item.get("plugin") or entrypoint, "entrypoint": entrypoint, "config": config}
        declared = {key: base[key] for key in DECLARED if key in base}
        if declared:
            spec["declared"] = declared
        if base.get("_python_paths"):
            spec["python_paths"] = list(base["_python_paths"])
        out.append(spec)
    return out


def declared_class(spec):
    """A stand-in class built from a registry entry's declaration (hooks of its stage assumed)."""
    from .base import HOOKS, NOT_YET
    declared = spec.get("declared") or {}
    fields = {key: declared[key] for key in DECLARED if key in declared}
    fields["requires"] = tuple(fields.get("requires") or ())
    hooks = [hook for hook in HOOKS.get(fields.get("stage"), ()) if hook not in NOT_YET]
    fields.update({hook: (lambda self, value, ctx: value) for hook in hooks})
    fields["__module__"] = spec["entrypoint"].partition(":")[0]
    return type(spec["entrypoint"].rpartition(":")[2] or "Declared", (Plugin,), fields)


def mismatches(cls, spec):
    """Where a loaded class disagrees with its registry declaration."""
    out = []
    for key, value in (spec.get("declared") or {}).items():
        actual = getattr(cls, key, None)
        if key == "requires":
            actual, value = sorted(actual or ()), sorted(value or ())
        if key == "threat":
            continue                     # a config may refine the threat card
        if actual != value:
            out.append(f"{key}: registry says {value!r}, class says {actual!r}")
    return out


def check(config):
    """Problems with a resolved run configuration's plugins (empty list = fine)."""
    params = config.get("params") or {}
    specs = params.get("plugins") or []
    if not specs:
        return []
    out = []
    model, env = config.get("model") or {}, config.get("environment") or {}
    probes = model.get("probes")
    abilities = env.get("capabilities")
    executor = params.get("executor", "model")
    for spec in specs:
        try:
            cls = _load_class(spec["entrypoint"], spec.get("python_paths", ()))
        except ModuleNotFoundError as exc:
            if not spec.get("declared", {}).get("stage"):
                out.append(f"plugin {spec['id']}: {spec['entrypoint']} needs {exc.name!r}, which the service "
                           "does not have; declare stage and role (and requires, privileged, threat) in its "
                           "registry entry so it can be checked before the run")
                continue
            cls = declared_class(spec)
        except Exception as exc:  # noqa: BLE001 - report import problems as configuration errors
            out.append(f"plugin {spec['id']}: cannot import {spec['entrypoint']}: {type(exc).__name__}: {exc}")
            continue
        out += [f"plugin {spec['id']}: {text}" for text in mismatches(cls, spec)]
        out += [f"plugin {spec['id']}: {text}" for text in problems(cls, spec.get("config"))]
        for need in getattr(cls, "requires", ()) or ():
            if need in PROBES:
                if executor != "model":
                    out.append(f"plugin {spec['id']} needs model probe '{need}'; executor {executor!r} has no model")
                elif probes is not None and need not in probes:
                    out.append(f"plugin {spec['id']} needs model probe '{need}'; the model declares {probes}")
                elif probes is None:
                    out.append(f"plugin {spec['id']} needs model probe '{need}'; the model entry declares no probes")
            elif need in ENV_ABILITIES and abilities is not None and need not in abilities:
                out.append(f"plugin {spec['id']} needs environment ability '{need}'; the environment declares {abilities}")
    return out
