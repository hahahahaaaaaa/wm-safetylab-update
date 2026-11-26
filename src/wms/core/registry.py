"""Load independent recipe files and resolve a run's machine configuration."""
from copy import deepcopy
import json
from pathlib import Path, PureWindowsPath
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GROUPS = ("models", "environments", "algorithms", "runtimes", "plugins", "recipes")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def merge(base, override):
    value = deepcopy(base)
    for key, item in override.items():
        value[key] = merge(value[key], item) if isinstance(item, dict) and isinstance(value.get(key), dict) else deepcopy(item)
    return value


def _path(value, base):
    """Resolve relative references without rewriting another OS's absolute path."""
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute() or str(value).startswith("/"):
        return str(value)
    return str((base / value).resolve())


def _runtime_paths(value, base):
    value = deepcopy(value)
    for key in ("config_file", "cwd"):
        if value.get(key):
            value[key] = _path(value[key], base)
    if value.get("python") and ("/" in value["python"] or "\\" in value["python"]):
        value["python"] = _path(value["python"], base)
    if "python_paths" in value:
        value["python_paths"] = [_path(p, base) for p in value["python_paths"]]
    return value


def _bind_runtime(value, runtime):
    """Two explicit references for separately constructed components."""
    if isinstance(value, dict):
        return {key: _bind_runtime(item, runtime) for key, item in value.items()}
    if isinstance(value, list):
        return [_bind_runtime(item, runtime) for item in value]
    if value == "$runtime.env":
        return deepcopy(runtime.get("env", {}))
    if value == "$runtime.config_file":
        if not runtime.get("config_file"):
            raise ValueError("$runtime.config_file requires a selected runtime.config_file")
        return runtime["config_file"]
    return value


class Registry:
    """A collection of names, not a compatibility engine or permission system."""

    def __init__(self, paths=(), python=None, native_configs=None):
        self.python = python or sys.executable
        self.items = {group: {} for group in GROUPS}
        self.sources = {group: {} for group in GROUPS}
        self.loaded_paths = set()
        self.native_configs = {key: _path(value, PROJECT_ROOT) for key, value in (native_configs or {}).items()}
        # Explicit legacy server mappings remain valid. This is not discovery:
        # profiles only select machine files for recipes declared independently.
        collection = self.native_configs.get("world_model")
        if collection and Path(collection).is_file():
            for profile in read_json(collection).get("profiles", []):
                path = _path(profile, Path(collection).parent)
                if Path(path).is_file():
                    key = "predict-" + read_json(path)["adapter"]
                    self.native_configs.setdefault(key, path)
        example = PROJECT_ROOT / "examples/minimal/registry.json"
        if example.exists():
            self.load(example)
        for path in sorted((PROJECT_ROOT / "native").glob("*/recipes.json")):
            self.load(path)
        for path in paths:
            self.load(path)

    def load(self, path):
        path = Path(path).resolve()
        if path in self.loaded_paths:
            return self
        data = read_json(path)
        if "specs" in data or "bridges" in data:
            raise ValueError(f"{path}: use the simple registry format documented in docs/DEVELOPMENT.zh-CN.md")
        extra = [_path(p, path.parent) for p in data.get("python_paths", [])]
        pending = []
        for group in GROUPS:
            entries = data.get(group, {})
            entries = list(entries.items()) if isinstance(entries, dict) else [(item["id"], item) for item in entries]
            seen = set()
            for name, original in entries:
                if name in self.items[group] or name in seen:
                    first = self.sources[group].get(name, str(path))
                    raise ValueError(f"Duplicate {group} ID '{name}': {first} and {path}")
                seen.add(name)
                entry = deepcopy(original)
                if group == "recipes":
                    entry["_python_paths"] = extra + [_path(p, path.parent) for p in entry.get("_python_paths", [])]
                    if isinstance(entry.get("runtime"), dict):
                        entry["runtime"] = _runtime_paths(entry["runtime"], path.parent)
                elif group == "runtimes":
                    entry = _runtime_paths(entry, path.parent)
                elif group == "plugins":
                    entry["_python_paths"] = list(extra)
                pending.append((group, name, entry))
        for group, name, entry in pending:
            self.items[group][name] = entry
            self.sources[group][name] = str(path)
        self.loaded_paths.add(path)
        return self

    def recipes(self):
        values = []
        for key, value in self.items["recipes"].items():
            item = {"id": key, **{field: value.get(field, "") for field in ("name", "task", "direction", "description")}}
            resolved = self.recipe(key)
            item.update(configured=resolved["configured"], configuration_error=resolved.get("configuration_error"))
            values.append(item)
        return values

    def recipe(self, name):
        return self.resolve({"recipe": name}, allow_unconfigured=True)

    def component(self, group, value):
        return deepcopy(self.items[group][value] if isinstance(value, str) else value)

    def _runtime(self, recipe, request):
        base = self.component("runtimes", recipe.get("runtime", {}))
        explicit = self.component("runtimes", request.get("runtime", {}))
        # API paths are relative to the project. Files loaded by load() have
        # already had references resolved relative to their declaring JSON.
        explicit = _runtime_paths(explicit, PROJECT_ROOT)
        native_key = explicit.get("native_key", base.get("native_key"))
        supplied = request.get("params", {})
        alias = supplied.get("profile", supplied.get("runtime_config"))
        if "config_file" in explicit:
            selected = explicit["config_file"]
        elif isinstance(alias, str):
            selected = _path(alias, PROJECT_ROOT)
        else:
            selected = self.native_configs.get(native_key, base.get("config_file"))
        defaults, error = {}, None
        if selected:
            path = Path(selected)
            if not path.is_file():
                error = f"Machine configuration not found: {selected}"
            else:
                try:
                    settings = read_json(path)
                    if not isinstance(settings, dict):
                        raise ValueError("Machine configuration must be a JSON object")
                    if not isinstance(settings.get("env", {}), dict):
                        raise ValueError("Machine configuration env must be an object")
                    defaults = _runtime_paths({key: settings[key] for key in ("python", "env", "cwd", "python_paths", "timeout_s") if key in settings}, path.parent)
                    device = settings.get("cuda_visible_devices", settings.get("gpu_device"))
                    if device is not None:
                        defaults["env"] = {"CUDA_VISIBLE_DEVICES": str(device), **defaults.get("env", {})}
                except (ValueError, TypeError, OSError) as exc:
                    error = f"Cannot read machine configuration {selected}: {exc}"
        runtime = merge(merge(defaults, base), explicit)
        if selected:
            runtime["config_file"] = str(selected)
        else:
            runtime.pop("config_file", None)
        if not runtime.get("python") or runtime["python"] == "auto":
            runtime["python"] = self.python
        runtime.setdefault("env", {})
        return runtime, error

    def resolve(self, config, *, allow_unconfigured=False):
        if not isinstance(config, dict):
            raise ValueError("An experiment configuration must be a JSON object")
        request = deepcopy(config)
        name = request.pop("recipe", None)
        recipe = deepcopy(self.items["recipes"][name]) if name else {}
        config = merge(recipe, request)
        if not config.get("task"):
            raise ValueError("Configuration needs task: generate, predict, rollout, plan, native or evaluate")
        runtime, error = self._runtime(recipe, request)
        runtime["python_paths"] = list(dict.fromkeys(runtime.get("python_paths", []) + config.pop("_python_paths", [])))
        config["runtime"] = runtime
        config["configured"] = error is None
        config.pop("configuration_error", None)
        if error:
            config["configuration_error"] = error
            if not allow_unconfigured:
                raise ValueError(error)
        for key, group in (("model", "models"), ("environment", "environments"), ("algorithm", "algorithms")):
            if config.get(key):
                config[key] = self.component(group, config[key])
        config = _bind_runtime(config, runtime)
        if (config.get("params") or {}).get("plugins"):
            from wms.plugins.resolve import check, resolve_specs
            try:
                config["params"]["plugins"] = resolve_specs(config["params"]["plugins"], self.items["plugins"])
                paths = [p for spec in config["params"]["plugins"] for p in spec.get("python_paths", [])]
                config["runtime"]["python_paths"] = list(dict.fromkeys(config["runtime"]["python_paths"] + paths))
                plugin_errors = check(config)
            except ValueError as exc:
                plugin_errors = [str(exc)]
            if plugin_errors:
                message = "Plugin check failed: " + "; ".join(plugin_errors)
                if not allow_unconfigured:
                    raise ValueError(message)
                config["configured"] = False
                config["configuration_error"] = "; ".join(filter(None, [config.get("configuration_error"), message]))
        for key, value in {"name": config["task"], "direction": "control", "inputs": {}, "params": {},
                           "seed": 0, "steps": 8, "metrics": [], "tags": []}.items():
            config.setdefault(key, value)
        return config
