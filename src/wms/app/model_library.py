"""Model workbench: resource readiness is separate from successful execution."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from fastapi import APIRouter, Body
from wms.core.registry import PROJECT_ROOT, merge


def resource_issues(config):
    """Inspect declared resources without importing a model or starting CUDA."""
    issues, visited = [], set()
    runtime = config.get("runtime", {})
    python = runtime.get("python", "")
    if not (Path(python).is_file() if "/" in python or "\\" in python else shutil.which(python)):
        issues.append("Python 环境不存在")
    path_keys = {"source_dir", "source_root", "checkpoint_dir", "checkpoint", "model_dir", "weights_dir",
                 "simpler_dir", "memoryvla_dir", "tdmpc2_dir", "assets_dir", "llama_config_dir",
                 "model_path", "tokenizer_path", "dataset_root", "episode_checkpoint", "asset_dir"}

    def inspect(path):
        path = Path(path).resolve()
        if path in visited:
            return
        visited.add(path)
        if not path.is_file():
            issues.append("缺少运行配置：" + path.name)
            return
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("Runtime configuration must be an object")
            for key in path_keys:
                item = value.get(key)
                if isinstance(item, str):
                    target = Path(item)
                    if not target.is_absolute():
                        target = path.parent / target
                    if not target.exists():
                        issues.append("缺少资源：" + key + " / " + target.name)
            if isinstance(value.get("config"), str):
                target = Path(value["config"])
                inspect(target if target.is_absolute() else path.parent / target)
            return value
        except (ValueError, OSError, TypeError):
            issues.append("运行配置无法读取：" + path.name)

    machine = inspect(runtime["config_file"]) if runtime.get("config_file") else {}
    provider = config.get("params", {}).get("provider")
    if provider in {"chat", "jev"}:
        # Use the adapter's connection validation; only report the exception type,
        # never return credentials or the raw machine configuration to the browser.
        from wms.engines.embodied_jev.connections import build_connection
        try:
            connection = build_connection(machine or {}, provider,
                config.get("params", {}).get("connection_profile"),
                base_dir=Path(runtime["config_file"]).parent, environ={**os.environ, **runtime.get("env", {})})
            if not connection:
                issues.append("尚未配置模型连接")
        except (ValueError, KeyError, OSError, RuntimeError):
            issues.append("尚未配置可用的 API 凭据或连接配置")
    return list(dict.fromkeys(issues))


def catalog(registry, runs=()):
    metadata = json.loads((PROJECT_ROOT / "native/model-library.json").read_text(encoding="utf-8"))
    local_path = PROJECT_ROOT / "model-library.local.json"
    local = json.loads(local_path.read_text()) if local_path.is_file() else {}
    result = []
    for item in registry.recipes():
        recipe_id = item["id"]
        raw = registry.items["recipes"][recipe_id]
        family = Path(registry.sources["recipes"][recipe_id]).parent.name
        details = metadata["families"].get(family, {"name": family, "scope": "接口示例", "stages": []})
        resolved = registry.recipe(recipe_id)
        issues = resource_issues(resolved) if resolved["configured"] else ["缺少或无效的机器配置"]
        template = {"seed": raw.get("seed", 0), "inputs": deepcopy(raw.get("inputs", {})),
                    "params": deepcopy(raw.get("params", {}))}
        # Machine-bound values must be resolved server-side, not sent back by UI.
        for key in ("runtime_config", "profile", "adapter_entrypoint"):
            template["params"].pop(key, None)
        template = merge(template, metadata.get("smoke", {}).get(recipe_id, {}))
        template = merge(template, {k: v for k, v in local.get(recipe_id, {}).items()
                                    if k in {"inputs", "params", "seed", "timeout_s", "steps"}})
        latest = next((r for r in runs if r.get("config", {}).get("workbench_recipe") == recipe_id), None)
        stages = []
        if resolved.get("entrypoint") == "wms.episodes.kernel:run":
            probes = resolved.get("model", {}).get("probes", [])
            stages = ["S1"] + [stage for probe, stage in (("state", "S2"), ("predict", "S3"), ("plan", "S4"))
                                 if probe in probes] + ["S5", "S6"]
        result.append({**{k: v for k, v in item.items() if k != "configuration_error"},
            "family": family, "family_name": details["name"], "scope": details["scope"],
            "stages": stages,
            "notes": details.get("notes", ""), "ready": not issues, "issues": issues,
            "availability_note": local.get(recipe_id, {}).get("availability_note", "")
                if not latest or latest.get("state") != "completed" else "",
            "template": template,
            "latest_run": {k: latest.get(k) for k in ("id", "state", "created_at", "evaluation_status")} if latest else None})
    return result


def create_model_library_router(services, registry):
    router = APIRouter(prefix="/api/v1/model-library")

    @router.get("")
    def models():
        return {"items": catalog(registry, services["store"].list_workbench_runs())}

    @router.get("/runs")
    def runs():
        items = []
        for run in services["store"].list_workbench_runs():
            public = {key: run.get(key) for key in ("id", "name", "task", "state", "created_at", "result", "error", "evaluation_status")}
            public["config"] = {"name": run["name"], "task": run["task"], "runtime": {},
                                "workbench_recipe": run["config"]["workbench_recipe"]}
            items.append(public)
        return {"items": items}

    @router.post("/{recipe_id}/runs")
    def submit(recipe_id: str, options: dict = Body(default={})):
        allowed = {"inputs", "params", "seed", "name", "timeout_s", "steps"}
        if set(options) - allowed:
            raise ValueError("模型工作台仅接受输入、参数、种子、名称、步数和超时")
        for key in ("inputs", "params"):
            if key in options and not isinstance(options[key], dict):
                raise ValueError(key + " must be an object")
        if "seed" in options and (type(options["seed"]) is not int or options["seed"] < 0):
            raise ValueError("seed must be a nonnegative integer")
        config = registry.resolve({"recipe": recipe_id, **options, "workbench_recipe": recipe_id})
        issues = resource_issues(config)
        if issues:
            raise ValueError("；".join(issues))
        return services["runner"].submit(config)

    return router
