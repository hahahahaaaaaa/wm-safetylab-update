"""Run one recipe directly with its runtime's Python, without the service (development and acceptance runs).

    python scripts/run_direct.py RECIPE RUN_ID [key=json ...] [--runs-dir runs]

    python scripts/run_direct.py tdmpc2-pushcube-episode run_a01 params.episodes=50 \
        'params.plugins=[{"plugin": "a-obs-corruption", "config": {"severity": 3}}]'

key=json overrides any field of the request with a dotted path (params.seeds=[0,1,2],
environment=simpler-bridge-spoon, model.config.x=1); a value that is not JSON is a string.
The recipe is resolved through the platform registry (plus --registry files, e.g. your own
plugins), written to RUNS/RUN_ID/config.json, run by `wms.runners.unified` with the recipe's
runtime Python and environment, and its result printed. Output goes to RUNS/RUN_ID/output.

Use it on a machine whose runtime files (native/<engine>/runtime.json) point at installed
environments, such as the GPU server. It does not go through the service's queue, so check
the GPU is free first (nvidia-smi). Compare two runs step by step with scripts/compare_runs.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def request_from(recipe, overrides):
    request = {"recipe": recipe}
    for item in overrides:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"override {item!r} must be key=value")
        target = request
        *parents, leaf = key.split(".")
        for part in parents:
            target = target.setdefault(part, {})
        try:
            target[leaf] = json.loads(value)
        except json.JSONDecodeError:
            target[leaf] = value
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one recipe directly (no service).")
    parser.add_argument("recipe")
    parser.add_argument("run_id")
    parser.add_argument("overrides", nargs="*", help="key=json, dotted paths into the request")
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    parser.add_argument("--registry", action="append", default=[], help="extra recipes.json / registry.json")
    args = parser.parse_args(argv)

    from wms.core.registry import Registry
    config = Registry(paths=args.registry).resolve(request_from(args.recipe, args.overrides))
    if config.get("configuration_error"):
        raise SystemExit(f"configuration error: {config['configuration_error']}")
    run_dir = Path(args.runs_dir) / args.run_id
    if (run_dir / "output" / "result.json").exists():
        raise SystemExit(f"{run_dir} already has a result; choose another RUN_ID")
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime = config["runtime"]
    python = runtime.get("python") or sys.executable
    env = {**os.environ, **{str(k): str(v) for k, v in runtime.get("env", {}).items()},
           "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), *runtime.get("python_paths", [])])}
    device = runtime.get("cuda_visible_devices", runtime.get("gpu_device"))
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    with (run_dir / "run.log").open("w") as log:
        code = subprocess.call([python, "-m", "wms.runners.unified", "--config", str(run_dir / "config.json"),
                                "--output-dir", str(run_dir / "output")], env=env, stdout=log,
                               stderr=subprocess.STDOUT, cwd=str(run_dir))
    result_path = run_dir / "output" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    print(json.dumps({"run": args.run_id, "exit": code, "status": result.get("status"), "data": result.get("data"),
                      "error": result.get("error")}, ensure_ascii=False, default=str)[:3000])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
