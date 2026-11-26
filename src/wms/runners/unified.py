"""Execute one experiment in its selected Python environment.

    python -m wms.runners.unified --config experiment.json --output-dir run/output

Components are ordinary Python classes constructed with a config dict. Model
libraries are imported inside the worker, rather than in the Web service.
"""
from __future__ import annotations

import argparse
import importlib
import json
import mimetypes
import os
from pathlib import Path
import random
import sys
import time
import traceback

from wms.core.payloads import json_value, resolve, is_payload


class Cancelled(Exception):
    """Raised when the run's cancel.flag is present."""


def load_entrypoint(name):
    module, separator, attr = name.partition(":")
    if not separator:
        module, _, attr = name.rpartition(".")
    if not module or not attr:
        raise ValueError("Entrypoint must be 'module:callable'")
    return getattr(importlib.import_module(module), attr)


class RunContext:
    def __init__(self, output_dir):
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)

    def check_cancelled(self):
        if (self.output / "cancel.flag").exists():
            raise Cancelled("Run cancelled")

    def event(self, kind, data):
        print(json.dumps({"event": kind, **json_value(data)}, ensure_ascii=False), flush=True)

    def write_json(self, name, data):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_value(data), ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")
        return path


class UnifiedWorker(RunContext):
    def __init__(self, config, output_dir):
        super().__init__(output_dir)
        self.config = config
        self.seed = int(config.get("seed", 0))
        self.inputs = config.get("inputs", {})
        self.params = config.get("params", {})
        self.model = self.env = self.planner = None

    def component(self, spec):
        if not spec:
            return None
        cls = load_entrypoint(spec["entrypoint"])
        config = {**spec.get("config", {}), "output_dir": str(self.output)}
        if spec.get("constructor") == "manifest":
            return cls(spec.get("manifest", {}), config)
        return cls(config)

    def request(self):
        return {**self.inputs, **self.params, "seed": self.seed,
                "output_dir": str(self.output)}

    def execute(self):
        runtime = self.config.get("runtime", {})
        for path in reversed(runtime.get("python_paths", [])):
            sys.path.insert(0, str(path))
        os.environ.update({str(k): str(v) for k, v in runtime.get("env", {}).items()})
        self.check_cancelled()
        task = self.config["task"]
        self.event("progress", {"phase": "start", "task": task, "python": sys.executable})
        try:
            if task == "evaluate":
                from wms.core.evaluation import execute_evaluation
                return execute_evaluation(self.config, self.output, self)
            if task == "native":
                return load_entrypoint(self.config["entrypoint"])(self.config, self.output, self)
            self.model = self.component(self.config.get("model"))
            self.env = self.component(self.config.get("environment"))
            self.planner = self.component(self.config.get("algorithm"))
            if task == "generate":
                return self.model.generate(self.request())
            if task == "predict":
                return self.predict()
            if task == "rollout":
                return self.episode(lambda obs, t: self.model.act(obs, self.inputs.get("instruction")))
            if task == "plan":
                return self.plan()
            raise ValueError(f"Unknown task: {task}; use generate, predict, rollout, plan or native")
        finally:
            active_error = sys.exc_info()[0] is not None
            close_error = None
            for component in (self.env, self.model, self.planner):
                close = getattr(component, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        self.event("error", {"phase": "close", "message": str(exc)})
                        close_error = close_error or exc
            if close_error is not None and not active_error:
                raise close_error

    def reset(self):
        if self.env is None:
            raise ValueError("This task requires an environment component")
        return self.env.reset(self.seed, self.inputs)

    def predict(self):
        request = self.request()
        if self.env is None:
            return self.model.predict(request)
        obs = self.reset()
        actions = request.get("actions")
        if actions is None:
            raise ValueError("Prediction against a simulator requires inputs.actions")
        predicted = self.model.predict({**request, "observation": obs})
        real, transitions = [], []
        for action in actions:
            self.check_cancelled()
            transition = self.env.step(action)
            transitions.append(transition)
            real.append(transition["obs"])
            if transition.get("terminated") or transition.get("truncated"):
                break
        return {"initial_obs": obs, "actions": actions, "predicted": predicted,
                "real": real, "transitions": transitions}

    def episode(self, choose):
        obs = self.reset()
        rows = []
        end_reason = "step_limit"
        for t in range(int(self.config.get("steps", 8))):
            self.check_cancelled()
            action = choose(obs, t)
            self.check_cancelled()
            transition = self.env.step(action)
            rows.append({"t": t, **transition, "obs": obs, "action": action,
                         "next_obs": transition["obs"]})
            obs = transition["obs"]
            self.event("progress", {"phase": "step", "step": t + 1})
            if transition.get("terminated") or transition.get("truncated"):
                end_reason = "terminated" if transition.get("terminated") else "truncated"
                break
        data = {"steps": rows, "final_obs": obs, "seed": self.seed,
                "summary": {"steps": len(rows), "end_reason": end_reason,
                            "return": sum(float(row.get("reward", 0)) for row in rows)}}
        self.write_json("trajectory.json", data)
        return {"outputs": [{"kind": "trajectory", "path": "trajectory.json",
                              "media_type": "application/json"}], "data": data["summary"]}

    def plan(self):
        from wms.core.planners import PlannerContext
        if self.planner is None:
            raise ValueError("plan requires an algorithm component")
        cost = load_entrypoint(self.params["cost"])
        rng, records = random.Random(self.seed), []

        def predict(obs, actions):
            self.check_cancelled()
            out = self.model.predict({**self.request(), "observation": obs, "actions": actions})
            value = out.get("predictions", out) if isinstance(out, dict) else out
            return resolve(value, base=self.output) if is_payload(value) else value

        def choose(obs, t):
            record = {"t": t}
            context = PlannerContext(predict=predict, cost=cost,
                action_range=self.params["action_range"], rng=rng,
                case=self.inputs, record=record)
            action = self.planner.plan(obs, t, context)
            records.append(record)
            return action

        out = self.episode(choose)
        self.write_json("plans.json", records)
        out["outputs"].append({"kind": "plans", "path": "plans.json", "media_type": "application/json"})
        return out

    def result(self, returned):
        """Normalize ordinary values or an explicit result into the run result."""
        returned = returned if isinstance(returned, dict) else {"data": returned}
        envelope = any(key in returned for key in ("outputs", "data", "metrics", "status"))
        result = {"status": returned.get("status", "completed"), "outputs": [],
                  "metrics": returned.get("metrics", {}), "error": returned.get("error")}
        for key in ("metric_errors", "metric_specs", "evaluation_status", "source_run"):
            if key in returned:
                result[key] = returned[key]
        data = returned.get("data", {k: v for k, v in returned.items()
                                    if k not in {"outputs", "metrics", "status", "error"}}) if envelope else returned
        if data is not None:
            result["data"] = json_value(data)
        for output in returned.get("outputs", []):
            item = dict(output)
            name = item.pop("relative_path", None) or item.get("path")
            if name is None:
                raise ValueError("Output needs a path")
            path = Path(name)
            path = (path if path.is_absolute() else self.output / path).resolve()
            # Output URLs are rooted in a run directory, not arbitrary file downloads.
            item["path"] = path.relative_to(self.output).as_posix()
            if not path.is_file():
                raise FileNotFoundError(path)
            item.setdefault("kind", "file")
            item.setdefault("media_type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            result["outputs"].append(item)
        return result

    def run(self):
        try:
            returned = self.execute()
            # Native workers can stop cooperatively and return their partial
            # evidence. Keep that explicit cancellation result, while still
            # validating every output path through result().
            cancelled_native = (self.config.get("task") == "native"
                                and isinstance(returned, dict)
                                and returned.get("status") == "cancelled")
            if not cancelled_native:
                self.check_cancelled()
            result = self.result(returned)
            self.write_json("result.json", result)
        except Cancelled as exc:
            result = {"status": "cancelled", "outputs": [], "metrics": {}, "error": str(exc)}
        except Exception as exc:
            traceback.print_exc()
            result = {"status": "failed", "outputs": [], "metrics": {},
                      "error": f"{type(exc).__name__}: {exc}"}
        if result["status"] != "completed":
            self.write_json("result.json", result)
        return 0 if result["status"] == "completed" else (130 if result["status"] == "cancelled" else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--managed", action="store_true")
    args = parser.parse_args()
    if args.managed:
        # Do not start model code until the service has persisted our identity.
        marker = Path(args.config).parent / "start.flag"
        deadline = time.monotonic() + 30
        while not marker.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("Service did not acknowledge this worker")
            time.sleep(0.05)
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    code = UnifiedWorker(config, args.output_dir).run()
    if args.managed:
        directory = Path(args.config).parent
        (directory / "finish.ready").touch()
        while not (directory / "finish.flag").exists():
            time.sleep(0.05)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
