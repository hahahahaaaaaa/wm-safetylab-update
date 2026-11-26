"""Run the official MemoryVLA SimplerEnv evaluation inside a platform task.

This follows the upstream entry `evaluation/simpler_env/simpler_env_inference.py` and
`script/eval/bridge/eval_bridge.sh` of MemoryVLA (openvla-codebase): the same SimplerEnv
arguments, the same `config.yaml` merge, `VLAInference` and the upstream single-episode
function `run_maniskill2_eval_single_episode`. Upstream files are imported, never edited.

Differences from the upstream script, all recorded in evaluation.json:
  * bf16 weights by default (params.use_bf16): fp32 needs ~30 GB of GPU memory;
  * the model is loaded once for all tasks instead of once per task;
  * torch / numpy / random are seeded per episode (upstream does not seed the diffusion head);
  * the Llama-2 config and tokenizer come from a local directory (runtime.llama2_dir) because
    meta-llama/Llama-2-7b-hf is gated; the weights themselves come from the MemoryVLA checkpoint;
  * VK_ICD_FILENAMES comes from the runtime file (upstream hard-codes a path missing on some hosts);
  * TensorFlow is hidden from the GPU (upstream reserves 3 GB; this evaluation does not use TF).
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

from .common import (argv_for_task as _argv, load_inference, merged_args as _merged_args, overlay_dir as _overlay_dir,
                     prepare_process as _prepare_process, seed_all as _seed, versions as _versions)

TASK_KEYS = ("name", "env_name", "scene_name", "robot", "rgb_overlay", "robot_init_x", "robot_init_y")


class _ActionRecorder:
    """Pass-through around VLAInference that writes every step's output to actions.jsonl.

    Only used with params.log_actions; it does not change what the model computes."""

    def __init__(self, model, log):
        self._model, self._log, self.episode, self._t = model, log, {}, 0

    def reset(self, task_description):
        self._t = 0
        return self._model.reset(task_description)

    def step(self, image, task_description, **kwargs):
        import numpy as np
        raw, action = self._model.step(image, task_description, **kwargs)
        sent = np.concatenate([action["world_vector"], action["rot_axangle"], np.asarray(action["gripper"]).reshape(-1)])
        self._log.write(json.dumps({**self.episode, "t": self._t, "action": [float(v) for v in sent],
                                    "terminate_episode": float(np.asarray(action["terminate_episode"]).reshape(-1)[0])})
                        + "\n")
        self._log.flush()
        self._t += 1
        return raw, action

    def __getattr__(self, name):
        return getattr(self._model, name)


def evaluate(runtime, params, output, context, seed=0):
    started = time.time()
    tasks = params.get("tasks") or []
    only = params.get("only_tasks")
    if only:
        tasks = [task for task in tasks if task["name"] in only]
    if not tasks:
        raise ValueError("params.tasks is empty")
    for task in tasks:
        missing = [key for key in TASK_KEYS if key not in task]
        if missing:
            raise ValueError(f"task {task.get('name')}: missing {missing}")
    episodes = [int(v) for v in params.get("episodes", [0, 24])]
    use_bf16 = bool(params.get("use_bf16", True))
    checkpoint = Path(runtime["checkpoint"])
    overlay_dir = _overlay_dir(runtime)
    output.mkdir(parents=True, exist_ok=True)
    videos = output / "videos"

    context.event("progress", {"phase": "loading_model", "checkpoint": str(checkpoint), "bf16": use_bf16})
    _prepare_process(runtime)
    from evaluation.simpler_env.maniskill2_evaluator import run_maniskill2_eval_single_episode
    from simpler_env.utils.env.env_builder import get_robot_control_mode

    _seed(seed)
    load_started = time.time()
    model, first_args = load_inference(runtime, tasks[0], params, episodes, use_bf16=use_bf16,
                                       cfg_scale=float(params.get("cfg_scale", 1.5)))
    action_log = None
    if params.get("log_actions"):
        # parity checks against the episode entry: record what the model returned at every step
        action_log = (output / "actions.jsonl").open("w", encoding="utf-8")
        model = _ActionRecorder(model, action_log)
    load_seconds = time.time() - load_started
    import torch
    memory_gb = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
    context.event("progress", {"phase": "model_loaded", "seconds": round(load_seconds, 1),
                               "max_gpu_memory_gb": memory_gb and round(memory_gb, 2)})

    rows, per_task = [], {}
    log = (output / "episodes.jsonl").open("w", encoding="utf-8")
    try:
        for task_index, task in enumerate(tasks):
            args, _ = _merged_args(_argv(task, checkpoint, overlay_dir, params, episodes), checkpoint, use_bf16)
            control_mode = get_robot_control_mode(args.robot, args.policy_model)
            successes = []
            for robot_init_x in args.robot_init_xs:
                for robot_init_y in args.robot_init_ys:
                    for robot_init_quat in args.robot_init_quats:
                        for episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                            context.check_cancelled()
                            episode_seed = seed + 1000 * task_index + episode_id
                            _seed(episode_seed)
                            if action_log is not None:
                                model.episode = {"task": task["name"], "episode_id": episode_id, "seed": episode_seed}
                            episode_started = time.time()
                            success = run_maniskill2_eval_single_episode(
                                model=model, ckpt_path=args.ckpt_path, robot_name=args.robot,
                                env_name=args.env_name, scene_name=args.scene_name,
                                robot_init_x=robot_init_x, robot_init_y=robot_init_y,
                                robot_init_quat=robot_init_quat, control_mode=control_mode,
                                additional_env_build_kwargs=args.additional_env_build_kwargs,
                                rgb_overlay_path=args.rgb_overlay_path, control_freq=args.control_freq,
                                sim_freq=args.sim_freq, max_episode_steps=args.max_episode_steps,
                                enable_raytracing=args.enable_raytracing,
                                additional_env_save_tags=args.additional_env_save_tags,
                                obs_camera_name=args.obs_camera_name, logging_dir=str(videos / task["name"]),
                                obj_episode_id=episode_id)
                            row = {"task": task["name"], "env_name": args.env_name, "episode_id": episode_id,
                                   "success": bool(success), "seed": episode_seed,
                                   "seconds": round(time.time() - episode_started, 2)}
                            rows.append(row)
                            successes.append(bool(success))
                            log.write(json.dumps(row) + "\n")
                            log.flush()
                            context.event("progress", {"phase": "episode", "task": task["name"],
                                                       "episode": episode_id, "success": bool(success),
                                                       "task_success_so_far": sum(successes) / len(successes)})
            per_task[task["name"]] = {"env_name": args.env_name, "episodes": len(successes),
                                      "successes": sum(successes),
                                      "success_rate_pct": 100.0 * sum(successes) / len(successes)}
    finally:
        log.close()
        if action_log is not None:
            action_log.close()

    reference = params.get("reference", {})
    average = sum(v["success_rate_pct"] for v in per_task.values()) / len(per_task)
    comparison = {name: {"ours_pct": per_task[name]["success_rate_pct"], "official_pct": reference.get(name),
                         "difference_pct": (per_task[name]["success_rate_pct"] - reference[name]
                                            if name in reference else None)} for name in per_task}
    summary = {
        "path": "official", "task": "simpler_bridge_evaluate", "benchmark": "SimplerEnv-Bridge (visual matching)",
        "model": "MemoryVLA (openvla-codebase)", "per_task": per_task, "average_success_pct": average,
        "reference_average_pct": reference.get("average") if len(per_task) == 4 else None,
        "comparison": comparison, "episodes_range": episodes, "seed": seed, "use_bf16": use_bf16,
        "cfg_scale": float(params.get("cfg_scale", 1.5)), "load_seconds": round(load_seconds, 1),
        "max_gpu_memory_gb": memory_gb, "elapsed_seconds": round(time.time() - started, 1),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": _versions(runtime), "merged_args_first_task": {k: v for k, v in vars(first_args).items()
                                                                  if isinstance(v, (str, int, float, bool, type(None)))},
        "differences_from_upstream_script": [
            f"weights in {'bf16' if use_bf16 else 'fp32'} (upstream eval_bridge.sh: fp32)",
            "model loaded once for all tasks (upstream: once per task)",
            "torch/numpy/random seeded per episode (upstream: unseeded)",
            "Llama-2 config/tokenizer from runtime.llama2_dir (public copy of the gated meta-llama repo)",
            "VK_ICD_FILENAMES from the runtime file; TensorFlow hidden from the GPU",
        ],
        "claim_limit": ("Reproduction of the published SimplerEnv-Bridge evaluation for one released checkpoint; "
                        "24 episodes per task means one episode is 4.2 points. Not a safety evaluation."),
    }
    (output / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                            encoding="utf-8")
    return summary
