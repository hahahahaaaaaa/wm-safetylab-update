"""Shared by both MemoryVLA entries: the official evaluation (simpler.py, path A) and the
step-by-step policy (adapter.py, path B).

Everything that decides what the model sees and how it is loaded lives here, so the two
entries cannot drift apart: the SimplerEnv arguments of eval_bridge.sh, the config.yaml
merge, process preparation (paths, TensorFlow off the GPU, local Llama-2 tokenizer),
seeding, and the VLAInference construction.
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import sys


def argv_for_task(task, checkpoint, overlay_dir, params, episodes):
    """The arguments script/eval/bridge/eval_bridge.sh passes for one task."""
    x, y = str(task["robot_init_x"]), str(task["robot_init_y"])
    return ["--ckpt-path", str(checkpoint), "--robot", task["robot"], "--policy-setup", "widowx_bridge",
            "--control-freq", str(params.get("control_freq", 5)), "--sim-freq", str(params.get("sim_freq", 500)),
            "--max-episode-steps", str(params.get("max_episode_steps", 120)),
            "--env-name", task["env_name"], "--scene-name", task["scene_name"],
            "--rgb-overlay-path", str(Path(overlay_dir) / task["rgb_overlay"]),
            "--robot-init-x-range", x, x, "1", "--robot-init-y-range", y, y, "1",
            "--obj-variation-mode", "episode", "--obj-episode-range", str(episodes[0]), str(episodes[1]),
            "--robot-init-rot-quat-center", "0", "0", "0", "1",
            "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1"]


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def merged_args(argv, checkpoint, use_bf16):
    """Same steps as the upstream __main__ block: SimplerEnv get_args, then config.yaml underneath."""
    from argparse import Namespace
    import yaml
    from simpler_env.evaluation.argparse import get_args
    saved = sys.argv
    try:
        sys.argv = ["simpler_env_inference.py", *argv]
        args = get_args()
    finally:
        sys.argv = saved
    config_yaml = Path(checkpoint).parents[1] / "config.yaml"
    yaml_args = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}
    merged = Namespace(**deep_update(dict(yaml_args), {**vars(args), "use_bf16": use_bf16}))
    filtered = {k: v for k, v in vars(merged).items() if k not in {"pretrained_checkpoint", "use_bf16"}}
    return merged, filtered


def seed_all(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def prepare_process(runtime):
    os.environ["DISPLAY"] = ""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"            # backbones and tokenizer are prepared by prepare_server.sh
    for key in ("source_dir", "simpler_dir"):
        if not Path(runtime[key]).is_dir():
            raise FileNotFoundError(f"runtime.{key} does not exist: {runtime[key]}")
    paths = [runtime["source_dir"], runtime["simpler_dir"], str(Path(runtime["simpler_dir"]) / "ManiSkill2_real2sim")]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    import tensorflow as tf                       # imported by MemoryVLA's data modules anyway
    tf.config.set_visible_devices([], "GPU")
    from prismatic.models.backbones.llm import llama2
    llama2.LLAMA2_MODELS["llama2-7b-pure"]["hf_hub_path"] = str(runtime["llama2_dir"])


def versions(runtime):
    import importlib.metadata as metadata
    import torch
    packages = {}
    for name in ("torch", "transformers", "timm", "tensorflow", "sapien", "numpy", "gymnasium", "mediapy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"packages": packages, "python": sys.version.split()[0],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "source_commits": runtime.get("source_commits", {}),
            "checkpoint": str(runtime["checkpoint"])}


def overlay_dir(runtime):
    return Path(runtime["simpler_dir"]) / "ManiSkill2_real2sim" / "data" / "real_inpainting"


def load_inference(runtime, task, params, episodes, use_bf16=True, cfg_scale=1.5):
    """Construct upstream VLAInference with the arguments eval_bridge.sh gives `task`.

    Call prepare_process(runtime) first. Returns (model, merged_args)."""
    from evaluation.simpler_env import VLAInference
    checkpoint = Path(runtime["checkpoint"])
    merged, filtered = merged_args(argv_for_task(task, checkpoint, overlay_dir(runtime), params, episodes),
                                   checkpoint, use_bf16)
    model = VLAInference(saved_model_path=str(checkpoint), cfg_scale=float(cfg_scale), use_bf16=use_bf16,
                         **filtered)
    return model, merged
