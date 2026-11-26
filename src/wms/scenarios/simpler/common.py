"""SimplerEnv-Bridge pieces shared by the official evaluation loop and SimplerBridgeEnv.

The task table is the one in script/eval/bridge/eval_bridge.sh (and in the
memoryvla-simpler-bridge-eval recipe); build_kwargs() and reset_options() repeat what
MemoryVLA's run_maniskill2_eval_single_episode passes to SimplerEnv, so an environment
built here starts every episode exactly as the official evaluation does.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

CAMERA = "3rd_view_camera"
TASKS = {
    "spoon": {"env_name": "PutSpoonOnTableClothInScene-v0", "scene_name": "bridge_table_1_v1", "robot": "widowx",
              "rgb_overlay": "bridge_real_eval_1.png", "robot_init_x": 0.147, "robot_init_y": 0.028},
    "carrot": {"env_name": "PutCarrotOnPlateInScene-v0", "scene_name": "bridge_table_1_v1", "robot": "widowx",
               "rgb_overlay": "bridge_real_eval_1.png", "robot_init_x": 0.147, "robot_init_y": 0.028},
    "cube": {"env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0", "scene_name": "bridge_table_1_v1",
             "robot": "widowx", "rgb_overlay": "bridge_real_eval_1.png", "robot_init_x": 0.147, "robot_init_y": 0.028},
    "eggplant": {"env_name": "PutEggplantInBasketScene-v0", "scene_name": "bridge_table_1_v2",
                 "robot": "widowx_sink_camera_setup", "rgb_overlay": "bridge_sink.png",
                 "robot_init_x": 0.127, "robot_init_y": 0.06},
}
TASK_FIELDS = tuple(TASKS["carrot"])
CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"   # SimplerEnv's choice for widowx
DEFAULTS = {"control_freq": 5, "sim_freq": 500, "max_episode_steps": 120}   # eval_bridge.sh


def load_runtime_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def task_settings(name, overrides=None):
    """Settings of one Bridge task, with any of TASK_FIELDS overridden."""
    if name not in TASKS and not overrides:
        raise KeyError(f"unknown Bridge task {name!r}; use one of {sorted(TASKS)}")
    return {**TASKS.get(name, {}), **{k: v for k, v in (overrides or {}).items() if k in TASK_FIELDS}}


def prepare_paths(runtime):
    """Make SimplerEnv importable and rendering headless (call before importing simpler_env)."""
    for key, value in (runtime.get("env") or {}).items():
        os.environ.setdefault(str(key), str(value))
    os.environ["DISPLAY"] = ""
    simpler = Path(runtime["simpler_dir"])
    if not simpler.is_dir():
        raise FileNotFoundError(f"runtime.simpler_dir does not exist: {simpler}")
    for path in reversed([str(simpler), str(simpler / "ManiSkill2_real2sim")]):
        if path not in sys.path:
            sys.path.insert(0, path)


def overlay_path(runtime, task):
    return Path(runtime["simpler_dir"]) / "ManiSkill2_real2sim" / "data" / "real_inpainting" / task["rgb_overlay"]


def build_kwargs(runtime, task, control_freq=5, sim_freq=500, max_episode_steps=120):
    """Keyword arguments of build_maniskill2_env, as the official single-episode evaluation sets them."""
    return {"obs_mode": "rgbd", "robot": task["robot"], "sim_freq": int(sim_freq), "control_mode": CONTROL_MODE,
            "control_freq": int(control_freq), "max_episode_steps": int(max_episode_steps),
            "scene_name": task["scene_name"], "camera_cfgs": {"add_segmentation": True},
            "rgb_overlay_path": str(overlay_path(runtime, task))}


def reset_options(task, episode_id):
    """env.reset options of the official evaluation in obj-variation mode 'episode'."""
    import numpy as np
    return {"robot_init_options": {"init_xy": np.array([task["robot_init_x"], task["robot_init_y"]]),
                                   "init_rot_quat": np.array([0.0, 0.0, 0.0, 1.0])},
            "obj_init_options": {"episode_id": int(episode_id)}}


def build_env(runtime, task, **settings):
    """A fresh SimplerEnv gym environment for `task` (call prepare_paths first)."""
    from simpler_env.utils.env.env_builder import build_maniskill2_env
    return build_maniskill2_env(task["env_name"], **build_kwargs(runtime, task, **settings))
