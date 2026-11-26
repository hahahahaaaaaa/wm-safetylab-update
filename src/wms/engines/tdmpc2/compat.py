"""Small compatibility boundary: upstream imports, configuration, and raw state env.

This module deliberately bypasses the upstream Hydra launcher and legacy Gym
attribute forwarding. It does not replace TD-MPC2's network, losses or planner.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys


def prepare(runtime):
    from wms.engines.maniskill.runtime import native_environment
    environment = native_environment(runtime)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ.update(environment)
    source = Path(runtime["source_dir"]).resolve()
    baseline = source / "examples/baselines/tdmpc2"
    if not (baseline / "tdmpc2.py").is_file():
        raise FileNotFoundError(f"Official ManiSkill TD-MPC2 source missing: {baseline}")
    if str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    import torch
    torch.set_num_threads(int(runtime.get("cpu_threads", 4)))
    if not torch.cuda.is_available():
        raise RuntimeError("This upstream TD-MPC2 implementation requires CUDA")
    # SAPIEN 3.0.3 otherwise auto-downloads into ~/.sapien without a timeout.
    # Match its loader, but use an explicitly prepared project-local library.
    if runtime.get("physx_gpu_library"):
        import ctypes
        import sapien
        if not sapien.physx.is_gpu_enabled():
            library = Path(runtime["physx_gpu_library"])
            if not library.is_file():
                raise FileNotFoundError(f"Prepare project-local PhysX GPU library first: {library}")
            ctypes.CDLL("libcuda.so.1", ctypes.RTLD_GLOBAL)
            ctypes.CDLL(str(library), ctypes.RTLD_LOCAL)
            sapien.physx._enable_gpu()
        sapien.render.set_imgui_ini_filename(str(Path(runtime["physx_gpu_library"]).parent / "imgui.ini"))
    return baseline


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_config(baseline, params, output, seed):
    """Load official hyperparameters without Hydra's process-global cwd state."""
    from omegaconf import OmegaConf
    from common import MODEL_SIZE
    cfg = OmegaConf.load(Path(baseline) / "config.yaml")
    if "defaults" in cfg:
        del cfg["defaults"]
    values = dict(env_id="PushCube-v1", obs="state", control_mode="pd_joint_delta_pos",
                  model_size=5, num_envs=16, num_eval_envs=1, steps=100000,
                  buffer_size=100000, steps_per_update=4, batch_size=256,
                  seed=int(seed), exp_name="wms", wandb=False, multitask=False,
                  task_dim=0, tasks=["PushCube-v1"], work_dir=str(output))
    allowed = set(values) | {"iterations", "num_samples", "num_elites", "num_pi_trajs",
                             "horizon", "lr", "eval_freq", "eval_episodes_per_env"}
    values.update({key: value for key, value in params.items() if key in allowed})
    if values["env_id"] not in ("PushCube-v1", "WMS-PushCubeCenter-v1") or values["obs"] != "state":
        raise ValueError("Select PushCube-v1 or WMS-PushCubeCenter-v1 with state observations")
    values["tasks"] = [values["env_id"]]
    values["task_profile"] = "center2cm-v1" if values["env_id"] == "WMS-PushCubeCenter-v1" else "region10cm"
    values["reward_version"] = "center-dense/1" if values["env_id"] == "WMS-PushCubeCenter-v1" else "maniskill-normalized-dense"
    if values["control_mode"] != "pd_joint_delta_pos":
        raise ValueError("Keep pd_joint_delta_pos; other controllers need a separate trained recipe")
    if values["model_size"] not in MODEL_SIZE:
        raise ValueError("Unknown official model_size")
    for key, value in {**values, **MODEL_SIZE[values["model_size"]]}.items():
        cfg[key] = value
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    if not 0 < cfg.num_elites <= cfg.num_samples or not 0 <= cfg.num_pi_trajs < cfg.num_samples:
        raise ValueError("Planner population, elites and policy candidates are inconsistent")
    for key in ("steps", "num_envs", "steps_per_update", "batch_size", "horizon", "iterations"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg.horizon > 50 or min(cfg.buffer_size, cfg.steps) < cfg.horizon + 1:
        raise ValueError("Replay must hold horizon+1 observations, and planning horizon cannot exceed 50")
    if cfg.steps_per_update > cfg.num_envs or cfg.num_envs % cfg.steps_per_update:
        raise ValueError("steps_per_update must divide num_envs to preserve the specified update ratio")
    return cfg


def plain_config(cfg):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)


class StateEnvironment:
    """Raw ManiSkill GPU environment, fixed horizon, explicit reset, no auto-reset.

    Full real terminal observations are therefore available without depending on
    Gymnasium's final_observation/final_obs naming. Success does not cut training
    episodes short (the same continuing-task convention as the upstream baseline).
    """
    def __init__(self, cfg, num_envs, *, render=False):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 -- task registration
        import numpy as np
        self.cfg, self.num_envs = cfg, int(num_envs)
        if cfg.env_id == "WMS-PushCubeCenter-v1":
            from wms.scenarios.maniskill import push_cube_center  # noqa: F401 -- registration
            if cfg.get("reward_version") != "center-dense/1":
                raise ValueError("Center task requires its separately trained reward version")
        self.env = gym.make(cfg.env_id, robot_uids="panda", num_envs=self.num_envs,
                            obs_mode="state", control_mode=cfg.control_mode,
                            reward_mode="normalized_dense", sim_backend="physx_cuda",
                            render_backend="gpu", render_mode="rgb_array" if render else None,
                            max_episode_steps=50)
        self.native = self.env.unwrapped
        self.horizon = 50
        self.steps = 0
        self.action_dim = int(self.env.action_space.shape[-1])
        self.obs_dim = int(self.env.observation_space.shape[-1])
        self.low = np.asarray(self.env.action_space.low)
        self.high = np.asarray(self.env.action_space.high)
        if not (np.allclose(self.low, -1) and np.allclose(self.high, 1)):
            self.close()
            raise ValueError("TD-MPC2 assumes normalized continuous actions in [-1,1]")
        cfg.obs_shape = {"state": [self.obs_dim]}
        cfg.action_dim = self.action_dim
        cfg.episode_length = self.horizon
        cfg.seed_steps = max(1000, int(cfg.num_envs) * self.horizon)

    def reset(self, seed=None, reconfigure=False):
        obs, info = self.env.reset(seed=seed, options={"reconfigure": True} if reconfigure else None)
        self.steps = 0
        return obs, info

    def step(self, action):
        import torch
        command = torch.as_tensor(action, dtype=torch.float32, device=self.native.device)
        if tuple(command.shape) != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape {(self.num_envs, self.action_dim)}, got {tuple(command.shape)}")
        if not bool(torch.isfinite(command).all()) or bool((command.abs() > 1.00001).any()):
            raise ValueError("Non-finite or out-of-range normalized action")
        if self.steps >= self.horizon:
            raise RuntimeError("Reset after the fixed 50-step episode")
        self.steps += 1
        return self.env.step(command)

    def oracle(self):
        """True simulator quantities for evidence only; not model predictions."""
        return {"cube_xyz": self.native.obj.pose.p.detach().cpu().numpy(),
                "tcp_xyz": self.native.agent.tcp.pose.p.detach().cpu().numpy(),
                "goal_xyz": self.native.goal_region.pose.p.detach().cpu().numpy()}

    def frame(self):
        import numpy as np
        frame = self.env.render()
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        return frame.astype(np.uint8)

    def close(self):
        self.env.close()


def episode_to_td(observations, actions, rewards):
    """[T+1,N,O], [T,N,A], [T,N] -> original Buffer's [N,T+1].

    Slot zero contains only s0. The original sampler discards action/reward at
    that slot; this convention pairs a_t and r_t with s_{t+1}, not s_t.
    """
    import torch
    from tensordict import TensorDict
    obs = torch.stack(observations).transpose(0, 1).cpu()
    act = torch.stack(actions).transpose(0, 1).cpu()
    rew = torch.stack(rewards).transpose(0, 1).cpu()
    n, t, _ = obs.shape
    if act.shape[:2] != (n, t - 1) or rew.shape != (n, t - 1):
        raise ValueError("Replay episode observations/actions/rewards are not time-aligned")
    act = torch.cat([torch.full((n, 1, act.shape[-1]), float("nan")), act], dim=1)
    rew = torch.cat([torch.full((n, 1), float("nan")), rew], dim=1)
    return TensorDict({"obs": obs, "action": act, "reward": rew}, batch_size=[n, t])


def checkpoint_save(agent, cfg, path, metadata):
    import torch
    torch.save({"model": agent.model.state_dict(), "config": plain_config(cfg),
                "metadata": metadata, "format": "wms.tdmpc2/1"}, path)


def checkpoint_load(path):
    """These checkpoints contain tensors and primitive metadata only."""
    import torch
    from omegaconf import OmegaConf
    from tdmpc2 import TDMPC2
    data = torch.load(path, map_location="cuda", weights_only=True)
    if data.get("format") != "wms.tdmpc2/1":
        raise ValueError("Expected a platform TD-MPC2 checkpoint with its observation/controller config")
    cfg = OmegaConf.create(data["config"])
    if cfg.env_id == "WMS-PushCubeCenter-v1":
        if cfg.get("reward_version") != "center-dense/1":
            raise ValueError("Checkpoint center reward version is incompatible")
        if data["metadata"].get("task_sha256") != task_definition(cfg.env_id)["task_sha256"]:
            raise ValueError("Checkpoint center task implementation differs from current task")
    cfg.num_envs = cfg.num_eval_envs = 1
    agent = TDMPC2(cfg)
    agent.load({"model": data["model"]})
    agent.model.eval()
    return agent, cfg, data.get("metadata", {})


def task_definition(env_id="PushCube-v1"):
    import inspect
    from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
    base_file = Path(inspect.getfile(PushCubeEnv))
    base_hash = file_sha256(base_file)
    result = {"task_source": str(base_file), "task_sha256": base_hash,
              "env_id": env_id, "reward_version": "maniskill-normalized-dense"}
    if env_id == "WMS-PushCubeCenter-v1":
        from wms.scenarios.maniskill.push_cube_center import PushCubeCenterEnv
        custom = Path(inspect.getfile(PushCubeCenterEnv))
        custom_hash = file_sha256(custom)
        result.update(task_source=str(custom), base_task_source=str(base_file),
                      base_task_sha256=base_hash, custom_task_sha256=custom_hash,
                      reward_version="center-dense/1",
                      task_sha256=hashlib.sha256((base_hash + ":" + custom_hash + ":center-dense/1").encode()).hexdigest())
    elif env_id != "PushCube-v1":
        raise ValueError("Unknown PushCube task definition")
    return result


def provenance(runtime, env_id="PushCube-v1"):
    import torch
    source = Path(runtime["source_dir"])
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    versions = {}
    for name in ("torch", "mani_skill", "sapien", "gymnasium", "numpy", "torchrl", "tensordict", "omegaconf"):
        versions[name] = importlib.metadata.version(name)
    return {"upstream": "https://github.com/mani-skill/ManiSkill", "commit": commit,
            "upstream_dirty": bool(dirty), "versions": versions,
            "python": sys.executable, "gpu": torch.cuda.get_device_name(0),
            **task_definition(env_id),
            "compatibility": "original network/loss/MPPI; platform config, raw fixed-horizon env and training orchestration",
            "safety_signals": "not measured; task success is not a safety verdict"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
