"""Action replay: does ManiSkillVisionEnv behave like a plain ManiSkill environment?

Native task, no model, runs in the ManiSkill runtime. For each task and seed the same
seeded random action sequence drives two environments built with the same gym.make
arguments: the plain gymnasium environment (reference) and
wms.scenarios.maniskill.vision:ManiSkillVisionEnv. Per step it compares object and TCP
positions, success flags and camera images.

params: env_ids (default PickCube-v1, StackCube-v1, PushCube-v1), seeds (default [0, 1]),
        steps (default 30), control_mode (default pd_joint_pos), action_scale (default 0.3 of the
        action range), pose_tol (default 1e-6), image_tol (default 0).
"""
from __future__ import annotations

import numpy as np

from .vision import TASKS, ManiSkillVisionEnv, _first


def _positions(native, task):
    parts = [_first(getattr(native, attr).pose.p) for attr in task["objects"].values()]
    parts.append(_first(native.agent.tcp.pose.p))
    return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts])


def _image(obs):
    data = next(iter(obs["sensor_data"].values()))
    return _first(data["rgb"]).astype(np.int16)


def replay(config, output_dir, context):
    import gymnasium as gym
    params = dict(config.get("params", {}))
    env_ids = params.get("env_ids", list(TASKS))
    seeds = [int(s) for s in params.get("seeds", [0, 1])]
    steps = int(params.get("steps", 30))
    control_mode = params.get("control_mode", "pd_joint_pos")
    scale = float(params.get("action_scale", 0.3))
    pose_tol, image_tol = float(params.get("pose_tol", 1e-6)), float(params.get("image_tol", 0))
    runtime_config = config["runtime"]["config_file"]
    rows = []
    for env_id in env_ids:
        settings = {"env_id": env_id, "obs_mode": "rgb", "control_mode": control_mode, "camera_size": [128, 128],
                    "max_episode_steps": steps, "sim_backend": "physx_cpu", "runtime_config": runtime_config}
        env = ManiSkillVisionEnv(settings)
        reference = gym.make(env_id, robot_uids="panda", num_envs=1, obs_mode="rgb", control_mode=control_mode,
                             render_mode="rgb_array", sim_backend="physx_cpu", render_backend="gpu",
                             sensor_configs=dict(width=128, height=128), max_episode_steps=steps)
        low = np.asarray(reference.action_space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(reference.action_space.high, dtype=np.float64).reshape(-1)
        try:
            for seed in seeds:
                context.check_cancelled()
                rng = np.random.default_rng(2000 + seed)
                ref_obs, _ = reference.reset(seed=seed)
                env.reset(seed)
                start = (low + high) / 2 if control_mode.startswith("pd_ee") else _first(
                    reference.unwrapped.agent.robot.get_qpos()).astype(np.float64)[:low.size]
                worst_pose = float(np.abs(_positions(reference.unwrapped, TASKS[env_id])
                                          - _positions(env.native, TASKS[env_id])).max())
                worst_image = float(np.abs(_image(ref_obs) - _image(env.native.get_obs())).max())
                success_mismatch = 0
                for _ in range(steps):
                    action = np.clip(start + scale * (high - low) / 2 * rng.uniform(-1, 1, low.size), low, high)
                    ref_obs, _, _, _, ref_info = reference.step(action.astype(np.float32).reshape(1, -1))
                    transition = env.step(action)
                    worst_pose = max(worst_pose, float(np.abs(_positions(reference.unwrapped, TASKS[env_id])
                                                              - _positions(env.native, TASKS[env_id])).max()))
                    seen = transition["obs"]["rgb"]
                    worst_image = max(worst_image, float(np.abs(_image(ref_obs)
                                                                - next(iter(seen.values())).astype(np.int16)).max()))
                    ref_success = bool(_first(ref_info["success"]).reshape(-1)[0])
                    success_mismatch += ref_success != transition["info"]["success"]
                rows.append({"env_id": env_id, "seed": seed, "max_position_diff": worst_pose,
                             "max_image_diff": worst_image, "success_mismatches": success_mismatch,
                             "ok": worst_pose <= pose_tol and worst_image <= image_tol and success_mismatch == 0})
                context.event("progress", {"phase": "replay", **rows[-1]})
        finally:
            reference.close()
            env.close()
    report = {"control_mode": control_mode, "steps": steps, "pose_tol": pose_tol, "image_tol": image_tol,
              "passed": all(r["ok"] for r in rows), "episodes": rows}
    context.write_json("replay.json", report)
    return {"data": {"passed": report["passed"], "episodes": len(rows), "env_ids": env_ids},
            "outputs": [{"path": "replay.json", "kind": "report", "media_type": "application/json"}]}
