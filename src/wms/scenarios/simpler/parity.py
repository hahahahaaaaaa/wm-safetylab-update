"""Action replay: does SimplerBridgeEnv behave like the environment the official evaluation builds?

Native task, no model, runs in the MemoryVLA runtime. For each layout, the same seeded random
action sequence drives two environments:

  reference  built from the official argument pipeline: eval_bridge.sh arguments ->
             SimplerEnv get_args + config.yaml (wms.engines.memoryvla.common), then
             build_maniskill2_env(env_name, obs_mode="rgbd", ..., control_mode from
             get_robot_control_mode) and reset with the official options, as
             run_maniskill2_eval_single_episode does
  episode    wms.scenarios.simpler.env:SimplerBridgeEnv

Per step it compares object and TCP positions, the "task done" flag and the camera image.
params: task (default carrot), seeds (default [0, 1, 2]), steps (default 30),
        action_scale (default 0.02 m / rad), pose_tol (default 1e-5), image_tol (default 0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import CAMERA, TASKS, load_runtime_file, prepare_paths


def _positions(native):
    tcp = native.tcp.pose if hasattr(native, "tcp") else native.agent.robot.get_links()[0].pose
    return np.concatenate([native.episode_source_obj.pose.p, native.episode_target_obj.pose.p, tcp.p])


def _reference_env(runtime, task_name, steps):
    from wms.engines.memoryvla.common import argv_for_task, merged_args, overlay_dir
    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    task = {"name": task_name, **TASKS[task_name]}
    args, _ = merged_args(argv_for_task(task, runtime["checkpoint"], overlay_dir(runtime),
                                        {"max_episode_steps": steps}, [0, 1]),
                          runtime["checkpoint"], True)
    control_mode = get_robot_control_mode(args.robot, args.policy_model)

    def build():
        return build_maniskill2_env(
            args.env_name, **(args.additional_env_build_kwargs or {}),
            obs_mode="rgbd", robot=args.robot, sim_freq=args.sim_freq, control_mode=control_mode,
            control_freq=args.control_freq, max_episode_steps=args.max_episode_steps,
            scene_name=args.scene_name, camera_cfgs={"add_segmentation": True},
            rgb_overlay_path=args.rgb_overlay_path)

    def options(episode_id):
        return {"robot_init_options": {"init_xy": np.array([args.robot_init_xs[0], args.robot_init_ys[0]]),
                                       "init_rot_quat": args.robot_init_quats[0]},
                "obj_init_options": {"episode_id": episode_id}}
    return build, options, control_mode


def replay(config, output_dir, context):
    from wms.engines.native_tasks import _runtime
    from wms.engines.memoryvla.common import prepare_process
    from .env import SimplerBridgeEnv
    params = dict(config.get("params", {}))
    runtime = _runtime(config)
    if not runtime:
        raise ValueError("Select the MemoryVLA runtime.config_file")
    task_name = params.get("task", "carrot")
    seeds = [int(s) for s in params.get("seeds", [0, 1, 2])]
    steps = int(params.get("steps", 30))
    scale = float(params.get("action_scale", 0.02))
    pose_tol, image_tol = float(params.get("pose_tol", 1e-5)), float(params.get("image_tol", 0))
    prepare_process(runtime)              # SimplerEnv argparse and config.yaml need the MemoryVLA paths
    prepare_paths(runtime)
    build, options, control_mode = _reference_env(runtime, task_name, steps)
    env = SimplerBridgeEnv({"task": task_name, "runtime_config": config["runtime"]["config_file"],
                            "max_episode_steps": steps, "output_dir": str(output_dir)})
    rows = []
    try:
        for seed in seeds:
            context.check_cancelled()
            rng = np.random.default_rng(1000 + seed)
            actions = np.concatenate([rng.uniform(-scale, scale, (steps, 6)),
                                      rng.choice([-1.0, 1.0], (steps, 1))], axis=1)
            reference = build()
            try:
                ref_obs, _ = reference.reset(options=options(seed))
                obs = env.reset(seed)
                ref_native = reference.unwrapped
                worst_pose = float(np.abs(_positions(ref_native) - _positions(env.native)).max())
                worst_image = float(np.abs(ref_obs["image"][CAMERA]["rgb"].astype(np.int16)
                                           - obs["rgb"][CAMERA].astype(np.int16)).max())
                done_mismatch = 0
                for action in actions:
                    ref_obs, _, ref_done, _, _ = reference.step(action)
                    transition = env.step(action)
                    worst_pose = max(worst_pose, float(np.abs(_positions(ref_native) - _positions(env.native)).max()))
                    worst_image = max(worst_image, float(np.abs(ref_obs["image"][CAMERA]["rgb"].astype(np.int16)
                                                                - transition["obs"]["rgb"][CAMERA].astype(np.int16)).max()))
                    done_mismatch += bool(ref_done) != transition["info"]["task_done_now"]
            finally:
                reference.close()
            rows.append({"seed": seed, "max_position_diff": worst_pose, "max_image_diff": worst_image,
                         "done_flag_mismatches": done_mismatch,
                         "ok": worst_pose <= pose_tol and worst_image <= image_tol and done_mismatch == 0})
            context.event("progress", {"phase": "replay", **rows[-1]})
    finally:
        env.close()
    report = {"task": task_name, "control_mode": control_mode, "steps": steps, "action_scale": scale,
              "pose_tol": pose_tol, "image_tol": image_tol, "passed": all(r["ok"] for r in rows), "episodes": rows}
    context.write_json("replay.json", report)
    return {"data": {k: report[k] for k in ("task", "passed", "control_mode", "steps")} | {"episodes": len(rows)},
            "outputs": [{"path": "replay.json", "kind": "report", "media_type": "application/json"}]}
