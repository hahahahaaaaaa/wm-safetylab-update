"""SimplerEnv-Bridge (WidowX, ManiSkill2-real2sim) as an episode-kernel environment.

Runs in a Python that has SimplerEnv installed (runtime native/memoryvla). One environment
component = one Bridge task; the official evaluation settings are the defaults:

  config: task               spoon | carrot | cube | eggplant (fills the task fields)
          env_name, scene_name, robot, rgb_overlay, robot_init_x, robot_init_y   (override)
          control_freq 5, sim_freq 500, max_episode_steps 120
          runtime_config     "$runtime.config_file" (simpler_dir and the Vulkan settings)
          rebuild_each_episode  true (default): build a fresh SimplerEnv environment on every reset,
                             as the official evaluation does; false: build once and reset it
          hazards            thresholds for wms.scenarios.hazards (workspace exit, object drop,
                             fast contact); hazards() reports them after each step

reset(seed, inputs): the object layout is SimplerEnv's episode id = inputs.episode_id or seed,
so seeds 0..23 are the official 24 episodes. Observation for a policy:
  {"rgb": {"3rd_view_camera": HxWx3 uint8}, "instruction": str, "t": step}
Action: 7 values [dx, dy, dz, axis-angle rx, ry, rz, gripper], as SimplerEnv's
widowx_bridge policies send them.

Success follows the official evaluation: whether the task is complete after the last step;
"success" is reported as an event on that step. Objects: "source" (the object to move) and
"target" (plate, towel, basket, cube below). Ported from the sub-platform's
sublab_ext.envs.simpler_bridge; the sub-platform's GPU picker is not used (the platform sets
CUDA_VISIBLE_DEVICES from the runtime file).
"""
from __future__ import annotations

import numpy as np

from wms.scenarios.hazards import SimpleHazards

from .common import (CAMERA, CONTROL_MODE, DEFAULTS, TASK_FIELDS, build_env, load_runtime_file, prepare_paths,
                     reset_options, task_settings)


class SimplerBridgeEnv:
    def __init__(self, config):
        self.config = dict(config)
        self.task_name = self.config.get("task", "carrot")
        self.task = task_settings(self.task_name, {k: self.config[k] for k in TASK_FIELDS if k in self.config})
        self.runtime = load_runtime_file(self.config.get("runtime_config"))
        prepare_paths(self.runtime)
        self.settings = {key: int(self.config.get(key, value)) for key, value in DEFAULTS.items()}
        self.max_steps = self.settings["max_episode_steps"]
        self.rebuild = bool(self.config.get("rebuild_each_episode", True))
        self.hazard_check = SimpleHazards(self.config.get("hazards"), dt=1.0 / self.settings["control_freq"])
        self.gym_env = None
        self.t, self._last, self._raw_obs, self.instruction = 0, None, None, None

    def _build(self):
        if self.gym_env is not None:
            self.gym_env.close()
        self.gym_env = build_env(self.runtime, self.task, **self.settings)
        self.native = self.gym_env.unwrapped

    # ---------------------------------------------------------------- convert
    def _convert(self, obs):
        self._raw_obs = obs
        return {"rgb": {CAMERA: np.asarray(obs["image"][CAMERA]["rgb"], dtype=np.uint8)},
                "instruction": self.instruction, "t": self.t}

    def _wrapped_obs(self):
        """The current observation as env.step would return it (fresh render, background overlay)."""
        import gymnasium as gym
        obs = self.native.get_obs()
        chain, env = [], self.gym_env
        while isinstance(env, gym.Wrapper):
            chain.append(env)
            env = env.env
        for wrapper in reversed(chain):
            if isinstance(wrapper, gym.ObservationWrapper):
                obs = wrapper.observation(obs)
        return obs

    # ---------------------------------------------------------------- episode
    def reset(self, seed, inputs=None):
        inputs = inputs or {}
        episode_id = int(inputs.get("episode_id", seed))
        if self.gym_env is None or self.rebuild:
            self._build()
        obs, _ = self.gym_env.reset(options=reset_options(self.task, episode_id))
        self.t, self.episode_id = 0, episode_id
        self.instruction = self.native.get_language_instruction()
        self._last = self._convert(obs)
        state = self.get_state()
        self.hazard_check.reset(state["tcp_xyz"], {k: v[:3] for k, v in state["objects"].items()})
        return self._last

    def hazards(self):
        """Simple hazard predicates after the last step (true positions; see wms.scenarios.hazards)."""
        state = self.get_state()
        return self.hazard_check.update(state["tcp_xyz"], {k: v[:3] for k, v in state["objects"].items()})

    def step(self, action):
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.size != 7 or not np.isfinite(command).all():
            raise ValueError("SimplerBridgeEnv expects 7 finite values")
        obs, reward, done, truncated, info = self.gym_env.step(command)
        self.t += 1
        self.instruction = self.native.get_language_instruction()
        self._last = self._convert(obs)
        last_step = bool(truncated) or self.t >= self.max_steps
        events = ["success"] if last_step and bool(done) else []
        stats = {k: bool(v) for k, v in (info.get("episode_stats") or {}).items()}
        return {"obs": self._last, "reward": float(reward), "terminated": False, "truncated": last_step,
                "events": events, "info": {"task_done_now": bool(done), "episode_stats": stats}}

    def observe(self):
        self._last = self._convert(self._wrapped_obs())
        return self._last

    def close(self):
        if self.gym_env is not None:
            self.gym_env.close()
            self.gym_env = None

    # ------------------------------------------------------ experiment hooks
    def _actor(self, name):
        if name in ("source", "object", "cube", "carrot", "spoon", "eggplant"):
            return self.native.episode_source_obj
        if name in ("target", "goal"):
            return self.native.episode_target_obj
        raise KeyError(f"objects are 'source' and 'target' (got {name!r})")

    def get_state(self):
        def pose(actor):
            p = actor.pose
            return [float(v) for v in [*p.p, *p.q]]
        source, target = self._actor("source"), self._actor("target")
        tcp = self.native.agent.robot.get_links()[0].pose if not hasattr(self.native, "tcp") else self.native.tcp.pose
        src, tgt = np.asarray(source.pose.p), np.asarray(target.pose.p)
        stats = getattr(self.native, "episode_stats", {}) or {}
        return {"t": self.t, "episode_id": getattr(self, "episode_id", None),
                "objects": {"source": pose(source), "target": pose(target)},
                "tcp_xyz": [float(v) for v in tcp.p],
                "distance_to_goal": float(np.linalg.norm(src[:2] - tgt[:2])),
                "episode_stats": {k: bool(v) for k, v in stats.items()}}

    def _teleport(self, actor, xyz=None, delta=None):
        import sapien.core as sapien
        p, q = np.array(actor.pose.p, dtype=np.float64), np.array(actor.pose.q, dtype=np.float64)
        if xyz is not None:
            for axis, value in enumerate(list(xyz)[:3]):
                if value is not None:
                    p[axis] = float(value)
        if delta is not None:
            p[:len(delta)] += np.asarray(list(delta)[:3], dtype=np.float64)
        actor.set_pose(sapien.Pose(p, q))
        for setter in ("set_velocity", "set_angular_velocity"):
            if hasattr(actor, setter):
                getattr(actor, setter)(np.zeros(3))

    def set_object_pose(self, name, xyz=None, delta=None):
        self._teleport(self._actor(name), xyz, delta)

    def set_goal(self, xyz=None, delta=None):
        self._teleport(self._actor("target"), xyz, delta)

    def image_box(self, name, camera=None):
        """[x0, y0, x1, y1] image fractions of the object's world bounding box, projected into the camera."""
        key = name.split(":", 1)[1] if name.startswith("object:") else ("target" if name == "goal" else name)
        actor = self._actor(key)
        size_attr = "episode_source_obj_bbox_world" if actor is self.native.episode_source_obj else "episode_target_obj_bbox_world"
        half = np.asarray(getattr(self.native, size_attr, np.array([0.05, 0.05, 0.05])), dtype=np.float64) / 2
        center = np.asarray(actor.pose.p, dtype=np.float64)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + center
        params = (self._raw_obs or {}).get("camera_param", {}).get(camera or CAMERA)
        if params is None:
            params = self.native.get_camera_params()[camera or CAMERA]
        extrinsic, intrinsic = np.asarray(params["extrinsic_cv"]), np.asarray(params["intrinsic_cv"])
        cam = (extrinsic @ np.c_[corners, np.ones(len(corners))].T)[:3]
        uv = (intrinsic @ cam)[:2] / cam[2]
        h, w = self._last["rgb"][camera or CAMERA].shape[:2]
        return [float(uv[0].min() / w), float(uv[1].min() / h), float(uv[0].max() / w), float(uv[1].max() / h)]

    def spec(self):
        return {"task": self.task_name, "env_name": self.task["env_name"], "action_dim": 7,
                "action_meaning": ["dx", "dy", "dz", "rot_axangle_x", "rot_axangle_y", "rot_axangle_z", "gripper"],
                "action_low": [-1.0] * 7, "action_high": [1.0] * 7,
                "cameras": {CAMERA: list(self._last["rgb"][CAMERA].shape)} if self._last else {},
                "objects": ["source", "target"], "has_goal": True, "max_steps": self.max_steps,
                "control_mode": CONTROL_MODE, "rebuild_each_episode": self.rebuild,
                "safety": self.hazard_check.spec()}
