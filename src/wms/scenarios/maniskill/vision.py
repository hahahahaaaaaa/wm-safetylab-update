"""ManiSkill 3 camera environments with the hooks the episode kernel needs.

Runs only in a native Python that has ManiSkill installed (runtime native/maniskill),
on a machine with a GPU renderer. Ported from the sub-platform's
sublab_ext.envs.maniskill_vision; the sub-platform's GPU picker is not used (the platform sets
CUDA_VISIBLE_DEVICES from the runtime file). The older PandaPushCubeEnv (adapter.py, state
observations) stays for the maniskill-hold rollout recipe. Supported tasks and their object names:

  PickCube-v1   objects: cube           goal: goal_site (not drawn by the camera)
  StackCube-v1  objects: cubeA, cubeB   goal: none (the target is cubeB)
  PushCube-v1   objects: cube           goal: goal_region

Observation given to a policy:
  {"rgb": {camera: HxWx3 uint8}, "depth": {camera: HxWx1} (rgbd only),
   "proprio": robot qpos, "extra": task inputs ManiSkill exposes in camera mode
   (tcp_pose, goal_pos for PickCube), "t": step}
True object poses are only available through get_state(), for logging.

executor "planner" uses ManiSkill's official motion-planning solutions
(control_mode pd_joint_pos). They read privileged simulator state once per
phase and ignore images, so image perturbations do not change their actions;
target moves after planning do. A visual policy is plugged in as a model.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np

from wms.scenarios.hazards import SimpleHazards

TASKS = {
    "PickCube-v1": {"objects": {"cube": "cube"}, "goal": "goal_site", "planner": "pick_cube"},
    "StackCube-v1": {"objects": {"cubeA": "cubeA", "cubeB": "cubeB"}, "goal": None, "planner": "stack_cube"},
    # PushCube succeeds when the cube's xy lies inside the flat goal region, so its distance is xy only
    "PushCube-v1": {"objects": {"cube": "obj"}, "goal": "goal_region", "planner": "push_cube", "goal_xy": True},
}


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _first(value):
    """Value of the single parallel environment."""
    array = _array(value)
    return array[0] if array.ndim and array.shape[0] == 1 else array


class EpisodeFinished(Exception):
    """Raised inside the planner's env.step once the episode has ended."""


class ManiSkillVisionEnv:
    def __init__(self, config):
        self.config = dict(config)
        self.env_id = self.config.get("env_id", "PickCube-v1")
        if self.env_id not in TASKS:
            raise ValueError(f"env_id must be one of {sorted(TASKS)}")
        self.task = TASKS[self.env_id]
        self._apply_runtime(self.config.get("runtime_config"))
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 - registers the tasks
        width, height = self.config.get("camera_size", [128, 128])
        self.control_mode = self.config.get("control_mode", "pd_joint_pos")
        self.sim_backend = self.config.get("sim_backend", "physx_cpu")
        self.max_episode_steps = int(self.config.get("max_episode_steps", 100))
        self.gym_env = gym.make(
            self.env_id, robot_uids=self.config.get("robot_uids", "panda"), num_envs=1,
            obs_mode=self.config.get("obs_mode", "rgb"), control_mode=self.control_mode,
            render_mode="rgb_array", sim_backend=self.sim_backend,
            render_backend=self.config.get("render_backend", "gpu"),
            sensor_configs=dict(width=int(width), height=int(height)),
            max_episode_steps=self.max_episode_steps)
        self.native = self.gym_env.unwrapped
        self.action_dim = int(self.gym_env.action_space.shape[-1])
        self.t, self._succeeded, self._done, self._last = 0, False, False, None
        freq = float(getattr(self.native, "control_freq", 20) or 20)
        self.hazard_check = SimpleHazards(self.config.get("hazards"), dt=1.0 / freq)

    @staticmethod
    def _apply_runtime(path):
        if not path:
            return
        from wms.engines.maniskill.runtime import native_environment
        from wms.engines.runtime import load_runtime
        values = native_environment(load_runtime(path))
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            values["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]   # keep the selected card
        os.environ.update(values)
        try:  # keep SAPIEN's UI state file inside the workspace cache
            import sapien
            cache = Path(os.environ.get("XDG_CACHE_HOME", "."))
            cache.mkdir(parents=True, exist_ok=True)
            sapien.render.set_imgui_ini_filename(str(cache / "imgui.ini"))
        except Exception:  # noqa: BLE001 - older SAPIEN builds have no such call
            pass

    # ---------------------------------------------------------------- convert
    def _convert(self, obs):
        out = {"rgb": {}, "t": self.t}
        for camera, data in (obs.get("sensor_data") or {}).items():
            if "rgb" in data:
                out["rgb"][camera] = _first(data["rgb"]).astype(np.uint8)
            if "depth" in data:
                out.setdefault("depth", {})[camera] = _first(data["depth"])
        agent = obs.get("agent") or {}
        if "qpos" in agent:
            out["proprio"] = _first(agent["qpos"]).astype(float).tolist()
        extra = obs.get("extra") or {}
        out["extra"] = {key: _first(value).tolist() for key, value in extra.items()}
        return out

    def _transition(self, obs, reward, terminated, truncated, info):
        self.t += 1
        success = bool(_first(info["success"]).reshape(-1)[0]) if "success" in info else False
        events = ["success"] if success and not self._succeeded else []
        self._succeeded = self._succeeded or success
        self._last = self._convert(obs)
        done_t = bool(_first(terminated).reshape(-1)[0])
        done_r = bool(_first(truncated).reshape(-1)[0])
        self._done = done_t or done_r
        return {"obs": self._last, "reward": float(_first(reward).reshape(-1)[0]),
                "terminated": done_t, "truncated": done_r, "events": events,
                "info": {"success": success, **{k: _first(v).tolist() for k, v in info.items()
                                                 if k != "success" and hasattr(v, "shape") and _array(v).size <= 16}}}

    # ---------------------------------------------------------------- episode
    def reset(self, seed, inputs=None):
        obs, _ = self.gym_env.reset(seed=int(seed))
        self.t, self._succeeded, self._done = 0, False, False
        self._last = self._convert(obs)
        self._reset_hazards()
        return self._last

    def _reset_hazards(self):
        state = self.get_state()
        self.hazard_check.reset(state["tcp_xyz"], {k: v[:3] for k, v in state["objects"].items()})

    def hazards(self):
        """Simple hazard predicates after the last step (true positions; see wms.scenarios.hazards)."""
        state = self.get_state()
        return self.hazard_check.update(state["tcp_xyz"], {k: v[:3] for k, v in state["objects"].items()})

    def step(self, action):
        command = np.asarray(action, dtype=np.float32).reshape(1, -1)
        if command.shape[1] != self.action_dim or not np.isfinite(command).all():
            raise ValueError(f"{self.env_id}/{self.control_mode} expects {self.action_dim} finite values")
        return self._transition(*self.gym_env.step(command))

    def observe(self):
        """Render the current state again (after a target move)."""
        self._last = self._convert(self.native.get_obs())
        return self._last

    def close(self):
        self.gym_env.close()

    # ------------------------------------------------------ experiment hooks
    def _actor(self, name):
        if name == "goal":
            if not self.task["goal"]:
                raise KeyError(f"{self.env_id} has no separate goal; move an object instead")
            return getattr(self.native, self.task["goal"])
        if name not in self.task["objects"]:
            raise KeyError(f"{self.env_id} objects: {sorted(self.task['objects'])} (got {name!r})")
        return getattr(self.native, self.task["objects"][name])

    def get_state(self):
        state = {"t": self.t,
                 "qpos": _first(self.native.agent.robot.get_qpos()).astype(float).tolist(),
                 "tcp_xyz": _first(self.native.agent.tcp.pose.p).astype(float).tolist(),
                 "objects": {name: _first(getattr(self.native, attr).pose.raw_pose).astype(float).tolist()
                             for name, attr in self.task["objects"].items()}}
        if self.task["goal"]:
            goal = _first(self._actor("goal").pose.p).astype(float)
            cube = np.asarray(state["objects"]["cube"][:3])
            state["goal"] = goal.tolist()
            dims = 2 if self.task.get("goal_xy") else 3
            state["distance_to_goal"] = float(np.linalg.norm(cube[:dims] - goal[:dims]))
        else:
            a, b = (np.asarray(state["objects"][k][:3]) for k in ("cubeA", "cubeB"))
            state["distance_to_goal"] = float(np.linalg.norm(a[:2] - b[:2]))
        state["success_so_far"] = self._succeeded
        return state

    def _teleport(self, actor, xyz=None, delta=None):
        import torch
        from mani_skill.utils.structs.pose import Pose
        p, q = actor.pose.p.clone(), actor.pose.q.clone()
        if xyz is not None:
            values = list(xyz) + [None] * (3 - len(xyz))
            for axis, value in enumerate(values[:3]):
                if value is not None:
                    p[0, axis] = float(value)
        if delta is not None:
            p[0, :len(delta)] += torch.as_tensor(list(delta)[:3], dtype=p.dtype, device=p.device)
        actor.set_pose(Pose.create_from_pq(p=p, q=q))
        for setter in ("set_linear_velocity", "set_angular_velocity"):
            try:
                getattr(actor, setter)(torch.zeros((1, 3), dtype=p.dtype, device=p.device))
            except Exception:  # noqa: BLE001 - kinematic goal markers have no velocity
                pass
        if getattr(self.native, "gpu_sim_enabled", False):
            scene = self.native.scene
            scene._gpu_apply_all()
            scene.px.gpu_update_articulation_kinematics()
            scene._gpu_fetch_all()

    def set_object_pose(self, name, xyz=None, delta=None):
        self._teleport(self._actor(name), xyz, delta)

    def set_goal(self, xyz=None, delta=None):
        self._teleport(self._actor("goal"), xyz, delta)

    def spec(self):
        space = self.gym_env.action_space
        return {"env_id": self.env_id, "control_mode": self.control_mode, "action_dim": self.action_dim,
                "action_low": _array(space.low).reshape(-1)[-self.action_dim:].tolist(),
                "action_high": _array(space.high).reshape(-1)[-self.action_dim:].tolist(),
                "cameras": {cam: list(img.shape) for cam, img in (self._last or {}).get("rgb", {}).items()},
                "objects": sorted(self.task["objects"]), "has_goal": bool(self.task["goal"]),
                "max_steps": self.max_episode_steps, "sim_backend": self.sim_backend,
                "safety": self.hazard_check.spec()}

    # ------------------------------------------------------- planner executor
    def run_planner(self, seed, hooks, max_steps):
        """Run the official motion-planning solution once.

        hooks.on_reset(obs), hooks.before_step(t, action) and
        hooks.after_step(t, action, transition) are called around every
        simulator step, so targets can move and every frame is logged.
        """
        import gymnasium as gym
        if self.control_mode not in ("pd_joint_pos", "pd_joint_pos_vel"):
            raise ValueError("executor 'planner' needs control_mode pd_joint_pos")
        solve = importlib.import_module(
            "mani_skill.examples.motionplanning.panda.solutions." + self.task["planner"]).solve
        adapter = self

        class Hooked(gym.Wrapper):
            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                adapter.t, adapter._succeeded, adapter._done = 0, False, False
                adapter._last = adapter._convert(obs)
                adapter._reset_hazards()
                hooks.on_reset(adapter._last)
                return obs, info

            def step(self, action):
                if adapter._done:
                    raise EpisodeFinished("terminated_or_truncated")
                if adapter.t >= max_steps:
                    raise EpisodeFinished("step_budget")
                t = adapter.t
                proposed = _array(action).reshape(-1).tolist()
                executed = hooks.before_step(t, proposed)      # an S5 plugin may change the action
                if executed is not None and list(executed) != proposed:
                    original = _array(action)
                    action = np.asarray(executed, dtype=original.dtype).reshape(original.shape)
                out = self.env.step(action)
                transition = adapter._transition(*out)
                hooks.after_step(t, _array(action).reshape(-1).tolist(), transition)
                return out

        try:
            result = solve(Hooked(self.gym_env), seed=int(seed), debug=False, vis=False)
            status = "no_plan" if isinstance(result, (int, np.integer)) and result == -1 else "planner_completed"
        except EpisodeFinished as finished:
            status = str(finished)
        except Exception as stopped:  # noqa: BLE001 - a plugin's stop carries the episode status
            if not hasattr(stopped, "episode_status"):
                raise
            status = stopped.episode_status
        return status
