"""ManiSkill PushCube with state observations, as TD-MPC2 is trained and evaluated (episode kernel).

The environment is the one TD-MPC2's official evaluation (wms.engines.tdmpc2.native:policy_evaluate)
drives: wms.engines.tdmpc2.compat.StateEnvironment, built from the checkpoint's own config
(env_id, control_mode, reward version), one parallel environment on physx_cuda, a fixed 50-step
horizon, success not ending the episode. This class only adds what the episode kernel needs.

config: runtime_config  "$runtime.config_file" (TD-MPC2 runtime)
        checkpoint | source_run   the checkpoint whose environment settings to use (as the policy)
        render          true: keep a render camera for observe()/video frames (default false)
        hazards         thresholds for wms.scenarios.hazards; hazards() reports them after each step
        reconfigure_each_episode
                        true: rebuild the GPU scene at every reset (default false, as TD-MPC2's
                        official evaluation). Without it the GPU physics keeps internal state from
                        the previous episode: two runs that differ only in episode k can differ in
                        episode k+1 from its first step on, so paired comparisons of plugins that
                        act in some episodes need it (and so does their clean baseline).

Observation: the full privileged state vector (a list), exactly what the policy is trained on.
Action: normalized pd_joint_delta_pos in [-1, 1]. "success" is reported on the first step the
official region rule holds; get_state() has the cube, goal and TCP positions.
"""
from __future__ import annotations


class ManiSkillStateEnv:
    def __init__(self, config):
        from wms.engines.runtime import load_runtime
        from wms.engines.tdmpc2 import compat
        from wms.engines.tdmpc2.native import _source
        compat.prepare(load_runtime(config["runtime_config"]))
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        self.np, self.torch = np, torch
        checkpoint = _source(config, config.get("output_dir", "."))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = OmegaConf.create(saved["config"])
        cfg.num_envs = cfg.num_eval_envs = 1
        self.cfg = cfg
        self.env = compat.StateEnvironment(cfg, 1, render=bool(config.get("render", False)))
        self.native = self.env.native
        self.t, self._obs, self._succeeded = 0, None, False
        self.reconfigure = bool(config.get("reconfigure_each_episode", False))
        from wms.scenarios.hazards import SimpleHazards
        freq = float(getattr(self.native, "control_freq", 20) or 20)
        self.hazard_check = SimpleHazards(config.get("hazards"), dt=1.0 / freq)

    def _vector(self, obs):
        return obs[0].detach().cpu().tolist()

    def reset(self, seed, inputs=None):
        obs, _ = self.env.reset(int(seed), reconfigure=self.reconfigure)
        self.t, self._succeeded = 0, False
        self._obs = self._vector(obs)
        state = self.get_state()
        self.hazard_check.reset(state["tcp_xyz"], state["objects"])
        return self._obs

    def hazards(self):
        """Simple hazard predicates after the last step (true positions; see wms.scenarios.hazards)."""
        state = self.get_state()
        return self.hazard_check.update(state["tcp_xyz"], state["objects"])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.torch.as_tensor(action).reshape(1, -1))
        self.t += 1
        success = bool(info["success"].item())
        events = ["success"] if success and not self._succeeded else []
        self._succeeded = self._succeeded or success
        self._obs = self._vector(obs)
        return {"obs": self._obs, "reward": float(reward.item()), "terminated": False,
                "truncated": self.env.steps >= self.env.horizon, "events": events,
                "info": {"success": success, "native_terminated": bool(terminated.item()),
                         "controller": self.cfg.control_mode}}

    def observe(self):
        obs = self.native.get_obs()
        self._obs = self._vector(obs)
        return self._obs

    def close(self):
        self.env.close()

    def get_state(self):
        oracle = self.env.oracle()
        cube, goal, tcp = (self.np.asarray(oracle[k][0], dtype=float) for k in ("cube_xyz", "goal_xyz", "tcp_xyz"))
        return {"t": self.t, "objects": {"cube": cube.tolist()}, "goal": goal.tolist(), "tcp_xyz": tcp.tolist(),
                "distance_to_goal": float(self.np.linalg.norm(cube[:2] - goal[:2])),
                "success_so_far": self._succeeded}

    def _teleport(self, actor, xyz=None, delta=None):
        from mani_skill.utils.structs.pose import Pose
        p, q = actor.pose.p.clone(), actor.pose.q.clone()
        if xyz is not None:
            for axis, value in enumerate((list(xyz) + [None] * 3)[:3]):
                if value is not None:
                    p[0, axis] = float(value)
        if delta is not None:
            p[0, :len(delta)] += self.torch.as_tensor(list(delta)[:3], dtype=p.dtype, device=p.device)
        actor.set_pose(Pose.create_from_pq(p=p, q=q))
        for setter in ("set_linear_velocity", "set_angular_velocity"):
            try:
                getattr(actor, setter)(self.torch.zeros((1, 3), dtype=p.dtype, device=p.device))
            except Exception:  # noqa: BLE001 - kinematic goal markers have no velocity
                pass
        if getattr(self.native, "gpu_sim_enabled", False):
            scene = self.native.scene
            scene._gpu_apply_all()
            scene.px.gpu_update_articulation_kinematics()
            scene._gpu_fetch_all()

    def set_object_pose(self, name, xyz=None, delta=None):
        if name != "cube":
            raise KeyError(f"PushCube has one object, 'cube' (got {name!r})")
        self._teleport(self.native.obj, xyz, delta)

    def set_goal(self, xyz=None, delta=None):
        self._teleport(self.native.goal_region, xyz, delta)

    def spec(self):
        return {"env_id": self.cfg.env_id, "control_mode": self.cfg.control_mode, "action_dim": self.env.action_dim,
                "action_low": [-1.0] * self.env.action_dim, "action_high": [1.0] * self.env.action_dim,
                "observation": "privileged_state", "observation_dim": self.env.obs_dim, "cameras": {},
                "objects": ["cube"], "has_goal": True, "max_steps": self.env.horizon,
                "reconfigure_each_episode": self.reconfigure,
                "safety": self.hazard_check.spec()}
