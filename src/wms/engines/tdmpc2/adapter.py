"""Optional small components for the platform's generic predict/rollout tasks.

The official MPPI policy stays inside act(). Generic platform CEM is not layered
over TD-MPC2 a second time. Predictions are explicitly latent representations.

Model probes (for S2-S4 plugins, through set_hooks; numpy arrays, float32):
  state    on_state({"latent": z (latent_dim,), "plan_mean": (horizon, action_dim) | None}, "read"):
           the encoded observation, and MPPI's warm start carried over from the previous step.
           Returning plan_mean None clears the warm start: this step plans as the first step of
           an episode does (t0), which is what "wipe the memory" means for TD-MPC2.
  predict  on_prediction(z_next (N, latent_dim), {"kind": "rollout", "h": h, "source": "prior" | "value"})
           for every latent dynamics step inside planning: h = 0, 1, ... counts the imagined steps
           of one rollout; source "prior" for the policy-prior rollouts at the start of planning,
           "value" for each MPPI iteration's value estimate (h restarts at 0 for each); after the choice,
           on_prediction(z_next (latent_dim,), {"kind": "chosen_next"}) for the executed action (a report;
           also made when only the kernel wants it, for transition["predicted_next"])
  plan     on_plan({"candidates": (N, horizon, action_dim), "scores": (N,)}, {"iteration": i, "iterations": n})
           once per MPPI iteration; returned scores replace the value estimates (re-ranking)
  embed    embed(obs) -> model.encode(obs) (latent_dim,): an observation in the space of the
           predictions (transition["next_state"]); no gradient, no random draw, no policy state
Without hooks, act() is the official call unchanged; with hooks it makes the same calls in the
same order (encode, then plan), so plugins that return their input leave every action unchanged.
"""
from . import compat


class ModelAdapter:
    probes = ("state", "predict", "plan", "embed")

    def __init__(self, config):
        self.config = config
        self.agent = self.cfg = None
        self.first = True
        self.cold = False
        self.hooks = None

    def _load(self, seed):
        if self.agent is None:
            from wms.engines.runtime import load_runtime
            runtime = load_runtime(self.config["runtime_config"])
            compat.prepare(runtime)
            compat.seed_all(seed)
            from .native import _source
            checkpoint = _source(self.config, self.config.get("output_dir", "."))
            self.config = {**self.config, "checkpoint": str(checkpoint)}
            self.agent, self.cfg, self.metadata = compat.checkpoint_load(checkpoint)
            # Checkpoint construction initializes then replaces weights. Use an
            # explicit policy RNG stream, independently of that initialization.
            compat.seed_all(seed)

    def predict(self, request):
        seed = int(request.get("seed", self.config.get("seed", 0)))
        self._load(seed)
        import torch
        from common.math import two_hot_inv
        obs = torch.as_tensor(request["observation"], device="cuda", dtype=torch.float32).reshape(1, -1)
        if obs.shape[1] != self.cfg.obs_shape["state"][0]:
            raise ValueError("Observation must match the checkpoint's full ManiSkill state layout")
        predictions, rewards = [], []
        with torch.no_grad():
            z = self.agent.model.encode(obs, None)
            for raw in request["actions"]:
                action = torch.as_tensor(raw, device="cuda", dtype=torch.float32).reshape(1, -1)
                if action.shape[1] != self.cfg.action_dim or not bool(torch.isfinite(action).all()) or bool((action.abs() > 1).any()):
                    raise ValueError("Actions must be normalized pd_joint_delta_pos with matching dimension")
                rewards.append(float(two_hot_inv(self.agent.model.reward(z, action, None), self.cfg).item()))
                z = self.agent.model.next(z, action, None)
                predictions.append(z[0].cpu().tolist())
        return {"predictions": predictions, "predicted_rewards": rewards,
                "representation": "tdmpc2_latent", "physical_state_decoded": False,
                "checkpoint": str(self.config["checkpoint"]), "seed": seed}

    def act(self, observation, instruction=None):
        self._load(int(self.config.get("seed", 0)))
        import torch
        from .backend import act
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        if self.hooks is None:
            action = act(self.agent, obs, first=self.first)
        else:
            action = self._act_probed(obs)
        self.first = False
        return action[0].tolist()

    # ------------------------------------------------------------ probes
    def set_hooks(self, hooks):
        self.hooks = hooks

    def embed(self, observation):
        self._load(int(self.config.get("seed", 0)))
        import torch
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            z = self.agent.model.encode(obs.to(self.agent.device, non_blocking=True), None)
        return z[0].cpu().numpy()

    def export_state(self):
        mean = getattr(self.agent, "_prev_mean_eval", None) if self.agent is not None else None
        return {"plan_mean": None if mean is None or self.first else mean[0].detach().cpu().numpy()}

    def import_state(self, state):
        import torch
        value = state.get("plan_mean")
        if value is None:
            self.cold = True                       # the next plan starts without a warm start
            return
        ref = self.agent._prev_mean_eval
        mean = torch.as_tensor(value, dtype=ref.dtype, device=ref.device).reshape(ref.shape)
        self.agent._prev_mean_eval = mean
        self.agent._prev_mean = mean

    def _act_probed(self, obs):
        """agent.act(obs, t0=first, eval_mode=True) spelled out, with the probes in between."""
        import numpy as np
        import torch
        agent, hooks, first = self.agent, self.hooks, self.first
        self.cold = False
        if not first and hasattr(agent, "_prev_mean_eval") and not hasattr(agent, "_prev_mean"):
            agent._prev_mean = agent._prev_mean_eval            # as backend.act does
        with torch.no_grad():
            z = agent.model.encode(obs.to(agent.device, non_blocking=True), None)
            if hooks.wants("on_state"):
                state = {"latent": z[0].cpu().numpy(), **self.export_state()}
                changed = hooks.state(state, "read")
                if changed["latent"] is not state["latent"]:
                    z = torch.as_tensor(np.asarray(changed["latent"]), dtype=z.dtype, device=z.device).reshape(z.shape)
                if changed.get("plan_mean") is not state["plan_mean"]:
                    self.import_state(changed)
            if not agent.cfg.mpc:
                a = agent.model.pi(z, None)[0]
            else:
                undo = self._wrap(agent, hooks, np, torch)
                try:
                    a = agent.plan(z, t0=first or self.cold, eval_mode=True)
                finally:
                    for restore in undo:
                        restore()
            if hooks.wants("chosen_next"):
                hooks.prediction(agent.model.next(z, a, None)[0].cpu().numpy(), {"kind": "chosen_next"})
        return a.cpu()

    def _wrap(self, agent, hooks, np, torch):
        """Route planning's value estimates and latent steps through the hooks; returns undo calls."""
        undo = []
        wants_plan, wants_prediction = hooks.wants("on_plan"), hooks.wants("on_prediction")
        rollout = {"h": 0, "source": "prior"}         # plan() runs the policy-prior rollouts first
        if wants_plan or wants_prediction:
            estimate, iterations = agent._estimate_value, int(agent.cfg.iterations)
            counter = iter(range(10 ** 9))

            def estimate_value(z, actions, task):
                rollout.update(h=0, source="value")
                value = estimate(z, actions, task)
                if not wants_plan:
                    return value
                plan = {"candidates": actions[0].permute(1, 0, 2).cpu().numpy(), "scores": value[0, :, 0].cpu().numpy()}
                changed = hooks.plan(plan, {"iteration": next(counter), "iterations": iterations})
                if changed["scores"] is not plan["scores"]:
                    value = torch.as_tensor(np.asarray(changed["scores"]), dtype=value.dtype,
                                            device=value.device).reshape(value.shape)
                return value
            agent._estimate_value = estimate_value
            undo.append(lambda: delattr(agent, "_estimate_value"))
        if wants_prediction:
            model, step = agent.model, agent.model.next

            def next_latent(z, a, task):
                out = step(z, a, task)
                flat = out.reshape(-1, out.shape[-1]).cpu().numpy()
                meta = {"kind": "rollout", "h": rollout["h"], "source": rollout["source"]}
                rollout["h"] += 1
                changed = hooks.prediction(flat, meta)
                if changed is not flat:
                    out = torch.as_tensor(np.asarray(changed), dtype=out.dtype, device=out.device).reshape(out.shape)
                return out
            model.next = next_latent
            undo.append(lambda: delattr(model, "next"))
        return undo

    def reset(self, seed=0):
        """For direct Python multi-episode use; generic rollout creates a fresh adapter.

        The generic runner does not call this hook; its recipe must explicitly
        set model.config.seed to the same value as the top-level episode seed.
        """
        self.config = {**self.config, "seed": int(seed)}
        self._load(seed)
        compat.seed_all(seed)
        self.first = True

    def close(self):
        self.agent = None


class EnvironmentAdapter:
    def __init__(self, config):
        from wms.engines.runtime import load_runtime
        compat.prepare(load_runtime(config["runtime_config"]))
        import torch
        from omegaconf import OmegaConf
        from .native import _source
        checkpoint = _source(config, config.get("output_dir", "."))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = OmegaConf.create(saved["config"])
        cfg.num_envs = cfg.num_eval_envs = 1
        self.env = compat.StateEnvironment(cfg, 1)

    def reset(self, seed, inputs):
        obs, _ = self.env.reset(seed)
        return obs[0].detach().cpu().tolist()

    def step(self, action):
        import torch
        obs, reward, terminated, truncated, info = self.env.step(torch.as_tensor(action).reshape(1, -1))
        success = bool(info["success"].item())
        return {"obs": obs[0].detach().cpu().tolist(), "reward": float(reward.item()),
                "terminated": False, "truncated": self.env.steps >= self.env.horizon,
                "events": ["task_success"] if success else [],
                "info": {"task_success": success, "native_terminated": bool(terminated.item()),
                         "safety_signal_status": "not_measured", "controller": "pd_joint_delta_pos"}}

    def close(self):
        self.env.close()
