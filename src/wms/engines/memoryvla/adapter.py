"""MemoryVLA (openvla-codebase, memvla-bridge checkpoint) as a step-by-step policy (path B).

Loading comes from .common, the same code the official evaluation (simpler.py, path A)
uses; the model is upstream VLAInference, called once per step on the (possibly
perturbed) camera image, exactly as SimplerEnv's evaluation loop calls it. Ported from the
sub-platform's sublab_ext.models.memoryvla.

config: runtime_config  "$runtime.config_file" (source_dir, simpler_dir, checkpoint, llama2_dir)
        memory          true (default): the memory bank carries over within an episode, as upstream.
                        false: the memory bank is cleared before every step (memory ablation;
                        the action ensemble is unchanged)
        task            Bridge task whose eval_bridge.sh arguments build the model (default carrot;
                        they do not change the model, upstream merely passes them along)
        camera          default 3rd_view_camera
        use_bf16, cfg_scale   default true, 1.5

reset(seed) seeds torch / numpy / random with `seed`, as the official evaluation does before
each episode. info per step: memory flag, whether the bank was reset, terminate_episode, and
the raw output (translation, rotation delta, gripper).

Model probe "state" (for S2 plugins, through set_hooks): the two memory banks of upstream
MemoryVLA, {"timestep": int, "cognitive": [(timestep, tensor), ...], "perceptual": [...]}
(torch tensors on the GPU, bf16; return new lists, never change the tensors in place).
on_state(state, "read") runs before the step reads the banks (not on steps where the model
clears them: the first step of an episode and every step with memory false);
on_state(state, "written") runs after the step stored this frame. A returned state replaces
the banks. Without hooks, act() is unchanged.
"""
from __future__ import annotations

import numpy as np

from wms.scenarios.simpler.common import CAMERA, DEFAULTS, load_runtime_file, task_settings

from .common import load_inference, prepare_process, seed_all


class MemoryVLAPolicy:
    probes = ("state",)

    def __init__(self, config):
        self.config = dict(config)
        self.runtime = load_runtime_file(self.config.get("runtime_config"))
        self.memory = bool(self.config.get("memory", True))
        self.camera = self.config.get("camera", CAMERA)
        task_name = self.config.get("task", "carrot")
        task = {"name": task_name, **task_settings(task_name)}
        prepare_process(self.runtime)
        self.vla, _ = load_inference(self.runtime, task, DEFAULTS, [0, 24],
                                     use_bf16=bool(self.config.get("use_bf16", True)),
                                     cfg_scale=float(self.config.get("cfg_scale", 1.5)))
        self.first = True
        self.info = {}
        self.hooks = None

    # ------------------------------------------------------------ probes
    def set_hooks(self, hooks):
        self.hooks = hooks

    def export_state(self):
        model = self.vla.vla
        return {"timestep": int(model.cur_timestep),
                "cognitive": list(model.cog_mem_bank.bank.get(0, [])),
                "perceptual": list(model.per_mem_bank.bank.get(0, []))}

    def import_state(self, state):
        model = self.vla.vla
        if "timestep" in state:
            model.cur_timestep = int(state["timestep"])
        for key, bank in (("cognitive", model.cog_mem_bank), ("perceptual", model.per_mem_bank)):
            if key in state:
                bank.bank[0] = list(state[key])

    def _offer_state(self, phase):
        state = self.export_state()
        changed = self.hooks.state(state, phase)
        if changed is not state:
            self.import_state(changed)

    def reset(self, seed=None):
        if seed is not None:
            seed_all(int(seed))
        self.first = True

    def act(self, observation, instruction=None):
        image = np.asarray(observation["rgb"][self.camera], dtype=np.uint8)
        text = observation.get("instruction") or instruction
        if self.first:
            self.vla.reset(text)                 # upstream: model.reset(task_description) per episode
        reset_bank = self.first or not self.memory
        probed = self.hooks is not None and self.hooks.wants("on_state")
        if probed and not reset_bank:
            self._offer_state("read")
        raw, action = self.vla.step(image, text, episode_first_frame="True" if reset_bank else "False")
        if probed:
            self._offer_state("written")
        self.first = False
        self.info = {"memory": self.memory, "memory_reset": reset_bank,
                     "terminate_episode": bool(np.asarray(action["terminate_episode"]).reshape(-1)[0] > 0),
                     "raw_world_vector": np.asarray(raw["world_vector"]).tolist(),
                     "raw_rotation_delta": np.asarray(raw["rotation_delta"]).tolist(),
                     "raw_open_gripper": float(np.asarray(raw["open_gripper"]).reshape(-1)[0])}
        return np.concatenate([action["world_vector"], action["rot_axangle"],
                               np.asarray(action["gripper"]).reshape(-1)]).astype(float).tolist()

    def close(self):
        import torch
        del self.vla
        torch.cuda.empty_cache()
