"""Image-only scripted baseline for wms.scenarios.toy.env:ToyPickEnv.

It finds the cube (red) and goal (green) by colour in the camera image, moves
the gripper there, closes, carries and opens. Gripper position comes from
proprioception, which perturbations do not touch. With memory=True it keeps
using the last estimate for up to memory_max_age steps when the cube or goal
cannot be seen; without memory it stops and waits.

It is a reference point for the pipeline, not a learned model.

Model probe "state" (for S2 plugins): the memory, {"cube": [x, y] | None, "goal": ...,
"age": {"cube": n, "goal": n}}. With hooks set, act() offers it to on_state twice: "read"
before the image is used (a plugin may clear or replace the memory) and "written" after the
new estimates are stored (a plugin may refuse the write by returning the read value).
"""
from __future__ import annotations

import numpy as np


class CentroidPickPolicy:
    probes = ("state",)

    def __init__(self, config):
        self.camera = config.get("camera", "top")
        self.memory = bool(config.get("memory", False))
        self.memory_max_age = int(config.get("memory_max_age", 30))
        self.step_size = float(config.get("step_size", 0.06))
        self.grasp_distance = float(config.get("grasp_distance", 0.035))
        self.place_distance = float(config.get("place_distance", 0.03))
        self.min_pixels = int(config.get("min_pixels", 3))
        self.colour_margin = float(config.get("colour_margin", 60))
        self.hooks = None
        self.reset(0)

    # ---------------------------------------------------------------- probes
    def set_hooks(self, hooks):
        self.hooks = hooks

    def export_state(self):
        return {name: (None if self.last[name] is None else [float(v) for v in self.last[name]])
                for name in ("cube", "goal")} | {"age": dict(self.age)}

    def import_state(self, state):
        for name in ("cube", "goal"):
            value = state.get(name)
            self.last[name] = None if value is None else np.asarray(value, dtype=float)
        self.age = {name: int((state.get("age") or {}).get(name, self.age[name])) for name in ("cube", "goal")}

    def _offer_state(self, phase):
        state = self.export_state()
        changed = self.hooks.state(state, phase)
        if changed is not state:
            self.import_state(changed)

    def reset(self, seed=None):
        self.last = {"cube": None, "goal": None}
        self.age = {"cube": 0, "goal": 0}
        self.info = {}

    def bind_env(self, spec):
        self.step_size = float(spec.get("step_size", self.step_size))

    def _detect(self, image, channel):
        data = image.astype(np.int16)
        others = np.delete(data, channel, axis=2).max(axis=2)
        mask = data[..., channel] - others > self.colour_margin
        count = int(mask.sum())
        if count < self.min_pixels:
            return None, count
        rows, cols = np.nonzero(mask)
        height, width = mask.shape
        return np.array([(cols.mean() + 0.5) / width, (rows.mean() + 0.5) / height]), count

    def _estimate(self, name, image, channel):
        found, pixels = self._detect(image, channel)
        return self._remember(name, found, pixels)

    def _remember(self, name, found, pixels=None):
        """Store a fresh estimate, or fall back on memory when the object is not seen."""
        if found is not None:
            self.last[name], self.age[name] = found, 0
            return found, "image", pixels
        if self.memory and self.last[name] is not None and self.age[name] < self.memory_max_age:
            self.age[name] += 1
            return self.last[name], "memory", pixels
        return None, "lost", pixels

    def act(self, observation, instruction=None):
        image = np.asarray(observation["rgb"][self.camera])
        gripper = np.asarray(observation["proprio"][:2], dtype=float)
        holding = bool(observation["proprio"][2] > 0.5)
        probed = self.hooks is not None and self.hooks.wants("on_state")
        if probed:
            self._offer_state("read")
        cube, cube_source, cube_pixels = self._estimate("cube", image, 0)
        goal, goal_source, goal_pixels = self._estimate("goal", image, 1)
        if probed:
            self._offer_state("written")
        target = goal if holding else cube
        phase = "carry" if holding else "reach"
        if target is None:
            move, grip = np.zeros(2), (1.0 if holding else -1.0)
            phase += "_wait"
        else:
            move = np.clip((target - gripper) / self.step_size, -1.0, 1.0)
            after = gripper + move * self.step_size
            near = np.linalg.norm(after - target)
            if holding:
                grip = -1.0 if near <= self.place_distance else 1.0
            else:
                grip = 1.0 if near <= self.grasp_distance else -1.0
        self.info = {"phase": phase,
                     "estimate_cube": None if cube is None else cube.tolist(), "source_cube": cube_source,
                     "pixels_cube": cube_pixels, "age_cube": self.age["cube"],
                     "estimate_goal": None if goal is None else goal.tolist(), "source_goal": goal_source,
                     "pixels_goal": goal_pixels, "age_goal": self.age["goal"]}
        return [float(move[0]), float(move[1]), float(grip)]

    def close(self):
        pass
