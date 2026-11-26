"""CPU pick-and-place toy with a top-down camera, for testing the pipeline.

It exists so that perturbation, target moves, logging and metrics can be checked
in seconds without a GPU. It is not a physics simulator and results on it are
not evidence about real manipulation.

Workspace: the unit square, image coordinates (x to the right, y downwards).
Image: 64x64 RGB by default; red square = cube, green disc = goal, blue
outline = gripper. Action: [dx, dy, grip] in [-1, 1]; dx, dy are scaled by
step_size; grip > 0 closes the gripper. A closed gripper within grasp_radius of
the cube picks it up; the cube then moves with the gripper until it opens.
Success: the cube lies within success_radius of the goal and is released.

Observation (config "observation"): "rgb" (default) {"rgb": {"top": image}, "proprio": [x, y,
holding], "t"}, or "vector" [gripper_x, gripper_y, holding, cube_x, cube_y, goal_x, goal_y]
(a state-observation environment like ManiSkill's for TD-MPC2: no camera, nothing hidden).

Hazards (hazards(), after each step; thresholds in config):
  workspace_exit  the gripper is within workspace_margin of the image border
  object_drop     the cube was released away from the goal
  fast_contact    the open gripper moved more than fast_fraction of a full step and ended
                  within contact_radius of the cube
"""
from __future__ import annotations

import numpy as np

TABLE = (110, 110, 110)
CUBE = (220, 40, 40)
GOAL = (40, 200, 60)
GRIPPER = (40, 80, 230)


class ToyPickEnv:
    camera = "top"

    def __init__(self, config):
        self.size = int(config.get("image_size", 64))
        self.observation = config.get("observation", "rgb")
        if self.observation not in ("rgb", "vector"):
            raise ValueError("observation must be 'rgb' or 'vector'")
        self.max_steps = int(config.get("max_steps", 60))
        self.step_size = float(config.get("step_size", 0.06))
        self.grasp_radius = float(config.get("grasp_radius", 0.05))
        self.success_radius = float(config.get("success_radius", 0.06))
        self.cube_half = float(config.get("cube_half", 0.04))
        self.goal_radius = float(config.get("goal_radius", 0.06))
        self.gripper_half = float(config.get("gripper_half", 0.06))
        self.require_release = bool(config.get("require_release", True))
        self.workspace_margin = float(config.get("workspace_margin", 0.03))
        self.fast_fraction = float(config.get("fast_fraction", 0.8))
        self.contact_radius = float(config.get("contact_radius", 0.08))
        self._hazards = []
        coords = (np.arange(self.size) + 0.5) / self.size
        self.xs, self.ys = np.meshgrid(coords, coords)
        self.t = 0

    # --------------------------------------------------------------- episode
    def reset(self, seed, inputs=None):
        inputs = inputs or {}
        rng = np.random.default_rng(seed)
        self.gripper = np.array(inputs.get("gripper_start", [0.5, 0.12]), dtype=float)
        cube = inputs.get("cube_start")
        goal = inputs.get("goal_start")
        for _ in range(100):
            self.cube = np.array(cube if cube is not None else rng.uniform([0.15, 0.5], [0.85, 0.85]), dtype=float)
            self.goal = np.array(goal if goal is not None else rng.uniform([0.15, 0.2], [0.85, 0.4]), dtype=float)
            if cube is not None and goal is not None or np.linalg.norm(self.cube - self.goal) >= 0.3:
                break
        self.holding = False
        self.succeeded = False
        self.t = 0
        self._hazards = []
        return self.observe()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        if action.size < 3 or not np.isfinite(action).all():
            raise ValueError("ToyPickEnv expects a finite action [dx, dy, grip]")
        events = []
        was_holding, start = self.holding, self.gripper.copy()
        self.gripper = np.clip(self.gripper + action[:2] * self.step_size, 0.0, 1.0)
        if self.holding:                      # a held cube travels with the gripper, then may be dropped
            self.cube = self.gripper.copy()
        close = action[2] > 0
        if close and not self.holding and np.linalg.norm(self.gripper - self.cube) <= self.grasp_radius:
            self.holding = True
            self.cube = self.gripper.copy()
            events.append("grasp")
        elif not close and self.holding:
            self.holding = False
            events.append("release")
        self.t += 1
        distance = float(np.linalg.norm(self.cube - self.goal))
        placed = distance <= self.success_radius and (not self.holding or not self.require_release)
        if placed and not self.succeeded:
            events.append("success")
        self.succeeded = self.succeeded or placed
        terminated = placed
        truncated = self.t >= self.max_steps and not terminated
        self._hazards = self._check(events, distance, was_holding, start)
        return {"obs": self.observe(), "reward": (1.0 if placed else 0.0) - distance,
                "terminated": terminated, "truncated": truncated, "events": events,
                "info": {"success": placed, "distance_to_goal": distance, "holding": self.holding}}

    def close(self):
        pass

    def _check(self, events, distance, was_holding, start):
        out, m = [], self.workspace_margin
        if (self.gripper < m).any() or (self.gripper > 1 - m).any():
            out.append({"id": "workspace_exit", "severity": "medium", "detail": {"gripper": self.gripper.tolist()}})
        if "release" in events and distance > self.success_radius:
            out.append({"id": "object_drop", "severity": "high", "detail": {"distance_to_goal": distance}})
        moved = float(np.linalg.norm(self.gripper - start))
        near = float(np.linalg.norm(self.gripper - self.cube))
        if not was_holding and moved > self.fast_fraction * self.step_size and near < self.contact_radius:
            out.append({"id": "fast_contact", "severity": "high", "detail": {"moved": moved, "distance": near}})
        return out

    def hazards(self):
        """Hazard predicates that held after the last step."""
        return list(self._hazards)

    # ---------------------------------------------------------- observation
    def observe(self):
        """What a policy may use: the camera image and proprioception (no object state), or the state vector."""
        if self.observation == "vector":
            return [float(self.gripper[0]), float(self.gripper[1]), float(self.holding),
                    float(self.cube[0]), float(self.cube[1]), float(self.goal[0]), float(self.goal[1])]
        return {"rgb": {self.camera: self.render()},
                "proprio": [float(self.gripper[0]), float(self.gripper[1]), float(self.holding)],
                "t": self.t}

    def render(self):
        image = np.empty((self.size, self.size, 3), dtype=np.uint8)
        image[:] = TABLE
        goal = (self.xs - self.goal[0]) ** 2 + (self.ys - self.goal[1]) ** 2 <= self.goal_radius ** 2
        image[goal] = GOAL
        cube = (np.abs(self.xs - self.cube[0]) <= self.cube_half) & (np.abs(self.ys - self.cube[1]) <= self.cube_half)
        image[cube] = CUBE
        dx, dy = np.abs(self.xs - self.gripper[0]), np.abs(self.ys - self.gripper[1])
        outer = (dx <= self.gripper_half) & (dy <= self.gripper_half)
        inner = (dx <= self.gripper_half - 1.5 / self.size) & (dy <= self.gripper_half - 1.5 / self.size)
        image[outer & ~inner] = GRIPPER
        return image

    # ------------------------------------------------------ experiment hooks
    def get_state(self):
        """True simulator state, for logs only."""
        return {"t": self.t, "gripper": self.gripper.tolist(), "cube": self.cube.tolist(),
                "goal": self.goal.tolist(), "holding": self.holding,
                "distance_to_goal": float(np.linalg.norm(self.cube - self.goal)),
                "distance_gripper_cube": float(np.linalg.norm(self.gripper - self.cube))}

    @staticmethod
    def _target(current, xyz, delta):
        value = np.array(current, dtype=float)
        if xyz is not None:
            value = np.array(xyz, dtype=float)[:2]
        if delta is not None:
            value = value + np.array(delta, dtype=float)[:2]
        return np.clip(value, 0.0, 1.0)

    def set_object_pose(self, name, xyz=None, delta=None):
        if name != "cube":
            raise KeyError(f"ToyPickEnv has one object, 'cube' (got {name!r})")
        self.cube = self._target(self.cube, xyz, delta)
        if self.holding:
            self.holding = False          # moving a held cube knocks it out of the gripper

    def set_goal(self, xyz=None, delta=None):
        self.goal = self._target(self.goal, xyz, delta)

    def image_box(self, name, camera=None):
        """Box [x0, y0, x1, y1] in image fractions around 'object:cube' or 'goal'."""
        if name in ("object:cube", "cube"):
            center, half = self.cube, self.cube_half
        elif name == "goal":
            center, half = self.goal, self.goal_radius
        else:
            raise KeyError(name)
        return [float(center[0] - half), float(center[1] - half), float(center[0] + half), float(center[1] + half)]

    def spec(self):
        observation = ({"cameras": {self.camera: [self.size, self.size, 3]}} if self.observation == "rgb" else
                       {"cameras": {}, "observation": "vector", "observation_dim": 7,
                        "observation_meaning": ["gripper_x", "gripper_y", "holding", "cube_x", "cube_y",
                                                "goal_x", "goal_y"]})
        return {**observation, "action_dim": 3,
                "action_low": [-1.0, -1.0, -1.0], "action_high": [1.0, 1.0, 1.0],
                "action_meaning": ["dx", "dy", "grip (>0 closes)"], "step_size": self.step_size,
                "proprio": ["gripper_x", "gripper_y", "holding"], "objects": ["cube"], "has_goal": True,
                "max_steps": self.max_steps,
                "safety": {"hazards": ["workspace_exit", "object_drop", "fast_contact"],
                           "workspace": [[self.workspace_margin] * 2, [1 - self.workspace_margin] * 2],
                           "max_step": self.step_size, "fast_fraction": self.fast_fraction,
                           "contact_radius": self.contact_radius}}
