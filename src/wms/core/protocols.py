"""Small in-process calling conventions; implement only the methods a task uses."""
from typing import Any, Protocol


class EnvTransition(dict):
    """obs, reward, terminated, truncated and optional cost/events/info."""


class Env(Protocol):
    def __init__(self, config: dict): ...
    def reset(self, seed: int, inputs: dict) -> Any: ...
    def step(self, action: Any) -> EnvTransition: ...
    def close(self) -> None: ...


class ModelAdapter(Protocol):
    def __init__(self, config: dict): ...
    def generate(self, request: dict) -> Any: ...
    def predict(self, request: dict) -> Any: ...
    def act(self, observation: Any, instruction: str | None = None) -> Any: ...


class Planner(Protocol):
    def __init__(self, config: dict): ...
    def plan(self, observation: Any, step: int, context: Any) -> Any: ...


# The episode kernel (wms.episodes.kernel) needs more than rollout does: per-episode
# reset of the policy, a re-render after objects move, and the true state for logs.
# Only that kernel checks these; rollout and the older components are unaffected.

class EpisodePolicy(Protocol):
    info: dict                                     # raw output of the last act(), logged per step
    def reset(self, seed: int) -> None: ...        # new episode: clear memory, seed randomness
    def act(self, observation: Any, instruction: str | None = None) -> Any: ...


class EpisodeEnv(Env, Protocol):
    def spec(self) -> dict: ...                    # action_dim, cameras, objects, max_steps
    def observe(self) -> Any: ...                  # current observation, re-rendered
    def get_state(self) -> dict: ...               # true state; logged, never shown to the policy


EPISODE_POLICY = ("reset", "act")
EPISODE_ENV = ("reset", "step", "close", "spec", "observe", "get_state")
# Optional: image_box(name, camera) for occluders that follow an object;
# set_object_pose(name, xyz, delta) / set_goal(xyz, delta) for target moves;
# policy.bind_env(spec).


def missing_methods(component, names):
    return [name for name in names if not callable(getattr(component, name, None))]
