"""The two interventions the episode kernel had before plugins, now as plugins.

params.perturbations and params.target_moves still work: the kernel turns them into
these two plugins (placed first), so old recipes and sub-platform configurations keep
their exact behaviour and logs. They are also the reference examples for new plugins.
"""
from __future__ import annotations

from wms.perturb import Perturbations, apply_target_moves

from .base import Plugin


class ImagePerturbation(Plugin):
    """S1 fault: a wms.perturb schedule (occlude, blur, noise, drop, freeze) on the camera images.

    config: schedule   list of wms.perturb items (see wms.perturb)
    An occluder that follows an object ("box": "object:cube") reads the object's image box
    from the simulator, so such a schedule makes this plugin privileged and need image_box.
    For compatibility its random stream is seeded with the episode seed itself, as before.
    """
    stage, role = "S1", "fault"
    neutral = {"schedule": []}

    def __init__(self, config):
        super().__init__(config)
        self.schedule = Perturbations(self.config.get("schedule"), seed=0)
        if self.schedule.needs_locate():
            self.privileged, self.requires = True, ("image_box",)

    def reset(self, episode, ctx):
        self.schedule.reset(ctx.seed)

    def on_observation(self, obs, ctx):
        locate = ctx.locate if self.schedule.needs_locate() else None
        seen, records = self.schedule.apply(obs, ctx.t, locate=locate)
        for record in records:
            ctx.log(record)
        return seen


class TargetMoves(Plugin):
    """ENV fault: move an object or the goal at given steps (the world changes; the policy must notice).

    config: moves   [{"step": 6, "object": "cube", "delta": [0.15, 0.0]},
                     {"step": 10, "goal": {"xyz": [...]}}]   (the kernel's normalised form)
    """
    stage, role = "ENV", "fault"
    neutral = {"moves": []}

    def __init__(self, config):
        super().__init__(config)
        self.moves = list(self.config.get("moves") or [])
        needed = ["set_goal" if "goal" in move else "set_object_pose" for move in self.moves]
        self.requires = tuple(dict.fromkeys(needed))

    def on_env(self, t, ctx):
        for record in apply_target_moves(ctx.env, self.moves, t):
            ctx.log(record)
