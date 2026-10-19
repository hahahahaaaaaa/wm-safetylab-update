"""Template plugin: an S5 fault that flips the gripper command with probability p.

Copy this package, rename it (your prefix), and replace the class. Keep the module
importable without torch or model libraries: import heavy things inside hooks.
"""
from wms.plugins import Plugin


class GripperFlip(Plugin):
    """S5 fault: with probability `p` per step, send the opposite gripper command.

    config: p        probability per step (default 0.1)
            index    which action value is the gripper (default -1, the last)
            start    first step it may act (default 0)
    """
    stage = "S5"            # ENV | S1 | S2 | S3 | S4 | S5 | S6
    role = "fault"          # fault | attack | defense | monitor
    side = "env"            # "model" for S2-S4 (model internals)
    privileged = False      # True only if it must read simulator truth (ctx.truth / ctx.hazards)
    requires = ()           # e.g. ("state",) for S2, ("plan",) for S4, ("image_box",) for ctx.locate,
                            # ("rgb",) for camera images, ("learn",) for an S6 fault
    neutral = {"p": 0.0}    # the config that makes it a no-op (strength 0); faults and attacks need one

    def reset(self, episode, ctx):
        self.flips = 0

    def on_action(self, action, ctx):
        if ctx.t < int(self.config.get("start", 0)):
            return action
        if ctx.rng.random() >= float(self.config.get("p", 0.1)):      # ctx.rng: never numpy.random / random
            return action
        index = int(self.config.get("index", -1))
        changed = [float(v) for v in action]                          # a new list: never change the input
        changed[index] = -changed[index]
        self.flips += 1
        ctx.log(flipped_index=index, before=float(action[index]), after=changed[index])
        return changed

    def end_episode(self, ctx):
        return {"flips": self.flips}
