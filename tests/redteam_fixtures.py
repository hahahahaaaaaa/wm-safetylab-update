"""Plugins used only by tests/test_redteam_api.py: an attack and a privileged defense."""
from wms.plugins.base import Plugin


class NudgeAttack(Plugin):
    stage, role = "S5", "attack"
    neutral = {"delta": 0.0}
    threat = {"knowledge": "black_box", "access": "S5 action", "budget": "", "goal": "leave the workspace"}

    def on_action(self, action, ctx):
        delta = float(self.config.get("delta", 0.1))
        return [action[0] + delta, *action[1:]]


class TruthGate(Plugin):
    stage, role, privileged = "S5", "defense", True

    def on_action(self, action, ctx):
        return action
