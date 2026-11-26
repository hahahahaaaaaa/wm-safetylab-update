"""#9: read-only residual, excluding predictions for a different executed action."""
from ..base import Plugin
from .b_numeric import nonnegative
from .b_residual import residual, normalized


class ResidualMonitor(Plugin):
    stage, role, side, requires = 'S3', 'monitor', 'model', ('predict', 'embed')

    def __init__(self, config):
        super().__init__(config)
        self.scale = nonnegative(self.config.get('scale', 1), 'scale', positive=True)
        self.threshold = nonnegative(self.config.get('threshold', 1), 'threshold')

    def reset(self, episode, ctx):
        self.values, self.missing, self.mismatch = [], 0, 0

    def on_prediction(self, prediction, ctx):
        return prediction

    def after_step(self, step, ctx):
        import numpy as np
        transition = step.get('transition') or {}
        proposed, action = step.get('action_proposed'), step.get('action')
        reason = None
        if transition.get('taken_over') or (proposed is not None and action is not None
                                           and not np.array_equal(proposed, action)):
            reason = 'prediction_action_mismatch'
            self.mismatch += 1
        value = None if reason else residual(transition.get('predicted_next'), transition.get('next_state'))
        if value is None and not reason:
            reason = 'missing_prediction_or_state'
            self.missing += 1
        if value is not None:
            self.values.append(value)
        ctx.log(residual=value, normalized=normalized(value, self.scale), scale=self.scale,
                threshold=self.threshold, exceeds_p99=None if value is None else value > self.threshold,
                valid=value is not None, invalid_reason=reason)

    def end_episode(self, ctx):
        return {'valid_steps': len(self.values), 'missing_steps': self.missing,
                'mismatch_steps': self.mismatch, 'mean': sum(self.values) / len(self.values) if self.values else None,
                'max': max(self.values) if self.values else None,
                'exceed_p99_steps': sum(v > self.threshold for v in self.values)}
