"""#10: controlled misspecification of candidate scores (all planning iterations)."""
from ..base import Plugin
from .b_numeric import nonnegative, plan_arrays, cast_finite


class ScoreMisspec(Plugin):
    stage, role, side, requires = 'S4', 'fault', 'model', ('plan',)
    neutral = {'c': 0, 'scale': 1}

    def __init__(self, config):
        super().__init__(config)
        self.mode = self.config.get('mode', 'amplitude_bias')
        if self.mode not in ('amplitude_bias', 'scale'):
            raise ValueError('mode must be amplitude_bias or scale')
        self.c = nonnegative(self.config.get('c', 1), 'c')
        self.scale = nonnegative(self.config.get('scale', 1), 'scale', positive=True)

    def on_plan(self, plan, ctx):
        import numpy as np
        candidates, scores = plan_arrays(plan)
        if (self.mode == 'amplitude_bias' and self.c == 0) or (self.mode == 'scale' and self.scale == 1):
            return plan
        original = scores.astype(np.float64)
        with np.errstate(over='ignore', invalid='ignore'):
            changed = (original + self.c * original.std() * np.abs(candidates.astype(np.float64)).mean((1, 2))
                       if self.mode == 'amplitude_bias' else original * self.scale)
        new = cast_finite(changed, scores)
        ctx.log(mode=self.mode, c=self.c, scale=self.scale, iteration=(ctx.meta or {}).get('iteration'),
                score_mean_before=float(original.mean()), score_std_before=float(original.std()),
                score_mean_after=float(new.astype(float).mean()), score_std_after=float(new.astype(float).std()),
                top_candidate_changed=bool(np.argmax(scores) != np.argmax(new)))
        return {**plan, 'scores': new}
