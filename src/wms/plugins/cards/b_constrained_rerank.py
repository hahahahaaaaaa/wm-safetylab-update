"""#11: feasible-first score ranking using a first-action norm proxy."""
from ..base import Plugin
from .b_numeric import nonnegative, plan_arrays, cast_finite


class ConstrainedRerank(Plugin):
    stage, role, side, requires = 'S4', 'defense', 'model', ('plan',)
    neutral = {'v_max': None}

    def __init__(self, config):
        super().__init__(config)
        value = self.config.get('v_max')
        self.v_max = None if value is None else nonnegative(value, 'v_max')

    def on_plan(self, plan, ctx):
        import numpy as np
        candidates, scores = plan_arrays(plan)
        if self.v_max is None:
            return plan
        norms = np.linalg.norm(candidates[:, 0, :].astype(float), axis=-1)
        excess = np.maximum(0, norms - self.v_max)
        violates = excess > 0
        original = scores.astype(float)
        with np.errstate(over='ignore', invalid='ignore'):
            new = cast_finite(original - (np.ptp(original) + 1) * (violates + excess), scores)
        # Float casting can erase a unit-sized separation at extreme magnitudes.
        # Fail explicitly rather than silently claim feasible-first semantics.
        if violates.any() and (~violates).any() and new[violates].max() >= new[~violates].min():
            raise ValueError('score precision cannot represent feasible-first separation')
        if violates.all():
            ctx.alarm('all_candidates_violate', score=float(excess.min()))
        ctx.log(v_max=self.v_max, violating_candidates=int(violates.sum()), candidates=int(len(scores)),
                min_excess=float(excess.min()), iteration=(ctx.meta or {}).get('iteration'))
        return {**plan, 'scores': new}
