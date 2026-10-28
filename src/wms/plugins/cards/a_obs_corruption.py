"""Minimal vector variant of existing S1 card #1, needed by S3 #9 acceptance."""
from ..base import Plugin
from .b_numeric import nonnegative, std_vector, cast_finite


class VectorObsNoise(Plugin):
    stage, role, requires = 'S1', 'fault', ('vector',)
    neutral = {'c': 0}

    def __init__(self, config):
        super().__init__(config)
        self.c = nonnegative(self.config.get('c', 0.1), 'c')
        self.start, self.end = self.config.get('start', 0), self.config.get('end')
        if not isinstance(self.start, int) or self.start < 0 or (self.end is not None and
                (not isinstance(self.end, int) or self.end <= self.start)):
            raise ValueError('start/end must define a nonempty nonnegative integer window')

    def reset(self, episode, ctx):
        self.std = std_vector(self.config, 'observation_vector')

    def on_observation(self, obs, ctx):
        import numpy as np
        if self.c == 0 or ctx.t < self.start or (self.end is not None and ctx.t >= self.end):
            return obs
        array = np.asarray(obs)
        if array.shape != self.std.shape or not np.isfinite(array).all():
            raise ValueError('vector observation must match std dimensions and contain finite values')
        noise = ctx.rng.normal(size=array.shape) * self.c * self.std
        changed = array.astype(float) + noise
        if not np.isfinite(changed).all():
            raise ValueError('observation noise overflow')
        ctx.log(injected=True, c=self.c, noise_norm=float(np.linalg.norm(noise)), start=self.start, end=self.end)
        if isinstance(obs, np.ndarray):
            if obs.dtype.kind != 'f':
                raise ValueError('array observation must have floating dtype')
            return cast_finite(changed, obs)
        return tuple(changed.tolist()) if isinstance(obs, tuple) else changed.tolist()
