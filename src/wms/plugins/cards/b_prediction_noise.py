"""#8: dimension-scaled Gaussian noise on imagined rollout states only."""
from ..base import Plugin
from .b_numeric import nonnegative, std_vector, cast_finite


class PredictionNoise(Plugin):
    stage, role, side, requires = 'S3', 'fault', 'model', ('predict',)
    neutral = {'c': 0}

    def __init__(self, config):
        super().__init__(config)
        self.c = nonnegative(self.config.get('c', 0.3), 'c')
        self.source = self.config.get('source', 'both')
        if self.source not in ('prior', 'value', 'both'):
            raise ValueError('source must be prior, value or both')

    def reset(self, episode, ctx):
        self.std = std_vector(self.config, 'latent')
        self.calls = 0

    def on_prediction(self, prediction, ctx):
        import numpy as np
        meta = ctx.meta or {}
        if self.c == 0 or meta.get('kind') != 'rollout' or (self.source != 'both' and meta.get('source') != self.source):
            return prediction
        array = np.asarray(prediction)
        if array.ndim < 1 or array.shape[-1] != len(self.std) or array.dtype.kind != 'f' or not np.isfinite(array).all():
            raise ValueError('prediction must be finite floating values with last dimension matching std')
        h = meta.get('h')
        if not isinstance(h, (int, np.integer)) or h < 0:
            raise ValueError('rollout h must be a nonnegative integer')
        sigma = self.c * self.std * np.sqrt(h + 1)
        noise = ctx.rng.normal(size=array.shape) * sigma
        new = cast_finite(array.astype(float) + noise, array)
        self.calls += 1
        ctx.log(c=self.c, source=meta.get('source'), h=int(h), noise_rms=float(np.sqrt(np.mean(noise ** 2))))
        return new

    def end_episode(self, ctx):
        return {'noise_calls': self.calls}
