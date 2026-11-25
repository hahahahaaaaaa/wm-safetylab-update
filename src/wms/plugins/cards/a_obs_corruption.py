"""Card #1: image corruptions and calibrated vector noise."""
from ..base import Plugin
from .b_numeric import nonnegative, std_vector, cast_finite
from .a_common import active, window, images, changed_images


class ObsCorruption(Plugin):
    stage, role, requires = 'S1', 'fault', ('rgb',)
    neutral = {'severity': 0}

    def __init__(self, config):
        super().__init__(config)
        self.severity = self.config.get('severity', 3)
        if type(self.severity) is not int or not 0 <= self.severity <= 5:
            raise ValueError('severity must be an integer in 0..5')
        self.kind = self.config.get('kind', 'noise')
        if self.kind not in ('noise', 'brightness', 'contrast', 'color_bias', 'blur'):
            raise ValueError('unknown image corruption kind')
        window(self.config)

    def on_observation(self, obs, ctx):
        if self.severity == 0 or not active(self.config, ctx.t):
            return obs
        from wms import perturb
        out = {}
        for camera, image in images(obs, self.config.get('camera', '*')):
            if self.kind == 'noise':
                out[camera] = perturb.noise(image, 255*(.08,.12,.18,.26,.38)[self.severity-1], ctx.rng)
            elif self.kind == 'blur':
                out[camera] = perturb.blur(image, (.5,1,2,3,4)[self.severity-1])
            else:
                out[camera] = getattr(perturb, self.kind)(image, self.severity)
            ctx.log(injected=True, kind=self.kind, severity=self.severity, camera=camera)
        return changed_images(obs, out)


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
