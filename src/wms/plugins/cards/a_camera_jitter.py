"""Card #2: image-plane affine jitter, not a simulator camera-pose change."""
from ..base import Plugin
from .a_common import active, window, images, changed_images, nonnegative


class CameraJitter(Plugin):
    stage, role, requires = 'S1', 'fault', ('rgb',)
    neutral = {'shift': 0, 'rotate': 0, 'scale': 0}

    def __init__(self, config):
        super().__init__(config)
        self.shift = nonnegative(self.config.get('shift', .05), 'shift')
        self.rotate = nonnegative(self.config.get('rotate', 5), 'rotate')
        self.scale = nonnegative(self.config.get('scale', .05), 'scale')
        self.mode = self.config.get('mode', 'persistent')
        if self.scale >= 1 or self.shift > 1 or self.mode not in ('persistent', 'per_step'):
            raise ValueError('scale must be <1, shift <=1; mode persistent or per_step')
        window(self.config)

    def _sample(self, ctx):
        return (ctx.rng.uniform(-self.shift,self.shift), ctx.rng.uniform(-self.shift,self.shift),
                ctx.rng.uniform(-self.rotate,self.rotate), 1+ctx.rng.uniform(-self.scale,self.scale))

    def reset(self, episode, ctx):
        self.transforms = {}

    def on_observation(self, obs, ctx):
        if self.shift == self.rotate == self.scale == 0 or not active(self.config, ctx.t):
            return obs
        from wms.perturb import affine
        out = {}
        for camera, image in images(obs, self.config.get('camera', '*')):
            if camera not in self.transforms or self.mode == 'per_step':
                self.transforms[camera] = self._sample(ctx)
            dx, dy, angle, scale = self.transforms[camera]
            dx, dy = dx*image.shape[1], dy*image.shape[0]
            out[camera] = affine(image, dx, dy, angle, scale)
            ctx.log(injected=True, camera=camera, dx=dx, dy=dy, angle=angle, scale=scale, mode=self.mode)
        return changed_images(obs, out)
