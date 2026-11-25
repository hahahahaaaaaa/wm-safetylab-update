"""Card #4: accepted-frame mean absolute difference, in pixel units."""
from ..base import Plugin
from .a_common import images, changed_images, nonnegative


def frame_score(image, previous):
    if previous is None:
        return 0.0
    import numpy as np
    if image.shape != previous.shape:
        raise ValueError('frame shapes differ')
    return float(np.abs(image.astype(float)-previous.astype(float)).mean())


class FrameGuard(Plugin):
    stage, role, requires = 'S1', 'defense', ('rgb',)
    neutral = {'tau': None}

    def __init__(self, config):
        super().__init__(config)
        self.tau = self.config.get('tau')
        if self.tau is not None:
            self.tau = nonnegative(self.tau, 'tau')
        self.limit = self.config.get('max_replacements', 5)
        if type(self.limit) is not int or self.limit < 1:
            raise ValueError('max_replacements must be a positive integer')

    def reset(self, episode, ctx):
        self.previous, self.count = {}, {}

    def on_observation(self, obs, ctx):
        out = {}
        for camera, image in images(obs, self.config.get('camera', '*')):
            previous = self.previous.get(camera)
            score = frame_score(image, previous)
            replaced = previous is not None and self.tau is not None and score > self.tau and self.count.get(camera,0) < self.limit
            forced = previous is not None and self.tau is not None and score > self.tau and self.count.get(camera,0) >= self.limit
            if replaced:
                out[camera] = previous.copy()
                self.count[camera] = self.count.get(camera,0)+1
                ctx.alarm('frame_replaced', score=score)
            else:
                self.previous[camera] = image.copy()
                self.count[camera] = 0
            ctx.log(camera=camera, score=score, replaced=replaced, forced_accept=forced, threshold=self.tau)
        return changed_images(obs, out) if out else obs
