"""Validation and image selection shared by the S1 cards (lazy NumPy imports)."""
from .b_numeric import nonnegative


def window(config):
    start, end = config.get('start', 0), config.get('end')
    if type(start) is not int or start < 0 or (end is not None and
            (type(end) is not int or end <= start)):
        raise ValueError('start/end must define a nonempty integer window')
    return start, end


def active(config, t):
    start, end = window(config)
    return t >= start and (end is None or t < end)


def images(obs, camera='*'):
    import numpy as np
    if not isinstance(obs, dict) or not isinstance(obs.get('rgb'), dict):
        raise ValueError('RGB observation must contain rgb camera dictionary')
    selected = [(k, v) for k, v in obs['rgb'].items() if camera in ('*', k)]
    if not selected:
        raise ValueError('configured camera is absent: ' + camera)
    for key, image in selected:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('camera image must be nonempty HxWx3 uint8')
        if min(image.shape) == 0:
            raise ValueError('empty image')
        yield key, image


def changed_images(obs, replacements):
    return {**obs, 'rgb': {**obs['rgb'], **replacements}}
