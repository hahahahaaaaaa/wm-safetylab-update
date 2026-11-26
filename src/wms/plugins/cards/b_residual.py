"""Shared one-transition residual for #9 and future S6 detectors."""
from .b_numeric import nonnegative


def residual(predicted_next, next_state):
    import numpy as np
    if predicted_next is None or next_state is None:
        return None
    predicted, actual = np.asarray(predicted_next, dtype=float), np.asarray(next_state, dtype=float)
    if predicted.shape != actual.shape:
        raise ValueError('residual shapes differ; broadcasting is not allowed')
    with np.errstate(invalid='ignore', over='ignore'):
        gap = (predicted - actual).reshape(-1)
    gap = gap[np.isfinite(gap)]
    if not gap.size:
        return None
    value = float(np.linalg.norm(gap))
    if not np.isfinite(value):
        raise ValueError('residual norm overflow')
    return value


def normalized(value, scale):
    scale = nonnegative(scale, 'scale', positive=True)
    if value is None:
        return None
    return nonnegative(value / scale, 'normalized residual')
