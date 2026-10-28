"""Validation shared by the S3/S4 cards; no model-specific branches."""
import math
import json
from pathlib import Path


def nonnegative(value, name, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def plan_arrays(plan):
    import numpy as np
    candidates, scores = np.asarray(plan['candidates']), np.asarray(plan['scores'])
    if candidates.ndim != 3 or any(n == 0 for n in candidates.shape):
        raise ValueError('candidates must have nonempty shape (N,H,A)')
    if scores.shape != (candidates.shape[0],) or scores.dtype.kind != 'f':
        raise ValueError('scores must be a floating array of shape (N,)')
    if not np.isfinite(candidates).all() or not np.isfinite(scores).all():
        raise ValueError('plan contains nonfinite values')
    return candidates, scores


def cast_finite(values, reference):
    import numpy as np
    with np.errstate(over='ignore', invalid='ignore'):
        out = np.asarray(values).astype(reference.dtype)
    if out.shape != reference.shape or not np.isfinite(out).all():
        raise ValueError('plugin output has invalid shape or nonfinite/overflowed values')
    return out


def std_vector(config, field):
    """Read once at reset. Relative paths resolve against the project, not run cwd."""
    import numpy as np
    if config.get('stats'):
        path = Path(config['stats'])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[4] / path
        values = json.loads(path.read_text(encoding='utf-8'))[field]['std']
    else:
        values = config.get('std')
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError('std must be an explicit finite nonnegative vector, or supply stats')
    return array
