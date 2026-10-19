"""Small helpers for plugins that work on nested model values (states, predictions, plans).

A model's state is whatever its probe offers: a dict of numpy arrays (toy planner, TD-MPC2),
torch tensors, lists of memory entries (MemoryVLA). A plugin that should run on all of them
treats the state as an opaque tree and changes only its floating-point leaves:

    from wms.plugins.tree import map_floats
    noisy = map_floats(state, lambda x: x + ctx.rng.normal(0, 0.1, x.shape))

Leaves that are not floating point (ints, bools, strings, None) are kept as they are.
`fn` receives a numpy float array and returns one of the same shape; tensors and Python
floats are converted to numpy and back. Nothing is changed in place.

    check_like(new, old)   None if `new` has the structure of `old` (keys, lengths, shapes,
                           dtypes), else a sentence saying where it differs; the kernel uses it
                           to refuse a hook that returns a value of the wrong form
    action_span(spec)      high - low of each action dimension, for parameters in relative units
"""
from __future__ import annotations

import numpy as np


def _is_tensor(value):
    return hasattr(value, "detach") and hasattr(value, "dtype") and hasattr(value, "shape")


def map_floats(tree, fn):
    """A copy of `tree` with fn applied to every floating-point leaf."""
    if isinstance(tree, dict):
        return {key: map_floats(value, fn) for key, value in tree.items()}
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        return type(tree)(*(map_floats(value, fn) for value in tree))
    if isinstance(tree, (list, tuple)):
        return type(tree)(map_floats(value, fn) for value in tree)
    if _is_tensor(tree):
        if not tree.dtype.is_floating_point:
            return tree
        import torch
        array = tree.detach().cpu().numpy()
        out = np.asarray(fn(array.astype(np.float64)), dtype=array.dtype).reshape(array.shape)
        return torch.as_tensor(out, dtype=tree.dtype, device=tree.device)
    if isinstance(tree, np.ndarray):
        if not np.issubdtype(tree.dtype, np.floating):
            return tree
        return np.asarray(fn(tree.astype(np.float64)), dtype=tree.dtype).reshape(tree.shape)
    if isinstance(tree, (float, np.floating)) and not isinstance(tree, bool):
        return type(tree)(np.asarray(fn(np.asarray([float(tree)])), dtype=float).reshape(-1)[0])
    return tree


def float_leaves(tree):
    """Every floating-point leaf as a numpy array (read-only use: norms, statistics)."""
    out = []
    map_floats(tree, lambda x: out.append(np.array(x)) or x)
    return out


def _kind(value):
    if isinstance(value, dict):
        return "dict"
    if _is_tensor(value) or isinstance(value, np.ndarray):
        return "array"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "number"
    return "none" if value is None else type(value).__name__


def check_like(new, old, path="value"):
    """None when `new` has the structure of `old`; else where and how it differs."""
    kind_new, kind_old = _kind(new), _kind(old)
    if kind_new != kind_old:
        return f"{path} is {kind_new}, expected {kind_old}"
    if kind_old == "dict":
        if set(new) != set(old):
            missing, extra = sorted(map(str, set(old) - set(new))), sorted(map(str, set(new) - set(old)))
            return f"{path} keys differ (missing {missing}, extra {extra})"
        for key in old:
            problem = check_like(new[key], old[key], f"{path}[{key!r}]")
            if problem:
                return problem
    elif kind_old == "list":
        if len(new) != len(old):
            return f"{path} has length {len(new)}, expected {len(old)}"
        for index, (a, b) in enumerate(zip(new, old)):
            problem = check_like(a, b, f"{path}[{index}]")
            if problem:
                return problem
    elif kind_old == "array":
        if tuple(new.shape) != tuple(old.shape):
            return f"{path} has shape {tuple(new.shape)}, expected {tuple(old.shape)}"
        if str(new.dtype) != str(old.dtype):
            return f"{path} has dtype {new.dtype}, expected {old.dtype}"
    return None


def action_span(spec):
    """high - low per action dimension (None when the environment spec gives no bounds)."""
    low, high = (spec or {}).get("action_low"), (spec or {}).get("action_high")
    if low is None or high is None:
        return None
    return np.asarray(high, dtype=float) - np.asarray(low, dtype=float)
