"""Versioned residual for S6 closed-loop deviation, not a physical safety score."""
from __future__ import annotations

RESIDUAL_VERSION = "planned-next-rms/v1"


def planned_next_residual(step, scale=1.0, include_changed=False):
    """RMS((predicted_next - embed(next_obs)) / scale), with strict validity gates.

    Predictions refer to the policy's proposed command. Modified commands and
    takeovers are excluded by default; including them measures loop deviation,
    not model error under the executed command. Never read simulator truth.
    """
    import numpy as np
    transition = step.get("transition")
    if not transition:
        return {"valid": False, "reason": "no_feedback", "residual": None}
    try:
        proposed = np.asarray(step["action_proposed"], dtype=float)
        sent = np.asarray(step["action"], dtype=float)
        if proposed.shape != sent.shape or not np.isfinite(proposed).all() or not np.isfinite(sent).all():
            return {"valid": False, "reason": "invalid_action", "residual": None}
        changed = not np.allclose(proposed, sent, atol=1e-7, rtol=0)
        if transition.get("taken_over"):
            return {"valid": False, "reason": "takeover", "action_changed": changed, "residual": None}
        if changed and not include_changed:
            return {"valid": False, "reason": "action_changed", "action_changed": True, "residual": None}
        predicted, actual = transition.get("predicted_next"), transition.get("next_state")
        if predicted is None or actual is None:
            return {"valid": False, "reason": "missing_prediction_or_embedding", "residual": None}
        a, b = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
        divisor = np.asarray(scale, dtype=float)
        if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
            return {"valid": False, "reason": "invalid_prediction_or_embedding", "residual": None}
        if not np.isfinite(divisor).all() or (divisor <= 0).any() or divisor.ndim > 1:
            raise ValueError("residual scale must be finite and positive")
        flat = (a - b).reshape(-1)
        if divisor.size not in (1, flat.size):
            raise ValueError("residual scale must be scalar or match the flattened latent dimension")
        value = float(np.sqrt(np.mean((flat / divisor.reshape(-1)) ** 2)))
        if not np.isfinite(value):
            return {"valid": False, "reason": "nonfinite_residual", "residual": None}
        return {"valid": True, "reason": None, "residual": value, "dimensions": int(a.size),
                "action_changed": changed, "version": RESIDUAL_VERSION,
                "predicted": a.reshape(-1).tolist(), "observed": b.reshape(-1).tolist(),
                "scale": divisor.reshape(-1).tolist()}
    except (KeyError, TypeError):
        return {"valid": False, "reason": "invalid_feedback_structure", "residual": None}
