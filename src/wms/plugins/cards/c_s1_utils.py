"""Shared helpers for C-workspace S1 observation plugins."""
from __future__ import annotations

def camera_items(obs, camera="*"):
    if not isinstance(obs, dict): return []
    rgb=obs.get("rgb")
    if not isinstance(rgb, dict) or not rgb: return []
    if camera in (None, "*"): return list(rgb.items())
    if camera in rgb: return [(camera, rgb[camera])]
    return list(rgb.items()) if len(rgb)==1 else []

def replace_rgb(obs, updates):
    if not updates: return obs
    out=dict(obs); rgb=dict(obs.get("rgb") or {}); rgb.update(updates); out["rgb"]=rgb
    return out

def in_window(t, start=0, end=None):
    return int(t)>=int(start) and (end is None or int(t)<int(end))
