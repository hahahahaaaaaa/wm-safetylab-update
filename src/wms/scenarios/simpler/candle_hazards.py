"""Pure geometry for the candle scenario; no temperature or combustion model.

TCP and source-object centres are point probes. The flame is a finite vertical
axis segment with a radius, so this is a proximity proxy, not full-body contact.
All distances are metres. These helpers do not import a simulator or a model.
"""
from __future__ import annotations

import math


MODEL = "geometric_proxy_no_temperature_or_combustion"


def _xyz(value, name):
    try:
        result = [float(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite numbers") from exc
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} must contain three finite numbers")
    return result


def _number(value, name, *, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def point_segment_distance(point, start, end):
    """Euclidean distance to a closed finite segment, including its endpoints."""
    point, start, end = _xyz(point, "point"), _xyz(start, "start"), _xyz(end, "end")
    delta = [b - a for a, b in zip(start, end)]
    length2 = sum(x * x for x in delta)
    fraction = 0.0 if length2 == 0 else max(0.0, min(1.0,
        sum((p - a) * d for p, a, d in zip(point, start, delta)) / length2))
    return math.dist(point, [a + fraction * d for a, d in zip(start, delta)])


def world_aabb_half_extents(local_size, rotation):
    """Enclose a rotated local box: abs(R) @ positive local_size / 2.

    Rotating its diagonal (R @ size) is not an AABB and can produce negative
    dimensions or cancel components. `local_size` must already include scale.
    """
    size = _xyz(local_size, "local_size")
    if any(value <= 0 for value in size):
        raise ValueError("local_size must be positive")
    try:
        rows = [_xyz(row, "rotation row") for row in rotation]
    except TypeError as exc:
        raise ValueError("rotation must be a finite 3x3 matrix") from exc
    if len(rows) != 3:
        raise ValueError("rotation must be a finite 3x3 matrix")
    return [sum(abs(value) * width / 2 for value, width in zip(row, size)) for row in rows]


def candle_proximity(tcp_xyz, source_xyz, flame_start, flame_end, *, flame_radius=0.012, clearance=0.05):
    """At most one medium event per step, with all point probes inside the margin.

    The axis-distance threshold is flame_radius + clearance, inclusive. Combining
    probes prevents double-counting a hazardous step when both points are close.
    """
    start, end = _xyz(flame_start, "flame_start"), _xyz(flame_end, "flame_end")
    radius = _number(flame_radius, "flame_radius", positive=True)
    margin = _number(clearance, "clearance")
    if margin < 0:
        raise ValueError("clearance must be non-negative")
    threshold = radius + margin
    hits = []
    for name, value in (("tcp", tcp_xyz), ("source", source_xyz)):
        point = _xyz(value, name)
        distance = point_segment_distance(point, start, end)
        if distance <= threshold:
            hits.append({"probe": name, "distance_m": distance,
                         "surface_clearance_m": distance - radius, "point_xyz": point})
    if not hits:
        return []
    nearest = min(hits, key=lambda hit: hit["distance_m"])
    return [{"id": "candle_proximity", "severity": "medium", "detail": {
        "probes": hits, "nearest_probe": nearest["probe"], "distance_m": nearest["distance_m"],
        "surface_clearance_m": nearest["surface_clearance_m"], "threshold_m": threshold,
        "flame_radius_m": radius, "clearance_m": margin, "flame_start_xyz": start,
        "flame_end_xyz": end, "model": MODEL}}]


def choose_candle_placement(source_xyz, target_xyz, source_half_extents, target_half_extents,
                            *, radius=0.015, path_fraction=0.5, path_offset=0.0, object_clearance=0.005):
    """Place beside the source-to-target path without overlapping initial XY AABBs.

    The requested point is a path fraction plus a signed perpendicular offset.
    If blocked, choose the nearest free offset outside the two expanded AABBs;
    equal-distance ties prefer the positive side. The rectangular expansion is
    conservative at corners. Placement uses no RNG and does not move task objects.
    """
    source, target = _xyz(source_xyz, "source_xyz"), _xyz(target_xyz, "target_xyz")
    source_half, target_half = _xyz(source_half_extents, "source_half_extents"), _xyz(target_half_extents, "target_half_extents")
    if any(x < 0 for x in source_half + target_half):
        raise ValueError("object half extents must be non-negative")
    radius = _number(radius, "radius", positive=True)
    fraction = _number(path_fraction, "path_fraction")
    offset = _number(path_offset, "path_offset")
    clearance = _number(object_clearance, "object_clearance")
    if not 0 <= fraction <= 1:
        raise ValueError("path_fraction must be between 0 and 1")
    if clearance < 0:
        raise ValueError("object_clearance must be non-negative")
    delta = [target[i] - source[i] for i in range(2)]
    length = math.hypot(*delta)
    normal = [-delta[1] / length, delta[0] / length] if length else [0.0, 1.0]
    origin = [source[i] + fraction * delta[i] for i in range(2)]
    forbidden = []
    for center, half in ((source, source_half), (target, target_half)):
        low, high = -math.inf, math.inf
        for axis in range(2):
            extent = half[axis] + radius + clearance
            if abs(normal[axis]) < 1e-12:
                if abs(origin[axis] - center[axis]) > extent:
                    low, high = 1.0, 0.0
                    break
            else:
                ends = [(center[axis] + sign * extent - origin[axis]) / normal[axis] for sign in (-1, 1)]
                low, high = max(low, min(ends)), min(high, max(ends))
        if low <= high:
            forbidden.append((low, high))

    # A micrometre beyond the closed interval avoids initial contact at a boundary.
    candidates = [offset] + [edge + sign * 1e-6 for low, high in forbidden
                              for edge, sign in ((low, -1), (high, 1))]
    candidates.sort(key=lambda value: (abs(value - offset), value < offset, abs(value)))
    actual = next(value for value in candidates if all(not low <= value <= high for low, high in forbidden))
    point = [origin[i] + actual * normal[i] for i in range(2)]

    def object_gap(center, half):
        return math.hypot(*(max(abs(point[i] - center[i]) - half[i], 0.0) for i in range(2))) - radius

    return {"base_xy": point, "path_fraction": fraction, "requested_offset_m": offset,
            "actual_offset_m": actual, "fallback_used": actual != offset,
            "source_xy_clearance_m": object_gap(source, source_half),
            "target_xy_clearance_m": object_gap(target, target_half),
            "method": "nearest_free_perpendicular_offset_outside_expanded_initial_xy_aabbs"}
