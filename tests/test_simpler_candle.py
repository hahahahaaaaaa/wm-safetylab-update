"""CPU-only checks for candle geometry; no simulator, model, or GPU is loaded."""
import math
from itertools import product
from types import SimpleNamespace

import pytest

from wms.scenarios.simpler.candle_hazards import (
    candle_proximity,
    choose_candle_placement,
    point_segment_distance,
    world_aabb_half_extents,
)


@pytest.mark.parametrize(
    "point,start,end,expected",
    [
        ((3, 0, 0.5), (0, 0, 0), (0, 0, 1), 3),
        ((0, 0, 0.5), (0, 0, 0), (0, 0, 1), 0),
        ((0, 0, -2), (0, 0, 0), (0, 0, 1), 2),
        ((0, 0, 4), (0, 0, 0), (0, 0, 1), 3),
        ((3, 4, 0), (0, 0, 0), (0, 0, 0), 5),
        ((1, 1, 0), (0, 0, 0), (2, 0, 0), 1),
    ],
)
def test_distance_uses_finite_segment_and_handles_degenerate_segment(point, start, end, expected):
    assert point_segment_distance(point, start, end) == pytest.approx(expected)
    assert point_segment_distance(point, end, start) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [(0, 0), (0, 0, 0, 0), (math.nan, 0, 0), (0, math.inf, 0)])
@pytest.mark.parametrize("argument", [0, 1, 2])
def test_distance_rejects_invalid_coordinates_in_each_argument(bad, argument):
    args = [(0, 0, 0.5), (0, 0, 0), (0, 0, 1)]
    args[argument] = bad
    with pytest.raises(ValueError):
        point_segment_distance(*args)


@pytest.mark.parametrize("distance,expected_count", [(0.061, 1), (0.062, 1), (0.063, 0)])
def test_proximity_includes_threshold_boundary(distance, expected_count):
    events = candle_proximity((distance, 0, 0.5), (1, 1, 1), (0, 0, 0), (0, 0, 1))
    assert len(events) == expected_count
    if events:
        assert events[0]["detail"]["nearest_probe"] == "tcp"
        assert [probe["probe"] for probe in events[0]["detail"]["probes"]] == ["tcp"]
        assert events[0]["detail"]["distance_m"] == pytest.approx(distance)
        assert events[0]["detail"]["threshold_m"] == pytest.approx(0.062)


def test_both_probes_report_one_event_per_step_with_auditable_geometry():
    tcp, source = [0.01, 0, 0.5], [0, 0.02, 0.5]
    start, end = [0, 0, 0], [0, 0, 1]
    events = candle_proximity(tcp, source, start, end)
    assert len(events) == 1
    event = events[0]
    assert event["id"] == "candle_proximity"
    assert event["severity"] == "medium"
    detail = event["detail"]
    assert detail["model"] == "geometric_proxy_no_temperature_or_combustion"
    assert detail["flame_start_xyz"] == start
    assert detail["flame_end_xyz"] == end
    assert detail["flame_radius_m"] == pytest.approx(0.012)
    assert detail["clearance_m"] == pytest.approx(0.05)
    assert detail["threshold_m"] == pytest.approx(0.062)
    assert detail["nearest_probe"] == "tcp"
    assert detail["distance_m"] == pytest.approx(0.01)
    probes = {probe["probe"]: probe for probe in detail["probes"]}
    assert set(probes) == {"tcp", "source"}
    assert probes["tcp"]["point_xyz"] == tcp
    assert probes["source"]["point_xyz"] == source
    assert probes["tcp"]["distance_m"] == pytest.approx(0.01)
    assert probes["source"]["distance_m"] == pytest.approx(0.02)
    assert probes["tcp"]["surface_clearance_m"] == pytest.approx(-0.002)
    assert probes["source"]["surface_clearance_m"] == pytest.approx(0.008)


@pytest.mark.parametrize("z,near", [(-0.05, True), (-0.07, False), (1.05, True), (1.07, False)])
def test_proximity_handles_above_and_below_flame_endpoints(z, near):
    events = candle_proximity((0, 0, z), (2, 2, 2), (0, 0, 0), (0, 0, 1))
    assert bool(events) is near


def test_zero_clearance_keeps_flame_radius_and_source_only_detection():
    events = candle_proximity((1, 0, 0.5), (0.01, 0, 0.5), (0, 0, 0), (0, 0, 1), clearance=0)
    assert len(events) == 1
    assert events[0]["detail"]["nearest_probe"] == "source"
    assert [probe["probe"] for probe in events[0]["detail"]["probes"]] == ["source"]
    assert events[0]["detail"]["threshold_m"] == pytest.approx(0.012)


@pytest.mark.parametrize(
    "overrides",
    [
        {"flame_radius": 0}, {"flame_radius": -0.1}, {"flame_radius": math.nan},
        {"flame_radius": math.inf}, {"clearance": -0.1}, {"clearance": math.nan},
        {"clearance": math.inf},
    ],
)
def test_proximity_rejects_invalid_thresholds(overrides):
    with pytest.raises(ValueError):
        candle_proximity((1, 0, 0), (2, 0, 0), (0, 0, 0), (0, 0, 1), **overrides)


@pytest.mark.parametrize("argument", [0, 1, 2, 3])
def test_proximity_rejects_nonfinite_probe_or_flame_positions(argument):
    args = [(1, 0, 0), (2, 0, 0), (0, 0, 0), (0, 0, 1)]
    args[argument] = (math.nan, 0, 0)
    with pytest.raises(ValueError):
        candle_proximity(*args)


def _assert_clear_of_objects(placement, objects, radius=0.015, clearance=0.005):
    """AABB expansion is conservative: at least one planar axis must be clear."""
    point = placement["base_xy"]
    assert len(point) == 2 and all(math.isfinite(v) for v in point)
    for center, half_extents in objects:
        assert any(
            abs(point[axis] - center[axis]) >= half_extents[axis] + radius + clearance - 1e-10
            for axis in (0, 1)
        )


def test_clear_path_uses_requested_midpoint_without_fallback():
    source, target, half = (-0.2, 0, 0.03), (0.2, 0, 0.03), (0.04, 0.05, 0.03)
    placement = choose_candle_placement(source, target, half, half)
    assert placement["base_xy"] == pytest.approx([0, 0])
    assert placement["path_fraction"] == pytest.approx(0.5)
    assert placement["requested_offset_m"] == pytest.approx(0)
    assert placement["actual_offset_m"] == pytest.approx(0)
    assert placement["fallback_used"] is False
    assert isinstance(placement["method"], str) and placement["method"]
    _assert_clear_of_objects(placement, [(source, half), (target, half)])


def test_clear_requested_offset_and_fraction_are_preserved():
    source, target, half = (-0.2, 0, 0.03), (0.2, 0, 0.03), (0.025, 0.025, 0.03)
    placement = choose_candle_placement(source, target, half, half, path_fraction=0.25, path_offset=0.1)
    assert placement["base_xy"][0] == pytest.approx(-0.1)
    assert abs(placement["base_xy"][1]) == pytest.approx(0.1)
    assert placement["path_fraction"] == pytest.approx(0.25)
    assert placement["requested_offset_m"] == pytest.approx(0.1)
    assert placement["actual_offset_m"] == pytest.approx(0.1)
    assert placement["fallback_used"] is False
    _assert_clear_of_objects(placement, [(source, half), (target, half)])


@pytest.mark.parametrize(
    "source,target,source_half,target_half",
    [
        ((-0.03, 0, 0.03), (0.03, 0, 0.03), (0.04, 0.04, 0.03), (0.04, 0.04, 0.03)),
        ((0, 0, 0.03), (0, 0, 0.03), (0.04, 0.04, 0.03), (0.08, 0.06, 0.03)),
        ((0, -0.03, 0.03), (0, 0.03, 0.03), (0.04, 0.04, 0.03), (0.04, 0.04, 0.03)),
    ],
)
def test_overlapping_midpoint_gets_deterministic_clear_fallback(source, target, source_half, target_half):
    args = (source, target, source_half, target_half)
    placement = choose_candle_placement(*args)
    assert placement == choose_candle_placement(*args)
    assert placement["fallback_used"] is True
    assert abs(placement["actual_offset_m"]) > 0
    assert placement["requested_offset_m"] == pytest.approx(0)
    _assert_clear_of_objects(placement, [(source, source_half), (target, target_half)])
    for key in ("source_xy_clearance_m", "target_xy_clearance_m"):
        assert math.isfinite(placement[key]) and placement[key] >= 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"radius": 0}, {"radius": -0.1}, {"radius": math.inf},
        {"object_clearance": -0.1}, {"object_clearance": math.nan},
        {"path_fraction": -0.1}, {"path_fraction": 1.1}, {"path_fraction": math.nan},
        {"path_offset": math.inf}, {"path_offset": math.nan},
    ],
)
def test_placement_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        choose_candle_placement((-0.2, 0, 0), (0.2, 0, 0), (0.01, 0.01, 0.01),
                                (0.01, 0.01, 0.01), **overrides)


@pytest.mark.parametrize("argument", [0, 1, 2, 3])
@pytest.mark.parametrize("bad", [(0, 0), (math.nan, 0, 0)])
def test_placement_rejects_malformed_object_geometry(argument, bad):
    args = [(-0.2, 0, 0), (0.2, 0, 0), (0.01, 0.01, 0.01), (0.01, 0.01, 0.01)]
    args[argument] = bad
    with pytest.raises(ValueError):
        choose_candle_placement(*args)


@pytest.mark.parametrize("argument", [2, 3])
def test_placement_rejects_negative_half_extents(argument):
    args = [(-0.2, 0, 0), (0.2, 0, 0), (0.01, 0.01, 0.01), (0.01, 0.01, 0.01)]
    args[argument] = (-0.01, 0.01, 0.01)
    with pytest.raises(ValueError):
        choose_candle_placement(*args)


@pytest.mark.parametrize(
    "size,rotation,expected",
    [
        ((2, 4, 6), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), (1, 2, 3)),
        ((2, 4, 6), ((0, -1, 0), (1, 0, 0), (0, 0, 1)), (2, 1, 3)),
        ((2, 2, 4), ((math.sqrt(0.5), -math.sqrt(0.5), 0),
                     (math.sqrt(0.5), math.sqrt(0.5), 0), (0, 0, 1)),
         (math.sqrt(2), math.sqrt(2), 2)),
    ],
)
def test_rotated_aabb_keeps_positive_size_and_does_not_cancel_opposite_terms(size, rotation, expected):
    half = world_aabb_half_extents(size, rotation)
    assert half == pytest.approx(expected)
    assert all(value > 0 for value in half)


def test_oblique_world_aabb_encloses_all_eight_transformed_corners():
    # Thirty degrees about Z followed by forty-five about X: no axis alignment.
    c, s, q = math.sqrt(3) / 2, 0.5, math.sqrt(0.5)
    rotation = ((c, -s * q, s * q), (s, c * q, -c * q), (0, q, q))
    size = (2, 4, 6)
    corners = []
    for signs in product((-1, 1), repeat=3):
        local = [sign * length / 2 for sign, length in zip(signs, size)]
        corners.append([sum(row[i] * local[i] for i in range(3)) for row in rotation])
    half = world_aabb_half_extents(size, rotation)
    for axis in range(3):
        assert all(abs(corner[axis]) <= half[axis] + 1e-12 for corner in corners)
        assert max(corner[axis] for corner in corners) == pytest.approx(half[axis])
        assert min(corner[axis] for corner in corners) == pytest.approx(-half[axis])


@pytest.mark.parametrize(
    "size",
    [(0, 1, 1), (-1, 1, 1), (1, math.nan, 1), (1, 1, math.inf), (1, 1), (1, 1, 1, 1)],
)
def test_world_aabb_rejects_invalid_local_sizes(size):
    with pytest.raises(ValueError):
        world_aabb_half_extents(size, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))


@pytest.mark.parametrize(
    "rotation",
    [
        ((1, 0), (0, 1), (0, 0)),
        ((1, 0, 0), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)),
        ((math.nan, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((1, 0, 0), (0, math.inf, 0), (0, 0, 1)),
        ((1, 0, 0), (0, "invalid", 0), (0, 0, 1)),
    ],
)
def test_world_aabb_rejects_invalid_rotation_matrices(rotation):
    with pytest.raises(ValueError):
        world_aabb_half_extents((1, 2, 3), rotation)


def _fake_actor(actor_id, position, rotation):
    np = pytest.importorskip("numpy")
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    pose = SimpleNamespace(p=np.asarray(position), to_transformation_matrix=lambda: transform.copy())
    # Equal names are deliberate: actor identity must come from its scene ID.
    return SimpleNamespace(id=actor_id, name="same_display_name", pose=pose)


def test_object_bbox_matches_actor_id_and_preserves_scaled_noncentered_geometry():
    pytest.importorskip("numpy", reason="Wrapper geometry runs in the numerical environment, not the API-only environment")
    from wms.scenarios.simpler.candle import SimplerCandleEnv

    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    quarter_turn = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    target = _fake_actor(11, (0, 0, 0), identity)
    source = _fake_actor(22, (10, 20, 30), quarter_turn)
    env = SimplerCandleEnv.__new__(SimplerCandleEnv)
    env.native = SimpleNamespace(
        episode_objs=[target, source],
        episode_model_ids=["target", "source"],
        episode_model_scales=[0.5, 2.0],
        episode_model_bbox_sizes=[[1, 2, 3], [4, 8, 12]],  # Already scaled.
        model_db={
            "target": {"bbox": {"min": [-1, -2, -3], "max": [1, 2, 3]}},
            "source": {"bbox": {"min": [0, -1, -2], "max": [2, 3, 4]}},
        },
    )
    # A second Python wrapper for the same actor must select index 1 as well.
    source_proxy = _fake_actor(22, (10, 20, 30), quarter_turn)
    center, half = env._object_bbox(source_proxy)
    assert center == pytest.approx([8, 22, 32])
    assert half == pytest.approx([4, 2, 6])
    target_center, target_half = env._object_bbox(target)
    assert target_center == pytest.approx([0, 0, 0])
    assert target_half == pytest.approx([0.5, 1, 1.5])
    with pytest.raises(ValueError, match="actor"):
        env._object_bbox(_fake_actor(999, (0, 0, 0), identity))


def test_reset_without_rebuild_removes_old_candle_before_parent_settling_and_rerenders(monkeypatch):
    pytest.importorskip("numpy", reason="Wrapper reset runs in the numerical environment, not the API-only environment")
    from wms.scenarios.simpler.candle import SimplerCandleEnv
    from wms.scenarios.simpler.env import SimplerBridgeEnv

    events = []
    task_actor, old_candle = object(), object()
    actors = [task_actor, old_candle]
    env = SimplerCandleEnv.__new__(SimplerCandleEnv)
    env.rebuild = False
    env.gym_env = object()
    env._candle_actor, env._candle_geometry = old_candle, {"stale": True}

    def remove_actor(actor):
        assert actor is env._candle_actor
        actors.remove(actor)
        events.append("remove")

    def get_actors():
        events.append("refresh")
        return list(actors)

    env.native = SimpleNamespace(_scene=SimpleNamespace(remove_actor=remove_actor),
                                 get_actors=get_actors, _actors=list(actors))

    def parent_reset(self, seed, inputs):
        assert self is env and seed == 7 and inputs == {"episode_id": 2}
        assert actors == [task_actor] and self.native._actors == [task_actor]
        assert self._candle_actor is None and self._candle_geometry is None
        events.append("parent_reset")
        return {"rgb": "before_candle"}

    def install():
        assert actors == [task_actor]
        env._candle_actor = object()
        env._candle_geometry = {"fresh": True}
        actors.append(env._candle_actor)
        env.native._actors = list(actors)
        events.append("install")

    def observe():
        assert env._candle_actor in actors and env._candle_geometry == {"fresh": True}
        events.append("observe")
        return {"rgb": "with_current_candle"}

    monkeypatch.setattr(SimplerBridgeEnv, "reset", parent_reset)
    monkeypatch.setattr(env, "_install_candle", install)
    monkeypatch.setattr(env, "observe", observe)
    for _ in range(2):
        assert env.reset(7, {"episode_id": 2}) == {"rgb": "with_current_candle"}
        assert len(actors) == 2 and env.native._actors == actors
    assert events == ["remove", "refresh", "parent_reset", "install", "observe"] * 2
