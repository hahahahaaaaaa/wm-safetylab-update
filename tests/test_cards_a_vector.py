"""Minimal S1 vector dependency: windows, replay and exact neutral behavior."""
import numpy as np
import pytest
from types import SimpleNamespace
from wms.plugins.cards.a_obs_corruption import VectorObsNoise


def test_window_dtype_replay_and_no_mutation():
    events = []
    ctx = SimpleNamespace(t=19, rng=np.random.default_rng(42), log=lambda **kw: events.append(kw))
    obj = VectorObsNoise({'std': [1,2], 'c': .3, 'start': 20, 'end': 50})
    obj.reset({}, ctx)
    obs = [1., 2.]
    assert obj.on_observation(obs, ctx) is obs
    ctx.t = 20
    actual = obj.on_observation(obs, ctx)
    np.testing.assert_allclose(actual, np.array(obs) + np.random.default_rng(42).normal(size=2) * [.3,.6])
    assert obs == [1., 2.] and events[0]['injected']
    ctx.t = 50
    assert obj.on_observation(obs, ctx) is obs
    ctx.t, obj.c = 21, 0
    assert obj.on_observation(obs, ctx) is obs


def test_wrong_dimensions():
    ctx = SimpleNamespace(t=0, rng=np.random.default_rng(42), log=lambda **kw: None)
    obj = VectorObsNoise({'std': [1,2]})
    obj.reset({}, ctx)
    with pytest.raises(ValueError):
        obj.on_observation([1,2,3], ctx)
