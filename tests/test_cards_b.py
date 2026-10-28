"""Algorithm and contract checks for the four S3/S4 cards."""
from types import SimpleNamespace
import pytest
import numpy as np
from wms.plugins.cards.b_score_misspec import ScoreMisspec
from wms.plugins.cards.b_constrained_rerank import ConstrainedRerank
from wms.plugins.cards.b_prediction_noise import PredictionNoise
from wms.plugins.cards.b_residual import residual, normalized
from wms.plugins.cards.b_residual_monitor import ResidualMonitor


def context(**values):
    events, alarms = [], []
    return SimpleNamespace(rng=np.random.default_rng(17), meta={'kind': 'rollout', 'source': 'value', 'h': 2},
                           log=lambda **kw: events.append(kw), alarm=lambda reason, **kw: alarms.append((reason, kw)),
                           events=events, alarms=alarms, **values)


def plan():
    return {'candidates': np.array([[[.1, .2], [.3, .4]], [[1, 1], [.8, .9]], [[.2, .2], [.1, .1]]], np.float32),
            'scores': np.array([0, 1, 2], np.float32), 'other': 'preserved'}


def test_score_formula_axes_dtype_and_no_mutation():
    p, ctx = plan(), context()
    before = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in p.items()}
    out = ScoreMisspec({'c': 2}).on_plan(p, ctx)
    expected = p['scores'] + 2 * np.std([0., 1., 2.]) * np.array([.25, .925, .15])
    np.testing.assert_allclose(out['scores'], expected, rtol=1e-6)
    assert out['scores'].dtype == np.float32 and out['other'] == 'preserved'
    for key in ('scores', 'candidates'):
        np.testing.assert_array_equal(p[key], before[key])
    assert out['candidates'] is p['candidates']


@pytest.mark.parametrize('cfg', [{'c': 0}, {'mode': 'scale', 'scale': 1}])
def test_score_neutral_exact(cfg):
    p = plan()
    assert ScoreMisspec(cfg).on_plan(p, context()) is p


def test_scale_and_constant():
    p = plan()
    out = ScoreMisspec({'mode': 'scale', 'scale': 2}).on_plan(p, context())
    np.testing.assert_array_equal(out['scores'], [0, 2, 4])
    assert np.argmax(out['scores']) == np.argmax(p['scores'])
    p['scores'][:] = 3
    np.testing.assert_array_equal(ScoreMisspec({'c': 5}).on_plan(p, context())['scores'], p['scores'])


@pytest.mark.parametrize('cfg', [{'c': -1}, {'c': float('nan')}, {'c': float('inf')}, {'scale': 0}, {'scale': -1}, {'mode': 'typo'}])
def test_bad_score_config(cfg):
    with pytest.raises(ValueError):
        ScoreMisspec(cfg)


@pytest.mark.parametrize('key,value', [('scores', np.ones(2)), ('scores', np.ones((3,1))),
    ('scores', np.array([1,2,3])), ('scores', np.array([1.,np.nan,2.])), ('candidates', np.ones((3,2)))])
def test_bad_plan(key, value):
    p = {**plan(), key: value}
    with pytest.raises(ValueError):
        ScoreMisspec({}).on_plan(p, context())


def test_score_overflow():
    with pytest.raises(ValueError):
        ScoreMisspec({'mode': 'scale', 'scale': 1e300}).on_plan(plan(), context())


def test_rerank_tiny_violation_beats_huge_original_score():
    p = {'candidates': np.array([[[1.00001]], [[.1]], [[.9]]], np.float32),
         'scores': np.array([10000., -100., 0.], np.float32)}
    out = ConstrainedRerank({'v_max': 1}).on_plan(p, context())
    assert out['scores'][0] < min(out['scores'][1:])
    assert out['scores'][2] > out['scores'][1]
    np.testing.assert_array_equal(p['scores'], [10000, -100, 0])
    assert out['candidates'] is p['candidates']


def test_rerank_feasible_all_violating_and_neutral():
    p, ctx = plan(), context()
    assert ConstrainedRerank({'v_max': None}).on_plan(p, ctx) is p
    np.testing.assert_array_equal(ConstrainedRerank({'v_max': 10}).on_plan(p, ctx)['scores'], p['scores'])
    out = ConstrainedRerank({'v_max': 0}).on_plan(p, ctx)
    assert np.isfinite(out['scores']).all()
    assert ctx.alarms[0][0] == 'all_candidates_violate'
    assert 'takeover' not in ctx.alarms[0][1]


def test_noise_formula_replay_source_and_neutral():
    p = np.ones((4, 2), np.float32)
    obj, ctx = PredictionNoise({'c': .3, 'std': [1, 2]}), context()
    obj.reset({}, ctx)
    expected = p + np.random.default_rng(17).normal(size=p.shape) * .3 * np.array([1,2]) * np.sqrt(3)
    out = obj.on_prediction(p, ctx)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)
    assert out.dtype == p.dtype and np.all(p == 1)
    ctx.meta['kind'] = 'chosen_next'
    assert obj.on_prediction(p, ctx) is p
    ctx.meta['kind'] = 'rollout'
    obj.source = 'prior'
    assert obj.on_prediction(p, ctx) is p
    obj.source, obj.c = 'both', 0
    assert obj.on_prediction(p, ctx) is p


def test_noise_dimension_validation_and_reset():
    obj, ctx = PredictionNoise({'std': [1, 2]}), context()
    obj.reset({}, ctx)
    with pytest.raises(ValueError):
        obj.on_prediction(np.ones((4,3)), ctx)
    obj.on_prediction(np.ones((4,2)), ctx)
    assert obj.end_episode(ctx)['noise_calls'] == 1
    obj.reset({}, ctx)
    assert obj.end_episode(ctx)['noise_calls'] == 0


def test_residual_matches_legacy_with_partial_missing():
    assert residual([3., 4., float('nan')], [0., 0., 0.]) == 5
    assert residual(None, [0]) is None
    assert residual([float('nan')], [0]) is None
    with pytest.raises(ValueError):
        residual([[1,2]], [1,2])
    assert normalized(5, 2) == 2.5 and normalized(None, 2) is None
    with pytest.raises(ValueError):
        normalized(None, 0)


def test_monitor_missing_mismatch_no_alarm_and_reset():
    obj, ctx = ResidualMonitor({'scale': 2, 'threshold': 4}), context()
    obj.reset({}, ctx)
    step = {'action_proposed': [1], 'action': [1], 'transition': {'predicted_next': [3,4], 'next_state': [0,0]}}
    obj.after_step(step, ctx)
    obj.after_step({**step, 'action': [2]}, ctx)
    obj.after_step({'transition': {}}, ctx)
    assert obj.end_episode(ctx) == {'valid_steps': 1, 'missing_steps': 1, 'mismatch_steps': 1,
                                   'mean': 5, 'max': 5, 'exceed_p99_steps': 1}
    assert ctx.events[0]['normalized'] == 2.5 and ctx.events[0]['exceeds_p99']
    assert not ctx.alarms
    obj.reset({}, ctx)
    assert obj.end_episode(ctx)['mean'] is None
