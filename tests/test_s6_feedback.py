import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from wms.plugins.base import Plugin
from wms.plugins.cards.c_feedback import CusumState, CusumDefense, FeedbackPoison
from wms.plugins.residuals import planned_next_residual
from wms.plugins.runtime import PluginSet, _same
from wms.plugins.conformance import TOY_ENVS, TOY_MODELS
from wms.runners.unified import UnifiedWorker
from wms.episodes.s6_metrics import audit


def ctx():
    log, alarms = [], []
    return SimpleNamespace(t=0, spec={'action_dim': 3, 'action_low': [-1.] * 3, 'action_high': [1.] * 3,
                           'action_meaning': ['dx', 'dy', 'grip (>0 closes)']},
                           log=lambda **kw: log.append(kw), records=log,
                           alarm=lambda *args, **kw: alarms.append((args, kw)), alarms=alarms)


def step():
    return {'action_proposed': [0., 0., -1.], 'action': [0., 0., -1.],
            'transition': {'predicted_next': [1., 2.], 'next_state': [0., 0.], 'taken_over': False}}


class AtZero(Plugin):
    stage, role = 'S6', 'defense'

    def on_feedback(self, transition, context):
        if context.t == 0:
            context.alarm('test stop priority', takeover='stop')
        return transition


def run(output, model='toy-planner', plugins=(), max_steps=20):
    config = {'task': 'native', 'entrypoint': 'wms.episodes.kernel:run', 'seed': 42,
              'model': TOY_MODELS[model], 'environment': TOY_ENVS['toy-pick'],
              'params': {'episodes': 1, 'max_steps': max_steps, 'video': False,
                         'plugin_strict': True, 'plugins': list(plugins)}}
    code = UnifiedWorker(config, output).run()
    return code, json.loads((output / 'result.json').read_text())


class S6Test(unittest.TestCase):
    def test_metric_replay_detects_tampered_alarm_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            code, result = run(output, plugins=[{'id': 'cusum', 'entrypoint': 'wms.plugins.cards.c_feedback:CusumDefense',
                                                'config': {'spread': .000001, 'threshold': .001}}])
            self.assertEqual(code, 0, result)
            self.assertTrue(audit(output)['consistent'])
            path = output / 'episodes/ep_000/summary.json'
            summary = json.loads(path.read_text())
            summary['plugin_reports']['0:cusum']['alarms'] += 1
            path.write_text(json.dumps(summary))
            self.assertFalse(audit(output)['consistent'])
    def test_unchanged_nan_feedback_is_not_mistaken_for_inplace_mutation(self):
        before = {'embedding': np.array([1., np.nan]), 'missing': float('nan')}
        self.assertTrue(_same(before, copy.deepcopy(before)))
        after = copy.deepcopy(before);after['embedding'][1] = 0.
        self.assertFalse(_same(before, after))
    def test_residual_definition_and_validity_gates(self):
        original = step()
        self.assertAlmostEqual(planned_next_residual(original)['residual'], np.sqrt(2.5))
        self.assertAlmostEqual(planned_next_residual(original, [1., 2.])['residual'], 1.)
        original['action'][0] = .2
        self.assertEqual(planned_next_residual(original)['reason'], 'action_changed')
        self.assertTrue(planned_next_residual(original, include_changed=True)['valid'])
        original['transition']['taken_over'] = True
        self.assertEqual(planned_next_residual(original, include_changed=True)['reason'], 'takeover')
        original = step()
        original['transition']['next_state'] = [float('nan'), 0.]
        self.assertFalse(planned_next_residual(original)['valid'])
        original['transition']['next_state'] = [0.]
        self.assertFalse(planned_next_residual(original)['valid'])
        with self.assertRaises(ValueError):
            planned_next_residual(step(), [1., 2., 3.])

    def test_cusum_cooldown_reset_and_invalid_samples(self):
        state = CusumState(1., 1., .5, 2., 2)
        self.assertFalse(state.update(0, 3.)['detected'])
        self.assertTrue(state.update(1, 3.)['detected'])
        self.assertTrue(state.update(2, 100.)['cooldown'])
        self.assertTrue(state.update(3, 100.)['cooldown'])
        self.assertFalse(state.update(4, 3.)['detected'])
        self.assertEqual(state.update(5, None)['score'], 0)
        self.assertFalse(state.update(6, 3.)['detected'])

    def test_alarm_modes_and_explicit_fallback_validation(self):
        context = ctx()
        config = {'center': 0., 'spread': 1., 'threshold': 1., 'drift': 0., 'takeover': 'fallback',
                  'steps': 2, 'cooldown': 2, 'fallback_action': [0., 0., -1.]}
        plugin = CusumDefense(config)
        plugin.reset({}, context)
        plugin.after_step(step(), context)
        self.assertEqual(context.alarms[0][1]['takeover'], 'fallback')
        self.assertEqual(context.alarms[0][1]['action'], [0., 0., -1.])
        plugin.reset({}, context)
        self.assertEqual(plugin.alarms, 0)
        for action in (None, [0.], [2., 0., 0.]):
            with self.assertRaises(ValueError):
                CusumDefense({**config, 'fallback_action': action}).reset({}, ctx())

    def test_stop_priority_and_pending_fallback_not_consumed(self):
        pending = {'kind': 'stop', 'plugin': 's6', 'reason': 'alarm', 'action': None,
                   'remaining': 1, 'via': 'alarm', 'raised_at': 0}
        owner = SimpleNamespace(pending=pending, takeover={'kind': 'fallback', 'plugin': 's5'})
        self.assertEqual(PluginSet.alarm_takeover(owner)['kind'], 'stop')
        self.assertIsNone(owner.pending)
        owner = SimpleNamespace(pending={**pending, 'kind': 'fallback', 'remaining': 2},
                                takeover={'kind': 'fallback', 'plugin': 's5'})
        self.assertEqual(PluginSet.alarm_takeover(owner)['plugin'], 's5')
        self.assertEqual(owner.pending['remaining'], 2)
        owner.takeover = None
        self.assertEqual(PluginSet.alarm_takeover(owner)['plugin'], 's6')
        self.assertEqual(owner.pending['remaining'], 1)

    def test_actual_s6_stop_overrides_s5_barrier_next_step(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            specs = [{'id': 'barrier', 'entrypoint': 'wms.plugins.cards.c_action:ActionBarrier',
                      'config': {'limit': .1, 'max_delta': .1}},
                     {'id': 'stop', 'entrypoint': 'test_s6_feedback:AtZero', 'config': {}}]
            code, result = run(output, plugins=specs)
            self.assertEqual(code, 0, result)
            rows = [json.loads(l) for l in (output / 'episodes/ep_000/trace.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['alarms'][0]['t'], 0)
            self.assertIsNone(rows[1]['action'])
            self.assertEqual(rows[1]['takeover']['kind'], 'stop')
            self.assertEqual(rows[1]['takeover']['raised_at'], 0)
            audited = audit(output)
            self.assertTrue(audited['consistent'])
            self.assertEqual(audited['episodes'][0]['alarms'][0]['served_control_steps'], 1)

    def test_poison_preserves_real_feedback_and_neutral(self):
        transition = {'obs': {'proprio': [.1, .2, 0.]}, 'next_obs': {'proprio': [.3, .4, 0.]},
                      'action': [1., 1., -1.], 'reward': 1.}
        original = copy.deepcopy(transition)
        plugin = FeedbackPoison({'factor': .5})
        context = ctx();plugin.reset({}, context)
        out = plugin.on_feedback(transition, context)
        self.assertEqual(transition, original)
        np.testing.assert_allclose(out['next_obs']['proprio'], [.2, .3, 0.])
        self.assertEqual(out['reward'], 1.)
        self.assertIs(FeedbackPoison({'factor': 1.}).on_feedback(transition, context), transition)
        with self.assertRaises(ValueError):
            plugin.reset({}, SimpleNamespace(spec={'action_dim': 8}))

    def test_feedback_poison_changes_online_learning_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, result = run(root / 'clean', model='toy-planner-online', max_steps=30)
            self.assertEqual(code, 0, result)
            code, result = run(root / 'poison', model='toy-planner-online', max_steps=30, plugins=[
                {'id': 'poison', 'entrypoint': 'wms.plugins.cards.c_feedback:FeedbackPoison', 'config': {'factor': .3}}])
            self.assertEqual(code, 0, result)
            def gain(path):
                row = json.loads((path / 'episodes/ep_000/steps.jsonl').read_text().splitlines()[-1])
                return row['policy_info']['move_gain'][0]
            self.assertLess(gain(root / 'poison'), gain(root / 'clean'))
            self.assertLess(gain(root / 'poison'), .03)


if __name__ == '__main__':
    unittest.main()
