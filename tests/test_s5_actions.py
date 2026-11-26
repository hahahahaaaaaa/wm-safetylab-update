"""Command faults, reset boundaries and barrier takeover semantics (CPU only)."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wms.plugins.cards.c_action import ActionNoise, ActionDelay, ActionBarrier
from wms.episodes.s5_metrics import recompute


def context():
    events, takeovers = [], []
    return SimpleNamespace(t=0, rng=np.random.default_rng(42),
                           spec={"action_low": [-1.] * 3, "action_high": [1.] * 3},
                           log=lambda **kw: events.append(kw), events=events,
                           intervene=lambda kind, **kw: takeovers.append((kind, kw)), takeovers=takeovers)


class S5ActionsTest(unittest.TestCase):
    def test_noise_clipping_is_explicit_and_keeps_raw_evidence(self):
        ctx = context()
        ctx.rng = SimpleNamespace(normal=lambda *args: np.array([.5, -.5]))
        p = ActionNoise({'std': .1, 'clip_to_bounds': True})
        self.assertEqual(p.on_action([1., -1., 1.], ctx), [1., -1., 1.])
        self.assertEqual(ctx.events[-1]['raw_noisy_action'], [1.5, -1.5, 1.])
        self.assertEqual(ctx.events[-1]['clipped_dimensions'], 2)
        with self.assertRaises(ValueError):
            ActionNoise({'clip_to_bounds': 'false'})
    def test_combined_episode_executes_faults_before_barrier(self):
        from wms.plugins.resolve import resolve_specs
        from wms.core.registry import Registry
        with tempfile.TemporaryDirectory() as directory:
            specs = resolve_specs([
                {'plugin': 's5-action-barrier', 'config': {'limit': .3, 'max_delta': .15}},
                {'plugin': 's5-action-noise', 'config': {'std': .4}},
                {'plugin': 's5-action-delay', 'config': {'steps': 2}},
            ], Registry().items['plugins'])
            from wms.plugins.conformance import TOY_MODELS, TOY_ENVS
            from wms.runners.unified import UnifiedWorker
            output = Path(directory) / 'combined'
            config = {'task': 'native', 'entrypoint': 'wms.episodes.kernel:run', 'seed': 42,
                      'model': TOY_MODELS['toy-planner'], 'environment': TOY_ENVS['toy-pick'],
                      'params': {'episodes': 2, 'max_steps': 12, 'video': False,
                                 'plugin_strict': True, 'plugins': specs}}
            self.assertEqual(UnifiedWorker(config, output).run(), 0)
            for path in sorted((output / 'episodes').glob('*/trace.jsonl')):
                previous = None
                for row in map(json.loads, path.read_text(encoding='utf8').splitlines()):
                    if row['action'] is None:
                        continue
                    action = np.asarray(row['action'])
                    self.assertTrue(np.all(np.abs(action[:2]) <= .300001))
                    if previous is not None:
                        self.assertTrue(np.all(np.abs(action[:2] - previous[:2]) <= .150001))
                    self.assertEqual(action[2], row['action_proposed'][2])
                    previous = action
            audited = recompute(output)
            self.assertEqual(len(audited['episodes']), 2)
            self.assertTrue(audited['summary_consistent'])
            for ep in audited['episodes']:
                self.assertEqual(len(ep['takeover_records']), ep['takeover_steps'])
                self.assertTrue(all(record['reason'] and record['reason_codes'] for record in ep['takeover_records']))

    def test_metrics_distinguish_contiguous_takeover_stop_and_unmeasured_hazards(self):
        with tempfile.TemporaryDirectory() as directory:
            ep = Path(directory) / 'episodes' / 'ep_000'
            ep.mkdir(parents=True)
            rows = [dict(t=0, action=[0], takeover={'plugin': 'shield', 'kind': 'fallback'},
                         hazards=[{'id': 'drop'}, {'id': 'drop'}], env_events=[]),
                    dict(t=1, action=[0], takeover={'plugin': 'shield', 'kind': 'fallback'}, hazards=[], env_events=['success']),
                    dict(t=2, action=None, takeover={'plugin': 'shield', 'kind': 'stop'}, hazards=[], env_events=[])]
            (ep / 'trace.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf8')
            (ep / 'summary.json').write_text(json.dumps({'hazard_steps': {'drop': 1}}), encoding='utf8')
            result = recompute(directory)
            self.assertEqual((result['takeover_steps'], result['takeover_events']), (2, 2))
            self.assertEqual(result['success_rate'], 1)
            self.assertEqual(result['episodes'][0]['hazard_steps'], {'drop': 1})
            self.assertEqual(len(result['episodes'][0]['takeover_event_groups']), 2)
            self.assertEqual(result['episodes'][0]['takeover_event_groups'][0]['steps'], 2)
            self.assertFalse(result['summary_consistent'])  # fixture intentionally lacks summary step/success fields
            (ep / 'summary.json').write_text('{}', encoding='utf8')
            self.assertIsNone(recompute(directory)['hazard_episode_rate'])

    def test_noise_seed_dimensions_window_and_identity(self):
        plugin = ActionNoise({"std": .2, "start": 1, "end": 3})
        a, b = context(), context()
        action = np.array([.5, .2, -1], dtype=np.float32)
        self.assertIs(plugin.on_action(action, a), action)
        a.t = b.t = 1
        first, second = plugin.on_action(action, a), plugin.on_action(action, b)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, action.dtype)
        self.assertEqual(first[2], -1)
        self.assertNotEqual(float(first[0]), float(action[0]))
        np.testing.assert_array_equal(action, [.5, np.float32(.2), -1])
        a.t = 3
        self.assertIs(plugin.on_action(action, a), action)
        self.assertIs(ActionNoise({"std": 0}).on_action(action, a), action)

    def test_delay_exact_queue_reset_and_unselected_dimension(self):
        p, ctx = ActionDelay({"steps": 2}), context()
        p.reset({}, ctx)
        outputs = []
        for t in range(5):
            ctx.t = t
            outputs.append(p.on_action([float(t), float(t), float(-t)], ctx))
        self.assertEqual([v[0] for v in outputs], [0., 0., 0., 1., 2.])
        self.assertEqual([v[2] for v in outputs], [0., -1., -2., -3., -4.])
        self.assertEqual(ctx.events[-1]['source_t'], 2)
        p.reset({}, ctx)
        self.assertEqual(p.on_action([9., 9., 1.], ctx), [9., 9., 1.])
        action = [1., 2., 3.]
        zero = ActionDelay({"steps": 0})
        zero.reset({}, ctx)
        self.assertIs(zero.on_action(action, ctx), action)

    def test_barrier_uses_actual_executed_action_and_counts_changes(self):
        p, ctx = ActionBarrier({"limit": .5, "max_delta": .2}), context()
        p.reset({}, ctx)
        self.assertEqual(p.on_action([2., -.8, 1.], ctx), [.5, -.5, 1.])
        self.assertEqual(len(ctx.takeovers), 1)
        p.after_step({"action": [0., 0., -1.]}, ctx)
        np.testing.assert_allclose(p.on_action([.5, -.5, 1.], ctx), [.2, -.2, 1.])
        p.reset({}, ctx)
        self.assertEqual(p.on_action([.4, -.4, 1.], ctx), [.4, -.4, 1.])
        self.assertEqual(p.end_episode(ctx)['modified_steps'], 0)

    def test_invalid_configuration_and_dimension_are_rejected(self):
        for cls, config in [(ActionNoise, {'std': -1}), (ActionNoise, {'std': float('nan')}),
                            (ActionDelay, {'steps': 1.5}), (ActionDelay, {'steps': True}),
                            (ActionBarrier, {'limit': 0}), (ActionBarrier, {'max_delta': -1}),
                            (ActionNoise, {'start': 3, 'end': 2})]:
            with self.subTest(cls=cls, config=config), self.assertRaises(ValueError):
                cls(config)
        with self.assertRaises(ValueError):
            ActionNoise({'dims': 4}).on_action([0., 0., 0.], context())


if __name__ == '__main__':
    unittest.main()
