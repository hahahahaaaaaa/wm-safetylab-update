"""Calibrate S6 on clean seeds, evaluate held-out closed loops, preserve all evidence."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np

from s5_acceptance import ROOT, STORE, api, save, wait
from wms.episodes.s6_metrics import audit
from wms.scenarios.toy.env import ToyPickEnv


class ShiftedToyEnv(ToyPickEnv):
    """Acceptance fixture: hidden actuator-gain change, not a production plugin."""
    def __init__(self, config):
        super().__init__(config)
        self.nominal_step = self.step_size
        self.shift_start = int(config.get('shift_start', 8))
        self.factor = float(config.get('shift_factor', .35))

    def reset(self, seed, inputs=None):
        self.step_size = self.nominal_step
        return super().reset(seed, inputs)

    def step(self, action):
        changed = self.t == self.shift_start
        if changed:
            self.step_size = self.nominal_step * self.factor
        transition = super().step(action)
        if changed:
            transition['events'].append('acceptance_dynamics_shift')
        return transition


def run_case(recipe, dest, label, plugins, seed, episodes, anomaly_start=None, shifted=False, model=None):
    dest = dest / label
    config = copy.deepcopy(recipe)
    config['name'] = 's6-acceptance-' + label
    config['metrics'] = []
    config['seed'] = seed
    config['timeout_s'] = 1800
    config['params'].update({'episodes': episodes, 'seeds': list(range(seed, seed + episodes)),
                            'plugins': plugins, 'perturbations': [], 'target_moves': [],
                            'save_raw': False, 'video': False, 'plugin_strict': True})
    if model:
        config['model'] = api('/components')['models'][model]
    if shifted:
        config['environment']['entrypoint'] = 's6_acceptance:ShiftedToyEnv'
        config['environment']['config'].update({'shift_start': anomaly_start, 'shift_factor': .35})
        config['runtime']['python_paths'] = [str(ROOT / 'scripts')]
    submitted = api('/runs', config)
    save(dest / 'submitted.json', submitted)
    completed = wait('/runs/' + submitted['id'])
    result = audit(STORE / submitted['id'] / 'output', anomaly_start)
    save(dest / 'audit.json', result)
    if not result['consistent']:
        raise RuntimeError({'run': submitted['id'], 'errors': result['errors'],
                            'summary': result['s5_metrics']['summary_consistent']})
    entry = {'label': label, 'run_id': submitted['id'], 'config': completed['config'],
             'success_rate': result['s5_metrics']['success_rate'],
             'mean_steps': result['s5_metrics']['mean_executed_steps'],
             'takeover_steps': result['s5_metrics']['takeover_steps'],
             'alarm_episode_rate': result['alarm_episode_rate'],
             'pre_anomaly_alarm_episode_rate': result['pre_anomaly_alarm_episode_rate'],
             'detection_rate': result['detection_rate'], 'mean_detection_delay': result['mean_detection_delay']}
    save(dest / 'summary.json', entry)
    print(label, entry['run_id'], 'success', entry['success_rate'], 'alarm', entry['alarm_episode_rate'],
          'detection', entry['detection_rate'], flush=True)
    return entry


def calibrate(run_id, dest):
    episodes, all_errors = [], []
    for path in sorted((STORE / run_id / 'output/episodes').glob('*/trace.jsonl')):
        values = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            samples = [e['data']['residual_sample'] for e in row['events'] if 'residual_sample' in e['data']]
            sample = next((s for s in samples if s['valid']), None)
            error = np.asarray(sample['predicted']) - np.asarray(sample['observed']) if sample else None
            values.append(error)
        episodes.append(values)
        all_errors.extend(v for v in values if v is not None)
    if len(all_errors) < 20:
        raise ValueError('insufficient eligible clean calibration samples')
    # Calibrate representation dimensions separately: a discontinuous holding bit
    # must not drown out predictable position coordinates. Use clean data only.
    scale = np.maximum(np.sqrt(np.mean(np.square(all_errors), axis=0)), .001)
    episodes = [[float(np.sqrt(np.mean((e / scale) ** 2))) if e is not None else None for e in errors]
                for errors in episodes]
    all_values = [v for values in episodes for v in values if v is not None]
    center, spread = float(np.mean(all_values)), max(float(np.std(all_values)), 1e-6)
    peak = 0.
    for values in episodes:
        score = 0.
        for value in values:
            score = max(0., score + (value - center) / spread - .5) if value is not None else 0.
            peak = max(peak, score)
    result = {'scale': scale.tolist(), 'center': center, 'spread': spread, 'drift': .5,
              'threshold': max(1., peak * 1.2 + .5), 'cooldown': 5,
              'include_changed': False, 'takeover': 'none', 'steps': 2}
    save(dest / 'calibration.json', {'config': result, 'source_run': run_id,
                                    'eligible_samples': len(all_values), 'max_clean_cusum': peak,
                                    'scale_method': 'clean-feature-rms/v1', 'feature_scale_floor': .001})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['cpu', 'gpu'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dest = args.output / args.phase
    dest.mkdir(parents=True, exist_ok=True)
    recipe = api('/recipes/' + ('toy-episode-occlusion' if args.phase == 'cpu' else 'tdmpc2-pushcube-episode'))
    monitor = {'plugin': 's6-residual-monitor'}
    results = []
    if args.phase == 'cpu':
        recipe['model'] = api('/components')['models']['toy-planner']
        recipe['params']['max_steps'] = 60
        seed = 700
        clean = run_case(recipe, dest, 'cpu-calibration', [monitor], 600, 20)
    else:
        free, utilization = map(int, subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')[:2])
        if free < 8000 or utilization > 10:
            raise RuntimeError('GPU busy; do not interfere with existing jobs')
        recipe['environment'] = api('/components')['environments']['maniskill-pushcube-state-independent']
        checkpoint = json.loads(Path(recipe['runtime']['config_file']).read_text())['episode_checkpoint']
        for component in ('model', 'environment'):
            recipe[component]['config']['checkpoint'] = checkpoint
        recipe['params']['max_steps'] = 50
        seed = 1500
        clean = run_case(recipe, dest, 'gpu-calibration', [monitor], 1400, 20)
    params = calibrate(clean['run_id'], dest)
    alarm = {'plugin': 's6-cusum', 'config': params}
    results.append(run_case(recipe, dest, args.phase + '-clean-alarm', [alarm], seed, 20))
    if args.phase == 'cpu':
        for label, mode in [('shift-alarm', 'none'), ('shift-stop', 'stop'), ('shift-fallback', 'fallback')]:
            config = {**params, 'takeover': mode}
            if mode == 'fallback':
                config['fallback_action'] = [0., 0., -1.]  # explicit toy test action, not a certified safe controller
            results.append(run_case(recipe, dest, label, [{'plugin': 's6-cusum', 'config': config}], seed, 20,
                                    anomaly_start=8, shifted=True))
        # Toy-only learning corruption. Keep raw learning gains separately from attack logs.
        for label, plugins in [('online-clean', []), ('online-poison', [{'plugin': 's6-feedback-poison', 'config': {'factor': .3}}]),
                               ('online-neutral', [{'plugin': 's6-feedback-poison', 'config': {'factor': 1.}}])]:
            result = run_case(recipe, dest, label, plugins, seed, 20, model='toy-planner-online')
            gains = []
            for path in sorted((STORE / result['run_id'] / 'output/episodes').glob('*/steps.jsonl')):
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                gains.append(rows[-1]['policy_info']['move_gain'])
            result['last_logged_gain_mean'] = np.mean(gains, axis=0).tolist()
            results.append(result)
    else:
        noise = {'plugin': 's5-action-noise', 'config': {'std': .4, 'dims': 7, 'start': 10, 'end': 40, 'clip_to_bounds': True}}
        results.append(run_case(recipe, dest, 'gpu-noise-only', [noise], seed, 20))
        for label, mode in [('gpu-noise-alarm', 'none'), ('gpu-noise-stop', 'stop')]:
            detector = {'plugin': 's6-cusum', 'config': {**params, 'include_changed': True, 'takeover': mode}}
            results.append(run_case(recipe, dest, label, [noise, detector], seed, 20, anomaly_start=10))
    save(dest / 'summary.json', {'calibration': clean, 'parameters': params, 'evaluation': results})


if __name__ == '__main__':
    main()
