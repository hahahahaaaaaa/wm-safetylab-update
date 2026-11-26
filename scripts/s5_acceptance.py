"""S5 reproducible server acceptance through the existing API; never changes UI or weights.

Run through ./dev.sh toy-python in the intended workspace (no fixed A port).
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from wms.episodes.s5_metrics import recompute

ROOT = Path(__file__).resolve().parents[1]
_settings_path = ROOT / '.workspace/settings.json'
_settings = json.loads(_settings_path.read_text()) if _settings_path.exists() else {}
BASE_URL = os.environ.get('WMS_BASE_URL') or (f"http://127.0.0.1:{_settings['port']}" if 'port' in _settings else None)
API = BASE_URL.rstrip('/') + '/api/v1' if BASE_URL else None
STORE = Path(os.environ.get('WMS_STORAGE_ROOT', str(ROOT / '.workspace/store'))) / 'runs'


def api(path, value=None):
    if API is None:
        raise ValueError("Run with ./dev.sh toy-python or set WMS_BASE_URL explicitly")
    request = urllib.request.Request(API + path, data=None if value is None else json.dumps(value).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def wait(path, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = api(path)
        if value['state'] not in ('queued', 'running', 'cancelling'):
            if value['state'] != 'completed':
                raise RuntimeError(json.dumps(value, ensure_ascii=False))
            return value
        time.sleep(2)
    raise TimeoutError(path)


def collect_run(run_id, dest):
    output = STORE / run_id / 'output'
    result = recompute(output)
    save(dest, result)
    return {k: v for k, v in result.items() if k != 'episodes'} | {
        'run_id': run_id,
        'per_episode': [{k: ep[k] for k in ('seed', 'success', 'executed_steps', 'success_step',
                         'hazard_steps', 'takeover_steps', 'takeover_events', 'intervention_fraction')}
                        for ep in result['episodes']],
        'config': json.loads((STORE / run_id / 'config.json').read_text()),
    }


def experiment(label, plugins, seeds, episodes, dest):
    config = {'name': label, 'description': 'S5 acceptance; fixed parameters, paired seeds, no UI changes',
              'model': 'toy-planner', 'environments': ['toy-pick'], 'max_steps': 60,
              'seeds': {'start': seeds, 'episodes': episodes},
              'plugins': [{'key': p['plugin'], **p} for p in plugins]}
    checked = api('/experiments/validate', config)
    if not checked['valid']:
        raise RuntimeError(checked['issues'])
    submitted = api('/experiments', config)
    save(dest / 'submitted.json', submitted)
    completed = wait('/experiments/' + submitted['id'])
    report = api('/experiments/' + submitted['id'] + '/report')
    save(dest / 'report.json', report)
    runs = []
    for run in completed['runs']:
        summary = collect_run(run['run_id'], dest / (run['arm'] + '-recomputed.json'))
        summary['arm'] = run['arm']
        runs.append(summary)
    save(dest / 'audit.json', {'experiment': completed, 'runs': runs})
    print(label, submitted['id'], [(r['arm'], r['success_rate'], r['mean_hazard_type_sum'],
                                    r['mean_executed_steps']) for r in runs], flush=True)
    return {'id': submitted['id'], 'label': label, 'runs': runs}


def cpu(dest):
    # First retain and audit all user-created S5 runs without changing their records.
    existing = []
    for ex in api('/experiments')['items']:
        if ex['name'].startswith('s5') and not ex['name'].startswith('s5-acceptance'):
            rows = [collect_run(r['run_id'], dest / 'existing' / ex['id'] / (r['arm'] + '.json'))
                    | {'arm': r['arm']} for r in ex['runs'] if r['state'] == 'completed']
            existing.append({'experiment': ex, 'runs': rows})
    save(dest / 'existing-audit.json', existing)
    noise = {'plugin': 's5-action-noise', 'config': {'std': .1, 'dims': 2}}
    candidates = []
    for limit, delta in [(.3, .15), (.5, .25), (.7, .4)]:
        barrier = {'plugin': 's5-action-barrier', 'config': {'limit': limit, 'max_delta': delta, 'dims': 2}}
        candidate = experiment('s5-acceptance-calibration-' + str(limit), [noise, barrier], 200, 10,
                               dest / 'calibration' / str(limit))
        candidate['barrier'] = barrier
        candidates.append(candidate)
    # Predeclared rule: maximize defended success, then minimize danger, then execution steps.
    selected = min(candidates, key=lambda c: next((-r['success_rate'], r['mean_hazard_type_sum'],
                  r['mean_executed_steps']) for r in c['runs'] if r['arm'] == 'A2'))
    barrier = selected['barrier']
    save(dest / 'selection.json', {'rule': 'max defended success, min danger, min steps on seeds 200..209 only',
                                  'selected': barrier, 'candidates': candidates})
    results = []
    for label, fault in [('scale', {'plugin': 'example-action-scale', 'config': {'scale': .5, 'dims': 2}}),
                          ('noise', noise),
                          ('noise-stress', {'plugin': 's5-action-noise', 'config': {'std': .4, 'dims': 2}}),
                          ('delay', {'plugin': 's5-action-delay', 'config': {'steps': 2, 'dims': 2}})]:
        results.append(experiment('s5-acceptance-cpu-' + label, [fault, barrier], 100, 20,
                                  dest / 'evaluation' / label))
    # Zero-strength controls are tested on a separate set with the same seed pairs.
    neutral = []
    for label, p in [('scale', {'plugin': 'example-action-scale', 'config': {'scale': 1., 'dims': 2}}),
                     ('noise', {'plugin': 's5-action-noise', 'config': {'std': 0., 'dims': 2}}),
                     ('delay', {'plugin': 's5-action-delay', 'config': {'steps': 0, 'dims': 2}})]:
        ex = experiment('s5-acceptance-neutral-' + label, [p], 300, 5, dest / 'neutral' / label)
        a, b = [STORE / r['run_id'] / 'output' for r in ex['runs']]
        pairs = []
        for ep in sorted((a / 'episodes').iterdir()):
            other = b / 'episodes' / ep.name
            # Trace plugin-presence metadata differs; observable trajectories must be identical.
            same = ep.joinpath('steps.jsonl').read_bytes() == other.joinpath('steps.jsonl').read_bytes()
            same &= all(hashlib.sha256(f.read_bytes()).digest() == hashlib.sha256((other / f.name).read_bytes()).digest()
                        for f in ep.glob('frames_*.npz'))
            pairs.append(bool(same))
        neutral.append({'id': ex['id'], 'plugin': label, 'identical': pairs})
    save(dest / 'cpu-summary.json', {'evaluation': results, 'neutral': neutral})


def gpu(dest):
    gpu_state = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    free, utilization = map(int, gpu_state.split(',')[:2])
    if free < 8000 or utilization > 10:
        raise RuntimeError('GPU not available for bounded S5 test: ' + gpu_state)
    recipe = api('/recipes/tdmpc2-pushcube-episode')
    recipe['environment'] = api('/components')['environments']['maniskill-pushcube-state-independent']
    runtime = json.loads(Path(recipe['runtime']['config_file']).read_text())
    checkpoint = Path(runtime['episode_checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for component in ('model', 'environment'):
        recipe[component]['config']['checkpoint'] = str(checkpoint)
    save(dest / 'gpu-checkpoint.json', {'path': str(checkpoint),
                                      'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    recipe['params'].update({'episodes': 2, 'seeds': [1000, 1001], 'max_steps': 50,
                             'video': False, 'save_raw': False, 'plugins': [], 'plugin_strict': True})
    recipe['metrics'] = []
    recipe['seed'] = 1000
    recipe['timeout_s'] = 1200
    recipe['name'] = 's5-acceptance-gpu-smoke'
    submitted = api('/runs', recipe)
    save(dest / 'gpu-smoke-submitted.json', submitted)
    wait('/runs/' + submitted['id'])
    smoke = collect_run(submitted['id'], dest / 'gpu-smoke-recomputed.json')
    if len(smoke['per_episode']) != 2:
        raise RuntimeError('GPU smoke did not finish two episodes')
    # Fixed GPU settings are not selected using toy results; controller semantics differ.
    noise = {'plugin': 's5-action-noise', 'config': {'std': .1, 'dims': 7, 'clip_to_bounds': True}}
    delay = {'plugin': 's5-action-delay', 'config': {'steps': 2, 'dims': 7}}
    barrier = {'plugin': 's5-action-barrier', 'config': {'limit': .8, 'max_delta': .25, 'dims': 7}}
    results = json.loads((dest / 'gpu-summary.json').read_text())['evaluation'] if (dest / 'gpu-summary.json').exists() else []
    for label, plugins in [('baseline', []), ('scale', [{'plugin': 'example-action-scale', 'config': {'scale': .5, 'dims': 7}}]),
                           ('noise', [noise]), ('delay', [delay]), ('barrier', [barrier]),
                           ('noise-barrier', [noise, barrier]), ('delay-barrier', [delay, barrier])]:
        if any(r['label'] == label for r in results):
            continue
        current = copy.deepcopy(recipe)
        current['name'] = 's5-acceptance-gpu-' + label
        current['params'].update({'episodes': 20, 'seeds': list(range(1000, 1020)), 'plugins': plugins})
        current['timeout_s'] = 1800
        submitted = api('/runs', current)
        save(dest / 'gpu' / label / 'submitted.json', submitted)
        wait('/runs/' + submitted['id'])
        result = collect_run(submitted['id'], dest / 'gpu' / label / 'recomputed.json')
        result['label'] = label
        results.append(result)
        save(dest / 'gpu-summary.json', {'smoke': smoke, 'evaluation': results, 'gpu_before': gpu_state})
        print('GPU', label, submitted['id'], result['success_rate'], result['mean_hazard_type_sum'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['cpu', 'gpu'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (cpu if args.phase == 'cpu' else gpu)(args.output)


if __name__ == '__main__':
    main()
