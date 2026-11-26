"""Submit/resume S3/S4 matrices through the same API as the Web workstation.

./dev.sh python scripts/b_s34_matrix.py --phase smoke --wait
./dev.sh python scripts/b_s34_matrix.py --phase formal --wait
No installs, training, shared model writes or changes to other services.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATS = 'native/plugins/stats/tdmpc2-pushcube-independent.json'


def use(key, plugin, **config):
    return {'key': key, 'plugin': plugin, 'config': config}


def matrices(phase, episodes=None):
    toy = phase == 'toy'
    data = json.loads((ROOT / STATS).read_text())
    std = {} if toy else {'stats': STATS}
    norm = data['action']['norm']
    watch = use('residual', 'b-residual-monitor', scale=.2478402774542645, threshold=.8173528137616873)
    score = use('score_c1', 'b-score-misspec', c=1)
    defense = use('rerank_p90', 'b-constrained-rerank', v_max=norm['p90'])
    noise = use('noise_both_c03', 'b-prediction-noise', c=.3, source='both', **std)
    obs = lambda c: use('obs_c' + str(c), 'a-obs-noise-vector', c=c, start=20, end=50, **std)
    if phase in ('smoke', 'toy'):
        groups = [('10-smoke', [score]), ('11-smoke', [score, defense]),
                  ('8-smoke', [noise]), ('9-smoke', [obs(.1), watch])]
    elif phase == 'neutral':
        groups = [('neutral', [use('score0', 'b-score-misspec', c=0),
                              use('noise0', 'b-prediction-noise', c=0, **std),
                              use('obs0', 'a-obs-noise-vector', c=0, **std),
                              use('rerank0', 'b-constrained-rerank', v_max=None)]),
                  ('monitor-neutral', [watch])]
    elif phase == 'scale':
        groups = [('10-scale', [use('scale' + str(s), 'b-score-misspec', mode='scale', scale=s) for s in (.5, 2)])]
    else:
        groups = [('10-dose', [use('score_c' + str(c), 'b-score-misspec', c=c) for c in (.5, 1, 2)]),
                  ('11-p90', [score, defense]),
                  ('11-p99', [score, use('rerank_p99', 'b-constrained-rerank', v_max=norm['p99'])]),
                  ('9-detection', [use('goal_move', 'target-moves', moves=[{'step': 20, 'goal': {'delta': [.1, 0, 0]}}]),
                                   *[obs(c) for c in (.1, .3, 1)], watch]),
                  ('8-dose', [use(f'noise_{source}_{c}', 'b-prediction-noise', c=c, source=source, **std)
                              for source in ('prior', 'value', 'both') for c in (.1, .3, 1)]),
                  ('g3-combination', [noise, defense])]
    n = episodes or (2 if phase in ('smoke', 'toy', 'scale') else 50)
    return [(key, {'name': 'S34-' + phase + '-' + key,
                   'description': 'S3/S4 isolated B acceptance; paired seeds; frozen fast_speed=1.02 for TD-MPC2.',
                   'model': 'toy-planner' if toy else 'tdmpc2-pushcube',
                   'environments': ['toy-pick-vector' if toy else 'maniskill-pushcube-state-independent'],
                   'plugins': plugins, 'seeds': {'start': 2000, 'episodes': n}, 'max_steps': 50,
                   'control': {'single': True, 'combined': False, 'defended': True,
                               'defense_only': True, 'monitors_everywhere': True}}) for key, plugins in groups]


def api(path, data=None):
    url = os.environ.get('WMS_BASE_URL', 'http://127.0.0.1:8782') + '/api/v1' + path
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=60) as response:
        return json.load(response)


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', choices=['toy', 'smoke', 'neutral', 'scale', 'formal'], required=True)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--only', help='one case key, e.g. 10-dose')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--output', default=str(ROOT / '.workspace/s34'))
    parser.add_argument('--export', action='store_true', help='write portable configs only; do not submit')
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    if args.export:
        for key,config in matrices(args.phase,args.episodes):
            if not args.only or key == args.only:
                save(folder / (args.phase+'-'+key+'.json'),config)
        return
    # Prevent two local submitters from both passing the resume check.
    import fcntl
    with (folder / 'matrix.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = folder / ('matrix-' + args.phase + '.json')
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'phase': args.phase, 'cases': {}}
        if args.phase == 'formal':
            provenance = json.loads((folder/'provenance.json').read_text())
            gate = json.loads((folder/'gate-neutral.json').read_text())
            if not gate.get('passed') or not provenance.get('stats_reproduced'):
                raise ValueError('formal submission requires neutral and provenance gates')
            for path,expected in provenance['source_sha256'].items():
                if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
                    raise ValueError('plugin source changed since the gates: '+path)
            manifest.setdefault('provenance',provenance)
        for key, config in matrices(args.phase, args.episodes):
            if args.only and key != args.only:
                continue
            digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            config['name'] += '-' + digest[:8]
            validation = api('/experiments/validate', config)
            if not validation['valid']:
                raise RuntimeError(validation['issues'])
            save(folder / (args.phase + '-' + key + '-config.json'), config)
            record = manifest['cases'].get(key)
            if record and record['config_sha256'] != digest:
                raise ValueError('Resume configuration changed: use a new --output directory')
            if record:
                experiment = api('/experiments/' + record['experiment_id'])
            else:
                # Recover when HTTP submission succeeded but the client lost its response.
                previous = [e for e in api('/experiments')['items'] if e['name'] == config['name']]
                if len(previous) > 1:
                    raise RuntimeError('Ambiguous duplicate experiment name; inspect before resuming')
                experiment = previous[0] if previous else api('/experiments', config)
                record = {'config_sha256': digest, 'experiment_id': experiment['id'], 'config': config}
                manifest['cases'][key] = record
            record['runs'] = experiment['runs']
            save(manifest_path, manifest)
            last = None
            while True:
                status = (experiment['state'], experiment['episodes_done'], experiment['episodes_total'])
                if status != last:
                    print(key, experiment['id'], *status, flush=True)
                    last = status
                if experiment['state'] not in ('queued', 'running', 'cancelling') or not args.wait:
                    break
                time.sleep(10)
                experiment = api('/experiments/' + experiment['id'])
            record['state'], record['runs'] = experiment['state'], experiment['runs']
            save(manifest_path, manifest)
            if experiment['state'] in ('failed', 'partial_failed', 'cancelled'):
                raise RuntimeError('Experiment did not complete; inspect logs, do not silently drop or retry seeds')


if __name__ == '__main__':
    main()
