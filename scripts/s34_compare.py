"""Strict seed-matched comparisons; --neutral requires exact full trajectories."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wms.redteam.stats import paired_bootstrap


def _output(runs_dir, run):
    path = Path(run)
    return path if (path / 'episodes.json').exists() else Path(runs_dir) / run / 'output'


def _steps(output, episode):
    path = output / 'episodes' / f'ep_{episode:03d}' / 'steps.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_difference(a, b):
    if len(a) != len(b) or not a:
        raise ValueError('paired samples must be nonempty and have equal length')
    mean, low, high = paired_bootstrap([float(y)-float(x) for x,y in zip(a,b)])
    return {'n': len(a), 'mean': mean, 'ci95': [low, high]}


def load(output):
    result = json.loads((output / 'result.json').read_text())
    if result.get('status') != 'completed':
        raise ValueError(f'{output}: run status is not completed')
    rows = json.loads((output / 'episodes.json').read_text())['episodes']
    manifest = json.loads((output / 'run_manifest.json').read_text())
    expected = manifest['seeds']
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('manifest seeds are missing/duplicated')
    indexed, episode_ids = {}, set()
    for row in rows:
        seed = row['seed']
        if seed in indexed or row['episode'] in episode_ids:
            raise ValueError('duplicate seed or episode index')
        if not isinstance(row.get('success'), bool):
            raise ValueError('missing boolean task outcome')
        steps = _steps(output, row['episode'])
        if len(steps) != row['steps'] or [s['t'] for s in steps] != list(range(len(steps))):
            raise ValueError('missing, duplicate or unordered trajectory steps')
        for step in steps:
            action = step.get('action')
            if action is not None and (not isinstance(action,list) or not action or
                    not all(isinstance(x,(int,float)) and math.isfinite(x) for x in action)):
                raise ValueError('invalid/nonfinite action')
        indexed[seed] = (row, steps)
        episode_ids.add(row['episode'])
    if set(indexed) != set(expected) or len(rows) != len(expected):
        raise ValueError('incomplete expected seed coverage')
    return manifest, indexed


def compare(runs_dir, a, b, neutral=False, expected_seeds=None):
    ma, ia = load(_output(runs_dir,a))
    mb, ib = load(_output(runs_dir,b))
    for key in ('model','environment','packages','inputs'):
        if ma.get(key) != mb.get(key):
            raise ValueError(f'incompatible {key}')
    for key in ('max_steps','reset_order','stop_on_terminate'):
        if ma['params'].get(key) != mb['params'].get(key):
            raise ValueError(f'incompatible {key}')
    if ia.keys() != ib.keys() or (expected_seeds is not None and set(ia) != set(expected_seeds)):
        raise ValueError('seed coverage differs')
    worst, lengths, outcomes, shapes, hazard_mismatch = 0., [], [], [], []
    hazards = {'a': {}, 'b': {}}
    for seed in sorted(ia):
        ra,sa = ia[seed]
        rb,sb = ib[seed]
        if len(sa) != len(sb):
            lengths.append(seed)
        for x,y in zip(sa,sb):
            xa,ya = x.get('action'),y.get('action')
            if xa is None or ya is None:
                if xa != ya:
                    shapes.append(seed)
            elif len(xa) != len(ya):
                shapes.append(seed)
            else:
                worst = max(worst,max(abs(p-q) for p,q in zip(xa,ya)))
        if ra['success'] != rb['success']:
            outcomes.append(seed)
        if ra.get('hazard_steps',{}) != rb.get('hazard_steps',{}):
            hazard_mismatch.append(seed)
        for label,row in (('a',ra),('b',rb)):
            for key,value in row.get('hazard_steps',{}).items():
                hazards[label][key] = hazards[label].get(key,0) + value
    exact = not lengths and not outcomes and not shapes and not hazard_mismatch and worst == 0
    if neutral and not exact:
        raise ValueError(f'neutral gate failed: lengths={lengths}, outcomes={outcomes}, shapes={shapes}, hazards={hazard_mismatch}, diff={worst}')
    seeds = sorted(ia)
    return {'a':str(a),'b':str(b),'episodes':[len(ia),len(ib)],'seed_mismatch':[],
            'full_trajectory_equal':exact,'max_abs_action_diff':None if lengths or shapes else worst,
            'overlap_max_abs_action_diff':worst,'length_mismatch':lengths,'action_shape_mismatch':shapes,
            'outcome_mismatch':outcomes,'success':[sum(ia[s][0]['success'] for s in seeds),sum(ib[s][0]['success'] for s in seeds)],
            'hazard_mismatch':hazard_mismatch,
            'success_difference':paired_difference([ia[s][0]['success'] for s in seeds],[ib[s][0]['success'] for s in seeds]),
            'hazard_steps':hazards,'executed_steps':[sum(len(ia[s][1]) for s in seeds),sum(len(ib[s][1]) for s in seeds)]}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('runs',nargs='+')
    parser.add_argument('--runs-dir',default='runs')
    parser.add_argument('--neutral',action='store_true')
    parser.add_argument('--seed-start',type=int)
    parser.add_argument('--episodes',type=int)
    args = parser.parse_args()
    if len(args.runs) % 2:
        parser.error('give runs in pairs')
    expected = range(args.seed_start,args.seed_start+args.episodes) if args.seed_start is not None and args.episodes else None
    for a,b in zip(args.runs[::2],args.runs[1::2]):
        print(json.dumps(compare(args.runs_dir,a,b,args.neutral,expected),ensure_ascii=False))


if __name__ == '__main__':
    main()
