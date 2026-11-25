"""Independent S5 action/record checks for saved episode-kernel run directories."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np


def audit_run(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    specs = config.get('params', {}).get('plugins', [])
    specs = [(i, p) for i, p in enumerate(specs)
             if p['entrypoint'].endswith((':ActionScale', ':ActionNoise', ':ActionDelay', ':ActionBarrier'))]
    specs.sort(key=lambda pair: pair[1]['entrypoint'].endswith(':ActionBarrier'))
    errors, checked = [], 0
    for ep in sorted((run_dir / 'output/episodes').glob('ep_*')):
        summary = json.loads((ep / 'summary.json').read_text())
        rng = {i: np.random.default_rng(np.random.SeedSequence([summary['seed'] & 0xffffffff, i]))
               for i, _ in specs}
        queues, first = {}, {}
        previous = None
        for line in (ep / 'trace.jsonl').read_text().splitlines():
            row = json.loads(line)
            action = np.array(row['action_proposed'], dtype=float)
            for i, p in specs:
                cfg = p.get('config', {})
                n = len(action) if cfg.get('dims') is None else cfg['dims']
                cls = p['entrypoint'].split(':')[-1]
                active = cfg.get('start', 0) <= row['t'] and (cfg.get('end') is None or row['t'] < cfg['end'])
                if not active:
                    queues.pop(i, None);first.pop(i, None)
                    continue
                if cls == 'ActionScale':
                    action[:n] *= cfg.get('scale', .5)
                elif cls == 'ActionNoise':
                    if cfg.get('std', .1):
                        action[:n] += rng[i].normal(0, cfg.get('std', .1), n)
                        if cfg.get('clip_to_bounds', False):
                            action[:n] = np.clip(action[:n], -1, 1)
                elif cls == 'ActionDelay' and cfg.get('steps', 1):
                    q = queues.setdefault(i, deque())
                    first.setdefault(i, (row['t'], action[:n].copy()))
                    q.append((row['t'], action[:n].copy()))
                    source, delayed = first[i] if len(q) <= cfg.get('steps', 1) else q.popleft()
                    action[:n] = delayed
                    events = [e['data'] for e in row['events'] if e['plugin'].startswith(str(i) + ':') and e['hook'] == 'on_action']
                    if not events or events[-1].get('source_t') != source:
                        errors.append([ep.name, row['t'], 'delay source step mismatch'])
                elif cls == 'ActionBarrier':
                    incoming = action.copy()
                    limit = min(1., cfg.get('limit', 1.))
                    bounded = np.clip(action[:n], -limit, limit)
                    delta = cfg.get('max_delta', .25)
                    if previous is not None and delta is not None:
                        bounded = np.clip(np.clip(bounded, previous[:n] - delta, previous[:n] + delta), -limit, limit)
                    action[:n] = bounded
                    changed = bool(np.any(action != incoming))
                    if changed and (not row.get('takeover') or row['takeover'].get('kind') != 'fallback'):
                        errors.append([ep.name, row['t'], 'modified barrier action missing takeover'])
            if row['action'] is not None:
                sent = np.array(row['action'], dtype=float)
                if not np.allclose(action, sent, atol=1e-6, rtol=0):
                    errors.append([ep.name, row['t'], 'S5 output mismatch', action.tolist(), sent.tolist()])
                previous = sent
                checked += 1
        # Validate actual toy state evolution separately from the sent command.
        if config['environment']['entrypoint'].endswith(':ToyPickEnv'):
            step_size = config['environment'].get('config', {}).get('step_size', .06)
            for line in (ep / 'steps.jsonl').read_text().splitlines():
                row = json.loads(line)
                before = np.array(row['state_before']['gripper'])
                expected = np.clip(before + np.clip(np.array(row['action'])[:2], -1, 1) * step_size, 0, 1)
                if not np.allclose(expected, row['state_after']['gripper'], atol=1e-6, rtol=0):
                    errors.append([ep.name, row['t'], 'toy environment state mismatch'])
    return {'run_id': run_dir.name, 'checked_steps': checked, 'passed': not errors, 'errors': errors}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run_dir), ensure_ascii=False, indent=2))
