"""Independently replay S6 residuals, CUSUM decisions, alarms and takeover links."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from wms.episodes.s5_metrics import recompute


def audit(output, anomaly_start=None):
    output = Path(output)
    base = recompute(output)
    episodes, errors = [], []
    for path in sorted((output / 'episodes').glob('ep_*/trace.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        alarms, eligible, pre = [], [], []
        served = {}
        scores, until = {}, {}
        counts = {}
        for row in rows:
            for event in row.get('events', []):
                data = event.get('data', {})
                sample = data.get('residual_sample')
                if sample is None:
                    continue
                pid = event['plugin']
                count = counts.setdefault(pid, {'valid_samples': 0, 'skipped_samples': 0, 'alarms': 0})
                count['valid_samples' if sample['valid'] else 'skipped_samples'] += 1
                if sample['valid']:
                    a, b, scales = sample['predicted'], sample['observed'], sample['scale']
                    r = math.sqrt(sum(((x - y) / scales[i if len(scales) > 1 else 0]) ** 2 for i, (x, y) in enumerate(zip(a, b))) / len(a))
                    if not math.isclose(r, sample['residual'], rel_tol=1e-8, abs_tol=1e-12):
                        errors.append([path.parent.name, row['t'], 'residual', pid])
                    eligible.append(row['t'])
                if 'detected' not in data:
                    continue
                # Replay from recorded calibrated parameters, independently of plugin code.
                if not sample['valid'] or row['t'] < until.get(pid, -1):
                    score, detected = 0., False
                else:
                    z = (sample['residual'] - data['center']) / data['spread']
                    score = max(0., scores.get(pid, 0.) + z - data['drift'])
                    detected = score >= data['threshold']
                if not math.isclose(score, data['score'], rel_tol=1e-8, abs_tol=1e-10) or detected != data['detected']:
                    errors.append([path.parent.name, row['t'], 'cusum_replay', pid])
                scores[pid] = 0. if detected else score
                if detected:
                    count['alarms'] += 1
                    until[pid] = row['t'] + data['cooldown_steps'] + 1
                actual = [a for a in row.get('alarms', []) if a['plugin'] == pid]
                if bool(actual) != detected:
                    errors.append([path.parent.name, row['t'], 'alarm_record', pid])
            alarms += [a for a in row.get('alarms', []) if a.get('stage') == 'S6']
            takeover = row.get('takeover')
            if takeover and takeover.get('via') == 'alarm':
                key = (takeover['plugin'], takeover['raised_at'])
                served[key] = served.get(key, 0) + 1
                if takeover['raised_at'] >= row['t']:
                    errors.append([path.parent.name, row['t'], 'takeover_not_after_alarm'])
                if not any(a['plugin'] == takeover['plugin'] and a['t'] == takeover['raised_at'] for a in alarms):
                    errors.append([path.parent.name, row['t'], 'takeover_without_alarm'])
        post = [t for t in eligible if anomaly_start is not None and t >= anomaly_start]
        before = [a for a in alarms if anomaly_start is None or a['t'] < anomaly_start]
        detected = [a for a in alarms if anomaly_start is not None and a['t'] >= anomaly_start]
        for alarm in alarms:
            alarm['served_control_steps'] = served.get((alarm['plugin'], alarm['t']), 0)
            alarm['requested_takeover_not_observed'] = bool(alarm.get('takeover')) and alarm['served_control_steps'] == 0
        summary = json.loads(path.with_name('summary.json').read_text())
        for pid, report in summary.get('plugin_reports', {}).items():
            for field in ('alarms', 'valid_samples', 'skipped_samples'):
                if field in report and report[field] != counts.get(pid, {}).get(field, 0):
                    errors.append([path.parent.name, pid, 'plugin_summary_' + field])
        episodes.append({'episode': path.parent.name, 'alarms': alarms, 'eligible_samples': len(eligible),
                         'pre_anomaly_alarms': len(before), 'post_anomaly_eligible_samples': len(post),
                         'detected': bool(detected) if anomaly_start is not None and post else None,
                         'detection_delay': detected[0]['t'] - anomaly_start if detected else None})
    observable = [ep for ep in episodes if ep['detected'] is not None]
    delays = [ep['detection_delay'] for ep in episodes if ep['detection_delay'] is not None]
    return {'s5_metrics': base, 'episodes': episodes, 'errors': errors,
            'consistent': not errors and base['summary_consistent'],
            'alarm_episode_rate': sum(bool(ep['alarms']) for ep in episodes) / len(episodes),
            'pre_anomaly_alarm_episode_rate': sum(bool(ep['pre_anomaly_alarms']) for ep in episodes) / len(episodes),
            'observable_anomaly_episodes': len(observable),
            'detection_rate': sum(ep['detected'] for ep in observable) / len(observable) if observable else None,
            'mean_detection_delay': sum(delays) / len(delays) if delays else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--anomaly-start', type=int)
    args = parser.parse_args()
    result = audit(args.output, args.anomaly_start)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result['consistent'] else 1)
