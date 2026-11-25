"""Audit completed S6 suites and expose their existing runs in the original UI."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from s5_acceptance import ROOT, STORE, api, save, BASE_URL
from wms.episodes.s6_metrics import audit


def register(dest, label, entries, model):
    marker = dest / (label + '-ui.json')
    if marker.exists():
        return json.loads(marker.read_text())['id']
    eid = 'rt_' + uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    arms, runs, uses = [], [], []
    env = 'maniskill-pushcube-state' if model == 'tdmpc2-pushcube' else 'toy-pick'
    for i, entry in enumerate(entries):
        keys = []
        for j, p in enumerate(entry['config']['params']['plugins']):
            key = f'arm{i}_plugin{j}';keys.append(key)
            uses.append({'key': key, 'plugin': p['id'], 'config': p['config']})
        arms.append({'id': 'A' + str(i), 'name': entry['label'], 'kind': 'baseline' if i == 0 else 'combined',
                     'plugins': keys, 'oracle': False})
        runs.append({'arm': 'A' + str(i), 'environment': env, 'run_id': entry['run_id']})
    value = {'id': eid, 'name': label, 'created_at': now, 'updated_at': now,
             'config': {'name': label, 'description': 'Grouping already completed S6 runs; per-arm calibrated parameters and fixture semantics are in each original run config.',
                        'model': model, 'environments': [env], 'plugins': uses,
                        'seeds': {'start': entries[0]['config']['seed'], 'episodes': 20}}, 'arms': arms, 'runs': runs}
    with (ROOT / '.workspace/store/redteam/experiments' / (eid + '.json')).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    save(marker, value)
    api('/experiments/' + eid + '/report')
    return eid


def generate(dest, initial):
    phases = {phase: json.loads((dest / phase / 'summary.json').read_text()) for phase in ('cpu', 'gpu')}
    checks = []
    for root in [initial, dest]:
        for phase in ('cpu', 'gpu'):
            for path in sorted((root / phase).glob('*/submitted.json')):
                submitted = json.loads(path.read_text())
                current = api('/runs/' + submitted['id'])
                if current['state'] != 'completed':
                    checks.append({'run_id': current['id'], 'consistent': False, 'state': current['state']})
                    continue
                config = current['config'];env = config['environment']['config']
                is_noise = any(p['entrypoint'].endswith(':ActionNoise') for p in config['params']['plugins'])
                start = env.get('shift_start') if 'shift_start' in env else (10 if is_noise and any(p['entrypoint'].endswith(':CusumDefense') for p in config['params']['plugins']) else None)
                result = audit(STORE / current['id'] / 'output', start)
                save(dest / 'recomputed' / (current['id'] + '.json'), result)
                checks.append({'run_id': current['id'], 'consistent': result['consistent'], 'errors': result['errors'],
                               'episodes': len(result['episodes']), 'steps': sum(e['executed_steps'] for e in result['s5_metrics']['episodes'])})
    # Check the toy poison's reported positions against independent raw environment records.
    cpu = phases['cpu']['evaluation']
    clean = next(e for e in cpu if e['label'] == 'online-clean')
    neutral = next(e for e in cpu if e['label'] == 'online-neutral')
    poison = next(e for e in cpu if e['label'] == 'online-poison')
    identical, poison_checks = [], []
    for ep in sorted((STORE / clean['run_id'] / 'output/episodes').iterdir()):
        other = STORE / neutral['run_id'] / 'output/episodes' / ep.name
        same = (ep / 'steps.jsonl').read_bytes() == (other / 'steps.jsonl').read_bytes()
        same &= all(hashlib.sha256(f.read_bytes()).digest() == hashlib.sha256((other / f.name).read_bytes()).digest() for f in ep.glob('frames_*.npz'))
        identical.append(bool(same))
    for ep in sorted((STORE / poison['run_id'] / 'output/episodes').iterdir()):
        steps = {r['t']: r for r in map(json.loads, (ep / 'steps.jsonl').read_text().splitlines())}
        for row in map(json.loads, (ep / 'trace.jsonl').read_text().splitlines()):
            for event in row['events']:
                data = event['data']
                if data.get('scope') == 'learning_feedback_only':
                    raw = steps[row['t']]
                    before, after = np.array(raw['state_before']['gripper']), np.array(raw['state_after']['gripper'])
                    good = np.allclose(data['clean_position'], after, atol=1e-12, rtol=0)
                    good &= np.allclose(data['poisoned_position'], before + data['factor'] * (after - before), atol=1e-12, rtol=0)
                    poison_checks.append(bool(good))
    save(dest / 'poison-verification.json', {'neutral_trajectory_identical': identical,
                                           'poisoned_steps_checked': len(poison_checks), 'raw_feedback_preserved': all(poison_checks)})
    pages = {
        'cpu': register(dest, 's6-acceptance-CPU-CUSUM-v2', [e for e in cpu if not e['label'].startswith('online-')], 'toy-planner'),
        'online': register(dest, 's6-acceptance-online-poison-v2', [e for e in cpu if e['label'].startswith('online-')], 'toy-planner-online'),
        'gpu': register(dest, 's6-acceptance-GPU-CUSUM-v2', phases['gpu']['evaluation'], 'tdmpc2-pushcube'),
    }
    fields = ['run_id', 'episode', 't', 'plugin', 'score', 'takeover', 'steps', 'served_control_steps', 'requested_takeover_not_observed']
    with (dest / 'alarms.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        for path in (dest / 'recomputed').glob('*.json'):
            result = json.loads(path.read_text())
            for ep in result['episodes']:
                for alarm in ep['alarms']:
                    writer.writerow({k: ({'run_id': path.stem, 'episode': ep['episode'], **alarm}).get(k) for k in fields})
    for entry in [e for e in cpu if e['label'] in ('shift-alarm', 'shift-stop', 'online-poison')] + [phases['gpu']['evaluation'][-1]]:
        target = dest / 'representative-traces' / entry['label'];target.mkdir(parents=True, exist_ok=True)
        source = STORE / entry['run_id'] / 'output/episodes/ep_000'
        for name in ('trace.jsonl', 'summary.json'):
            shutil.copy2(source / name, target / name)
    summary = {'runs': len(checks), 'episodes': sum(c.get('episodes', 0) for c in checks),
               'checked_steps': sum(c.get('steps', 0) for c in checks), 'all_consistent': all(c['consistent'] for c in checks),
               'neutral_identical': all(identical), 'poison_feedback_checks': len(poison_checks),
               'poison_feedback_preserved': bool(poison_checks) and all(poison_checks), 'pages': pages, 'checks': checks}
    save(dest / 'acceptance-summary.json', summary)
    lines = ['# S6 开发与验证记录（2026-10-30）', '',
             '已实现残差监控、CUSUM告警/停止/备用控制、在线玩具反馈投毒，复用原接管机制。原版八步 UI、导航与 CSS 保留，新增配置仅在 S6 卡片内。', '',
             '## 预设与校准', '',
             '残差 planned-next-rms/v1：预测下一潜状态与执行后观测编码差异的逐维尺度归一化RMS。尺度仅由干净校准数据各维误差RMS确定，下限0.001；再用干净标量残差均值和标准差标准化。阈值为干净序列CUSUM峰值的1.2倍加0.5（至少1），drift=0.5。', '',
             'CPU校准seeds600–619，评估700–719；GPU校准1400–1419，评估1500–1519。每个评估组20回合。默认排除动作改变和接管步骤；GPU故障组显式include_changed=true，解释为执行闭环偏离，不是模型失效概率。', '',
             '第一轮统一尺度保留在原始验收目录：CPU动力学变化检出率仅5%，GPU干净回合告警率25%。逐维尺度针对表示中不同维度的量级差异，修改后换新种子独立校准/评估，没有用评估异常样本选择阈值。', '',
             '## 核对结果', '',
             f"两轮共核对{summary['runs']}个运行、{summary['episodes']}回合、{summary['checked_steps']}个执行步；残差、CUSUM回放、告警与接管来源、原回合汇总一致：{summary['all_consistent']}。",
             f"反馈投毒独立核对{len(poison_checks)}步；原环境反馈保留：{summary['poison_feedback_preserved']}。20个零投毒回合的轨迹和帧与干净组逐字节相同：{summary['neutral_identical']}。", '',
             '16项S5/S6 unittest通过；三个S6插件的契约检查通过。停止请求优先于备用动作，被S5临时覆盖的S6备用请求不消耗待执行步数；未改变的NaN不再被严格模式误认为原地修改，但仍不参与残差。', '',
             '## 独立评估结果', '',
             '| 环境 | 组 | 成功率 | 回合告警率 | 异常前告警率 | 可观察异常检出率 | 检测延迟/步 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for phase, data in phases.items():
        for e in data['evaluation']:
            detection = '—' if e['detection_rate'] is None else f"{e['detection_rate']:.0%}"
            delay = '—' if e['mean_detection_delay'] is None else f"{e['mean_detection_delay']:.2f}"
            lines.append(f"| {phase} | {e['label']} | {e['success_rate']:.0%} | {e['alarm_episode_rate']:.0%} | {e['pre_anomaly_alarm_episode_rate']:.0%} | {detection} | {delay} |")
    lines += ['', '在线投毒组没有启用检测器，其告警率0不能解释为通过了投毒检测。', '',
              f"在线学习的末次动作前增益均值：干净组{clean['last_logged_gain_mean']}，投毒组{poison['last_logged_gain_mean']}，无投毒组{neutral['last_logged_gain_mean']}；这些是玩具动力学参数，不是神经网络权重。", '',
              'CPU异常为第8步开始的隐藏执行增益变化（仅验收fixture，不暴露给检测器）；GPU异常为第10–39步std=0.4的动作噪声，前7维、显式限幅，使用冻结TD-MPC2 checkpoint及独立场景重建。', '',
              '当前逐维版本在GPU正常组没有告警，但GPU噪声异常组也没有检出，因此GPU检测效果未通过。初版GPU检出率100%同时有25%正常回合告警率，两版都不能据此称为可靠检测器；保留两版数据，不继续依据这些评估样本调参。', '',
              '## 页面与证据', '']
    for key, eid in pages.items():
        lines.append(f"- [{key}结果]({(BASE_URL or '').rstrip('/')}/?s6=acceptance#results/report?exp={eid})")
    lines += ['', '`alarms.csv`包含每次告警、请求类型、实际服务步数和未观察到接管的请求；`recomputed/`包含完整逐回合重算和接管记录；`poison-verification.json`核对真实/投毒反馈；原运行配置与trace保留在A工作区。', '',
              '## 解释边界', '',
              '停止模式结束回合，任务成功率下降是预期代价，不能据此认为停止链路失效。备用动作[0,0,-1]仅用于玩具链路验收，它会打开夹爪，可能导致物体释放/掉落；该组成功率45%、危险计数10.2/回合，不构成安全改善。该动作不是推荐的安全控制器。CUSUM只判断预设残差异常，不保证覆盖所有危险；默认跳过动作修改也意味着对该部分偏离不可观察。', '',
              '正常回合告警率和异常检测率只描述这批20个种子的结果，不是总体保证。未将反馈投毒扩展到不具备learn能力的TD-MPC2，不实施on_update训练。S3将来提供统一残差时可替换版本化函数，再重新校准。']
    (dest / 'acceptance.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'checks'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--initial', type=Path, required=True)
    args = parser.parse_args()
    generate(args.output, args.initial)
