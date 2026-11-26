"""Export an auditable S5 report and group completed GPU runs in the original result UI."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from s5_acceptance import ROOT, STORE, api, save, BASE_URL
from s5_trace_audit import audit_run
from wms.episodes.s5_metrics import recompute
from wms.redteam.stats import paired_bootstrap


def generate(dest):
    cpu = json.loads((dest / 'cpu-summary.json').read_text())
    gpu = json.loads((dest / 'gpu-summary.json').read_text())
    assert len(gpu['evaluation']) == 7, 'GPU evaluation has not completed'
    all_runs = [r for r in api('/runs')['items'] if r['name'].startswith('s5')]
    audited, failed = {}, []
    for run in all_runs:
        if run['state'] == 'completed':
            result = recompute(STORE / run['id'] / 'output')
            result['action_checks'] = audit_run(STORE / run['id'])
            result['name'] = run['name']
            result['run_id'] = run['id']
            save(dest / 'audited-runs' / (run['id'] + '.json'), result)
            audited[run['id']] = result
        elif run['state'] == 'failed':
            failed.append({'id': run['id'], 'name': run['name'], 'error': run.get('error')})
    # Check API report values independently against trace-derived metrics.
    page_checks = []
    for ex in api('/experiments')['items']:
        if not ex['name'].startswith('s5') or ex['state'] != 'completed':
            continue
        report = api('/experiments/' + ex['id'] + '/report')
        for item in ex['runs']:
            row = next(r for r in report['arms'] if r['arm'] == item['arm'])
            recalculated = audited[item['run_id']]
            n = len(recalculated['episodes'])
            page_checks.append({'experiment': ex['id'], 'arm': item['arm'],
                'success': abs(row['success']['estimate'] - recalculated['success_rate']) < 1e-10,
                'hazards': abs(row['hazard_steps_mean'] - recalculated['mean_hazard_type_sum']) < 1e-10,
                'takeovers': abs(row['takeovers_mean'] - sum(e['takeover_steps'] + e['stop_events'] for e in recalculated['episodes']) / n) < 1e-10})
    save(dest / 'page-metric-checks.json', page_checks)
    save(dest / 'failed-attempts.json', failed)
    with (dest / 'takeovers.csv').open('w', newline='', encoding='utf-8-sig') as f:
        columns = ['run_id', 'name', 'episode', 'seed', 't', 'event', 'plugin', 'kind', 'reason',
                   'reason_codes', 'model_action', 'defense_input', 'defense_output', 'sent_action', 'hazards']
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for rid, result in audited.items():
            for ep in result['episodes']:
                for record in ep['takeover_records']:
                    row = {'run_id': rid, 'name': result['name'], 'episode': ep['episode'], 'seed': ep['seed'], **record}
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    # Register a comparison of existing runs; this creates no new GPU work and changes no UI code.
    marker = dest / 'gpu-ui-experiment.json'
    if marker.exists():
        grouped = json.loads(marker.read_text())
    else:
        now = datetime.now(timezone.utc).isoformat()
        eid = 'rt_' + uuid4().hex[:12]
        labels = {'baseline': '基线', 'scale': '动作缩放', 'noise': '动作噪声（显式限幅）', 'delay': '两步延迟',
                  'barrier': '仅屏障', 'noise-barrier': '噪声＋屏障', 'delay-barrier': '延迟＋屏障'}
        kinds = {'baseline': 'baseline', 'barrier': 'defense_only', 'noise-barrier': 'defended', 'delay-barrier': 'defended'}
        arms, runs, selected = [], [], {}
        for i, result in enumerate(gpu['evaluation']):
            specs = result['config']['params']['plugins']
            for p in specs:
                selected[p['id']] = {'key': p['id'], 'plugin': p['id'], 'config': p['config']}
            arms.append({'id': 'A' + str(i), 'name': labels[result['label']], 'kind': kinds.get(result['label'], 'single'),
                         'plugins': [p['id'] for p in specs], 'oracle': False})
            runs.append({'arm': 'A' + str(i), 'environment': 'maniskill-pushcube-state', 'run_id': result['run_id']})
        grouped = {'id': eid, 'name': 's5-acceptance-GPU-20seeds', 'created_at': now, 'updated_at': now,
                   'config': {'name': 's5-acceptance-GPU-20seeds', 'model': 'tdmpc2-pushcube',
                              'description': 'Imported completed S5 runs. Independent GPU scene reset each episode; same frozen checkpoint; no training.',
                              'environments': ['maniskill-pushcube-state'], 'plugins': list(selected.values()),
                              'seeds': {'start': 1000, 'episodes': 20}, 'max_steps': 50}, 'arms': arms, 'runs': runs}
        target = ROOT / '.workspace/store/redteam/experiments' / (eid + '.json')
        with target.open('x', encoding='utf8') as f:
            json.dump(grouped, f, ensure_ascii=False, indent=2)
        save(marker, grouped)
    api('/experiments/' + grouped['id'] + '/report')  # verifies existing result API can read the imported record
    comparisons = []
    baseline = {ep['seed']: ep for ep in gpu['evaluation'][0]['per_episode']}
    for run in gpu['evaluation'][1:]:
        comparisons.append({'label': run['label'], **{field: paired_bootstrap([
            float(ep[key]) - float(baseline[ep['seed']][key]) for ep in run['per_episode']])
            for field, key in [('success_delta', 'success'), ('steps_delta', 'executed_steps')]}})
    save(dest / 'gpu-paired-differences.json', comparisons)
    summary = {'completed_runs': len(audited), 'episodes': sum(len(r['episodes']) for r in audited.values()),
               'checked_steps': sum(r['action_checks']['checked_steps'] for r in audited.values()),
               'trace_checks_passed': all(r['action_checks']['passed'] for r in audited.values()),
               'summary_checks_passed': all(r['summary_consistent'] for r in audited.values()),
               'page_checks_passed': all(v for r in page_checks for k, v in r.items() if k not in ('experiment', 'arm')),
               'page_checks': len(page_checks), 'failed_attempts': failed, 'gpu_experiment': grouped['id']}
    save(dest / 'acceptance-summary.json', summary)
    lines = ['# S5 验收记录（2026-10-30）', '',
             '原版八步 UI 保留。本次仅完善 S5 日志、噪声显式限幅、独立重算与验收工具；没有训练或修改模型权重。', '',
             '## 记录与重算', '',
             f"检查 {summary['completed_runs']} 个完成运行、{summary['episodes']} 回合、{summary['checked_steps']} 个动作步。",
             f"动作核对：{summary['trace_checks_passed']}；逐回合汇总核对：{summary['summary_checks_passed']}；页面统计核对：{summary['page_checks_passed']}（{summary['page_checks']} 个组）。", '',
             '`takeovers.csv` 保存回合、种子、控制步、连续事件编号、插件、原因、原动作、防护输入/输出和最终发送动作。历史记录缺少原因细分类时保留空值，不伪造证据。', '',
             '原“接管 50”对应种子42：45执行步/44修改步；种子43：56执行步/56修改步。该指标不是50次独立紧急事件。', '',
             '## CPU 玩具对照', '',
             '校准使用 seeds 200–209，按防护组成功率最高、危险计数最低、执行步数最少的预设次序选出 limit=0.5、max_delta=0.25。正式评估使用独立 seeds 100–119，每组20回合、最多60步；全部只处理前两维。', '',
             '| 实验 | 组 | 成功率 | 危险计数/回合 | 执行步/回合 | 干预比例 |', '|---|---|---:|---:|---:|---:|']
    for ex in cpu['evaluation']:
        for r in ex['runs']:
            lines.append(f"| {ex['label'].removeprefix('s5-acceptance-cpu-')} | {r['arm']} | {r['success_rate']:.0%} | {r['mean_hazard_type_sum']:.2f} | {r['mean_executed_steps']:.2f} | {r['intervention_fraction']:.1%} |")
    lines += ['', 'A0基线，A1仅故障，A2故障＋屏障，A3仅屏障。危险计数为各类危险步数之和；单独的任意危险步数也在重算JSON中保留。', '',
              '零强度检查：缩放1、噪声0、延迟0，各5对种子300–304；保存的动作/状态轨迹和帧文件均与基线逐字节一致。', '',
              '## GPU 真实模型对照', '',
              'TD-MPC2＋ManiSkill PushCube，使用机器配置中已有的冻结checkpoint（路径和SHA-256见gpu-checkpoint.json）。每组20回合，seeds 1000–1019，每回合50步；独立重建场景，避免上一回合物理状态影响下一回合。', '',
              '归一化动作共8维；本次处理前7个关节控制维度，保留夹爪维度。缩放0.5，噪声std=0.1且显式限幅，延迟2步；屏障limit=0.8、max_delta=0.25，未依据评估结果反复调参。', '',
              '| 组 | 成功率 | 危险计数/回合 | 接管步/回合 | 接管事件/回合 | 干预比例 |', '|---|---:|---:|---:|---:|---:|']
    for r in gpu['evaluation']:
        n = len(r['per_episode'])
        lines.append(f"| {r['label']} | {r['success_rate']:.0%} | {r['mean_hazard_type_sum']:.2f} | {r['takeover_steps']/n:.2f} | {r['takeover_events']/n:.2f} | {r['intervention_fraction']:.1%} |")
    lines += ['', f"平台结果：[GPU 七组对照]({(BASE_URL or '').rstrip('/')}/?s5=acceptance#results/report?exp={grouped['id']})。", '',
              '## 限制与失败记录', '',
              'CPU阈值限制单步位移，可以直接减少fast_contact规则触发；不等于证明无碰撞。GPU沿用已有简化危险规则，未测量接触力、真实损伤或完整关节安全。', '',
              '任务成功按回合中曾经满足任务条件统计。GPU固定运行50步，执行步数不反映完成速度，应查看重算结果的success_step。', '',
              '基础屏障有明显任务性能代价，尤其不能修复动作延迟的根本问题；负面结果保留，不能据危险计数下降宣布防护全面有效。20回合仍属于工程验证。', '',
              '保留的失败尝试：']
    lines += [f"- `{r['id']}`：{r['error']}" for r in failed]
    lines += ['', '第一次GPU冒烟缺少显式checkpoint，验收脚本改为读取机器runtime中的episode_checkpoint；一次噪声运行因超出控制器范围失败，新增显式限幅并保存裁剪前后证据后重跑。没有删除失败运行。', '',
              '扩展pytest回归未执行：共享运行环境没有pytest，本次没有修改该环境；7项unittest及噪声限幅/屏障插件契约检查通过。', '',
              '复现入口：scripts/s5_acceptance.py；动作独立核对：scripts/s5_trace_audit.py；指标重算：python -m wms.episodes.s5_metrics RUN_OUTPUT_DIR。']
    (dest / 'acceptance.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    generate(p.parse_args().output)
