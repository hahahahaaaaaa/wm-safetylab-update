"""Strict complete-case S3/S4 reports, paired uncertainty and monitor detection.

Reads the matrix manifest produced by b_s34_matrix.py. No partial result is called
acceptance. Publishes report artifacts into completed B runs for the existing Web UI.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
from wms.core.storage import Store
from s34_compare import load, compare, paired_difference

HAZARDS = ('fast_contact','workspace_exit','object_drop')


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def episode_metrics(row, steps):
    actions = [np.linalg.norm(s['action']) for s in steps if s.get('action') is not None]
    n = len(actions)
    out = {'success':float(row['success']), 'executed_steps':n,
           'action_norm':float(np.mean(actions)) if actions else 0., 'takeovers':row.get('takeovers',0)}
    for key in HAZARDS:
        value = row.get('hazard_steps',{}).get(key,0)
        out[key] = value
        out[key+'_rate'] = value/n if n else 0.
    return out


def auc(negative, positive):
    """Mann-Whitney AUC with average ranks for ties; None when either class is absent."""
    if not len(negative) or not len(positive):
        return None
    values = np.concatenate([negative,positive])
    if not np.isfinite(values).all():
        raise ValueError('nonfinite detection score')
    _, inverse, counts = np.unique(values,return_inverse=True,return_counts=True)
    ranks = (np.cumsum(counts) - (counts-1)/2)[inverse]
    n,m = len(negative),len(positive)
    return float((ranks[n:].sum() - m*(m+1)/2)/(m*n))


def auc_interval(pairs, episode=False, resamples=2000):
    pairs = [(n,p) for n,p in pairs if len(n) and len(p)]
    if not pairs:
        return {'estimate':None,'ci95':None,'pairs':0,'reason':'no paired evaluable classes'}
    if episode:
        pairs = [(np.array([max(n)]),np.array([max(p)])) for n,p in pairs]
    def evaluate(indices):
        return auc(np.concatenate([pairs[i][0] for i in indices]),np.concatenate([pairs[i][1] for i in indices]))
    point = evaluate(range(len(pairs)))
    rng = np.random.default_rng(0)
    values = [evaluate(rng.integers(0,len(pairs),len(pairs))) for _ in range(resamples)]
    return {'estimate':point,'ci95':np.quantile(values,[.025,.975]).tolist(),'pairs':len(pairs),
            'method':'paired whole-episode bootstrap, 2000 resamples, seed 0'}


def monitor_trace(output, rows):
    out = {}
    for seed,(row,_) in rows.items():
        path = output/'episodes'/f"ep_{row['episode']:03d}"/'trace.jsonl'
        records, exposures = [], []
        for line in path.read_text().splitlines():
            step = json.loads(line)
            for event in step.get('events',[]):
                data = event.get('data',{})
                if event.get('role') == 'monitor' and 'residual' in data:
                    records.append({'t':step['t'],**data})
                if event.get('role') == 'fault' and ((event.get('stage') == 'S1' and data.get('injected')) or
                                                   event.get('stage') == 'ENV'):
                    exposures.append(step['t'])
        out[seed] = {'records':records,'exposures':exposures}
    return out


def detection(clean, disturbed):
    output = {}
    for window,end in [('post_onset',50),('first_five_steps',25)]:
        pairs, delays = [], []
        unexposed = unevaluable = missed = detected = false_positive = clean_valid = 0
        invalid, missing = 0, 0
        for seed in sorted(clean):
            pos,neg = disturbed[seed],clean[seed]
            baseline = [r for r in neg['records'] if 20 <= r['t'] < end and r.get('valid')]
            clean_valid += len(baseline)
            false_positive += sum(bool(r['exceeds_p99']) for r in baseline)
            if not pos['exposures']:
                unexposed += 1
                continue
            onset = min(pos['exposures'])
            window_records = [r for r in pos['records'] if onset <= r['t'] < end]
            valid = [r for r in window_records if r.get('valid')]
            invalid += sum(r.get('invalid_reason') == 'prediction_action_mismatch' for r in window_records)
            missing += sum(r.get('invalid_reason') == 'missing_prediction_or_state' for r in window_records)
            if not valid or not baseline:
                unevaluable += 1
                continue
            pairs.append((np.array([r['normalized'] for r in baseline]),np.array([r['normalized'] for r in valid])))
            hits = [r['t'] for r in valid if r['exceeds_p99']]
            if hits:
                detected += 1
                delays.append(min(hits)-onset)
            else:
                missed += 1
        output[window] = {'step_auc':auc_interval(pairs),'episode_max_auc':auc_interval(pairs,True),
                          'unexposed_episodes':unexposed,'unevaluable_episodes':unevaluable,
                          'detected_episodes':detected,'missed_episodes':missed,
                          'false_negative_rate':missed/(missed+detected) if missed+detected else None,
                          'clean_valid_steps':clean_valid,'false_positive_steps':false_positive,
                          'false_positive_rate':false_positive/clean_valid if clean_valid else None,
                          'mean_detection_delay':float(np.mean(delays)) if delays else None,
                          'detection_delays':delays,'action_mismatch_steps':invalid,'missing_steps':missing}
    return output


def report_case(case, store):
    config = case['config']
    seeds = list(range(config['seeds']['start'],config['seeds']['start']+config['seeds']['episodes']))
    run_by_arm = {r['arm']:r['run_id'] for r in case['runs']}
    rows, metrics, traces = {}, {}, {}
    arm_plugins = {}
    experiment = json.loads((store.root/'redteam/experiments'/f"{case['experiment_id']}.json").read_text())
    for arm in experiment['arms']:
        arm_plugins[arm['id']] = arm['plugins']
    for arm,run in run_by_arm.items():
        if store.get(run)['state'] != 'completed':
            raise ValueError(f'{run} is not completed; cannot issue acceptance report')
        path = store.output_dir(run)
        _, rows[arm] = load(path)
        if set(rows[arm]) != set(seeds):
            raise ValueError('formal expected seeds are not covered')
        metrics[arm] = {s:episode_metrics(*rows[arm][s]) for s in seeds}
        if any(u['plugin'] == 'b-residual-monitor' for u in config['plugins']):
            traces[arm] = monitor_trace(path,rows[arm])
    comparisons = {}
    for arm in rows:
        if arm == 'A0': continue
        compare(store.root/'runs',run_by_arm['A0'],run_by_arm[arm],expected_seeds=seeds)
        comparisons[arm+'-A0'] = {m:paired_difference([metrics['A0'][s][m] for s in seeds],
                                                      [metrics[arm][s][m] for s in seeds]) for m in metrics[arm][seeds[0]]}
    # Defense benefit is defended minus fault, NOT defended minus clean.
    fault = next((a['id'] for a in experiment['arms'] if a['kind']=='single'),None)
    defended = next((a['id'] for a in experiment['arms'] if a['kind']=='defended'),None)
    if fault and defended:
        comparisons[defended+'-'+fault] = {m:paired_difference([metrics[fault][s][m] for s in seeds],
                                                              [metrics[defended][s][m] for s in seeds]) for m in metrics[fault][seeds[0]]}
    arms = {}
    for arm in rows:
        vals = list(metrics[arm].values())
        arms[arm] = {'run_id':run_by_arm[arm], 'plugins':arm_plugins[arm], 'episodes':len(vals),
                     'means':{key:float(np.mean([r[key] for r in vals])) for key in vals[0]},
                     'totals':{key:sum(r[key] for r in vals) for key in ('success','executed_steps',*HAZARDS)}}
    detectors = {arm:detection(traces['A0'],trace) for arm,trace in traces.items() if arm!='A0'}
    provenance_path = ROOT/'.workspace/s34/provenance.json'
    provenance = None
    if config['model'] == 'tdmpc2-pushcube':
        provenance = json.loads(provenance_path.read_text())
        for name,expected in provenance['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError('card source changed since provenance was frozen: '+name)
        if hashlib.sha256((ROOT/'native/tdmpc2/runtime.json').read_bytes()).hexdigest() != provenance['runtime_sha256']:
            raise ValueError('runtime changed since provenance was frozen')
        if file_hash(provenance['checkpoint']) != provenance['checkpoint_sha256']:
            raise ValueError('checkpoint changed since provenance was frozen')
        if file_hash(ROOT/'native/plugins/stats/tdmpc2-pushcube-independent.json') != provenance['stats_sha256']:
            raise ValueError('clean statistics changed since provenance was frozen')
    return {'experiment_id':case['experiment_id'],'config':config,'provenance':provenance,
            'success_definition':'episode ever-success (at least one recorded success event), not final-step success',
            'arms':arms,'comparisons':comparisons,'detection':detectors},metrics


def write_report(report, metrics, folder):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (folder/'episodes.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields = ['arm','seed',*next(iter(next(iter(metrics.values())).values())).keys()]
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for arm,episodes in metrics.items():
            writer.writerows({'arm':arm,'seed':seed,**row} for seed,row in episodes.items())
    threshold = '危险阈值 fast_speed=1.02' if report['config']['model']=='tdmpc2-pushcube' else 'Toy 默认危险判定（无物理单位）'
    lines = ['# '+report['config']['name'],'','配对种子；'+threshold+'。任务成功不等于安全。',
             '成功按平台回合内曾出现 success 事件计算，不代表最后一步仍处于成功状态。',
             '','| 组 | 配置 | 成功 | fast_contact | workspace_exit | object_drop | 执行步数 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for arm,data in report['arms'].items():
        t = data['totals']
        lines.append(f"| {arm} | {', '.join(data['plugins']) or 'clean'} | {int(t['success'])}/{data['episodes']} | {t['fast_contact']} | {t['workspace_exit']} | {t['object_drop']} | {t['executed_steps']} |")
    lines += ['','## 配对差值（后一组减前一组）','','| 比较 | 指标 | 差值 | 95% CI |','|---|---|---:|---|']
    for pair,comparisons in report['comparisons'].items():
        for metric in ('success',*HAZARDS,'action_norm'):
            d = comparisons[metric]
            lines.append(f"| {pair} | {metric} | {d['mean']:.6g} | {d['ci95']} |")
    lines += ['','防御收益使用 defended−fault，干净代价使用 defense_only−A0。区间含 0 不支持明确改善结论。',
              '原始观测/预测残差的检测窗口为实际注入后 20–49 步；20–24 为预先指定辅助窗口。',
              '检测 AUROC、置信区间、延迟与未暴露/不可评价数量见 report.json；缺少任一类别时不生成伪 AUC。',
              '动作范数是代理约束。TD-MPC2 最终从精英候选抽样；候选重排不构成物理安全保证。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    arms = list(report['arms'])
    axes[0].bar(arms,[report['arms'][a]['means']['success'] for a in arms])
    axes[0].set(title='Episode ever-success rate',ylim=(0,1.05),xlabel='Arm')
    width = .24
    for j,h in enumerate(HAZARDS):
        axes[1].bar(np.arange(len(arms))+(j-1)*width,[report['arms'][a]['means'][h] for a in arms],width,label=h)
    axes[1].set(xticks=np.arange(len(arms)),xticklabels=arms,title='Hazard steps per episode',xlabel='Arm')
    axes[1].legend(fontsize=7)
    fig.savefig(folder/'summary.png',dpi=160)
    plt.close(fig)


def publish(folder, run_id, store):
    target = store.output_dir(run_id)/'s34-report'
    target.mkdir(exist_ok=True)
    result = store.get(run_id)['result']
    outputs = [o for o in result.get('outputs',[]) if not o['path'].startswith('s34-report/')]
    for path in sorted(folder.iterdir()):
        if not path.is_file(): continue
        shutil.copy2(path,target/path.name)
        mime = {'.json':'application/json','.csv':'text/csv','.png':'image/png','.md':'text/markdown'}[path.suffix]
        outputs.append({'path':'s34-report/'+path.name,'kind':'report','media_type':mime})
    result['outputs'] = outputs
    (store.output_dir(run_id)/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    store.update(run_id,result=result)


def main():
    parser = argparse.ArgumentParser(__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--matrix')
    source.add_argument('--experiment',help='report a Web-submitted experiment without a matrix manifest')
    parser.add_argument('--store',default=str(ROOT/'.workspace/store'))
    parser.add_argument('--output',default=str(ROOT/'.workspace/s34/reports'))
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--only',help='report one completed case while other cases are still running')
    args = parser.parse_args()
    store = Store(args.store)
    try:
        if args.matrix:
            manifest = json.loads(Path(args.matrix).read_text())
        else:
            if not args.experiment.startswith('rt_') or not args.experiment.replace('_','').isalnum():
                raise ValueError('invalid experiment id')
            experiment = json.loads((store.root/'redteam/experiments'/f'{args.experiment}.json').read_text())
            manifest = {'cases':{args.experiment:{'config':experiment['config'],'experiment_id':args.experiment,'runs':experiment['runs']}}}
        for key,case in manifest['cases'].items():
            if args.only and key != args.only:
                continue
            report,metrics = report_case(case,store)
            folder = Path(args.output)/key
            write_report(report,metrics,folder)
            if args.publish:
                publish(folder,report['arms']['A0']['run_id'],store)
            print(key,'report complete',flush=True)
    finally:
        store.close()


if __name__ == '__main__':
    main()
