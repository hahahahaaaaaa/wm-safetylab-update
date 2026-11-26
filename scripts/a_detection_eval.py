"""Aligned per-step detection metrics, episode-cluster bootstrap; no hidden truth.

Use clean and disturbed run outputs together so AUROC has both classes.
Labels come only from explicit injected=True events of the configured disturbance ID.
"""
import argparse,json
from pathlib import Path
import numpy as np

def auc(labels,scores):
    labels=np.asarray(labels,bool);scores=np.asarray(scores,float)
    n=int(labels.sum());m=len(labels)-n
    if not n or not m:return None
    order=np.argsort(scores,kind='stable');ranks=np.empty(len(scores),float)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and scores[order[j]]==scores[order[i]]:j+=1
        ranks[order[i:j]]=(i+j+1)/2;i=j
    return float((ranks[labels].sum()-n*(n+1)/2)/(n*m))

def evaluate(runs,monitor,disturbance,field='score',threshold=None,bootstrap=2000,min_seed=None):
    def matches_id(actual,expected):
        return actual==expected or actual.partition(':')[2]==expected
    episodes=[]
    for run in map(Path,runs):
        if json.loads((run/'result.json').read_text()).get('status')!='completed':raise ValueError('incomplete run')
        rows=json.loads((run/'episodes.json').read_text())['episodes']
        manifest=json.loads((run/'run_manifest.json').read_text())
        if sorted(r['seed'] for r in rows)!=sorted(manifest['seeds']):raise ValueError('incomplete expected seeds')
        for row in rows:
            if min_seed is not None and row['seed']<min_seed:continue
            trace=[json.loads(s) for s in (run/'episodes'/f"ep_{row['episode']:03d}"/'trace.jsonl').read_text().splitlines()]
            if len(trace)!=row['steps'] or [s['t'] for s in trace]!=list(range(row['steps'])):
                raise ValueError('missing, duplicated, or unordered trace steps')
            labels=[];scores=[];times=[];hits=[]
            for step in trace:
                events=step.get('events',[])
                matches=[e['data'] for e in events if matches_id(e['plugin'],monitor) and field in e.get('data',{}) and not e.get('final')]
                if len(matches)!=1:raise ValueError('exactly one detector score per step required; select one camera')
                score=matches[0][field]
                if score is None or not np.isfinite(score):raise ValueError('nonfinite/missing detector score')
                tau=threshold if threshold is not None else matches[0].get('threshold')
                if tau is None:raise ValueError('detector threshold is required')
                labels.append(any(matches_id(e['plugin'],disturbance) and e.get('data',{}).get('injected') and not e.get('final') for e in events))
                scores.append(float(score));times.append(step['t']);hits.append(score>tau)
            l,h=np.asarray(labels),np.asarray(hits)
            exposed=np.flatnonzero(l); detected=np.flatnonzero(l & h)
            episodes.append({'seed':row['seed'],'run':str(run),'labels':labels,'scores':scores,
                'false_positive_steps':int((~l & h).sum()),'clean_steps':int((~l).sum()),
                'false_negative_steps':int((l & ~h).sum()),'exposed_steps':int(l.sum()),
                'delay':times[detected[0]]-times[exposed[0]] if len(detected) else None})
    if not episodes:raise ValueError('no evaluation episodes selected')
    labels=sum([e['labels'] for e in episodes],[]);scores=sum([e['scores'] for e in episodes],[])
    if len({(e['run'],e['seed']) for e in episodes})!=len(episodes):raise ValueError('duplicate episode seed within a run')
    # Cluster all paired arms sharing a seed, retaining temporal dependence.
    seeds=sorted({e['seed'] for e in episodes});rng=np.random.default_rng(0);values=[]
    groups={s:[e for e in episodes if e['seed']==s] for s in seeds}
    for _ in range(bootstrap):
        batch=[e for s in rng.choice(seeds,len(seeds),replace=True) for e in groups[s]]
        value=auc(sum([e['labels'] for e in batch],[]),sum([e['scores'] for e in batch],[]))
        if value is not None:values.append(value)
    exposed=[e for e in episodes if e['exposed_steps']]; delays=[e['delay'] for e in exposed if e['delay'] is not None]
    fp=sum(e['false_positive_steps'] for e in episodes);clean=sum(e['clean_steps'] for e in episodes)
    fn=sum(e['false_negative_steps'] for e in episodes);pos=sum(e['exposed_steps'] for e in episodes)
    return {'auroc':auc(labels,scores),'auroc_ci95':np.quantile(values,[.025,.975]).tolist() if values else None,
      'bootstrap_valid':len(values),'false_positive_rate':fp/clean if clean else None,
      'false_negative_rate':fn/pos if pos else None,'exposed_episodes':len(exposed),'detected_episodes':len(delays),
      'miss_rate':1-len(delays)/len(exposed) if exposed else None,'mean_detected_delay':float(np.mean(delays)) if delays else None,
      'clean_episode_false_alarm_rate':float(np.mean([e['false_positive_steps']>0 for e in episodes if e['clean_steps']])) if clean else None,
      'evaluation_min_seed':min_seed,'episodes':episodes}

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('runs',nargs='+');p.add_argument('--monitor',required=True)
    p.add_argument('--disturbance',required=True);p.add_argument('--field',default='score');p.add_argument('--threshold',type=float)
    p.add_argument('--output',required=True);p.add_argument('--min-seed',type=int);a=p.parse_args()
    out=evaluate(a.runs,a.monitor,a.disturbance,a.field,a.threshold,min_seed=a.min_seed)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
