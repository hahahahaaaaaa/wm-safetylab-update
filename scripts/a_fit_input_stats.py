"""Fit input statistics only from declared clean, completed run outputs.

python scripts/a_fit_input_stats.py OUT1 OUT2 --kind image --camera 3rd_view_camera --output STATS
The first half of each episode fits moments; the second half sets the p99 threshold.
These are calibration episodes, not held-out evaluation. Terminal frames are excluded.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from wms.plugins.cards.a_input_monitor import image_features

def readlines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def clean_rows(run):
    result=json.loads((run/'result.json').read_text())
    if result.get('status')!='completed': raise ValueError('incomplete calibration run')
    manifest=json.loads((run/'run_manifest.json').read_text())
    if any(p.get('role') in ('fault','attack') for p in manifest.get('plugins',[])):
        raise ValueError('calibration must not contain fault/attack plugins')
    rows=json.loads((run/'episodes.json').read_text())['episodes']
    if len({r['seed'] for r in rows})!=len(rows) or sorted(r['seed'] for r in rows)!=sorted(manifest['seeds']):
        raise ValueError('incomplete/duplicate calibration seeds')
    for row in rows:
        folder=run/'episodes'/f"ep_{row['episode']:03d}"
        trace=readlines(folder/'trace.jsonl')
        if len(trace)!=row['steps'] or [s['t'] for s in trace]!=list(range(row['steps'])):
            raise ValueError('incomplete calibration trajectory')
        if any(e.get('role') in ('fault','attack') and e.get('data',{}).get('injected') for s in trace for e in s.get('events',[])):
            raise ValueError('calibration contains injected disturbance')
        yield row,folder,trace

def fit(runs,kind,camera,max_seed=None):
    train,calibrate,frame_diffs,write_scores=[],[],[],[]
    sources=[];memory={}
    for run in map(Path,runs):
        for row,folder,trace in clean_rows(run):
            if max_seed is not None and row['seed']>max_seed:continue
            for report in row.get('plugin_reports',{}).values():
                for bank,m in report.get('memory_moments',{}).items():
                    v=memory.setdefault(bank,{'n':0,'sum':np.zeros(len(m['sum'])),'sumsq':np.zeros(len(m['sum']))})
                    v['n']+=m['n'];v['sum']+=m['sum'];v['sumsq']+=m['sumsq']
            if kind=='image':
                with np.load(folder/f'frames_{camera}.npz') as data: frames=data['seen'][:row['steps']]
                features=np.asarray([image_features(frame,frames[t-1] if t else None) for t,frame in enumerate(frames)])
                frame_diffs.extend(features[1:,-1].tolist())
            else:
                features=np.asarray([s['obs_seen'] for s in readlines(folder/'steps.jsonl')],float)
            if len(features)<4: raise ValueError('at least four steps per calibration episode')
            cut=len(features)//2
            train.extend(features[:cut]); calibrate.extend(features[cut:])
            for step in trace:
                for event in step.get('events',[]):
                    d=event.get('data',{})
                    if event.get('stage')=='S2' and 'distance' in d and d['distance'] is not None:
                        if d.get('rejected'): raise ValueError('calibration gate must be neutral')
                        write_scores.append(float(d['distance']))
            sources.append({'output':str(run),'seed':row['seed'],'steps':row['steps'],
                            'trace_sha256':hashlib.sha256((folder/'trace.jsonl').read_bytes()).hexdigest()})
    x,y=np.asarray(train),np.asarray(calibrate)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('invalid calibration features')
    mean=x.mean(0); variance=np.var(x,0)
    if kind=='vector': covariance=np.diag(np.maximum(variance,1e-12))
    else:
        covariance=np.cov(x,rowvar=False)+np.eye(x.shape[1])*max(float(variance.mean())*1e-6,1e-8)
    precision=np.linalg.inv(covariance)
    delta=y-mean
    scores=np.sqrt(np.maximum(0,np.einsum('ni,ij,nj->n',delta,precision,delta)))
    return {'version':'s12-input-v1','kind':kind,'camera':camera if kind=='image' else None,
       'mean':mean.tolist(),'std':np.sqrt(variance).tolist(),'precision':precision.tolist(),
       'threshold':float(np.quantile(scores,.99)),'fit_samples':len(x),'threshold_samples':len(y),
       'frame_tau':float(np.quantile(frame_diffs,.99)) if frame_diffs else None,
       'write_tau':float(np.quantile(write_scores,.99)) if write_scores else None,
       'memory':{k:{'n':v['n'],'mean':(v['sum']/v['n']).tolist(),
                   'std':np.sqrt(np.maximum(0,v['sumsq']/v['n']-(v['sum']/v['n'])**2)).tolist()} for k,v in memory.items() if v['n']},
       'source_episodes':sources,'split':'first-half moments / second-half threshold within clean calibration episodes'}

def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('runs',nargs='+');p.add_argument('--kind',choices=['image','vector'],required=True)
    p.add_argument('--camera',default='3rd_view_camera');p.add_argument('--max-seed',type=int);p.add_argument('--output',required=True);args=p.parse_args()
    out=fit(args.runs,args.kind,args.camera,args.max_seed);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2)+'\n');print(path)
if __name__=='__main__':main()
