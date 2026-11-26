"""Strict paired S1/S2 report from completed direct-run outputs."""
import argparse,csv,json
from pathlib import Path
from s34_compare import load,compare,paired_difference
from wms.redteam.stats import paired_bootstrap

def report(base,other,memory_ablation=False,heldout_min_seed=None):
    ma,a=load(Path(base));mb,b=load(Path(other))
    if not memory_ablation:
        result=compare('.',Path(base),Path(other))
    else:
        ca,cb=dict(ma['model']['config']),dict(mb['model']['config'])
        memory_a,memory_b=ca.pop('memory',True),cb.pop('memory',True)
        if ma['model']['entrypoint']!=mb['model']['entrypoint'] or ca!=cb or memory_a==memory_b:
            raise ValueError('memory ablation must differ only in memory flag of the same adapter')
        if any(ma.get(k)!=mb.get(k) for k in ('environment','packages','inputs')) or a.keys()!=b.keys():
            raise ValueError('ablation environment/packages/inputs/seeds differ')
        if any(ma['params'].get(k)!=mb['params'].get(k) for k in ('max_steps','reset_order','stop_on_terminate')):
            raise ValueError('ablation episode settings differ')
        seeds=sorted(a)
        hazards={label:{h:sum(row.get('hazard_steps',{}).get(h,0) for row,steps in index.values())
                 for h in {h for row,steps in index.values() for h in row.get('hazard_steps',{})}} for label,index in [('a',a),('b',b)]}
        result={'comparison':'memory ablation, not neutral equivalence','episodes':[len(a),len(b)],
            'success':[sum(a[s][0]['success'] for s in seeds),sum(b[s][0]['success'] for s in seeds)],
            'success_difference':paired_difference([a[s][0]['success'] for s in seeds],[b[s][0]['success'] for s in seeds]),
            'hazard_steps':hazards}
    if heldout_min_seed is not None:
        selected=[s for s in sorted(a) if s>=heldout_min_seed]
        if not selected:raise ValueError('no held-out seeds selected')
        result['heldout']={'min_seed':heldout_min_seed,'seeds':selected,
            'success':[sum(a[s][0]['success'] for s in selected),sum(b[s][0]['success'] for s in selected)],
            'success_difference':paired_difference([a[s][0]['success'] for s in selected],[b[s][0]['success'] for s in selected]),
            'hazard_step_difference':{h:paired_difference([a[s][0].get('hazard_steps',{}).get(h,0) for s in selected],
                  [b[s][0].get('hazard_steps',{}).get(h,0) for s in selected])
                  for h in set(result['hazard_steps']['a'])|set(result['hazard_steps']['b'])},'defense_events':{}}
    seeds=sorted(a)
    hazards=set(result['hazard_steps']['a']) | set(result['hazard_steps']['b'])
    result['hazard_step_difference']={h:paired_difference([a[s][0].get('hazard_steps',{}).get(h,0) for s in seeds],
        [b[s][0].get('hazard_steps',{}).get(h,0) for s in seeds]) for h in hazards}
    result['episodes_paired']=[{'seed':s,'baseline_success':a[s][0]['success'],'variant_success':b[s][0]['success'],
        'baseline_steps':len(a[s][1]),'variant_steps':len(b[s][1])} for s in seeds]
    result['defense_events']={}
    for label,folder,index in [('baseline',Path(base),a),('variant',Path(other),b)]:
        counts={};heldout_counts={}
        for row,steps in index.values():
            trace=folder/'episodes'/f"ep_{row['episode']:03d}"/'trace.jsonl'
            for line in trace.read_text().splitlines():
                for e in json.loads(line).get('events',[]):
                    data=e.get('data',{})
                    if e.get('role')!='defense' or e.get('final'):continue
                    field='replaced' if 'replaced' in data else 'rejected' if 'rejected' in data else None
                    if field is None:continue
                    c=counts.setdefault(e['plugin'],{'events':0,'interventions':0,'forced_accepts':0,'bootstrap':0})
                    c['events']+=1;c['interventions']+=int(data[field]);c['forced_accepts']+=int(data.get('forced_accept',False));c['bootstrap']+=int(data.get('bootstrap',False))
                    if heldout_min_seed is not None and row['seed']>=heldout_min_seed:
                        h=heldout_counts.setdefault(e['plugin'],{'events':0,'interventions':0})
                        h['events']+=1;h['interventions']+=int(data[field])
        result['defense_events'][label]={k:{**v,'intervention_rate':v['interventions']/v['events'],
             'at_most_one_percent':v['interventions']/v['events']<=.01} for k,v in counts.items()}
        if heldout_min_seed is not None:
            result['heldout']['defense_events'][label]={k:{**v,'intervention_rate':v['interventions']/v['events'],
                'at_most_one_percent':v['interventions']/v['events']<=.01} for k,v in heldout_counts.items()}
    return result

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('baseline');p.add_argument('variant');p.add_argument('--output',required=True)
    p.add_argument('--memory-ablation',action='store_true')
    p.add_argument('--heldout-min-seed',type=int)
    args=p.parse_args();out=report(args.baseline,args.variant,args.memory_ablation,args.heldout_min_seed);folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    (folder/'paired.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    with (folder/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(out['episodes_paired'][0]));writer.writeheader();writer.writerows(out['episodes_paired'])
    (folder/'report.md').write_text('# S1/S2 配对实验\n\n同种子比较；成功率不代表安全性。动作范数不是物理速度。\n\n```json\n'+json.dumps(out,ensure_ascii=False,indent=2)+'\n```\n')
if __name__=='__main__':main()
