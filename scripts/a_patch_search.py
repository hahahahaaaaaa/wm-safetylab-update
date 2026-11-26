"""Seeded black-box patch search. Reads only episodes.json public outcome summaries.

Search uses layout IDs 0..11 only. Evaluation must use 12..23; no evaluation
outcomes are read here. A query is one model/environment episode, not a candidate.
"""
import argparse,json,math,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--recipe',default='toy-episode-occlusion')
    p.add_argument('--model',default='toy-centroid-memory');p.add_argument('--area',type=float,default=.1)
    p.add_argument('--rounds',type=int,default=5);p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--queries',type=int,default=480);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();folder=Path(a.output).resolve();folder.mkdir(parents=True,exist_ok=True)
    if (folder/'search_log.jsonl').exists():raise ValueError('search output exists; use a new directory')
    if not 0<a.area<=1 or not 1<=a.rounds<=5 or not 1<=a.candidates<=8 or not 12<=a.queries<=480:raise ValueError('invalid area/query budget')
    rng=random.Random(a.seed);side=math.sqrt(a.area);best=None;used=0
    for round_id in range(a.rounds):
        for candidate in range(a.candidates):
            if used+12>a.queries:break
            radius=(1-side)*(.5**round_id)
            center=best['center'] if best else [rng.random()*(1-side)+side/2 for _ in range(2)]
            center=[min(1-side/2,max(side/2,v+rng.uniform(-radius,radius))) for v in center]
            box=[center[0]-side/2,center[1]-side/2,center[0]+side/2,center[1]+side/2]
            config={'area':a.area,'box':box,'threat':{'budget':{'area':a.area,'queries':a.queries}}}
            run_id=f'query_{round_id}_{candidate}'
            args=[sys.executable,str(ROOT/'scripts/run_direct.py'),a.recipe,run_id,'model='+a.model,
                'params.seeds='+json.dumps(list(range(12))),'params.episodes=12','params.perturbations=[]','params.target_moves=[]',
                'params.video=false','params.save_raw=false','params.plugins='+json.dumps([{'plugin':'a-patch-attack','config':config}]),
                '--runs-dir',str(folder/'runs')]
            used+=12 # charge attempted queries even if worker fails
            with (folder/'search_log.jsonl').open('a') as f:f.write(json.dumps({'round':round_id,'candidate':candidate,'box':box,'queries_used':used,'state':'submitted'})+'\n')
            subprocess.run(args,cwd=ROOT,check=True)
            rows=json.loads((folder/'runs'/run_id/'output/episodes.json').read_text())['episodes']
            if sorted(r['seed'] for r in rows)!=list(range(12)):raise ValueError('search requires exactly 12 training layouts')
            success=sum(bool(r['success']) for r in rows)/12
            distances=[r.get('final_distance') for r in rows if r.get('final_distance') is not None]
            score=(success,-sum(distances)/len(distances) if distances else 0)
            record={'round':round_id,'candidate':candidate,'center':center,'box':box,'score':score,'queries_used':used,'config':config,'run':run_id}
            if best is None or score<tuple(best['score']):best=record
            with (folder/'search_log.jsonl').open('a') as f:f.write(json.dumps({**record,'best_score':best['score'],'state':'completed'})+'\n')
            (folder/'best.json').write_text(json.dumps({'state':'running','best':best,'queries_used':used,'training_seeds':list(range(12)),'heldout_seeds':list(range(12,24))},indent=2)+'\n')
    (folder/'best.json').write_text(json.dumps({'state':'completed','best':best,'queries_used':used,'training_seeds':list(range(12)),'heldout_seeds':list(range(12,24))},indent=2)+'\n')
    print(json.dumps(best))
if __name__=='__main__':main()
