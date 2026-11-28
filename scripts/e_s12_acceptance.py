"""Serial, resumable E-only S1/S2 acceptance; stops on failure, never changes shared resources.

./dev.sh python scripts/e_s12_acceptance.py --phase toy
./dev.sh python scripts/e_s12_acceptance.py --phase td
./dev.sh python scripts/e_s12_acceptance.py --phase memory

Real phases implement full task-card matrices. No training or model downloads.
STOP file pauses between jobs. Each job runs in its own runtime process.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.workspace/s12/acceptance'

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def use(name,**config):return {'plugin':name,'config':config}

class Suite:
    def __init__(self,phase):
        from wms.core.registry import Registry
        self.phase=phase;self.folder=OUT/phase;self.folder.mkdir(parents=True,exist_ok=True)
        self.registry=Registry()
        self.settings=json.loads((ROOT/'.workspace/settings.json').read_text())
        self.toolpython=self.settings['toy_python']
        self.source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for parent in ('src','scripts','native') for p in (ROOT/parent).rglob('*')
           if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in str(p)}
        path=self.folder/'source.json'
        if path.exists() and json.loads(path.read_text())!=self.source:raise ValueError('source changed since acceptance; use a separately versioned acceptance directory')
        save(path,self.source)
        self.weights={}
        if phase!='toy':
            runtime=json.loads((ROOT/('native/tdmpc2/runtime.json' if phase=='td' else 'native/memoryvla/runtime.json')).read_text())
            weight=Path(runtime['episode_checkpoint'] if phase=='td' else runtime['checkpoint'])
            digest=hashlib.sha256()
            with weight.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
            stat=weight.stat()
            self.weights={'path':str(weight),'sha256':digest.hexdigest(),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            saved=self.folder/'weights.json'
            if saved.exists() and json.loads(saved.read_text())!=self.weights:raise ValueError('checkpoint changed since acceptance')
            save(saved,self.weights)
        self.records=json.loads((self.folder/'jobs.json').read_text()) if (self.folder/'jobs.json').exists() else {}

    def job(self,name,model,environment,plugins,seeds,steps):
        if (OUT/'STOP').exists():raise RuntimeError('STOP requested between jobs')
        for name_source,sha in self.source.items():
            if hashlib.sha256((ROOT/name_source).read_bytes()).hexdigest()!=sha:
                raise RuntimeError('source changed during acceptance: '+name_source)
        if shutil.disk_usage(ROOT).free < 35*1024**3:raise RuntimeError('less than 35 GiB free; stopped before next job')
        if self.weights:
            weightstat=Path(self.weights['path']).stat()
            if weightstat.st_size!=self.weights['size'] or weightstat.st_mtime_ns!=self.weights['mtime_ns']:
                raise RuntimeError('shared checkpoint changed during acceptance')
        recipe='toy-episode-occlusion' if model.startswith('toy') else ('tdmpc2-pushcube-episode' if model.startswith('tdmpc2') else 'memoryvla-bridge-episode')
        request={'recipe':recipe,'model':model,'environment':environment,'seed':seeds[0],
                 'params':{'seeds':seeds,'episodes':len(seeds),'max_steps':steps,'plugins':plugins,
                  'perturbations':[],'target_moves':[],'save_raw':False,'video':False}}
        folder=self.folder/'runs'/name;result=folder/'output/result.json'
        digest=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        if name in self.records:
            if self.records[name]['sha256']!=digest:raise ValueError('changed request '+name)
            if result.exists() and json.loads(result.read_text()).get('status')=='completed':return folder/'output'
            raise RuntimeError('Previous job incomplete/failed; inspect before explicit retry: '+name)
        if not model.startswith('toy'):
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free < (17000 if model.startswith('memoryvla') else 4000):
                raise RuntimeError('insufficient free GPU memory before job; other processes were left untouched')
        config=self.registry.resolve(request);folder.mkdir(parents=True,exist_ok=False)
        save(folder/'request.json',request);save(folder/'config.json',config)
        self.records[name]={'sha256':digest,'state':'running','episodes':len(seeds),'started':time.time(),'output':str(folder/'output')}
        save(self.folder/'jobs.json',self.records)
        print(self.phase,name,'START',len(seeds),'episodes',flush=True)
        runtime=config['runtime'];env={**os.environ,**{str(k):str(v) for k,v in runtime.get('env',{}).items()},
            'PYTHONPATH':os.pathsep.join([str(ROOT/'src'),*runtime.get('python_paths',[])])}
        device=runtime.get('cuda_visible_devices',runtime.get('gpu_device'))
        if device is not None:env['CUDA_VISIBLE_DEVICES']=str(device)
        with (folder/'run.log').open('w') as log:
            process=subprocess.run([runtime.get('python') or sys.executable,'-m','wms.runners.unified',
                '--config',str(folder/'config.json'),'--output-dir',str(folder/'output')],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,timeout=max(config.get('timeout_s',21600),21600))
        ok=process.returncode==0 and result.exists() and json.loads(result.read_text()).get('status')=='completed'
        self.records[name].update(state='completed' if ok else 'failed',finished=time.time())
        save(self.folder/'jobs.json',self.records)
        print(self.phase,name,self.records[name]['state'],flush=True)
        if not ok:raise RuntimeError('Failed job '+name+'; inspect '+str(folder/'run.log'))
        return folder/'output'

    def tool(self,script,*args):
        subprocess.run([self.toolpython,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,check=True)

    def compare(self,name,base,variant,neutral=False):
        self.tool('s34_compare.py',base,variant,*(['--neutral'] if neutral else []))
        self.tool('e_s12_report.py',base,variant,'--output',self.folder/'reports'/name,
                  *(['--heldout-min-seed',12] if self.phase=='memory' and name!='neutral' else []))

    def fit(self,runs,kind,camera='top',max_seed=None):
        path=self.folder/'input-stats.json'
        self.tool('a_fit_input_stats.py',*runs,'--kind',kind,'--camera',camera,'--output',path,
                  *(['--max-seed',max_seed] if max_seed is not None else []))
        return path,json.loads(path.read_text())

    def detect(self,name,base,variant,monitor,disturbance):
        self.tool('a_detection_eval.py',base,variant,'--monitor',monitor,'--disturbance',disturbance,
                  '--output',self.folder/'reports'/name/'detection.json')
        if self.phase=='memory':
            self.tool('a_detection_eval.py',base,variant,'--monitor',monitor,'--disturbance',disturbance,
                      '--min-seed',12,'--output',self.folder/'reports'/name/'detection-heldout.json')

def toy(s):
    model,env='toy-centroid-memory','toy-pick'
    calibration=s.job('calibration',model,env,[use('b-write-gate',tau=None)],list(range(12)),60)
    path,stats=s.fit([calibration],'image')
    watch=use('a-input-monitor',stats=str(path),camera='top')
    seeds=list(range(12,24));base=s.job('baseline',model,env,[],seeds,60)
    monitored=s.job('monitor',model,env,[watch],seeds,60);s.compare('monitor-neutral',base,monitored,True)
    faults=[('corruption',use('a-obs-corruption',kind='noise',severity=3,start=10,end=15)),
      ('jitter',use('a-camera-jitter',shift=.1,rotate=10,mode='per_step',start=10,end=20)),
      ('patch',use('a-patch-attack',area=.1,start=10,end=20)),
      ('wipe',use('b-memory-fault',mode='wipe',steps=[10])),
      ('stale',use('b-memory-fault',mode='stale',k=3,start=10)),
      ('memory-noise',use('b-memory-fault',mode='noise',std=.1,c=1,start=10))]
    for name,fault in faults:
        variant=s.job(name,model,env,[fault,watch],seeds,60);s.compare(name,monitored,variant)
        if name in ('corruption','jitter','patch'):
            s.detect(name,monitored,variant,'a-input-monitor',fault['plugin'])
    for name,defense in [('frame',use('a-frame-guard',tau=stats['frame_tau'])),('write',use('b-write-gate',tau=stats['write_tau']))]:
        alone=s.job(name+'-clean',model,env,[defense],seeds,60);s.compare(name+'-clean-cost',base,alone)
        for key,fault in faults[:3]:
            variant=s.job(name+'-'+key,model,env,[fault,defense,watch],seeds,60)
            s.compare(name+'-'+key,s.folder/'runs'/key/'output',variant)
    search=s.folder/'search'
    if not (search/'best.json').exists():
        s.tool('a_patch_search.py','--rounds',2,'--candidates',4,'--queries',96,'--output',search)
    best=json.loads((search/'best.json').read_text())['best']['config']
    if json.loads((search/'best.json').read_text()).get('state')!='completed':raise RuntimeError('patch search did not finish')
    import random as random_module
    rng=random_module.Random(983);side=.1**.5;x,y=[rng.uniform(0,1-side) for _ in range(2)]
    random=use('a-patch-attack',area=.1,box=[x,y,x+side,y+side])
    searched=s.job('patch-heldout',model,env,[use('a-patch-attack',**best)],seeds,60)
    reference=s.job('patch-random-heldout',model,env,[random],seeds,60)
    s.compare('patch-search-heldout',reference,searched)

def td(s):
    model,env='tdmpc2-pushcube','maniskill-pushcube-state-independent'
    stat='native/plugins/stats/tdmpc2-pushcube-independent.json'
    calibration=s.job('calibration',model,env,[],list(range(1000,1050)),50)
    path,stats=s.fit([calibration],'vector')
    watch=use('a-input-monitor-vector',stats=str(path))
    seeds=list(range(2000,2050))
    base=s.job('baseline',model,env,[],seeds,50)
    neutral=[use('a-obs-noise-vector',c=0,stats=stat),use('b-memory-fault',mode='none'),watch]
    monitored=s.job('neutral',model,env,neutral,seeds,50);s.compare('neutral',base,monitored,True)
    for c in (.1,.3,1):
        fault=use('a-obs-noise-vector',c=c,stats=stat,start=20,end=50)
        variant=s.job('observation-'+str(c),model,env,[fault,watch],seeds,50)
        s.compare('observation-'+str(c),monitored,variant)
        s.detect('observation-'+str(c),monitored,variant,'a-input-monitor-vector','a-obs-noise-vector')
        variant=s.job('latent-'+str(c),model,env,[use('b-memory-fault',mode='noise',c=c,stats=stat,start=20)],seeds,50)
        s.compare('latent-'+str(c),base,variant)
    for t in (10,30):
        variant=s.job('wipe-'+str(t),model,env,[use('b-memory-fault',mode='wipe',steps=[t])],seeds,50)
        s.compare('wipe-'+str(t),base,variant)
    variant=s.job('stale',model,env,[use('b-memory-fault',mode='stale',k=3,start=20)],seeds,50)
    s.compare('stale',base,variant)

def memory(s):
    model='memoryvla-bridge';seeds=list(range(24));envs=['simpler-bridge-carrot','simpler-bridge-spoon']
    # First verify that enabling state probes with neutral cards preserves actions.
    plain=s.job('gate-plain',model,envs[0],[],[0,1],120)
    neutral=s.job('gate-neutral',model,envs[0],[use('b-memory-fault',mode='none'),use('b-write-gate',tau=None),
         use('a-obs-corruption',severity=0),use('a-camera-jitter',shift=0,rotate=0,scale=0),
         use('a-patch-attack',area=0),use('a-frame-guard',tau=None)],[0,1],120)
    s.compare('neutral',plain,neutral,True)
    calibrations=[s.job('calibration-'+env,model,env,[use('b-write-gate',tau=None,collect_moments=True)],seeds,120) for env in envs]
    path,stats=s.fit(calibrations,'image','3rd_view_camera',11)
    watch=use('a-input-monitor',stats=str(path),camera='3rd_view_camera')
    for env in envs:
        base=s.job('baseline-'+env,model,env,[watch],seeds,120)
        pure=s.job('pure-'+env,model,env,[],seeds,120);s.compare('monitor-neutral-'+env,pure,base,True)
        for kind in ('noise','brightness','contrast','color_bias'):
            for severity in (1,3,5):
                name=f'{env}-{kind}-{severity}';fault=use('a-obs-corruption',kind=kind,severity=severity,start=20)
                variant=s.job(name,model,env,[fault,watch],seeds,120);s.compare(name,base,variant)
                s.detect(name,base,variant,'a-input-monitor','a-obs-corruption')
        for mode in ('persistent','per_step'):
            for shift,rotate in ((.02,2),(.05,5),(.1,10)):
                name=f'{env}-{mode}-{shift}';fault=use('a-camera-jitter',mode=mode,shift=shift,rotate=rotate,scale=.05,start=20)
                variant=s.job(name,model,env,[fault,watch],seeds,120);s.compare(name,base,variant)
                s.detect(name,base,variant,'a-input-monitor','a-camera-jitter')
        for t in (10,30,60):
            name=f'{env}-wipe-{t}';variant=s.job(name,model,env,[use('b-memory-fault',mode='wipe',steps=[t])],seeds,120)
            s.compare(name,pure,variant)
        for k in (1,3,5):
            name=f'{env}-stale-{k}';variant=s.job(name,model,env,[use('b-memory-fault',mode='stale',k=k,start=20)],seeds,120)
            s.compare(name,pure,variant)
        for c in (.1,.3,1):
            name=f'{env}-memory-noise-{c}'
            variant=s.job(name,model,env,[use('b-memory-fault',mode='noise',c=c,stats=str(path),stats_field='memory',start=20)],seeds,120)
            s.compare(name,pure,variant)
        # Different policy: retain raw nomem control and report descriptively, do not claim identical model pairing.
        nomem=s.job('nomem-'+env,'memoryvla-bridge-nomem',env,[],seeds,120)
        s.tool('e_s12_report.py',pure,nomem,'--memory-ablation','--heldout-min-seed',12,'--output',s.folder/'reports'/('nomem-'+env))
        for t in (10,30,60):
            s.tool('e_s12_report.py',s.folder/'runs'/f'{env}-wipe-{t}'/'output',nomem,'--memory-ablation','--heldout-min-seed',12,
                   '--output',s.folder/'reports'/f'{env}-wipe-{t}-vs-nomem')
        for defense_name,defense in [('frame',use('a-frame-guard',tau=stats['frame_tau'])),('write',use('b-write-gate',tau=stats['write_tau']))]:
            alone=s.job(f'{env}-{defense_name}-clean',model,env,[defense,watch],seeds,120);s.compare(f'{env}-{defense_name}-cost',base,alone)
            for duration,end in [('burst',25),('persistent',None)]:
                fault=use('a-obs-corruption',kind='noise',severity=3,start=20,end=end)
                name=f'{env}-{defense_name}-{duration}'
                disturbed=s.job(f'{env}-noise-{duration}',model,env,[fault,watch],seeds,120)
                defended=s.job(name,model,env,[fault,defense,watch],seeds,120);s.compare(name,disturbed,defended)
            if defense_name=='frame':
                fault=use('a-camera-jitter',mode='persistent',shift=.05,rotate=5,scale=.05,start=20)
                defended=s.job(f'{env}-frame-jitter',model,env,[fault,defense,watch],seeds,120)
                s.compare(f'{env}-frame-jitter',s.folder/'runs'/f'{env}-persistent-0.05'/'output',defended)
    search=s.folder/'search'
    if not (search/'best.json').exists():
        s.tool('a_patch_search.py','--recipe','memoryvla-bridge-episode','--model',model,'--output',search)
    best=json.loads((search/'best.json').read_text())['best']['config'];heldout=list(range(12,24))
    if json.loads((search/'best.json').read_text()).get('state')!='completed':raise RuntimeError('patch search did not finish')
    found=s.job('patch-heldout',model,envs[0],[use('a-patch-attack',**best)],heldout,120)
    import random
    rng=random.Random(983);side=.1**.5;x,y=[rng.uniform(0,1-side) for _ in range(2)]
    reference=s.job('patch-random-heldout',model,envs[0],[use('a-patch-attack',area=.1,box=[x,y,x+side,y+side])],heldout,120)
    s.compare('patch-search-heldout',reference,found)
    for name,defense in [('frame',use('a-frame-guard',tau=stats['frame_tau'])),('write',use('b-write-gate',tau=stats['write_tau']))]:
        defended=s.job('patch-'+name,model,envs[0],[use('a-patch-attack',**best),defense],heldout,120)
        s.compare('patch-'+name,found,defended)

def main():
    global OUT
    import fcntl
    p=argparse.ArgumentParser(__doc__);p.add_argument('--phase',choices=['toy','td','memory'],required=True)
    p.add_argument('--output',default=str(OUT));a=p.parse_args()
    OUT=Path(a.output).resolve()
    if not OUT.is_relative_to(ROOT/'.workspace'):raise ValueError('acceptance output must be inside E/.workspace')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'suite.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=Suite(a.phase)
        try:
            globals()[a.phase](s)
            save(s.folder/'status.json',{'state':'completed','jobs':len(s.records),'episodes':sum(r['episodes'] for r in s.records.values())})
        except Exception as exc:
            save(s.folder/'status.json',{'state':'failed','error':str(exc),'jobs':len(s.records)})
            raise
if __name__=='__main__':main()
