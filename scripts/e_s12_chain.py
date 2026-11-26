"""Single sequential real-model acceptance chain; no other workspace processes touched."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'.workspace/s12/acceptance-final'
if not out.is_relative_to(ROOT/'.workspace'):raise ValueError('output must be inside E/.workspace')
out.mkdir(parents=True,exist_ok=True)
states={}
for phase in ('td','memory'):
    states[phase]={'state':'running','started':time.time()}
    (out/'chain.json').write_text(json.dumps(states,indent=2))
    with (out/(phase+'.log')).open('a') as log:
        result=subprocess.run([str(ROOT/'dev.sh'),'python','scripts/e_s12_acceptance.py',
            '--phase',phase,'--output',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    states[phase].update(state='completed' if result.returncode==0 else 'failed',finished=time.time())
    (out/'chain.json').write_text(json.dumps(states,indent=2))
    if result.returncode:raise SystemExit(result.returncode)
