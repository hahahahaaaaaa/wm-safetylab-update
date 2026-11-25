"""Run all S1/S2 contracts plus active defense and all memory-fault modes."""
import json
from pathlib import Path
from wms.plugins.conformance import _spec, check
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.workspace/s12/conformance'
OUT.mkdir(parents=True,exist_ok=True)
cases=[('a-obs-corruption',{}),('a-obs-noise-vector',{}),('a-camera-jitter',{}),
 ('a-patch-attack',{}),('a-frame-guard',{'tau':10}),('a-input-monitor',{}),('a-input-monitor-vector',{}),
 ('b-memory-fault',{'mode':'wipe','start':4}),('b-memory-fault',{'mode':'stale','k':3}),
 ('b-memory-fault',{'mode':'noise','std':.1}),('b-write-gate',{'tau':.1})]
results=[]
for index,(name,config) in enumerate(cases):
    folder=OUT/(str(index)+'-'+name)
    folder.mkdir(parents=True,exist_ok=True)
    report=check(_spec(name,config,[]),folder)
    passed=not any(row[0]=='FAIL' for row in report)
    results.append({'plugin':name,'config':config,'passed':passed,'checks':report})
    (OUT/'report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(name,config,'PASS' if passed else 'FAIL',flush=True)
    if not passed:
        print(report,flush=True)
raise SystemExit(0 if all(r['passed'] for r in results) else 1)
