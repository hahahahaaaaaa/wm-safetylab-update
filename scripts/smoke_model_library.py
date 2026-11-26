"""Serial, resumable integration smoke tests through the actual HTTP API."""
import argparse, json, time, urllib.request, urllib.error
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=8785); p.add_argument('--recipes',nargs='*'); p.add_argument('--report',required=True)
a=p.parse_args(); base=f'http://127.0.0.1:{a.port}/api/v1'; report=Path(a.report); report.parent.mkdir(parents=True,exist_ok=True)
def api(path,body=None):
    req=urllib.request.Request(base+path, data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
ids=a.recipes or ['demo-predict','toy-episode-occlusion','native-dreamer','native-renew','native-maniskill','state-wm-maniskill','tdmpc2-pushcube-episode','embodied-jev-baseline','native-cosmos','native-modelscope','native-wan','predict-rlvr_video','predict-rlvr_language','native-openvla','openvla-bridge-episode','memoryvla-bridge-episode','native-cosmos-policy']
if not a.recipes:
    ids += ['openvla-libero-spatial','tdmpc2-pushcube-center-eval','predict-dreamer','embodied-jev-chat','embodied-jev-jev']
rows=json.loads(report.read_text()) if report.exists() else []
terminal={'completed','failed','cancelled','interrupted'}
def save():report.write_text(json.dumps(rows,indent=2,ensure_ascii=False))
for rid in ids:
    if any(r['recipe']==rid and r.get('finished') for r in rows):continue
    row=next((r for r in rows if r['recipe']==rid and not r.get('finished')),None)
    if row is None:
        item=next(i for i in api('/model-library')['items'] if i['id']==rid)
        row={'recipe':rid,'started':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}; rows.append(row)
        if not item['ready']:
            row.update(state='blocked',issues=item['issues'],finished=True);save();print(json.dumps(row),flush=True);continue
        body={**item['template'],'name':'E model integration smoke: '+rid,'timeout_s':1800}
        try:
            run=api('/model-library/'+rid+'/runs',body);row['run_id']=run['id'];save();print('START',rid,run['id'],flush=True)
        except urllib.error.HTTPError as e:
            row.update(state='submission_failed',error=e.read().decode(),finished=True);save();print(json.dumps(row),flush=True);continue
    while True:
        run=api('/runs/'+row['run_id'])
        if run['state'] in terminal:break
        time.sleep(5)
    evaluation_id=run.get('automatic_evaluation_id')
    if evaluation_id:
        while True:
            ev=api('/runs/'+evaluation_id)
            if ev['state'] in terminal:break
            time.sleep(2)
        row['evaluation']={'id':evaluation_id,'state':ev['state'],'result':ev.get('result')}
    result=run.get('result') or {}
    row.update(state=run['state'],error=run.get('error'),outputs=result.get('outputs',[]),data=result.get('data'),finished=True)
    save();print('END',rid,run['state'],str(run.get('error'))[:300],flush=True)
