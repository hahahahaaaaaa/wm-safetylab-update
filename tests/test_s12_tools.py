import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from a_detection_eval import auc

def test_auc_ranking_ties_and_missing_class():
    assert auc([0,0,1,1],[0,1,2,3])==1
    assert auc([0,1],[1,1])==.5
    assert auc([0,1],[1,0])==0
    assert auc([1,1],[0,1]) is None

def test_registered_write_gate_phases():
    from wms.core.registry import Registry
    from wms.plugins.resolve import check
    registry=Registry()
    config=registry.resolve({'recipe':'toy-episode-occlusion','model':'toy-centroid-memory',
        'params':{'plugins':[{'plugin':'b-write-gate'}]}})
    assert check(config)==[]
    with pytest.raises(ValueError,match='state phases'):
        registry.resolve({'recipe':'toy-episode-occlusion','model':'toy-planner',
            'params':{'plugins':[{'plugin':'b-write-gate'}]}})


def run_fixture(root,name,memory=True):
    import json
    out=root/name;out.mkdir()
    rows=[]
    for i,seed in enumerate((0,12)):
        folder=out/'episodes'/f'ep_{i:03d}';folder.mkdir(parents=True)
        rows.append({'episode':i,'seed':seed,'steps':2,'success':bool(i),'hazard_steps':{}})
        (folder/'steps.jsonl').write_text(''.join(json.dumps({'t':t,'action':[0.]})+'\n' for t in range(2)))
        (folder/'trace.jsonl').write_text(''.join(json.dumps({'t':t,'events':[
            {'plugin':'0:monitor','data':{'score':float(t),'threshold':.5}},
            {'plugin':'1:fault','data':{'injected':bool(t)}},
            {'plugin':'2:guard','role':'defense','data':{'replaced':bool(t)}}]})+'\n' for t in range(2)))
    for file,data in [('result.json',{'status':'completed'}),('episodes.json',{'episodes':rows}),
                      ('run_manifest.json',{'seeds':[0,12],'model':{'entrypoint':'adapter','config':{'memory':memory}},
                        'environment':'env','params':{},'packages':{},'inputs':{}})]:
        (out/file).write_text(json.dumps(data))
    return out


def test_memory_ablation_is_explicit_and_heldout_is_separate(tmp_path):
    from e_s12_report import report
    a=run_fixture(tmp_path,'a');b=run_fixture(tmp_path,'b',False)
    with pytest.raises(ValueError,match='model'):report(a,b)
    result=report(a,b,memory_ablation=True,heldout_min_seed=12)
    assert result['episodes']==[2,2]
    assert result['heldout']['seeds']==[12]
    assert result['heldout']['success_difference']['n']==1
    assert result['heldout']['defense_events']['baseline']['2:guard']['intervention_rate']==.5
    with pytest.raises(ValueError,match='differ only'):report(a,a,memory_ablation=True)


def test_detector_requires_complete_trace_and_filters_calibration(tmp_path):
    from a_detection_eval import evaluate
    a=run_fixture(tmp_path,'a')
    result=evaluate([a],'monitor','fault',bootstrap=10,min_seed=12)
    assert result['auroc']==1 and len(result['episodes'])==1
    assert result['episodes'][0]['seed']==12
    (a/'episodes/ep_001/trace.jsonl').write_text('')
    with pytest.raises(ValueError,match='trace steps'):
        evaluate([a],'monitor','fault',bootstrap=10,min_seed=12)
