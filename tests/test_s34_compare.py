import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('s34_compare', Path(__file__).parents[1] / 'scripts/s34_compare.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(root, name, seeds=(1,2)):
    out = root / name / 'output'
    out.mkdir(parents=True)
    rows = []
    for i,seed in enumerate(seeds):
        folder = out / 'episodes' / f'ep_{i:03d}'
        folder.mkdir(parents=True)
        (folder/'steps.jsonl').write_text(json.dumps({'t':0,'action':[1.,2.]})+'\n')
        rows.append({'episode':i,'seed':seed,'steps':1,'success':True,'hazard_steps':{}})
    for file,value in [('result.json',{'status':'completed'}),('episodes.json',{'episodes':rows}),
                        ('run_manifest.json',{'seeds':list(seeds),'params':{},'environment':{'name':'env'}})]:
        (out/file).write_text(json.dumps(value))
    return out


def test_matches_seed_not_position(tmp_path):
    fixture(tmp_path,'a')
    fixture(tmp_path,'b',(2,1))
    assert m.compare(tmp_path,'a','b',True)['full_trajectory_equal']


@pytest.mark.parametrize('fault',['missing','duplicate','nonfinite','shape','length','failed','environment'])
def test_invalid_pairs_cannot_pass(tmp_path,fault):
    fixture(tmp_path,'a')
    out = fixture(tmp_path,'b')
    path = out/'episodes/ep_001/steps.jsonl'
    if fault in ('missing','duplicate'):
        data = json.loads((out/'episodes.json').read_text())
        if fault == 'missing': data['episodes'].pop()
        else: data['episodes'][1]['seed']=1
        (out/'episodes.json').write_text(json.dumps(data))
    elif fault == 'failed':
        (out/'result.json').write_text('{"status":"failed"}')
    elif fault == 'environment':
        data = json.loads((out/'run_manifest.json').read_text())
        data['environment']['name']='other'
        (out/'run_manifest.json').write_text(json.dumps(data))
    else:
        action = [float('nan'),2] if fault == 'nonfinite' else [1] if fault == 'shape' else [1,2]
        path.write_text('' if fault == 'length' else json.dumps({'t':0,'action':action})+'\n')
    with pytest.raises(ValueError):
        m.compare(tmp_path,'a','b',True)
