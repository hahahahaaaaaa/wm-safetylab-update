"""After the matrix, verify repeated clean/c=1 arms across experiments on all 50 seeds."""
import json
from pathlib import Path
from s34_compare import compare

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'.workspace/s34'
cases = json.loads((folder/'matrix-formal.json').read_text())['cases']
required = {'10-dose','11-p90','11-p99','9-detection','8-dose','g3-combination'}
assert set(cases) == required, 'formal matrix is incomplete'
runs = ROOT/'.workspace/store/runs'
baseline = next(r['run_id'] for r in cases['10-dose']['runs'] if r['arm']=='A0')
checks = []
for key in sorted(required-{'10-dose'}):
    other = next(r['run_id'] for r in cases[key]['runs'] if r['arm']=='A0')
    checks.append(compare(runs,baseline,other,True,range(2000,2050)))
fault = next(r['run_id'] for r in cases['10-dose']['runs'] if r['arm']=='A2')
for key in ('11-p90','11-p99'):
    other = next(r['run_id'] for r in cases[key]['runs'] if r['arm']=='A1')
    checks.append(compare(runs,fault,other,True,range(2000,2050)))
for left_case,left_arm,right_arm in [('8-dose','A8','A1'),('11-p90','A3','A3')]:
    left = next(r['run_id'] for r in cases[left_case]['runs'] if r['arm']==left_arm)
    right = next(r['run_id'] for r in cases['g3-combination']['runs'] if r['arm']==right_arm)
    checks.append(compare(runs,left,right,True,range(2000,2050)))
(folder/'gate-repeated-arms.json').write_text(json.dumps({'passed':True,'comparisons':checks},indent=2))
print('PASS: repeated clean/monitor/c=1/noise/defense arms match full trajectories on all 50 seeds')
