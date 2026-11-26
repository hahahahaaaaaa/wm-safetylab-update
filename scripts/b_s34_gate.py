"""Strict full-trajectory gate on neutral arms and a separate monitor-only run."""
import json
from pathlib import Path
from s34_compare import compare

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / '.workspace/s34'
manifest = json.loads((folder / 'matrix-neutral.json').read_text())
runs = manifest['cases']['neutral']['runs']
baseline = next(r['run_id'] for r in runs if r['arm'] == 'A0')
others = [r['run_id'] for r in runs if r['arm'] != 'A0']
others += [r['run_id'] for r in manifest['cases']['monitor-neutral']['runs']]
expected = manifest['cases']['neutral']['config']['seeds']
seeds = list(range(expected['start'], expected['start'] + expected['episodes']))
rows = [compare(ROOT / '.workspace/store/runs', baseline, other, True, seeds) for other in others]
(folder / 'gate-neutral.json').write_text(json.dumps({'passed':True,'seeds':seeds,'comparisons':rows},indent=2))
print('PASS full-trajectory neutral/monitor gate:',len(rows),'comparisons,',len(seeds),'seeds each')
