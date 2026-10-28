"""Verify frozen clean statistics against their original evidence before G2."""
import hashlib
import json
from pathlib import Path
import subprocess
from wms.episodes.clean_stats import collect
from wms.engines.tdmpc2.native import episode_checkpoint

ROOT = Path(__file__).resolve().parents[1]
original = ROOT.parents[1] / 'deployments/wm-safetylab-dual-entry/runs/run_7e0320'
stats_path = ROOT / 'native/plugins/stats/tdmpc2-pushcube-independent.json'
config = json.loads((original / 'config.json').read_text())
old_checkpoint = episode_checkpoint({**config['model']['config'], 'output_dir': str(original / 'output')})
current_checkpoint = episode_checkpoint({'runtime_config': str(ROOT / 'native/tdmpc2/runtime.json')})
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
assert digest(old_checkpoint) == digest(current_checkpoint), 'checkpoint differs from clean-stat source'
assert config['environment']['config']['hazards']['fast_speed'] == 1.02
assert config['environment']['config']['reconfigure_each_episode']
assert config['seed'] == 1000 and config['params']['episodes'] == 50 and config['params']['max_steps'] == 50
saved = json.loads(stats_path.read_text())
rebuilt = collect([original / 'output'])
for key in ('observation_vector', 'action', 'latent', 'events'):
    assert saved[key] == rebuilt[key], f'clean statistics do not reproduce: {key}'
payload = {'source_run': str(original), 'checkpoint': str(current_checkpoint), 'checkpoint_sha256': digest(current_checkpoint),
           'stats_sha256': digest(stats_path), 'stats_reproduced': True, 'calibration_seeds': list(range(1000,1050)),
           'acceptance_seeds': list(range(2000,2050)), 'fast_speed': 1.02, 'threshold_version': 'pushcube-fast-contact-2026-10-18',
           'runtime_sha256': digest(ROOT / 'native/tdmpc2/runtime.json'),
           'base_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'src/wms/plugins/cards').glob('*.py'))}}
out = ROOT / '.workspace/s34/provenance.json'
out.write_text(json.dumps(payload,indent=2))
print(json.dumps(payload,indent=2))
