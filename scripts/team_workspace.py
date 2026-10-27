"""Run a developer checkout with shared dependencies and private writable state."""
from pathlib import Path
import json
import os
import subprocess
import sys
import re
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((ROOT / '.workspace/settings.json').read_text())
ENV = {**os.environ, **SETTINGS['environment']}
ENV['PYTHONPATH'] = str(ROOT / 'src')
ENV['WMS_BASE_URL'] = 'http://127.0.0.1:' + str(SETTINGS['port'])
ENV['WMS_STORAGE_ROOT'] = str(ROOT / '.workspace/store')
PYTHON = SETTINGS['control_python']

def execute(command):
    os.chdir(ROOT)
    os.execve(command[0], command, ENV)

def main():
    args = sys.argv[1:]
    action = args.pop(0) if args else 'info'
    if action == 'info':
        print(json.dumps({'workspace': str(ROOT), 'developer': SETTINGS['developer'],
                         'branch': subprocess.check_output(['git', '-C', str(ROOT), 'branch', '--show-current'], text=True).strip(),
                         'url': ENV['WMS_BASE_URL'], 'store': ENV['WMS_STORAGE_ROOT'],
                         'direct_runs': str(ROOT / '.workspace/runs'), 'control_python': PYTHON,
                         'native_configs': SETTINGS['native_configs']}, indent=2, ensure_ascii=False))
    elif action == 'serve':
        if args:
            raise SystemExit('serve uses the workspace port and storage; edit .workspace/settings.json to change them')
        execute([PYTHON, '-m', 'wms.app.cli', 'serve', '--host', '127.0.0.1',
                 '--port', str(SETTINGS['port']), '--storage-root', ENV['WMS_STORAGE_ROOT'],
                 '--worker-python', PYTHON, '--native-configs', str(ROOT / 'native-configs.local.json')])
    elif action == 'run':
        if len(args) < 2:
            raise SystemExit('Usage: ./dev.sh run RECIPE UNIQUE_RUN_ID [key=json ...]')
        if any(x == '--runs-dir' or x.startswith('--runs-dir=') for x in args):
            raise SystemExit('The workspace output directory is fixed by dev.sh')
        if Path(args[1]).name != args[1] or args[1] in ('.', '..'):
            raise SystemExit('RUN_ID must be a single directory name')
        execute([PYTHON, str(ROOT / 'scripts/run_direct.py'), *args,
                 '--runs-dir', str(ROOT / '.workspace/runs')])
    elif action == 'plugin-test':
        if not args:
            raise SystemExit('Usage: ./dev.sh plugin-test PLUGIN [options]')
        label = re.sub(r'[^A-Za-z0-9_.-]+', '_', args[0])
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = ROOT / '.workspace/results' / f'plugin-test-{label}-{stamp}-{uuid4().hex[:8]}'
        print('Plugin test output:', output, flush=True)
        execute([PYTHON, '-m', 'wms.app.cli', 'plugin', 'test', *args,
                 '--output', str(output)])
    elif action in ('python', 'toy-python'):
        execute([PYTHON if action == 'python' else SETTINGS['toy_python'], *args])
    elif action == 'wms':
        if args and args[0] == 'serve':
            raise SystemExit('Use ./dev.sh serve to retain workspace isolation')
        execute([PYTHON, '-m', 'wms.app.cli', *args])
    else:
        raise SystemExit('Commands: info, serve, run, plugin-test, python, toy-python, wms')

if __name__ == '__main__':
    main()
