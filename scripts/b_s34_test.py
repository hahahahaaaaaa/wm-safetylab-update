"""Use shared pure-Python pytest with the model runtime's NumPy; install nothing."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
# Shared control environment contains pytest; append so the active runtime wins.
control = root.parents[1] / 'deployments/wm-safetylab/.venv/lib'
for path in control.glob('python*/site-packages'):
    sys.path.append(str(path))
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
