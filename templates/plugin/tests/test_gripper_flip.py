"""Run with a Python that has numpy, from the platform root:

    $env:PYTHONPATH = "src;templates/plugin"
    python -m pytest templates/plugin/tests
"""
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from wms.core.registry import Registry
from wms.plugins.conformance import check

REGISTRY = Path(__file__).resolve().parents[1] / "registry.json"


def test_contract(tmp_path):
    entry = Registry(paths=[REGISTRY]).items["plugins"]["template-gripper-flip"]
    spec = {"id": "template-gripper-flip", "entrypoint": entry["entrypoint"], "config": {"p": 0.3},
            "declared": {"stage": entry["stage"], "role": entry["role"]}}
    report = check(spec, tmp_path)
    assert not [row for row in report if row[0] == "FAIL"], report


def test_flips_only_the_gripper():
    from wms_plugin_template.gripper_flip import GripperFlip

    class Ctx:
        t = 0
        rng = __import__("numpy").random.default_rng(0)
        logged = []

        def log(self, **fields):
            self.logged.append(fields)

    plugin = GripperFlip({"p": 1.0})
    plugin.reset({"index": 0, "seed": 0}, Ctx())
    action = [0.1, 0.2, 1.0]
    assert plugin.on_action(action, Ctx()) == [0.1, 0.2, -1.0]
    assert action == [0.1, 0.2, 1.0]                     # the input is untouched
