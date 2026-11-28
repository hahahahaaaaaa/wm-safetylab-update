"""Reuse the pinned visual sublab adapter with E-owned execution and outputs."""
import sys
from wms.engines.native_tasks import _runtime


def run(config, output_dir, context):
    runtime = _runtime(config)
    sys.path.insert(0, runtime["source_dir"] + "/src")
    from wms_sublab.tasks.libero import run as upstream_run
    inputs = {**runtime.get("input", {}), **config.get("inputs", {})}
    return upstream_run({**config, "inputs": inputs}, output_dir, context)
