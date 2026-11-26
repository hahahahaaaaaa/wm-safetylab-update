"""Platform task: SimplerEnv-Bridge evaluation of an official MemoryVLA checkpoint."""
from __future__ import annotations

from pathlib import Path


def _result(output, data):
    import mimetypes
    files = []
    for path in sorted(Path(output).rglob("*")):
        if not path.is_file() or path.name in {"result.json", "cancel.flag"} or path.suffix == ".png":
            continue
        kind = "video" if path.suffix == ".mp4" else "evidence"
        files.append({"path": path.relative_to(output).as_posix(), "kind": kind,
                      "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"})
    return {"data": data, "outputs": files}


def simpler_evaluate(config, output_dir, context):
    from wms.engines.native_tasks import _runtime
    from .simpler import evaluate
    runtime = _runtime(config)
    if not runtime:
        raise ValueError("Select a MemoryVLA runtime.config_file")
    summary = evaluate(runtime, dict(config.get("params", {})), Path(output_dir), context,
                       seed=int(config.get("seed", 0)))
    return _result(output_dir, summary)
