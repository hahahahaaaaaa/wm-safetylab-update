"""Clean statistics for plugin units and thresholds (made by python -m wms.episodes.clean_stats).

    from wms.plugins.stats import clean_stats
    stats = clean_stats(ctx.config.get("stats", "tdmpc2-pushcube-independent"))
    std = stats["latent"]["std"]                              # per latent dimension
    threshold = stats["events"]["monitor-surprise.surprise"]["p99"]

A name is a file in native/plugins/stats/ (without .json); anything else is a path. The
files say which clean runs they come from ("runs"). Read them in reset(), not per step.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

STATS_DIR = Path(__file__).resolve().parents[3] / "native" / "plugins" / "stats"


@lru_cache(maxsize=16)
def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_stats(name_or_path):
    path = STATS_DIR / f"{name_or_path}.json"
    if not path.is_file():
        path = Path(name_or_path)
    if not path.is_file():
        known = sorted(p.stem for p in STATS_DIR.glob("*.json"))
        raise FileNotFoundError(f"no clean statistics {name_or_path!r}; known: {known}")
    return _load(str(path.resolve()))
