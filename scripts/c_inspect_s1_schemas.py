#!/usr/bin/env python3
"""Read-only schema inspector for steps.jsonl, episodes.json and trace.jsonl."""
import argparse
import json
from pathlib import Path


def episode_dirs(run):
    run = Path(run)
    for root in (run / "output" / "episodes", run / "episodes", run):
        if root.is_dir():
            found = sorted(p for p in root.glob("ep_*") if p.is_dir())
            if found:
                return found
    return []


def shape(value, depth=0):
    if isinstance(value, dict):
        return {k: shape(v, depth + 1) for k, v in list(value.items())[:30]} if depth < 2 else "object"
    if isinstance(value, list):
        return {"type": "array", "length": len(value), "item": shape(value[0], depth + 1) if value else None}
    return type(value).__name__


def first_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories")
    args = ap.parse_args()
    for run in args.runs:
        run = Path(run)
        print(f"\n===== RUN {run} =====")
        episodes_json = next((p for p in (run / "output" / "episodes.json", run / "episodes.json") if p.is_file()), None)
        if episodes_json:
            data = json.loads(episodes_json.read_text(encoding="utf-8"))
            print("episodes.json:", json.dumps(shape(data), ensure_ascii=False))
        else:
            print("episodes.json: MISSING")
        dirs = episode_dirs(run)
        if not dirs:
            print("episode dirs: MISSING")
        for ep in dirs[:1]:
            for name in ("steps.jsonl", "trace.jsonl"):
                path = ep / name
                if path.is_file():
                    sample = first_jsonl(path)
                    print(f"{ep.name}/{name}:", json.dumps(shape(sample), ensure_ascii=False))
                else:
                    print(f"{ep.name}/{name}: MISSING")


if __name__ == "__main__":
    main()
