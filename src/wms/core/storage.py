"""Small local run store: one SQLite row and one working directory per run.

The previous records.sqlite3/artifacts/jobs layout is left untouched. New runs
live in runs.sqlite3 and runs/<id>; there are no snapshots or artifact copies.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self._db = sqlite3.connect(self.root / "runs.sqlite3", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        self._db.commit()

    def _prepare(self, config):
        config = deepcopy(config)
        encoded_config = json.dumps(config, ensure_ascii=False, indent=2)
        run_id = "run_" + uuid4().hex[:16]
        now = utcnow()
        run = {
            "id": run_id,
            "name": config.get("name", config.get("task", "experiment")),
            "task": config.get("task", "native"),
            "direction": config.get("direction", ""),
            "state": "queued",
            "created_at": now,
            "updated_at": now,
            "config": config,
            "result": None,
            "error": None,
            "record_version": 3,
        }
        if config.get("source_run"):
            run["source_run"] = config["source_run"]
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True)
        self.output_dir(run_id).mkdir()
        (directory / "config.json").write_text(encoded_config, encoding="utf-8")
        return run

    def create(self, config):
        run = self._prepare(config)
        with self.lock, self._db:
            self._db.execute("INSERT INTO runs VALUES (?, ?)", (run["id"], json.dumps(run, ensure_ascii=False)))
        return deepcopy(run)

    def finish(self, run_id, fields, evaluation_config=None):
        """Commit completion and its optional automatic evaluation atomically."""
        with self.lock, self._db:
            run = self.get(run_id)
            evaluation = None
            if evaluation_config and not run.get("automatic_evaluation_id"):
                evaluation = self._prepare(evaluation_config)
                self._db.execute("INSERT INTO runs VALUES (?, ?)",
                                 (evaluation["id"], json.dumps(evaluation, ensure_ascii=False)))
                run["automatic_evaluation_id"] = evaluation["id"]
            run.update(deepcopy(fields), updated_at=utcnow())
            self._db.execute("UPDATE runs SET body=? WHERE id=?", (json.dumps(run, ensure_ascii=False), run_id))
        return run, evaluation

    def get(self, run_id):
        with self.lock:
            row = self._db.execute("SELECT body FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row[0])

    def list_runs(self):
        with self.lock:
            rows = self._db.execute("SELECT body FROM runs ORDER BY rowid DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def list_workbench_runs(self, limit=200):
        """Bound model-workbench polling independently of large red-team stores."""
        with self.lock:
            rows = self._db.execute(
                "SELECT body FROM runs WHERE json_extract(body, '$.config.workbench_recipe') IS NOT NULL "
                "ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def update(self, run_id, **fields):
        with self.lock:
            run = self.get(run_id)
            run.update(deepcopy(fields))
            run["id"] = run_id
            run["updated_at"] = utcnow()
            # result.json belongs to the worker, never to score bookkeeping.
            self._db.execute("UPDATE runs SET body = ? WHERE id = ?", (json.dumps(run, ensure_ascii=False), run_id))
            self._db.commit()
        return run

    def run_dir(self, run_id):
        if not isinstance(run_id, str) or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
            raise ValueError("Invalid run id")
        return self.root / "runs" / run_id

    def output_dir(self, run_id):
        return self.run_dir(run_id) / "output"

    def close(self):
        with self.lock:
            self._db.close()

