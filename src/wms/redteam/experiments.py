"""Red-team experiments: one episode-kernel run per (arm, environment), all with the same seeds.

Experiments and drafts are JSON files under <storage root>/redteam; the runs themselves
are ordinary runs of the run store (runs.sqlite3, runs/<id>), so logs, files, cancel and
retry work as for any other run. Results are read from each run's output directory:
episodes/ep_XXX/summary.json (written when the episode ends), steps.jsonl, trace.jsonl and
frames_<camera>.npz.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import threading
from uuid import uuid4
import zipfile
import zlib

from wms.redteam import stats
from wms.redteam.catalog import load_catalog
from wms.redteam.schema import (ArmStats, Delta, EpisodeDetail, EpisodePlugin, EpisodeRow, Experiment, ExperimentConfig,
                                ExperimentRun, Interval, Matrix, MatrixCell, MatrixRow, OutcomeRow, Report, Trace,
                                TraceStep)
from wms.redteam.validation import validate_config

ACTIVE = ("queued", "running", "cancelling")
STAGES = ("ENV", "S1", "S2", "S3", "S4", "S5", "S6")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def _id_ok(value):
    return isinstance(value, str) and 0 < len(value) <= 64 and all(c.isalnum() or c in "_-" for c in value)


class Experiments:
    def __init__(self, store, runner, registry):
        self.store, self.runner, self.registry = store, runner, registry
        self.root = Path(store.root) / "redteam"
        for name in ("experiments", "drafts"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    # ---------- files ----------

    def _path(self, kind, item_id):
        if not _id_ok(item_id):
            raise ValueError(f"Invalid id: {item_id!r}")
        return self.root / kind / f"{item_id}.json"

    def _read(self, kind, item_id):
        path = self._path(kind, item_id)
        if not path.is_file():
            raise KeyError(item_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, kind, item_id, value):
        path = self._path(kind, item_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def catalog(self):
        return load_catalog(self.registry)

    def validate(self, config):
        return validate_config(config, self.catalog(), self.registry)

    # ---------- drafts ----------

    def get_draft(self, draft_id):
        return self._read("drafts", draft_id)

    def put_draft(self, draft_id, config):
        value = {"id": draft_id, "config": config.model_dump(), "updated_at": utcnow()}
        self._write("drafts", draft_id, value)
        return value

    # ---------- experiments ----------

    def submit(self, config: ExperimentConfig):
        result = self.validate(config)
        if not result.valid:
            errors = "; ".join(i.message for i in result.issues if i.severity == "error")
            raise ValueError(f"配置有错误：{errors}")
        experiment_id = "rt_" + uuid4().hex[:12]
        resolved = [self.registry.resolve(recipe) for recipe in result.recipes]
        runs = []
        for recipe, run_config, (arm, env) in zip(result.recipes, resolved, _pairs(result.arms, config.environments)):
            run_config["tags"] = list(run_config.get("tags") or []) + [experiment_id]
            run = self.runner.submit(run_config)
            runs.append({"arm": arm.id, "environment": env, "run_id": run["id"]})
        now = utcnow()
        value = {"id": experiment_id, "name": config.name, "created_at": now, "updated_at": now,
                 "config": config.model_dump(), "arms": [a.model_dump() for a in result.arms], "runs": runs}
        with self.lock:
            self._write("experiments", experiment_id, value)
        return self.get(experiment_id)

    def _run_state(self, item, episodes):
        try:
            run = self.store.get(item["run_id"])
        except KeyError:
            return ExperimentRun(**item, state="failed", episodes_total=episodes, error="运行记录不存在")
        done = len(list(self.store.output_dir(item["run_id"]).glob("episodes/ep_*/summary.json")))
        return ExperimentRun(**item, state=run["state"], episodes_done=min(done, episodes), episodes_total=episodes,
                             error=run.get("error"))

    def get(self, experiment_id):
        value = self._read("experiments", experiment_id)
        episodes = value["config"]["seeds"]["episodes"]
        runs = [self._run_state(item, episodes) for item in value["runs"]]
        states = [r.state for r in runs]
        if any(s == "cancelling" for s in states):
            state = "cancelling"
        elif any(s == "running" for s in states) or (any(s == "queued" for s in states) and not all(s == "queued" for s in states)):
            state = "running"
        elif all(s == "queued" for s in states):
            state = "queued"
        elif all(s == "completed" for s in states):
            state = "completed"
        elif any(s == "completed" for s in states):
            state = "partial_failed"
        elif any(s == "cancelled" for s in states) and not any(s in ("failed", "interrupted") for s in states):
            state = "cancelled"
        else:
            state = "failed"
        return Experiment(id=value["id"], name=value["name"], created_at=value["created_at"],
                          updated_at=value["updated_at"], state=state, config=ExperimentConfig(**value["config"]),
                          arms=value["arms"], runs=runs, episodes_done=sum(r.episodes_done for r in runs),
                          episodes_total=sum(r.episodes_total for r in runs))

    def list(self):
        paths = sorted((self.root / "experiments").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [self.get(p.stem) for p in paths]

    def cancel(self, experiment_id):
        for item in self._read("experiments", experiment_id)["runs"]:
            try:
                self.runner.cancel(item["run_id"])
            except KeyError:
                pass
        return self.get(experiment_id)

    def logs(self, experiment_id, run_id, offset=0):
        if run_id not in {item["run_id"] for item in self._read("experiments", experiment_id)["runs"]}:
            raise KeyError(run_id)
        path = self.store.run_dir(run_id) / "run.log"
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        start = min(max(offset, 0), len(text))
        chunk = text[start:start + 200000]
        return {"run_id": run_id, "text": chunk, "next_offset": start + len(chunk)}

    # ---------- episodes ----------

    def _episode_dirs(self, item):
        return sorted(self.store.output_dir(item["run_id"]).glob("episodes/ep_*/summary.json"))

    def rows(self, experiment_id):
        value = self._read("experiments", experiment_id)
        out = []
        for item in value["runs"]:
            for path in self._episode_dirs(item):
                out.append(_row(item, json.loads(path.read_text(encoding="utf-8"))))
        return value, out

    def episodes(self, experiment_id, arm=None, environment=None, success=None, end_reason=None, hazard=None,
                 cursor=None, limit=100):
        _, rows = self.rows(experiment_id)
        rows = [r for r in rows if (arm is None or r.arm == arm) and (environment is None or r.environment == environment)
                and (success is None or r.success == success) and (end_reason is None or r.end_reason == end_reason)
                and (hazard is None or r.hazard_steps.get(hazard))]
        start = int(cursor or 0)
        page = rows[start:start + limit]
        return {"items": page, "total": len(rows), "next_cursor": str(start + limit) if start + limit < len(rows) else None}

    def _locate(self, experiment_id, episode_id):
        value = self._read("experiments", experiment_id)
        run_id, _, index = episode_id.partition(":")
        item = next((i for i in value["runs"] if i["run_id"] == run_id), None)
        if item is None or not index.isdigit():
            raise KeyError(episode_id)
        directory = self.store.output_dir(run_id) / "episodes" / f"ep_{int(index):03d}"
        if not (directory / "summary.json").is_file():
            raise KeyError(episode_id)
        return value, item, directory

    def episode(self, experiment_id, episode_id):
        value, item, directory = self._locate(experiment_id, episode_id)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        row = _row(item, summary)
        baseline = None
        if item["arm"] != "A0":
            twin = next((i for i in value["runs"] if i["arm"] == "A0" and i["environment"] == item["environment"]), None)
            if twin:
                for path in self._episode_dirs(twin):
                    other = json.loads(path.read_text(encoding="utf-8"))
                    if other.get("seed") == row.seed:
                        baseline = f"{twin['run_id']}:{other['episode']}"
                        break
        arm = next(a for a in value["arms"] if a["id"] == item["arm"])
        uses = {u["key"]: u for u in value["config"]["plugins"]}
        plugins = []
        for key in arm["plugins"]:
            use = uses.get(key, {})
            entry = self.registry.items["plugins"].get(use.get("plugin") or "", {})
            plugins.append(EpisodePlugin(key=key, id=use.get("plugin") or use.get("entrypoint") or key,
                                         stage=entry.get("stage"), role=entry.get("role"),
                                         config={**(entry.get("config") or {}), **(use.get("config") or {})}))
        return EpisodeDetail(row=row, baseline=baseline, initial_state=summary.get("initial_state") or {},
                             final_state=summary.get("final_state") or {}, events=summary.get("events") or [],
                             frames=summary.get("frames") or {}, plugins=plugins)

    def trace(self, experiment_id, episode_id, start=0, end=None):
        _, _, directory = self._locate(experiment_id, episode_id)
        steps = _jsonl(directory / "steps.jsonl")
        trace = {row.get("t"): row for row in _jsonl(directory / "trace.jsonl")}
        total = len(steps)
        start = max(0, start)
        end = total if end is None else min(total, end)
        out = []
        for step in steps[start:end]:
            extra = trace.get(step.get("t"), {})
            out.append(TraceStep(
                t=step.get("t", 0), action=step.get("action"), action_proposed=extra.get("action_proposed"),
                reward=step.get("reward"), state_before=step.get("state_before") or {},
                state_after=step.get("state_after") or {}, policy_info=step.get("policy_info") or {},
                hazards=extra.get("hazards") or [],
                plugins={stage: [_key(n) for n in names] for stage, names in (extra.get("plugins") or {}).items()},
                events=[{**e, "plugin": _key(e.get("plugin"))} for e in extra.get("events") or []],
                takeover=extra.get("takeover"),
                env_events=extra.get("env_events") or step.get("events") or []))
        return Trace(episode=episode_id, start=start, end=end, total=total, steps=out)

    def frame_png(self, experiment_id, episode_id, camera, index, raw=False):
        _, _, directory = self._locate(experiment_id, episode_id)
        if not _id_ok(camera):
            raise ValueError("Invalid camera")
        path = directory / f"{'raw' if raw else 'frames'}_{camera}.npz"
        if not path.is_file():
            raise KeyError(f"{camera} frames")
        return _npz_frame_png(path, index)

    # ---------- report and matrix ----------

    def report(self, experiment_id):
        experiment = self.get(experiment_id)
        value, rows = self.rows(experiment_id)
        by_arm = {}
        for row in rows:
            by_arm.setdefault(row.arm, {})[(row.environment, row.seed)] = row
        base = by_arm.get("A0", {})
        out = []
        for arm in experiment.arms:
            mine = by_arm.get(arm.id, {})
            items = list(mine.values())
            n = len(items)
            rate, low, high = stats.wilson(sum(r.success for r in items), n)
            kinds = sorted({k for r in items for k in r.hazard_steps})
            stat = ArmStats(
                arm=arm.id, name=arm.name, kind=arm.kind, oracle=arm.oracle, episodes=n,
                success=Interval(estimate=rate, low=low, high=high),
                hazard_steps_mean=sum(r.hazard_total for r in items) / n if n else None,
                hazard_by_kind={k: sum(r.hazard_steps.get(k, 0) for r in items) / n for k in kinds},
                takeovers_mean=sum(r.takeovers for r in items) / n if n else None)
            if arm.id != "A0":
                pairs = [key for key in mine if key in base]
                for field, measure in (("delta_success", lambda r: float(r.success)),
                                       ("delta_hazard", lambda r: float(r.hazard_total)),
                                       ("delta_takeovers", lambda r: float(r.takeovers))):
                    result = stats.paired_bootstrap([measure(mine[k]) - measure(base[k]) for k in pairs])
                    if result:
                        setattr(stat, field, Delta(estimate=result[0], low=result[1], high=result[2], pairs=len(pairs)))
            out.append(stat)
        return Report(experiment=experiment_id, baseline="A0", complete=experiment.state == "completed",
                      method=stats.METHOD, arms=out)

    def matrix(self, experiment_id):
        experiment = self.get(experiment_id)
        value, rows = self.rows(experiment_id)
        uses = {u.key: u for u in experiment.config.plugins}
        stage_of = {}
        for key, use in uses.items():
            entry = self.registry.items["plugins"].get(use.plugin or "", {})
            stage_of[key] = entry.get("stage") or "?"
        events = self._events_per_episode(value)
        matrix_rows = []
        for arm in experiment.arms:
            cells = {}
            for stage in STAGES:
                keys = [k for k in arm.plugins if stage_of.get(k) == stage]
                if keys:
                    cells[stage] = MatrixCell(plugins=[uses[k].plugin or uses[k].entrypoint or k for k in keys],
                                              events_per_episode=events.get((arm.id, stage)))
            matrix_rows.append(MatrixRow(arm=arm.id, name=arm.name, cells=cells))
        seeds = list(range(experiment.config.seeds.start, experiment.config.seeds.start + experiment.config.seeds.episodes))
        found = {(r.arm, r.environment, r.seed): r.success for r in rows}
        outcomes = [OutcomeRow(arm=arm.id, environment=env, outcomes=[found.get((arm.id, env, s)) for s in seeds])
                    for arm in experiment.arms for env in experiment.config.environments]
        return Matrix(stages=list(STAGES), rows=matrix_rows, outcomes=outcomes)

    def _events_per_episode(self, value):
        """Mean number of plugin log records per episode, by (arm, stage), from trace.jsonl."""
        counts, episodes = {}, {}
        for item in value["runs"]:
            for summary in self._episode_dirs(item):
                episodes[item["arm"]] = episodes.get(item["arm"], 0) + 1
                for step in _jsonl(summary.parent / "trace.jsonl"):
                    for event in step.get("events") or []:
                        key = (item["arm"], event.get("stage"))
                        counts[key] = counts.get(key, 0) + 1
        return {key: count / episodes[key[0]] for key, count in counts.items() if episodes.get(key[0])}


def _key(name):
    """The kernel names plugins "<index>:<id>"; the id is the PluginUse key."""
    return name.partition(":")[2] if isinstance(name, str) and name.partition(":")[0].isdigit() else name


def _pairs(arms, environments):
    return [(arm, env) for arm in arms for env in environments]


def _row(item, summary):
    hazards = {k: int(v) for k, v in (summary.get("hazard_steps") or {}).items()}
    return EpisodeRow(id=f"{item['run_id']}:{summary['episode']}", arm=item["arm"], environment=item["environment"],
                      run_id=item["run_id"], episode=summary["episode"], seed=summary.get("seed", 0),
                      success=bool(summary.get("success")), success_step=summary.get("success_step"),
                      steps=summary.get("steps", 0), end_reason=summary.get("end_reason", ""),
                      hazard_steps=hazards, hazard_total=sum(hazards.values()),
                      takeovers=int(summary.get("takeovers") or 0), plugin_reports=summary.get("plugin_reports") or {})


def _jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _npz_frame_png(path, index):
    """Frame `index` of an (N, H, W, 3) uint8 array in an .npz, as PNG, without numpy."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".npy"))
        with archive.open(name) as stream:
            if stream.read(6) != b"\x93NUMPY":
                raise ValueError("Not an .npy array")
            major = stream.read(2)[0]
            size = struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]
            header = ast.literal_eval(stream.read(size).decode("latin1"))
            shape = tuple(header["shape"])
            if header["descr"] not in ("|u1", "<u1") or header.get("fortran_order") or len(shape) != 4 or shape[3] not in (3, 4):
                raise ValueError(f"Unsupported frame array {header}")
            count, height, width, channels = shape
            if not 0 <= index < count:
                raise KeyError(f"frame {index}")
            frame_bytes = height * width * channels
            remaining = index * frame_bytes
            while remaining:
                remaining -= len(stream.read(min(remaining, 1 << 20)))
            pixels = stream.read(frame_bytes)
    row_bytes = width * channels
    raw = b"".join(b"\x00" + pixels[y * row_bytes:(y + 1) * row_bytes] for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    colour = 2 if channels == 3 else 6
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
