"""Request and response models for the red-team web API.

These models are the contract between the web UI and the backend: FastAPI turns
them into OpenAPI, and the web types are generated from that document.

A red-team experiment runs one model on one or more environments with S1-S6
plugins (wms.plugins). The service builds comparison arms from the selected
plugins (a baseline, each disturbance alone, all disturbances, disturbances plus
defenses, defenses alone), submits one episode-kernel run per arm and
environment with the same seeds, and compares every arm with the baseline on
the same (environment, seed) pairs.
"""
from typing import Any, Literal

from pydantic import BaseModel, Field

Stage = Literal["ENV", "S1", "S2", "S3", "S4", "S5", "S6"]
Role = Literal["fault", "attack", "defense", "monitor"]
StepId = Literal["basic", "model", "envtask", "plugins", "threat", "judge", "control", "review"]
Location = Literal["local", "server"]
ArmKind = Literal["baseline", "single", "combined", "defended", "defense_only"]
RunState = Literal["queued", "running", "cancelling", "completed", "failed", "cancelled", "interrupted"]
ExperimentState = Literal["queued", "running", "cancelling", "completed", "partial_failed", "failed", "cancelled"]


# ---------- catalog (read-only, from the registry) ----------

class Label(BaseModel):
    id: str
    name: str
    description: str = ""


class ModelEntry(BaseModel):
    id: str
    name: str
    kind: Literal["policy", "planner"] = Field(description="planner: the environment's own motion planner, no model")
    probes: list[str] = Field(description="Model probes plugins can rely on (wms.plugins.base.PROBES)")
    environments: list[str] = Field(description="Environments this model runs on")
    location: Location = Field(description="local: CPU on this machine; server: needs the GPU server")
    configured: bool = Field(description="This service has the machine configuration the model needs")
    configuration_error: str | None = None
    memory: bool | None = None
    seconds_per_episode: float | None = Field(None, description="Measured wall time per episode; null = not measured")
    note: str = ""


class EnvironmentEntry(BaseModel):
    id: str
    name: str
    family: str
    capabilities: list[str] = Field(description="Environment abilities (wms.plugins.base.ENV_ABILITIES)")
    observation: Literal["rgb", "vector"] | None = None
    robot: str = ""
    instruction: str = ""
    max_steps: int | None = None


class PluginEntry(BaseModel):
    id: str
    name: str
    description: str = ""
    entrypoint: str
    stage: Stage
    role: Role
    side: Literal["env", "model"] = "env"
    privileged: bool = False
    requires: list[str] = []
    threat: dict[str, Any] | None = Field(None, description="Default threat card (attacks)")
    config: dict[str, Any] = Field(default_factory=dict, description="Default config")
    hooks: list[str] = []
    config_dependent: str = Field("", description="How the config changes privileged / requires, if it does")


class HazardKind(BaseModel):
    id: str
    name: str
    severity: Literal["high", "medium", "low"]
    description: str


class HazardDefaults(BaseModel):
    family: str
    thresholds: dict[str, float]
    note: str = ""


class PlannedScenario(BaseModel):
    id: str
    name: str
    description: str


class Catalog(BaseModel):
    catalog_version: str
    models: list[ModelEntry]
    environments: list[EnvironmentEntry]
    plugins: list[PluginEntry]
    stages: list[Label]
    roles: list[Label]
    probes: list[Label]
    abilities: list[Label]
    knowledge: list[Label]
    hazards: list[HazardKind]
    hazard_defaults: list[HazardDefaults]
    planned: list[PlannedScenario] = Field(description="Scenarios without a simulator backend yet; not selectable")


# ---------- experiment configuration ----------

class PluginUse(BaseModel):
    key: str = Field(description="Unique within the experiment, e.g. p1")
    plugin: str | None = Field(None, description="Registry plugin id")
    entrypoint: str | None = Field(None, description="module:Class of a plugin not in the registry")
    config: dict[str, Any] = Field(default_factory=dict, description="Merged over the registry default; may hold oracle")
    threat: dict[str, Any] | None = Field(None, description="Threat card of an attack, merged into config.threat")


class SeedPlan(BaseModel):
    start: int = 0
    episodes: int = Field(20, ge=1, le=1000, description="Episodes per arm and environment; seeds start..start+episodes-1")


class ControlPlan(BaseModel):
    single: bool = Field(True, description="One arm per disturbance (fault or attack)")
    combined: bool = Field(True, description="All disturbances together (when there are two or more)")
    defended: bool = Field(True, description="All disturbances plus all defenses")
    defense_only: bool = Field(True, description="Defenses without disturbances (their cost)")
    monitors_everywhere: bool = Field(True, description="Attach monitors to every arm, the baseline included")


class ExperimentConfig(BaseModel):
    name: str = ""
    description: str = ""
    model: str = ""
    environments: list[str] = []
    plugins: list[PluginUse] = []
    seeds: SeedPlan = SeedPlan()
    max_steps: int | None = Field(None, ge=1, description="Step budget per episode; null = the environment's own")
    control: ControlPlan = ControlPlan()


class StepIssue(BaseModel):
    step: StepId
    field: str
    message: str
    severity: Literal["error", "warning"] = "error"
    plugin: str | None = Field(None, description="PluginUse.key the issue is about")


class PluginCheck(BaseModel):
    key: str
    id: str
    stage: Stage | None = None
    role: Role | None = None
    privileged: bool = False
    oracle: bool = False
    requires: list[str] = []
    available: bool = Field(description="Runs with the selected model and environments")
    reasons: list[str] = []


class Arm(BaseModel):
    id: str = Field(description="A0 is the baseline")
    name: str
    kind: ArmKind
    plugins: list[str] = Field(description="PluginUse keys in this arm")
    oracle: bool = Field(False, description="Contains an oracle defense: an upper bound, not a deployable result")


class Estimate(BaseModel):
    runs: int
    episodes: int
    minutes: float | None = Field(None, description="null when the model's time per episode is not measured")


class ValidationResult(BaseModel):
    valid: bool
    issues: list[StepIssue]
    plugins: list[PluginCheck]
    arms: list[Arm]
    estimate: Estimate
    recipes: list[dict[str, Any]] = Field(description="The episode-kernel run configurations that would be submitted")


class Draft(BaseModel):
    id: str
    config: ExperimentConfig
    updated_at: str


# ---------- experiments and runs ----------

class ExperimentRun(BaseModel):
    arm: str
    environment: str
    run_id: str
    state: RunState
    episodes_done: int = 0
    episodes_total: int
    error: str | None = None


class Experiment(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    state: ExperimentState
    config: ExperimentConfig
    arms: list[Arm]
    runs: list[ExperimentRun]
    episodes_done: int
    episodes_total: int


class ExperimentPage(BaseModel):
    items: list[Experiment]


class LogChunk(BaseModel):
    run_id: str
    text: str
    next_offset: int


class RunEvent(BaseModel):
    type: Literal["progress", "episode", "state"]
    experiment: str
    data: dict[str, Any]


# ---------- results ----------

class EpisodeRow(BaseModel):
    id: str = Field(description="<run_id>:<episode index>")
    arm: str
    environment: str
    run_id: str
    episode: int
    seed: int
    success: bool
    success_step: int | None = None
    steps: int
    end_reason: str
    hazard_steps: dict[str, int] = {}
    hazard_total: int = 0
    takeovers: int = 0
    plugin_reports: dict[str, Any] = {}


class EpisodePage(BaseModel):
    items: list[EpisodeRow]
    total: int
    next_cursor: str | None = None


class EpisodePlugin(BaseModel):
    key: str
    id: str
    stage: Stage | None = None
    role: Role | None = None
    config: dict[str, Any] = {}


class EpisodeDetail(BaseModel):
    row: EpisodeRow
    baseline: str | None = Field(None, description="Episode id of the baseline with the same environment and seed")
    initial_state: dict[str, Any] = {}
    final_state: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    frames: dict[str, int] = Field(default_factory=dict, description="Camera -> number of frames")
    plugins: list[EpisodePlugin] = []


class TraceStep(BaseModel):
    t: int
    action: list[float] | None = None
    action_proposed: list[float] | None = None
    reward: float | None = None
    state_before: dict[str, Any] = {}
    state_after: dict[str, Any] = {}
    policy_info: dict[str, Any] = {}
    hazards: list[dict[str, Any]] = []
    plugins: dict[str, list[str]] = Field(default_factory=dict, description="Stage -> active plugins")
    events: list[dict[str, Any]] = Field(default_factory=list, description="Plugin log records of this step")
    takeover: dict[str, Any] | None = None
    env_events: list[Any] = []


class Trace(BaseModel):
    episode: str
    start: int
    end: int
    total: int
    steps: list[TraceStep]


class Interval(BaseModel):
    estimate: float | None
    low: float | None
    high: float | None


class Delta(BaseModel):
    estimate: float
    low: float
    high: float
    pairs: int


class ArmStats(BaseModel):
    arm: str
    name: str
    kind: ArmKind
    oracle: bool
    episodes: int
    success: Interval = Field(description="Success rate with a Wilson 95% interval")
    hazard_steps_mean: float | None
    hazard_by_kind: dict[str, float] = Field(description="Mean hazard steps per episode, by hazard id")
    takeovers_mean: float | None
    delta_success: Delta | None = Field(None, description="Minus the baseline, paired by (environment, seed)")
    delta_hazard: Delta | None = None
    delta_takeovers: Delta | None = None


class Report(BaseModel):
    experiment: str
    baseline: str
    complete: bool
    method: str
    arms: list[ArmStats]


class MatrixCell(BaseModel):
    plugins: list[str]
    events_per_episode: float | None = None


class MatrixRow(BaseModel):
    arm: str
    name: str
    cells: dict[str, MatrixCell] = Field(description="Stage -> plugins of this arm at that stage")


class OutcomeRow(BaseModel):
    arm: str
    environment: str
    outcomes: list[bool | None] = Field(description="One entry per seed in order; null = not run yet")


class Matrix(BaseModel):
    stages: list[str]
    rows: list[MatrixRow]
    outcomes: list[OutcomeRow]
