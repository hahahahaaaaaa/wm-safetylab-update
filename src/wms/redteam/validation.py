"""Server-side validation of a red-team configuration, the comparison arms and the run configurations.

The web UI shows the issues per setup step; compatibility rules live only here.
Plugin rules are those of wms.plugins.base.problems and wms.plugins.resolve.check;
the episode kernel checks the loaded model and environment again at run time.
"""
from __future__ import annotations

from wms.plugins import base
from wms.redteam.catalog import META, plugin_class, plugin_meta
from wms.redteam.schema import Arm, Catalog, Estimate, ExperimentConfig, PluginCheck, StepIssue, ValidationResult

DISTURBANCES = ("fault", "attack")
STAGE_STEP = {"S2": "state", "S3": "predict", "S4": "plan"}


class _Plugin:
    """A selected plugin with its class and effective config."""

    def __init__(self, use, registry):
        self.use, self.cls, self.error = use, None, None
        entry = registry.items["plugins"].get(use.plugin) if use.plugin else None
        self.id = use.plugin or use.entrypoint or use.key
        self.entrypoint = (entry or {}).get("entrypoint") or use.entrypoint
        self.config = {**((entry or {}).get("config") or {}), **use.config}
        if use.threat:
            self.config["threat"] = {**(self.config.get("threat") or {}), **use.threat}
        if use.plugin and entry is None:
            self.error = f"未登记的插件：{use.plugin}"
        elif not self.entrypoint:
            self.error = "需要插件 id 或 entrypoint（module:Class）"
        else:
            try:
                self.cls = plugin_class(self.entrypoint, entry)
            except Exception as exc:  # noqa: BLE001 - shown to the user as a configuration error
                self.error = f"无法导入 {self.entrypoint}：{type(exc).__name__}: {exc}"
        self.role = getattr(self.cls, "role", None)
        self.stage = getattr(self.cls, "stage", None)
        self.privileged, self.requires = plugin_meta(self.cls, self.entrypoint, self.config) if self.cls else (False, [])
        self.oracle = bool(self.config.get("oracle")) and self.role == "defense" and self.privileged

    def spec(self):
        item = {"plugin": self.use.plugin} if self.use.plugin else {"entrypoint": self.entrypoint}
        item["id"] = self.use.key
        if self.use.config or self.use.threat:
            item["config"] = {key: value for key, value in self.config.items()
                              if key in self.use.config or key == "threat" and self.use.threat}
        return item


def build_arms(selected, control):
    disturb = [p for p in selected if p.role in DISTURBANCES]
    defend = [p for p in selected if p.role == "defense"]
    watch = [p.use.key for p in selected if p.role == "monitor"] if control.monitors_everywhere else []
    names = {p.use.key: META["plugins"].get(p.id, {}).get("name", p.id) for p in selected}
    arms = [Arm(id="A0", name="基线（只挂监控）" if watch else "基线（不挂插件）", kind="baseline", plugins=watch)]

    def add(kind, name, members):
        keys = [p.use.key for p in members]
        arms.append(Arm(id=f"A{len(arms)}", name=name, kind=kind, plugins=keys + watch,
                        oracle=any(p.oracle for p in members)))

    if control.single:
        for p in disturb:
            add("single", names[p.use.key], [p])
    if control.combined and len(disturb) > 1:
        add("combined", "全部扰动", disturb)
    if control.defended and disturb and defend:
        add("defended", "扰动 + 防御", disturb + defend)
    if control.defense_only and defend:
        add("defense_only", "只开防御", defend)
    return arms


def build_recipes(config, model, arms, selected, registry):
    """One episode-kernel run configuration per (arm, environment)."""
    meta = META["models"].get(config.model) or META["planners"].get(config.model) or {}
    by_key = {p.use.key: p for p in selected}
    out = []
    for arm in arms:
        for env in config.environments:
            params = {"episodes": config.seeds.episodes, "perturbations": [], "target_moves": [],
                      "plugins": [by_key[key].spec() for key in arm.plugins], "video": False, "save_raw": False}
            if config.max_steps:
                params["max_steps"] = config.max_steps
            recipe = {"recipe": meta.get("template"), "name": f"{config.name} · {arm.id} · {env}",
                      "environment": env, "seed": config.seeds.start, "params": params, "metrics": [],
                      "tags": ["redteam", arm.id, env]}
            if model is not None and model.kind == "policy":
                recipe["model"] = config.model
            out.append(recipe)
    return out


def validate_config(config: ExperimentConfig, catalog: Catalog, registry) -> ValidationResult:
    issues: list[StepIssue] = []

    def add(step, field, message, severity="error", plugin=None):
        issues.append(StepIssue(step=step, field=field, message=message, severity=severity, plugin=plugin))

    models = {m.id: m for m in catalog.models}
    envs = {e.id: e for e in catalog.environments}

    if not config.name.strip():
        add("basic", "name", "请填写实验名称")

    model = models.get(config.model)
    if not config.model:
        add("model", "model", "请选择被测模型")
    elif model is None:
        add("model", "model", f"未知模型：{config.model}")
    elif not model.configured:
        where = "需要在实验服务器上的服务提交" if model.location == "server" else "请检查 runtime.json"
        add("model", "model", f"此服务没有 {model.name} 的机器配置（{where}）：{model.configuration_error}")

    chosen = [envs[e] for e in config.environments if e in envs]
    for env in config.environments:
        if env not in envs:
            add("envtask", "environments", f"未知任务：{env}")
        elif model and env not in model.environments:
            add("envtask", "environments", f"{model.name} 不支持 {envs[env].name}")
    if not chosen:
        add("envtask", "environments", "至少选择 1 个任务")
    if len(chosen) > 1 and len({e.family for e in chosen}) > 1:
        add("envtask", "environments", "一次实验的任务需来自同一个仿真环境", "warning")

    keys = [use.key for use in config.plugins]
    for key in {k for k in keys if keys.count(k) > 1}:
        add("plugins", "plugins", f"插件 key 重复：{key}")
    selected = [_Plugin(use, registry) for use in config.plugins]
    checks = []
    probes = set(model.probes) if model else set()
    for p in selected:
        reasons = []
        if p.error:
            reasons.append(p.error)
        else:
            for text in base.problems(p.cls, p.config):
                if "privileged defense" in text:
                    add("threat", f"plugins.{p.use.key}.config.oracle",
                        "这个防御读取仿真真值，部署时拿不到；打开 oracle 作为上界运行（结果会标注）", plugin=p.use.key)
                elif "threat card" in text or "threat.knowledge" in text:
                    add("threat", f"plugins.{p.use.key}.threat", "攻击插件需要完整的威胁卡：" + text.split(";")[-1].strip(),
                        plugin=p.use.key)
                else:
                    reasons.append(text.split(": ", 1)[-1])
            for need in p.requires:
                if need in base.PROBES:
                    if model and model.kind == "planner":
                        reasons.append(f"需要探针 {need}；规划器没有模型")
                    elif model and need not in probes:
                        reasons.append(f"需要探针 {need}")
                elif need in base.ENV_ABILITIES:
                    lacking = [e.name for e in chosen if need not in e.capabilities]
                    if lacking:
                        reasons.append(f"需要环境能力 {need}（{'、'.join(lacking)} 没有）")
        for reason in reasons:
            add("plugins", f"plugins.{p.use.key}", f"{p.id}：{reason}", plugin=p.use.key)
        checks.append(PluginCheck(key=p.use.key, id=p.id, stage=p.stage, role=p.role, privileged=p.privileged,
                                  oracle=p.oracle, requires=p.requires, available=not reasons))

    usable = [p for p, check in zip(selected, checks) if check.available]
    if not any(p.role in DISTURBANCES + ("defense",) for p in usable):
        add("plugins", "plugins", "没有故障、攻击或防御插件：只会运行基线", "warning")
    if any(p.oracle for p in usable):
        add("control", "plugins", "含 oracle 防御的对照组是上界，不代表可部署的防御", "warning")
    if config.seeds.episodes < 10:
        add("control", "seeds.episodes", "每组少于 10 个 episode，置信区间会很宽", "warning")

    arms = build_arms(usable, config.control)
    recipes = build_recipes(config, model, arms, usable, registry) if model and chosen else []
    episodes = len(recipes) * config.seeds.episodes
    seconds = model.seconds_per_episode if model else None
    minutes = round(episodes * seconds / 60, 1) if seconds is not None else None
    return ValidationResult(valid=not any(i.severity == "error" for i in issues), issues=issues, plugins=checks,
                            arms=arms, estimate=Estimate(runs=len(recipes), episodes=episodes, minutes=minutes),
                            recipes=recipes)
