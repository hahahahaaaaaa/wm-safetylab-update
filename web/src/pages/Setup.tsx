// 1 新建实验 + 2 实验配置: an eight-step form with a step rail. The backend validates the whole
// configuration on every change (POST /experiments/validate); the rail shows its issues per step.
import { useEffect, useMemo, useRef, useState } from 'react';
import type { Go } from '../App';
import { redteam, type Catalog, type EnvironmentEntry, type ExperimentConfig, type ModelEntry, type PluginEntry,
  type PluginUse, type Role, type Stage, type ValidationResult } from '../api/redteam';
import { cx, Empty, ErrorBox, fmt, PageHead, ROLE_NAME, RoleTag, STAGE_NAME, STAGES, StageTag, Tag } from '../ui';
import TaskPreview from '../components/TaskPreview';

export type StepId = 'basic' | 'model' | 'envtask' | 'plugins' | 'threat' | 'judge' | 'control' | 'review';
const STEPS: { id: StepId; sec: 'new' | 'config'; title: string }[] = [
  { id: 'basic', sec: 'new', title: '基本信息' }, { id: 'model', sec: 'new', title: '被测模型' }, { id: 'envtask', sec: 'new', title: '环境与任务' },
  { id: 'plugins', sec: 'config', title: '插件' }, { id: 'threat', sec: 'config', title: '威胁卡与 oracle' }, { id: 'judge', sec: 'config', title: '判定' },
  { id: 'control', sec: 'config', title: '对照' }, { id: 'review', sec: 'config', title: '检查并运行' },
];
const DRAFT_ID = 'current';
const LOCAL_KEY = 'wms-rt-draft';
const KNOWLEDGE = ['black_box', 'gray_box', 'white_box'];
const today = () => new Date().toISOString().slice(5, 10).replace('-', '');
const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));
const objectName = (family?: string) => family === 'bridge' ? 'source' : 'cube';

function example(catalog: Catalog): ExperimentConfig {
  const has = (id: string) => catalog.plugins.some(p => p.id === id);
  const plugins: PluginUse[] = [
    { key: 'image-perturbation', plugin: 'image-perturbation', config: { schedule: [{ type: 'occlude', start: 4, end: 12, box: 'object:cube', margin: 0.04 }] } },
    { key: 'target-moves', plugin: 'target-moves', config: { moves: [{ step: 6, object: 'cube', delta: [0.15, 0] }] } },
    { key: 'example-action-scale', plugin: 'example-action-scale', config: {} },
    { key: 'monitor-plan-summary', plugin: 'monitor-plan-summary', config: {} },
  ].filter(p => has(p.plugin));
  return { name: `rt-${today()}-toyplanner`, description: '', model: 'toy-planner', environments: ['toy-pick'], plugins,
    seeds: { start: 0, episodes: 20 }, max_steps: null,
    control: { single: true, combined: true, defended: true, defense_only: true, monitors_everywhere: true } };
}
const blank = (): ExperimentConfig => ({ name: '', description: '', model: '', environments: [], plugins: [], seeds: { start: 0, episodes: 20 }, max_steps: null,
  control: { single: true, combined: true, defended: true, defense_only: true, monitors_everywhere: true } });

function useDraft(catalog: Catalog) {
  const [config, setConfig] = useState<ExperimentConfig | null>(null);
  useEffect(() => {
    let local: ExperimentConfig | null = null;
    try { local = JSON.parse(localStorage.getItem(LOCAL_KEY) ?? 'null'); } catch { /* optional browser state */ }
    redteam.getDraft(DRAFT_ID).then(d => setConfig(d.config)).catch(() => setConfig(local ?? example(catalog)));
  }, [catalog]);
  const timer = useRef(0);
  useEffect(() => {
    if (!config) return;
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      try { localStorage.setItem(LOCAL_KEY, JSON.stringify(config)); } catch { /* optional */ }
      redteam.putDraft(DRAFT_ID, config).catch(() => { /* the local copy remains */ });
    }, 500);
  }, [config]);
  return [config, setConfig] as const;
}

function useValidation(config: ExperimentConfig | null) {
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!config) return;
    const timer = window.setTimeout(() => {
      redteam.validate(config).then(r => { setResult(r); setError(''); }).catch(e => setError(e instanceof Error ? e.message : String(e)));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [config]);
  return { result, error };
}

/** Why a plugin cannot run with this model and these environments (display only; the backend decides). */
function unmet(plugin: PluginEntry, model?: ModelEntry, envs: EnvironmentEntry[] = []) {
  const out: string[] = [];
  const probes = ['state', 'predict', 'plan', 'embed', 'learn'];
  for (const need of plugin.requires ?? []) {
    if (probes.includes(need)) {
      if (model?.kind === 'planner') out.push(`需要探针 ${need}；规划器没有模型`);
      else if (model && !(model.probes ?? []).includes(need)) out.push(`需要探针 ${need}`);
    } else if (envs.some(e => !(e.capabilities ?? []).includes(need))) out.push(`需要环境能力 ${need}`);
  }
  return out;
}

interface Ctx {
  catalog: Catalog; config: ExperimentConfig; set: (f: (c: ExperimentConfig) => void) => void;
  result: ValidationResult | null; model?: ModelEntry; envs: EnvironmentEntry[]; go: Go;
}

export default function Setup({ catalog, sec, step, go }: { catalog: Catalog; sec: 'new' | 'config'; step: StepId; go: Go }) {
  const [config, setConfig] = useDraft(catalog);
  const { result, error } = useValidation(config);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState('');
  const [recipeOpen, setRecipeOpen] = useState(false);
  const current = STEPS.find(s => s.id === step && s.sec === sec) ?? STEPS.find(s => s.sec === sec)!;
  if (!config) return <div className="empty-state">加载草稿…</div>;

  const set = (f: (c: ExperimentConfig) => void) => setConfig(prev => { const next = clone(prev!); f(next); return next; });
  const model = catalog.models.find(m => m.id === config.model);
  const envs = catalog.environments.filter(e => (config.environments ?? []).includes(e.id));
  const ctx: Ctx = { catalog, config, set, result, model, envs, go };
  const issues = (id: StepId) => (result?.issues ?? []).filter(i => i.step === id && i.severity === 'error');
  const stepError = (id: StepId) => issues(id)[0]?.message ?? '';
  const firstBad = STEPS.find(s => stepError(s.id));
  const index = STEPS.indexOf(current);
  const prev = STEPS[index - 1], next = STEPS[index + 1];
  const err = stepError(current.id);
  const counts: Partial<Record<StepId, string>> = {
    model: config.model, envtask: String(envs.length), plugins: String(config.plugins?.length ?? 0),
    threat: String((result?.plugins ?? []).filter(p => p.role === 'attack' || (p.role === 'defense' && p.privileged)).length),
    control: String(result?.arms.length ?? ''),
  };
  const launch = async () => {
    setLaunching(true); setLaunchError('');
    try { const e = await redteam.submit(config); go({ sec: 'progress', sub: 'live', exp: e.id }); }
    catch (e) { setLaunchError(e instanceof Error ? e.message : String(e)); }
    finally { setLaunching(false); }
  };
  const Body = { basic: Basic, model: Model, envtask: EnvTask, plugins: Plugins, threat: Threat, judge: Judge, control: Control, review: Review }[current.id];
  const railItem = (s: typeof STEPS[number]) => {
    const e = stepError(s.id);
    return <button key={s.id} className={cx('rail-item', e ? 'bad' : 'ok')} aria-current={s === current ? 'step' : undefined} title={e || '已完成'}
      onClick={() => go({ sec: s.sec, sub: s.id })}><span className="n">{e ? '!' : STEPS.indexOf(s) + 1}</span><span>{s.title}</span><span className="cnt">{counts[s.id] ?? ''}</span></button>;
  };
  return <div className="cfg-layout">
    <nav className="rail" aria-label="配置步骤"><div className="rail-h">{sec === 'new' ? '新建实验' : '实验配置'}</div>
      <div className="rail-sep">1 · 新建实验</div>{STEPS.filter(s => s.sec === 'new').map(railItem)}
      <div className="rail-sep">2 · 实验配置</div>{STEPS.filter(s => s.sec === 'config').map(railItem)}
      <div className="rail-foot"><button onClick={() => setRecipeOpen(true)}>{'{ }'} 查看配方 JSON</button>
        <button onClick={() => setConfig(example(catalog))}>↺ 载入示例</button><button onClick={() => setConfig(blank())}>✕ 清空</button></div></nav>
    <div className="stack">
      {error && <ErrorBox message={'校验服务出错：' + error} />}
      <Body {...ctx} />
      {current.id === 'review' && launchError && <ErrorBox message={launchError} />}
      <div className="stepbar">
        {prev && <button className="btn" onClick={() => go({ sec: prev.sec, sub: prev.id })}>‹ 上一步</button>}
        <span className={cx('hint', (err || (current.id === 'review' && firstBad)) && 'err')}>
          {err || (current.id === 'review' && firstBad ? `第 ${STEPS.indexOf(firstBad) + 1} 步：${stepError(firstBad.id)}` : '')}</span>
        {current.id === 'review'
          ? <button className="btn primary lg" disabled={!result?.valid || launching} onClick={launch}>{launching ? '提交中…' : '▶ 立即运行'}</button>
          : <button className="btn primary" disabled={!!err} onClick={() => go({ sec: next.sec, sub: next.id })}>下一步 ›</button>}
      </div>
    </div>
    {recipeOpen && <Drawer title="配方 JSON" text={JSON.stringify(result?.recipes ?? [], null, 2)} onClose={() => setRecipeOpen(false)} />}
  </div>;
}

function Drawer({ title, text, onClose }: { title: string; text: string; onClose: () => void }) {
  return <><div className="overlay" onClick={onClose} /><aside className="drawer" role="dialog" aria-label={title}>
    <div className="card-h"><h2>{title}</h2><span className="right">
      <button className="btn" onClick={() => navigator.clipboard?.writeText(text)}>复制</button>
      <button className="icon-btn" aria-label="关闭" onClick={onClose}>×</button></span></div>
    <div className="body"><pre className="code" style={{ maxHeight: 'none' }}>{text}</pre></div></aside></>;
}

const Probes = ({ model, catalog }: { model?: ModelEntry; catalog: Catalog }) => <>{catalog.probes.map(p =>
  <Tag key={p.id} mono kind={(model?.probes ?? []).includes(p.id) ? 'ok' : 'off'} title={p.description}>{p.id}</Tag>)}</>;
const Caps = ({ caps, catalog }: { caps: string[]; catalog: Catalog }) => <>{catalog.abilities.map(a =>
  <Tag key={a.id} mono kind={caps.includes(a.id) ? 'ok' : 'off'} title={a.description}>{a.id}</Tag>)}</>;

function Basic({ config, set, catalog, go }: Ctx) {
  const [recent, setRecent] = useState<{ id: string; name: string; model: string; arms: number; envs: number; config: ExperimentConfig }[]>([]);
  useEffect(() => { redteam.list().then(p => setRecent(p.items.slice(0, 6).map(e => ({ id: e.id, name: e.name, model: e.config.model ?? '', arms: e.arms.length, envs: (e.config.environments ?? []).length, config: e.config })))).catch(() => setRecent([])); }, []);
  return <>
    <PageHead title="基本信息" />
    <section className="card"><div className="card-b">
      <div className="field" style={{ maxWidth: 420 }}><label htmlFor="dName">实验名称</label>
        <input type="text" id="dName" value={config.name ?? ''} onChange={e => set(c => { c.name = e.target.value; })} /></div>
      <div className="field" style={{ maxWidth: 640, marginBottom: 0 }}><label htmlFor="dDesc">说明</label>
        <textarea id="dDesc" rows={2} value={config.description ?? ''} onChange={e => set(c => { c.description = e.target.value; })} /></div>
    </div></section>
    <section className="card"><div className="card-h"><h2>最近的实验</h2><span className="right"><button className="link" onClick={() => go({ sec: 'progress', sub: 'runs' })}>全部 →</button></span></div>
      {recent.length ? <div className="scroll"><table><thead><tr><th>实验</th><th>模型</th><th className="num">任务</th><th className="num">对照组</th><th /></tr></thead><tbody>
        {recent.map(r => <tr key={r.id}><td>{r.name}<div className="hint mono">{r.id}</div></td><td className="mono">{r.model}</td><td className="num">{r.envs}</td><td className="num">{r.arms}</td>
          <td><button className="link" onClick={() => set(c => { Object.assign(c, clone(r.config), { name: r.config.name + '-copy' }); })}>复制为新实验</button></td></tr>)}
      </tbody></table></div> : <div className="empty-state">还没有实验 · <button className="link" onClick={() => set(c => Object.assign(c, example(catalog)))}>载入示例</button></div>}
    </section>
  </>;
}

function Model({ catalog, config, set }: Ctx) {
  const [cat, setCat] = useState<'all' | 'local' | 'server'>('all');
  const list = catalog.models.filter(m => cat === 'all' || m.location === cat);
  const choose = (m: ModelEntry) => set(c => { c.model = m.id; c.environments = (c.environments ?? []).filter(e => (m.environments ?? []).includes(e)); if (!c.environments.length && m.environments?.length) c.environments = [m.environments[0]]; });
  return <>
    <PageHead title="被测模型" />
    <section className="card"><div className="card-b">
      <div className="cat-chips">{([['all', `全部 (${catalog.models.length})`], ['local', 'CPU'], ['server', '实验服务器 GPU']] as const).map(([k, v]) =>
        <button key={k} aria-pressed={cat === k} onClick={() => setCat(k)}>{v}</button>)}</div>
      <div className="opt-list">{list.map(m => {
        const sel = config.model === m.id;
        return <label key={m.id} className={cx('opt', sel && 'sel')}><input type="radio" name="model" checked={sel} onChange={() => choose(m)} />
          <h3>{m.name}<Tag mono>{m.id}</Tag>{m.kind === 'planner' && <Tag>规划器</Tag>}{m.note && <Tag kind="ok">{m.note}</Tag>}
            {!m.configured && <Tag kind="warn" title={m.configuration_error ?? ''}>此服务未配置</Tag>}</h3>
          <div className="meta">{m.location === 'local' ? '当前服务 CPU' : '实验服务器 GPU'}{m.memory != null && ` · ${m.memory ? '有记忆' : '无记忆'}`}{m.seconds_per_episode != null && ` · 约 ${m.seconds_per_episode} s / episode`}</div>
          <div className="tags"><span className="hint" style={{ marginRight: 2 }}>探针</span><Probes model={m} catalog={catalog} />
            <span className="hint" style={{ margin: '0 2px 0 8px' }}>任务</span>{(m.environments ?? []).length > 2 ? <Tag>{m.environments!.length} 个</Tag> : (m.environments ?? []).map(e => <Tag key={e} mono>{e}</Tag>)}</div>
        </label>;
      })}</div>
    </div></section>
  </>;
}

function EnvTask({ catalog, config, set, model }: Ctx) {
  if (!model) return <><PageHead title="环境与任务" /><Empty>先选择被测模型</Empty></>;
  const options = catalog.environments.filter(e => (model.environments ?? []).includes(e.id));
  const toggle = (id: string) => set(c => { const s = new Set(c.environments ?? []); s.has(id) ? s.delete(id) : s.add(id); c.environments = options.map(o => o.id).filter(o => s.has(o)); });
  const first = options[0];
  return <>
    <PageHead title="环境与任务" />
    <section className="card"><div className="card-h"><h2>仿真环境</h2><span className="right"><Tag>由 {model.id} 决定</Tag></span></div>
      <div className="card-b"><div className="lockbox"><Lock /><div><b>{first ? ({ toy: 'ToyPick（CPU 玩具，非物理）', bridge: 'SimplerEnv Bridge', maniskill: 'ManiSkill 3' } as Record<string, string>)[first.family] ?? first.family : '—'}</b>
        <p>{first?.robot}</p></div></div></div></section>
    <section className="card"><div className="card-h"><h2>任务</h2><span className="sub">已选 {(config.environments ?? []).length} / {options.length}</span>
      {options.length > 1 && <span className="right"><button className="btn" onClick={() => set(c => { c.environments = options.map(o => o.id); })}>全选</button><button className="btn" onClick={() => set(c => { c.environments = []; })}>清空</button></span>}</div>
      <div className="card-b"><div className="trows">{options.map(e => {
        const sel = (config.environments ?? []).includes(e.id);
        return <label key={e.id} className={cx('trow', sel && 'sel')}><input type="checkbox" checked={sel} onChange={() => toggle(e.id)} /><TaskPreview name={e.name} preview={e.preview} />
          <span className="tmain"><b>{e.name} <Tag mono>{e.id}</Tag></b><span>{e.instruction && `“${e.instruction}” · `}{e.max_steps ? `max_steps ${e.max_steps}` : ''}{e.observation === 'vector' ? ' · 向量观测' : ''}</span>
            {e.preview && <span className="task-preview-caption">{e.preview.caption}{e.preview.source_environment && e.preview.source_environment !== e.id && <> · 来源环境：<span className="mono">{e.preview.source_environment}</span></>}</span>}</span>
          <span className="tside"><span className="tags"><Caps caps={e.capabilities ?? []} catalog={catalog} /></span></span></label>;
      })}</div>
        <div className="sec-t">具身安全场景</div><div className="trows">{catalog.planned.map(p =>
          <label key={p.id} className="trow dis"><input type="checkbox" disabled /><TaskPreview name={p.name} placeholder="场景尚未实现" /><span className="tmain"><b>{p.id} · {p.name}</b><span>{p.description}</span></span>
            <span className="tside"><span className="tags"><Tag kind="plan">规划中</Tag></span></span></label>)}</div>
      </div></section>
  </>;
}
const Lock = () => <svg viewBox="0 0 16 16"><path d="M4 7V5a4 4 0 1 1 8 0v2h1v8H3V7zm2 0h4V5a2 2 0 1 0-4 0z" /></svg>;

function Plugins(ctx: Ctx) {
  const { catalog, config, set, model, envs, result } = ctx;
  const [stage, setStage] = useState<'all' | Stage>('all');
  const [role, setRole] = useState<'all' | Role>('all');
  const [entry, setEntry] = useState('');
  const uses = config.plugins ?? [];
  const checks = Object.fromEntries((result?.plugins ?? []).map(p => [p.key, p]));
  const inStage = (p: { stage: Stage }) => stage === 'all' || p.stage === stage;
  const list = catalog.plugins.filter(p => inStage(p) && (role === 'all' || p.role === role));
  const caps = [...new Set(envs.flatMap(e => e.capabilities ?? []))];
  const toggle = (p: PluginEntry) => set(c => {
    const i = (c.plugins ?? []).findIndex(u => u.plugin === p.id);
    if (i >= 0) c.plugins!.splice(i, 1); else (c.plugins ??= []).push({ key: p.id, plugin: p.id, config: starterConfig(p, envs[0]?.family) });
  });
  const custom = uses.filter(u => !u.plugin);
  const add = () => {
    const v = entry.trim();
    if (!/^[\w.]+:\w+$/.test(v)) return;
    set(c => { let n = 1; while ((c.plugins ?? []).some(u => u.key === `c${n}`)) n++; (c.plugins ??= []).push({ key: `c${n}`, entrypoint: v, config: {} }); });
    setEntry('');
  };
  const card = (p: PluginEntry) => {
    const use = uses.find(u => u.plugin === p.id), sel = !!use, check = use ? checks[use.key] : undefined;
    const why = sel ? check?.reasons ?? [] : unmet(p, model, envs);
    const requires = check?.requires ?? p.requires ?? [];
    const privileged = check?.privileged ?? p.privileged;
    return <div key={p.id} className={cx('opt', sel && 'sel', !sel && why.length > 0 && 'dis')}>
      <label className="opt-lab"><input type="checkbox" checked={sel} disabled={!sel && why.length > 0} onChange={() => toggle(p)} aria-label={p.id} />
        <h3><StageTag stage={p.stage} /><RoleTag role={p.role} /><span className="mono" style={{ fontSize: 13 }}>{p.id}</span><span>{p.name !== p.id ? p.name : ''}</span><Tag kind="ok">已登记</Tag></h3>
        <div className="meta">{p.description}</div>
        <div className="tags"><Tag mono>side: {p.side}</Tag>
          {requires.map(r => <Tag key={r} mono kind={(model?.probes ?? []).includes(r) || caps.includes(r) ? 'ok' : 'bad'}>requires {r}</Tag>)}
          {privileged && <Tag mono kind="warn">privileged</Tag>}{p.config_dependent && <Tag title={p.config_dependent}>随配置变化</Tag>}<Tag mono>{p.entrypoint}</Tag></div>
        {why.length > 0 && <div className="why">{why.join('；')}</div>}</label>
      {sel && <div className="opt-body"><ConfigForm use={use!} entry={p} family={envs[0]?.family} set={set} /></div>}
    </div>;
  };
  const grouped = (Object.keys(ROLE_NAME) as Role[]).map(r => {
    const items = uses.filter(u => (checks[u.key]?.role ?? catalog.plugins.find(p => p.id === u.plugin)?.role) === r);
    return items.length ? <div key={r}><div className="sel-g">{ROLE_NAME[r]}</div>{items.map(u => <SelItem key={u.key} use={u} stage={checks[u.key]?.stage} set={set} />)}</div> : null;
  });
  const unknown = custom.filter(u => !checks[u.key]?.role);
  return <>
    <PageHead title="插件" />
    <section className="card"><div className="card-b">
      <div className="capbar"><div><span className="hint">探针</span><span className="mono">{model?.id ?? '—'}</span><span className="tags"><Probes model={model} catalog={catalog} /></span></div>
        <div><span className="hint">环境能力</span><span className="tags"><Caps caps={caps.filter(c => envs.every(e => (e.capabilities ?? []).includes(c)))} catalog={catalog} /></span></div></div>
      <div className="loop" role="group" aria-label="环节">
        <button aria-pressed={stage === 'all'} onClick={() => setStage('all')}><b>全部</b><span>闭环各环节</span><i>{catalog.plugins.length}</i></button>
        {STAGES.map(s => <button key={s} aria-pressed={stage === s} onClick={() => setStage(s)}><b>{s}</b><span>{STAGE_NAME[s]}</span><i>{catalog.plugins.filter(p => p.stage === s).length}</i></button>)}
      </div>
      <div className="cat-chips"><button aria-pressed={role === 'all'} onClick={() => setRole('all')}>全部角色</button>
        {(Object.keys(ROLE_NAME) as Role[]).map(r => <button key={r} aria-pressed={role === r} onClick={() => setRole(r)}>{ROLE_NAME[r]} <span className="mono">{catalog.plugins.filter(p => p.role === r && inStage(p)).length}</span></button>)}</div>
      <div className="plug-layout">
        <div className="opt-list">{list.map(card)}{list.length === 0 && <div className="empty-state">该环节还没有登记的插件</div>}
          {custom.map(u => { const check = checks[u.key];
            return <div key={u.key} className="opt sel"><div className="opt-lab"><h3><StageTag stage={check?.stage} /><RoleTag role={check?.role} /><span className="mono" style={{ fontSize: 13 }}>{u.entrypoint}</span><Tag kind="warn">自定义</Tag></h3>
              {check?.reasons?.length ? <div className="why">{check.reasons.join('；')}</div> : null}</div>
              <div className="opt-body"><ConfigForm use={u} family={envs[0]?.family} set={set} /></div></div>; })}
        </div>
        <aside className="selected-panel"><h3>已选（{uses.length}）</h3>{grouped}
          {unknown.length > 0 && <div><div className="sel-g">未识别</div>{unknown.map(u => <SelItem key={u.key} use={u} set={set} />)}</div>}
          {!uses.length && <p className="hint">尚未选择</p>}
          <div className="sec-t" style={{ marginTop: 14 }}>添加未登记的插件</div>
          <div className="toolbar"><input type="text" value={entry} onChange={e => setEntry(e.target.value)} onKeyDown={e => e.key === 'Enter' && add()} placeholder="module:Class" aria-label="插件入口" style={{ flex: 1, minWidth: 0 }} />
            <button className="btn" onClick={add} disabled={!/^[\w.]+:\w+$/.test(entry.trim())}>添加</button></div>
          <div className="hint mono" style={{ marginTop: 6 }}>先用 wms plugin test &lt;ID&gt; 检查</div></aside>
      </div>
    </div></section>
  </>;
}
const SelItem = ({ use, stage, set }: { use: PluginUse; stage?: Stage | null; set: Ctx['set'] }) =>
  <div className="sel-item"><span>{stage ? stage + ' · ' : ''}{use.plugin ?? use.entrypoint}</span>
    <button aria-label={'移除 ' + use.key} onClick={() => set(c => { c.plugins = (c.plugins ?? []).filter(u => u.key !== use.key); })}>−</button></div>;

function starterConfig(p: PluginEntry, family?: string): Record<string, unknown> {
  const obj = objectName(family);
  if (p.id === 'image-perturbation') return { schedule: [{ type: 'occlude', start: 4, end: 12, box: `object:${obj}`, margin: 0.04 }] };
  if (p.id === 'target-moves') return { moves: [{ step: 6, object: obj, delta: family === 'toy' ? [0.15, 0] : [0.04, 0, 0] }] };
  return {};
}

function ConfigForm({ use, entry, family, set }: { use: PluginUse; entry?: PluginEntry; family?: string; set: Ctx['set'] }) {
  const config = { ...(entry?.config ?? {}), ...(use.config ?? {}) } as Record<string, unknown>;
  const update = (f: (cfg: Record<string, unknown>) => void) => set(c => { const u = c.plugins!.find(x => x.key === use.key)!; const cfg = { ...(entry?.config ?? {}), ...(u.config ?? {}) }; f(cfg); u.config = cfg; });
  const obj = objectName(family);
  if (entry?.stage === 'S6' && ['s6-residual-monitor', 's6-cusum', 's6-feedback-poison'].includes(entry.id)) {
    const poison = entry.id === 's6-feedback-poison', detector = entry.id === 's6-cusum';
    const fields = (poison ? [['factor', '反馈位移比例', 0], ['start', '开始步', 0], ['end', '结束步（留空持续）', 1]] :
      [['scale', '残差尺度', 0.000000001], ...(detector ? [['center', '干净残差均值', 0], ['spread', '干净残差标准差', 0.000000001], ['drift', '容忍偏差', 0], ['threshold', '累计告警阈值', 0.000000001], ['cooldown', '冷却步数', 0], ['steps', '备用接管步数', 1]] : [])]) as [string, string, number][];
    return <div className="sched"><div className="sched-row">{fields.map(([key, label, min]) => <label key={key}>{label} <input aria-label={key} type="number" min={min} max={key === 'factor' ? 1 : undefined} step={['start', 'end', 'cooldown', 'steps'].includes(key) ? 1 : 'any'} value={typeof config[key] === 'number' ? Number(config[key]) : ''} onChange={e => {
      if (!e.target.value) { if (key === 'end') update(c => { c.end = null; }); return; }
      const value = Number(e.target.value);
      if (Number.isFinite(value) && value >= min && (key !== 'factor' || value <= 1) && (!['start', 'end', 'cooldown', 'steps'].includes(key) || Number.isInteger(value))) update(c => { c[key] = value; });
    }} /></label>)}</div>
      {!poison && <><label className="check"><input type="checkbox" aria-label="include_changed" checked={config.include_changed === true} onChange={e => update(c => { c.include_changed = e.target.checked; })} />纳入 S5 改变动作的步骤（解释为闭环偏离）</label><div className="hint">默认排除动作改变；接管步骤始终排除。残差不是物理安全概率。</div></>}
      {detector && <><div className="sched-row"><label>告警处理 <select aria-label="takeover" value={String(config.takeover ?? 'none')} onChange={e => update(c => { c.takeover = e.target.value; })}><option value="none">只告警</option><option value="stop">下一步停止</option><option value="fallback">下一步备用动作</option></select></label></div>
        <div className="hint">先校准 center、spread、threshold，再做独立种子评估。备用模式必须提供控制器适用的动作，且冷却步数不能小于接管步数。</div></>}
      {poison && <div className="hint">仅用于 toy-planner-online；factor=1 为无投毒对照。只修改学习反馈，不修改真实环境。</div>}
      <details><summary>高级 JSON（备用动作或逐维残差尺度）</summary><GenericConfig config={config} onChange={next => set(c => { c.plugins!.find(x => x.key === use.key)!.config = next; })} /></details>
    </div>;
  }
  if (entry?.stage === 'S5' && ['s5-action-noise', 's5-action-delay', 's5-action-barrier'].includes(entry.id)) {
    const fields = [
      ...(entry.id === 's5-action-noise' ? [['std', '噪声标准差', 0, false]] : []),
      ...(entry.id === 's5-action-delay' ? [['steps', '延迟步数', 0, false]] : []),
      ...(entry.id === 's5-action-barrier' ? [['limit', '幅度上限', 0.000001, false], ['max_delta', '每步最大变化量', 0.000001, true]] : []),
      ['dims', '前几个动作维度', 1, true], ['start', '开始步（含）', 0, false], ['end', '结束步（不含）', 1, true],
    ] as [string, string, number, boolean][];
    return <div className="sched">
      <div className="sched-row">{fields.map(([key, label, min, nullable]) => <label key={key}>{label} <input
        type="number" min={min} step={['steps', 'dims', 'start', 'end'].includes(key) ? 1 : 'any'}
        aria-label={key} value={typeof config[key] === 'number' ? Number(config[key]) : ''}
        placeholder={key === 'dims' ? '全部' : key === 'end' ? '持续' : key === 'max_delta' ? '关闭' : undefined}
        onChange={e => {
          if (!e.target.value) { if (nullable) update(c => { c[key] = null; }); return; }
          const value = Number(e.target.value);
          if (Number.isFinite(value) && value >= min && (!['steps', 'dims', 'start', 'end'].includes(key) || Number.isInteger(value))) update(c => { c[key] = value; });
        }} /></label>)}</div>
      <div className="hint">默认只作用前两维，请按控制器含义调整，避免误改夹爪。结束步留空表示持续。</div>
      {entry.id === 's5-action-noise' && <><label className="check"><input type="checkbox" aria-label="clip_to_bounds" checked={config.clip_to_bounds === true} onChange={e => update(c => { c.clip_to_bounds = e.target.checked; })} />按控制器范围限幅（TD-MPC2 噪声实验需开启）</label><div className="hint">std=0 时不改变动作；强度单位为控制指令单位。限幅前后动作均记录。</div></>}
      {entry.id === 's5-action-delay' && <div className="hint">steps=0 时无延迟；队列预热时重复首个动作。</div>}
      {entry.id === 's5-action-barrier' && <div className="hint">首步只限幅，修改动作时记录接管；仅约束控制命令，不保证碰撞安全。</div>}
      {typeof config.end === 'number' && config.end <= Number(config.start ?? 0) && <div className="why">结束步必须大于开始步。</div>}
    </div>;
  }
  if (entry?.id === 'image-perturbation') {
    const rows = (config.schedule as Record<string, unknown>[] | undefined) ?? [];
    return <div className="sched">{rows.map((s, i) => <div key={i} className="sched-row">
      <select aria-label="type" value={String(s.type)} onChange={e => update(c => { (c.schedule as Record<string, unknown>[])[i].type = e.target.value; })}>{['occlude', 'blur', 'noise', 'drop', 'freeze'].map(t => <option key={t}>{t}</option>)}</select>
      start <input type="number" aria-label="start" value={Number(s.start ?? 0)} onChange={e => update(c => { (c.schedule as Record<string, unknown>[])[i].start = +e.target.value; })} />
      end <input type="number" aria-label="end" value={Number(s.end ?? 0)} onChange={e => update(c => { (c.schedule as Record<string, unknown>[])[i].end = +e.target.value; })} />
      {s.type === 'occlude' && <select aria-label="box" value={typeof s.box === 'string' ? 'object' : 'fixed'} onChange={e => update(c => { (c.schedule as Record<string, unknown>[])[i].box = e.target.value === 'object' ? `object:${obj}` : [0.3, 0.3, 0.7, 0.7]; })}>
        <option value="object">box: object:{obj}（跟随物体，读真值）</option><option value="fixed">box: [0.3, 0.3, 0.7, 0.7]</option></select>}
      <button className="link" onClick={() => update(c => { (c.schedule as unknown[]).splice(i, 1); })}>删除</button></div>)}
      <div><button className="btn sm" onClick={() => update(c => { c.schedule = [...((c.schedule as unknown[]) ?? []), { type: 'blur', start: 10, end: 20, sigma: 2.0 }]; })}>+ 条目</button></div></div>;
  }
  if (entry?.id === 'target-moves') {
    const rows = (config.moves as Record<string, unknown>[] | undefined) ?? [];
    const delta = (m: Record<string, unknown>) => ((('goal' in m ? (m.goal as Record<string, unknown>).delta : m.delta) as number[] | undefined) ?? [0, 0]);
    const setRow = (i: number, f: (m: Record<string, unknown>) => Record<string, unknown>) => update(c => { const list = c.moves as Record<string, unknown>[]; list[i] = f(list[i]); });
    const make = (step: number, goal: boolean, d: number[]) => goal ? { step, goal: { delta: d } } : { step, object: obj, delta: d };
    return <div className="sched">{rows.map((m, i) => { const d = delta(m), goal = 'goal' in m;
      return <div key={i} className="sched-row">step <input type="number" aria-label="step" value={Number(m.step ?? 0)} onChange={e => setRow(i, x => ({ ...x, step: +e.target.value }))} />
        <select aria-label="target" value={goal ? 'goal' : 'object'} onChange={e => setRow(i, x => make(Number(x.step), e.target.value === 'goal', d))}><option value="object">object:{obj}</option><option value="goal">goal</option></select>
        Δx <input type="number" step="0.01" aria-label="dx" value={d[0]} onChange={e => setRow(i, x => make(Number(x.step), goal, [+e.target.value, ...d.slice(1)]))} />
        Δy <input type="number" step="0.01" aria-label="dy" value={d[1]} onChange={e => setRow(i, x => make(Number(x.step), goal, [d[0], +e.target.value, ...d.slice(2)]))} />
        <button className="link" onClick={() => update(c => { (c.moves as unknown[]).splice(i, 1); })}>删除</button></div>; })}
      <div><button className="btn sm" onClick={() => update(c => { c.moves = [...((c.moves as unknown[]) ?? []), make(10, false, family === 'toy' ? [0.1, 0] : [0.03, 0, 0])]; })}>+ 移动</button></div></div>;
  }
  return <GenericConfig config={config} onChange={next => set(c => { c.plugins!.find(x => x.key === use.key)!.config = next; })} />;
}

function GenericConfig({ config, onChange }: { config: Record<string, unknown>; onChange: (c: Record<string, unknown>) => void }) {
  const [raw, setRaw] = useState<string | null>(null);
  const fields = Object.entries(config).filter(([k]) => k !== 'oracle' && k !== 'threat');
  const simple = fields.every(([, v]) => ['number', 'boolean', 'string'].includes(typeof v));
  if (raw !== null || !simple) {
    const text = raw ?? JSON.stringify(Object.fromEntries(fields), null, 2);
    let bad = false; try { JSON.parse(text); } catch { bad = true; }
    return <div><textarea className="mono" rows={Math.min(10, text.split('\n').length + 1)} value={text} style={bad ? { borderColor: 'var(--bad)' } : undefined}
      onChange={e => { setRaw(e.target.value); try { const v = JSON.parse(e.target.value); onChange({ ...config, ...v }); } catch { /* keep typing */ } }} aria-label="config JSON" /></div>;
  }
  return <div className="sched-row">{fields.map(([k, v]) => typeof v === 'boolean'
    ? <label key={k} className="check"><input type="checkbox" checked={v} onChange={e => onChange({ ...config, [k]: e.target.checked })} />{k}</label>
    : <span key={k}>{k} <input type={typeof v === 'number' ? 'number' : 'text'} step="any" aria-label={k} value={String(v)} onChange={e => onChange({ ...config, [k]: typeof v === 'number' ? +e.target.value : e.target.value })} /></span>)}
    {!fields.length && <span className="hint">无参数</span>}<button className="link" onClick={() => setRaw(JSON.stringify(Object.fromEntries(fields), null, 2))}>JSON</button></div>;
}

function Threat({ config, set, result, catalog }: Ctx) {
  const checks = result?.plugins ?? [];
  const attacks = checks.filter(p => p.role === 'attack'), defenses = checks.filter(p => p.role === 'defense' && p.privileged);
  const otherPriv = checks.filter(p => p.privileged && p.role !== 'defense');
  const issueFor = (key: string) => (result?.issues ?? []).filter(i => i.step === 'threat' && i.plugin === key);
  const use = (key: string) => (config.plugins ?? []).find(u => u.key === key)!;
  const setThreat = (key: string, k: string, v: unknown) => set(c => { const u = c.plugins!.find(x => x.key === key)!; u.threat = { ...(u.threat ?? {}), [k]: v }; });
  const setOracle = (key: string, v: boolean) => set(c => { const u = c.plugins!.find(x => x.key === key)!; u.config = { ...(u.config ?? {}), oracle: v }; });
  const defaultThreat = (id: string) => catalog.plugins.find(p => p.id === id)?.threat ?? {};
  return <>
    <PageHead title="威胁卡与 oracle" />
    {!attacks.length && !defenses.length && <Empty>未选择攻击插件或读真值的防御</Empty>}
    {attacks.length > 0 && <div className="sec-t" style={{ margin: 0 }}>攻击插件 · 威胁卡</div>}
    {attacks.map(p => { const card = { ...defaultThreat(p.id), ...(use(p.key).threat ?? {}) } as Record<string, unknown>, issues = issueFor(p.key);
      const field = (k: string) => <div className="field"><label htmlFor={`th-${p.key}-${k}`}>{k}</label>
        <input type="text" id={`th-${p.key}-${k}`} value={String(card[k] ?? '')} onChange={e => setThreat(p.key, k, e.target.value)} style={!String(card[k] ?? '').trim() ? { borderColor: 'var(--bad)' } : undefined} /></div>;
      return <section key={p.key} className="card"><div className="card-h"><StageTag stage={p.stage} /><RoleTag role="attack" /><h2 className="mono">{p.id}</h2>
        <span className="right">{issues.length ? <span className="pill bad">不完整</span> : <span className="pill good">完整</span>}</span></div>
        <div className="card-b">
          <div className="field" style={{ maxWidth: 260 }}><label htmlFor={`th-${p.key}-knowledge`}>knowledge</label>
            <select id={`th-${p.key}-knowledge`} value={String(card.knowledge ?? '')} onChange={e => setThreat(p.key, 'knowledge', e.target.value)}><option value="">—</option>{KNOWLEDGE.map(k => <option key={k}>{k}</option>)}</select></div>
          <div className="cols2" style={{ gap: '0 16px' }}>{field('access')}{field('budget')}</div>{field('goal')}
          <label className="check"><input type="checkbox" checked={!!card.adaptive} onChange={e => setThreat(p.key, 'adaptive', e.target.checked)} />adaptive</label>
          {issues.map((i, n) => <div key={n} className="err" style={{ marginTop: 6 }}>{i.message}</div>)}
        </div></section>; })}
    {defenses.length > 0 && <div className="sec-t" style={{ margin: 0 }}>读真值的防御</div>}
    {defenses.map(p => { const on = !!(use(p.key).config as Record<string, unknown> | undefined)?.oracle;
      return <section key={p.key} className="card"><div className="card-h"><StageTag stage={p.stage} /><RoleTag role="defense" /><h2 className="mono">{p.id}</h2><Tag mono kind="warn">privileged</Tag>
        <span className="right"><label className="check"><input type="checkbox" checked={on} onChange={e => setOracle(p.key, e.target.checked)} />oracle</label></span></div>
        <div className="card-b">{on ? <div className="okbox">结果标为 oracle 上界</div> : <div className="errbox">读取仿真真值（ctx.truth），部署时没有；不打开 oracle 不能运行</div>}</div></section>; })}
    {otherPriv.length > 0 && <section className="card"><div className="card-h"><h2>其他读真值的插件</h2></div>
      <div className="card-b" style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>{otherPriv.map(p => <span key={p.key} className={'chip-x ' + p.role}>{p.id}</span>)}<Tag kind="warn" mono>privileged</Tag><Tag>故障与监控可读真值，不需要 oracle</Tag></div></section>}
  </>;
}

function Judge({ catalog, config, set, envs }: Ctx) {
  return <>
    <PageHead title="判定" />
    {envs.map(e => { const d = { thresholds: e.hazard_thresholds ?? {}, note: e.hazard_note };
      return <section key={e.id} className="card"><div className="card-h"><h2>危险判定 · {e.name}</h2><span className="right"><Tag>当前环境配方生效值</Tag></span></div>
        <div className="scroll"><table><thead><tr><th>参数</th><th className="num">值</th></tr></thead><tbody>
          {Object.entries(d.thresholds).map(([k, v]) => <tr key={k}><td className="mono">{k}</td><td className="num">{v}</td></tr>)}</tbody></table></div>
        <div className="card-b hint" style={{ borderTop: '1px solid var(--line)' }}>{d.note}</div></section>; })}
    <section className="card"><div className="card-h"><h2>危险类型</h2></div><div className="scroll"><table><thead><tr><th>id</th><th>名称</th><th>严重度</th><th>条件</th></tr></thead><tbody>
      {catalog.hazards.map(h => <tr key={h.id}><td className="mono">{h.id}</td><td>{h.name}</td><td><Tag kind={h.severity === 'high' ? 'bad' : 'warn'}>{h.severity}</Tag></td><td className="reason">{h.description}</td></tr>)}</tbody></table></div></section>
    <section className="card"><div className="card-h"><h2>任务成功</h2></div><div className="scroll"><table><thead><tr><th>任务</th><th>成功条件</th><th className="num">默认 max_steps</th></tr></thead><tbody>
      {envs.map(e => <tr key={e.id}><td className="mono">{e.id}</td><td className="reason">环境 success 事件</td><td className="num">{e.max_steps ?? '—'}</td></tr>)}</tbody></table></div>
      <div className="card-b" style={{ borderTop: '1px solid var(--line)' }}><div className="inline-fields"><div className="field"><label htmlFor="maxSteps">max_steps（留空用环境默认）</label>
        <input type="number" id="maxSteps" min={1} value={config.max_steps ?? ''} onChange={e => set(c => { c.max_steps = e.target.value ? +e.target.value : null; })} /></div></div></div></section>
    <section className="card"><div className="card-h"><h2>语义判定</h2><span className="right"><Tag kind="plan">规划中</Tag></span></div>
      <div className="card-b" style={{ opacity: .55 }}><label className="check"><input type="checkbox" disabled />视觉语言模型复核</label></div></section>
  </>;
}

function Estimates({ result, model }: { result: ValidationResult | null; model?: ModelEntry }) {
  const m = result?.estimate.minutes;
  return <div className="estimates"><div className="est">对照组<b>{result?.arms.length ?? '—'}</b></div><div className="est">运行<b>{result?.estimate.runs ?? '—'}</b></div>
    <div className="est">episode<b>{result?.estimate.episodes.toLocaleString() ?? '—'}</b></div>
    <div className="est">预计时长<b>{m == null ? '未测' : m < 1 ? '<1m' : m < 90 ? `~${Math.ceil(m)}m` : `~${fmt(m / 60, 1)}h`}</b></div>
    <div className="est">运行位置<b style={{ fontSize: 14 }}>{model ? model.location === 'local' ? '当前服务 CPU' : '实验服务器 GPU' : '—'}</b></div></div>;
}

function ArmTable({ result, config }: { result: ValidationResult | null; config: ExperimentConfig }) {
  const name = (key: string) => { const u = (config.plugins ?? []).find(p => p.key === key); return u?.plugin ?? u?.entrypoint ?? key; };
  const role = (key: string) => result?.plugins.find(p => p.key === key)?.role ?? '';
  return <div className="scroll"><table><thead><tr><th>组</th><th>名称</th><th>插件</th><th /></tr></thead><tbody>
    {(result?.arms ?? []).map(a => <tr key={a.id}><td className="mono">{a.id}</td><td>{a.name}</td>
      <td>{a.plugins.length ? a.plugins.map(k => <span key={k} className={'chip-x ' + role(k)} style={{ marginRight: 4 }}>{name(k)}</span>) : <span className="hint">—</span>}</td>
      <td>{a.kind === 'baseline' && <Tag>配对基准</Tag>}{a.oracle && <Tag kind="warn">oracle 上界</Tag>}</td></tr>)}</tbody></table></div>;
}

function Control({ config, set, result, model }: Ctx) {
  const ctl = config.control!;
  const ck = (k: keyof typeof ctl, label: string) => <label className="check"><input type="checkbox" checked={!!ctl[k]} onChange={e => set(c => { c.control![k] = e.target.checked; })} />{label}</label>;
  const seeds = config.seeds!;
  return <>
    <PageHead title="对照" /><Estimates result={result} model={model} />
    <section className="card"><div className="card-h"><h2>对照组</h2><span className="right">{ck('single', '逐个扰动')}{ck('combined', '全部扰动')}{ck('defended', '扰动 + 防御')}{ck('defense_only', '只开防御')}{ck('monitors_everywhere', '每组都挂监控')}</span></div>
      <ArmTable result={result} config={config} /></section>
    <section className="card"><div className="card-h"><h2>种子</h2></div><div className="card-b">
      <div className="inline-fields"><div className="field"><label htmlFor="nEp">每组每任务 episode</label><input type="number" id="nEp" min={1} max={1000} value={seeds.episodes} onChange={e => set(c => { c.seeds!.episodes = Math.max(1, +e.target.value); })} /></div>
        <div className="field"><label htmlFor="seed0">起始种子</label><input type="number" id="seed0" min={0} value={seeds.start} onChange={e => set(c => { c.seeds!.start = Math.max(0, +e.target.value); })} /></div>
        <span className="hint mono">seeds {seeds.start} … {seeds.start! + seeds.episodes! - 1}</span></div>
      <div className="lockbox" style={{ marginTop: 12 }}><Lock /><div><b>所有组使用同一组种子</b><p>按（任务，种子）与基线配对 · 差值的 bootstrap 95% 区间 · 成功率 Wilson 区间</p></div></div>
    </div></section>
    {(result?.issues ?? []).filter(i => i.step === 'control').map((i, n) => <div key={n} className={i.severity === 'error' ? 'errbox' : 'hint'} style={i.severity === 'warning' ? { color: 'var(--warn)' } : undefined}>{i.message}</div>)}
  </>;
}

function Review({ config, set, result, model, envs, go }: Ctx) {
  const warnings = (result?.issues ?? []).filter(i => i.severity === 'warning');
  const recipes = useMemo(() => JSON.stringify(result?.recipes ?? [], null, 2), [result]);
  return <>
    <PageHead title="检查并运行" />
    <div className="field" style={{ maxWidth: 520, margin: 0 }}><label htmlFor="dName2">实验名称</label><input type="text" id="dName2" value={config.name ?? ''} onChange={e => set(c => { c.name = e.target.value; })} /></div>
    <Estimates result={result} model={model} />
    {warnings.map((w, i) => <div key={i} className="hint" style={{ color: 'var(--warn)' }}>⚠ {w.message}</div>)}
    <div className="cols2">
      <section className="card"><div className="card-h"><h2>模型与任务</h2><span className="right"><button className="link" onClick={() => go({ sec: 'new', sub: 'model' })}>修改</button></span></div>
        <div className="card-b"><dl className="kv"><dt>模型</dt><dd className="mono">{model?.id ?? '—'}</dd><dt>探针</dt><dd className="mono">{(model?.probes ?? []).join(', ') || '无'}</dd>
          <dt>任务</dt><dd className="mono">{envs.map(e => e.id).join(', ') || '—'}</dd><dt>max_steps</dt><dd className="mono">{config.max_steps ?? '环境默认'}</dd></dl></div></section>
      <section className="card"><div className="card-h"><h2>插件（{(config.plugins ?? []).length}）</h2><span className="right"><button className="link" onClick={() => go({ sec: 'config', sub: 'plugins' })}>修改</button></span></div>
        <div className="card-b" style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>{(config.plugins ?? []).map(u => <span key={u.key} className={'chip-x ' + (result?.plugins.find(p => p.key === u.key)?.role ?? '')}>{u.plugin ?? u.entrypoint}</span>)}</div></section>
    </div>
    <section className="card"><div className="card-h"><h2>对照组（{result?.arms.length ?? 0}）</h2><span className="right"><button className="link" onClick={() => go({ sec: 'config', sub: 'control' })}>修改</button></span></div><ArmTable result={result} config={config} /></section>
    <details className="acc"><summary>配方 JSON（{result?.recipes.length ?? 0} 个运行，提交给 wms.episodes.kernel:run）</summary><pre className="code">{recipes}</pre></details>
  </>;
}
