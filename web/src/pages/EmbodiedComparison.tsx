import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, post, terminal, statusTitle, Run, Config, fileUrl } from '../api';
import EmbodiedScene, { EmbodiedSceneData } from '../components/EmbodiedScene';
import { Episode, FramesResponse, LiveSnapshot, VisualFrame, Decision, Observation, DecisionView, isEmbodiedRun, frameAtTime, decisionAtFrame, comparisonWarnings, cycleStartTime, normalizeDecision } from '../embodied';
import './embodiedComparison.css';

type Descriptor = { platform_state?: string; ready: boolean; state: string; reason?: string; source?: string; scene_url?: string; frames_url?: string; episode_url?: string; live_url?: string; prepare_url?: string; prepare_run_id?: string };
type VisualData = { descriptor?: Descriptor; scene?: EmbodiedSceneData; episode?: Episode; frames?: FramesResponse; live?: LiveSnapshot; error?: string };
const titles: Record<string, string> = { baseline: '规则基线', chat: 'Chat 模型', jev: 'JEV 模型' };
const tasks: Record<string, string> = { transfer: '搬运入盘', stack: '方块堆叠', barrier: '越障搬运' };
const phases: Record<string, string> = { approach: '靠近物体', descend: '下降对齐', grasp: '抓取', lift: '抬升', carry: '搬运', lower: '放下', release: '松爪', retreat: '撤离', withdraw: '撤离', done: '完成', hold: '保持', direct: '直接执行', gentle: '减速执行' };
const outcomes: Record<string, string> = { success: '任务成功', uncertain: '置信度不足 · 停止', exhausted: '达到步数上限', stalled: '动作停滞', error: '执行错误', timeout: '超时', stopped: '停止', cancelled: '已取消' };
const pretty = (value: unknown): string => value === undefined || value === null ? '—' : typeof value === 'boolean' ? value ? '是' : '否' : typeof value === 'number' ? Number(value.toFixed(3)).toString() : String(value);
const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const vector = (value: unknown) => Array.isArray(value) ? value.map(v => typeof v === 'number' ? v.toFixed(3) : '—').join(' / ') : '—';
const providerOf = (run?: Run) => String(run?.config.params?.provider ?? '');
const taskOf = (run?: Run) => String(run?.config.params?.task ?? '');
const labelOf = (run: Run) => `${titles[providerOf(run)] ?? run.name} · ${tasks[taskOf(run)] ?? taskOf(run)} · seed ${run.config.seed ?? '?'} · ${run.id.slice(-8)}`;
const pathOf = (id: string) => '/runs/' + encodeURIComponent(id) + '/embodied-jev';
async function resource<T>(url: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal, cache: 'no-store' });
  if (!response.ok) throw new Error(`场景数据加载失败（${response.status}）`);
  return response.json() as Promise<T>;
}
function useVisual(id: string, revision: number): VisualData {
  const [data, setData] = useState<VisualData>({});
  useEffect(() => {
    setData({});
    if (!id) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let current: VisualData = {};
    async function poll() {
      let done = false;
      try {
        const descriptor = await api<Descriptor>(pathOf(id), { signal: controller.signal });
        current = { ...current, descriptor, error: undefined };
        if (descriptor.scene_url && !current.scene) current.scene = await resource<EmbodiedSceneData>(descriptor.scene_url, controller.signal);
        if (descriptor.ready && descriptor.frames_url && descriptor.episode_url) {
          const [frames, episode] = await Promise.all([resource<FramesResponse>(descriptor.frames_url, controller.signal), resource<Episode>(descriptor.episode_url, controller.signal)]);
          current = { ...current, frames, episode }; done = true;
        } else if (descriptor.live_url) current.live = await resource<LiveSnapshot>(descriptor.live_url, controller.signal);
        if (descriptor.state === 'unavailable' && terminal(descriptor.platform_state ?? '')) done = true;
        if (!controller.signal.aborted) setData({ ...current });
      } catch (error) {
        if (!controller.signal.aborted) setData({ ...current, error: message(error) });
      }
      if (!controller.signal.aborted && !done) timer = setTimeout(() => void poll(), 1200);
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [id, revision]);
  return data;
}
function DecisionDetails({ decision, view, live }: { decision: Decision | null; view: DecisionView | null; live?: boolean }) {
  const probabilities = decision?.probabilities;
  const items = probabilities ? Object.entries(probabilities).sort((a, b) => b[1] - a[1]) : [];
  return <section className="ev-section ev-decision">
    <div className="ev-section-title"><h3>{view?.decisionRole === 'intent' ? '阶段决策' : '动作决策'}</h3><span>{live ? '最近响应' : view?.cycle != null ? `CYCLE ${view.cycle}` : 'RECORDED'}</span></div>
    <div className="ev-choice"><strong>{decision?.choice ? phases[decision.choice] ?? decision.choice : '等待决策记录'}</strong><span>{typeof decision?.latency_ms === 'number' ? `${(decision.latency_ms / 1000).toFixed(2)} s 请求耗时` : '—'}</span></div>
    {view?.phase && <p className="ev-muted">阶段：{phases[view.phase] ?? view.phase}{view.action?.label ? ` · ${view.action.label}` : ''}</p>}
    {items.length ? <div className="ev-probabilities">{items.map(([id, probability]) => <div key={id} className={'ev-probability ' + (id === decision?.choice ? 'selected' : '')}><div><span>{phases[id] ?? id}{id === decision?.choice ? ' · 选中' : ''}</span><b>{(probability * 100).toFixed(1)}%</b></div><div className="ev-prob-track"><i style={{ width: `${probability * 100}%` }} /></div></div>)}</div> : <p className="ev-empty-prob">{decision?.channel_decisions ? '分通道决策 · 未提供联合概率' : decision ? '该决策未提供候选概率' : '该时刻暂无决策'}</p>}
    {decision?.channel_decisions && <div className="ev-channel-decisions">{Object.entries(decision.channel_decisions).map(([channel, item]) => <div key={channel}><b>{channel}: {item.choice ?? '—'}</b>{item.probabilities && Object.entries(item.probabilities).map(([choice, p]) => <div className={'ev-probability ' + (choice === item.choice ? 'selected' : '')} key={choice}><div><span>{choice}</span><b>{(p * 100).toFixed(1)}%</b></div><div className="ev-prob-track"><i style={{ width: `${p * 100}%` }} /></div></div>)}</div>)}</div>}
    {view?.intent && <p className="ev-muted">阶段选择：{phases[view.intent.choice ?? ''] ?? view.intent.choice}{typeof view.intent.selected_probability === 'number' ? ` · ${(view.intent.selected_probability * 100).toFixed(1)}%` : ''}</p>}
    <p className="ev-footnote">{live ? '最近一次响应可能属于尚未执行的动作。' : view?.timingLabel ?? '从动作开始处读取对应记录。'}</p>
    {!!view?.candidates.length && <details><summary>候选动作与预演 · {view.candidates.length}</summary><div className="ev-candidates">{view.candidates.map(candidate => <div key={candidate.id}><strong>{candidate.id} <span>{candidate.label}</span></strong><p>目标 TCP：{vector(candidate.target)}</p><p>夹爪：{pretty(candidate.gripper)} · 时长 {pretty(candidate.seconds)} s · 检查通过：{pretty(candidate.admitted)}</p>{candidate.rejection && <p>{candidate.rejection}</p>}{candidate.preview && <p>预演目标误差：{pretty(candidate.preview.target_error_m)} m</p>}</div>)}</div></details>}
  </section>;
}
function ObservationDetails({ observation }: { observation?: Observation | null }) {
  return <section className="ev-section"><div className="ev-section-title"><h3>当前仿真状态</h3><span>XYZ · m</span></div><dl className="ev-observations"><dt>末端 TCP</dt><dd>{vector(observation?.tcp)}</dd><dt>物体位置</dt><dd>{vector(observation?.object)}</dd><dt>目标位置</dt><dd>{vector(observation?.destination)}</dd><dt>夹爪 / 持物</dt><dd>{pretty(observation?.gripper)} / {pretty(observation?.held)}</dd><dt>支撑接触</dt><dd>{pretty(observation?.support_contact)}</dd><dt>禁区接触 / 成功</dt><dd>{pretty(observation?.forbidden_contact)} / {pretty(observation?.success)}</dd></dl></section>;
}
function Lane({ index, run, data, time, mode, camera, onCamera, onSeek, onPrepare, onRun, onResults }: {
  index: number; run?: Run; data: VisualData; time: number; mode: 'replay' | 'live'; camera: 'orbit' | 'top'; onCamera: (mode: 'orbit' | 'top') => void;
  onSeek: (time: number) => void; onPrepare: () => void; onRun: () => void; onResults: () => void;
}) {
  const episode = data.episode;
  const selection = useMemo(() => frameAtTime(data.frames?.frames ?? [], time), [data.frames, time]);
  const isLive = mode === 'live' && !data.frames;
  const frame: VisualFrame | null = isLive ? data.live?.frame ?? null : selection.frame;
  const replayEpisode = useMemo(() => episode && data.frames ? { ...episode, frames: data.frames.frames } : null, [episode, data.frames]);
  const view: DecisionView | null = isLive && data.live ? {
    cycle: data.live.cycle ?? null, phase: data.live.last_intent?.choice ?? frame?.phase ?? null,
    intent: normalizeDecision(data.live.last_intent), decision: normalizeDecision(data.live.last_decision),
    candidates: data.live.candidates ?? [], action: null, before: null, after: null,
    completed: null, executed: null, historyIndex: null, source: 'terminal', decisionRole: 'unknown',
    timingLabel: '实时快照最近响应；未必已执行', timingUncertain: true,
  } : replayEpisode && !isLive ? decisionAtFrame(replayEpisode, frame, selection.index) : null;
  const decision = isLive ? normalizeDecision(data.live?.last_decision) : view?.decision ?? null;
  const history = episode?.history ?? data.live?.history ?? [];
  const evidence = useMemo(() => JSON.stringify({ history: episode?.history ?? data.live?.history ?? [], events: episode?.events ?? data.live?.events ?? [], last_decision_inputs: episode?.last_decision_inputs }, null, 2), [episode, data.live]);
  const status = episode?.status ?? data.live?.status;
  const provider = episode?.provider ?? providerOf(run);
  const model = episode?.model ?? (provider === 'baseline' ? 'deterministic policy' : '使用平台已配置模型');
  const missing = ['missing', 'stale'].includes(data.descriptor?.state ?? '');
  const video = run?.result?.outputs.find(output => (output.media_type ?? '').startsWith('video/'));
  return <article className={'ev-lane ev-lane-' + index} data-testid={'lane-' + index}>
    <header className="ev-lane-header"><span className="ev-lane-letter">{String.fromCharCode(65 + index)}</span><div><h2>{titles[provider] ?? '选择实验'}</h2><p>{String(model ?? '—')}</p></div><span className={'ev-status ' + (episode?.success ? 'success' : '')}>{episode?.success ? '任务成功' : status ? outcomes[status] ?? statusTitle[status] ?? status : run ? statusTitle[run.state] ?? run.state : '未选择'}</span></header>
    <div className="ev-lane-meta"><span>{tasks[taskOf(run)] ?? taskOf(run)} · seed {run?.config.seed ?? '—'}</span><button onClick={onRun} disabled={!run}>{run?.id.slice(-8) ?? '运行详情'} ↗</button></div>
    <EmbodiedScene scene={data.scene ?? null} frame={frame} label={`${titles[provider] ?? '实验'} · ${isLive ? '实时场景' : '轨迹回放'}`} cameraMode={camera} onCameraModeChange={onCamera} />
    {data.error && <p className="ev-warning" role="alert">{data.error}</p>}
    {missing && <div className="ev-prepare"><p>此历史回合尚未生成交互场景。</p><button className="secondary" onClick={onPrepare}>准备 3D 回放</button></div>}
    {data.descriptor?.state === 'preparing' && <p className="ev-muted ev-inset" role="status">正在排队生成姿态缓存，完成后自动载入…</p>}
    {data.descriptor?.state === 'unavailable' && <p className="ev-warning">{run?.state === 'queued' ? '排队等待执行，开始后将自动载入场景。' : run && !terminal(run.state) ? '正在初始化场景与模型连接…' : run?.state === 'cancelled' ? '实验已取消；此回合没有可还原的轨迹。' : '此回合没有可还原的场景数据，可查看运行详情。'}</p>}
    <div className="ev-frame-status"><span>{isLive ? `实时 · ${pretty(data.live?.stage)}` : selection.frame ? `${selection.frameElapsed.toFixed(2)} / ${selection.duration.toFixed(2)} s` : '尚无回放帧'}</span><b>{isLive ? `周期 ${data.live?.cycle ?? 0}` : selection.ended && selection.frame ? '已到末帧 · 保持' : frame?.cycle ? `周期 ${frame.cycle}` : '初始状态'}</b></div>
    <div className="ev-cycle-strip" aria-label={`通道 ${index + 1} 动作时间线`}>{history.map((record, step) => {
      const start = replayEpisode ? cycleStartTime(replayEpisode, record.cycle) : null;
      return <button key={`${record.cycle}-${step}`} title={`${record.cycle} · ${record.label ?? record.phase}`} aria-pressed={view?.cycle === record.cycle} disabled={start === null} onClick={() => start !== null && onSeek(start)}>{record.cycle}<span>{phases[record.phase ?? ''] ?? record.phase ?? '动作'}</span></button>;
    })}{!history.length && <span className="ev-muted">执行后将显示逐步动作时间线</span>}</div>
    <DecisionDetails decision={decision} view={view} live={isLive} />
    <ObservationDetails observation={frame?.observation} />
    {view?.after && <p className="ev-outcome">当前动作已完成 · 持物 {pretty(view.after.held)} · 成功 {pretty(view.after.success)}</p>}
    <section className="ev-section ev-run-summary"><div className="ev-section-title"><h3>全回合记录</h3><span>非当前帧指标</span></div><div className="ev-stats"><div><b>{pretty(episode?.model_calls ?? data.live?.model_calls)}</b><span>模型调用</span></div><div><b>{history.length}</b><span>动作记录</span></div><div><b>{typeof (episode?.wall_seconds ?? data.live?.wall_seconds) === 'number' ? Number(episode?.wall_seconds ?? data.live?.wall_seconds).toFixed(1) : '—'}</b><span>墙钟耗时 / s</span></div></div>
      <details><summary>完整动作、输入与事件</summary><p className="ev-footnote">此处展示归档的全回合证据，不随回放时间裁剪。</p><pre>{evidence}</pre></details>
      <div className="ev-links"><button onClick={onResults} disabled={!run}>结果与评价 ↗</button>{video && run && <a href={fileUrl(run.id, video.path)} target="_blank" rel="noreferrer">姿态回放视频 ↗</a>}{data.descriptor?.episode_url && <a href={data.descriptor.episode_url} target="_blank" rel="noreferrer">回合 JSON ↗</a>}</div>
    </section>
  </article>;
}
export default function EmbodiedComparison({ runs, onRun, onResults, onRefresh }: { runs: Run[]; onRun: (id: string) => void; onResults: (id: string) => void; onRefresh: () => Promise<void> }) {
  const [ids, setIds] = useState<string[]>(() => { try { const stored = JSON.parse(localStorage.getItem('wms.embodied.ids') ?? '[]'); return Array.isArray(stored) ? stored.slice(0, 3).filter(id => typeof id === 'string') : []; } catch { return []; } });
  const [newRuns, setNewRuns] = useState<Run[]>([]);
  const available = useMemo(() => [...runs, ...newRuns.filter(run => !runs.some(item => item.id === run.id))].filter(isEmbodiedRun), [runs, newRuns]);
  const [tab, setTab] = useState<'history' | 'new'>('history');
  const [count, setCount] = useState(3);
  const [providers, setProviders] = useState(['baseline', 'chat', 'jev']);
  const [task, setTask] = useState('transfer');
  const [seed, setSeed] = useState(0);
  const [preview, setPreview] = useState(true);
  const [threshold, setThreshold] = useState(0.55);
  const [control, setControl] = useState('skills');
  const [observation, setObservation] = useState('privileged');
  const [maxCalls, setMaxCalls] = useState(20);
  const [maxCycles, setMaxCycles] = useState(30);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [revision, setRevision] = useState(0);
  const [mode, setMode] = useState<'live' | 'replay'>('replay');
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [camera, setCamera] = useState<'orbit' | 'top'>('orbit');
  const initialized = useRef(false);
  useEffect(() => {
    if (initialized.current || !available.length) return;
    initialized.current = true;
    const valid = ids.filter(id => available.some(run => run.id === id));
    if (valid.length >= 2) { setIds(valid); setCount(valid.length); return; }
    const matched = ['baseline', 'chat', 'jev'].map(provider => available.find(run => providerOf(run) === provider && taskOf(run) === 'transfer' && run.config.seed === 0 && run.state === 'completed')).filter((run): run is Run => !!run);
    setIds(matched.length >= 2 ? matched.map(run => run.id) : available.slice(0, 3).map(run => run.id));
  }, [available, ids]);
  useEffect(() => { try { localStorage.setItem('wms.embodied.ids', JSON.stringify(ids)); } catch { /* optional */ } setTime(0); setPlaying(false); }, [ids]);
  const first = useVisual(ids[0] ?? '', revision), second = useVisual(ids[1] ?? '', revision), third = useVisual(count === 3 ? ids[2] ?? '' : '', revision);
  const datasets = [first, second, third].slice(0, count);
  const selected = ids.slice(0, count).map(id => available.find(run => run.id === id));
  const duration = Math.max(0, ...datasets.map(data => data.frames?.duration_s ?? 0));
  const warnings = comparisonWarnings(datasets.map(data => data.episode));
  const running = selected.some(run => run && !terminal(run.state));
  const readyCount = datasets.filter(data => data.frames?.frames.length).length;
  const seek = useCallback((next: number) => { setTime(next); setMode('replay'); setPlaying(false); }, []);
  useEffect(() => {
    if (!playing || mode !== 'replay' || duration <= 0) return;
    let handle = 0, previous = performance.now();
    const tick = (now: number) => {
      const delta = Math.min((now - previous) / 1000, 0.15); previous = now;
      setTime(current => Math.min(duration, current + delta * speed));
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing, mode, speed, duration]);
  useEffect(() => { if (duration > 0 && time >= duration) setPlaying(false); }, [time, duration]);
  const act = async (fn: () => Promise<void>) => { setBusy(true); setError(''); try { await fn(); } catch (error) { setError(message(error)); } finally { setBusy(false); } };
  const prepare = (id: string) => void act(async () => { await post(pathOf(id) + '/prepare'); setRevision(value => value + 1); await onRefresh(); });
  const start = () => void act(async () => {
    if (!Number.isInteger(seed) || seed < 0 || !Number.isInteger(maxCycles) || maxCycles < 1 || maxCycles > 100 || !Number.isInteger(maxCalls) || maxCalls < 1 || maxCalls > 100 || threshold < 0 || threshold > 1) throw new Error('请检查种子、步数、调用上限与阈值。');
    setNotice(''); setPlaying(false); setMode('live');
    const created: Run[] = [];
    try {
      for (const [index, provider] of providers.slice(0, count).entries()) {
        const config = await api<Config>('/recipes/embodied-jev-' + provider);
        const run = await post<Run>('/runs', { ...config, name: `过程对比 ${String.fromCharCode(65 + index)} · ${titles[provider]} · ${tasks[task]}`, seed, params: { ...config.params, task, preview, threshold, control_mode: control, observation_mode: observation, max_cycles: maxCycles, max_api_calls: provider === 'baseline' ? 0 : maxCalls, speed: 1.5 } });
        created.push(run); setNewRuns(previous => [run, ...previous]); setIds(created.map(item => item.id));
      }
      setNotice(`已提交 ${created.length} 路实验，平台逐路执行。运行中的场景自动更新，完成后可同步回放。`);
      setTab('history');
    } catch (error) { setNotice(`已创建 ${created.length} 路实验；其余提交失败。可停止已创建的实验或稍后重试。`); throw error; }
    finally { await onRefresh(); }
  });
  const cancel = () => void act(async () => { for (const run of selected) if (run && !terminal(run.state)) await post('/runs/' + run.id + '/cancel'); await onRefresh(); setNotice('已请求停止所选未完成实验。已有记录保留。'); });
  const jump = (direction: -1 | 1) => {
    const stops = [...new Set(datasets.flatMap(data => data.episode && data.frames ? data.episode.history.flatMap(record => { const value = cycleStartTime({ frames: data.frames!.frames }, record.cycle); return value === null ? [] : [value]; }) : []))].sort((a, b) => a - b);
    seek(direction === 1 ? stops.find(value => value > time + 0.001) ?? duration : stops.filter(value => value < time - 0.001).slice(-1)[0] ?? 0);
  };
  const download = () => {
    const blob = new Blob([JSON.stringify({ format: 'wm-safetylab-process-comparison-v1', run_ids: ids.slice(0, count), alignment: 'relative_simulation_seconds', cursor_s: time, conditions: datasets.map(data => { const ep = data.episode; return ep ? { task: ep.task, seed: ep.seed, scene_hash: ep.scene_hash, provider: ep.provider, model: ep.model, control_mode: ep.control_mode, observation_mode: ep.observation_mode, preview: ep.preview, threshold: ep.threshold } : null; }) }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'process-comparison.json'; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <div className="embodied-workbench">
    <header className="ev-heading"><div><p className="ev-eyebrow">EMBODIED JEV / PROCESS COMPARISON</p><h1>同一个场景，不同的决策</h1><p>并排观察机器人如何理解、选择、执行，再回到每一步的证据。</p></div><div className="ev-heading-actions"><button className="secondary" onClick={download} disabled={!ids.length}>导出对比清单 ↓</button><span>MuJoCo · Panda · {count} 路对比</span></div></header>
    <section className="ev-setup"><div className="ev-setup-top"><div className="ev-tabs"><button aria-pressed={tab === 'history'} onClick={() => setTab('history')}>历史实验</button><button aria-pressed={tab === 'new'} onClick={() => setTab('new')}>新建对比</button></div><label className="ev-inline">对比通道<select aria-label="对比通道数量" value={count} disabled={busy || running} onChange={event => { setCount(Number(event.target.value)); setPlaying(false); }}><option value={2}>2 路</option><option value={3}>3 路</option></select></label></div>
      {tab === 'history' ? <div className="ev-selectors" style={{ '--lane-count': count } as React.CSSProperties}>{Array.from({ length: count }, (_, index) => <label key={index}><span>通道 {String.fromCharCode(65 + index)}</span><select aria-label={`通道 ${index + 1} 实验`} value={ids[index] ?? ''} disabled={busy} onChange={event => setIds(previous => { const next = [...previous]; next[index] = event.target.value; return next; })}><option value="">选择实验…</option>{available.map(run => <option key={run.id} value={run.id}>{labelOf(run)}</option>)}</select></label>)}</div> : <div className="ev-new-form">
        <div className="ev-form-grid"><label>任务<select aria-label="任务" value={task} onChange={e => setTask(e.target.value)}>{Object.entries(tasks).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label><label>随机种子<input aria-label="随机种子" type="number" min={0} value={seed} onChange={e => setSeed(Number(e.target.value))} /></label>{providers.slice(0, count).map((provider, index) => <label key={index}>通道 {String.fromCharCode(65 + index)}<select aria-label={`通道 ${index + 1} 模型`} value={provider} onChange={e => setProviders(previous => previous.map((item, i) => i === index ? e.target.value : item))}>{Object.entries(titles).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>)}</div>
        <details><summary>共同执行参数</summary><div className="ev-form-grid"><label>控制模式<select value={control} onChange={e => setControl(e.target.value)}><option value="skills">技能动作</option><option value="incremental">增量控制</option><option value="hierarchical">分层控制</option></select></label><label>观测模式<select value={observation} onChange={e => setObservation(e.target.value)}><option value="privileged">结构化仿真状态</option><option value="rgbd">RGB-D 感知</option></select></label><label>决策阈值<input type="number" min={0} max={1} step={0.05} value={threshold} onChange={e => setThreshold(Number(e.target.value))} /></label><label>最多动作周期<input type="number" min={1} max={100} value={maxCycles} onChange={e => setMaxCycles(Number(e.target.value))} /></label><label>每路 API 调用上限<input type="number" min={1} max={100} value={maxCalls} onChange={e => setMaxCalls(Number(e.target.value))} /></label><label className="ev-check"><input type="checkbox" checked={preview} onChange={e => setPreview(e.target.checked)} />动作预演</label></div></details>
        <div className="ev-form-submit"><p>共用任务、种子和执行参数。Chat / JEV 使用后台已配置的 API，调用计入各自额度。</p><button className="primary" disabled={busy || running} onClick={start}>{busy ? '提交中…' : '开始对比（逐路排队）'}</button></div>
      </div>}
    </section>
    {error && <div className="ev-warning" role="alert">{error}</div>}{notice && <div className="ev-notice" role="status">{notice}</div>}
    {!!warnings.length && <details className="ev-warning"><summary>对比条件需要注意 · {warnings.length}</summary><ul>{warnings.map(warning => <li key={warning}>{warning}</li>)}</ul></details>}
    <section className="ev-playback" aria-label="同步回放控制"><div className="ev-playback-top"><div className="ev-tabs"><button aria-pressed={mode === 'replay'} onClick={() => { setMode('replay'); setPlaying(false); }}>同步回放</button><button aria-pressed={mode === 'live'} onClick={() => { setMode('live'); setPlaying(false); }}>实时进程</button></div><div className="ev-playback-meta"><span>{readyCount}/{count} 路回放已就绪</span><button className="ev-text-button" onClick={() => setCamera(camera === 'top' ? 'orbit' : 'top')}>{camera === 'top' ? '全部自由视角' : '全部俯视'}</button>{running && <button className="danger-button" disabled={busy} onClick={cancel}>停止所选实验</button>}</div></div>
      <div className="ev-transport"><button aria-label="回到起点" disabled={!duration || mode !== 'replay'} onClick={() => seek(0)}>↤</button><button aria-label="上一个动作" disabled={!duration || mode !== 'replay'} onClick={() => jump(-1)}>‹</button><button className="ev-play" aria-label={playing ? '暂停回放' : '播放回放'} disabled={!duration || mode !== 'replay'} onClick={() => { if (time >= duration) setTime(0); setPlaying(value => !value); }}>{playing ? 'Ⅱ' : '▶'}</button><button aria-label="下一个动作" disabled={!duration || mode !== 'replay'} onClick={() => jump(1)}>›</button><div className="ev-scrubber"><input aria-label="同步仿真时间" type="range" min={0} max={duration || 1} step="any" value={Math.min(time, duration)} disabled={!duration || mode !== 'replay'} onChange={event => seek(Number(event.target.value))} /><div><span>{time.toFixed(2)} s</span><span>{duration.toFixed(2)} s</span></div></div><select aria-label="回放倍速" value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option></select></div>
      <p>{mode === 'replay' ? '按相对仿真时间同步；较短回合停在末帧。暂停只影响回放，不会停止实验。' : '每路显示最近收到的场景快照；排队执行不按墙钟同步。完成后切换同步回放查看全程。'} 候选置信度不等于任务成功概率或风险概率。</p>
    </section>
    <div className="ev-lanes" style={{ '--lane-count': count } as React.CSSProperties}>{datasets.map((data, index) => <Lane key={ids[index] || `empty-${index}`} index={index} run={selected[index]} data={data} time={mode === 'live' && data.frames ? data.frames.duration_s : time} mode={mode} camera={camera} onCamera={setCamera} onSeek={seek} onPrepare={() => prepare(ids[index])} onRun={() => ids[index] && onRun(ids[index])} onResults={() => ids[index] && onResults(ids[index])} />)}</div>
    <p className="ev-provenance">交互场景来自原实验几何与记录姿态。历史回放不重新运行策略或物理仿真；决策按动作段关联，具体请求时刻未记录时会明确标注。场景下方显示仿真状态，实际模型输入见完整证据。</p>
  </div>;
}