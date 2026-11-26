// Episode replay: recorded baseline/current cameras share a step axis (including the final frame).
// True-state trajectories, plots and timeline come from the recorded trace; no video or 3D is synthesized.
import { useEffect, useMemo, useRef, useState } from 'react';
import { redteam, type TraceStep } from '../api/redteam';
import { clamp, fmt, HAZARDS, type HazardDisplay } from '../ui';

type Vec = number[];
type State = Record<string, unknown>;
const num = (v: unknown): Vec | null => Array.isArray(v) && v.every(x => typeof x === 'number') ? v as Vec : null;
const dist = (a: Vec | null, b: Vec | null) => a && b ? Math.hypot(a[0] - b[0], a[1] - b[1], (a[2] ?? 0) - (b[2] ?? 0)) : null;

/** Gripper, main object and goal of a state, for the toy (gripper/cube/goal) and simulator (tcp_xyz/objects) layouts. */
function geometry(s: State) {
  const objects = (s.objects && typeof s.objects === 'object' ? s.objects : s.cube ? { cube: s.cube } : {}) as Record<string, unknown>;
  const tcp = num(s.gripper) ?? num(s.tcp_xyz);
  const obj = num(objects.cube) ?? num(objects.source) ?? num(Object.values(objects)[0]);
  const goal = num(s.goal) ?? num(objects.target);
  return { tcp, obj, goal, held: Boolean(s.holding ?? s.grasped) };
}

export interface Series {
  n: number; tcp: (Vec | null)[]; obj: (Vec | null)[]; goal: (Vec | null)[];
  speed: (number | null)[]; dObj: (number | null)[]; dGoal: (number | null)[];
  best: (number | null)[]; mean: (number | null)[]; chosen: (number | null)[]; residual: (number | null)[];
  hazards: string[][]; stages: Record<string, { plugin: string; role: string }[]>[]; takeover: (string | null)[]; phase: (string | null)[];
}

export function series(steps: TraceStep[]): Series {
  const g = steps.map(s => geometry(s.state_before ?? {})), after = steps.map(s => geometry(s.state_after ?? {}));
  const positions = after.length ? [...g, after[after.length - 1]] : g;
  const planEvent = (s: TraceStep) => (s.events ?? []).find(e => e.stage === 'S4' && e.data && typeof (e.data as State).best === 'number')?.data as State | undefined;
  return {
    n: steps.length, tcp: positions.map(x => x.tcp), obj: positions.map(x => x.obj), goal: positions.map(x => x.goal),
    speed: g.map((x, i) => dist(x.tcp, after[i].tcp)),
    dObj: g.map(x => dist(x.tcp, x.obj)),
    dGoal: steps.map((s, i) => typeof (s.state_before as State)?.distance_to_goal === 'number' ? (s.state_before as State).distance_to_goal as number : dist(g[i].obj, g[i].goal)),
    best: steps.map(s => (planEvent(s)?.best as number) ?? null), mean: steps.map(s => (planEvent(s)?.mean as number) ?? null),
    chosen: steps.map(s => typeof s.policy_info?.chosen_score === 'number' ? s.policy_info.chosen_score as number : null),
    residual: steps.map((s, i) => { const p = num(s.policy_info?.predicted_next); return p && after[i].tcp ? dist(p.slice(0, 2), after[i].tcp!.slice(0, 2)) : null; }),
    hazards: steps.map(s => (s.hazards ?? []).map(h => String(h.id))),
    stages: steps.map(s => { const out: Record<string, { plugin: string; role: string }[]> = {};
      for (const e of s.events ?? []) if (e.role !== 'monitor') (out[String(e.stage)] ??= []).push({ plugin: String(e.plugin), role: String(e.role) });
      return out; }),
    takeover: steps.map(s => s.takeover ? String((s.takeover as State).kind ?? 'takeover') : null),
    phase: steps.map(s => typeof s.policy_info?.phase === 'string' ? s.policy_info.phase as string : null),
  };
}

let colors: Record<string, string> = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  colors = Object.fromEntries(['surface', 'surface-2', 'ink', 'muted', 'faint', 'line', 'accent', 'accent-soft', 'bad', 'bad-soft', 'warn', 'high', 'good', 'good-soft', 'mon', 'scene', 'scene-grid', 'table', 'cube', 'goal', 'grip']
    .map(n => [n, cs.getPropertyValue('--' + n).trim()]));
  colors.mono = cs.getPropertyValue('--f-mono');
}
function fit(cv: HTMLCanvasElement): [CanvasRenderingContext2D, number, number] {
  const r = cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const c = cv.getContext('2d')!; c.setTransform(dpr, 0, 0, dpr, 0, 0); return [c, r.width, r.height];
}

function drawScene(cv: HTMLCanvasElement, s: Series, f: number, yDown: boolean, ghost: Series | null, cursor = f) {
  const [c, W, H] = fit(cv);
  c.fillStyle = colors.scene; c.fillRect(0, 0, W, H);
  const pts = [...s.tcp, ...s.obj, ...s.goal, ...(ghost?.tcp ?? [])].filter(Boolean) as Vec[];
  if (!pts.length) { c.fillStyle = colors.muted; c.font = '12px ' + colors.mono; c.fillText('状态中没有位置信息', 12, 20); return; }
  let [x0, x1, y0, y1] = [Math.min(...pts.map(p => p[0])), Math.max(...pts.map(p => p[0])), Math.min(...pts.map(p => p[1])), Math.max(...pts.map(p => p[1]))];
  const span = Math.max(x1 - x0, y1 - y0, 1e-3) * 1.25, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  [x0, x1, y0, y1] = [mx - span / 2, mx + span / 2, my - span / 2, my + span / 2];
  const side = Math.min(W, H) - 16, ox = (W - side) / 2, oy = (H - side) / 2;
  if (side < 20) return;
  const px = (v: Vec) => ox + (v[0] - x0) / span * side, py = (v: Vec) => yDown ? oy + (v[1] - y0) / span * side : oy + (1 - (v[1] - y0) / span) * side;
  c.fillStyle = colors.table; c.fillRect(ox, oy, side, side);
  c.strokeStyle = colors['scene-grid']; c.lineWidth = 1;
  for (let k = 1; k < 10; k++) { c.beginPath(); c.moveTo(ox + k * side / 10, oy); c.lineTo(ox + k * side / 10, oy + side); c.moveTo(ox, oy + k * side / 10); c.lineTo(ox + side, oy + k * side / 10); c.stroke(); }
  const r = side * 0.03, goal = s.goal[f], obj = s.obj[f], tcp = s.tcp[f];
  const trail = (list: (Vec | null)[], upTo: number, color: string, dash: number[], alpha: number) => {
    c.strokeStyle = color; c.globalAlpha = alpha; c.setLineDash(dash); c.lineWidth = 2; c.beginPath(); let on = false;
    list.slice(0, upTo + 1).forEach(p => { if (!p) { on = false; return; } on ? c.lineTo(px(p), py(p)) : c.moveTo(px(p), py(p)); on = true; });
    c.stroke(); c.setLineDash([]); c.globalAlpha = 1;
  };
  if (ghost) trail(ghost.tcp, Math.min(cursor, ghost.tcp.length - 1), colors.muted, [3, 4], 0.5);
  trail(s.tcp, f, colors.accent, [], 1);
  if (goal) { c.fillStyle = colors.goal; c.globalAlpha = .22; c.beginPath(); c.arc(px(goal), py(goal), r * 2, 0, 7); c.fill(); c.globalAlpha = 1;
    c.strokeStyle = colors.goal; c.lineWidth = 2; c.beginPath(); c.arc(px(goal), py(goal), r * 2, 0, 7); c.stroke(); }
  if (obj) { c.fillStyle = colors.cube; c.fillRect(px(obj) - r, py(obj) - r, 2 * r, 2 * r); }
  if (tcp) { c.strokeStyle = colors.grip; c.lineWidth = 2.5; c.strokeRect(px(tcp) - r * 1.4, py(tcp) - r * 1.4, r * 2.8, r * 2.8);
    if (s.hazards[f]?.length) { c.strokeStyle = colors.bad; c.lineWidth = 2; c.beginPath(); c.arc(px(tcp), py(tcp), r * 2.6, 0, 7); c.stroke(); } }
  c.font = '12px ' + colors.mono; c.textAlign = 'left'; let y = 18;
  for (const [t, col] of [[`t = ${f} / ${Math.max(0, s.tcp.length - 1)}${f >= s.n ? ' · 最终状态' : ''}`, colors.ink], [s.phase[f] ? `phase ${s.phase[f]}` : '', colors.muted],
    [s.takeover[f] ? `S5 takeover ${s.takeover[f]}` : '', colors.warn], ...(s.hazards[f] ?? []).map(h => [h, colors.bad])] as [string, string][]) {
    if (t) { c.fillStyle = col; c.fillText(t, 8, y); y += 17; }
  }
  if (ghost) { c.fillStyle = colors.muted; c.font = '11px sans-serif'; c.fillText('虚线：基线同种子', 8, H - 8); }
}

const PL = 44, PR = 10, PT = 8, PB = 18, TL = 112;
interface PlotCfg { series: { arr: (number | null)[]; color: string; w?: number }[]; bad?: (i: number) => boolean; marks?: number[] }
function drawPlot(cv: HTMLCanvasElement, cfg: PlotCfg, f: number, n: number) {
  const [c, W, H] = fit(cv); c.clearRect(0, 0, W, H);
  const vals = cfg.series.flatMap(s => s.arr).filter((v): v is number => v != null);
  let lo = Math.min(0, ...vals), hi = Math.max(...vals, lo + 1e-6); if (lo < 0) { const pad = (hi - lo) * .05; lo -= pad; hi += pad; } else hi *= 1.08;
  const x = (i: number) => PL + i / Math.max(1, n - 1) * (W - PL - PR), y = (v: number) => PT + (1 - (v - lo) / (hi - lo)) * (H - PT - PB);
  if (cfg.bad) { c.fillStyle = colors['bad-soft']; for (let i = 0; i < n; i++) if (cfg.bad(i)) c.fillRect(x(i) - .5, PT, (W - PL - PR) / Math.max(1, n - 1) + 1, H - PT - PB); }
  c.font = '10px ' + colors.mono; c.fillStyle = colors.faint; c.strokeStyle = colors.line; c.lineWidth = 1; c.textAlign = 'right';
  [lo, (lo + hi) / 2, hi].forEach(t => { c.beginPath(); c.moveTo(PL, y(t)); c.lineTo(W - PR, y(t)); c.stroke(); c.fillText(fmt(t, Math.abs(hi) < 0.1 ? 3 : 2), PL - 6, y(t) + 3); });
  c.textAlign = 'center'; const st = Math.max(1, Math.ceil(n / 6 / 5) * 5); for (let s = 0; s < n; s += st) c.fillText(String(s), x(s), H - 4);
  cfg.series.forEach(sr => { c.strokeStyle = sr.color; c.lineWidth = sr.w ?? 1.6; c.beginPath(); let on = false;
    sr.arr.forEach((v, i) => { if (v == null) { on = false; return; } on ? c.lineTo(x(i), y(v)) : c.moveTo(x(i), y(v)); on = true; }); c.stroke();
    const v = sr.arr[f]; if (v != null) { c.fillStyle = sr.color; c.beginPath(); c.arc(x(f), y(v), 3.5, 0, 7); c.fill(); } });
  (cfg.marks ?? []).forEach(i => { c.fillStyle = colors.warn; c.fillRect(x(i) - .5, PT, 1, H - PT - PB); });
  c.strokeStyle = colors.accent; c.lineWidth = 1.5; c.beginPath(); c.moveTo(x(f), PT - 4); c.lineTo(x(f), H - PB); c.stroke();
}

type Row = [string, (i: number) => [string, string]];
export function timelineRows(s: Series, hazards: HazardDisplay[] = HAZARDS): Row[] {
  const used = ['ENV', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'].filter(st => s.stages.some(m => m[st]?.length));
  const hazardTypes = new Map(hazards.map(h => [h.id, h]));
  for (const id of s.hazards.flat()) if (!hazardTypes.has(id)) hazardTypes.set(id, { id, name: id, color: 'warn' });
  return [
    ...[...hazardTypes.values()].filter(h => s.hazards.some(l => l.includes(h.id))).map(h => [h.id, (i: number) => s.hazards[i]?.includes(h.id) ? [h.color, ''] : ['ok', '']] as Row),
    ...used.map(st => [st + ' 插件', (i: number) => { const e = s.stages[i][st]?.[0]; return e ? [e.role === 'attack' ? 'bad' : e.role === 'defense' ? 'good' : 'warn', e.plugin] : ['ok', '']; }] as Row),
    ['接管', i => s.takeover[i] ? ['warn', s.takeover[i]!] : ['ok', '']],
    ...(s.phase.some(Boolean) ? [['阶段', (i: number) => [s.phase[i] ? 'phase' : 'ok', s.phase[i] ?? '']] as Row] : []),
  ];
}
function drawTimeline(cv: HTMLCanvasElement, s: Series, f: number, axisLength = s.n, hazards: HazardDisplay[] = HAZARDS) {
  const rows = timelineRows(s, hazards); cv.style.height = (rows.length * 28 + 8) + 'px';
  const [c, W, H] = fit(cv); c.clearRect(0, 0, W, H);
  const n = s.n, x = (i: number) => TL + i / Math.max(1, axisLength - 1) * (W - TL - PR), cw = (W - TL - PR) / Math.max(1, axisLength - 1);
  const fill: Record<string, string> = { bad: colors.bad, warn: colors.warn, high: colors.high, ok: colors['surface-2'], good: colors.good, phase: colors['accent-soft'] };
  const ink: Record<string, string> = { bad: '#fff', warn: '#fff', high: '#fff', good: '#fff', phase: colors.accent, ok: colors.muted };
  rows.forEach(([label, at], ri) => {
    const y0 = 4 + ri * 28; c.font = '11px ' + colors.mono; c.fillStyle = colors.muted; c.textAlign = 'left'; c.fillText(label, 0, y0 + 15);
    let i = 0;
    while (i < n) { const [k, t] = at(i); let j = i; while (j + 1 < n && at(j + 1)[0] === k && at(j + 1)[1] === t) j++;
      c.fillStyle = fill[k] ?? colors['surface-2']; c.beginPath(); c.roundRect(x(i), y0, Math.max(2, x(j) - x(i) + cw), 22, 3); c.fill();
      if (t && x(j) - x(i) + cw > c.measureText(t).width + 10) { c.fillStyle = ink[k] ?? colors.muted; c.fillText(t, x(i) + 5, y0 + 15); }
      i = j + 1; }
  });
  c.strokeStyle = colors.accent; c.lineWidth = 1.5; c.beginPath(); c.moveTo(x(f) + cw / 2, 0); c.lineTo(x(f) + cw / 2, H); c.stroke();
}

export interface ReplayBaseline {
  episode: string; seed: number; arm: string; steps: TraceStep[]; cameras: Record<string, number>; endReason: string;
}

/** Frame counts include the terminal observation; never request an out-of-range frame. */
export function replayPosition(step: number, count: number) {
  return { index: count > 0 ? clamp(step, 0, count - 1) : null, ended: count > 0 && step >= count - 1 };
}

function CameraView({ experiment, episode, label, seed, arm, steps, cameras, cursor, endReason }: ReplayBaseline & {
  experiment: string; label: string; cursor: number;
}) {
  const names = Object.keys(cameras).filter(k => cameras[k] > 0);
  const [selected, setSelected] = useState('');
  const [failedUrl, setFailedUrl] = useState('');
  const camera = names.includes(selected) ? selected : names[0];
  const count = camera ? cameras[camera] : 0;
  const { index, ended } = replayPosition(cursor, count);
  const url = camera && index !== null ? redteam.frameUrl(experiment, episode, camera, index) : '';
  const planner = steps.some(s => s.policy_info?.executor === 'official_motion_planner');
  const statePolicy = steps.some(s => s.policy_info?.observation_kind === 'privileged_state' || s.policy_info?.executor === 'tdmpc2');
  const observerOnly = camera === 'render_camera' || steps.some(s => s.policy_info?.uses_images === false);
  const memoryvla = steps.find(s => s.policy_info?.executor === 'memoryvla')?.policy_info;
  const memoryNote = memoryvla?.memory === true ? '记忆开启（回合内保留）'
    : memoryvla?.memory === false ? '记忆关闭（每步清空记忆库）' : '记忆状态未记录';
  const cameraNote = planner ? '仿真相机记录；官方规划器使用真值决策，不使用图像。'
    : statePolicy ? '状态向量决策；画面仅用于记录，不作为模型视觉输入。'
    : observerOnly ? '仿真相机仅用于记录，不作为模型视觉输入。'
    : memoryvla ? `MemoryVLA · RGB + 指令决策 · ${memoryNote}。画面为 S1 处理后的 seen 帧；${index !== null && index >= steps.length ? '末帧为动作后记录，不再送入策略。' : '动作前的帧实际送入策略。'}`
    : '回合相机观测记录（S1 处理后）。';
  return <figure aria-label={`${label} · ${arm} · seed ${seed}`}>
    <div className="card-h"><h2>{label} · {arm} · seed {seed}</h2><span className="right">
      {ended && <span className="pill idle">画面已结束 · 停留末帧</span>}</span></div>
    {url && failedUrl !== url ? <img alt={`${label} ${arm} seed ${seed} · ${camera} · 记录帧 ${index}`} src={url} onError={() => setFailedUrl(url)} />
      : <div className="empty-state" style={{ aspectRatio: '1', display: 'grid', placeContent: 'center' }}>
          {url ? `记录帧 ${index} 加载失败` : '此回合没有保存相机帧'}<span className="hint">仍可查看下方记录的轨迹与事件</span></div>}
    <figcaption style={{ flexWrap: 'wrap' }}>
      <span className="mono">{camera ?? '无相机'}{index !== null ? ` · 帧 ${index} / ${count - 1}` : ''}</span>
      {names.length > 1 && <select aria-label={`${label}相机`} value={camera} onChange={e => setSelected(e.target.value)}>{names.map(k => <option key={k}>{k}</option>)}</select>}
      <span style={{ flexBasis: '100%' }}>{cameraNote}结束原因：{endReason}</span>
    </figcaption>
  </figure>;
}

export default function Replay({ experiment, episode, seed, arm, endReason, steps, cameras, baseline, hazards = HAZARDS, yDown, ghost, frame, onFrame }: {
  experiment: string; episode: string; seed: number; arm: string; endReason: string; steps: TraceStep[]; cameras: Record<string, number>; baseline: ReplayBaseline | null; yDown: boolean;
  hazards?: HazardDisplay[]; ghost: TraceStep[] | null; frame: number; onFrame: React.Dispatch<React.SetStateAction<number>>;
}) {
  const s = useMemo(() => series(steps), [steps]);
  const g = useMemo(() => ghost ? series(ghost) : null, [ghost]);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const scene = useRef<HTMLCanvasElement>(null), timeline = useRef<HTMLCanvasElement>(null);
  const plots = { V: useRef<HTMLCanvasElement>(null), O: useRef<HTMLCanvasElement>(null), G: useRef<HTMLCanvasElement>(null), P: useRef<HTMLCanvasElement>(null), R: useRef<HTMLCanvasElement>(null) };
  const lastFrame = Math.max(0, s.n, baseline?.steps.length ?? 0,
    ...Object.values(cameras).map(n => n - 1), ...Object.values(baseline?.cameras ?? {}).map(n => n - 1));
  const cursor = clamp(frame, 0, lastFrame), axisLength = lastFrame + 1;
  const sceneFrame = clamp(cursor, 0, Math.max(0, s.tcp.length - 1));
  const hasPlan = s.best.some(v => v != null) || s.chosen.some(v => v != null), hasRes = s.residual.some(v => v != null);
  const takes = s.takeover.map((t, i) => t ? i : -1).filter(i => i >= 0);

  useEffect(() => {
    readColors();
    const draw = () => {
      if (scene.current) drawScene(scene.current, s, sceneFrame, yDown, g, cursor);
      if (plots.V.current) drawPlot(plots.V.current, { series: [{ arr: s.speed, color: colors.ink }], bad: i => s.hazards[i]?.includes('fast_contact') ?? false, marks: takes }, cursor, axisLength);
      if (plots.O.current) drawPlot(plots.O.current, { series: [{ arr: s.dObj, color: colors.ink }] }, cursor, axisLength);
      if (plots.G.current) drawPlot(plots.G.current, { series: [{ arr: s.dGoal, color: colors.ink }], bad: i => s.hazards[i]?.includes('object_drop') ?? false }, cursor, axisLength);
      if (plots.P.current) drawPlot(plots.P.current, { series: [{ arr: s.mean, color: colors.faint, w: 1.2 }, { arr: s.best, color: colors.ink }, { arr: s.chosen.map((v, i) => v != null && s.best[i] != null && Math.abs(v - s.best[i]!) > 1e-9 ? v : null), color: colors.bad, w: 2 }], marks: takes }, cursor, axisLength);
      if (plots.R.current) drawPlot(plots.R.current, { series: [{ arr: s.residual, color: colors.mon }], marks: takes }, cursor, axisLength);
      if (timeline.current) drawTimeline(timeline.current, s, cursor, axisLength, hazards);
    };
    draw();
    window.addEventListener('resize', draw);
    return () => window.removeEventListener('resize', draw);
  });

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => onFrame(x => Math.min(lastFrame, x + 1)), 125 / rate);
    return () => window.clearInterval(timer);
  }, [playing, rate, lastFrame, onFrame]);
  useEffect(() => { if (playing && cursor >= lastFrame) setPlaying(false); }, [cursor, playing, lastFrame]);

  const seek = (e: React.PointerEvent<HTMLCanvasElement>, left: number) => {
    if (e.buttons !== 1 && e.type !== 'pointerdown') return;
    const r = e.currentTarget.getBoundingClientRect();
    onFrame(clamp(Math.round((e.clientX - r.left - left) / Math.max(1, r.width - left - PR) * lastFrame), 0, lastFrame));
  };
  const plot = (key: keyof typeof plots, title: string, unit: string, value: string, over = false) =>
    <div className="plot"><div className="plot-h"><b>{title}</b><span>{unit}</span><span className={'val mono' + (over ? ' over' : '')}>{value}</span></div>
      <canvas ref={plots[key]} onPointerDown={e => seek(e, PL)} onPointerMove={e => seek(e, PL)} /></div>;
  return <>
    <section className="card sim" style={{ margin: 16 }}>
      <div className="card-h"><h2>{baseline ? '基线与当前回合同步回放' : '当前回合回放'}</h2></div>
      <div className="card-b hint">按相同步号对齐（非墙钟时间），使用保存的真实相机帧，包含动作后的最终帧。较短的一侧停留末帧；最终帧与结束后的步号不补造动作或事件。</div>
        <div className="replay-views">
          {baseline && <CameraView key={baseline.episode} {...baseline} experiment={experiment} label="基线" cursor={cursor} />}
          <CameraView key={episode} experiment={experiment} episode={episode} seed={seed} arm={arm} endReason={endReason} steps={steps} cameras={cameras} label="当前" cursor={cursor} />
        </div>
        <div className="transport">
          <button className="icon-btn" aria-label="上一步" disabled={cursor === 0} onClick={() => onFrame(Math.max(0, cursor - 1))}><svg viewBox="0 0 16 16"><path d="M3 2h2v12H3zM14 2v12L6 8z" /></svg></button>
          <button className="icon-btn" aria-label="播放或暂停" disabled={lastFrame === 0} onClick={() => { if (!playing && cursor >= lastFrame) onFrame(0); setPlaying(p => !p); }}><svg viewBox="0 0 16 16">{playing ? <path d="M3 2h4v12H3zM9 2h4v12H9z" /> : <path d="M4 2l10 6-10 6z" />}</svg></button>
          <button className="icon-btn" aria-label="下一步" disabled={cursor === lastFrame} onClick={() => onFrame(Math.min(lastFrame, cursor + 1))}><svg viewBox="0 0 16 16"><path d="M11 2h2v12h-2zM2 2v12l8-6z" /></svg></button>
          <select aria-label="倍速" value={rate} onChange={e => setRate(+e.target.value)}>{[0.5, 1, 2, 4].map(v => <option key={v} value={v}>{v}×</option>)}</select>
          <input type="range" min={0} max={lastFrame} value={cursor} onChange={e => onFrame(+e.target.value)} aria-label="时间" />
          <span className="clock mono">t {cursor} / {lastFrame}</span>
        </div>
    </section>
    <div className="cols-replay" style={{ padding: 16 }}>
      <section className="card sim"><div className="card-h"><h2>记录的俯视轨迹</h2><span className="hint">当前：实线；同 seed 基线：虚线</span></div>
        <canvas ref={scene} aria-label="俯视场景（仿真真值）" /><div className="card-b hint">由记录的仿真状态绘制的二维示意，非仿真视频或 3D 渲染。</div></section>
      <section className="card">
        {cursor >= s.n && <div className="card-b hint">当前回合在此步没有动作记录；保留历史曲线，不延用上一条事件。</div>}
        {plot('V', 'TCP 位移', '每步', fmt(s.speed[cursor], 3), s.hazards[cursor]?.includes('fast_contact'))}
        {plot('O', '与物体距离', '', fmt(s.dObj[cursor], 3))}
        {plot('G', '物体到目标距离', '', fmt(s.dGoal[cursor], 3))}
        {hasPlan && plot('P', '规划打分', '最高 / 平均 / 被选中（偏离最高时红色）', `${fmt(s.best[cursor])} / ${fmt(s.chosen[cursor])}`, s.chosen[cursor] != null && s.best[cursor] != null && Math.abs(s.chosen[cursor]! - s.best[cursor]!) > 1e-9)}
        {hasRes && plot('R', '预测残差', '|预测 − 实际|', fmt(s.residual[cursor], 4))}
      </section>
    </div>
    <div className="tl" style={{ padding: '0 16px 16px' }}><canvas ref={timeline} onPointerDown={e => seek(e, TL)} onPointerMove={e => seek(e, TL)} /></div>
  </>;
}
