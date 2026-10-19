/** Pure mappings for archived embodied-jev evidence. No API or renderer imports. */
export type Observation = Record<string, unknown> & {
  sim_seconds?: number; tcp?: number[]; object?: number[]; destination?: number[];
  success?: boolean; forbidden_contact?: boolean; gripper?: string; held?: boolean;
};
export interface Decision {
  choice?: string;
  probabilities?: Record<string, number> | null;
  selected_probability?: number | null;
  provider_confidence?: number | null;
  provider?: string; model?: string; model_call?: boolean; latency_ms?: number;
  channel_decisions?: Record<string, Decision>;
  [key: string]: unknown;
}
export interface Candidate {
  id: string; label?: string; phase?: string; target?: number[] | null;
  gripper?: string | null; seconds?: number; admitted?: boolean;
  rejection?: string | null; preview?: Record<string, unknown> | null;
  [key: string]: unknown;
}
export interface Frame {
  time: number; cycle?: number | null; phase?: string | null;
  observation?: Observation; qpos?: number[]; evaluation_target?: number[];
  [key: string]: unknown;
}
export interface VisualFrame extends Frame {
  index?: number; sim_time?: number;
  positions: number[][]; rotations: number[][];
}
export interface HistoryRecord {
  cycle: number; phase?: string | null; label?: string;
  intent?: Decision | null; decision?: Decision | null;
  action?: Candidate | null; candidates?: Candidate[];
  before?: Observation | null; after?: Observation | null;
  executed?: boolean; rejected_count?: number; rejection?: string | null;
  decision_inputs?: Record<string, unknown> | null;
  [key: string]: unknown;
}
export interface Episode {
  format?: string; id?: string; task?: string; seed?: number; scene_hash?: string;
  provider?: string; model?: string; control_mode?: string; observation_mode?: string;
  preview?: boolean; threshold?: number; status?: string; success?: boolean;
  history: HistoryRecord[]; frames: Frame[];
  last_decision?: Decision | null; last_intent?: Decision | null;
  last_decision_inputs?: Record<string, unknown> | null;
  candidates?: Candidate[]; events?: Record<string, unknown>[];
  model_calls?: number; wall_seconds?: number;
  [key: string]: unknown;
}
export interface FramesResponse {
  schema_version: string; run_id: string; time_basis: 'simulation_seconds';
  duration_s: number; frames: VisualFrame[];
}
export interface LiveSnapshot {
  run_id: string; status: string; stage?: string; cycle?: number;
  frame?: VisualFrame | null; wall_seconds?: number; history?: HistoryRecord[];
  last_decision?: Decision | null; last_intent?: Decision | null;
  candidates?: Candidate[]; events?: Record<string, unknown>[];
  model_calls?: number; updated_at?: string;
  [key: string]: unknown;
}
export interface FrameSelection {
  frame: Frame | null; index: number;
  /** Requested elapsed simulation time, clamped to this lane's duration. */
  elapsed: number;
  /** Elapsed time of the actual preceding recorded frame (no interpolation). */
  frameElapsed: number;
  duration: number; ended: boolean; clamped: boolean;
}
export interface DecisionView {
  cycle: number | null; phase: string | null;
  intent: Decision | null; decision: Decision | null;
  candidates: Candidate[]; action: Candidate | null;
  before: Observation | null; after: Observation | null;
  completed: boolean | null; executed: boolean | null;
  historyIndex: number | null; source: 'history' | 'terminal';
  decisionRole: 'action' | 'intent' | 'unknown';
  timingLabel: string; timingUncertain: boolean;
}
const ENTRYPOINT = 'wms.engines.embodied_jev.native:run_episode';
const RECIPES = new Set(['embodied-jev-baseline', 'embodied-jev-chat', 'embodied-jev-jev']);
const EPSILON = 1e-9;
const object = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

/** Select native source experiments, never evaluation jobs or replay/cache jobs. */
export function isEmbodiedRun(value: unknown): boolean {
  const run = object(value);
  if (!run || run.task !== 'native' || run.source_run) return false;
  const config = object(run.config);
  if (!config || config.task === 'evaluate' || config.source_run) return false;
  if (typeof config.entrypoint === 'string') return config.entrypoint === ENTRYPOINT;
  return typeof config.recipe === 'string' && RECIPES.has(config.recipe);
}

/** Preserve reported values without inventing or renormalizing probabilities. */
export function normalizeDecision(value: Decision | null | undefined): Decision | null {
  if (!value) return null;
  const probabilities = value.probabilities;
  const valid = object(probabilities) && Object.keys(probabilities!).length > 0
    && Object.values(probabilities!).every(item => finite(item) && item >= 0 && item <= 1);
  const selected = value.selected_probability;
  return {
    ...value,
    probabilities: valid ? { ...probabilities } as Record<string, number> : null,
    selected_probability: finite(selected) && selected >= 0 && selected <= 1 ? selected : null,
    ...(value.channel_decisions ? { channel_decisions: Object.fromEntries(
      Object.entries(value.channel_decisions).map(([key, decision]) => [key, normalizeDecision(decision)!]),
    ) } : {}),
  };
}

/** Binary seek on each lane's own zero-origin simulation clock. */
export function frameAtTime<T extends Frame>(frames: readonly T[], time: number): FrameSelection & { frame: T | null } {
  if (!finite(time)) throw new RangeError('Replay time must be finite');
  if (!frames.length) return { frame: null, index: -1, elapsed: 0, frameElapsed: 0,
    duration: 0, ended: false, clamped: time !== 0 };
  for (let index = 0; index < frames.length; index++) {
    if (!finite(frames[index].time) || (index > 0 && frames[index].time < frames[index - 1].time)) {
      throw new RangeError('Recorded frame times must be finite and nondecreasing');
    }
  }
  const origin = frames[0].time;
  const duration = Math.max(0, frames[frames.length - 1].time - origin);
  const elapsed = Math.min(duration, Math.max(0, time));
  let low = 0, high = frames.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (frames[mid].time - origin <= elapsed + EPSILON) low = mid + 1;
    else high = mid;
  }
  const index = Math.max(0, low - 1);
  return { frame: frames[index], index, elapsed,
    frameElapsed: Math.max(0, frames[index].time - origin), duration,
    ended: time >= duration - EPSILON, clamped: time < 0 || time > duration + EPSILON };
}

export function episodeDuration(episode: Pick<Episode, 'frames'>): number {
  return frameAtTime(episode.frames, 0).duration;
}

/** Jump to the first recorded frame of this cycle; never invent a decision timestamp. */
export function cycleStartTime(episode: Pick<Episode, 'frames'>, cycle: number): number | null {
  const frame = episode.frames.find(item => item.cycle === cycle);
  if (!frame || !episode.frames.length) return null;
  return Math.max(0, frame.time - episode.frames[0].time);
}

function pendingCandidates(episode: Episode, role: 'intent' | 'action'): Candidate[] {
  if (episode.candidates?.length) return episode.candidates;
  const inputs = object(episode.last_decision_inputs);
  const request = object(inputs?.[role === 'intent' ? 'phase' : 'action']);
  const spec = object(request?.decision);
  const criteria = object(spec?.criteria);
  if (!criteria) return [];
  return Object.entries(criteria).map(([id, description]) => {
    let details: Record<string, unknown> | null = null;
    if (typeof description === 'string') {
      try { details = object(JSON.parse(description)); } catch { /* plain description */ }
    }
    return { ...(details ?? {}), id,
      label: typeof details?.label === 'string' ? details.label :
        typeof description === 'string' && !details ? description : id } as Candidate;
  });
}

function terminalDecision(episode: Episode, frame: Frame): DecisionView {
  const intent = normalizeDecision(episode.last_intent);
  const decision = normalizeDecision(episode.last_decision);
  const role = episode.last_decision && episode.last_intent
    && JSON.stringify(episode.last_decision) === JSON.stringify(episode.last_intent) ? 'intent' : 'action';
  return {
    cycle: null, phase: typeof intent?.choice === 'string' ? intent.choice : frame.phase ?? null,
    intent, decision, candidates: pendingCandidates(episode, role),
    action: null, before: null, after: null, completed: false,
    executed: episode.status === 'uncertain' ? false : null,
    historyIndex: null, source: 'terminal', decisionRole: role,
    timingLabel: episode.status === 'uncertain'
      ? '末帧保留的未执行决策；请求时刻未记录'
      : '结束时保留的最后决策；执行范围与请求时刻未完整记录',
    timingUncertain: true,
  };
}

/** Match only the frame's cycle; hide post-action outcomes until their time. */
export function decisionAtFrame(episode: Episode, frame: Frame | null | undefined, index: number): DecisionView | null {
  if (!frame || !Number.isInteger(index) || index < 0 || index >= episode.frames.length) return null;
  const saved = episode.frames[index];
  if (finite(saved.cycle) && finite(frame.cycle) && saved.cycle !== frame.cycle) return null;
  const cycle = finite(frame.cycle) ? frame.cycle : saved.cycle;
  const historyIndex = finite(cycle) && cycle > 0
    ? episode.history.findIndex(record => record.cycle === cycle) : -1;
  // JEV uncertain can keep cycle N in its final frame while last_decision is
  // an unexecuted cycle N+1 proposal. Show it only at the archived endpoint.
  const atEnd = index === episode.frames.length - 1;
  if (atEnd && episode.last_decision && (episode.status === 'uncertain' ||
      (historyIndex < 0 && ['error', 'timeout', 'cancelled', 'stopped', 'exhausted', 'stalled', 'interrupted'].includes(episode.status ?? '')))) {
    return terminalDecision(episode, frame);
  }
  if (historyIndex < 0) return null;
  const record = episode.history[historyIndex];
  const observedTime = frame.observation?.sim_seconds;
  const afterTime = record.after?.sim_seconds;
  const completed = finite(observedTime) && finite(afterTime) ? afterTime <= observedTime + 1e-6 : null;
  const knownExecution = typeof record.executed === 'boolean' ? record.executed
    : episode.control_mode === 'skills' && record.action?.admitted === true ? true : null;
  return {
    cycle: record.cycle, phase: record.phase ?? frame.phase ?? null,
    intent: normalizeDecision(record.intent), decision: normalizeDecision(record.decision),
    candidates: record.candidates ?? [], action: record.action ?? null,
    before: record.before ?? null, after: completed === true ? record.after ?? null : null,
    completed, executed: completed === true ? knownExecution : null,
    historyIndex, source: 'history', decisionRole: 'action',
    timingLabel: '该动作段记录；非逐请求实时记录', timingUncertain: true,
  };
}

const COMPARISON_FIELDS = [
  ['task', '任务'], ['seed', '随机种子'], ['scene_hash', '场景哈希'],
  ['control_mode', '控制模式'], ['observation_mode', '观测模式'],
  ['preview', '预演开关'], ['threshold', '决策阈值'],
] as const;

/** Differences and missing evidence both prevent claiming a matched comparison. */
export function comparisonWarnings(episodes: readonly (Episode | null | undefined)[]): string[] {
  // A pending/empty lane is not evidence, but must retain its screen position.
  const loaded = episodes.flatMap((episode, index) => episode ? [{ episode, lane: index + 1 }] : []);
  if (loaded.length < 2) return [];
  const warnings: string[] = [];
  for (const [field, label] of COMPARISON_FIELDS) {
    const values = loaded.map(({ episode, lane }) => ({ value: episode[field], lane }));
    const known = (value: unknown) => field === 'preview' ? typeof value === 'boolean'
      : field === 'seed' || field === 'threshold' ? finite(value)
      : typeof value === 'string' && value.length > 0;
    const missing = values.flatMap(({ value, lane }) => known(value) ? [] : [lane]);
    if (missing.length) warnings.push(`${label}缺失（通道 ${missing.join('、')}），无法确认条件一致。`);
    const entries = values.flatMap(({ value, lane }) => known(value) ? [{
      value, text: `通道 ${lane}：${typeof value === 'boolean' ? value ? '开启' : '关闭' : value}`,
    }] : []);
    if (new Set(entries.map(item => item.value)).size > 1) {
      warnings.push(`${label}不同（${entries.map(item => item.text).join('；')}），不能作为相同条件对照。`);
    }
  }
  return warnings;
}

