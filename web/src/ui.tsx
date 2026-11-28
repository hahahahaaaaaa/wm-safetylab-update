// Small shared pieces of the red-team UI: labels, tags, pills, intervals.
import type { ReactNode } from 'react';
import type { Catalog, Delta, EnvironmentEntry, Role, Stage } from './api/redteam';

export const ROLE_NAME: Record<Role, string> = { fault: '故障', attack: '攻击', defense: '防御', monitor: '监控' };
export const STAGE_NAME: Record<Stage, string> = { ENV: '仿真世界', S1: '观测 / 指令', S2: '记忆 / 潜状态', S3: '预测', S4: '规划', S5: '动作', S6: '反馈' };
export const STAGES = Object.keys(STAGE_NAME) as Stage[];
export const HAZARDS = [
  { id: 'workspace_exit', name: '离开工作区', color: 'warn' },
  { id: 'object_drop', name: '物体掉落', color: 'high' },
  { id: 'fast_contact', name: '高速接触', color: 'bad' },
];
export interface HazardDisplay { id: string; name: string; color: string; severity?: string; description?: string }

/** Old environments keep their three hazard columns; recorded unknown IDs must remain visible. */
export function environmentHazards(catalog: Catalog, environments: EnvironmentEntry[], recorded: string[] = []): HazardDisplay[] {
  const defaults = HAZARDS.map(h => h.id);
  const ids = new Set([...(environments.length ? environments.flatMap(e => e.hazard_ids ?? defaults) : defaults), ...recorded]);
  return [...ids].map(id => {
    const entry = catalog.hazards.find(h => h.id === id), legacy = HAZARDS.find(h => h.id === id);
    return { id, name: entry?.name ?? legacy?.name ?? id, severity: entry?.severity,
      description: entry?.description ?? '记录中的危险类型；当前目录未提供判定说明。',
      color: legacy?.color ?? (entry?.severity === 'high' ? 'bad' : 'warn') };
  });
}
export const STATE_NAME: Record<string, [string, string]> = {
  queued: ['idle', '排队中'], running: ['run', '运行中'], cancelling: ['warn', '正在取消'], completed: ['good', '已完成'],
  partial_failed: ['warn', '部分失败'], failed: ['bad', '失败'], cancelled: ['idle', '已取消'], interrupted: ['bad', '已中断'],
};

export const fmt = (n: number | null | undefined, p = 2) => n == null || !Number.isFinite(n) ? '—' : Number(n).toFixed(p);
export const pct = (n: number | null | undefined) => n == null ? '—' : Math.round(n * 100) + '%';
export const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
export const cx = (...names: (string | false | null | undefined)[]) => names.filter(Boolean).join(' ');

export const Pill = ({ state }: { state: string }) => {
  const [kind, text] = STATE_NAME[state] ?? ['idle', state];
  return <span className={'pill ' + kind}>{text}</span>;
};
export const RoleTag = ({ role }: { role?: Role | null }) => role ? <span className={'role ' + role}>{ROLE_NAME[role]}</span> : null;
export const StageTag = ({ stage }: { stage?: Stage | null }) => stage ? <span className="st">{stage}</span> : null;
export const Tag = ({ kind, mono, children, title }: { kind?: string; mono?: boolean; children: ReactNode; title?: string }) =>
  <span className={cx('tag', kind, mono && 'm')} title={title}>{children}</span>;
export const PageHead = ({ title, children }: { title: string; children?: ReactNode }) =>
  <div className="page-title"><h1>{title}</h1>{children}</div>;
export const Empty = ({ children }: { children: ReactNode }) => <div className="card empty-state">{children}</div>;
export const ErrorBox = ({ message }: { message: string }) => <div className="errbox" role="alert">{message}</div>;

/** A difference with its 95% interval; `higherIsWorse` colours a significant increase red. */
export function DeltaCell({ d, unit, scale = 1, digits = 0, higherIsWorse }: { d?: Delta | null; unit: string; scale?: number; digits?: number; higherIsWorse: boolean }) {
  if (!d) return <span className="d ns">—</span>;
  const sig = d.low > 0 || d.high < 0;
  const worse = higherIsWorse ? d.estimate > 0 : d.estimate < 0;
  const v = (x: number) => (x > 0 ? '+' : '') + fmt(x * scale, digits);
  return <span className={cx('d', !sig ? 'ns' : worse ? 'worse' : 'better')}>{v(d.estimate)}{unit && ' ' + unit}
    <span className="ci"> [{fmt(d.low * scale, digits)}, {fmt(d.high * scale, digits)}]</span></span>;
}

export function Forest({ d, range, higherIsWorse }: { d?: Delta | null; range: number; higherIsWorse: boolean }) {
  if (!d) return null;
  const pos = (x: number) => clamp(50 + x / range * 50, 0, 100);
  const sig = d.low > 0 || d.high < 0;
  const worse = higherIsWorse ? d.estimate > 0 : d.estimate < 0;
  const color = !sig ? 'var(--faint)' : worse ? 'var(--bad)' : 'var(--good)';
  return <span className="forest" aria-hidden="true"><span className="zero" style={{ left: '50%' }} />
    <span className="bar" style={{ left: pos(d.low) + '%', width: Math.max(1, pos(d.high) - pos(d.low)) + '%', background: color }} />
    <span className="pt" style={{ left: pos(d.estimate) + '%', background: color }} /></span>;
}

export function Seg<T extends string>({ value, options, onChange }: { value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return <div className="seg">{options.map(([k, label]) =>
    <button key={k} aria-pressed={value === k} onClick={() => onChange(k)}>{label}</button>)}</div>;
}

export const dateText = (value: string) => new Date(value).toLocaleString('zh-CN', { hour12: false, month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
