// 3 实验进度: the experiment list, a live view (server-sent progress events), and run logs.
import { useEffect, useMemo, useRef, useState } from 'react';
import type { Go, Route } from '../App';
import { redteam, type Catalog, type EpisodeRow, type Experiment, type Report } from '../api/redteam';
import { cx, dateText, DeltaCell, Empty, ErrorBox, fmt, PageHead, Pill, pct, Seg, Tag } from '../ui';

const ACTIVE = ['queued', 'running', 'cancelling'];
const isActive = (e: Experiment) => ACTIVE.includes(e.state);
const errText = (e: unknown) => e instanceof Error ? e.message : String(e);

function useExperiments() {
  const [items, setItems] = useState<Experiment[] | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let alive = true, timer = 0;
    const load = () => redteam.list().then(p => { if (!alive) return; setItems(p.items); setError('');
      timer = window.setTimeout(load, p.items.some(isActive) ? 2000 : 8000); })
      .catch(e => { if (alive) { setError(errText(e)); timer = window.setTimeout(load, 5000); } });
    load();
    return () => { alive = false; window.clearTimeout(timer); };
  }, []);
  return { items, error, replace: (e: Experiment) => setItems(list => (list ?? []).map(x => x.id === e.id ? e : x)) };
}

export const ExperimentSelect = ({ items, value, onChange }: { items: Experiment[]; value?: string; onChange: (id: string) => void }) =>
  <select aria-label="选择实验" value={value ?? ''} onChange={e => onChange(e.target.value)}>
    {items.map(e => <option key={e.id} value={e.id}>{e.name} · {e.id}{isActive(e) ? '（运行中）' : ''}</option>)}</select>;

const Bar = ({ e }: { e: Experiment }) => <><div className={cx('progress', e.state === 'completed' && 'done', ['failed', 'partial_failed'].includes(e.state) && 'fail')}>
  <i style={{ width: (e.episodes_total ? e.episodes_done / e.episodes_total * 100 : 0) + '%' }} /></div>
  <span className="hint mono">{e.episodes_done} / {e.episodes_total} episode</span></>;

export default function Progress({ catalog, route, go }: { catalog: Catalog; route: Route; go: Go }) {
  const list = useExperiments();
  if (list.error && !list.items) return <ErrorBox message={list.error} />;
  if (!list.items) return <div className="empty-state">加载…</div>;
  if (route.sub === 'live') return <Live items={list.items} route={route} go={go} />;
  if (route.sub === 'logs') return <Logs items={list.items} route={route} go={go} />;
  return <Runs items={list.items} catalog={catalog} go={go} onChange={list.replace} />;
}

function Runs({ items, catalog, go, onChange }: { items: Experiment[]; catalog: Catalog; go: Go; onChange: (e: Experiment) => void }) {
  const [state, setState] = useState<'all' | 'active' | 'completed' | 'failed'>('all');
  const [q, setQ] = useState('');
  const shown = items.filter(e => (state === 'all' || (state === 'active' ? isActive(e) : state === 'failed' ? ['failed', 'partial_failed', 'cancelled'].includes(e.state) : e.state === state))
    && (!q || (e.id + e.name + e.config.model).toLowerCase().includes(q.toLowerCase())));
  const modelName = (id?: string) => catalog.models.find(m => m.id === id)?.name ?? id;
  const cancel = (id: string) => redteam.cancel(id).then(onChange).catch(() => { /* the list refreshes */ });
  return <>
    <PageHead title="运行列表"><button className="btn primary" onClick={() => go({ sec: 'new', sub: 'basic' })}>+ 新建实验</button></PageHead>
    <section className="card"><div className="card-h"><Seg value={state} onChange={setState} options={[['all', '全部'], ['active', '运行中'], ['completed', '完成'], ['failed', '失败 / 取消']]} />
      <span className="right"><input type="search" placeholder="搜索实验、模型" value={q} onChange={e => setQ(e.target.value)} aria-label="搜索" /></span></div>
      <div className="scroll"><table><thead><tr><th>状态</th><th>实验</th><th>模型 · 任务</th><th className="num">对照组</th><th style={{ minWidth: 150 }}>进度</th><th>创建</th><th /></tr></thead><tbody>
        {shown.map(e => { const failed = e.runs.filter(r => r.error);
          return <tr key={e.id}><td><Pill state={e.state} /></td>
            <td><button className="link" style={{ fontSize: 13 }} onClick={() => go({ sec: 'results', sub: 'report', exp: e.id })}>{e.name}</button><div className="hint mono">{e.id}</div>
              {failed.slice(0, 1).map(r => <div key={r.run_id} className="err">{r.arm} · {r.environment}：{r.error}</div>)}</td>
            <td>{modelName(e.config.model)}<div className="hint mono">{(e.config.environments ?? []).length > 1 ? `${e.config.environments!.length} 个任务` : e.config.environments?.[0]}</div></td>
            <td className="num">{e.arms.length}</td><td><Bar e={e} /></td><td className="mono">{dateText(e.created_at)}</td>
            <td style={{ whiteSpace: 'nowrap' }}>{isActive(e) ? <><button className="link" onClick={() => go({ sec: 'progress', sub: 'live', exp: e.id })}>监控</button> · <button className="link" onClick={() => cancel(e.id)}>取消</button></>
              : <><button className="link" onClick={() => go({ sec: 'results', sub: 'report', exp: e.id })}>查看结果</button> · <button className="link" onClick={() => go({ sec: 'progress', sub: 'logs', exp: e.id })}>日志</button></>}</td></tr>; })}
        {!shown.length && <tr><td colSpan={7} className="empty-state">没有符合条件的实验</td></tr>}
      </tbody></table></div></section>
  </>;
}

function useLive(id?: string) {
  const [exp, setExp] = useState<Experiment | null>(null);
  useEffect(() => {
    if (!id) return;
    let source: EventSource | null = null, alive = true;
    redteam.get(id).then(e => { if (!alive) return; setExp(e);
      source = new EventSource(redteam.eventsUrl(id));
      source.addEventListener('progress', m => { const d = JSON.parse((m as MessageEvent).data).data;
        setExp(prev => prev ? { ...prev, state: d.state, episodes_done: d.episodes_done, episodes_total: d.episodes_total, runs: d.runs } : prev); });
      source.addEventListener('state', () => source?.close());
    }).catch(() => setExp(null));
    return () => { alive = false; source?.close(); };
  }, [id]);
  return exp;
}

function Live({ items, route, go }: { items: Experiment[]; route: Route; go: Go }) {
  const live = items.filter(isActive);
  const id = route.exp && items.some(e => e.id === route.exp) ? route.exp : (live[0] ?? items[0])?.id;
  const exp = useLive(id);
  const [report, setReport] = useState<Report | null>(null);
  const [latest, setLatest] = useState<EpisodeRow[]>([]);
  useEffect(() => {
    if (!exp) return;
    redteam.report(exp.id).then(setReport).catch(() => setReport(null));
    redteam.episodes(exp.id, { limit: 500 }).then(p => setLatest(p.items)).catch(() => setLatest([]));
  }, [exp?.id, exp?.episodes_done]);
  if (!items.length) return <><PageHead title="实时监控" /><Empty>还没有实验<br /><button className="btn" style={{ marginTop: 10 }} onClick={() => go({ sec: 'new', sub: 'basic' })}>新建实验</button></Empty></>;
  if (!exp) return <div className="empty-state">加载…</div>;
  const base = report?.arms[0];
  return <>
    <PageHead title="实时监控"><div className="toolbar"><ExperimentSelect items={items} value={exp.id} onChange={v => go({ sec: 'progress', sub: 'live', exp: v })} /><Pill state={exp.state} />
      {isActive(exp) && <button className="btn" onClick={() => redteam.cancel(exp.id)}>取消</button>}
      <button className="btn" onClick={() => go({ sec: 'results', sub: 'report', exp: exp.id })}>结果 →</button></div></PageHead>
    <div className="kpis"><div className="kpi"><span>进度</span><b>{exp.episodes_total ? Math.round(exp.episodes_done / exp.episodes_total * 100) : 0}%</b><small className="mono">{exp.episodes_done} / {exp.episodes_total}</small></div>
      <div className="kpi"><span>基线成功率</span><b>{pct(base?.success.estimate)}</b><small className="mono">n = {base?.episodes ?? 0}</small></div>
      {report?.arms.slice(1).map(a => <div key={a.arm} className={cx('kpi', a.delta_success && a.delta_success.high < 0 && 'bad')}><span>{a.arm} {a.name}</span><b>{pct(a.success.estimate)}</b>
        <small>Δ <DeltaCell d={a.delta_success} unit="pp" scale={100} higherIsWorse={false} /> · 危险 {fmt(a.hazard_steps_mean, 1)}/ep</small></div>)}</div>
    <section className="card"><div className="card-h"><h2>运行</h2><span className="sub">每个对照组 × 任务一个运行</span></div>
      <div className="scroll"><table><thead><tr><th>组</th><th>任务</th><th>状态</th><th style={{ minWidth: 150 }}>进度</th><th>run</th><th /></tr></thead><tbody>
        {exp.runs.map(r => <tr key={r.run_id}><td className="mono">{r.arm}</td><td className="mono">{r.environment}</td><td><Pill state={r.state} />{r.error && <div className="err">{r.error}</div>}</td>
          <td><div className={cx('progress', r.state === 'completed' && 'done', r.state === 'failed' && 'fail')}><i style={{ width: r.episodes_done / Math.max(1, r.episodes_total) * 100 + '%' }} /></div><span className="hint mono">{r.episodes_done} / {r.episodes_total}</span></td>
          <td className="mono hint">{r.run_id}</td><td><button className="link" onClick={() => go({ sec: 'progress', sub: 'logs', exp: exp.id, ep: r.run_id })}>日志</button></td></tr>)}</tbody></table></div></section>
    <section className="card"><div className="card-h"><h2>已完成的 episode</h2><span className="sub">{latest.length}</span></div>
      <pre className="log">{latest.slice(-40).map(e => `${e.arm} ${e.environment} seed=${e.seed} → ${e.success ? `success@${e.success_step}` : e.end_reason}${e.hazard_total ? ' · hazards ' + Object.entries(e.hazard_steps ?? {}).map(([k, v]) => `${k}×${v}`).join(' ') : ''}${e.takeovers ? ` · takeovers ${e.takeovers}` : ''}`).join('\n') || '等待第一个 episode…'}</pre></section>
  </>;
}

function Logs({ items, route, go }: { items: Experiment[]; route: Route; go: Go }) {
  const exp = items.find(e => e.id === route.exp) ?? items[0];
  const runId = exp?.runs.find(r => r.run_id === route.ep)?.run_id ?? exp?.runs[0]?.run_id;
  const [text, setText] = useState('');
  const [error, setError] = useState('');
  const offset = useRef(0);
  const box = useRef<HTMLPreElement>(null);
  const active = exp ? isActive(exp) : false;
  useEffect(() => {
    if (!exp || !runId) return;
    setText(''); offset.current = 0;
    let alive = true, timer = 0;
    const load = () => redteam.logs(exp.id, runId, offset.current).then(c => {
      if (!alive) return; offset.current = c.next_offset; if (c.text) setText(t => t + c.text); setError('');
      if (active) timer = window.setTimeout(load, 1500);
    }).catch(e => alive && setError(errText(e)));
    load();
    return () => { alive = false; window.clearTimeout(timer); };
  }, [exp?.id, runId, active]);
  useEffect(() => { const el = box.current; if (el) el.scrollTop = el.scrollHeight; }, [text]);
  const lines = useMemo(() => text.split('\n'), [text]);
  if (!exp) return <><PageHead title="运行日志" /><Empty>还没有实验</Empty></>;
  return <>
    <PageHead title="运行日志"><div className="toolbar"><ExperimentSelect items={items} value={exp.id} onChange={v => go({ sec: 'progress', sub: 'logs', exp: v, ep: '' })} />
      <select aria-label="选择运行" value={runId} onChange={e => go({ sec: 'progress', sub: 'logs', exp: exp.id, ep: e.target.value })}>
        {exp.runs.map(r => <option key={r.run_id} value={r.run_id}>{r.arm} · {r.environment} · {r.state}</option>)}</select><Pill state={exp.state} /></div></PageHead>
    {error && <ErrorBox message={error} />}
    <section className="card"><div className="card-h"><span className="hint mono">runs/{runId}/run.log</span>{active && <span className="right"><Tag kind="ok">跟随中</Tag></span>}</div>
      <pre className="log" ref={box} style={{ maxHeight: 640 }}>{lines.map((l, i) => <span key={i} className={/error|traceback|exception/i.test(l) ? 'e' : /warn/i.test(l) ? 'w' : undefined}>{l}{'\n'}</span>)}</pre></section>
  </>;
}
