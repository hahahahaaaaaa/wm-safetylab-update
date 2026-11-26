// 4 实验结果: paired report against the baseline, stage × arm matrix, episode list and episode detail.
import { useEffect, useMemo, useState } from 'react';
import type { Go, Route } from '../App';
import { redteam, type Catalog, type EpisodeDetail, type EpisodeRow, type Experiment, type Matrix, type Report, type TraceStep } from '../api/redteam';
import Replay, { type ReplayBaseline } from '../components/Replay';
import { ExperimentSelect } from './Progress';
import { cx, DeltaCell, Empty, environmentHazards, ErrorBox, fmt, Forest, type HazardDisplay, PageHead, Pill, pct, Seg, STAGE_NAME, STAGES, StageTag, Tag } from '../ui';

const errText = (e: unknown) => e instanceof Error ? e.message : String(e);

async function allEpisodes(id: string) {
  const out: EpisodeRow[] = [];
  let cursor: string | undefined;
  do { const page = await redteam.episodes(id, { cursor, limit: 500 }); out.push(...page.items); cursor = page.next_cursor ?? undefined; } while (cursor);
  return out;
}

function useResults(id?: string) {
  const [data, setData] = useState<{ exp: Experiment; report: Report; matrix: Matrix; rows: EpisodeRow[] } | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!id) return;
    let alive = true, timer = 0;
    const load = async () => {
      try {
        const [exp, report, matrix, rows] = await Promise.all([redteam.get(id), redteam.report(id), redteam.matrix(id), allEpisodes(id)]);
        if (!alive) return; setData({ exp, report, matrix, rows }); setError('');
        if (['queued', 'running', 'cancelling'].includes(exp.state)) timer = window.setTimeout(load, 4000);
      } catch (e) { if (alive) setError(errText(e)); }
    };
    setData(null); load();
    return () => { alive = false; window.clearTimeout(timer); };
  }, [id]);
  return { data, error };
}

export default function Results({ catalog, route, go }: { catalog: Catalog; route: Route; go: Go }) {
  const [items, setItems] = useState<Experiment[] | null>(null);
  useEffect(() => { redteam.list().then(p => setItems(p.items)).catch(() => setItems([])); }, []);
  const withData = (items ?? []).filter(e => e.episodes_done > 0);
  const id = route.exp ?? withData[0]?.id;
  const { data, error } = useResults(id);
  if (!items) return <div className="empty-state">加载…</div>;
  if (!id) return <><PageHead title="实验结果" /><Empty>还没有完成的 episode<br /><button className="btn" style={{ marginTop: 10 }} onClick={() => go({ sec: 'new', sub: 'basic' })}>新建实验</button></Empty></>;
  if (error) return <ErrorBox message={error} />;
  if (!data) return <div className="empty-state">加载结果…</div>;
  const { exp } = data;
  const environments = catalog.environments.filter(e => exp.config.environments?.includes(e.id));
  const hazards = environmentHazards(catalog, environments, [
    ...data.rows.flatMap(r => Object.keys(r.hazard_steps ?? {})),
    ...data.report.arms.flatMap(a => Object.keys(a.hazard_by_kind ?? {})),
  ]);
  const toy = (exp.config.environments ?? []).some(e => catalog.environments.find(x => x.id === e)?.family === 'toy');
  const bar = <div className="card toolbar" style={{ padding: '12px 16px' }}><span className="hint">实验</span>
    <ExperimentSelect items={items.some(e => e.id === exp.id) ? items : [exp, ...items]} value={exp.id} onChange={v => go({ sec: 'results', exp: v, ep: '' })} /><Pill state={exp.state} />
    <span className="hint mono">{exp.config.model} · {(exp.config.environments ?? []).length > 1 ? `${exp.config.environments!.length} 个任务` : exp.config.environments?.[0]}</span>
    {toy && <Tag kind="warn">玩具环境：流程检查，不是机械臂证据</Tag>}<span style={{ flex: 1 }} />
    {exp.state !== 'completed' ? <span className="pill warn">部分结果 {exp.episodes_done} / {exp.episodes_total}</span> : <span className="hint mono">{exp.episodes_total} episode</span>}</div>;
  const view = route.sub === 'matrix' ? <MatrixView {...data} go={go} /> : route.sub === 'episodes' ? <Episodes {...data} go={go} hazards={hazards} />
    : route.sub === 'episode' ? <Episode {...data} route={route} go={go} catalog={catalog} hazards={hazards} /> : <ReportView {...data} hazards={hazards} />;
  return <div className="stack">{bar}{view}<ReportArtifacts exp={exp} /></div>;
}

const pluginName = (exp: Experiment, key: string) => { const u = (exp.config.plugins ?? []).find(p => p.key === key); return u?.plugin ?? u?.entrypoint ?? key; };

function ReportView({ exp, report, rows, hazards }: { exp: Experiment; report: Report; rows: EpisodeRow[]; hazards: HazardDisplay[] }) {
  const base = report.arms[0];
  const hzRange = Math.max(2, ...report.arms.slice(1).map(a => a.delta_hazard ? Math.max(Math.abs(a.delta_hazard.low), Math.abs(a.delta_hazard.high)) : 0));
  const maxHz = Math.max(0.5, ...report.arms.map(a => a.hazard_steps_mean ?? 0));
  const reports = useMemo(() => {
    const out: Record<string, { arm: string; plugin: string; field: string; values: number[] }> = {};
    for (const r of rows) for (const [plugin, value] of Object.entries(r.plugin_reports ?? {}))
      for (const [field, v] of Object.entries((value ?? {}) as Record<string, unknown>)) if (typeof v === 'number') {
        const k = `${r.arm}|${plugin}|${field}`; (out[k] ??= { arm: r.arm, plugin, field, values: [] }).values.push(v); }
    return Object.values(out);
  }, [rows]);
  const copy = () => navigator.clipboard?.writeText(report.arms.map(a => `${a.arm} ${a.name}: 成功率 ${pct(a.success.estimate)} (n=${a.episodes})` +
    (a.delta_success ? `, Δ ${fmt(a.delta_success.estimate * 100, 0)} pp [${fmt(a.delta_success.low * 100, 0)}, ${fmt(a.delta_success.high * 100, 0)}]` : '') +
    `, 危险 ${fmt(a.hazard_steps_mean, 1)}/ep${a.oracle ? ' (oracle 上界)' : ''}`).join('\n'));
  return <>
    <PageHead title="对照报告"><button className="btn" onClick={copy}>复制摘要</button></PageHead>
    <div className="kpis"><div className="kpi"><span>基线成功率</span><b>{pct(base?.success.estimate)}</b><small className="mono">n = {base?.episodes} · [{pct(base?.success.low)}, {pct(base?.success.high)}]</small></div>
      <div className="kpi"><span>基线危险步数</span><b>{fmt(base?.hazard_steps_mean, 1)}</b><small>每 episode</small></div>
      <div className="kpi"><span>配对</span><b>{report.arms[1]?.delta_success?.pairs ?? 0}</b><small>对 · 同任务同种子</small></div>
      <div className="kpi"><span>接管</span><b>{fmt(Math.max(0, ...report.arms.map(a => a.takeovers_mean ?? 0)), 1)}</b><small>每 episode（最高组）</small></div></div>
    <section className="card"><div className="card-h"><h2>与基线配对比较</h2><span className="right"><span className="legend"><span><i style={{ background: 'var(--bad)' }} />显著变差</span><span><i style={{ background: 'var(--good)' }} />显著变好</span><span><i style={{ background: 'var(--faint)' }} />区间含 0</span></span></span></div>
      <div className="scroll"><table><thead><tr><th>组</th><th>插件</th><th className="num">n</th><th className="num">成功率</th><th>Δ 成功率 · 95% 区间</th><th className="num">危险步数/ep</th><th>Δ 危险步数</th><th className="num">接管/ep</th></tr></thead><tbody>
        {report.arms.map((a, i) => { const arm = exp.arms.find(x => x.id === a.arm);
          return <tr key={a.arm} style={i ? undefined : { background: 'var(--surface-2)' }}><td className="mono">{a.arm}</td>
            <td>{a.name}{a.oracle && <> <Tag kind="warn">oracle 上界</Tag></>}<div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>{(arm?.plugins ?? []).map(k => <span key={k} className="chip-x">{pluginName(exp, k)}</span>)}</div></td>
            <td className="num">{a.episodes}</td><td className="num">{pct(a.success.estimate)}<div className="ci">[{pct(a.success.low)}, {pct(a.success.high)}]</div></td>
            <td>{i ? <><DeltaCell d={a.delta_success} unit="pp" scale={100} higherIsWorse={false} /><Forest d={a.delta_success} range={1} higherIsWorse={false} /></> : <Tag>配对基准</Tag>}</td>
            <td className="num">{fmt(a.hazard_steps_mean, 1)}</td>
            <td>{i ? <><DeltaCell d={a.delta_hazard} unit="" digits={1} higherIsWorse /><Forest d={a.delta_hazard} range={hzRange} higherIsWorse /></> : null}</td>
            <td className="num">{fmt(a.takeovers_mean, 1)}</td></tr>; })}</tbody></table></div>
      <div className="card-b hint" style={{ borderTop: '1px solid var(--line)' }}>{report.method}</div></section>
    <div className="cols2">
      <section className="card"><div className="card-h"><h2>危险类型</h2><span className="sub">每 episode 步数</span><span className="right"><span className="legend">{hazards.map(h => <span key={h.id} title={h.name}><i style={{ background: `var(--${h.color})` }} />{h.id}</span>)}</span></span></div>
        <div className="scroll"><table><thead><tr><th>组</th>{hazards.map(h => <th key={h.id} className="num" title={h.name}>{h.id}</th>)}<th /></tr></thead><tbody>
          {report.arms.map(a => <tr key={a.arm}><td className="mono">{a.arm}</td>{hazards.map(h => <td key={h.id} className="num">{fmt(a.hazard_by_kind[h.id] ?? 0, 2)}</td>)}
            <td style={{ minWidth: 160 }}><div className="hzbar" style={{ width: Math.max(4, (a.hazard_steps_mean ?? 0) / maxHz * 100) + '%' }}>{hazards.map(h =>
              <i key={h.id} style={{ width: (a.hazard_steps_mean ? (a.hazard_by_kind[h.id] ?? 0) / a.hazard_steps_mean * 100 : 0) + '%', background: `var(--${h.color})` }} />)}</div></td></tr>)}</tbody></table></div></section>
      <section className="card"><div className="card-h"><h2>插件输出</h2><span className="sub">end_episode，均值</span></div>
        <div className="scroll"><table><thead><tr><th>组</th><th>插件</th><th>字段</th><th className="num">均值</th><th className="num">ep</th></tr></thead><tbody>
          {reports.map(r => <tr key={r.arm + r.plugin + r.field}><td className="mono">{r.arm}</td><td className="mono">{r.plugin}</td><td className="mono">{r.field}</td>
            <td className="num">{fmt(r.values.reduce((s, v) => s + v, 0) / r.values.length, 3)}</td><td className="num">{r.values.length}</td></tr>)}
          {!reports.length && <tr><td colSpan={5} className="empty-state">无</td></tr>}</tbody></table></div></section>
    </div>
  </>;
}

function ReportArtifacts({ exp }: { exp: Experiment }) {
  const [files, setFiles] = useState<{run: string; path: string; media_type?: string}[]>([]);
  const [error, setError] = useState('');
  useEffect(() => {
    let alive = true;
    Promise.all(exp.runs.map(async run => {
      const response = await fetch(`/api/v1/runs/${run.run_id}/files`);
      if (!response.ok) throw new Error('无法加载报告附件');
      const data: {items: {path: string; media_type?: string}[]} = await response.json();
      return data.items.filter(f => f.path.startsWith('s34-report/')).map(f => ({...f, run: run.run_id}));
    })).then(groups => { if (alive) { setFiles(groups.flat()); setError(''); } }).catch(e => { if (alive) setError(errText(e)); });
    return () => { alive = false; };
  }, [exp.id, exp.state]);
  if (error) return <ErrorBox message={error} />;
  if (!files.length) return null;
  return <section className="card"><div className="card-h"><h2>S3/S4 验收报告附件</h2></div><div className="card-b stack">
    {files.map(f => { const url = `/api/v1/runs/${f.run}/files/${f.path.split('/').map(encodeURIComponent).join('/')}`;
      return <div key={f.run + f.path}><a href={url} download>{f.path}</a>
        {f.media_type === 'image/png' && <img src={url} alt="S3/S4 配对结果图" style={{width:'100%', maxWidth:960}} />}</div>; })}
  </div></section>;
}

const dotClass = (r?: EpisodeRow) => !r ? 'todo' : r.hazard_total ? (r.success ? 'warn' : 'bad') : r.success ? 'good' : 'fail';

function MatrixView({ exp, report, matrix, rows, go }: { exp: Experiment; report: Report; matrix: Matrix; rows: EpisodeRow[]; go: Go }) {
  const used = STAGES.filter(s => matrix.rows.some(r => r.cells[s]));
  const stat = Object.fromEntries(report.arms.map(a => [a.arm, a]));
  const find = (arm: string, env: string, seed: number) => rows.find(r => r.arm === arm && r.environment === env && r.seed === seed);
  const start = exp.config.seeds?.start ?? 0;
  const head = <tr><th>环节</th>{matrix.rows.map(r => <th key={r.arm}>{r.arm}<div className="hint" style={{ fontWeight: 400, whiteSpace: 'normal', maxWidth: 150 }}>{r.name}</div></th>)}</tr>;
  return <>
    <section className="card"><div className="card-h"><h2>环节 × 对照组</h2><span className="sub">插件与每 episode 日志条数</span></div>
      <div className="scroll"><table className="mx"><thead>{head}</thead><tbody>
        {used.map(s => <tr key={s}><td><StageTag stage={s} /> <span className="hint">{STAGE_NAME[s]}</span></td>{matrix.rows.map(r => { const c = r.cells[s];
          return <td key={r.arm}>{c ? c.plugins.map(p => <div key={p}><span className="chip-x">{p}</span></div>) : <span className="hint">—</span>}
            {c?.events_per_episode != null && <div className="hint mono">{fmt(c.events_per_episode, 1)}/ep</div>}</td>; })}</tr>)}
        <tr><td><b style={{ fontWeight: 500 }}>成功率</b></td>{matrix.rows.map(r => <td key={r.arm} className="cellv">{pct(stat[r.arm]?.success.estimate)}</td>)}</tr>
        <tr><td><b style={{ fontWeight: 500 }}>Δ 成功率</b></td>{matrix.rows.map((r, i) => <td key={r.arm} className="cellv">{i ? <DeltaCell d={stat[r.arm]?.delta_success} unit="pp" scale={100} higherIsWorse={false} /> : null}</td>)}</tr>
        <tr><td><b style={{ fontWeight: 500 }}>Δ 危险步数</b></td>{matrix.rows.map((r, i) => <td key={r.arm} className="cellv">{i ? <DeltaCell d={stat[r.arm]?.delta_hazard} unit="" digits={1} higherIsWorse /> : null}</td>)}</tr>
        <tr><td><b style={{ fontWeight: 500 }}>接管/ep</b></td>{matrix.rows.map(r => <td key={r.arm} className="cellv">{fmt(stat[r.arm]?.takeovers_mean, 1)}</td>)}</tr>
      </tbody></table></div></section>
    <section className="card"><div className="card-h"><h2>任务 × 对照组</h2><span className="right"><span className="legend"><span><i style={{ background: 'var(--good)' }} />成功</span><span><i style={{ background: 'var(--warn)' }} />成功但有危险</span><span><i style={{ background: 'var(--bad)' }} />失败且有危险</span><span><i style={{ background: 'var(--surface-2)', boxShadow: 'inset 0 0 0 1px var(--faint)' }} />失败</span></span></span></div>
      <div className="scroll"><table className="mx"><thead><tr><th>任务</th>{matrix.rows.map(r => <th key={r.arm}>{r.arm}</th>)}</tr></thead><tbody>
        {(exp.config.environments ?? []).map(env => <tr key={env}><td className="mono">{env}</td>{matrix.rows.map(r => { const o = matrix.outcomes.find(x => x.arm === r.arm && x.environment === env);
          const done = (o?.outcomes ?? []).filter(v => v != null), k = done.filter(Boolean).length;
          return <td key={r.arm}><span className="cellv">{done.length ? Math.round(k / done.length * 100) + '%' : '—'}</span> <span className="hint mono">{k}/{done.length}</span>
            <div className="dots">{(o?.outcomes ?? []).map((_, i) => { const row = find(r.arm, env, start + i);
              return row ? <button key={i} className={'dot ' + dotClass(row)} title={`seed ${row.seed} · ${row.success ? 'success' : row.end_reason}${row.hazard_total ? ' · hazards ' + row.hazard_total : ''}`}
                onClick={() => go({ sec: 'results', sub: 'episode', ep: row.id })} /> : <span key={i} className="dot todo" title={`seed ${start + i}：未执行`} />; })}</div></td>; })}</tr>)}
      </tbody></table></div></section>
  </>;
}

type Outcome = 'all' | 'ok' | 'fail' | 'hz' | 'take';
function Episodes({ exp, rows, go, hazards }: { exp: Experiment; rows: EpisodeRow[]; go: Go; hazards: HazardDisplay[] }) {
  const [arm, setArm] = useState('all'), [env, setEnv] = useState('all'), [out, setOut] = useState<Outcome>('all');
  const base = (r: EpisodeRow) => rows.find(x => x.arm === 'A0' && x.environment === r.environment && x.seed === r.seed);
  const list = rows.filter(r => (arm === 'all' || r.arm === arm) && (env === 'all' || r.environment === env)
    && (out === 'all' || (out === 'ok' ? r.success : out === 'fail' ? !r.success : out === 'hz' ? r.hazard_total > 0 : r.takeovers > 0)));
  return <section className="card"><div className="card-h"><h2>Episodes</h2><span className="sub">{list.length}</span><span className="right">
    <select aria-label="组" value={arm} onChange={e => setArm(e.target.value)}><option value="all">全部组</option>{exp.arms.map(a => <option key={a.id} value={a.id}>{a.id} {a.name}</option>)}</select>
    {(exp.config.environments ?? []).length > 1 && <select aria-label="任务" value={env} onChange={e => setEnv(e.target.value)}><option value="all">全部任务</option>{exp.config.environments!.map(e => <option key={e}>{e}</option>)}</select>}
    <Seg value={out} onChange={setOut} options={[['all', '全部'], ['ok', '成功'], ['fail', '失败'], ['hz', '有危险'], ['take', '有接管']]} /></span></div>
    <div className="scroll"><table><thead><tr><th>组</th><th>任务</th><th className="num">seed</th><th>结果</th><th className="num">步数</th>{hazards.map(h => <th key={h.id} className="num" title={h.name}>{h.id}</th>)}<th className="num">接管</th><th>基线同种子</th></tr></thead><tbody>
      {list.slice(0, 500).map(r => { const b = r.arm !== 'A0' ? base(r) : undefined;
        return <tr key={r.id} className="click" onClick={() => go({ sec: 'results', sub: 'episode', ep: r.id })}><td className="mono">{r.arm}</td><td className="mono">{r.environment}</td><td className="num">{r.seed}</td>
          <td><Outcome r={r} /></td><td className="num">{r.steps}</td>
          {hazards.map(h => <td key={h.id} className="num" style={r.hazard_steps?.[h.id] ? { color: 'var(--bad)' } : undefined}>{r.hazard_steps?.[h.id] || '·'}</td>)}
          <td className="num">{r.takeovers || '·'}</td><td className="hint">{b ? (b.success ? 'success' : b.end_reason) + (b.hazard_total ? ` · 危险 ${b.hazard_total}` : '') : '—'}</td></tr>; })}
      {!list.length && <tr><td colSpan={7 + hazards.length} className="empty-state">没有符合条件的 episode</td></tr>}</tbody></table></div></section>;
}
const Outcome = ({ r }: { r?: EpisodeRow }) => !r ? <>—</> : r.success ? <span className="pill good">success@{r.success_step}</span> : <span className={'pill ' + (r.end_reason === 'plugin_stop' ? 'warn' : 'idle')}>{r.end_reason}</span>;

function pickEpisode(rows: EpisodeRow[]) {
  const base = (r: EpisodeRow) => rows.find(x => x.arm === 'A0' && x.environment === r.environment && x.seed === r.seed);
  return rows.find(r => r.arm !== 'A0' && !r.success && base(r)?.success) ?? rows.find(r => r.arm !== 'A0') ?? rows[0];
}

function Episode({ exp, rows, route, go, catalog, hazards }: { exp: Experiment; rows: EpisodeRow[]; route: Route; go: Go; catalog: Catalog; hazards: HazardDisplay[] }) {
  const row = rows.find(r => r.id === route.ep) ?? pickEpisode(rows);
  const [detail, setDetail] = useState<EpisodeDetail | null>(null);
  const [steps, setSteps] = useState<TraceStep[] | null>(null);
  const [ghost, setGhost] = useState<TraceStep[] | null>(null);
  const [baseline, setBaseline] = useState<ReplayBaseline | null>(null);
  const [baselineError, setBaselineError] = useState('');
  const [showGhost, setShowGhost] = useState(true);
  const [tab, setTab] = useState<'replay' | 'trace' | 'plugins' | 'meta' | 'json'>('replay');
  const [frame, setFrame] = useState(0);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!row) return;
    let alive = true;
    setDetail(null); setSteps(null); setGhost(null); setBaseline(null); setBaselineError(''); setError('');
    Promise.all([redteam.episode(exp.id, row.id), redteam.trace(exp.id, row.id)]).then(async ([d, t]) => {
      if (!alive) return; setDetail(d); setSteps(t.steps);
      const first = t.steps.findIndex(s => (s.hazards ?? []).some(h => h.id !== 'fast_contact')), any = t.steps.findIndex(s => (s.hazards ?? []).length);
      setFrame(Math.max(0, (first >= 0 ? first : any) - 2));
      if (d.baseline) {
        try {
          const [b, trace] = await Promise.all([redteam.episode(exp.id, d.baseline), redteam.trace(exp.id, d.baseline)]);
          if (!alive) return;
          if (b.row.seed !== row.seed || b.row.environment !== row.environment || b.row.arm !== 'A0') throw new Error('基线与当前回合的环境或 seed 不一致，无法配对回放。');
          setBaseline({ episode: d.baseline, seed: b.row.seed, arm: b.row.arm, steps: trace.steps, cameras: b.frames ?? {}, endReason: b.row.end_reason });
          setGhost(trace.steps);
        } catch (e) { if (alive) setBaselineError(errText(e)); }
      }
    }).catch(e => alive && setError(errText(e)));
    return () => { alive = false; };
  }, [exp.id, row?.id]);
  if (!row) return <Empty>还没有 episode</Empty>;
  const list = rows, idx = list.findIndex(r => r.id === row.id), prev = list[idx - 1], next = list[idx + 1];
  const base = rows.find(r => r.id === detail?.baseline);
  const arm = exp.arms.find(a => a.id === row.arm);
  const family = catalog.environments.find(e => e.id === row.environment)?.family;
  const f = steps ? Math.max(0, Math.min(frame, steps.length - 1)) : 0;
  const isTdmpc2 = exp.config.model === 'tdmpc2-pushcube';
  const isMemoryvlaBridge = family === 'bridge' && ['memoryvla-bridge', 'memoryvla-bridge-nomem'].includes(exp.config.model ?? '');
  const isCandle = row.environment === 'simpler-bridge-carrot-candle';
  const distance = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? `${fmt(value * 100, 2)} cm` : '未记录';
  return <section className="card">
    <div className="card-h"><button className="icon-btn" aria-label="上一个" disabled={!prev} onClick={() => go({ sec: 'results', sub: 'episode', ep: prev.id })}>‹</button>
      <h2 className="mono">{row.arm} · {row.environment} · seed {row.seed}</h2>
      <button className="icon-btn" aria-label="下一个" disabled={!next} onClick={() => go({ sec: 'results', sub: 'episode', ep: next.id })}>›</button>
      <span className="sub">{idx + 1} / {list.length}</span>
      <span className="right">{detail?.baseline && <button className="btn" onClick={() => go({ sec: 'results', sub: 'episode', ep: detail.baseline! })}>打开基线同种子</button>}
        <button className="link" onClick={() => go({ sec: 'results', sub: 'episodes' })}>← Episodes</button></span></div>
    <div className="sample-head"><div><div className="k">组</div><div className="v">{arm?.name}{arm?.oracle && <> <Tag kind="warn">oracle</Tag></>}</div></div>
      <div><div className="k">{isTdmpc2 ? '回合内曾成功' : isMemoryvlaBridge || isCandle ? '最后一步成功' : '结果'}</div><div className="v"><Outcome r={row} /></div></div>
      <div><div className="k">{isTdmpc2 ? '基线回合内曾成功（同种子）' : isMemoryvlaBridge || isCandle ? '基线最后一步成功（同种子）' : '基线（同种子）'}</div><div className="v">{row.arm === 'A0' ? '—' : <Outcome r={base} />}</div></div>
      <div><div className="k">危险步数</div><div className="v mono">{row.hazard_total}{base && <span className="hint"> 基线 {base.hazard_total}</span>}</div></div>
      <div><div className="k">接管</div><div className="v mono">{row.takeovers}</div></div></div>
    <div className="itabs">{([['replay', '回放'], ['trace', 'Trace'], ['plugins', '插件'], ['meta', '元数据'], ['json', 'JSON']] as const).map(([k, v]) =>
      <button key={k} aria-pressed={tab === k} onClick={() => setTab(k)}>{v}</button>)}
      {tab === 'replay' && ghost && <span className="right"><label className="check hint"><input type="checkbox" checked={showGhost} onChange={e => setShowGhost(e.target.checked)} />叠加基线轨迹</label></span>}</div>
    {error ? <div className="card-b"><ErrorBox message={error} /></div> : !detail || detail.row.id !== row.id || !steps ? <div className="empty-state">加载 trace…</div>
      : tab === 'replay' ? <>
          {isTdmpc2 && <div className="card-b hint">success@N 表示首次成功步；任务成功后仍继续执行，需另看最终状态。
            <div className="mono" style={{ marginTop: 4 }}>最终到目标距离：当前 {distance(detail.final_state?.distance_to_goal)}
              {detail.baseline && <> · 基线 {distance(baseline?.steps[baseline.steps.length - 1]?.state_after?.distance_to_goal)}</>}</div></div>}
          {isMemoryvlaBridge && <div className="card-b hint">正常完成回合时，SimplerEnv 按最后一步任务完成情况评价；success@N 表示成功评价步，并非首次到达目标。提前结束的回合请结合结束原因查看。</div>}
          {isCandle && <div className="card-b hint">纯仿真 · OpenVLA 胡萝卜放盘。正常完成回合时按最后一步任务完成情况评价，提前结束请结合结束原因查看；靠近蜡烛火焰使用末端或胡萝卜中心到有限火焰轴段的几何距离代理，与任务成功独立，不代表燃烧或温度测量。</div>}
          {baselineError && <div className="card-b"><ErrorBox message={'基线回放加载失败，当前回合仍可查看：' + baselineError} /></div>}
          {detail.baseline && !baseline && !baselineError && <div className="card-b hint">加载同环境、同 seed 基线画面…</div>}
          <Replay key={`${exp.id}:${row.id}`} experiment={exp.id} episode={row.id} seed={row.seed} arm={row.arm} endReason={row.end_reason}
            steps={steps} cameras={detail.frames ?? {}} baseline={baseline} hazards={hazards} yDown={family === 'toy'} ghost={showGhost ? ghost : null} frame={frame} onFrame={setFrame} />
        </>
      : tab === 'trace' ? <TraceTable steps={steps} frame={f} onFrame={setFrame} />
      : tab === 'plugins' ? <div className="card-b"><div className="opt-list">{(detail.plugins ?? []).map(p => <div key={p.key} className="opt" style={{ cursor: 'default' }}>
          <h3><StageTag stage={p.stage} /><span className="mono" style={{ fontSize: 13 }}>{p.id}</span>{(p.config as Record<string, unknown>)?.oracle ? <Tag kind="warn">oracle</Tag> : null}</h3>
          <pre className="code" style={{ marginTop: 8, maxHeight: 200 }}>{JSON.stringify({ config: p.config, end_episode: row.plugin_reports?.[p.key] ?? null }, null, 2)}</pre></div>)}
          {!(detail.plugins ?? []).length && <div className="empty-state">基线不挂插件</div>}</div></div>
      : tab === 'meta' ? <div className="card-b"><dl className="kv"><dt>实验</dt><dd className="mono">{exp.id}</dd><dt>运行</dt><dd className="mono">{row.run_id}</dd><dt>组</dt><dd>{row.arm} · {arm?.name}</dd>
          <dt>模型</dt><dd className="mono">{exp.config.model}</dd><dt>环境</dt><dd className="mono">{row.environment}</dd><dt>seed</dt><dd className="mono">{row.seed}（environment.reset 与 policy.reset 同种子）</dd>
          <dt>end_reason</dt><dd className="mono">{row.end_reason}</dd><dt>步数</dt><dd className="mono">{row.steps}</dd><dt>相机</dt><dd className="mono">{Object.entries(detail.frames ?? {}).map(([k, v]) => `${k} × ${v}`).join(' · ') || '无'}</dd>
          <dt>文件</dt><dd className="mono">runs/{row.run_id}/output/episodes/ep_{String(row.episode).padStart(3, '0')}/</dd></dl></div>
      : <pre className="code">{JSON.stringify({ ...row, initial_state: detail.initial_state, final_state: detail.final_state, events: detail.events }, null, 2)}</pre>}
  </section>;
}

function TraceTable({ steps, frame, onFrame }: { steps: TraceStep[]; frame: number; onFrame: (f: number) => void }) {
  const [monitors, setMonitors] = useState(false);
  const lo = Math.max(0, frame - 8), hi = Math.min(steps.length - 1, frame + 8);
  const stages = STAGES.filter(s => steps.some(t => (t.events ?? []).some(e => e.stage === s)));
  return <><div className="card-h"><span className="hint mono">trace.jsonl · t {lo}–{hi}</span><span className="right"><label className="check hint"><input type="checkbox" checked={monitors} onChange={e => setMonitors(e.target.checked)} />显示监控</label></span></div>
    <div className="scroll"><table className="trace"><thead><tr><th>t</th><th>action</th>{stages.map(s => <th key={s}>{s}</th>)}<th>takeover</th><th>hazards</th></tr></thead><tbody>
      {steps.slice(lo, hi + 1).map(t => <tr key={t.t} className={cx('click', t.t === frame && 'cur')} onClick={() => onFrame(t.t)}><td>{t.t}</td>
        <td>{(t.action ?? []).map(v => fmt(v, 2)).join(', ')}{t.action_proposed && JSON.stringify(t.action_proposed) !== JSON.stringify(t.action) && <div className="hint">提议 {t.action_proposed.map(v => fmt(v, 2)).join(', ')}</div>}</td>
        {stages.map(s => <td key={s} className="ev">{(t.events ?? []).filter(e => e.stage === s && (monitors || e.role !== 'monitor')).map((e, i) =>
          <div key={i}><span className="hint">{String(e.plugin)}</span> {JSON.stringify(e.data ?? {}).slice(0, 80)}</div>)}</td>)}
        <td>{t.takeover && <Tag kind="warn">{String((t.takeover as Record<string, unknown>).kind ?? 'takeover')}</Tag>}</td>
        <td>{(t.hazards ?? []).map((h, i) => <Tag key={i} kind="bad">{String(h.id)}</Tag>)}</td></tr>)}</tbody></table></div></>;
}
