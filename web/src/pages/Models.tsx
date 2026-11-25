import { useEffect, useState } from 'react';
import { api, post, fileUrl, statusTitle, terminal, type Run, type JsonObject } from '../api';
import { ErrorBox } from '../ui';

interface ModelRecipe {
  id: string; name: string; description: string; family: string; family_name: string;
  scope: string; stages: string[]; notes: string; ready: boolean; issues: string[];
  template: JsonObject; latest_run: { id: string; state: string } | null;
}

export default function Models() {
  const [items, setItems] = useState<ModelRecipe[]>([]);
  const [family, setFamily] = useState('');
  const [selected, setSelected] = useState('');
  const [text, setText] = useState('');
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState('');
  const [log, setLog] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const model = items.find(i => i.id === selected);
  const active = runs.find(r => r.id === runId);
  const outputs = active?.result?.outputs ?? [];
  const previews = [...outputs.filter(o => o.media_type?.startsWith('video/')).slice(0, 3),
    ...outputs.filter(o => o.media_type?.startsWith('image/')).slice(0, 6)];

  useEffect(() => {
    let live = true;
    const refresh = async () => {
      try {
        const [models, history] = await Promise.all([
          api<{ items: ModelRecipe[] }>('/model-library'), api<{ items: Run[] }>('/model-library/runs')]);
        if (live) { setItems(models.items); setRuns(history.items.filter(r => r.config.workbench_recipe)); }
      } catch (e) { if (live) setError(String(e)); }
    };
    void refresh(); const timer = setInterval(refresh, 5000);
    return () => { live = false; clearInterval(timer); };
  }, []);

  useEffect(() => {
    let live = true, offset = 0, pending = false; setLog('');
    if (!runId) return;
    const refresh = async () => {
      if (pending) return;
      pending = true;
      try {
        const v = await api<{ text: string; next_offset: number }>(`/runs/${encodeURIComponent(runId)}/logs?offset=${offset}`);
        if (live) { offset = v.next_offset; setLog(prev => (prev + v.text).slice(-200000)); }
      } catch (e) { if (live) setError(String(e)); }
      finally { pending = false; }
    };
    void refresh(); const timer = setInterval(refresh, 3000);
    return () => { live = false; clearInterval(timer); };
  }, [runId]);

  function choose(item: ModelRecipe) {
    setSelected(item.id); setFamily(item.family); setText(JSON.stringify(item.template, null, 2)); setError('');
  }
  async function submit() {
    setError(''); setBusy(true);
    try {
      const body: unknown = JSON.parse(text);
      if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error('配置必须是 JSON 对象');
      const run = await post<Run>(`/model-library/${encodeURIComponent(selected)}/runs`, body);
      setRuns(prev => [run, ...prev]); setRunId(run.id);
    } catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  }

  const families = [...new Map(items.map(i => [i.family, i])).values()];
  return <div className="model-workbench">
    <div className="model-heading"><h1>模型工作台</h1><p>选择已有模型，提交原生任务，查看日志与产物。闭环安全对照实验仍从「新建实验」进入。</p></div>
    {error && <ErrorBox message={error} />}
    <div className="model-family-grid">{families.map(f => <button key={f.family}
      className={family === f.family ? 'selected' : ''} onClick={() => choose(items.find(i => i.family === f.family && i.ready) ?? f)}>
      <b>{f.family_name}</b><span>{f.scope}</span><small>{items.filter(i => i.family === f.family && i.ready).length} / {items.filter(i => i.family === f.family).length} 配方资源就绪</small>
    </button>)}</div>
    <p className="model-note">资源就绪只表示本机配置与资源可用；运行完成也不等于任务成功或安全验收通过。默认轻量参数用于接入验证，训练与回放配方仍需填写相应输入。</p>
    {model && <section className="model-panel">
      <label>任务配方<select value={selected} onChange={e => choose(items.find(i => i.id === e.target.value)!)}>
        {items.filter(i => i.family === family).map(i => <option key={i.id} value={i.id}>{i.name}</option>)}
      </select></label>
      <p>{model.description}</p><p>{model.notes}</p>
      <p><b>插件挂点：</b>{model.stages.length ? model.stages.join(' · ') : '此配方为原生任务入口，不使用通用逐步插件内核'}</p>
      <p><b>接入状态：</b>{model.ready ? '资源就绪，支持提交' : model.issues.join('；')}　<b>最近运行：</b>{model.latest_run ? statusTitle[model.latest_run.state] ?? model.latest_run.state : '尚无工作台运行记录'}</p>
      <label>运行配置（JSON）<textarea aria-label="模型运行配置" spellCheck={false} rows={16} value={text} onChange={e => setText(e.target.value)} /></label>
      <button className="btn primary" disabled={!model.ready || busy} onClick={submit}>{busy ? '提交中…' : '提交实验'}</button>
    </section>}
    <section className="model-panel"><h2>运行与结果</h2>
      <label>选择运行<select value={runId} onChange={e => setRunId(e.target.value)}><option value="">选择一条记录</option>
        {runs.map(r => <option key={r.id} value={r.id}>{r.name} · {statusTitle[r.state] ?? r.state} · {r.id}</option>)}</select></label>
      {active && <>
        <p>{statusTitle[active.state] ?? active.state} · {active.id} · 独立评价：{active.evaluation_status ?? '未提供'}
          {!terminal(active.state) && <button className="btn" onClick={() => post(`/runs/${active.id}/cancel`).catch(e => setError(String(e)))}>取消此任务</button>}</p>
        {active.error != null && <ErrorBox message={typeof active.error === 'string' ? active.error : JSON.stringify(active.error)} />}
        <div className="model-artifacts">{previews.map((o, i) => <div key={`${o.path}-${i}`}>
          <a href={fileUrl(active.id, o.path)} target="_blank" rel="noreferrer">{o.kind} · {o.path}</a>
          {o.media_type?.startsWith('video/') && <video controls preload="metadata" src={fileUrl(active.id, o.path)} />}
          {o.media_type?.startsWith('image/') && <img loading="lazy" alt={o.kind} src={fileUrl(active.id, o.path)} />}
        </div>)}</div>
        <details><summary>全部产物（{outputs.length}）</summary><ul>{outputs.map((o, i) =>
          <li key={`${o.path}-${i}`}><a href={fileUrl(active.id, o.path)} target="_blank" rel="noreferrer">{o.kind} · {o.path}</a></li>)}</ul></details>
        <details><summary>运行结果 JSON</summary><pre>{JSON.stringify(active.result, null, 2)}</pre></details>
        <h3>运行日志</h3><pre aria-label="模型运行日志" className="model-log">{log || '暂无日志'}</pre>
      </>}
    </section>
  </div>;
}
