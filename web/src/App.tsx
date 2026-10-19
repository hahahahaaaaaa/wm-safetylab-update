import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { api, post, Config, Recipe, Run, Result, Comparison, LogChunk, terminal } from './api';
import { ErrorPanel } from './components/Common';
import Experiments from './pages/Experiments';
import Runs from './pages/Runs';
import Cases from './pages/Cases';
import Comparisons from './pages/Comparisons';

const EmbodiedComparison = lazy(() => import('./pages/EmbodiedComparison'));

type Page = 'experiments' | 'runs' | 'cases' | 'comparisons' | 'embodied';
const pages: { id: Page; title: string; subtitle: string; icon: string }[] = [
  { id: 'experiments', title: '实验', subtitle: '配方与参数', icon: '▧' },
  { id: 'runs', title: '运行', subtitle: '状态与日志', icon: '▷' },
  { id: 'cases', title: '结果', subtitle: '文件与指标', icon: '◇' },
  { id: 'comparisons', title: '对比', subtitle: '多次实验', icon: '▤' },
  { id: 'embodied', title: '过程对比', subtitle: '场景与逐步决策', icon: '◫' },
];
const stored = (key: string) => { try { return localStorage.getItem(key) ?? ''; } catch { return ''; } };
const save = (key: string, value: string) => { try { localStorage.setItem(key, value); } catch { /* Optional browser state. */ } };
const readPage = (): Page => { const hash = location.hash.slice(1); return pages.some(page => page.id === hash) ? hash as Page : 'experiments'; };

export default function App() {
  const [page, setPage] = useState<Page>(readPage);
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [recipeId, setRecipeId] = useState('');
  const [raw, setRaw] = useState('');
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState(stored('wms.simple.runId'));
  const [run, setRun] = useState<Run | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [evaluations, setEvaluations] = useState<Run[]>([]);
  const [logs, setLogs] = useState('');
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [connected, setConnected] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const config = useMemo(() => {
    try { const value = JSON.parse(raw); return value && typeof value === 'object' && !Array.isArray(value) ? value as Config : null; }
    catch { return null; }
  }, [raw]);
  const report = (error: unknown) => setError(error instanceof Error ? error.message : String(error));
  const navigate = (next: Page) => { setPage(next); location.hash = next; };
  const chooseRun = (id: string) => { setRunId(id); save('wms.simple.runId', id); setError(''); setRefresh(value => value + 1); };
  const updateRun = (next: Run) => { setRun(next); setResult(next.result); setRuns(previous => [next, ...previous.filter(item => item.id !== next.id)]); };
  const refreshRuns = useCallback(async () => {
    const response = await api<{ items: Run[] }>('/runs'); setRuns(response.items);
  }, []);
  const initialize = useCallback(async () => {
    try {
      const [, recipeList] = await Promise.all([api('/health'), api<{ items: Recipe[] }>('/recipes')]);
      setConnected(true); setRecipes(recipeList.items); await refreshRuns();
      if (recipeList.items.length) {
        const first = recipeList.items[0];
        const initial = await api<Config>('/recipes/' + encodeURIComponent(first.id));
        setRaw(previous => previous || JSON.stringify(initial, null, 2));
        setRecipeId(previous => previous || first.id);
      } else {
        setRaw(previous => previous || JSON.stringify({ name: '我的实验', task: 'predict', runtime: {}, inputs: {}, params: {}, seed: 0, steps: 1, metrics: [] }, null, 2));
      }
    } catch (error) { setConnected(false); report(error); }
  }, [refreshRuns]);
  useEffect(() => { void initialize(); }, [initialize]);
  useEffect(() => {
    if (!connected) return;
    const timer = setInterval(() => { void refreshRuns().catch(report); }, 3000);
    return () => clearInterval(timer);
  }, [connected, refreshRuns]);
  useEffect(() => {
    const handler = () => setPage(readPage());
    window.addEventListener('hashchange', handler);
    return () => window.removeEventListener('hashchange', handler);
  }, []);

  useEffect(() => {
    if (!runId) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    let offset = 0;
    setRun(null); setResult(null); setLogs(''); setEvaluations([]);
    const poll = async () => {
      try {
        const path = '/runs/' + encodeURIComponent(runId);
        const [next, chunk] = await Promise.all([api<Run>(path), api<LogChunk>(path + '/logs?offset=' + offset)]);
        if (!active) return;
        setConnected(true); updateRun(next);
        if (chunk.text) setLogs(previous => previous + chunk.text);
        offset = chunk.next_offset;
        if (terminal(next.state)) {
          // A finished run can have more logs than the API's 200000-character chunk.
          let page = chunk;
          while (active && page.text.length >= 200000) {
            page = await api<LogChunk>(path + '/logs?offset=' + offset);
            if (!active) return;
            if (page.text) setLogs(previous => previous + page.text);
            if (page.next_offset <= offset) break;
            offset = page.next_offset;
          }
          const saved = await api<Result | null>(path + '/result');
          if (active) setResult(saved);
        }
        if (next.task !== 'evaluate') {
          const history = await api<{ items: Run[] }>(path + '/evaluations');
          if (active) setEvaluations(history.items);
        }
      } catch (error) { if (active) { report(error); setConnected(false); } }
      if (active) timer = setTimeout(poll, 1200);
    };
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [runId, refresh]);

  const perform = async (action: () => Promise<void>) => {
    setBusy(true); setError('');
    try { await action(); } catch (error) { report(error); } finally { setBusy(false); }
  };
  const loadRecipe = (id: string) => perform(async () => {
    const value = await api<Config>('/recipes/' + encodeURIComponent(id));
    setRaw(JSON.stringify(value, null, 2)); setRecipeId(id);
  });
  const submit = () => perform(async () => {
    if (!config) throw new Error('请填写有效的 JSON 实验配置。');
    const next = await post<Run>('/runs', config);
    chooseRun(next.id); navigate('runs'); await refreshRuns();
  });
  const cancel = () => perform(async () => {
    if (!runId) return;
    updateRun(await post<Run>('/runs/' + encodeURIComponent(runId) + '/cancel'));
    setRefresh(value => value + 1);
  });
  const retry = () => perform(async () => {
    if (!runId) return;
    const next = await post<Run>('/runs/' + encodeURIComponent(runId) + '/retry');
    chooseRun(next.id); navigate('runs'); await refreshRuns();
  });
  const evaluate = () => perform(async () => {
    if (!runId || run?.task === 'evaluate') return;
    const next = await post<Run>('/runs/' + encodeURIComponent(runId) + '/evaluate');
    setEvaluations(previous => [next, ...previous.filter(item => item.id !== next.id)]);
    await refreshRuns();
  });
  const compare = (ids: string[], choices: Record<string, string>) => perform(async () => {
    setComparison(await post<Comparison>('/comparisons', { run_ids: ids, evaluations: choices }));
  });
  const viewRun = (id: string) => { chooseRun(id); navigate('runs'); };
  const viewResult = (id: string) => { chooseRun(id); navigate('cases'); };
  const reuse = () => {
    if (!run) return;
    setRaw(JSON.stringify(run.config, null, 2)); setRecipeId(''); navigate('experiments');
  };

  return <div className="app-shell">
    <aside className="sidebar"><a className="brand" href="#experiments" onClick={() => navigate('experiments')}><span className="brand-mark">W<span>·</span></span><span>WM SafetyLab<small>实验工作台</small></span></a><p className="nav-label">WORKSPACE</p><nav>{pages.map(item => <button key={item.id} className={'nav-item ' + (page === item.id ? 'active' : '')} aria-label={item.title} title={item.title} onClick={() => navigate(item.id)}><span className="nav-icon">{item.icon}</span><span>{item.title}<small>{item.subtitle}</small></span></button>)}</nav><div className="sidebar-bottom"><strong>统一实验入口</strong><p>Python 核心 · CLI · Web<br />独立环境执行 · 文件输出</p><span>SIMPLE WORKSPACE</span></div></aside>
    <div className="main-shell"><header className="topbar"><div className="breadcrumb">实验空间 <span>/</span> <strong>{pages.find(item => item.id === page)?.title}</strong></div><button className={'connection ' + (connected ? 'online' : 'offline')} onClick={() => void initialize()}><i />{connected ? '服务已连接' : '重新连接服务'}</button></header><main>
      {error && <ErrorPanel message={error} onClose={() => setError('')} />}
      {page === 'experiments' && <Experiments recipes={recipes} recipeId={recipeId} raw={raw} config={config} busy={busy} onRecipe={id => void loadRecipe(id)} onChange={setRaw} onRun={() => void submit()} />}
      {page === 'runs' && <Runs runs={runs} run={run} logs={logs} busy={busy} onSelect={chooseRun} onRefresh={() => void perform(async () => { await refreshRuns(); setRefresh(value => value + 1); })} onCancel={() => void cancel()} onRetry={() => void retry()} onResults={() => navigate('cases')} onReuse={reuse} />}
      {page === 'cases' && <Cases key={runId} run={run} result={result} evaluations={evaluations} busy={busy} onEvaluate={() => void evaluate()} onRun={viewRun} onSource={viewResult} />}
      {page === 'comparisons' && <Comparisons runs={runs} comparison={comparison} busy={busy} onCompare={(ids, choices) => void compare(ids, choices)} onRun={viewRun} />}
      {page === 'embodied' && <Suspense fallback={<p>正在载入三维过程对比…</p>}><EmbodiedComparison runs={runs} onRun={viewRun} onResults={viewResult} onRefresh={refreshRuns} /></Suspense>}
    </main><footer>WM SafetyLab <span>配置 → 执行 → 输出 → 可选评价</span></footer></div>
  </div>;
}

