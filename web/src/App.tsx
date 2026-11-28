import { useCallback, useEffect, useState } from 'react';
import { redteam, type Catalog } from './api/redteam';
import Setup, { type StepId } from './pages/Setup';
import Progress from './pages/Progress';
import Results from './pages/Results';
import { ErrorBox } from './ui';

export type Section = 'new' | 'config' | 'progress' | 'results';
export interface Route { sec: Section; sub: string; exp?: string; ep?: string }
export type Go = (route: Partial<Route> & { sec: Section }) => void;

const SECTIONS: { id: Section; title: string; subs?: [string, string][] }[] = [
  { id: 'new', title: '新建实验' },
  { id: 'config', title: '实验配置' },
  { id: 'progress', title: '实验进度', subs: [['runs', '运行列表'], ['live', '实时监控'], ['logs', '运行日志']] },
  { id: 'results', title: '实验结果', subs: [['report', '对照报告'], ['matrix', '环节 × 插件'], ['episodes', 'Episodes'], ['episode', 'Episode 详情']] },
];
const DEFAULT_SUB: Record<Section, string> = { new: 'basic', config: 'plugins', progress: 'runs', results: 'report' };

function readRoute(): Route {
  if (/^#models(?:[/?]|$)/.test(location.hash)) {
    history.replaceState(null, '', '#new/basic');
  }
  const [path, search] = location.hash.slice(1).split('?');
  const [sec, sub] = path.split('/') as [Section, string];
  const params = new URLSearchParams(search ?? '');
  const known = SECTIONS.some(s => s.id === sec);
  return { sec: known ? sec : 'new', sub: known && sub ? sub : DEFAULT_SUB[known ? sec : 'new'],
    exp: params.get('exp') ?? undefined, ep: params.get('ep') ?? undefined };
}
function writeRoute(r: Route) {
  const params = new URLSearchParams();
  if (r.exp) params.set('exp', r.exp);
  if (r.ep) params.set('ep', r.ep);
  const text = `#${r.sec}/${r.sub}${params.size ? '?' + params : ''}`;
  if (location.hash !== text) history.pushState(null, '', text);
}

export default function App() {
  const [route, setRoute] = useState<Route>(readRoute);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    redteam.catalog().then(setCatalog).catch(e => setError('无法连接后台服务：' + (e instanceof Error ? e.message : String(e))));
    const pop = () => setRoute(readRoute());
    window.addEventListener('popstate', pop);
    return () => window.removeEventListener('popstate', pop);
  }, []);

  const go: Go = useCallback(next => {
    setRoute(prev => {
      const sameSec = next.sec === prev.sec;
      const r: Route = { sub: next.sub ?? (sameSec ? prev.sub : DEFAULT_SUB[next.sec]), exp: next.exp ?? prev.exp, ep: next.ep ?? (next.exp && next.exp !== prev.exp ? undefined : prev.ep), sec: next.sec };
      writeRoute(r);
      if (!sameSec) window.scrollTo({ top: 0 });
      return r;
    });
  }, []);

  const section = SECTIONS.find(s => s.id === route.sec)!;
  return <>
    <header className="top">
      <div className="wrap bar1">
        <div className="brand"><b>WM-SafetyLab</b><span>具身仿真红队测试</span></div>
        <nav className="main-nav" aria-label="主要功能">{SECTIONS.map((s, i) => <span key={s.id} style={{ display: 'contents' }}>
          {i > 0 && <span className="sep" aria-hidden="true">›</span>}
          <button aria-current={route.sec === s.id ? 'page' : undefined} onClick={() => go({ sec: s.id })}><span className="num">{i + 1}</span>{s.title}</button>
        </span>)}</nav>
      </div>
      {section.subs && <div className="subbar"><div className="wrap"><nav className="sub-nav" aria-label="子功能">
        {section.subs.map(([id, title]) => <button key={id} aria-current={route.sub === id ? 'page' : undefined} onClick={() => go({ sec: route.sec, sub: id })}>{title}</button>)}
      </nav></div></div>}
    </header>
    <main className="wrap">
      {error ? <ErrorBox message={error} /> : !catalog ? <div className="empty-state">加载目录…</div>
        : route.sec === 'new' || route.sec === 'config' ? <Setup catalog={catalog} sec={route.sec} step={route.sub as StepId} go={go} />
        : route.sec === 'progress' ? <Progress catalog={catalog} route={route} go={go} />
        : <Results catalog={catalog} route={route} go={go} />}
    </main>
  </>;
}
