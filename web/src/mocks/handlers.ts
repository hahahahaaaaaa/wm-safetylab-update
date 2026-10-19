// MSW handlers for the red-team API. Only mock runs (ids starting with "rt-mock-")
// and not-yet-implemented routes are answered here; catalog and validation go to
// the real backend, which owns the compatibility rules.
import { bypass, http, HttpResponse, passthrough } from 'msw';
import type { S } from '../api/redteam';
import { detail, executed, matrix, progress, report, RUNS, summary, trace, type MockRun } from './fixtures';

const API = '/api/v1';
const runOf = (id: string): MockRun | undefined => RUNS.find(r => r.id === id);
const notFound = (what: string) => HttpResponse.json({ detail: `Not found: ${what}` }, { status: 404 });

function asRun(r: MockRun) {
  const p = progress(r);
  return { id: r.id, name: r.id, task: 'redteam', direction: 'vla', state: r.state, created_at: r.created_at,
    config: { name: r.id, kind: 'redteam', task: 'redteam', runtime: 'mock', target: { id: r.target }, labels: { operator: r.operator, operation: r.operation } },
    result: null, progress: p };
}

// Advance the running mock run so progress and live views move.
setInterval(() => RUNS.forEach(r => {
  if (r.state !== 'running') return;
  const total = progress(r).total;
  r.done = Math.min(total, (r.done || Math.round(total * 0.62)) + 1);
  if (r.done >= total) r.state = 'completed';
}), 1500);

export const handlers = [
  http.get(`${API}/runs`, async ({ request }) => {
    let real: unknown[] = [];
    try { const res = await fetch(bypass(request)); if (res.ok) real = await res.json(); } catch { /* backend offline: mock runs only */ }
    return HttpResponse.json([...RUNS.map(asRun), ...real]);
  }),
  http.post(`${API}/runs`, async ({ request }) => {
    const body = await request.clone().json() as Partial<S['ExperimentConfig']>;
    if (body.kind !== 'redteam') return passthrough();
    const run: MockRun = { id: `rt-mock-${Date.now().toString(36)}`, target: body.target?.id ?? 'openvla-7b', state: 'running',
      tasks: body.tasks?.length ? body.tasks : ['cup'], strategies: ['basic', ...(body.strategies ?? [])], defenses: body.defenses ?? ['none'],
      plugins: (body.plugins ?? []).map(p => p.id), rounds: 3, seedOff: 1000 + RUNS.length * 37, done: 1, created_at: new Date().toISOString(),
      operator: body.labels?.operator ?? '', operation: body.labels?.operation ?? '' };
    RUNS.unshift(run);
    return HttpResponse.json(asRun(run), { status: 201 });
  }),
  http.post(`${API}/targets/test`, async ({ request }) => {
    const body = await request.clone().json() as S['TargetRef'];
    const result: S['TargetTestResult'] = body.id === 'custom' && !/^https?:\/\//.test(body.endpoint ?? '')
      ? { ok: false, message: '请填写有效的策略服务地址' } : { ok: true, latency_ms: 96, action_dim: 7, message: 'mock' };
    return HttpResponse.json(result);
  }),
  ...['pause', 'resume'].map(action => http.post(`${API}/runs/:runId/${action}`, ({ params }) => {
    const r = runOf(String(params.runId)); if (!r) return passthrough();
    r.state = action === 'pause' ? 'paused' : 'running';
    return HttpResponse.json(progress(r));
  })),
  http.post(`${API}/runs/:runId/cancel`, ({ params }) => {
    const r = runOf(String(params.runId)); if (!r) return passthrough();
    r.state = 'cancelled'; return HttpResponse.json(asRun(r));
  }),
  http.get(`${API}/runs/:runId/progress`, ({ params }) => {
    const r = runOf(String(params.runId)); return r ? HttpResponse.json(progress(r)) : passthrough();
  }),
  http.get(`${API}/runs/:runId/episodes`, ({ params, request }) => {
    const r = runOf(String(params.runId)); if (!r) return passthrough();
    const q = new URL(request.url).searchParams;
    const limit = Number(q.get('limit') ?? 100), offset = Number(q.get('cursor') ?? 0);
    let items = executed(r).map(summary).reverse();
    for (const key of ['task', 'strategy', 'defense', 'verdict'] as const) { const v = q.get(key); if (v) items = items.filter(e => e[key] === v); }
    if (q.get('include_baseline') === 'false') items = items.filter(e => e.strategy !== 'basic');
    const text = q.get('q'); if (text) items = items.filter(e => (e.task + e.input_text + e.judge_summary).includes(text));
    const page: S['EpisodePage'] = { items: items.slice(offset, offset + limit), total: items.length, next_cursor: offset + limit < items.length ? String(offset + limit) : null };
    return HttpResponse.json(page);
  }),
  http.get(`${API}/runs/:runId/episodes/:episodeId`, ({ params }) => {
    const r = runOf(String(params.runId)); if (!r) return passthrough();
    const e = executed(r).find(x => x.id === params.episodeId);
    return e ? HttpResponse.json(detail(e)) : notFound(String(params.episodeId));
  }),
  http.get(`${API}/runs/:runId/episodes/:episodeId/trace`, ({ params }) => {
    const r = runOf(String(params.runId)); if (!r) return passthrough();
    const e = executed(r).find(x => x.id === params.episodeId);
    return e ? HttpResponse.json(trace(e)) : notFound(String(params.episodeId));
  }),
  http.get(`${API}/runs/:runId/report`, ({ params }) => {
    const r = runOf(String(params.runId)); return r ? HttpResponse.json(report(r)) : passthrough();
  }),
  http.get(`${API}/runs/:runId/matrix`, ({ params, request }) => {
    const r = runOf(String(params.runId)); if (!r) return passthrough();
    return HttpResponse.json(matrix(r, new URL(request.url).searchParams.get('defense') ?? r.defenses[0]));
  }),
];
