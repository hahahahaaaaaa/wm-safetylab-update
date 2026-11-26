// Node built-ins are runtime-only; no frontend test framework is required.
// @ts-ignore -- compiling this standalone test does not require @types/node.
import assert from 'node:assert/strict';
// @ts-ignore -- Node's built-in test runner.
import test from 'node:test';
// @ts-ignore -- optional local historical evidence audit.
import { readFileSync } from 'node:fs';
// @ts-ignore -- optional local historical evidence audit.
import { join } from 'node:path';
import {
  type Episode, type Frame, comparisonWarnings, cycleStartTime, decisionAtFrame,
  episodeDuration, frameAtTime, isEmbodiedRun, normalizeDecision,
} from '../src/embodied';

declare const process: { env: Record<string, string | undefined> };

function fixture(): Episode {
  return {
    id: 'example', task: 'transfer', seed: 0, scene_hash: 'same-scene',
    provider: 'jev', control_mode: 'skills', observation_mode: 'privileged',
    preview: true, threshold: 0.55, status: 'completed',
    frames: [
      { time: 0.5, cycle: 0, phase: null, observation: { sim_seconds: 0 } },
      { time: 0.54, cycle: 1, phase: 'approach', observation: { sim_seconds: 0.04 } },
      { time: 1.3, cycle: 1, phase: 'approach', observation: { sim_seconds: 0.8 } },
      { time: 1.34, cycle: 2, phase: 'lower', observation: { sim_seconds: 0.84 } },
      { time: 2.1, cycle: 2, phase: 'lower', observation: { sim_seconds: 1.6 } },
    ],
    history: [
      { cycle: 1, phase: 'approach', intent: { choice: 'approach' },
        decision: { choice: 'direct', probabilities: { direct: 0.8, hold: 0.2 }, selected_probability: 0.8 },
        action: { id: 'direct', admitted: true }, candidates: [{ id: 'direct', admitted: true }, { id: 'hold' }],
        before: { sim_seconds: 0 }, after: { sim_seconds: 0.8, object: [1, 0, 0] } },
      { cycle: 2, phase: 'lower', intent: { choice: 'lower' },
        decision: { choice: 'gentle' }, action: { id: 'gentle', admitted: true },
        before: { sim_seconds: 0.8 }, after: { sim_seconds: 1.6, object: [2, 0, 0] } },
    ],
    last_intent: { choice: 'lower' }, last_decision: { choice: 'gentle' },
  };
}

test('filters source experiments and excludes evaluate, cache and name-only matches', () => {
  const config = { task: 'native', recipe: 'embodied-jev-chat',
    entrypoint: 'wms.engines.embodied_jev.native:run_episode' };
  for (const state of ['queued', 'running', 'completed', 'failed']) {
    assert.equal(isEmbodiedRun({ task: 'native', state, config }), true);
  }
  assert.equal(isEmbodiedRun({ task: 'evaluate', config }), false);
  assert.equal(isEmbodiedRun({ task: 'native', source_run: 'old', config }), false);
  assert.equal(isEmbodiedRun({ task: 'native', config: {
    ...config, entrypoint: 'wms.engines.embodied_jev.video:render_episode',
  } }), false);
  assert.equal(isEmbodiedRun({ task: 'native', config: { recipe: 'embodied-jev-cache' } }), false);
  assert.equal(isEmbodiedRun({ task: 'native', name: 'embodied-jev-chat', config: {} }), false);
  assert.equal(isEmbodiedRun({ task: 'native', config: { recipe: 'embodied-jev-baseline' } }), true);
  assert.equal(isEmbodiedRun(null), false);
});

test('seeks by independent elapsed origins and uses preceding recorded frame', () => {
  const frames = fixture().frames;
  const selected = frameAtTime(frames, 0.5);
  assert.equal(selected.index, 1);
  assert.equal(selected.elapsed, 0.5);
  assert.ok(Math.abs(selected.frameElapsed - 0.04) < 1e-9);
  const shifted = frames.map(frame => ({ ...frame, time: frame.time + 123 }));
  assert.equal(frameAtTime(shifted, 0.5).index, selected.index);
  const visual = frames.map((frame, index) => ({ ...frame, index, sim_time: frame.time, time: frame.time - 0.5 }));
  assert.equal(frameAtTime(visual, 0.5).index, selected.index);
});

test('clamps early and exhausted lanes independently without looping', () => {
  const frames = fixture().frames;
  assert.equal(frameAtTime(frames, -1).index, 0);
  assert.equal(frameAtTime(frames, -1).clamped, true);
  const end = frameAtTime(frames, 999);
  assert.equal(end.index, 4);
  assert.equal(end.ended, true);
  assert.equal(end.clamped, true);
  assert.ok(Math.abs(end.elapsed - 1.6) < 1e-9);
  assert.equal(frameAtTime(frames, episodeDuration({ frames })).clamped, false);
  assert.equal(frameAtTime(frames, episodeDuration({ frames })).ended, true);
  assert.equal(frameAtTime(frames.slice(0, 3), 1.2).ended, true);
  assert.equal(frameAtTime(frames, 1.2).ended, false);
});

test('empty and single-frame lanes are explicit, invalid clocks are rejected', () => {
  assert.equal(frameAtTime([], 0).frame, null);
  assert.equal(frameAtTime([], 0).index, -1);
  assert.equal(frameAtTime([], 0).ended, false);
  assert.equal(frameAtTime([{ time: 5 }], 0).ended, true);
  assert.throws(() => frameAtTime([{ time: 1 }, { time: 0 }], 0), RangeError);
  assert.throws(() => frameAtTime([{ time: NaN }], 0), RangeError);
  assert.throws(() => frameAtTime([], Infinity), RangeError);
});

test('equal timestamps choose latest recorded state at that instant', () => {
  const frames: Frame[] = [{ time: 1, cycle: 0 }, { time: 1, cycle: 1 }, { time: 2, cycle: 2 }];
  assert.equal(frameAtTime(frames, 0).index, 1);
});

test('initial frame has no future decision and cycles do not forward-fill', () => {
  const episode = fixture();
  assert.equal(decisionAtFrame(episode, episode.frames[0], 0), null);
  episode.history = episode.history.slice(1);
  assert.equal(decisionAtFrame(episode, episode.frames[1], 1), null);
  assert.equal(decisionAtFrame(episode, episode.frames[1], 99), null);
});

test('action-segment decisions hide future after observations and execution outcomes', () => {
  const episode = fixture();
  const during = decisionAtFrame(episode, episode.frames[1], 1)!;
  assert.equal(during.decision?.choice, 'direct');
  assert.equal(during.after, null);
  assert.equal(during.completed, false);
  assert.equal(during.executed, null);
  assert.equal(during.timingLabel, '该动作段记录；非逐请求实时记录');
  const completed = decisionAtFrame(episode, episode.frames[2], 2)!;
  assert.deepEqual(completed.after?.object, [1, 0, 0]);
  assert.equal(completed.completed, true);
  assert.equal(completed.executed, true);
  assert.equal(decisionAtFrame(episode, episode.frames[3], 3)?.decision?.choice, 'gentle');
});

test('missing completion time remains unknown even at a cycle final frame', () => {
  const episode = fixture();
  delete episode.history[0].after!.sim_seconds;
  const view = decisionAtFrame(episode, episode.frames[2], 2)!;
  assert.equal(view.completed, null);
  assert.equal(view.after, null);
});

test('uncertain pending decision appears only at final frame despite unchanged cycle', () => {
  const episode = fixture();
  episode.status = 'uncertain';
  episode.frames = episode.frames.slice(0, 3);
  episode.history = episode.history.slice(0, 1);
  episode.last_intent = { choice: 'lower' };
  episode.last_decision = { choice: 'gentle', probabilities: { gentle: 0.4, hold: 0.6 }, selected_probability: 0.4 };
  episode.last_decision_inputs = { action: { decision: { criteria: {
    gentle: JSON.stringify({ phase: 'lower', motion: 'gentle', target_tcp: [1, 2, 3] }),
    hold: 'Keep the current pose',
  } } } };
  assert.equal(decisionAtFrame(episode, episode.frames[0], 0), null);
  assert.equal(decisionAtFrame(episode, episode.frames[1], 1)?.decision?.choice, 'direct');
  const pending = decisionAtFrame(episode, episode.frames[2], 2)!;
  assert.equal(pending.source, 'terminal');
  assert.equal(pending.phase, 'lower');
  assert.equal(pending.decision?.choice, 'gentle');
  assert.equal(pending.decision?.selected_probability, 0.4);
  assert.equal(pending.cycle, null);
  assert.equal(pending.executed, false);
  assert.equal(pending.action, null);
  assert.equal(pending.after, null);
  assert.equal(pending.candidates.length, 2);
  assert.match(pending.timingLabel, /未执行/);
});

test('uncertain phase decision keeps its phase role and no action is invented', () => {
  const episode = fixture();
  episode.status = 'uncertain';
  episode.last_intent = { choice: 'lower', probabilities: { lower: 0.4, hold: 0.6 } };
  episode.last_decision = { ...episode.last_intent };
  episode.last_decision_inputs = { phase: { decision: { criteria: { lower: 'Lower', hold: 'Hold' } } } };
  const pending = decisionAtFrame(episode, episode.frames[4], 4)!;
  assert.equal(pending.decisionRole, 'intent');
  assert.equal(pending.candidates[0].id, 'lower');
});

test('last decision is not borrowed for arbitrary missing or stale frame cycles', () => {
  const episode = fixture();
  episode.status = 'running';
  episode.history = [];
  assert.equal(decisionAtFrame(episode, episode.frames[4], 4), null);
  assert.equal(decisionAtFrame(fixture(), { ...episode.frames[1], cycle: 2 }, 1), null);
});

test('absent probabilities stay null and reported probabilities are not renormalized', () => {
  assert.equal(normalizeDecision({ choice: 'direct', probabilities: {} })?.probabilities, null);
  assert.equal(normalizeDecision({ choice: 'direct' })?.selected_probability, null);
  const reported = normalizeDecision({ probabilities: { a: 0.2, b: 0.3 }, selected_probability: 0.2 })!;
  assert.deepEqual(reported.probabilities, { a: 0.2, b: 0.3 });
  assert.equal(normalizeDecision({ probabilities: { a: Infinity } })?.probabilities, null);
  assert.equal(normalizeDecision({ selected_probability: 2 })?.selected_probability, null);
  assert.equal(normalizeDecision({ channel_decisions: { x: { choice: 'hold' } } })?.channel_decisions?.x.selected_probability, null);
});

test('hierarchical channels retain independent probabilities without a fabricated joint value', () => {
  const episode = fixture();
  episode.control_mode = 'hierarchical';
  episode.history[0].decision = {
    choice: 'assembled-motion', probabilities: {}, selected_probability: null,
    channel_decisions: {
      x: { choice: 'positive', probabilities: { positive: 0.7, hold: 0.3 }, selected_probability: 0.7 },
      gripper: { choice: 'close', probabilities: {}, selected_probability: null },
    },
  };
  const before = JSON.stringify(episode);
  const view = decisionAtFrame(episode, episode.frames[1], 1)!;
  assert.equal(view.decision?.probabilities, null);
  assert.equal(view.decision?.selected_probability, null);
  assert.deepEqual(view.decision?.channel_decisions?.x.probabilities, { positive: 0.7, hold: 0.3 });
  assert.equal(view.decision?.channel_decisions?.gripper.probabilities, null);
  assert.equal(view.after, null);
  assert.equal(view.executed, null);
  assert.equal(JSON.stringify(episode), before);
});

test('normalization and replay selection do not mutate original evidence', () => {
  const episode = fixture();
  const before = JSON.stringify(episode);
  frameAtTime(episode.frames, 1);
  decisionAtFrame(episode, episode.frames[1], 1);
  normalizeDecision(episode.history[0].decision);
  assert.equal(JSON.stringify(episode), before);
});

test('comparison warns about each mismatched condition and missing evidence', () => {
  const a = fixture(), b = fixture();
  assert.deepEqual(comparisonWarnings([a, { ...b, provider: 'chat' }]), []);
  Object.assign(b, { task: 'barrier', seed: 1, scene_hash: 'different', control_mode: 'hierarchical',
    observation_mode: 'rgbd', preview: false, threshold: 0.7 });
  assert.equal(comparisonWarnings([a, b]).length, 7);
  for (const name of ['任务', '随机种子', '场景哈希', '控制模式', '观测模式', '预演开关', '决策阈值']) {
    assert.ok(comparisonWarnings([a, b]).some(warning => warning.startsWith(name + '不同')));
  }
  delete b.scene_hash;
  assert.ok(comparisonWarnings([a, b]).some(warning => warning.startsWith('场景哈希缺失')));
  assert.deepEqual(comparisonWarnings([a]), []);
});

test('comparison keeps actual lane numbers when an earlier or middle slot is pending', () => {
  const a = fixture(), b = { ...fixture(), task: 'barrier' };
  assert.deepEqual(comparisonWarnings([undefined, a, b]), [
    '任务不同（通道 2：transfer；通道 3：barrier），不能作为相同条件对照。',
  ]);
  assert.deepEqual(comparisonWarnings([a, null, b]), [
    '任务不同（通道 1：transfer；通道 3：barrier），不能作为相同条件对照。',
  ]);
  delete b.scene_hash;
  assert.ok(comparisonWarnings([a, null, b]).includes('场景哈希缺失（通道 3），无法确认条件一致。'));
  assert.deepEqual(comparisonWarnings([undefined, a, null]), []);
  assert.deepEqual(comparisonWarnings([null, undefined]), []);
});

test('action jump uses first recorded frame instead of an invented request time', () => {
  const episode = fixture();
  assert.ok(Math.abs(cycleStartTime(episode, 1)! - 0.04) < 1e-9);
  assert.equal(cycleStartTime(episode, 9), null);
});

const evidenceRoot = process.env.WMS_EVIDENCE_ROOT;
test('nine archived runs retain initial silence and terminal uncertain decisions', { skip: !evidenceRoot }, () => {
  const report = JSON.parse(readFileSync(join(evidenceRoot!, 'results/embodied-jev-integration-20261012/live-acceptance.json'), 'utf8'));
  assert.equal(report.runs.length, 9);
  let uncertain = 0;
  for (const run of report.runs) {
    const episode = JSON.parse(readFileSync(join(evidenceRoot!, 'data/runs', run.run_id, 'output/episode.raw.json'), 'utf8')) as Episode;
    assert.equal(decisionAtFrame(episode, episode.frames[0], 0), null);
    const last = episode.frames.length - 1;
    assert.equal(frameAtTime(episode.frames, 1000).index, last);
    for (let index = 1; index < last; index++) {
      const view = decisionAtFrame(episode, episode.frames[index], index);
      if (view) {
        assert.equal(view.source, 'history');
        assert.equal(view.cycle, episode.frames[index].cycle);
        if (view.after) assert.ok(view.after.sim_seconds! <= episode.frames[index].observation!.sim_seconds! + 1e-6);
      }
    }
    const final = decisionAtFrame(episode, episode.frames[last], last)!;
    if (episode.status === 'uncertain') {
      uncertain++;
      assert.equal(final.source, 'terminal');
      assert.equal(final.phase, 'lower');
      assert.equal(final.decision?.choice, episode.last_decision?.choice);
      assert.equal(final.executed, false);
    } else assert.equal(final.source, 'history');
  }
  assert.equal(uncertain, 3);
});

