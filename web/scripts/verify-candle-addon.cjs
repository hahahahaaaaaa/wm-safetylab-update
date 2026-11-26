// Exercise the real checkbox callbacks without starting a browser or GPU worker.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const esbuild = require('esbuild');
const web = path.resolve(__dirname, '..');
const out = path.resolve(process.argv[2]);
fs.mkdirSync(out, { recursive: true });
const setup = path.join(web, 'src/pages/Setup.tsx');
const extra = `
import { renderToStaticMarkup } from 'react-dom/server';
import { createElement } from 'react';
export { EnvTask, Model, Review };
export * from '../taskSelection';
export const ssr = (component, props) => renderToStaticMarkup(createElement(component, props));
`;
const bundle = path.join(out, 'frontend-test-bundle.cjs');
esbuild.buildSync({ stdin: { contents: fs.readFileSync(setup, 'utf8') + extra, loader: 'tsx',
  resolveDir: path.dirname(setup), sourcefile: setup }, bundle: true, platform: 'node',
  format: 'cjs', jsx: 'automatic', outfile: bundle });
const ui = require(bundle);
const catalog = JSON.parse(fs.readFileSync(path.join(out, 'catalog-live.json'), 'utf8').replace(/^\uFEFF/, ''));
const { CARROT_TASK: base, CANDLE_VARIANT: candle } = ui;
const model = catalog.models.find(m => m.id === 'openvla-bridge');
const baseName = catalog.environments.find(e => e.id === base).name;
let config = { name: 'checkbox integration', model: model.id, environments: [], plugins: [],
  seeds: { start: 0, episodes: 1 }, max_steps: 120 };
function ctx() { return { catalog, config, model, envs: catalog.environments.filter(e => config.environments.includes(e.id)),
  result: null, go() {}, set(change) { const next = structuredClone(config); change(next);
    next.environments = ui.normalizeTaskSelection(next.environments); config = next; } }; }
function nodes(node) {
  if (Array.isArray(node)) return node.flatMap(nodes);
  if (!node || typeof node !== 'object') return [];
  return [node, ...nodes(node.props?.children)];
}
function control(label) { const input = nodes(ui.EnvTask(ctx())).find(n => n.type === 'input' && n.props['aria-label'] === label);
  assert.ok(input, label); return input.props; }
function button(label) { const found = nodes(ui.EnvTask(ctx())).find(n => n.type === 'button' && n.props.children === label);
  assert.ok(found, label); found.props.onClick(); }
function html() { return ui.ssr(ui.EnvTask, ctx()); }
const checks = [];
function check(name, test) { test(); checks.push(name); console.log('PASS ' + name); }

check('unselected task disables the candle add-on', () => {
  assert.equal(control('在场景中添加蜡烛').disabled, true);
  assert.deepEqual(ui.setCandleAddon([], true), []);
});
check('select task then candle creates one effective task', () => {
  control(baseName).onChange();
  assert.deepEqual(config.environments, [base]);
  assert.equal(control('在场景中添加蜡烛').disabled, false);
  control('在场景中添加蜡烛').onChange({ target: { checked: true } });
  assert.deepEqual(config.environments, [candle]);
  assert.equal(control(baseName).checked, true);
  assert.equal(control('在场景中添加蜡烛').checked, true);
  assert.ok(html().includes('已选 1 / 4'));
  fs.writeFileSync(path.join(out, 'frontend-selected-config.json'), JSON.stringify(config, null, 2));
  fs.writeFileSync(path.join(out, 'frontend-selected.html'), html());
});
check('unchecking candle restores plain task', () => {
  control('在场景中添加蜡烛').onChange({ target: { checked: false } });
  assert.deepEqual(config.environments, [base]);
  assert.equal(control(baseName).checked, true);
});
check('unchecking the base task clears candle and disables it', () => {
  control('在场景中添加蜡烛').onChange({ target: { checked: true } });
  control(baseName).onChange();
  assert.deepEqual(config.environments, []);
  assert.equal(control('在场景中添加蜡烛').checked, false);
  assert.equal(control('在场景中添加蜡烛').disabled, true);
});
check('select all never adds a second carrot task or enables candle implicitly', () => {
  button('全选');
  assert.equal(config.environments.length, 4);
  assert.ok(config.environments.includes(base));
  assert.ok(!config.environments.includes(candle));
  control('在场景中添加蜡烛').onChange({ target: { checked: true } });
  button('全选');
  assert.equal(config.environments.length, 4);
  assert.ok(config.environments.includes(candle));
  assert.ok(!config.environments.includes(base));
  button('清空');
  assert.deepEqual(config.environments, []);
});
check('legacy draft variants merge while other tasks survive', () => {
  assert.deepEqual(ui.normalizeTaskSelection([base, 'simpler-bridge-spoon', candle]), [candle, 'simpler-bridge-spoon']);
  assert.deepEqual(ui.normalizeTaskSelection([candle, base, candle]), [candle]);
});
check('switching model retains the supported plain task without candle', () => {
  const memory = catalog.models.find(m => m.id === 'memoryvla-bridge');
  assert.deepEqual(ui.tasksForModel([candle], memory.environments), [base]);
  assert.deepEqual(ui.tasksForModel([candle], model.environments), [candle]);
  assert.ok(!ui.ssr(ui.EnvTask, { ...ctx(), model: memory }).includes('aria-label="在场景中添加蜡烛"'));
});
check('model count and review describe one task with an add-on', () => {
  config.environments = [candle];
  const rendered = ui.ssr(ui.Model, ctx());
  const label = rendered.split('<label').find(s => s.includes('<h3>OpenVLA') && s.includes('>openvla-bridge<'));
  assert.ok(label && label.includes('4 个') && !label.includes('5 个'));
  assert.ok(ui.ssr(ui.Review, ctx()).includes('胡萝卜放盘（添加蜡烛）'));
  assert.ok(html().includes('具身危险场景'));
});
fs.writeFileSync(path.join(out, 'frontend-checks.json'), JSON.stringify({ passed: checks.length, checks }, null, 2));
