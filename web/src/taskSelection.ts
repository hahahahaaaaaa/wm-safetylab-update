export const CARROT_TASK = 'simpler-bridge-carrot';
export const CANDLE_VARIANT = 'simpler-bridge-carrot-candle';

/** A candle variant replaces its base task in submitted configurations. */
export function normalizeTaskSelection(ids: readonly string[] = []): string[] {
  const candle = ids.includes(CANDLE_VARIANT);
  return [...new Set(ids.map(id => candle && id === CARROT_TASK ? CANDLE_VARIANT : id))];
}

/** Count/display the underlying tasks, not their optional scene variants. */
export function taskIds(ids: readonly string[] = []): string[] {
  return [...new Set(ids.map(id => id === CANDLE_VARIANT ? CARROT_TASK : id))];
}

export function toggleTask(ids: readonly string[], task: string): string[] {
  const selected = normalizeTaskSelection(ids);
  if (taskIds(selected).includes(task)) {
    return selected.filter(id => id !== task && !(task === CARROT_TASK && id === CANDLE_VARIANT));
  }
  return [...selected, task];
}

export function setCandleAddon(ids: readonly string[], enabled: boolean): string[] {
  const selected = normalizeTaskSelection(ids);
  if (!taskIds(selected).includes(CARROT_TASK)) return selected;
  return selected.map(id => id === CARROT_TASK || id === CANDLE_VARIANT ? (enabled ? CANDLE_VARIANT : CARROT_TASK) : id);
}

export function selectAllTasks(ids: readonly string[], supported: readonly string[]): string[] {
  const ordinary = [...new Set(supported.filter(id => id !== CANDLE_VARIANT))];
  return ids.includes(CANDLE_VARIANT) && supported.includes(CANDLE_VARIANT) ? setCandleAddon(ordinary, true) : ordinary;
}

export function tasksForModel(ids: readonly string[], supported: readonly string[]): string[] {
  const selected = normalizeTaskSelection(ids).map(id =>
    id === CANDLE_VARIANT && !supported.includes(CANDLE_VARIANT) ? CARROT_TASK : id);
  const compatible = normalizeTaskSelection(selected.filter(id => supported.includes(id)));
  const first = supported.find(id => id !== CANDLE_VARIANT);
  return compatible.length || !first ? compatible : [first];
}
