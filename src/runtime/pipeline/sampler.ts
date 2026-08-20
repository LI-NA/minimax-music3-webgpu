export interface SamplingOptions {
  guidance: number;
  topK: number;
  draw: () => number;
}

const finiteLogit = (value: number) =>
  Number.isNaN(value) ? -1e9 : value === Infinity ? 1e9 : value === -Infinity ? -1e9 : value;

function stableTopK(values: ArrayLike<number>, topK: number, allowed?: ReadonlySet<number>) {
  const candidates: { index: number; value: number }[] = [];
  for (let index = 0; index < values.length; index++)
    if (!allowed || allowed.has(index)) candidates.push({ index, value: finiteLogit(values[index]) });
  candidates.sort((left, right) => right.value - left.value || left.index - right.index);
  return candidates.slice(0, Math.min(topK, candidates.length));
}

function categorical(candidates: readonly { index: number; value: number }[], draw: () => number) {
  if (!candidates.length) throw new Error('sampling distribution is empty');
  const random = draw();
  if (!(random >= 0 && random < 1)) throw new Error('random draw must be in [0, 1)');
  const maximum = candidates[0].value;
  const weights = candidates.map(({ value }) => Math.exp(value - maximum));
  const total = weights.reduce((sum, value) => sum + value, 0);
  let target = random * total;
  for (let index = 0; index < candidates.length; index++) {
    target -= weights[index];
    if (target < 0) return candidates[index].index;
  }
  return candidates.at(-1)!.index;
}

function guidedTopK(
  conditional: Float32Array,
  unconditional: Float32Array,
  options: SamplingOptions,
  restrictToConditionalTopK: boolean,
) {
  if (conditional.length !== unconditional.length) throw new Error('classifier-free guidance lanes disagree');
  if (!Number.isInteger(options.topK) || options.topK < 1) throw new Error('topK must be a positive integer');
  const allowed = restrictToConditionalTopK
    ? new Set(stableTopK(conditional, options.topK).map(({ index }) => index))
    : undefined;
  const guided = new Float64Array(conditional.length);
  for (let index = 0; index < conditional.length; index++) {
    const cond = finiteLogit(conditional[index]);
    const uncond = finiteLogit(unconditional[index]);
    guided[index] = uncond + options.guidance * (cond - uncond);
  }
  return categorical(stableTopK(guided, options.topK, allowed), options.draw);
}

export function sampleSemantic(
  conditional: Float32Array,
  unconditional: Float32Array,
  options: SamplingOptions,
) {
  if (conditional.length !== 16_385) throw new Error('semantic logits must contain 16,384 rows plus audio end');
  return guidedTopK(conditional, unconditional, options, true);
}

export function sampleResidual(
  conditional: Float32Array,
  unconditional: Float32Array,
  options: SamplingOptions,
) {
  if (conditional.length !== 1_024) throw new Error('residual logits must contain 1,024 rows');
  return guidedTopK(conditional, unconditional, options, false);
}

export function createDeterministicDraw(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}
