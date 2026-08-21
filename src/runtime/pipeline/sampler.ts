export interface SamplingOptions {
  guidance: number;
  topK: number;
  temperature: number;
  /** Injected draws make trace replay deterministic, without claiming PyTorch RNG equivalence. */
  draw: () => number;
}

const f32 = Math.fround;
const finiteLogit = (value: number) =>
  Number.isNaN(value) ? -1e9 : value === Infinity ? 1e9 : value === -Infinity ? -1e9 : value;

function kthThreshold(
  values: ArrayLike<number>,
  topK: number,
  allowed?: ReadonlySet<number>,
  excludedIndex?: number,
) {
  const candidates: number[] = [];
  for (let index = 0; index < values.length; index++)
    if (index !== excludedIndex && (!allowed || allowed.has(index)))
      candidates.push(finiteLogit(values[index]));
  if (!candidates.length) throw new Error('sampling distribution is empty');
  candidates.sort((left, right) => right - left);
  return candidates[Math.min(topK, candidates.length) - 1];
}

function categorical(
  values: ArrayLike<number>,
  threshold: number,
  draw: () => number,
  allowed?: ReadonlySet<number>,
) {
  const random = draw();
  if (!(random >= 0 && random < 1)) throw new Error('random draw must be in [0, 1)');
  let maximum = -Infinity;
  for (let index = 0; index < values.length; index++) {
    if (allowed && !allowed.has(index)) continue;
    const value = finiteLogit(values[index]);
    if (value >= threshold) maximum = Math.max(maximum, value);
  }
  let total = f32(0);
  for (let index = 0; index < values.length; index++) {
    if (allowed && !allowed.has(index)) continue;
    const value = finiteLogit(values[index]);
    if (value < threshold) continue;
    const weight = f32(Math.exp(f32(value - maximum)));
    total = f32(total + weight);
  }
  let target = f32(random * total);
  let lastWeightedIndex = -1;
  // Recompute weights in vocabulary order instead of retaining a full-vocabulary array.
  for (let index = 0; index < values.length; index++) {
    if (allowed && !allowed.has(index)) continue;
    const value = finiteLogit(values[index]);
    if (value < threshold) continue;
    const weight = f32(Math.exp(f32(value - maximum)));
    if (weight !== 0) lastWeightedIndex = index;
    target = f32(target - weight);
    if (target < 0) return index;
  }
  if (lastWeightedIndex >= 0) return lastWeightedIndex;
  throw new Error('sampling distribution is empty');
}

function guidedTopK(
  conditional: Float32Array,
  unconditional: Float32Array,
  options: SamplingOptions,
  restrictToConditionalTopK: boolean,
  excludedIndex?: number,
) {
  if (conditional.length !== unconditional.length) throw new Error('classifier-free guidance lanes disagree');
  if (!Number.isInteger(options.topK) || options.topK < 1) throw new Error('topK must be a positive integer');
  if (!Number.isFinite(options.temperature) || options.temperature <= 0)
    throw new Error('temperature must be finite and greater than zero');
  const conditionalThreshold = restrictToConditionalTopK
    ? kthThreshold(conditional, options.topK, undefined, excludedIndex)
    : -Infinity;
  const guided = new Float32Array(conditional.length);
  const allowed = restrictToConditionalTopK || excludedIndex !== undefined
    ? new Set<number>()
    : undefined;
  for (let index = 0; index < conditional.length; index++) {
    if (index === excludedIndex) continue;
    const cond = f32(finiteLogit(conditional[index]));
    if (cond < conditionalThreshold) {
      guided[index] = -Infinity;
      continue;
    }
    const uncond = f32(finiteLogit(unconditional[index]));
    const guidedLogit = f32(uncond + f32(f32(cond - uncond) * f32(options.guidance)));
    guided[index] = f32(guidedLogit / f32(options.temperature));
    allowed?.add(index);
  }
  return categorical(
    guided,
    kthThreshold(guided, options.topK, allowed),
    options.draw,
    allowed,
  );
}

export function sampleSemantic(conditional: Float32Array, unconditional: Float32Array, options: SamplingOptions) {
  if (conditional.length !== 16_385) throw new Error('semantic logits must contain 16,384 rows plus audio end');
  return guidedTopK(conditional, unconditional, options, true);
}

export function sampleSemanticExcludingAudioEnd(
  conditional: Float32Array,
  unconditional: Float32Array,
  options: SamplingOptions,
) {
  if (conditional.length !== 16_385) throw new Error('semantic logits must contain 16,384 rows plus audio end');
  return guidedTopK(conditional, unconditional, options, true, 16_384);
}

export function sampleResidual(conditional: Float32Array, unconditional: Float32Array, options: SamplingOptions) {
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
