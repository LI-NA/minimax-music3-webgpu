import type * as ort from 'onnxruntime-web/jspi';
import { readGpuFp16Bits } from '../model/fp16-readback';
import type { DurationChunkPlan, RetainedFramesPlan } from './duration-plan';

const defaultFlowSteps = 30;
const defaultFlowGuidance = 1.7;
const maximumFrames = 200;
const maximumLatents = 689;
const overlapLatents = 172;
const latentChannels = 128;
const conditionWidth = 2048;
const frameHiddenWidth = 32_768;
const fp16One = 0x3c00;
const fp16Minimum = 0xfbff;
export interface FlowSchedule {
  timesteps: Float32Array;
  dts: Float32Array;
}

export interface FlowGenerationRuntime {
  ort: typeof ort;
  session: ort.InferenceSession;
}

export type FlowSmokeRuntime = FlowGenerationRuntime;

export interface ChunkedFlowGenerationRuntime {
  ort: typeof ort;
  conditionSession: ort.InferenceSession;
  flowSession: ort.InferenceSession;
}

export interface ChunkedFlowGenerationRequest {
  plan: RetainedFramesPlan;
  frameHiddens: Uint16Array;
  initialLatents: readonly Uint16Array[];
  flowGuidance: number;
  flowSteps: number;
  onConditionStart?: (chunkIndex: number) => void;
  onConditionComplete?: (timing: FlowChunkTiming) => void;
  onChunkStart?: (chunkIndex: number, completedSteps: number) => void;
  onChunkComplete?: (timing: FlowChunkTiming) => void;
  onStep?: (completedSteps: number) => void;
}

export interface FlowChunkTiming {
  chunkIndex: number;
  elapsedMs: number;
}

export interface FlowChunkResult extends DurationChunkPlan {
  latentBits: Uint16Array;
}

export interface FlowSmokeMetrics {
  oneStepMs: number;
  generationMs: number;
  stepMs: readonly number[];
  shape: readonly number[];
  oneStepLocation: string;
  finalLocation: string;
  oneStepFinite: boolean;
  finalFinite: boolean;
}

function requireFlowStepCount(numSteps: number) {
  if (!Number.isSafeInteger(numSteps) || numSteps < 1)
    throw new Error('flow step count must be a positive safe integer');
}

export function exactFlowSchedule(numSteps = defaultFlowSteps): FlowSchedule {
  requireFlowStepCount(numSteps);
  const sourceSigmas = new Float32Array(numSteps);
  const timesteps = new Float32Array(numSteps);
  const dts = new Float32Array(numSteps);
  const stop = 1 / numSteps;
  const delta = numSteps === 1 ? 0 : (stop - 1) / (numSteps - 1);
  for (let index = 0; index < numSteps; index++) {
    const sigma = index === numSteps - 1 ? stop : 1 + index * delta;
    sourceSigmas[index] = sigma;
    timesteps[index] = Math.fround(1 - sourceSigmas[index]);
  }
  for (let index = 0; index < numSteps; index++) {
    const next = index + 1 === numSteps ? 1 : timesteps[index + 1];
    dts[index] = Math.fround(next - timesteps[index]);
  }
  return { timesteps, dts };
}

export function float32ToFloat16Bits(value: number) {
  const f32 = new Float32Array([value]);
  const bits = new Uint32Array(f32.buffer)[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const fraction = bits & 0x7fffff;
  if (exponent === 0xff) return sign | (fraction ? 0x7e00 : 0x7c00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const mantissa = fraction | 0x800000;
    const shift = 14 - halfExponent;
    const rounded = (mantissa + (1 << (shift - 1)) - 1 + ((mantissa >> shift) & 1)) >> shift;
    return sign | rounded;
  }
  const rounded = fraction + 0xfff + ((fraction >>> 13) & 1);
  if (rounded & 0x800000) {
    const nextExponent = halfExponent + 1;
    return sign | (nextExponent >= 0x1f ? 0x7c00 : nextExponent << 10);
  }
  return sign | (halfExponent << 10) | (rounded >>> 13);
}

function requireFloat16(tensor: ort.Tensor, name: string, shape: readonly number[], requireGpu = false) {
  if (
    tensor.type !== 'float16' ||
    (requireGpu && tensor.location !== 'gpu-buffer') ||
    tensor.dims.length !== shape.length ||
    tensor.dims.some((value, index) => value !== shape[index])
  )
    throw new Error(
      `${name} must be a ${requireGpu ? 'GPU-resident ' : ''}float16 tensor with shape [${shape.join(',')}]`,
    );
}

export async function runFixedFlowStep(
  runtime: FlowGenerationRuntime,
  latents: ort.Tensor,
  condition: ort.Tensor,
  step: number,
  flowGuidance = defaultFlowGuidance,
  flowStepCount = defaultFlowSteps,
) {
  requireFloat16(latents, 'latents', [1, 128, 430]);
  requireFloat16(condition, 'condition', [1, 430, 2048]);
  const schedule = exactFlowSchedule(flowStepCount);
  if (!Number.isInteger(step) || step < 0 || step >= flowStepCount)
    throw new Error(`flow step must be between 0 and ${flowStepCount - 1}`);
  const timestep = new runtime.ort.Tensor(
    'float16',
    new Uint16Array([float32ToFloat16Bits(schedule.timesteps[step])]),
    [1],
  );
  const dt = new runtime.ort.Tensor('float32', new Float32Array([schedule.dts[step]]), [1]);
  const guidance = new runtime.ort.Tensor('float16', new Uint16Array([float32ToFloat16Bits(flowGuidance)]), [1]);
  let next: ort.Tensor | undefined;
  try {
    const outputs = await runtime.session.run({ latents, condition, timestep, dt, guidance });
    next = outputs.next_latents;
    if (!next) throw new Error('flow session did not return next_latents');
    requireFloat16(next, 'next_latents', [1, 128, 430], true);
    return next;
  } catch (error) {
    next?.dispose();
    throw error;
  } finally {
    timestep.dispose();
    dt.dispose();
    guidance.dispose();
  }
}

export async function runFixedFlowGeneration(
  runtime: FlowGenerationRuntime,
  initialLatents: ort.Tensor,
  condition: ort.Tensor,
  onStep?: (completedSteps: number) => void,
  flowGuidance = defaultFlowGuidance,
  flowStepCount = defaultFlowSteps,
) {
  requireFloat16(initialLatents, 'latents', [1, 128, 430]);
  requireFloat16(condition, 'condition', [1, 430, 2048]);
  requireFlowStepCount(flowStepCount);
  let latents = initialLatents;
  try {
    for (let index = 0; index < flowStepCount; index++) {
      const next = await runFixedFlowStep(runtime, latents, condition, index, flowGuidance, flowStepCount);
      latents.dispose();
      latents = next;
      onStep?.(index + 1);
    }
    return latents;
  } catch (error) {
    latents.dispose();
    throw error;
  }
}

function paddedChannelValues(values: Uint16Array, sourceLength: number) {
  const padded = new Uint16Array(latentChannels * maximumLatents);
  for (let channel = 0; channel < latentChannels; channel++)
    padded.set(values.subarray(channel * sourceLength, (channel + 1) * sourceLength), channel * maximumLatents);
  return padded;
}

function activeChannelValues(values: Uint16Array, activeLength: number) {
  const active = new Uint16Array(latentChannels * activeLength);
  for (let channel = 0; channel < latentChannels; channel++)
    active.set(
      values.subarray(channel * maximumLatents, channel * maximumLatents + activeLength),
      channel * activeLength,
    );
  return active;
}

function channelInterval(values: Uint16Array, length: number, start: number, end: number) {
  const intervalLength = end - start;
  const selected = new Uint16Array(latentChannels * intervalLength);
  for (let channel = 0; channel < latentChannels; channel++)
    selected.set(values.subarray(channel * length + start, channel * length + end), channel * intervalLength);
  return selected;
}

function restoreChannelPrefix(values: Uint16Array, prefix: Uint16Array) {
  for (let channel = 0; channel < latentChannels; channel++)
    values.set(
      prefix.subarray(channel * overlapLatents, (channel + 1) * overlapLatents),
      channel * (values.length / latentChannels),
    );
}

function nearestIndices(frameLength: number, latentLength: number) {
  const result = new BigInt64Array(maximumLatents);
  const scale = Math.fround(frameLength / latentLength);
  for (let index = 0; index < latentLength; index++) result[index] = BigInt(Math.floor(Math.fround(index * scale)));
  return result;
}

function activeLatentMask(latentLength: number) {
  const mask = new Uint16Array(maximumLatents);
  mask.fill(fp16One, 0, latentLength);
  return mask;
}

function keyAttentionBias(latentLength: number) {
  const bias = new Uint16Array(maximumLatents + 1);
  bias.fill(fp16Minimum, latentLength + 1);
  return bias;
}

function conditionInterval(values: Uint16Array, start: number, end: number) {
  return values.slice(start * conditionWidth, end * conditionWidth);
}

async function encodeMaximumCondition(
  runtime: ChunkedFlowGenerationRuntime,
  frameHiddens: Uint16Array,
  chunk: DurationChunkPlan,
) {
  const paddedFrames = new Uint16Array(maximumFrames * frameHiddenWidth);
  paddedFrames.set(
    frameHiddens.subarray(
      chunk.startFrame * frameHiddenWidth,
      (chunk.startFrame + chunk.frameLength) * frameHiddenWidth,
    ),
  );
  const frames = new runtime.ort.Tensor('float16', paddedFrames, [1, maximumFrames, frameHiddenWidth]);
  const nearest = new runtime.ort.Tensor('int64', nearestIndices(chunk.frameLength, chunk.latentLength), [
    maximumLatents,
  ]);
  const mask = new runtime.ort.Tensor('float16', activeLatentMask(chunk.latentLength), [1, maximumLatents, 1]);
  let output: ort.Tensor | undefined;
  try {
    const started = performance.now();
    const outputs = await runtime.conditionSession.run({
      frame_hiddens: frames,
      nearest_index: nearest,
      active_latent_mask: mask,
    });
    output = outputs.condition;
    if (!output) throw new Error('condition session did not return condition');
    const bits = await readGpuFp16Bits(output, [1, maximumLatents, conditionWidth], 'condition');
    return { bits, elapsedMs: performance.now() - started };
  } finally {
    output?.dispose();
    frames.dispose();
    nearest.dispose();
    mask.dispose();
  }
}

async function generateMaximumFlowChunk(
  runtime: ChunkedFlowGenerationRuntime,
  request: ChunkedFlowGenerationRequest,
  chunkIndex: number,
  conditionBits: Uint16Array,
  previousLatent: Uint16Array | undefined,
) {
  const chunk = request.plan.chunks[chunkIndex];
  const initialLatents = request.initialLatents[chunkIndex];
  const flowGuidance = request.flowGuidance;
  const flowStepCount = request.flowSteps;
  const completedSteps = chunkIndex * flowStepCount;
  const noisePromptBits = previousLatent
    ? channelInterval(initialLatents, chunk.latentLength, 0, overlapLatents)
    : new Uint16Array(latentChannels * overlapLatents);
  const previousLatentBits = previousLatent ?? new Uint16Array(latentChannels * overlapLatents);
  const condition = new runtime.ort.Tensor('float16', conditionBits, [1, maximumLatents, conditionWidth]);
  const activeMask = new runtime.ort.Tensor('float16', activeLatentMask(chunk.latentLength), [1, maximumLatents, 1]);
  const attentionBias = new runtime.ort.Tensor('float16', keyAttentionBias(chunk.latentLength), [
    1,
    1,
    1,
    maximumLatents + 1,
  ]);
  const noisePrompt = new runtime.ort.Tensor('float16', noisePromptBits, [1, latentChannels, overlapLatents]);
  const previous = new runtime.ort.Tensor('float16', previousLatentBits, [1, latentChannels, overlapLatents]);
  const overlapEnabled = new runtime.ort.Tensor('float16', new Uint16Array([previousLatent ? fp16One : 0]), [1]);
  let latents: ort.Tensor = new runtime.ort.Tensor('float16', paddedChannelValues(initialLatents, chunk.latentLength), [
    1,
    latentChannels,
    maximumLatents,
  ]);
  try {
    const schedule = exactFlowSchedule(flowStepCount);
    request.onChunkStart?.(chunkIndex, completedSteps);
    const started = performance.now();
    for (let index = 0; index < flowStepCount; index++) {
      const timestep = new runtime.ort.Tensor(
        'float16',
        new Uint16Array([float32ToFloat16Bits(schedule.timesteps[index])]),
        [1],
      );
      const dt = new runtime.ort.Tensor('float32', new Float32Array([schedule.dts[index]]), [1]);
      const guidance = new runtime.ort.Tensor('float16', new Uint16Array([float32ToFloat16Bits(flowGuidance)]), [1]);
      let next: ort.Tensor | undefined;
      try {
        const outputs = await runtime.flowSession.run({
          latents,
          condition,
          timestep,
          dt,
          guidance,
          active_latent_mask: activeMask,
          key_attention_bias: attentionBias,
          noise_prompt: noisePrompt,
          previous_latent: previous,
          overlap_enabled: overlapEnabled,
        });
        next = outputs.next_latents;
        if (!next) throw new Error('flow session did not return next_latents');
        requireFloat16(next, 'next_latents', [1, latentChannels, maximumLatents], true);
        latents.dispose();
        latents = next;
        next = undefined;
        request.onStep?.(completedSteps + index + 1);
      } finally {
        next?.dispose();
        timestep.dispose();
        dt.dispose();
        guidance.dispose();
      }
    }
    const downloaded = await readGpuFp16Bits(latents, [1, latentChannels, maximumLatents], 'final latents');
    const active = activeChannelValues(downloaded, chunk.latentLength);
    if (previousLatent) restoreChannelPrefix(active, previousLatent);
    request.onChunkComplete?.({ chunkIndex, elapsedMs: performance.now() - started });
    return active;
  } finally {
    latents.dispose();
    condition.dispose();
    activeMask.dispose();
    attentionBias.dispose();
    noisePrompt.dispose();
    previous.dispose();
    overlapEnabled.dispose();
  }
}

export async function runChunkedFlowGeneration(
  runtime: ChunkedFlowGenerationRuntime,
  request: ChunkedFlowGenerationRequest,
): Promise<FlowChunkResult[]> {
  if (request.frameHiddens.length !== request.plan.retainedFrames * frameHiddenWidth)
    throw new Error('frame hiddens do not match the retained-frame plan');
  if (request.initialLatents.length !== request.plan.chunks.length)
    throw new Error('initial latent chunks do not match the duration plan');
  const result: FlowChunkResult[] = [];
  let previousLatent: Uint16Array | undefined;
  let previousCondition: Uint16Array | undefined;
  for (let chunkIndex = 0; chunkIndex < request.plan.chunks.length; chunkIndex++) {
    const chunk = request.plan.chunks[chunkIndex];
    const initialLatents = request.initialLatents[chunkIndex];
    if (initialLatents.length !== latentChannels * chunk.latentLength)
      throw new Error(`initial latents for chunk ${chunkIndex} have an invalid length`);
    request.onConditionStart?.(chunkIndex);
    const encoded = await encodeMaximumCondition(runtime, request.frameHiddens, chunk);
    const condition = encoded.bits;
    request.onConditionComplete?.({ chunkIndex, elapsedMs: encoded.elapsedMs });
    condition.fill(0, chunk.latentLength * conditionWidth);
    if (previousCondition) condition.set(previousCondition, 0);
    const latentBits = await generateMaximumFlowChunk(runtime, request, chunkIndex, condition, previousLatent);
    result.push({ ...chunk, latentBits });
    if (chunkIndex + 1 < request.plan.chunks.length) {
      const carryStart = chunk.latentLength - 2 * overlapLatents;
      const carryEnd = chunk.latentLength - overlapLatents;
      previousLatent = channelInterval(latentBits, chunk.latentLength, carryStart, carryEnd);
      previousCondition = conditionInterval(condition, carryStart, carryEnd);
    }
  }
  return result;
}

function analyticFp16(length: number, positive: number, negative: number) {
  const values = new Uint16Array(length);
  for (let index = 0; index < length; index++) values[index] = index % 2 === 0 ? positive : negative;
  return values;
}

function analyticFp16Tensor(runtime: FlowSmokeRuntime, values: Uint16Array, dims: readonly number[]) {
  return new runtime.ort.Tensor('float16', values, dims);
}

export function areFiniteFlowValues(data: unknown) {
  if (data instanceof Uint16Array) return data.every((value) => (value & 0x7c00) !== 0x7c00);
  if (data instanceof Float32Array || (ArrayBuffer.isView(data) && data.constructor.name === 'Float16Array'))
    return Array.from(data as unknown as ArrayLike<number>).every(Number.isFinite);
  throw new Error('flow output did not download as float16');
}

export async function runFlowSmoke(runtime: FlowSmokeRuntime): Promise<FlowSmokeMetrics> {
  const makeLatents = () => analyticFp16Tensor(runtime, analyticFp16(128 * 430, 0x2c00, 0xac00), [1, 128, 430]);
  const makeCondition = () => analyticFp16Tensor(runtime, analyticFp16(430 * 2048, 0x2800, 0xa800), [1, 430, 2048]);
  const oneLatents = makeLatents();
  const oneCondition = makeCondition();
  let oneOutput: ort.Tensor | undefined;
  let oneStepMs = 0;
  let oneStepLocation = '';
  let oneStepFinite = false;
  try {
    const started = performance.now();
    oneOutput = await runFixedFlowStep(runtime, oneLatents, oneCondition, 0, defaultFlowGuidance, defaultFlowSteps);
    oneStepMs = performance.now() - started;
    oneStepLocation = oneOutput.location;
    oneStepFinite = areFiniteFlowValues(await oneOutput.getData());
  } finally {
    oneOutput?.dispose();
    oneCondition.dispose();
    oneLatents.dispose();
  }

  const latents = makeLatents();
  const condition = makeCondition();
  const stepMs: number[] = [];
  let previous = performance.now();
  const started = previous;
  let final: ort.Tensor | undefined;
  try {
    final = await runFixedFlowGeneration(
      runtime,
      latents,
      condition,
      () => {
        const now = performance.now();
        stepMs.push(now - previous);
        previous = now;
      },
      defaultFlowGuidance,
      defaultFlowSteps,
    );
    const generationMs = performance.now() - started;
    const shape = [...final.dims];
    const finalLocation = final.location;
    const finalFinite = areFiniteFlowValues(await final.getData());
    return {
      oneStepMs,
      generationMs,
      stepMs,
      shape,
      oneStepLocation,
      finalLocation,
      oneStepFinite,
      finalFinite,
    };
  } finally {
    final?.dispose();
    condition.dispose();
  }
}
