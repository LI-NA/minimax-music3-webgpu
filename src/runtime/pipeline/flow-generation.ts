import type * as ort from 'onnxruntime-web/jspi';

const flowSteps = 30;
const timestepBits = new Uint32Array([
  0, 1023969424, 1032358024, 1036831952, 1040746632, 1042983596,
  1045220556, 1047457520, 1049135240, 1050253722, 1051372202, 1052490684,
  1053609164, 1054727646, 1055846126, 1056964608, 1057523848, 1058083089,
  1058642330, 1059201570, 1059760810, 1060320051, 1060879292, 1061438532,
  1061997773, 1062557013, 1063116254, 1063675494, 1064234735, 1064793975,
]);
const dtBits = new Uint32Array([
  1023969424, 1023969408, 1023969424, 1023969408, 1023969424, 1023969408,
  1023969424, 1023969408, 1023969424, 1023969408, 1023969424, 1023969408,
  1023969424, 1023969408, 1023969424, 1023969408, 1023969424, 1023969424,
  1023969408, 1023969408, 1023969424, 1023969424, 1023969408, 1023969424,
  1023969408, 1023969424, 1023969408, 1023969424, 1023969408, 1023969424,
]);

export interface FlowSchedule {
  timesteps: Float32Array;
  dts: Float32Array;
}

export interface FlowGenerationRuntime {
  ort: typeof ort;
  session: ort.InferenceSession;
}

export type FlowSmokeRuntime = FlowGenerationRuntime;

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

export function exactFlowSchedule(): FlowSchedule {
  return {
    timesteps: new Float32Array(timestepBits.slice().buffer),
    dts: new Float32Array(dtBits.slice().buffer),
  };
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

function requireFloat16(
  tensor: ort.Tensor,
  name: string,
  shape: readonly number[],
  requireGpu = false,
) {
  if (
    tensor.type !== 'float16'
    || (requireGpu && tensor.location !== 'gpu-buffer')
    || tensor.dims.length !== shape.length
    || tensor.dims.some((value, index) => value !== shape[index])
  ) throw new Error(
    `${name} must be a ${requireGpu ? 'GPU-resident ' : ''}float16 tensor with shape [${shape.join(',')}]`,
  );
}

export async function runFixedFlowStep(
  runtime: FlowGenerationRuntime,
  latents: ort.Tensor,
  condition: ort.Tensor,
  step: number,
) {
  requireFloat16(latents, 'latents', [1, 128, 430]);
  requireFloat16(condition, 'condition', [1, 430, 2048]);
  if (!Number.isInteger(step) || step < 0 || step >= flowSteps)
    throw new Error('flow step must be between 0 and 29');
  const schedule = exactFlowSchedule();
  const timestep = new runtime.ort.Tensor(
    'float16',
    new Uint16Array([float32ToFloat16Bits(schedule.timesteps[step])]),
    [1],
  );
  const dt = new runtime.ort.Tensor('float32', new Float32Array([schedule.dts[step]]), [1]);
  let next: ort.Tensor | undefined;
  try {
    const outputs = await runtime.session.run({ latents, condition, timestep, dt });
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
  }
}

export async function runFixedFlowGeneration(
  runtime: FlowGenerationRuntime,
  initialLatents: ort.Tensor,
  condition: ort.Tensor,
  onStep?: (completedSteps: number) => void,
) {
  requireFloat16(initialLatents, 'latents', [1, 128, 430]);
  requireFloat16(condition, 'condition', [1, 430, 2048]);
  let latents = initialLatents;
  try {
    for (let index = 0; index < flowSteps; index++) {
      const next = await runFixedFlowStep(runtime, latents, condition, index);
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

function analyticFp16(length: number, positive: number, negative: number) {
  const values = new Uint16Array(length);
  for (let index = 0; index < length; index++) values[index] = index % 2 === 0 ? positive : negative;
  return values;
}

function analyticFp16Tensor(
  runtime: FlowSmokeRuntime,
  values: Uint16Array,
  dims: readonly number[],
) {
  return new runtime.ort.Tensor('float16', values, dims);
}

export function areFiniteFlowValues(data: unknown) {
  if (data instanceof Uint16Array)
    return data.every((value) => (value & 0x7c00) !== 0x7c00);
  if (
    data instanceof Float32Array
    || (ArrayBuffer.isView(data) && data.constructor.name === 'Float16Array')
  ) return Array.from(data as unknown as ArrayLike<number>).every(Number.isFinite);
  throw new Error('flow output did not download as float16');
}

export async function runFlowSmoke(runtime: FlowSmokeRuntime): Promise<FlowSmokeMetrics> {
  const makeLatents = () => analyticFp16Tensor(
    runtime,
    analyticFp16(128 * 430, 0x2c00, 0xac00),
    [1, 128, 430],
  );
  const makeCondition = () => analyticFp16Tensor(
    runtime,
    analyticFp16(430 * 2048, 0x2800, 0xa800),
    [1, 430, 2048],
  );
  const oneLatents = makeLatents();
  const oneCondition = makeCondition();
  let oneOutput: ort.Tensor | undefined;
  let oneStepMs = 0;
  let oneStepLocation = '';
  let oneStepFinite = false;
  try {
    const started = performance.now();
    oneOutput = await runFixedFlowStep(runtime, oneLatents, oneCondition, 0);
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
    final = await runFixedFlowGeneration(runtime, latents, condition, () => {
      const now = performance.now();
      stepMs.push(now - previous);
      previous = now;
    });
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
