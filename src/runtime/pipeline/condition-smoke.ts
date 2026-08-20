import type * as ort from 'onnxruntime-web/jspi';

const frameCount = 125;
const hiddenGroups = 8;
const hiddenSize = 4096;
const conditionLength = 430;
const conditionSize = 2048;

export interface ConditionSmokeRuntime {
  ort: typeof ort;
  session: ort.InferenceSession;
}

export interface ConditionSmokeMetrics {
  elapsedMs: number;
  shape: readonly number[];
  outputLocation: string;
  finite: boolean;
}

function analyticInput() {
  const values = new Uint16Array(frameCount * hiddenGroups * hiddenSize);
  for (let index = 0; index < values.length; index++) values[index] = index % 2 === 0 ? 0x2800 : 0xa800;
  return values;
}

function finite(data: ort.Tensor['data']) {
  return data instanceof Uint16Array
    ? data.every((value) => (value & 0x7c00) !== 0x7c00)
    : Array.from(data as Float32Array).every(Number.isFinite);
}

export async function runConditionSmoke(runtime: ConditionSmokeRuntime): Promise<ConditionSmokeMetrics> {
  const input = new runtime.ort.Tensor(
    'float16',
    analyticInput(),
    [1, frameCount, hiddenGroups * hiddenSize],
  );
  let output: ort.Tensor | undefined;
  try {
    const started = performance.now();
    const outputs = await runtime.session.run({ frame_hiddens: input });
    const elapsedMs = performance.now() - started;
    output = outputs.condition;
    if (!output) throw new Error('condition encoder did not return condition');
    if (output.dims.join(',') !== `1,${conditionLength},${conditionSize}`)
      throw new Error('condition encoder returned an unexpected shape');
    if (output.location !== 'gpu-buffer') throw new Error('condition output is not GPU-resident');
    const outputLocation = output.location;
    const data = await output.getData();
    return {
      elapsedMs,
      shape: [...output.dims],
      outputLocation,
      finite: finite(data),
    };
  } finally {
    output?.dispose();
    input.dispose();
  }
}
