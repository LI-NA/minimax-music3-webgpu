import type * as ort from 'onnxruntime-web/jspi';

import { encodeStereoPcm16Wav } from '../audio/wav';

const latentShape = [1, 128, 430] as const;
const latentValues = 128 * 430;
const waveformSamples = 220_160;
const waveformValues = 2 * waveformSamples;
const waveformShape = [1, 2, waveformSamples] as const;

export interface VocoderGenerationRuntime {
  ort: typeof ort;
  session: ort.InferenceSession;
}

export function analyticVocoderLatents(): Uint16Array {
  const pattern = [0x2800, 0xa800, 0x3000, 0xb000, 0x3400, 0xb400] as const;
  const values = new Uint16Array(latentValues);
  for (let index = 0; index < values.length; index++) values[index] = pattern[index % pattern.length];
  return values;
}

export async function generateFixedVocoderWav(
  runtime: VocoderGenerationRuntime,
  latentBits: Uint16Array,
): Promise<ArrayBuffer> {
  if (latentBits.length !== latentValues)
    throw new Error(`latents must contain exactly ${latentValues} float16 values`);
  const input = new runtime.ort.Tensor('float16', latentBits, latentShape);
  let output: ort.Tensor | undefined;
  try {
    const outputs = await runtime.session.run({ latents: input });
    output = outputs.waveform;
    if (!output) throw new Error('vocoder did not return waveform');
    if (output.type !== 'float32') throw new Error('vocoder waveform must be float32');
    if (
      output.dims.length !== waveformShape.length
      || output.dims.some((value, index) => value !== waveformShape[index])
    ) throw new Error('vocoder waveform must have shape [1,2,220160]');
    const waveform = await output.getData();
    if (!(waveform instanceof Float32Array))
      throw new Error('vocoder waveform data must be a Float32Array');
    if (waveform.length !== waveformValues)
      throw new Error(`vocoder waveform must contain exactly ${waveformValues} samples`);
    if (!waveform.every(Number.isFinite))
      throw new Error('vocoder waveform samples must be finite');
    return encodeStereoPcm16Wav([
      waveform.subarray(0, waveformSamples),
      waveform.subarray(waveformSamples),
    ]);
  } finally {
    output?.dispose();
    input.dispose();
  }
}
