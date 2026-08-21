import type * as ort from 'onnxruntime-web/jspi';

import {
  createStereoPcm16Wav,
  encodeStereoPcm16Wav,
  writeStereoPcm16WavChannel,
} from '../audio/wav';
import type { RetainedFramesPlan } from './duration-plan';

const latentShape = [1, 128, 430] as const;
const latentValues = 128 * 430;
const waveformSamples = 220_160;
const waveformValues = 2 * waveformSamples;
const waveformShape = [1, 2, waveformSamples] as const;

export interface VocoderGenerationRuntime {
  ort: typeof ort;
  session: ort.InferenceSession;
}

export interface VocoderChannelCompletion {
  chunkIndex: number;
  channel: 'left' | 'right';
  completedCalls: number;
  totalCalls: number;
  inferenceMs: number;
  pcmWriteMs: number;
}

export type VocoderChannelCompletionObserver = (completion: VocoderChannelCompletion) => void;

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

export async function generateVariableVocoderWav(
  runtime: VocoderGenerationRuntime,
  plan: Pick<RetainedFramesPlan, 'chunks' | 'samplesPerChannel'>,
  latentChunks: readonly Uint16Array[],
  onChannelComplete?: VocoderChannelCompletionObserver,
  onWavStart?: () => void,
): Promise<ArrayBuffer> {
  if (latentChunks.length !== plan.chunks.length)
    throw new Error(`vocoder requires exactly ${plan.chunks.length} latent chunks`);
  let plannedSamples = 0;
  for (let index = 0; index < plan.chunks.length; index++) {
    const chunk = plan.chunks[index];
    const expectedValues = 128 * chunk.latentLength;
    if (latentChunks[index].length !== expectedValues)
      throw new Error(
        `vocoder chunk ${index} latents must contain exactly ${expectedValues} float16 values`,
      );
    if (
      chunk.cropLeftLatents < 0
      || chunk.cropRightLatents < 0
      || chunk.cropLeftLatents + chunk.cropRightLatents > chunk.latentLength
      || chunk.samplesPerChannel
        !== (chunk.latentLength - chunk.cropLeftLatents - chunk.cropRightLatents) * 512
    ) throw new Error(`vocoder chunk ${index} crop metadata is invalid`);
    plannedSamples += chunk.samplesPerChannel;
  }
  if (plannedSamples !== plan.samplesPerChannel)
    throw new Error('vocoder plan sample total does not match its chunks');

  onWavStart?.();
  const wav = createStereoPcm16Wav(plan.samplesPerChannel);
  let sampleOffset = 0;
  let completedCalls = 0;
  const totalCalls = plan.chunks.length * 2;
  for (let index = 0; index < plan.chunks.length; index++) {
    const chunk = plan.chunks[index];
    const channelValues = 64 * chunk.latentLength;
    const latentBits = latentChunks[index];
    const leftTiming = await runMonoVocoderChannel(
      runtime,
      latentBits.subarray(0, channelValues),
      chunk.latentLength,
      chunk.cropLeftLatents,
      chunk.cropRightLatents,
      wav,
      0,
      sampleOffset,
    );
    completedCalls++;
    onChannelComplete?.({
      chunkIndex: index,
      channel: 'left',
      completedCalls,
      totalCalls,
      ...leftTiming,
    });
    const rightTiming = await runMonoVocoderChannel(
      runtime,
      latentBits.subarray(channelValues),
      chunk.latentLength,
      chunk.cropLeftLatents,
      chunk.cropRightLatents,
      wav,
      1,
      sampleOffset,
    );
    completedCalls++;
    onChannelComplete?.({
      chunkIndex: index,
      channel: 'right',
      completedCalls,
      totalCalls,
      ...rightTiming,
    });
    sampleOffset += chunk.samplesPerChannel;
  }
  return wav;
}

async function runMonoVocoderChannel(
  runtime: VocoderGenerationRuntime,
  latentBits: Uint16Array,
  latentLength: number,
  cropLeftLatents: number,
  cropRightLatents: number,
  wav: ArrayBuffer,
  channel: 0 | 1,
  sampleOffset: number,
): Promise<Pick<VocoderChannelCompletion, 'inferenceMs' | 'pcmWriteMs'>> {
  const input = new runtime.ort.Tensor('float16', latentBits, [1, 64, latentLength]);
  let output: ort.Tensor | undefined;
  try {
    const inferenceStarted = performance.now();
    const outputs = await runtime.session.run({ latents: input });
    output = outputs.waveform;
    if (!output) throw new Error('vocoder did not return waveform');
    if (output.type !== 'float32') throw new Error('vocoder waveform must be float32');
    const waveformSamples = 512 * latentLength;
    if (
      output.dims.length !== 3
      || output.dims[0] !== 1
      || output.dims[1] !== 1
      || output.dims[2] !== waveformSamples
    ) throw new Error(`vocoder waveform must have shape [1,1,${waveformSamples}]`);
    const waveform = await output.getData();
    if (!(waveform instanceof Float32Array))
      throw new Error('vocoder waveform data must be a Float32Array');
    if (waveform.length !== waveformSamples)
      throw new Error(`vocoder waveform must contain exactly ${waveformSamples} samples`);
    if (!waveform.every(Number.isFinite))
      throw new Error('vocoder waveform samples must be finite');
    const inferenceMs = performance.now() - inferenceStarted;
    const pcmWriteStarted = performance.now();
    writeStereoPcm16WavChannel(
      wav,
      channel,
      sampleOffset,
      waveform.subarray(cropLeftLatents * 512, waveformSamples - cropRightLatents * 512),
    );
    return {
      inferenceMs,
      pcmWriteMs: performance.now() - pcmWriteStarted,
    };
  } finally {
    output?.dispose();
    input.dispose();
  }
}
