import type * as ort from 'onnxruntime-web/jspi';
import { readGpuFp16Bits } from '../model/fp16-readback';
import { planRetainedFrames, type Termination } from './duration-plan';
import { float32ToFloat16Bits } from './flow-generation';
import type { GeneratedFrame, GeneratedFrames } from './rvq-generation';
import { createDeterministicDraw } from './sampler';
import { EarlyAudioEndError } from './rvq-generation';

const groupsPerFrame = 8;
const hiddenSize = 4096;
const promptTokens = 40;
const conditionValues = 430 * 2048;
const latentValues = 128 * 430;
const wavBytes = 880_684;

export type MusicGenerationProgress =
  | { stage: 'autoregressive'; retainedFrames: number }
  | { stage: 'condition' }
  | { stage: 'flow'; completedSteps: number }
  | { stage: 'vocoder' }
  | { stage: 'wav' }
  | { stage: 'complete' };

export interface MusicGenerationStages {
  autoregressive(seed: number): Promise<readonly GeneratedFrame[]>;
  condition(frameBits: Uint16Array): Promise<Uint16Array>;
  flow(conditionBits: Uint16Array, seed: number, onStep: (step: number) => void): Promise<Uint16Array>;
  vocoder(latentBits: Uint16Array): Promise<ArrayBuffer>;
}

export interface MusicGenerationResult {
  wav: ArrayBuffer;
  attemptedSeeds: readonly number[];
  retainedFrames: number;
  termination: Termination;
  hiddenBytes: number;
  conditionBytes: number;
  latentBytes: number;
}

export function flattenFrameHiddens(frames: readonly GeneratedFrame[]) {
  const generated = frames as Partial<GeneratedFrames>;
  const termination: Termination = generated.termination === 'natural-end' ? 'natural-end' : 'max-frames';
  planRetainedFrames({ retainedFrames: frames.length, promptTokens, termination });
  const valuesPerFrame = groupsPerFrame * hiddenSize;
  frames.forEach((frame) => {
    if (frame.hiddenGroups.length !== valuesPerFrame)
      throw new Error('retained frame hidden groups have an invalid shape');
  });
  if (generated.hiddenGroups) {
    if (generated.hiddenGroups.length !== frames.length * valuesPerFrame)
      throw new Error('flat retained hidden groups have an invalid shape');
    frames.forEach((frame, index) => {
      if (
        frame.hiddenGroups.buffer !== generated.hiddenGroups!.buffer ||
        frame.hiddenGroups.byteOffset !== generated.hiddenGroups!.byteOffset + index * valuesPerFrame * 2
      ) throw new Error('retained frame hidden groups must be contiguous views of the flat buffer');
    });
    return generated.hiddenGroups;
  }
  const values = new Uint16Array(frames.length * valuesPerFrame);
  frames.forEach((frame, index) => values.set(frame.hiddenGroups, index * valuesPerFrame));
  return values;
}

export function deterministicGaussianFp16(seed: number, length: number) {
  if (!Number.isSafeInteger(length) || length < 0) throw new Error('Gaussian length must be non-negative');
  const draw = createDeterministicDraw(seed);
  const values = new Uint16Array(length);
  for (let index = 0; index < length; index += 2) {
    const radius = Math.sqrt(-2 * Math.log1p(-draw()));
    const angle = 2 * Math.PI * draw();
    values[index] = float32ToFloat16Bits(radius * Math.cos(angle));
    if (index + 1 < length) values[index + 1] = float32ToFloat16Bits(radius * Math.sin(angle));
  }
  return values;
}

export async function readExactGpuFp16(
  tensor: ort.Tensor,
  dims: readonly number[],
  name: string,
) {
  return readGpuFp16Bits(tensor, dims, name);
}

export async function generateFiveSecondMusic(
  stages: MusicGenerationStages,
  seed: number,
  onProgress?: (progress: MusicGenerationProgress) => void,
): Promise<MusicGenerationResult> {
  const attemptedSeeds: number[] = [];
  let frames: readonly GeneratedFrame[] | undefined;
  let selectedSeed = seed;
  for (let attempt = 0; attempt < 2; attempt++) {
    selectedSeed = seed + attempt;
    attemptedSeeds.push(selectedSeed);
    try {
      const candidate = await stages.autoregressive(selectedSeed);
      const candidateTermination = (candidate as Partial<GeneratedFrames>).termination;
      if (candidate.length !== 125 || candidateTermination === 'natural-end') {
        if (attempt === 1)
          throw new Error(`audio end sampled early for seeds ${attemptedSeeds.join(', ')}`);
        continue;
      }
      frames = candidate;
      break;
    } catch (error) {
      if (!(error instanceof EarlyAudioEndError)) throw error;
      if (attempt === 1)
        throw new Error(`audio end sampled early for seeds ${attemptedSeeds.join(', ')}`);
    }
  }
  if (!frames) throw new Error('autoregressive generation did not return frames');
  const termination: Termination = (frames as Partial<GeneratedFrames>).termination === 'natural-end'
    ? 'natural-end'
    : 'max-frames';
  onProgress?.({ stage: 'autoregressive', retainedFrames: frames.length });
  const frameBits = flattenFrameHiddens(frames);
  onProgress?.({ stage: 'condition' });
  const conditionBits = await stages.condition(frameBits);
  if (conditionBits.length !== conditionValues) throw new Error('condition must contain exactly 880,640 FP16 values');
  const latentBits = await stages.flow(conditionBits, selectedSeed, (completedSteps) =>
    onProgress?.({ stage: 'flow', completedSteps }),
  );
  if (latentBits.length !== latentValues) throw new Error('latents must contain exactly 55,040 FP16 values');
  onProgress?.({ stage: 'vocoder' });
  const wav = await stages.vocoder(latentBits);
  if (wav.byteLength !== wavBytes) throw new Error('WAV must contain exactly 880,684 bytes');
  onProgress?.({ stage: 'wav' });
  onProgress?.({ stage: 'complete' });
  return {
    wav,
    attemptedSeeds,
    retainedFrames: frames.length,
    termination,
    hiddenBytes: frameBits.byteLength,
    conditionBytes: conditionBits.byteLength,
    latentBytes: latentBits.byteLength,
  };
}
