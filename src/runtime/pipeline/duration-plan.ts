export type DurationChunkPlan = {
  startFrame: number;
  frameLength: number;
  latentLength: number;
  cropLeftLatents: number;
  cropRightLatents: number;
  samplesPerChannel: number;
};
export type Termination = 'max-frames' | 'natural-end';
export type RetainedFramesPlan = {
  retainedFrames: number;
  termination: Termination;
  chunks: DurationChunkPlan[];
  samplesPerChannel: number;
  wavBytes: number;
  flowCalls: number;
  vocoderCalls: number;
  semanticDecisions: number;
  rvqCalls: number;
  feedbackCalls: number;
};
export type RetainedFramesPlanRequest = {
  retainedFrames: number;
  promptTokens: number;
  termination: Termination;
  flowSteps?: number;
};
export type DurationPlanRequest = {
  durationSeconds: number;
  promptTokens: number;
  flowSteps?: number;
};

const FRAMES_PER_SECOND = 25;
const MAX_PRODUCT_DURATION_SECONDS = 300;
const MAX_RETAINED_FRAMES = 7500;
const MAX_CONTEXT_TOKENS = 10_240;
const CHUNK_FRAMES = 200;
const CHUNK_HOP_FRAMES = 100;
const SAMPLES_PER_LATENT = 512;

function validateNonNegativeInteger(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0)
    throw new Error(`${label} must be non-negative integers`);
}

function validateFlowSteps(value: unknown): asserts value is number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1)
    throw new Error('Flow steps must be a positive safe integer');
}

function validateDuration({ durationSeconds, promptTokens }: DurationPlanRequest): void {
  validateNonNegativeInteger(promptTokens, 'Prompt tokens');
  if (typeof durationSeconds !== 'number' || !Number.isFinite(durationSeconds)
    || durationSeconds <= 0 || durationSeconds > MAX_PRODUCT_DURATION_SECONDS)
    throw new Error('Duration seconds must be a finite number greater than zero and at most 300');
  if (Math.floor(durationSeconds * FRAMES_PER_SECOND) < 1)
    throw new Error('Duration must produce at least one frame');
}

export function planRetainedFrames({
  retainedFrames,
  promptTokens,
  termination,
  flowSteps = 30,
}: RetainedFramesPlanRequest): RetainedFramesPlan {
  validateNonNegativeInteger(retainedFrames, 'Retained frames');
  validateNonNegativeInteger(promptTokens, 'Prompt tokens');
  validateFlowSteps(flowSteps);
  if (retainedFrames < 1 || retainedFrames > MAX_RETAINED_FRAMES)
    throw new Error('Retained frames must be between 1 and 7500');
  if (termination !== 'max-frames' && termination !== 'natural-end') throw new Error('Invalid termination');
  if (promptTokens + retainedFrames > MAX_CONTEXT_TOKENS)
    throw new Error(`Prompt tokens plus frames must not exceed ${MAX_CONTEXT_TOKENS}`);
  const chunkCount = retainedFrames <= CHUNK_FRAMES ? 1 : Math.ceil(retainedFrames / CHUNK_HOP_FRAMES) - 1;
  const chunks = Array.from({ length: chunkCount }, (_, index) => {
    const startFrame = index * CHUNK_HOP_FRAMES;
    const frameLength = Math.min(CHUNK_FRAMES, retainedFrames - startFrame);
    const latentLength = Math.floor(frameLength * 441 / 128);
    const cropLeftLatents = index === 0 ? 0 : 86;
    const cropRightLatents = index === chunkCount - 1 ? 0 : 258;
    return {
      startFrame,
      frameLength,
      latentLength,
      cropLeftLatents,
      cropRightLatents,
      samplesPerChannel: (latentLength - cropLeftLatents - cropRightLatents) * SAMPLES_PER_LATENT,
    };
  });
  const samplesPerChannel = chunks.reduce((total, chunk) => total + chunk.samplesPerChannel, 0);
  const semanticDecisions = retainedFrames + (termination === 'natural-end' ? 2 : 1);
  const rvqCalls = (retainedFrames + 1) * 7;
  const feedbackCalls = retainedFrames + (termination === 'natural-end' ? 1 : 0);
  return {
    retainedFrames,
    termination,
    chunks,
    samplesPerChannel,
    wavBytes: 44 + samplesPerChannel * 4,
    flowCalls: chunkCount * flowSteps,
    vocoderCalls: chunkCount * 2,
    semanticDecisions,
    rvqCalls,
    feedbackCalls,
  };
}

export function planDuration(request: DurationPlanRequest): RetainedFramesPlan {
  validateDuration(request);
  return planRetainedFrames({
    retainedFrames: Math.floor(request.durationSeconds * FRAMES_PER_SECOND),
    promptTokens: request.promptTokens,
    termination: 'max-frames',
    flowSteps: request.flowSteps,
  });
}
