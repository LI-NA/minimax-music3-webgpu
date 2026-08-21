import {
  planDuration,
  planRetainedFrames,
  type DurationChunkPlan,
  type Termination,
} from '../runtime/pipeline/duration-plan';
import type { FixedComparisonMetadata } from '../runtime/reference/fixed-comparison';
import type { CapacityDiagnosticMetadata } from '../runtime/pipeline/rvq-generation';

export const CAPACITY_DIAGNOSTIC_PROMPT_TOKENS = 40;

export type MusicSamplingInput = {
  globalGuidance: number;
  semanticTopK: number;
  residualTopK: number;
  temperature: number;
  flowGuidance: number;
  flowSteps: number;
};

export type MusicGenerationRequestInput = {
  manifestUrl: string;
  prompt: string;
  lyrics: string;
  seed: number;
  durationSeconds: number;
  sampling: MusicSamplingInput;
};
export type MusicGenerationRequest = {
  type: 'generate-music';
} & MusicGenerationRequestInput;
export type ResolvedMusicGenerationRequest = MusicGenerationRequest & {
  promptTokens: number;
  requestedFrames: number;
};

export type MusicCapacityDiagnosticRequest = {
  type: 'diagnose-music-capacity';
  diagnostic: 'continue-after-audio-end';
  manifestUrl: string;
  seed: number;
  durationSeconds: 300;
  promptTokens: 40;
  requestedFrames: 7_500;
};

export function validateMusicCapacityDiagnosticRequest(
  raw: unknown,
): MusicCapacityDiagnosticRequest {
  if (typeof raw !== 'object' || raw === null)
    throw new Error('Music capacity diagnostic request must be an object');
  const request = raw as Record<string, unknown>;
  const keys = [
    'diagnostic',
    'durationSeconds',
    'manifestUrl',
    'promptTokens',
    'requestedFrames',
    'seed',
    'type',
  ];
  if (JSON.stringify(Object.keys(request).sort()) !== JSON.stringify(keys))
    throw new Error('Music capacity diagnostic request fields are invalid');
  if (request.type !== 'diagnose-music-capacity')
    throw new Error('Invalid music capacity diagnostic request type');
  if (request.diagnostic !== 'continue-after-audio-end')
    throw new Error('Invalid music capacity diagnostic policy');
  if (typeof request.manifestUrl !== 'string' || request.manifestUrl.length === 0)
    throw new Error('Music capacity diagnostic manifest URL must be a non-empty string');
  if (
    typeof request.seed !== 'number'
    || !Number.isInteger(request.seed)
    || request.seed < 0
    || request.seed > 4_294_967_295
  ) throw new Error('Music capacity diagnostic seed must be a uint32 integer');
  if (
    request.durationSeconds !== 300
    || request.promptTokens !== CAPACITY_DIAGNOSTIC_PROMPT_TOKENS
    || request.requestedFrames !== 7_500
  ) throw new Error('Music capacity diagnostic must use 300 seconds, 40 prompt tokens, and 7500 frames');
  return request as MusicCapacityDiagnosticRequest;
}

function sameKeys(value: Record<string, unknown>, keys: readonly string[]) {
  return JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...keys].sort());
}

function validateSampling(value: unknown): asserts value is MusicSamplingInput {
  if (typeof value !== 'object' || value === null)
    throw new Error('Music generation sampling fields are invalid');
  const sampling = value as Record<string, unknown>;
  if (!sameKeys(sampling, [
    'flowGuidance',
    'flowSteps',
    'globalGuidance',
    'residualTopK',
    'semanticTopK',
    'temperature',
  ])) throw new Error('Music generation sampling fields are invalid');
  if (
    typeof sampling.globalGuidance !== 'number'
    || !Number.isFinite(sampling.globalGuidance)
    || sampling.globalGuidance < 0
    || typeof sampling.flowGuidance !== 'number'
    || !Number.isFinite(sampling.flowGuidance)
    || sampling.flowGuidance < 0
    || sampling.flowGuidance > 65_504
    || typeof sampling.semanticTopK !== 'number'
    || !Number.isInteger(sampling.semanticTopK)
    || sampling.semanticTopK < 1
    || sampling.semanticTopK > 16_385
    || typeof sampling.residualTopK !== 'number'
    || !Number.isInteger(sampling.residualTopK)
    || sampling.residualTopK < 1
    || sampling.residualTopK > 1_024
    || typeof sampling.temperature !== 'number'
    || !Number.isFinite(sampling.temperature)
    || sampling.temperature <= 0
    || typeof sampling.flowSteps !== 'number'
    || !Number.isSafeInteger(sampling.flowSteps)
    || sampling.flowSteps < 1
  ) throw new Error('Music generation sampling values are invalid');
}

export function createMusicGenerationRequest(input: MusicGenerationRequestInput): MusicGenerationRequest {
  return validateMusicGenerationRequest({ type: 'generate-music', ...input });
}

export function validateMusicGenerationRequest(raw: unknown): MusicGenerationRequest {
  if (typeof raw !== 'object' || raw === null)
    throw new Error('Music generation request must be an object');
  const request = raw as Record<string, unknown>;
  if (!sameKeys(request, [
    'durationSeconds',
    'lyrics',
    'manifestUrl',
    'prompt',
    'sampling',
    'seed',
    'type',
  ])) throw new Error('Music generation request fields are invalid');
  if (request.type !== 'generate-music') throw new Error('Invalid music generation request type');
  if (typeof request.manifestUrl !== 'string' || request.manifestUrl.length === 0)
    throw new Error('Music generation manifest URL must be a non-empty string');
  if (typeof request.prompt !== 'string' || request.prompt.trim().length === 0)
    throw new Error('Music generation prompt must be a non-empty string');
  if (typeof request.lyrics !== 'string' || request.lyrics.trim().length === 0)
    throw new Error('Music generation lyrics must be a non-empty string');
  if (
    typeof request.seed !== 'number'
    || !Number.isInteger(request.seed)
    || request.seed < 0
    || request.seed > 4_294_967_295
  ) throw new Error('Music generation seed must be a uint32 integer');
  planDuration({ durationSeconds: request.durationSeconds as number, promptTokens: 0 });
  validateSampling(request.sampling);
  return request as MusicGenerationRequest;
}

export function createResolvedMusicGenerationRequest(
  request: MusicGenerationRequest,
  promptTokens: number,
): ResolvedMusicGenerationRequest {
  if (!Number.isInteger(promptTokens) || promptTokens < 1 || promptTokens > 5_000)
    throw new Error('Prepared prompt tokens must be between 1 and 5000');
  const plan = planDuration({
    durationSeconds: request.durationSeconds,
    promptTokens,
    flowSteps: request.sampling.flowSteps,
  });
  return { ...request, promptTokens, requestedFrames: plan.retainedFrames };
}

export type MusicGenerationResultPlan = {
  durationSeconds: number;
  requestedFrames: number;
  retainedFrames: number;
  termination: Termination;
  chunkCount: number;
  chunks: readonly DurationChunkPlan[];
  samplesPerChannel: number;
  wavBytes: number;
  flowCalls: number;
  vocoderCalls: number;
  semanticDecisions: number;
  rvqCalls: number;
  feedbackCalls: number;
};

export type MusicGenerationEffectiveInput = {
  prompt: string;
  lyrics: string;
  assembledPrompt: string;
  tokenRows: [number[], number[]];
  promptTokens: number;
  seed: number;
  durationSeconds: number;
  sampling: MusicSamplingInput;
};

export function createMusicGenerationResultPlan(
  request: ResolvedMusicGenerationRequest | MusicCapacityDiagnosticRequest,
  retainedFrames: number,
  termination: Termination,
): MusicGenerationResultPlan {
  const flowSteps = request.type === 'generate-music' ? request.sampling.flowSteps : 30;
  const requested = planDuration({
    durationSeconds: request.durationSeconds,
    promptTokens: request.promptTokens,
    flowSteps,
  });
  if (retainedFrames > requested.retainedFrames)
    throw new Error('Retained frames must not exceed requested frames');
  const retained = planRetainedFrames({
    retainedFrames,
    promptTokens: request.promptTokens,
    termination,
    flowSteps,
  });
  return {
    durationSeconds: request.durationSeconds,
    requestedFrames: requested.retainedFrames,
    retainedFrames: retained.retainedFrames,
    termination: retained.termination,
    chunkCount: retained.chunks.length,
    chunks: retained.chunks,
    samplesPerChannel: retained.samplesPerChannel,
    wavBytes: retained.wavBytes,
    flowCalls: retained.flowCalls,
    vocoderCalls: retained.vocoderCalls,
    semanticDecisions: retained.semanticDecisions,
    rvqCalls: retained.rvqCalls,
    feedbackCalls: retained.feedbackCalls,
  };
}

export type WorkerRequest =
  | { type: 'run-global-smoke'; manifestUrl: string }
  | { type: 'run-rvq-smoke'; manifestUrl: string }
  | { type: 'run-condition-smoke'; manifestUrl: string }
  | { type: 'run-flow-smoke'; manifestUrl: string }
  | { type: 'run-vocoder-smoke'; manifestUrl: string }
  | {
      type: 'generate-frames';
      globalManifestUrl: string;
      rvqManifestUrl: string;
      maxFrames: number;
      seed: number;
    }
  | MusicGenerationRequest
  | MusicCapacityDiagnosticRequest
  | { type: 'generate-music-5s'; manifestUrl: string; seed: number };
export type MusicStage = 'autoregressive' | 'acoustic' | 'condition' | 'flow' | 'vocoder' | 'wav';
export type WorkerProgress = {
  type: 'progress';
  stage: 'manifest' | 'artifact' | 'adapter' | 'session' | MusicStage | 'complete';
  detail: string;
  loaded?: number;
  total?: number;
  name?: MusicStage;
  activity?: 'indeterminate';
  currentFile?: string;
  completedBytes?: number;
  totalBytes?: number;
  cacheHit?: boolean;
  completed?: number;
  elapsedMs?: number;
  rate?: number;
  stepMs?: number;
  etaMs?: number;
  wavBytes?: number;
  totalElapsedMs?: number;
};
export type GlobalSmokeResult = {
  adapter: string;
  sessionCreateMs: number;
  stepMs: readonly number[];
  cacheLengths: readonly number[];
  tensorLocations: readonly string[];
  finiteLogits: boolean;
  ownedTensorBytes: number;
  artifactFetches: number;
  cacheReuseCount: number;
  status: 'passed';
};
export type RvqSmokeResult = {
  adapter: string;
  sessionCreateMs: number;
  lengths: readonly number[];
  stepMs: readonly number[];
  finiteLogits: boolean;
  hiddenLocations: readonly string[];
  feedbackLocation: string;
  artifactFetches: number;
  status: 'passed';
};
export type FrameGenerationResult = {
  adapter: string;
  frames: number;
  attemptedSeeds: readonly number[];
  semanticDecisions: number;
  rvqCalls: number;
  feedbackDecodes: number;
  cacheLengths: readonly number[];
  finiteHiddenGroups: boolean;
  codesInRange: boolean;
  hiddenBytes: number;
  artifactFetches: number;
  status: 'passed';
};
export type ConditionSmokeResult = {
  adapter: string;
  sessionCreateMs: number;
  elapsedMs: number;
  shape: readonly number[];
  outputLocation: string;
  finite: boolean;
  artifactFetches: number;
  status: 'passed';
};
export type FlowSmokeResult = {
  adapter: string;
  sessionCreateMs: number;
  oneStepMs: number;
  generationMs: number;
  stepMs: readonly number[];
  shape: readonly number[];
  oneStepLocation: string;
  finalLocation: string;
  oneStepFinite: boolean;
  finalFinite: boolean;
  artifactFetches: number;
  status: 'passed';
};
export type VocoderSmokeResult = {
  adapter: string;
  sessionCreateMs: number;
  generationMs: number;
  outputType: 'float32';
  shape: readonly [1, 2, 220160];
  finite: true;
  wavBytes: 880684;
  sampleRate: 44100;
  channels: 2;
  samples: 220160;
  bitsPerSample: 16;
  artifactFetches: number;
  status: 'passed';
};
type MusicGenerationWorkerMetrics = {
  wav: ArrayBuffer;
  adapters: readonly string[];
  attemptedSeeds: readonly number[];
  hiddenBytes: number;
  conditionBytes: number;
  latentBytes: number;
  wavBytes: number;
  artifactBytes: number;
  artifactFetches: number;
  manifestHash: string;
  sessionCreateMs: Readonly<Record<'autoregressive' | 'condition' | 'flow' | 'vocoder', number>>;
  stageMs: Readonly<Record<'autoregressive' | 'condition' | 'flow' | 'vocoder', number>>;
  inferenceMs: Readonly<Record<'autoregressive' | 'condition' | 'flow' | 'vocoder', number>>;
  flowStepMs: readonly number[];
  browser: string;
  ortVersion: string;
  status: 'passed';
};

export type MusicGenerationWorkerResult = MusicGenerationWorkerMetrics & {
  effectiveInput: MusicGenerationEffectiveInput;
  plan: MusicGenerationResultPlan;
  comparison?: FixedComparisonMetadata;
  capacityDiagnostic?: never;
};

export type LegacyFiveSecondMusicWorkerResult = MusicGenerationWorkerMetrics & {
  effectiveInput?: never;
  plan?: never;
  comparison?: never;
  capacityDiagnostic?: never;
};

export type MusicCapacityDiagnosticWorkerResult = MusicGenerationWorkerMetrics & {
  effectiveInput?: never;
  plan: MusicGenerationResultPlan;
  comparison?: never;
  capacityDiagnostic: CapacityDiagnosticMetadata;
};
export type AnyMusicGenerationWorkerResult =
  | MusicGenerationWorkerResult
  | LegacyFiveSecondMusicWorkerResult
  | MusicCapacityDiagnosticWorkerResult;
export type WorkerResponse =
  | WorkerProgress
  | { type: 'result'; result: GlobalSmokeResult }
  | { type: 'rvq-result'; result: RvqSmokeResult }
  | { type: 'frame-result'; result: FrameGenerationResult }
  | { type: 'condition-result'; result: ConditionSmokeResult }
  | { type: 'flow-result'; result: FlowSmokeResult }
  | { type: 'vocoder-result'; result: VocoderSmokeResult }
  | {
      type: 'music-result';
      result: AnyMusicGenerationWorkerResult;
    }
  | { type: 'error'; message: string };
