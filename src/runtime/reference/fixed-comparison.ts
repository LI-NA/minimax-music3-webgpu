import promptContract from '../../../tests/fixtures/prompt-contract.json';
import flowContract from '../../../tools/reference/fixed_case.json';
import {
  planRetainedFrames,
  type DurationChunkPlan,
  type Termination,
} from '../pipeline/duration-plan';

export const PINNED_COMPARISON_ORT_VERSION = '1.30.0-dev.20260813-72e1c9c9b8';

export type FixedComparisonPlan = {
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

export type FixedComparisonMetadata = {
  prompt: string;
  lyrics: string;
  assembledPrompt: string;
  tokenIds: [number[], number[]];
  seed: number;
  durationSeconds: number;
  retainedFrames: number;
  termination: Termination;
  globalGuidance: number;
  semanticTopK: number;
  residualTopK: number;
  temperature: number;
  samplerRevision: string;
  flowGuidance: number;
  flowSteps: number;
  flowScheduleRevision: string;
  manifestHash: string;
  browser: string;
  ortVersion: string;
  appVersion: string;
  appRevision: string;
};

export const FIXED_COMPARISON_CASE = {
  input: {
    prompt: flowContract.prompt,
    lyrics: flowContract.lyrics,
    assembledPrompt: flowContract.assembledPrompt,
    tokenRows: [promptContract.conditional, promptContract.unconditional] as [number[], number[]],
  },
  generation: { seed: flowContract.seed, durationSeconds: flowContract.durationSeconds },
  sampler: {
    globalGuidance: flowContract.globalGuidance,
    semanticTopK: flowContract.semanticTopK,
    residualTopK: flowContract.residualTopK,
    temperature: flowContract.temperature,
    samplerRevision: flowContract.samplerRevision,
    flowGuidance: flowContract.flowGuidance,
    flowSteps: flowContract.flowSteps,
    flowScheduleRevision: flowContract.flowScheduleRevision,
  },
} as const;

type FixedComparisonRuntime = {
  input: {
    prompt: string;
    lyrics: string;
    assembledPrompt: string;
    tokenRows: readonly [readonly number[], readonly number[]];
  };
  sampling: {
    globalGuidance: number;
    semanticTopK: number;
    residualTopK: number;
    temperature: number;
    flowGuidance: number;
    flowSteps: number;
  };
  seed: number;
  durationSeconds: number;
  plan: FixedComparisonPlan;
  manifestHash: string;
  browser: string;
  ortVersion: string;
};

const copyRows = (): [number[], number[]] => [
  [...FIXED_COMPARISON_CASE.input.tokenRows[0]],
  [...FIXED_COMPARISON_CASE.input.tokenRows[1]],
];

export function createFixedComparisonMetadata(
  runtime: FixedComparisonRuntime,
): FixedComparisonMetadata | undefined {
  if (
    runtime.seed !== FIXED_COMPARISON_CASE.generation.seed
    || runtime.durationSeconds !== FIXED_COMPARISON_CASE.generation.durationSeconds
    || !sameJson(runtime.input, FIXED_COMPARISON_CASE.input)
    || !Object.entries(runtime.sampling).every(
      ([key, value]) => FIXED_COMPARISON_CASE.sampler[
        key as keyof typeof FIXED_COMPARISON_CASE.sampler
      ] === value,
    )
    || Object.keys(runtime.sampling).length !== 6
  ) return undefined;
  validatePlan(runtime.plan);
  const metadata: FixedComparisonMetadata = {
    prompt: FIXED_COMPARISON_CASE.input.prompt,
    lyrics: FIXED_COMPARISON_CASE.input.lyrics,
    assembledPrompt: FIXED_COMPARISON_CASE.input.assembledPrompt,
    tokenIds: copyRows(),
    seed: runtime.seed,
    durationSeconds: runtime.durationSeconds,
    retainedFrames: runtime.plan.retainedFrames,
    termination: runtime.plan.termination,
    ...FIXED_COMPARISON_CASE.sampler,
    manifestHash: runtime.manifestHash,
    browser: runtime.browser,
    ortVersion: runtime.ortVersion,
    appVersion: __MINIMAX_APP_VERSION__,
    appRevision: __MINIMAX_APP_REVISION__,
  };
  return validateFixedComparisonMetadata(metadata);
}

const sameJson = (left: unknown, right: unknown) => JSON.stringify(left) === JSON.stringify(right);
const chromeUserAgent = /(?:^| )Chrome\/([1-9]\d*\.\d+\.\d+\.\d+)(?: |$)/;

function validatePlan(plan: FixedComparisonPlan) {
  let retained;
  try {
    retained = planRetainedFrames({
      retainedFrames: plan.retainedFrames,
      promptTokens: 40,
      termination: plan.termination,
    });
  } catch {
    throw new Error('fixed comparison result plan is invalid');
  }
  const expected: FixedComparisonPlan = {
    durationSeconds: 10,
    requestedFrames: 250,
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
  if (!sameJson(plan, expected)) throw new Error('fixed comparison result plan does not match');
}

export function validateFixedComparisonMetadata(value: unknown): FixedComparisonMetadata {
  if (typeof value !== 'object' || value === null)
    throw new Error('fixed comparison metadata must be an object');
  const metadata = value as FixedComparisonMetadata;
  const keys = Object.keys(metadata).sort();
  if (
    !sameJson(keys, [...flowContract.comparisonMetricKeys].sort())
    || metadata.prompt !== FIXED_COMPARISON_CASE.input.prompt
    || metadata.lyrics !== FIXED_COMPARISON_CASE.input.lyrics
    || metadata.assembledPrompt !== FIXED_COMPARISON_CASE.input.assembledPrompt
    || !sameJson(metadata.tokenIds, FIXED_COMPARISON_CASE.input.tokenRows)
    || metadata.seed !== FIXED_COMPARISON_CASE.generation.seed
    || metadata.durationSeconds !== FIXED_COMPARISON_CASE.generation.durationSeconds
    || !Object.entries(FIXED_COMPARISON_CASE.sampler)
      .every(([key, expected]) => metadata[key as keyof FixedComparisonMetadata] === expected)
  ) throw new Error('fixed comparison contract does not match');

  const retained = metadata.retainedFrames;
  const termination = metadata.termination;
  if (!Number.isInteger(retained) || (termination !== 'max-frames' && termination !== 'natural-end'))
    throw new Error('fixed comparison result plan is invalid');
  let expectedRetained;
  try {
    expectedRetained = planRetainedFrames({ retainedFrames: retained, promptTokens: 40, termination });
  } catch {
    throw new Error('fixed comparison result plan is invalid');
  }
  if (expectedRetained.termination === 'max-frames' && expectedRetained.retainedFrames !== 250)
    throw new Error('fixed comparison result plan does not match');

  if (typeof metadata.browser !== 'string' || !chromeUserAgent.test(metadata.browser))
    throw new Error('fixed comparison requires a four-part Chrome user agent');
  if (
    metadata.ortVersion !== PINNED_COMPARISON_ORT_VERSION
    || metadata.appVersion !== __MINIMAX_APP_VERSION__
    || metadata.appRevision !== __MINIMAX_APP_REVISION__
    || !/^[0-9a-f]{64}$/.test(metadata.manifestHash)
    || !/^[0-9a-f]{64}$/.test(metadata.appRevision)
  ) throw new Error('fixed comparison provenance does not match');
  return metadata;
}
