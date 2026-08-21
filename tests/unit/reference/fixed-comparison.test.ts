import { describe, expect, it } from 'vitest';
import { planRetainedFrames } from '../../../src/runtime/pipeline/duration-plan';
import fixedCaseFixture from '../../../tools/reference/fixed_case.json';
import {
  createFixedComparisonMetadata,
  FIXED_COMPARISON_CASE,
  validateFixedComparisonMetadata,
} from '../../../src/runtime/reference/fixed-comparison';

const provenance = {
  manifestHash: 'a'.repeat(64),
  browser: 'Mozilla/5.0 Chrome/140.0.7339.81 Safari/537.36',
  ortVersion: '1.30.0-dev.20260813-72e1c9c9b8',
};

const fixedInput = {
  prompt: FIXED_COMPARISON_CASE.input.prompt,
  lyrics: FIXED_COMPARISON_CASE.input.lyrics,
  assembledPrompt: FIXED_COMPARISON_CASE.input.assembledPrompt,
  tokenRows: FIXED_COMPARISON_CASE.input.tokenRows,
};

const fixedSampling = {
  globalGuidance: 1.5,
  semanticTopK: 50,
  residualTopK: 50,
  temperature: 1,
  flowGuidance: 1.7,
  flowSteps: 30,
};

function fixedMetadata() {
  const retained = planRetainedFrames({
    retainedFrames: 250,
    promptTokens: 40,
    termination: 'max-frames',
  });
  return createFixedComparisonMetadata({
    input: fixedInput,
    sampling: fixedSampling,
    seed: 7,
    durationSeconds: 10,
    plan: {
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
    },
    ...provenance,
  })!;
}

describe('fixed WebGPU comparison metadata', () => {
  it('records the exact prompt, token rows, sampler, schedule, runtime, and result plan', () => {
    const metadata = fixedMetadata();

    expect(metadata).toMatchObject({
      prompt: 'Global\nbpm is 96\nWarm female vocal',
      lyrics: '[verse]\nHello\n[chorus]\nStay\nTogether\n[bridge][solo]',
      assembledPrompt: '<|im_start|><|caption_start|>Global\nbpm is 96\nWarm female vocal<|caption_end|><|lyrics_start|>[start]\n[verse]\nHello\n[chorus]\nStay\nTogether\n[bridge][solo]<|lyrics_end|><|im_end|><|audio_start|>',
      seed: 7,
      durationSeconds: 10,
      retainedFrames: 250,
      termination: 'max-frames',
      globalGuidance: 1.5,
      semanticTopK: 50,
      residualTopK: 50,
      temperature: 1,
      samplerRevision: 'mulberry32-guided-threshold-topk-f32-box-muller-fp16-v1',
      flowGuidance: 1.7,
      flowSteps: 30,
      flowScheduleRevision: 'sha256:b1c658b8efb8382eac3a9d4115d1566ad11abd8a91b223c48d73699a0140bd92',
      ...provenance,
      appVersion: '0.1.0-experimental',
      appRevision: expect.stringMatching(/^[0-9a-f]{64}$/),
    });
    expect(metadata.tokenIds).toEqual([
      FIXED_COMPARISON_CASE.input.tokenRows[0],
      FIXED_COMPARISON_CASE.input.tokenRows[1],
    ]);
    expect(metadata.tokenIds[0]).toHaveLength(40);
    expect(metadata.tokenIds[1]).toHaveLength(40);
    expect(Object.keys(metadata).sort()).toEqual(
      [...(fixedCaseFixture as { comparisonMetricKeys: string[] }).comparisonMetricKeys].sort(),
    );
    expect(validateFixedComparisonMetadata(metadata)).toEqual(metadata);
  });

  it.each([
    ['prompt', (value: ReturnType<typeof fixedMetadata>) => { value.prompt = 'other'; }],
    ['lyrics', (value: ReturnType<typeof fixedMetadata>) => { value.lyrics = 'other'; }],
    ['assembled prompt', (value: ReturnType<typeof fixedMetadata>) => { value.assembledPrompt = 'other'; }],
    ['conditional tokens', (value: ReturnType<typeof fixedMetadata>) => { value.tokenIds[0][1] = 0; }],
    ['unconditional tokens', (value: ReturnType<typeof fixedMetadata>) => { value.tokenIds[1][1] = 0; }],
    ['global guidance', (value: ReturnType<typeof fixedMetadata>) => { value.globalGuidance = 1.6; }],
    ['semantic top-k', (value: ReturnType<typeof fixedMetadata>) => { value.semanticTopK = 49; }],
    ['residual top-k', (value: ReturnType<typeof fixedMetadata>) => { value.residualTopK = 49; }],
    ['temperature', (value: ReturnType<typeof fixedMetadata>) => { value.temperature = 0.9; }],
    ['sampler revision', (value: ReturnType<typeof fixedMetadata>) => { value.samplerRevision = 'other'; }],
    ['flow guidance', (value: ReturnType<typeof fixedMetadata>) => { value.flowGuidance = 1.8; }],
    ['flow steps', (value: ReturnType<typeof fixedMetadata>) => { value.flowSteps = 29; }],
    ['schedule', (value: ReturnType<typeof fixedMetadata>) => { value.flowScheduleRevision = 'other'; }],
  ])('rejects changed %s metadata', (_label, mutate) => {
    const metadata = structuredClone(fixedMetadata());
    mutate(metadata);
    expect(() => validateFixedComparisonMetadata(metadata)).toThrow('fixed comparison');
  });

  it('does not label changed input, sampling, seed, or duration as the fixed case', () => {
    const plan = {
      durationSeconds: 10,
      requestedFrames: 250,
      retainedFrames: fixedMetadata().retainedFrames,
      termination: fixedMetadata().termination,
      chunkCount: 2,
      chunks: [
        { startFrame: 0, frameLength: 200, latentLength: 689, cropLeftLatents: 0, cropRightLatents: 258, samplesPerChannel: 220_672 },
        { startFrame: 100, frameLength: 150, latentLength: 516, cropLeftLatents: 86, cropRightLatents: 0, samplesPerChannel: 220_160 },
      ],
      samplesPerChannel: 440_832,
      wavBytes: 1_763_372,
      flowCalls: 60,
      vocoderCalls: 4,
      semanticDecisions: 251,
      rvqCalls: 1_757,
      feedbackCalls: 250,
    } as const;
    expect(createFixedComparisonMetadata({ input: fixedInput, sampling: fixedSampling, seed: 8, durationSeconds: 10, plan, ...provenance }))
      .toBeUndefined();
    expect(createFixedComparisonMetadata({ input: fixedInput, sampling: fixedSampling, seed: 7, durationSeconds: 15, plan, ...provenance }))
      .toBeUndefined();
    expect(createFixedComparisonMetadata({
      input: { ...fixedInput, prompt: 'other' }, sampling: fixedSampling,
      seed: 7, durationSeconds: 10, plan, ...provenance,
    }))
      .toBeUndefined();
    expect(createFixedComparisonMetadata({
      input: fixedInput, sampling: { ...fixedSampling, temperature: 0.9 },
      seed: 7, durationSeconds: 10, plan, ...provenance,
    }))
      .toBeUndefined();
  });

  it('rejects a non-Chrome user agent', () => {
    expect(() => createFixedComparisonMetadata({
      input: fixedInput,
      sampling: fixedSampling,
      seed: 7,
      durationSeconds: 10,
      plan: {
        durationSeconds: 10,
        requestedFrames: 250,
        retainedFrames: 250,
        termination: 'max-frames',
        chunkCount: 2,
        chunks: [
          { startFrame: 0, frameLength: 200, latentLength: 689, cropLeftLatents: 0, cropRightLatents: 258, samplesPerChannel: 220_672 },
          { startFrame: 100, frameLength: 150, latentLength: 516, cropLeftLatents: 86, cropRightLatents: 0, samplesPerChannel: 220_160 },
        ],
        samplesPerChannel: 440_832,
        wavBytes: 1_763_372,
        flowCalls: 60,
        vocoderCalls: 4,
        semanticDecisions: 251,
        rvqCalls: 1_757,
        feedbackCalls: 250,
      },
      ...provenance,
      browser: 'synthetic worker',
    })).toThrow('Chrome');
  });

  it('rejects changed autoregressive plan counts', () => {
    const retained = planRetainedFrames({
      retainedFrames: 250,
      promptTokens: 40,
      termination: 'max-frames',
    });
    expect(() => createFixedComparisonMetadata({
      input: fixedInput,
      sampling: fixedSampling,
      seed: 7,
      durationSeconds: 10,
      plan: {
        durationSeconds: 10,
        requestedFrames: 250,
        retainedFrames: 250,
        termination: 'max-frames',
        chunkCount: 2,
        chunks: retained.chunks,
        samplesPerChannel: 440_832,
        wavBytes: 1_763_372,
        flowCalls: 60,
        vocoderCalls: 4,
        semanticDecisions: retained.semanticDecisions - 1,
        rvqCalls: retained.rvqCalls,
        feedbackCalls: retained.feedbackCalls,
      },
      ...provenance,
    })).toThrow('fixed comparison result plan does not match');
  });
});
