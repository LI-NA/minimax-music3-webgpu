import { describe, expect, expectTypeOf, it } from 'vitest';
import {
  createMusicGenerationRequest,
  createResolvedMusicGenerationRequest,
  createMusicGenerationResultPlan,
  validateArtifactCacheRequest,
  type ArtifactCacheRequest,
  type ArtifactCacheStatus,
  type ArtifactErrorCode,
  type ArtifactOperation,
  validateMusicCapacityDiagnosticRequest,
  validateMusicGenerationRequest,
  type LegacyFiveSecondMusicWorkerResult,
  type MusicGenerationEffectiveInput,
  type MusicGenerationResultPlan,
  type MusicGenerationWorkerResult,
  type WorkerResponse,
} from '../../../src/workers/protocol';

describe('artifact cache worker protocol', () => {
  it('defines cache operations, status, success responses, and structured errors', () => {
    expectTypeOf<ArtifactCacheRequest>().toEqualTypeOf<
      | { type: 'inspect-artifact-cache'; manifestUrl: string }
      | { type: 'download-artifacts'; manifestUrl: string }
      | { type: 'delete-artifact-caches'; manifestUrl: string }
    >();
    expectTypeOf<ArtifactOperation>().toEqualTypeOf<
      'inspect-artifact-cache' | 'download-artifacts' | 'delete-artifact-caches' | 'generate-music'
    >();
    expectTypeOf<ArtifactErrorCode>().toEqualTypeOf<
      | 'manifest-unavailable'
      | 'manifest-invalid'
      | 'storage-estimate-unavailable'
      | 'quota-insufficient'
      | 'cache-not-ready'
      | 'download-failed'
      | 'quota-exceeded'
      | 'cache-inspection-failed'
      | 'cache-delete-failed'
    >();

    const status: ArtifactCacheStatus = {
      manifestHash: 'abc',
      state: 'partial',
      artifactCount: 2,
      totalArtifactBytes: 100,
      completeArtifactCount: 1,
      completeArtifactBytes: 40,
      storedReferencedBytes: 50,
      additionalBytesNeeded: 50,
      largestPendingArtifactBytes: 60,
      projectCacheCount: 2,
      projectCacheBytes: 75,
      persistence: 'best-effort',
      requiredHeadroomBytes: 110,
    };
    const responses: WorkerResponse[] = [
      { type: 'artifact-cache-status', status },
      { type: 'artifact-download-complete', status },
      { type: 'artifact-cache-deleted', status },
      {
        type: 'error',
        message: 'cache is incomplete',
        code: 'cache-not-ready',
        operation: 'generate-music',
        retryable: true,
      },
    ];

    expect(responses).toHaveLength(4);
  });

  it.each(['inspect-artifact-cache', 'download-artifacts', 'delete-artifact-caches'] as const)(
    'accepts an exact %s request',
    (type) => {
      const request = { type, manifestUrl: '/music/manifest.json' };

      expect(validateArtifactCacheRequest(request)).toEqual(request);
    },
  );

  it.each([
    ['missing manifest URL', { type: 'inspect-artifact-cache' }],
    ['extra field', { type: 'download-artifacts', manifestUrl: '/music/manifest.json', extra: true }],
    ['empty manifest URL', { type: 'delete-artifact-caches', manifestUrl: '' }],
    [
      'decorated array',
      Object.assign([], {
        type: 'inspect-artifact-cache',
        manifestUrl: '/music/manifest.json',
      }),
    ],
  ])('rejects a request with an %s', (_label, request) => {
    expect(() => validateArtifactCacheRequest(request)).toThrow();
  });
});

describe('music generation worker protocol', () => {
  it('requires resolved product metadata without imposing it on the legacy result', () => {
    expectTypeOf<MusicGenerationWorkerResult>().toMatchTypeOf<{
      effectiveInput: MusicGenerationEffectiveInput;
      plan: MusicGenerationResultPlan;
    }>();
    expectTypeOf<LegacyFiveSecondMusicWorkerResult>().toMatchTypeOf<{
      effectiveInput?: never;
      plan?: never;
    }>();
  });

  const input = {
    manifestUrl: '/music/manifest.json',
    prompt: 'Warm female vocal',
    lyrics: '[verse]\nHello',
    seed: 7,
    durationSeconds: 10,
    sampling: {
      globalGuidance: 1.5,
      semanticTopK: 50,
      residualTopK: 50,
      temperature: 1,
      flowGuidance: 1.7,
      flowSteps: 30,
    },
  } as const;

  it('accepts exactly the complete raw product input', () => {
    expect(createMusicGenerationRequest(input)).toEqual({
      type: 'generate-music',
      ...input,
    });
    expect(validateMusicGenerationRequest({ type: 'generate-music', ...input })).toEqual({
      type: 'generate-music',
      ...input,
    });
  });

  it.each([
    ['unknown top-level field', { type: 'generate-music', ...input, promptTokens: 40 }],
    [
      'unknown sampling field',
      {
        type: 'generate-music',
        ...input,
        sampling: { ...input.sampling, topK: 50 },
      },
    ],
    [
      'missing sampling field',
      {
        type: 'generate-music',
        ...input,
        sampling: { ...input.sampling, flowSteps: undefined },
      },
    ],
  ])('rejects a product request with an %s', (_label, raw) => {
    expect(() => validateMusicGenerationRequest(raw)).toThrow('invalid');
  });

  it.each([
    ['empty prompt', { ...input, prompt: '' }],
    ['whitespace prompt', { ...input, prompt: '   ' }],
    ['empty lyrics', { ...input, lyrics: '' }],
    ['whitespace lyrics', { ...input, lyrics: '\n\t' }],
    ['negative seed', { ...input, seed: -1 }],
    ['oversized seed', { ...input, seed: 4_294_967_296 }],
    ['zero duration', { ...input, durationSeconds: 0 }],
    ['duration over five minutes', { ...input, durationSeconds: 300.01 }],
    ['negative global guidance', { ...input, sampling: { ...input.sampling, globalGuidance: -0.1 } }],
    ['non-finite flow guidance', { ...input, sampling: { ...input.sampling, flowGuidance: Infinity } }],
    ['out-of-range semantic top-k', { ...input, sampling: { ...input.sampling, semanticTopK: 16_386 } }],
    ['out-of-range residual top-k', { ...input, sampling: { ...input.sampling, residualTopK: 1_025 } }],
    ['non-positive temperature', { ...input, sampling: { ...input.sampling, temperature: 0 } }],
    ['unsafe flow steps', { ...input, sampling: { ...input.sampling, flowSteps: Number.MAX_SAFE_INTEGER + 1 } }],
  ])('rejects an invalid product %s', (_label, invalid) => {
    expect(() => createMusicGenerationRequest(invalid)).toThrow();
  });

  it('derives frame totals only after prepared prompt tokens are supplied internally', () => {
    const request = createResolvedMusicGenerationRequest(createMusicGenerationRequest(input), 40);

    expect(request).toMatchObject({
      type: 'generate-music',
      promptTokens: 40,
      requestedFrames: 250,
    });
    expect(createMusicGenerationResultPlan(request, 201, 'natural-end')).toMatchObject({
      durationSeconds: 10,
      requestedFrames: 250,
      retainedFrames: 201,
      termination: 'natural-end',
    });
  });

  it.each([
    [5.04, 126],
    [6, 150],
    [10.5, 262],
  ])('accepts %s seconds and derives %i frames', (durationSeconds, requestedFrames) => {
    const request = createResolvedMusicGenerationRequest(
      createMusicGenerationRequest({ ...input, durationSeconds }),
      40,
    );

    expect(request.requestedFrames).toBe(requestedFrames);
  });

  it('derives result totals from the actual retained frames', () => {
    const request = createResolvedMusicGenerationRequest(createMusicGenerationRequest(input), 40);

    expect(createMusicGenerationResultPlan(request, 201, 'natural-end')).toEqual({
      durationSeconds: 10,
      requestedFrames: 250,
      retainedFrames: 201,
      termination: 'natural-end',
      chunkCount: 2,
      chunks: [
        {
          startFrame: 0,
          frameLength: 200,
          latentLength: 689,
          cropLeftLatents: 0,
          cropRightLatents: 258,
          samplesPerChannel: 220_672,
        },
        {
          startFrame: 100,
          frameLength: 101,
          latentLength: 347,
          cropLeftLatents: 86,
          cropRightLatents: 0,
          samplesPerChannel: 133_632,
        },
      ],
      samplesPerChannel: 354_304,
      wavBytes: 1_417_260,
      flowCalls: 60,
      vocoderCalls: 4,
      semanticDecisions: 203,
      rvqCalls: 1_414,
      feedbackCalls: 202,
    });
    expect(() => createMusicGenerationResultPlan(request, 251, 'natural-end')).toThrow('requested frames');
  });

  it('derives flow totals from the requested flow step count', () => {
    const request = createResolvedMusicGenerationRequest(
      createMusicGenerationRequest({
        ...input,
        sampling: { ...input.sampling, flowSteps: 12 },
      }),
      40,
    );

    expect(createMusicGenerationResultPlan(request, 201, 'natural-end').flowCalls).toBe(24);
  });

  it.each([
    {
      durationSeconds: 10,
      expected: {
        semanticDecisions: 251,
        rvqCalls: 1_757,
        feedbackCalls: 250,
        chunks: [
          {
            startFrame: 0,
            frameLength: 200,
            latentLength: 689,
            cropLeftLatents: 0,
            cropRightLatents: 258,
            samplesPerChannel: 220_672,
          },
          {
            startFrame: 100,
            frameLength: 150,
            latentLength: 516,
            cropLeftLatents: 86,
            cropRightLatents: 0,
            samplesPerChannel: 220_160,
          },
        ],
      },
    },
  ])('reports exact max-frame AR counts for $durationSeconds seconds', ({ durationSeconds, expected }) => {
    const request = createResolvedMusicGenerationRequest(
      createMusicGenerationRequest({
        ...input,
        durationSeconds,
      }),
      40,
    );

    expect(createMusicGenerationResultPlan(request, request.requestedFrames, 'max-frames')).toMatchObject(expected);
  });

  it('accepts only the explicit five-minute capacity diagnostic contract', () => {
    const request = {
      type: 'diagnose-music-capacity',
      diagnostic: 'continue-after-audio-end',
      manifestUrl: '/music/manifest.json',
      seed: 7,
      durationSeconds: 300,
      promptTokens: 40,
      requestedFrames: 7_500,
    } as const;

    expect(validateMusicCapacityDiagnosticRequest(request)).toEqual(request);
    for (const invalid of [
      { ...request, type: 'generate-music' },
      { ...request, diagnostic: 'natural' },
      { ...request, durationSeconds: 120 },
      { ...request, promptTokens: 39 },
      { ...request, requestedFrames: 7_499 },
      { ...request, seed: -1 },
      { ...request, seed: 4_294_967_296 },
    ])
      expect(() => validateMusicCapacityDiagnosticRequest(invalid)).toThrow();
  });
});
