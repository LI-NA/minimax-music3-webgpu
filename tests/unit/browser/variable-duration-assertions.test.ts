import { describe, expect, it } from 'vitest';
import {
  assertAudioHealth,
  assertCancellationBoundary,
  assertGateResult,
  assertCapacityDiagnosticResult,
  assertStableProgressMetrics,
  assertWavIdentifiers,
  gateExpectation,
  parseLongDurationMode,
} from '../../browser/variable-duration-assertions';
import * as gateAssertions from '../../browser/variable-duration-assertions';

describe('variable-duration browser gate assertions', () => {
  it('pins the exact six and ten second maximum-frame plans', () => {
    expect(gateExpectation(6)).toEqual({
      durationSeconds: 6,
      requestedFrames: 150,
      retainedFrames: 150,
      termination: 'max-frames',
      chunkCount: 1,
      samplesPerChannel: 264_192,
      wavBytes: 1_056_812,
      flowCalls: 30,
      vocoderCalls: 2,
      semanticDecisions: 151,
      rvqCalls: 1_057,
      feedbackCalls: 150,
      chunks: [{
        startFrame: 0,
        frameLength: 150,
        latentLength: 516,
        cropLeftLatents: 0,
        cropRightLatents: 0,
        samplesPerChannel: 264_192,
      }],
    });
    expect(gateExpectation(10)).toEqual({
      durationSeconds: 10,
      requestedFrames: 250,
      retainedFrames: 250,
      termination: 'max-frames',
      chunkCount: 2,
      samplesPerChannel: 440_832,
      wavBytes: 1_763_372,
      flowCalls: 60,
      vocoderCalls: 4,
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
    });
  });

  it('pins exact long-duration plans through the five-minute boundary', () => {
    const cases = [
      {
        durationSeconds: 30,
        retainedFrames: 750,
        chunkCount: 7,
        samplesPerChannel: 1_324_032,
        wavBytes: 5_296_172,
        flowCalls: 210,
        vocoderCalls: 14,
        semanticDecisions: 751,
        rvqCalls: 5_257,
        feedbackCalls: 750,
        lastChunk: {
          startFrame: 600,
          frameLength: 150,
          latentLength: 516,
          cropLeftLatents: 86,
          cropRightLatents: 0,
          samplesPerChannel: 220_160,
        },
      },
      {
        durationSeconds: 60,
        retainedFrames: 1_500,
        chunkCount: 14,
        samplesPerChannel: 2_649_088,
        wavBytes: 10_596_396,
        flowCalls: 420,
        vocoderCalls: 28,
        semanticDecisions: 1_501,
        rvqCalls: 10_507,
        feedbackCalls: 1_500,
        lastChunk: {
          startFrame: 1_300,
          frameLength: 200,
          latentLength: 689,
          cropLeftLatents: 86,
          cropRightLatents: 0,
          samplesPerChannel: 308_736,
        },
      },
      {
        durationSeconds: 120,
        retainedFrames: 3_000,
        chunkCount: 29,
        samplesPerChannel: 5_298_688,
        wavBytes: 21_194_796,
        flowCalls: 870,
        vocoderCalls: 58,
        semanticDecisions: 3_001,
        rvqCalls: 21_007,
        feedbackCalls: 3_000,
        lastChunk: {
          startFrame: 2_800,
          frameLength: 200,
          latentLength: 689,
          cropLeftLatents: 86,
          cropRightLatents: 0,
          samplesPerChannel: 308_736,
        },
      },
      {
        durationSeconds: 300,
        retainedFrames: 7_500,
        chunkCount: 74,
        samplesPerChannel: 13_247_488,
        wavBytes: 52_989_996,
        flowCalls: 2_220,
        vocoderCalls: 148,
        semanticDecisions: 7_501,
        rvqCalls: 52_507,
        feedbackCalls: 7_500,
        lastChunk: {
          startFrame: 7_300,
          frameLength: 200,
          latentLength: 689,
          cropLeftLatents: 86,
          cropRightLatents: 0,
          samplesPerChannel: 308_736,
        },
      },
    ] as const;

    for (const expected of cases) {
      const plan = gateExpectation(expected.durationSeconds as never);
      expect(plan).toMatchObject({
        durationSeconds: expected.durationSeconds,
        requestedFrames: expected.retainedFrames,
        retainedFrames: expected.retainedFrames,
        termination: 'max-frames',
        chunkCount: expected.chunkCount,
        samplesPerChannel: expected.samplesPerChannel,
        wavBytes: expected.wavBytes,
        flowCalls: expected.flowCalls,
        vocoderCalls: expected.vocoderCalls,
        semanticDecisions: expected.semanticDecisions,
        rvqCalls: expected.rvqCalls,
        feedbackCalls: expected.feedbackCalls,
      });
      expect(plan.chunks).toHaveLength(expected.chunkCount);
      expect(plan.chunks[0]).toEqual({
        startFrame: 0,
        frameLength: 200,
        latentLength: 689,
        cropLeftLatents: 0,
        cropRightLatents: 258,
        samplesPerChannel: 220_672,
      });
      expect(plan.chunks.at(-1)).toEqual(expected.lastChunk);
    }
  });

  it('strictly parses the four staged long-duration gates', () => {
    const parseLongGateDuration = (gateAssertions as unknown as {
      parseLongGateDuration?: (value: string | undefined) => number;
    }).parseLongGateDuration;
    expect(typeof parseLongGateDuration).toBe('function');
    expect(['30', '60', '120', '300'].map((value) => parseLongGateDuration?.(value)))
      .toEqual([30, 60, 120, 300]);
    expect(() => parseLongGateDuration?.(undefined)).toThrow('required');
    expect(() => parseLongGateDuration?.('10')).toThrow('30, 60, 120, or 300');
  });

  it('allows the capacity diagnostic only for the five-minute gate', () => {
    expect(parseLongDurationMode('product', 120)).toBe('product');
    expect(parseLongDurationMode('capacity-diagnostic', 300)).toBe('capacity-diagnostic');
    expect(() => parseLongDurationMode(undefined, 300)).toThrow('required');
    expect(() => parseLongDurationMode('capacity-diagnostic', 120))
      .toThrow('300-second');
    expect(() => parseLongDurationMode('other', 300)).toThrow('allowlist');

    expect(() => assertCapacityDiagnosticResult({
      capacityDiagnostic: {
        kind: 'continue-after-audio-end',
        suppressedAudioEnds: 2,
        firstAudioEndAtRetainedFrame: 1_743,
      },
    })).not.toThrow();
    expect(() => assertCapacityDiagnosticResult({
      capacityDiagnostic: {
        kind: 'continue-after-audio-end',
        suppressedAudioEnds: 0,
        firstAudioEndAtRetainedFrame: null,
      },
    })).toThrow('at least one');
    expect(() => assertCapacityDiagnosticResult({
      capacityDiagnostic: {
        kind: 'continue-after-audio-end',
        suppressedAudioEnds: 1,
        firstAudioEndAtRetainedFrame: 1_742,
      },
    })).toThrow('1743');
    expect(() => assertCapacityDiagnosticResult({
      comparison: {},
      capacityDiagnostic: {
        kind: 'continue-after-audio-end',
        suppressedAudioEnds: 1,
        firstAudioEndAtRetainedFrame: 1_743,
      },
    })).toThrow('comparison');
  });

  it('returns compact long-run progress evidence at practical boundaries', () => {
    const assertLongDurationProgress = (gateAssertions as unknown as {
      assertLongDurationProgress?: (
        progress: Record<string, unknown>[],
        durationSeconds: number,
      ) => Record<string, unknown>;
    }).assertLongDurationProgress;
    expect(typeof assertLongDurationProgress).toBe('function');
    const expected = gateExpectation(30);
    const sessions = ['autoregressive', 'condition', 'flow', 'vocoder'].map((name) => ({
      stage: 'session',
      name,
      activity: 'indeterminate',
    }));
    const measured = (stage: 'autoregressive' | 'flow', completed: number, total: number) => ({
      stage,
      completed,
      total,
      elapsedMs: completed * 10,
      ...(completed >= 3
        ? stage === 'autoregressive'
          ? { rate: 2, etaMs: Math.max(0, total - completed) * 500 }
          : { stepMs: 20, etaMs: Math.max(0, total - completed) * 20 }
        : {}),
    });
    const progress = [
      ...sessions,
      measured('autoregressive', 1, expected.retainedFrames),
      measured('autoregressive', 2, expected.retainedFrames),
      measured('autoregressive', 3, expected.retainedFrames),
      measured('autoregressive', 250, expected.retainedFrames),
      measured('autoregressive', 500, expected.retainedFrames),
      measured('autoregressive', 750, expected.retainedFrames),
      ...Array.from({ length: expected.chunkCount }, () => ({ stage: 'condition' })),
      measured('flow', 1, expected.flowCalls),
      measured('flow', 2, expected.flowCalls),
      measured('flow', 3, expected.flowCalls),
      ...Array.from({ length: expected.chunkCount }, (_, index) =>
        measured('flow', (index + 1) * 30, expected.flowCalls)),
      ...Array.from({ length: expected.chunkCount }, (_, index) => ({
        stage: 'acoustic', completed: index + 1, total: expected.chunkCount,
      })),
      ...Array.from({ length: expected.vocoderCalls }, (_, index) => ({
        stage: 'vocoder', completed: index + 1, total: expected.vocoderCalls,
      })),
      { stage: 'wav' },
      { stage: 'complete' },
    ];

    expect(assertLongDurationProgress?.(progress, 30)).toMatchObject({
      durationSeconds: 30,
      eventCount: progress.length,
      stages: {
        autoregressive: {
          firstCompleted: 1,
          lastCompleted: 750,
          total: 750,
          samples: [1, 2, 3, 250, 500, 750],
          metricSamples: expect.arrayContaining([
            expect.objectContaining({ completed: 3, total: 750, rate: 2, etaMs: 373_500 }),
            expect.objectContaining({ completed: 750, total: 750, rate: 2, etaMs: 0 }),
          ]),
        },
        flow: {
          firstCompleted: 1,
          lastCompleted: 210,
          total: 210,
          samples: [1, 2, 3, 30, 60, 90, 120, 150, 180, 210],
          metricSamples: expect.arrayContaining([
            expect.objectContaining({ completed: 3, total: 210, stepMs: 20, etaMs: 4_140 }),
            expect.objectContaining({ completed: 210, total: 210, stepMs: 20, etaMs: 0 }),
          ]),
        },
      },
      conditionEvents: 7,
      sessionNames: ['autoregressive', 'condition', 'flow', 'vocoder'],
      terminalStages: ['wav', 'complete'],
    });
  });

  it('rejects incomplete or incorrectly totalled long-run progress', () => {
    const assertLongDurationProgress = (gateAssertions as unknown as {
      assertLongDurationProgress?: (
        progress: Record<string, unknown>[],
        durationSeconds: number,
      ) => unknown;
    }).assertLongDurationProgress;
    expect(typeof assertLongDurationProgress).toBe('function');
    const progress = [
      { stage: 'autoregressive', completed: 1, total: 749, elapsedMs: 1 },
      { stage: 'autoregressive', completed: 750, total: 749, elapsedMs: 2, rate: 1, etaMs: 0 },
    ];
    expect(() => assertLongDurationProgress?.(progress, 30)).toThrow('total');
  });

  it('summarizes natural-end progress without treating it as max-frame qualification', () => {
    const summarizeObservedProgress = (gateAssertions as unknown as {
      summarizeObservedProgress?: (
        progress: Record<string, unknown>[],
      ) => Record<string, unknown>;
    }).summarizeObservedProgress;
    expect(typeof summarizeObservedProgress).toBe('function');
    const progress = [
      { stage: 'session', name: 'autoregressive', activity: 'indeterminate' },
      { stage: 'autoregressive', completed: 1, total: 3_000, elapsedMs: 10 },
      { stage: 'autoregressive', completed: 1_500, total: 3_000, elapsedMs: 15_000, rate: 100, etaMs: 15_000 },
      { stage: 'autoregressive', completed: 1_743, total: 3_000, elapsedMs: 17_430, rate: 100, etaMs: 12_570 },
      { stage: 'session', name: 'condition', activity: 'indeterminate' },
      { stage: 'session', name: 'flow', activity: 'indeterminate' },
      { stage: 'flow', completed: 1, total: 510, elapsedMs: 20 },
      { stage: 'flow', completed: 510, total: 510, elapsedMs: 10_200, stepMs: 20, etaMs: 0 },
      { stage: 'acoustic', completed: 17, total: 17 },
      { stage: 'session', name: 'vocoder', activity: 'indeterminate' },
      { stage: 'vocoder', completed: 34, total: 34 },
      { stage: 'wav' },
      { stage: 'complete' },
    ];

    expect(summarizeObservedProgress?.(progress)).toEqual({
      eventCount: progress.length,
      sessionNames: ['autoregressive', 'condition', 'flow', 'vocoder'],
      terminalStages: ['wav', 'complete'],
      stages: {
        autoregressive: {
          eventCount: 3,
          firstCompleted: 1,
          lastCompleted: 1_743,
          reportedTotals: [3_000],
          samples: [
            { completed: 1, total: 3_000, elapsedMs: 10 },
            { completed: 1_500, total: 3_000, elapsedMs: 15_000, rate: 100, etaMs: 15_000 },
            { completed: 1_743, total: 3_000, elapsedMs: 17_430, rate: 100, etaMs: 12_570 },
          ],
        },
        flow: {
          eventCount: 2,
          firstCompleted: 1,
          lastCompleted: 510,
          reportedTotals: [510],
          samples: [
            { completed: 1, total: 510, elapsedMs: 20 },
            { completed: 510, total: 510, elapsedMs: 10_200, stepMs: 20, etaMs: 0 },
          ],
        },
        acoustic: {
          eventCount: 1,
          firstCompleted: 17,
          lastCompleted: 17,
          reportedTotals: [17],
          samples: [{ completed: 17, total: 17 }],
        },
        vocoder: {
          eventCount: 1,
          firstCompleted: 34,
          lastCompleted: 34,
          reportedTotals: [34],
          samples: [{ completed: 34, total: 34 }],
        },
      },
    });
  });

  it('accepts the exact 1743-frame natural end as a product outcome only', () => {
    const naturalEndExpectation = (gateAssertions as unknown as {
      naturalEnd1743Expectation?: () => Record<string, unknown>;
    }).naturalEnd1743Expectation;
    const assertProductOutcome = (gateAssertions as unknown as {
      assertProductOutcome?: (input: Record<string, unknown>) => Record<string, unknown>;
    }).assertProductOutcome;
    expect(typeof naturalEndExpectation).toBe('function');
    expect(typeof assertProductOutcome).toBe('function');
    const plan = naturalEndExpectation?.();
    expect(plan).toMatchObject({
      durationSeconds: 120,
      requestedFrames: 3_000,
      retainedFrames: 1_743,
      termination: 'natural-end',
      chunkCount: 17,
      samplesPerChannel: 3_078_144,
      wavBytes: 12_312_620,
      flowCalls: 510,
      vocoderCalls: 34,
      semanticDecisions: 1_745,
      rvqCalls: 12_208,
      feedbackCalls: 1_744,
    });
    expect((plan?.chunks as unknown[])).toHaveLength(17);
    expect((plan?.chunks as Record<string, unknown>[]).at(-1)).toEqual({
      startFrame: 1_600,
      frameLength: 143,
      latentLength: 492,
      cropLeftLatents: 86,
      cropRightLatents: 0,
      samplesPerChannel: 207_872,
    });

    const sessions = ['autoregressive', 'condition', 'flow', 'vocoder'].map((name) => ({
      stage: 'session', name, activity: 'indeterminate',
    }));
    const measured = (
      stage: 'autoregressive' | 'flow',
      completed: number,
      total: number,
    ) => ({
      stage,
      completed,
      total,
      elapsedMs: completed * 10,
      ...(completed >= 3
        ? stage === 'autoregressive'
          ? { rate: 2, etaMs: Math.max(0, total - completed) * 500 }
          : { stepMs: 20, etaMs: Math.max(0, total - completed) * 20 }
        : {}),
    });
    const progress = [
      ...sessions,
      measured('autoregressive', 1, 3_000),
      measured('autoregressive', 2, 3_000),
      measured('autoregressive', 3, 3_000),
      measured('autoregressive', 1_743, 3_000),
      ...Array.from({ length: 17 }, () => ({ stage: 'condition' })),
      measured('flow', 1, 510),
      measured('flow', 2, 510),
      measured('flow', 3, 510),
      measured('flow', 510, 510),
      { stage: 'acoustic', completed: 1, total: 17 },
      { stage: 'acoustic', completed: 17, total: 17 },
      { stage: 'vocoder', completed: 1, total: 34 },
      { stage: 'vocoder', completed: 34, total: 34 },
      { stage: 'wav' },
      { stage: 'complete' },
    ];
    const audio = {
      riff: 'RIFF', wave: 'WAVE', format: 1, sampleRate: 44_100, channels: 2,
      bitsPerSample: 16, riffSize: 12_312_612, dataBytes: 12_312_576,
      byteRate: 176_400, blockAlign: 4, samplesPerChannel: 3_078_144,
      wavBytes: 12_312_620, decodedSampleRate: 44_100, decodedChannels: 2,
      decodedSamples: 3_078_144, finite: true, stereoDiffers: true,
      longestConstantFrameRun: 10, finalSecondDelta: 1,
    };
    expect(assertProductOutcome?.({ plan, audio, progress })).toMatchObject({
      status: 'passed',
      termination: 'natural-end',
      retainedFrames: 1_743,
      wavBytes: 12_312_620,
    });
    expect(() => assertGateResult(plan ?? {}, 120)).toThrow();
  });

  it('builds observed evidence without claiming qualification passed', () => {
    const createObservedRunEvidence = (gateAssertions as unknown as {
      createObservedRunEvidence?: (input: Record<string, unknown>) => Record<string, unknown>;
    }).createObservedRunEvidence;
    expect(typeof createObservedRunEvidence).toBe('function');
    const result = {
      plan: { retainedFrames: 1_743, termination: 'natural-end' },
      status: 'passed',
    };
    const evidence = createObservedRunEvidence?.({
      durationSeconds: 120,
      seed: 7,
      promptTokens: 40,
      result,
      audio: { wavBytes: 12_345 },
      progress: [{ stage: 'autoregressive', completed: 1_743, total: 3_000 }],
      productOutcome: {
        status: 'passed',
        termination: 'natural-end',
        retainedFrames: 1_743,
        wavBytes: 12_312_620,
      },
    });
    expect(evidence).toMatchObject({
      schemaVersion: 1,
      qualification: {
        status: 'failed',
        requiredTermination: 'max-frames',
        observedTermination: 'natural-end',
      },
      productOutcome: {
        status: 'passed',
        termination: 'natural-end',
        retainedFrames: 1_743,
      },
      request: { durationSeconds: 120, seed: 7, promptTokens: 40 },
      result,
      audio: { wavBytes: 12_345 },
      progress: {
        stages: {
          autoregressive: {
            lastCompleted: 1_743,
            reportedTotals: [3_000],
          },
        },
      },
    });
    expect(evidence?.qualification).not.toMatchObject({ status: 'passed' });
  });

  it('rejects a natural end or mismatched output byte count', () => {
    const expected = gateExpectation(10);
    expect(() => assertGateResult({ ...expected, termination: 'natural-end' }, 10))
      .toThrow('termination');
    expect(() => assertGateResult({ ...expected, wavBytes: expected.wavBytes - 4 }, 10))
      .toThrow('wavBytes');
  });

  it('requires canonical decoded stereo audio with a varying tail', () => {
    expect(() => assertAudioHealth({
      riff: 'RIFF',
      wave: 'WAVE',
      format: 1,
      sampleRate: 44_100,
      channels: 2,
      bitsPerSample: 16,
      riffSize: 1_763_364,
      dataBytes: 1_763_328,
      byteRate: 176_400,
      blockAlign: 4,
      samplesPerChannel: 440_832,
      wavBytes: 1_763_372,
      decodedSampleRate: 44_100,
      decodedChannels: 2,
      decodedSamples: 440_832,
      finite: true,
      stereoDiffers: true,
      longestConstantFrameRun: 12,
      finalSecondDelta: 1,
    }, 10)).not.toThrow();

    expect(() => assertAudioHealth({
      riff: 'RIFF',
      wave: 'WAVE',
      format: 1,
      sampleRate: 44_100,
      channels: 2,
      bitsPerSample: 16,
      riffSize: 1_763_364,
      dataBytes: 1_763_328,
      byteRate: 176_400,
      blockAlign: 4,
      samplesPerChannel: 440_832,
      wavBytes: 1_763_372,
      decodedSampleRate: 44_100,
      decodedChannels: 2,
      decodedSamples: 440_832,
      finite: true,
      stereoDiffers: true,
      longestConstantFrameRun: 44_100,
      finalSecondDelta: 0,
    }, 10)).toThrow('constant');
  });

  it('requires stable AR and flow speed plus ETA metrics', () => {
    const ar = [1, 2, 3].map((completed) => ({
      stage: 'autoregressive',
      completed,
      elapsedMs: completed * 10,
      ...(completed >= 3 ? { rate: 2, etaMs: 100 } : {}),
    }));
    const flow = [1, 2, 3].map((completed) => ({
      stage: 'flow',
      completed,
      elapsedMs: completed * 10,
      ...(completed >= 3 ? { stepMs: 20, etaMs: 100 } : {}),
    }));
    expect(() => assertStableProgressMetrics(ar, 'autoregressive')).not.toThrow();
    expect(() => assertStableProgressMetrics(flow, 'flow')).not.toThrow();
    expect(() => assertStableProgressMetrics(ar.map((event) => ({ ...event, rate: undefined })), 'autoregressive'))
      .toThrow('rate');
    expect(() => assertStableProgressMetrics(flow.map((event) => ({ ...event, etaMs: undefined })), 'flow'))
      .toThrow('ETA');
  });

  it('rejects terminal work after either cancellation boundary', () => {
    expect(() => assertCancellationBoundary([
      { stage: 'autoregressive', completed: 100 },
      { stage: 'wav' },
    ], 'late-ar')).toThrow('wav');
    expect(() => assertCancellationBoundary([
      { stage: 'flow', completed: 31 },
      { stage: 'complete' },
    ], 'second-flow')).toThrow('complete');
  });

  it('requires direct canonical WAV chunk identifiers', () => {
    expect(() => assertWavIdentifiers({
      fmt: 'fmt ',
      fmtChunkSize: 16,
      data: 'data',
    })).not.toThrow();
    expect(() => assertWavIdentifiers({
      fmt: 'JUNK',
      fmtChunkSize: 16,
      data: 'data',
    })).toThrow('fmt');
  });
});
