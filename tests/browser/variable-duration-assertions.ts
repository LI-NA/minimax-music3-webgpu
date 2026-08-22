export type GateDuration = 6 | 10 | 30 | 60 | 120 | 300;

export type GateChunk = {
  startFrame: number;
  frameLength: number;
  latentLength: number;
  cropLeftLatents: number;
  cropRightLatents: number;
  samplesPerChannel: number;
};

export type GatePlan = {
  durationSeconds: GateDuration;
  requestedFrames: number;
  retainedFrames: number;
  termination: 'max-frames';
  chunkCount: number;
  samplesPerChannel: number;
  wavBytes: number;
  flowCalls: number;
  vocoderCalls: number;
  semanticDecisions: number;
  rvqCalls: number;
  feedbackCalls: number;
  chunks: GateChunk[];
};

export type ProductPlan = Omit<GatePlan, 'termination'> & {
  termination: 'max-frames' | 'natural-end';
};

export type AudioHealth = {
  riff: string;
  wave: string;
  format: number;
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  riffSize: number;
  dataBytes: number;
  byteRate: number;
  blockAlign: number;
  samplesPerChannel: number;
  wavBytes: number;
  decodedSampleRate: number;
  decodedChannels: number;
  decodedSamples: number;
  finite: boolean;
  stereoDiffers: boolean;
  longestConstantFrameRun: number;
  finalSecondDelta: number;
};

export type ProgressMetric = {
  stage: string;
  completed?: number;
  total?: number;
  elapsedMs?: number;
  rate?: number;
  stepMs?: number;
  etaMs?: number;
  name?: string;
  activity?: string;
};

export type LongGateDuration = 30 | 60 | 120 | 300;
export type LongDurationMode = 'product' | 'capacity-diagnostic';

const expectations: Record<GateDuration, GatePlan> = {
  6: {
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
    chunks: [
      {
        startFrame: 0,
        frameLength: 150,
        latentLength: 516,
        cropLeftLatents: 0,
        cropRightLatents: 0,
        samplesPerChannel: 264_192,
      },
    ],
  },
  10: {
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
  },
  ...(Object.fromEntries(
    [30, 60, 120, 300].map((durationSeconds) => {
      const retainedFrames = durationSeconds * 25;
      const chunkCount = Math.ceil(retainedFrames / 100) - 1;
      const chunks = Array.from({ length: chunkCount }, (_, index) => {
        const startFrame = index * 100;
        const frameLength = Math.min(200, retainedFrames - startFrame);
        const latentLength = Math.floor((frameLength * 441) / 128);
        const cropLeftLatents = index === 0 ? 0 : 86;
        const cropRightLatents = index === chunkCount - 1 ? 0 : 258;
        return {
          startFrame,
          frameLength,
          latentLength,
          cropLeftLatents,
          cropRightLatents,
          samplesPerChannel: (latentLength - cropLeftLatents - cropRightLatents) * 512,
        };
      });
      const samplesPerChannel = chunks.reduce((total, chunk) => total + chunk.samplesPerChannel, 0);
      return [
        durationSeconds,
        {
          durationSeconds,
          requestedFrames: retainedFrames,
          retainedFrames,
          termination: 'max-frames' as const,
          chunkCount,
          samplesPerChannel,
          wavBytes: 44 + samplesPerChannel * 4,
          flowCalls: chunkCount * 30,
          vocoderCalls: chunkCount * 2,
          semanticDecisions: retainedFrames + 1,
          rvqCalls: (retainedFrames + 1) * 7,
          feedbackCalls: retainedFrames,
          chunks,
        },
      ];
    }),
  ) as Record<30 | 60 | 120 | 300, GatePlan>),
};

export function gateExpectation(durationSeconds: GateDuration): GatePlan {
  return { ...expectations[durationSeconds] };
}

export function parseLongGateDuration(value: string | undefined): LongGateDuration {
  if (value === undefined) throw new Error('MINIMAX_LONG_DURATION_SECONDS is required');
  const duration = Number(value);
  if (![30, 60, 120, 300].includes(duration))
    throw new Error('MINIMAX_LONG_DURATION_SECONDS must be 30, 60, 120, or 300');
  return duration as LongGateDuration;
}

export function parseLongDurationMode(value: string | undefined, durationSeconds: LongGateDuration): LongDurationMode {
  if (value === undefined) throw new Error('MINIMAX_LONG_DURATION_MODE is required');
  if (value === 'product') return value;
  if (value !== 'capacity-diagnostic') throw new Error('MINIMAX_LONG_DURATION_MODE must use the mode allowlist');
  if (durationSeconds !== 300) throw new Error('capacity-diagnostic mode is restricted to the 300-second gate');
  return value;
}

export function assertCapacityDiagnosticResult(actual: Record<string, unknown>) {
  if (actual.comparison !== undefined)
    throw new Error('capacity diagnostic results must not contain comparison metadata');
  if (typeof actual.capacityDiagnostic !== 'object' || actual.capacityDiagnostic === null)
    throw new Error('capacity diagnostic metadata is missing');
  const metadata = actual.capacityDiagnostic as Record<string, unknown>;
  if (metadata.kind !== 'continue-after-audio-end') throw new Error('capacity diagnostic kind is invalid');
  if (!Number.isSafeInteger(metadata.suppressedAudioEnds) || (metadata.suppressedAudioEnds as number) < 1)
    throw new Error('capacity diagnostic must suppress at least one audio end');
  if (metadata.firstAudioEndAtRetainedFrame !== 1_743)
    throw new Error('capacity diagnostic first audio-end frame must be 1743');
}

export function assertGateResult(actual: Record<string, unknown>, durationSeconds: GateDuration) {
  const expected = gateExpectation(durationSeconds);
  for (const [name, value] of Object.entries(expected)) {
    if (JSON.stringify(actual[name]) !== JSON.stringify(value))
      throw new Error(`${name} must be ${String(value)}, received ${String(actual[name])}`);
  }
}

export function assertAudioHealthForPlan(actual: AudioHealth, expected: ProductPlan) {
  const fixed = {
    riff: 'RIFF',
    wave: 'WAVE',
    format: 1,
    sampleRate: 44_100,
    channels: 2,
    bitsPerSample: 16,
    riffSize: expected.wavBytes - 8,
    dataBytes: expected.wavBytes - 44,
    byteRate: 176_400,
    blockAlign: 4,
    samplesPerChannel: expected.samplesPerChannel,
    wavBytes: expected.wavBytes,
    decodedSampleRate: 44_100,
    decodedChannels: 2,
    decodedSamples: expected.samplesPerChannel,
  };
  for (const [name, value] of Object.entries(fixed)) {
    if (actual[name as keyof AudioHealth] !== value) throw new Error(`${name} must be ${String(value)}`);
  }
  if (!actual.finite) throw new Error('decoded samples must be finite');
  if (!actual.stereoDiffers) throw new Error('stereo channels must differ');
  if (actual.longestConstantFrameRun >= 44_100)
    throw new Error('longest constant frame run must be shorter than one second');
  if (!(actual.finalSecondDelta > 0)) throw new Error('final second must be varying, not constant');
}

export function assertAudioHealth(actual: AudioHealth, durationSeconds: GateDuration) {
  assertAudioHealthForPlan(actual, gateExpectation(durationSeconds));
}

function productPlan(
  durationSeconds: GateDuration,
  requestedFrames: number,
  retainedFrames: number,
  termination: 'max-frames' | 'natural-end',
): ProductPlan {
  const chunkCount = retainedFrames <= 200 ? 1 : Math.ceil(retainedFrames / 100) - 1;
  const chunks = Array.from({ length: chunkCount }, (_, index) => {
    const startFrame = index * 100;
    const frameLength = Math.min(200, retainedFrames - startFrame);
    const latentLength = Math.floor((frameLength * 441) / 128);
    const cropLeftLatents = index === 0 ? 0 : 86;
    const cropRightLatents = index === chunkCount - 1 ? 0 : 258;
    return {
      startFrame,
      frameLength,
      latentLength,
      cropLeftLatents,
      cropRightLatents,
      samplesPerChannel: (latentLength - cropLeftLatents - cropRightLatents) * 512,
    };
  });
  const samplesPerChannel = chunks.reduce((total, chunk) => total + chunk.samplesPerChannel, 0);
  return {
    durationSeconds,
    requestedFrames,
    retainedFrames,
    termination,
    chunkCount,
    samplesPerChannel,
    wavBytes: 44 + samplesPerChannel * 4,
    flowCalls: chunkCount * 30,
    vocoderCalls: chunkCount * 2,
    semanticDecisions: retainedFrames + (termination === 'natural-end' ? 2 : 1),
    rvqCalls: (retainedFrames + 1) * 7,
    feedbackCalls: retainedFrames + (termination === 'natural-end' ? 1 : 0),
    chunks,
  };
}

export function naturalEnd1743Expectation() {
  return productPlan(120, 3_000, 1_743, 'natural-end');
}

export function assertStableProgressMetrics(progress: readonly ProgressMetric[], stage: 'autoregressive' | 'flow') {
  const events = progress.filter((event) => event.stage === stage);
  if (events.length < 3) throw new Error(`${stage} progress must reach a stable sample window`);
  for (const event of events) {
    if (!Number.isFinite(event.elapsedMs) || event.elapsedMs! < 0)
      throw new Error(`${stage} elapsed time must be finite and non-negative`);
    if ((event.completed ?? 0) < 3 && event.etaMs !== undefined)
      throw new Error(`${stage} ETA must not appear before the stable sample window`);
    if ((event.completed ?? 0) < 3) continue;
    const speed = stage === 'autoregressive' ? event.rate : event.stepMs;
    const speedName = stage === 'autoregressive' ? 'rate' : 'stepMs';
    if (!Number.isFinite(speed) || speed! <= 0)
      throw new Error(`${stage} ${speedName} must be present and positive after stability`);
    if (!Number.isFinite(event.etaMs) || event.etaMs! < 0)
      throw new Error(`${stage} ETA must be present and non-negative after stability`);
  }
}

function assertCountedProgress(
  progress: readonly ProgressMetric[],
  stage: 'autoregressive' | 'flow' | 'acoustic' | 'vocoder',
  total: number,
) {
  const events = progress.filter((event) => event.stage === stage && event.completed !== undefined);
  if (events.length === 0) throw new Error(`${stage} progress must be present`);
  if (events.some((event) => event.total !== total)) throw new Error(`${stage} progress total must be ${total}`);
  const completed = events.map((event) => event.completed!);
  if (completed[0] !== 1) throw new Error(`${stage} progress must start at 1`);
  if (completed.at(-1) !== total) throw new Error(`${stage} progress must finish at ${total}`);
  if (completed.some((value, index) => index > 0 && value <= completed[index - 1]))
    throw new Error(`${stage} progress must be strictly increasing`);
  return { events, completed };
}

function assertProductProgress(progress: readonly ProgressMetric[], expected: ProductPlan) {
  const autoregressive = progress.filter((event) => event.stage === 'autoregressive' && event.completed !== undefined);
  if (autoregressive.length === 0) throw new Error('autoregressive progress must be present');
  if (autoregressive.some((event) => event.total !== expected.requestedFrames))
    throw new Error(`autoregressive progress total must be ${expected.requestedFrames}`);
  const arCompleted = autoregressive.map((event) => event.completed!);
  if (arCompleted[0] !== 1) throw new Error('autoregressive progress must start at 1');
  if (arCompleted.at(-1) !== expected.retainedFrames)
    throw new Error(`autoregressive progress must finish at ${expected.retainedFrames}`);
  if (arCompleted.some((value, index) => index > 0 && value <= arCompleted[index - 1]))
    throw new Error('autoregressive progress must be strictly increasing');
  assertStableProgressMetrics(autoregressive, 'autoregressive');

  const flow = assertCountedProgress(progress, 'flow', expected.flowCalls);
  assertStableProgressMetrics(flow.events, 'flow');
  assertCountedProgress(progress, 'acoustic', expected.chunkCount);
  assertCountedProgress(progress, 'vocoder', expected.vocoderCalls);
  const conditionEvents = progress.filter((event) => event.stage === 'condition').length;
  if (conditionEvents !== expected.chunkCount)
    throw new Error(`condition progress count must be ${expected.chunkCount}`);
  const sessions = progress.filter((event) => event.stage === 'session');
  const sessionNames = sessions.map((event) => event.name);
  if (JSON.stringify(sessionNames) !== JSON.stringify(['autoregressive', 'condition', 'flow', 'vocoder']))
    throw new Error('product session progress is incomplete');
  if (sessions.some((event) => event.activity !== 'indeterminate'))
    throw new Error('session progress must be indeterminate');
  const terminalStages = progress
    .filter((event) => event.stage === 'wav' || event.stage === 'complete')
    .map((event) => event.stage);
  if (JSON.stringify(terminalStages) !== JSON.stringify(['wav', 'complete']))
    throw new Error('product progress must end through wav and complete exactly once');
  if (progress.at(-1)?.stage !== 'complete') throw new Error('complete must be the final progress event');
}

export function assertProductOutcome(input: {
  plan: ProductPlan;
  audio: AudioHealth;
  progress: readonly ProgressMetric[];
}) {
  const actual = input.plan;
  if (![6, 10, 30, 60, 120, 300].includes(actual.durationSeconds)) throw new Error('product duration is unsupported');
  if (actual.requestedFrames !== actual.durationSeconds * 25)
    throw new Error('product requested frames do not match duration');
  if (
    !Number.isInteger(actual.retainedFrames) ||
    actual.retainedFrames < 1 ||
    actual.retainedFrames > actual.requestedFrames
  )
    throw new Error('product retained frames are invalid');
  if (actual.termination !== 'max-frames' && actual.termination !== 'natural-end')
    throw new Error('product termination is invalid');
  const expected = productPlan(
    actual.durationSeconds,
    actual.requestedFrames,
    actual.retainedFrames,
    actual.termination,
  );
  for (const [name, value] of Object.entries(expected)) {
    if (JSON.stringify(actual[name as keyof ProductPlan]) !== JSON.stringify(value))
      throw new Error(`product ${name} does not match the retained plan`);
  }
  assertAudioHealthForPlan(input.audio, expected);
  assertProductProgress(input.progress, expected);
  return {
    status: 'passed',
    termination: expected.termination,
    retainedFrames: expected.retainedFrames,
    wavBytes: expected.wavBytes,
  } as const;
}

export function assertLongDurationProgress(progress: readonly ProgressMetric[], durationSeconds: LongGateDuration) {
  const expected = gateExpectation(durationSeconds);
  const autoregressive = assertCountedProgress(progress, 'autoregressive', expected.retainedFrames);
  const flow = assertCountedProgress(progress, 'flow', expected.flowCalls);
  const acoustic = assertCountedProgress(progress, 'acoustic', expected.chunkCount);
  const vocoder = assertCountedProgress(progress, 'vocoder', expected.vocoderCalls);
  assertStableProgressMetrics(autoregressive.events, 'autoregressive');
  assertStableProgressMetrics(flow.events, 'flow');

  const conditionEvents = progress.filter((event) => event.stage === 'condition').length;
  if (conditionEvents !== expected.chunkCount)
    throw new Error(`condition progress count must be ${expected.chunkCount}`);
  const sessions = progress.filter((event) => event.stage === 'session');
  const sessionNames = sessions.map((event) => event.name);
  const expectedSessions = ['autoregressive', 'condition', 'flow', 'vocoder'];
  if (JSON.stringify(sessionNames) !== JSON.stringify(expectedSessions))
    throw new Error(`session progress must be ${expectedSessions.join(', ')}`);
  if (sessions.some((event) => event.activity !== 'indeterminate'))
    throw new Error('session progress must be indeterminate');
  const terminalStages = progress
    .filter((event) => event.stage === 'wav' || event.stage === 'complete')
    .map((event) => event.stage);
  if (JSON.stringify(terminalStages) !== JSON.stringify(['wav', 'complete']))
    throw new Error('progress must end through wav and complete exactly once');
  if (progress.at(-1)?.stage !== 'complete') throw new Error('complete must be the final progress event');

  const sample = (values: readonly number[], interval: number, total: number) =>
    values.filter((value) => value <= 3 || value % interval === 0 || value === total);
  const metricSamples = (events: readonly ProgressMetric[], interval: number, total: number) =>
    events.filter(
      ({ completed }) =>
        completed !== undefined && (completed <= 3 || completed % interval === 0 || completed === total),
    );
  return {
    durationSeconds,
    eventCount: progress.length,
    stages: {
      autoregressive: {
        eventCount: autoregressive.events.length,
        firstCompleted: autoregressive.completed[0],
        lastCompleted: autoregressive.completed.at(-1),
        total: expected.retainedFrames,
        samples: sample(autoregressive.completed, 250, expected.retainedFrames),
        metricSamples: metricSamples(autoregressive.events, 250, expected.retainedFrames),
      },
      flow: {
        eventCount: flow.events.length,
        firstCompleted: flow.completed[0],
        lastCompleted: flow.completed.at(-1),
        total: expected.flowCalls,
        samples: sample(flow.completed, 30, expected.flowCalls),
        metricSamples: metricSamples(flow.events, 30, expected.flowCalls),
      },
      acoustic: {
        eventCount: acoustic.events.length,
        firstCompleted: acoustic.completed[0],
        lastCompleted: acoustic.completed.at(-1),
        total: expected.chunkCount,
        samples: acoustic.completed,
      },
      vocoder: {
        eventCount: vocoder.events.length,
        firstCompleted: vocoder.completed[0],
        lastCompleted: vocoder.completed.at(-1),
        total: expected.vocoderCalls,
        samples: vocoder.completed,
      },
    },
    conditionEvents,
    sessionNames,
    terminalStages,
  };
}

export function summarizeObservedProgress(progress: readonly ProgressMetric[]) {
  const summarizeStage = (stage: 'autoregressive' | 'flow' | 'acoustic' | 'vocoder', interval: number) => {
    const events = progress.filter((event) => event.stage === stage && event.completed !== undefined);
    const snapshot = (event: ProgressMetric) => ({
      completed: event.completed,
      ...(event.total === undefined ? {} : { total: event.total }),
      ...(event.elapsedMs === undefined ? {} : { elapsedMs: event.elapsedMs }),
      ...(event.rate === undefined ? {} : { rate: event.rate }),
      ...(event.stepMs === undefined ? {} : { stepMs: event.stepMs }),
      ...(event.etaMs === undefined ? {} : { etaMs: event.etaMs }),
    });
    return {
      eventCount: events.length,
      firstCompleted: events[0]?.completed,
      lastCompleted: events.at(-1)?.completed,
      reportedTotals: [...new Set(events.flatMap((event) => (event.total === undefined ? [] : [event.total])))],
      samples: events
        .filter(
          (event, index) => event.completed! <= 3 || event.completed! % interval === 0 || index === events.length - 1,
        )
        .map(snapshot),
    };
  };
  return {
    eventCount: progress.length,
    sessionNames: progress.filter((event) => event.stage === 'session').map((event) => event.name),
    terminalStages: progress
      .filter((event) => event.stage === 'wav' || event.stage === 'complete')
      .map((event) => event.stage),
    stages: {
      autoregressive: summarizeStage('autoregressive', 250),
      flow: summarizeStage('flow', 30),
      acoustic: summarizeStage('acoustic', 1),
      vocoder: summarizeStage('vocoder', 1),
    },
  };
}

export function createObservedRunEvidence(input: {
  durationSeconds: LongGateDuration;
  mode?: LongDurationMode;
  seed: number;
  promptTokens: number;
  result: unknown;
  audio: unknown;
  progress: readonly ProgressMetric[];
  productOutcome: unknown;
}) {
  const result =
    typeof input.result === 'object' && input.result !== null ? (input.result as Record<string, unknown>) : undefined;
  const plan =
    typeof result?.plan === 'object' && result.plan !== null ? (result.plan as Record<string, unknown>) : undefined;
  const observedTermination = plan?.termination;
  return {
    schemaVersion: 1,
    qualification:
      observedTermination === 'natural-end'
        ? {
            status: 'failed',
            requiredTermination: 'max-frames',
            observedTermination,
          }
        : {
            status: 'not-evaluated',
            requiredTermination: 'max-frames',
            ...(observedTermination === undefined ? {} : { observedTermination }),
          },
    productOutcome: input.productOutcome,
    request: {
      durationSeconds: input.durationSeconds,
      mode: input.mode ?? 'product',
      seed: input.seed,
      promptTokens: input.promptTokens,
    },
    result: input.result,
    audio: input.audio,
    progress: summarizeObservedProgress(input.progress),
  };
}

export function assertCancellationBoundary(progress: readonly ProgressMetric[], boundary: 'late-ar' | 'second-flow') {
  for (const terminal of ['wav', 'complete']) {
    if (progress.some(({ stage }) => stage === terminal))
      throw new Error(`${boundary} cancellation must not reach ${terminal}`);
  }
  if (boundary === 'late-ar') {
    const completed = progress.filter(({ stage }) => stage === 'autoregressive').map((event) => event.completed ?? -1);
    const maximum = Math.max(...completed);
    if (maximum < 100 || maximum >= 150) throw new Error('late-ar cancellation must stop from frame 100 through 149');
    if (progress.some(({ stage }) => ['condition', 'acoustic', 'flow', 'vocoder'].includes(stage)))
      throw new Error('late-ar cancellation must not reach acoustic work');
    if (
      progress.some(({ stage, name }) => stage === 'session' && ['condition', 'flow', 'vocoder'].includes(name ?? ''))
    )
      throw new Error('late-ar cancellation must not create acoustic sessions');
    return;
  }
  const completed = progress.filter(({ stage }) => stage === 'flow').map((event) => event.completed ?? -1);
  const maximum = Math.max(...completed);
  if (maximum < 31 || maximum >= 60) throw new Error('second-flow cancellation must stop from flow step 31 through 59');
  if (progress.some(({ stage, completed: count }) => stage === 'acoustic' && count === 2))
    throw new Error('second-flow cancellation must not complete acoustic chunk 2');
  if (progress.some(({ stage, name }) => stage === 'vocoder' || (stage === 'session' && name === 'vocoder')))
    throw new Error('second-flow cancellation must not reach vocoder');
}

export function assertWavIdentifiers(actual: { fmt: string; fmtChunkSize: number; data: string }) {
  if (actual.fmt !== 'fmt ') throw new Error('WAV fmt identifier must be fmt ');
  if (actual.fmtChunkSize !== 16) throw new Error('WAV fmt chunk size must be 16');
  if (actual.data !== 'data') throw new Error('WAV data identifier must be data');
}
