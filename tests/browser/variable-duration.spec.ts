import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { chromium, expect, test, type Page } from '@playwright/test';
import type { MusicGenerationRequest, MusicGenerationWorkerResult, WorkerProgress } from '../../src/workers/protocol';
import {
  assertAudioHealth,
  assertCancellationBoundary,
  assertGateResult,
  assertStableProgressMetrics,
  assertWavIdentifiers,
  gateExpectation,
  type AudioHealth,
  type GateDuration,
} from './variable-duration-assertions';
import {
  assertFreshQualificationCapture,
  resolveQualificationCapture,
  resolveQualificationProfile,
} from './qualification-paths';

const fixedInput = JSON.parse(readFileSync(path.resolve('tools/reference/fixed_case.json'), 'utf8')) as {
  prompt: string;
  lyrics: string;
};

type CapturedProgress = { atMs: number; message: WorkerProgress };
type CompletedRun = {
  progress: CapturedProgress[];
  result: Omit<MusicGenerationWorkerResult, 'wav'>;
  audio: AudioHealth;
  wavIdentifiers: { fmt: string; fmtChunkSize: number; data: string };
};

const timeoutMs = Number(process.env.MINIMAX_VARIABLE_GATE_TIMEOUT_MS ?? 4 * 60 * 60_000);
if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 60_000)
  throw new Error('MINIMAX_VARIABLE_GATE_TIMEOUT_MS must be an integer of at least 60000');
test.setTimeout(timeoutMs);

const checkoutRoot = path.resolve('.');
const linkedWorktree = statSync(path.join(checkoutRoot, '.git')).isFile();
const captureDirectory = resolveQualificationCapture(checkoutRoot, process.env.MINIMAX_VARIABLE_CAPTURE_DIR);

function request(durationSeconds: GateDuration): MusicGenerationRequest {
  return {
    type: 'generate-music',
    manifestUrl: '/artifacts/music-variable/manifest.json',
    prompt: fixedInput.prompt,
    lyrics: fixedInput.lyrics,
    seed: 7,
    durationSeconds,
    sampling: {
      globalGuidance: 1.5,
      semanticTopK: 50,
      residualTopK: 50,
      temperature: 1,
      flowGuidance: 1.7,
      flowSteps: 30,
    },
  };
}

function persistJson(name: string, value: unknown) {
  writeFileSync(path.join(captureDirectory, `${name}.json`), `${JSON.stringify(value, null, 2)}\n`, {
    encoding: 'utf8',
    flag: 'wx',
  });
}

async function startCancelableRun(page: Page, generationRequest: MusicGenerationRequest) {
  await page.evaluate((input) => {
    type GateState = {
      worker: Worker | null;
      events: { atMs: number; message: Record<string, unknown> }[];
      error?: string;
    };
    const scope = globalThis as typeof globalThis & { __variableDurationGate?: GateState };
    scope.__variableDurationGate?.worker?.terminate();
    const worker = new Worker(new URL('/src/workers/inference.worker.ts', location.origin), {
      type: 'module',
    });
    const state: GateState = { worker, events: [] };
    scope.__variableDurationGate = state;
    worker.addEventListener('message', ({ data }: MessageEvent<Record<string, unknown>>) => {
      if (data.type === 'progress') state.events.push({ atMs: performance.now(), message: data });
      else if (data.type === 'error') state.error = String(data.message);
      else if (data.type === 'music-result') state.error = 'generation completed before cancellation';
    });
    worker.addEventListener('error', (event) => {
      state.error = event.message || 'inference worker failed';
    });
    worker.postMessage(input);
  }, generationRequest);
}

async function cancelAfter(page: Page, stage: string, completed: number) {
  await page.waitForFunction(
    ({ expectedStage, expectedCompleted }) => {
      const state = (
        globalThis as typeof globalThis & {
          __variableDurationGate?: {
            events: { message: { stage?: unknown; completed?: unknown } }[];
            error?: string;
          };
        }
      ).__variableDurationGate;
      return Boolean(
        state?.error ||
        state?.events.some(
          ({ message }) =>
            message.stage === expectedStage &&
            typeof message.completed === 'number' &&
            message.completed >= expectedCompleted,
        ),
      );
    },
    { expectedStage: stage, expectedCompleted: completed },
    { timeout: timeoutMs },
  );
  const before = await page.evaluate(() => {
    const state = (
      globalThis as typeof globalThis & {
        __variableDurationGate?: {
          worker: Worker | null;
          events: { atMs: number; message: Record<string, unknown> }[];
          error?: string;
        };
      }
    ).__variableDurationGate;
    if (!state) throw new Error('cancellation state is unavailable');
    if (state.error) throw new Error(state.error);
    state.worker?.terminate();
    state.worker = null;
    return { count: state.events.length, events: state.events };
  });
  await page.waitForTimeout(1_000);
  const after = await page.evaluate(() => {
    const state = (
      globalThis as typeof globalThis & {
        __variableDurationGate?: { events: unknown[]; error?: string };
      }
    ).__variableDurationGate;
    return { count: state?.events.length ?? -1, error: state?.error };
  });
  expect(after.error).toBeUndefined();
  expect(after.count).toBe(before.count);
  return before.events as CapturedProgress[];
}

async function runToCompletion(page: Page, generationRequest: MusicGenerationRequest) {
  return page.evaluate(async (input): Promise<CompletedRun> => {
    const worker = new Worker(new URL('/src/workers/inference.worker.ts', location.origin), {
      type: 'module',
    });
    const progress: CapturedProgress[] = [];
    try {
      return await new Promise<CompletedRun>((resolve, reject) => {
        worker.addEventListener('error', (event) => reject(new Error(event.message)));
        worker.addEventListener('message', ({ data }: MessageEvent<Record<string, unknown>>) => {
          if (data.type === 'progress') {
            progress.push({ atMs: performance.now(), message: data as WorkerProgress });
            return;
          }
          if (data.type === 'error') {
            reject(new Error(String(data.message)));
            return;
          }
          if (data.type !== 'music-result') return;
          void (async () => {
            const received = data.result as MusicGenerationWorkerResult;
            const wav = received.wav;
            const view = new DataView(wav);
            const ascii = (offset: number, length: number) =>
              String.fromCharCode(...new Uint8Array(wav, offset, length));
            const frames = view.getUint32(40, true) / 4;
            let longestConstantFrameRun = 1;
            let currentConstantFrameRun = 1;
            let stereoDiffers = false;
            let finalSecondDelta = 0;
            const finalSecondStart = Math.max(1, frames - 44_100);
            let previousLeft = view.getInt16(44, true);
            let previousRight = view.getInt16(46, true);
            for (let frame = 1; frame < frames; frame++) {
              const offset = 44 + frame * 4;
              const left = view.getInt16(offset, true);
              const right = view.getInt16(offset + 2, true);
              stereoDiffers ||= left !== right;
              if (left === previousLeft && right === previousRight) currentConstantFrameRun++;
              else currentConstantFrameRun = 1;
              longestConstantFrameRun = Math.max(longestConstantFrameRun, currentConstantFrameRun);
              if (frame >= finalSecondStart)
                finalSecondDelta += Math.abs(left - previousLeft) + Math.abs(right - previousRight);
              previousLeft = left;
              previousRight = right;
            }
            const audioContext = new AudioContext({ sampleRate: 44_100 });
            let decoded: AudioBuffer;
            try {
              decoded = await audioContext.decodeAudioData(wav.slice(0));
            } finally {
              await audioContext.close();
            }
            let finite = true;
            for (let channel = 0; channel < decoded.numberOfChannels && finite; channel++) {
              const samples = decoded.getChannelData(channel);
              for (let index = 0; index < samples.length; index++) {
                if (!Number.isFinite(samples[index])) {
                  finite = false;
                  break;
                }
              }
            }
            const { wav: _wav, ...result } = received;
            void _wav;
            resolve({
              progress,
              result,
              audio: {
                riff: ascii(0, 4),
                wave: ascii(8, 4),
                format: view.getUint16(20, true),
                channels: view.getUint16(22, true),
                sampleRate: view.getUint32(24, true),
                bitsPerSample: view.getUint16(34, true),
                riffSize: view.getUint32(4, true),
                dataBytes: view.getUint32(40, true),
                byteRate: view.getUint32(28, true),
                blockAlign: view.getUint16(32, true),
                samplesPerChannel: frames,
                wavBytes: wav.byteLength,
                decodedSampleRate: decoded.sampleRate,
                decodedChannels: decoded.numberOfChannels,
                decodedSamples: decoded.length,
                finite,
                stereoDiffers,
                longestConstantFrameRun,
                finalSecondDelta,
              },
              wavIdentifiers: {
                fmt: ascii(12, 4),
                fmtChunkSize: view.getUint32(16, true),
                data: ascii(36, 4),
              },
            });
          })().catch(reject);
        });
        worker.postMessage(input);
      });
    } finally {
      worker.terminate();
    }
  }, generationRequest);
}

function progressValues(progress: CapturedProgress[], stage: string) {
  return progress
    .filter(({ message }) => message.stage === stage && message.completed !== undefined)
    .map(({ message }) => message.completed as number);
}

function assertProgress(run: CompletedRun, durationSeconds: GateDuration) {
  const expected = gateExpectation(durationSeconds);
  const countedStages = {
    autoregressive: expected.retainedFrames,
    flow: expected.flowCalls,
    acoustic: expected.chunkCount,
    vocoder: expected.vocoderCalls,
  } as const;
  for (const [stage, total] of Object.entries(countedStages)) {
    const events = run.progress.filter(({ message }) => message.stage === stage && message.completed !== undefined);
    expect(events.every(({ message }) => message.total === total)).toBe(true);
  }
  const autoregressive = run.progress.filter(({ message }) => message.stage === 'autoregressive');
  const flow = run.progress.filter(({ message }) => message.stage === 'flow');
  assertStableProgressMetrics(
    autoregressive.map(({ message }) => message),
    'autoregressive',
  );
  assertStableProgressMetrics(
    flow.map(({ message }) => message),
    'flow',
  );
  expect(progressValues(run.progress, 'autoregressive')).toEqual(
    Array.from({ length: expected.retainedFrames }, (_, index) => index + 1),
  );
  expect(progressValues(run.progress, 'flow')).toEqual(
    Array.from({ length: expected.flowCalls }, (_, index) => index + 1),
  );
  expect(progressValues(run.progress, 'acoustic')).toEqual(
    Array.from({ length: expected.chunkCount }, (_, index) => index + 1),
  );
  expect(progressValues(run.progress, 'vocoder')).toEqual(
    Array.from({ length: expected.vocoderCalls }, (_, index) => index + 1),
  );
  expect(run.progress.filter(({ message }) => message.stage === 'condition')).toHaveLength(expected.chunkCount);
  const sessionNames = run.progress
    .filter(({ message }) => message.stage === 'session')
    .map(({ message }) => message.name);
  expect(sessionNames).toEqual(['autoregressive', 'condition', 'flow', 'vocoder']);
  expect(
    run.progress
      .filter(({ message }) => message.stage === 'session')
      .every(({ message }) => message.activity === 'indeterminate'),
  ).toBe(true);
  const completedArtifactBytes = run.progress
    .filter(({ message }) => message.stage === 'artifact' && message.completedBytes !== undefined)
    .map(({ message }) => message.completedBytes as number);
  expect(completedArtifactBytes).toEqual([...completedArtifactBytes].sort((left, right) => left - right));
  const stages = run.progress.map(({ message }) => message.stage);
  expect(stages.at(-1)).toBe('complete');
  expect(stages.lastIndexOf('wav')).toBeLessThan(stages.lastIndexOf('complete'));

  if (durationSeconds === 10) {
    const flow30 = run.progress.findIndex(({ message }) => message.stage === 'flow' && message.completed === 30);
    const acoustic1 = run.progress.findIndex(({ message }) => message.stage === 'acoustic' && message.completed === 1);
    const conditionEvents = run.progress
      .map(({ message }, index) => ({ index, message }))
      .filter(({ message }) => message.stage === 'condition');
    const flow31 = run.progress.findIndex(({ message }) => message.stage === 'flow' && message.completed === 31);
    const flow60 = run.progress.findIndex(({ message }) => message.stage === 'flow' && message.completed === 60);
    const acoustic2 = run.progress.findIndex(({ message }) => message.stage === 'acoustic' && message.completed === 2);
    expect(flow30).toBeLessThan(acoustic1);
    expect(acoustic1).toBeLessThan(conditionEvents[1].index);
    expect(conditionEvents[1].index).toBeLessThan(flow31);
    expect(flow60).toBeLessThan(acoustic2);
  }
}

function assertLateArCancellation(progress: CapturedProgress[]) {
  assertCancellationBoundary(
    progress.map(({ message }) => message),
    'late-ar',
  );
}

function assertSecondFlowCancellation(progress: CapturedProgress[]) {
  assertCancellationBoundary(
    progress.map(({ message }) => message),
    'second-flow',
  );
}

test('qualifies fallback-disabled six and ten second generation with practical cancellation', async () => {
  expect(test.info().config.workers).toBe(1);
  assertFreshQualificationCapture(captureDirectory, existsSync);
  mkdirSync(captureDirectory, { recursive: true });
  const sessionSource = readFileSync(path.resolve('src/runtime/model/ort-session.ts'), 'utf8');
  expect(sessionSource).toContain("executionProviders: ['webgpu']");
  expect(sessionSource).toContain("disable_cpu_ep_fallback: '1'");

  const profile = resolveQualificationProfile(
    checkoutRoot,
    process.env.MINIMAX_VARIABLE_CHROME_PROFILE,
    linkedWorktree,
  );

  const context = await chromium.launchPersistentContext(profile, {
    channel: 'chrome',
    headless: false,
  });
  const wasmRequests: string[] = [];
  context.on('request', (networkRequest) => {
    if (networkRequest.url().includes('ort-wasm-simd-threaded.jspi.wasm')) wasmRequests.push(networkRequest.url());
  });
  const page = context.pages()[0] ?? (await context.newPage());
  try {
    await page.goto('http://127.0.0.1:5173/');

    await startCancelableRun(page, request(6));
    const lateArCancellation = await cancelAfter(page, 'autoregressive', 100);
    assertLateArCancellation(lateArCancellation);
    persistJson('6s-late-ar-cancel-progress', lateArCancellation);

    const sixSeconds = await runToCompletion(page, request(6));
    if (!sixSeconds.result.plan) throw new Error('six-second result plan is missing');
    assertGateResult(sixSeconds.result.plan as unknown as Record<string, unknown>, 6);
    expect(sixSeconds.result.plan.chunks).toEqual(gateExpectation(6).chunks);
    assertAudioHealth(sixSeconds.audio, 6);
    assertWavIdentifiers(sixSeconds.wavIdentifiers);
    assertProgress(sixSeconds, 6);
    expect(sixSeconds.result.status).toBe('passed');
    expect(sixSeconds.result.wavBytes).toBe(1_056_812);
    expect(sixSeconds.result.artifactFetches).toBe(0);
    expect(sixSeconds.result.adapters).toHaveLength(3);
    persistJson('6s-progress', sixSeconds.progress);
    persistJson('6s-result', {
      ...sixSeconds.result,
      audio: sixSeconds.audio,
      wavIdentifiers: sixSeconds.wavIdentifiers,
    });

    await startCancelableRun(page, request(10));
    const flowCancellation = await cancelAfter(page, 'flow', 31);
    assertSecondFlowCancellation(flowCancellation);
    persistJson('10s-second-flow-cancel-progress', flowCancellation);

    const tenSeconds = await runToCompletion(page, request(10));
    if (!tenSeconds.result.plan) throw new Error('ten-second result plan is missing');
    assertGateResult(tenSeconds.result.plan as unknown as Record<string, unknown>, 10);
    expect(tenSeconds.result.plan.chunks).toEqual(gateExpectation(10).chunks);
    assertAudioHealth(tenSeconds.audio, 10);
    assertWavIdentifiers(tenSeconds.wavIdentifiers);
    assertProgress(tenSeconds, 10);
    expect(tenSeconds.result.status).toBe('passed');
    expect(tenSeconds.result.plan).toMatchObject({
      chunkCount: 2,
      flowCalls: 60,
      vocoderCalls: 4,
      samplesPerChannel: 440_832,
      wavBytes: 1_763_372,
    });
    expect(tenSeconds.result.wavBytes).toBe(1_763_372);
    expect(tenSeconds.result.artifactFetches).toBe(0);
    expect(tenSeconds.result.adapters).toHaveLength(3);
    expect(tenSeconds.result.comparison).toBeDefined();
    expect(tenSeconds.result.comparison).toMatchObject({
      durationSeconds: 10,
      seed: 7,
      retainedFrames: 250,
      termination: 'max-frames',
    });
    expect(tenSeconds.result.comparison).not.toHaveProperty('input');
    expect(tenSeconds.result.comparison).not.toHaveProperty('generation');
    expect(tenSeconds.result.comparison).not.toHaveProperty('sampler');
    expect(tenSeconds.result.comparison).not.toHaveProperty('provenance');
    expect(tenSeconds.result.comparison?.manifestHash).toBe(tenSeconds.result.manifestHash);
    persistJson('10s-progress', tenSeconds.progress);
    persistJson('10s-result', {
      ...tenSeconds.result,
      audio: tenSeconds.audio,
      wavIdentifiers: tenSeconds.wavIdentifiers,
    });

    expect(wasmRequests.length).toBeGreaterThan(0);
    expect(wasmRequests.every((url) => url.endsWith('.jspi.wasm?v=0569a267'))).toBe(true);
  } finally {
    await context.close();
  }
});
