import {
  constants,
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import { chromium, expect, test } from '@playwright/test';
import type {
  MusicCapacityDiagnosticRequest,
  MusicGenerationRequest,
  MusicGenerationWorkerResult,
  WorkerProgress,
} from '../../src/workers/protocol';
import {
  assertAudioHealth,
  assertCapacityDiagnosticResult,
  assertGateResult,
  assertLongDurationProgress,
  assertProductOutcome,
  createObservedRunEvidence,
  gateExpectation,
  parseLongGateDuration,
  parseLongDurationMode,
  type AudioHealth,
  type ProductPlan,
} from './variable-duration-assertions';
import {
  assertFreshQualificationCapture,
  resolveQualificationCapture,
  resolveQualificationProfile,
} from './qualification-paths';
import { analyzeCanonicalPcm16Wav } from './reference-capture-helpers';

const fixedInput = JSON.parse(
  readFileSync(path.resolve('tools/reference/fixed_case.json'), 'utf8'),
) as { prompt: string; lyrics: string };

type CapturedProgress = { atMs: number; message: WorkerProgress };
type DecodedAudio = {
  sampleRate: number;
  channels: number;
  samples: number;
  finite: boolean;
};
type CompletedRun = {
  progress: CapturedProgress[];
  result: Omit<MusicGenerationWorkerResult, 'wav'>;
  decoded: DecodedAudio;
};

const durationSeconds = parseLongGateDuration(process.env.MINIMAX_LONG_DURATION_SECONDS);
const mode = parseLongDurationMode(process.env.MINIMAX_LONG_DURATION_MODE, durationSeconds);
const timeoutMs = Number(process.env.MINIMAX_LONG_DURATION_TIMEOUT_MS ?? 4 * 60 * 60_000);
if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 60_000)
  throw new Error('MINIMAX_LONG_DURATION_TIMEOUT_MS must be an integer of at least 60000');
test.setTimeout(timeoutMs);

const checkoutRoot = path.resolve('.');
const linkedWorktree = statSync(path.join(checkoutRoot, '.git')).isFile();
const captureDirectory = resolveQualificationCapture(
  checkoutRoot,
  process.env.MINIMAX_VARIABLE_CAPTURE_DIR,
);

function writeJsonExclusive(name: string, value: unknown) {
  writeFileSync(
    path.join(captureDirectory, name),
    `${JSON.stringify(value, null, 2)}\n`,
    { encoding: 'utf8', flag: 'wx' },
  );
}

test(`qualifies one isolated ${durationSeconds}-second ${mode} generation`, async () => {
  expect(process.env.MINIMAX_RELEASE).toBe('music-variable');
  expect(process.env.MINIMAX_VARIABLE_GATE).toBe('long-duration');
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
    acceptDownloads: true,
  });
  const wasmRequests: string[] = [];
  context.on('request', (networkRequest) => {
    if (networkRequest.url().includes('ort-wasm-simd-threaded.jspi.wasm'))
      wasmRequests.push(networkRequest.url());
  });
  const page = context.pages()[0] ?? await context.newPage();
  let run!: CompletedRun;
  const evidenceStem = mode === 'capacity-diagnostic'
    ? `${durationSeconds}s-capacity-diagnostic`
    : `${durationSeconds}s`;
  const wavName = `${evidenceStem}.wav`;
  const wavPath = path.join(captureDirectory, wavName);
  try {
    await page.goto('http://127.0.0.1:5173/');
    const generationRequest: MusicGenerationRequest | MusicCapacityDiagnosticRequest =
      mode === 'capacity-diagnostic'
        ? {
            type: 'diagnose-music-capacity',
            diagnostic: 'continue-after-audio-end',
            manifestUrl: 'http://127.0.0.1:5174/manifest.json',
            seed: 7,
            durationSeconds: 300,
            promptTokens: 40,
            requestedFrames: 7_500,
          }
        : {
            type: 'generate-music',
            manifestUrl: 'http://127.0.0.1:5174/manifest.json',
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
    const downloadPromise = page.waitForEvent('download', { timeout: timeoutMs });
    const runPromise = page.evaluate(async (input): Promise<CompletedRun> => {
      const worker = new Worker(
        new URL('/src/workers/inference.worker.ts', location.origin),
        { type: 'module' },
      );
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
              const audioContext = new AudioContext({ sampleRate: 44_100 });
              let decoded: AudioBuffer;
              try {
                decoded = await audioContext.decodeAudioData(wav.slice(0));
              } finally {
                await audioContext.close();
              }
              let finite = true;
              for (let channel = 0; channel < decoded.numberOfChannels && finite; channel++) {
                const values = decoded.getChannelData(channel);
                for (let index = 0; index < values.length; index++) {
                  if (!Number.isFinite(values[index])) {
                    finite = false;
                    break;
                  }
                }
              }
              const objectUrl = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }));
              const anchor = document.createElement('a');
              anchor.href = objectUrl;
              const diagnostic = input.type === 'diagnose-music-capacity'
                ? '-capacity-diagnostic'
                : '';
              anchor.download = `minimax-music3-${input.durationSeconds}s${diagnostic}.wav`;
              document.body.append(anchor);
              anchor.click();
              anchor.remove();
              setTimeout(() => URL.revokeObjectURL(objectUrl), 1_000);
              const { wav: _wav, ...result } = received;
              void _wav;
              resolve({
                progress,
                result,
                decoded: {
                  sampleRate: decoded.sampleRate,
                  channels: decoded.numberOfChannels,
                  samples: decoded.length,
                  finite,
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
    const [download, completed] = await Promise.all([downloadPromise, runPromise]);
    run = completed;
    const downloadPath = await download.path();
    if (!downloadPath) throw new Error('long-duration WAV download path is unavailable');
    copyFileSync(downloadPath, wavPath, constants.COPYFILE_EXCL);
  } finally {
    await context.close();
  }

  const wav = readFileSync(wavPath);
  const analysis = analyzeCanonicalPcm16Wav(wav);
  const expected = gateExpectation(durationSeconds);
  const audio: AudioHealth = {
    ...analysis.structure,
    ...analysis.health,
    decodedSampleRate: run.decoded.sampleRate,
    decodedChannels: run.decoded.channels,
    decodedSamples: run.decoded.samples,
    finite: run.decoded.finite,
  };
  const progress = run.progress.map(({ message }) => message);
  let productFailure: Error | undefined;
  let productOutcome: unknown = { status: 'not-applicable', mode };
  if (mode === 'product') {
    try {
      if (!run.result.plan) throw new Error('long-duration result plan is missing');
      productOutcome = assertProductOutcome({
        plan: run.result.plan as unknown as ProductPlan,
        audio,
        progress,
      });
    } catch (error) {
      productFailure = error instanceof Error ? error : new Error(String(error));
      productOutcome = { status: 'failed', error: productFailure.message };
    }
  }

  writeJsonExclusive(`${evidenceStem}-observed.json`, createObservedRunEvidence({
    durationSeconds,
    mode,
    seed: 7,
    promptTokens: 40,
    result: run.result,
    audio,
    progress,
    productOutcome,
  }));

  if (productFailure) throw productFailure;
  if (!run.result.plan) throw new Error('long-duration result plan is missing');
  assertGateResult(run.result.plan as unknown as Record<string, unknown>, durationSeconds);
  expect(run.result.plan.chunks).toEqual(expected.chunks);
  assertAudioHealth(audio, durationSeconds);
  const progressEvidence = assertLongDurationProgress(
    progress,
    durationSeconds,
  );
  expect(run.result.status).toBe('passed');
  expect(run.result.attemptedSeeds).toEqual([7]);
  expect(run.result.wavBytes).toBe(expected.wavBytes);
  expect(run.result.artifactFetches).toBe(0);
  expect(run.result.adapters).toHaveLength(3);
  expect(run.result.flowStepMs).toHaveLength(expected.flowCalls);
  expect(run.result.flowStepMs.every((milliseconds) =>
    Number.isFinite(milliseconds) && milliseconds >= 0)).toBe(true);
  if (mode === 'capacity-diagnostic')
    assertCapacityDiagnosticResult(run.result as unknown as Record<string, unknown>);
  else
    expect(run.result).not.toHaveProperty('capacityDiagnostic');
  expect(run.result.comparison).toBeUndefined();
  expect(wasmRequests.length).toBeGreaterThan(0);
  expect(wasmRequests.every((url) => url.endsWith('.jspi.wasm?v=0569a267'))).toBe(true);

  writeJsonExclusive(`${evidenceStem}-result.json`, {
    durationSeconds,
    mode,
    seed: 7,
    promptTokens: 40,
    result: run.result,
    audio,
    wavFile: wavName,
  });
  writeJsonExclusive(`${evidenceStem}-progress.json`, progressEvidence);
});
