import { constants, copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { chromium, expect, test, type Page } from '@playwright/test';
import type { FixedComparisonMetadata } from '../../src/runtime/reference/fixed-comparison';
import type { MusicGenerationResultPlan, MusicGenerationWorkerResult } from '../../src/workers/protocol';
import { assertAudioHealth, type AudioHealth } from './variable-duration-assertions';
import {
  analyzeCanonicalPcm16Wav,
  assertFreshCaptureLayout,
  assertMatchingWavSha256,
  resolveCaptureLayout,
  type CanonicalWavAnalysis,
  type ReferenceCaptureLayout,
} from './reference-capture-helpers';

const timeoutMs = Number(process.env.MINIMAX_REFERENCE_TIMEOUT_MS ?? 4 * 60 * 60_000);
if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 60_000)
  throw new Error('MINIMAX_REFERENCE_TIMEOUT_MS must be an integer of at least 60000');
test.setTimeout(timeoutMs);

type RenderedMusicResult = Omit<MusicGenerationWorkerResult, 'wav'>;
type DecodedAudio = { sampleRate: number; channels: number; samples: number; finite: boolean };
type CapturedRun = {
  comparison: FixedComparisonMetadata;
  plan: MusicGenerationResultPlan;
  structure: CanonicalWavAnalysis['structure'];
  health: CanonicalWavAnalysis['health'];
  decoded: DecodedAudio;
  wavSha256: string;
  metricsFile: string;
  wavFile: string;
};

const fixedCase = JSON.parse(
  readFileSync(path.resolve('tools/reference/fixed_case.json'), 'utf8'),
) as { comparisonMetricKeys: string[] };

const expectedPlan: MusicGenerationResultPlan = {
  durationSeconds: 10,
  requestedFrames: 250,
  retainedFrames: 250,
  termination: 'max-frames',
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
      frameLength: 150,
      latentLength: 516,
      cropLeftLatents: 86,
      cropRightLatents: 0,
      samplesPerChannel: 220_160,
    },
  ],
  samplesPerChannel: 440_832,
  wavBytes: 1_763_372,
  flowCalls: 60,
  vocoderCalls: 4,
  semanticDecisions: 251,
  rvqCalls: 1_757,
  feedbackCalls: 250,
};

function writeJsonExclusive(file: string, value: unknown) {
  writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' });
}

function runReferenceTool(checkoutRoot: string, args: readonly string[]) {
  const result = spawnSync(
    'uv',
    ['run', 'python', 'tools/reference/reference_case.py', ...args],
    { cwd: checkoutRoot, encoding: 'utf8', maxBuffer: 1024 * 1024 },
  );
  if (result.error) throw result.error;
  if (result.status !== 0)
    throw new Error(
      `reference_case.py failed with status ${String(result.status)}\n${result.stdout}${result.stderr}`,
    );
}

async function decodeAudio(page: Page): Promise<DecodedAudio> {
  return page.getByTestId('generated-audio').evaluate(async (element) => {
    const wav = await fetch((element as HTMLAudioElement).src).then((response) => response.arrayBuffer());
    const context = new AudioContext({ sampleRate: 44_100 });
    try {
      const decoded = await context.decodeAudioData(wav);
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
      return {
        sampleRate: decoded.sampleRate,
        channels: decoded.numberOfChannels,
        samples: decoded.length,
        finite,
      };
    } finally {
      await context.close();
    }
  });
}

function assertHealthyWav(analysis: CanonicalWavAnalysis, decoded: DecodedAudio) {
  const audio: AudioHealth = {
    riff: analysis.structure.riff,
    wave: analysis.structure.wave,
    format: analysis.structure.format,
    sampleRate: analysis.structure.sampleRate,
    channels: analysis.structure.channels,
    bitsPerSample: analysis.structure.bitsPerSample,
    riffSize: analysis.structure.riffSize,
    dataBytes: analysis.structure.dataBytes,
    byteRate: analysis.structure.byteRate,
    blockAlign: analysis.structure.blockAlign,
    samplesPerChannel: analysis.structure.samplesPerChannel,
    wavBytes: analysis.structure.wavBytes,
    decodedSampleRate: decoded.sampleRate,
    decodedChannels: decoded.channels,
    decodedSamples: decoded.samples,
    finite: decoded.finite,
    ...analysis.health,
  };
  assertAudioHealth(audio, 10);
}

async function captureRun(
  page: Page,
  layout: ReferenceCaptureLayout,
  runNumber: 1 | 2,
): Promise<CapturedRun> {
  const resultSection = page.getByTestId('music-generation-result');
  const generate = page.getByRole('button', { name: 'Generate 10-second music' });
  await expect(generate).toBeEnabled();
  await generate.click();
  await expect(resultSection).toHaveCount(0);
  await page.waitForFunction(
    () =>
      document.querySelector('[data-testid="music-generation-result"]')
      || document.querySelector('output')?.textContent?.startsWith('Error:'),
    undefined,
    { timeout: timeoutMs },
  );
  await expect(page.locator('output')).not.toHaveText(/^Error:/);

  const renderedText = await resultSection.locator('pre').textContent();
  if (!renderedText) throw new Error(`run ${runNumber} rendered result is missing`);
  const result = JSON.parse(renderedText) as RenderedMusicResult;
  if (!result.plan || !result.comparison)
    throw new Error(`run ${runNumber} fixed comparison metadata or result plan is missing`);
  expect(result.plan).toEqual(expectedPlan);
  expect(result.artifactFetches).toBe(0);
  expect(result.attemptedSeeds).toEqual([7]);
  expect(result.comparison.manifestHash).toBe(result.manifestHash);
  expect(Object.keys(result.comparison).sort()).toEqual([...fixedCase.comparisonMetricKeys].sort());

  const decoded = await decodeAudio(page);
  const downloadPromise = page.waitForEvent('download');
  await page.getByTestId('download-music').click();
  const download = await downloadPromise;
  const downloadPath = await download.path();
  if (!downloadPath) throw new Error(`run ${runNumber} browser download path is unavailable`);
  const wavFile = `run-${runNumber}.wav`;
  const metricsFile = `run-${runNumber}.metrics.json`;
  const wavPath = path.join(layout.captureDirectory, wavFile);
  const metricsPath = path.join(layout.captureDirectory, metricsFile);
  copyFileSync(downloadPath, wavPath, constants.COPYFILE_EXCL);
  const wav = readFileSync(wavPath);
  const analysis = analyzeCanonicalPcm16Wav(wav);
  assertHealthyWav(analysis, decoded);
  writeJsonExclusive(metricsPath, result.comparison);

  return {
    comparison: result.comparison,
    plan: result.plan,
    structure: analysis.structure,
    health: analysis.health,
    decoded,
    wavSha256: createHash('sha256').update(wav).digest('hex'),
    metricsFile,
    wavFile,
  };
}

test('captures two fixed ten-second WebGPU runs and publishes a verified cloud receipt', async () => {
  const captureId = process.env.MINIMAX_REFERENCE_CAPTURE_ID;
  if (!captureId) throw new Error('MINIMAX_REFERENCE_CAPTURE_ID is required');
  if (process.env.MINIMAX_RELEASE !== 'music-variable')
    throw new Error('MINIMAX_RELEASE must be music-variable');
  expect(test.info().config.workers).toBe(1);

  const checkoutRoot = path.resolve('.');
  const layout = resolveCaptureLayout(
    checkoutRoot,
    captureId,
    process.env.MINIMAX_REFERENCE_CHROME_PROFILE
      ?? process.env.MINIMAX_VARIABLE_CHROME_PROFILE,
    statSync(path.join(checkoutRoot, '.git')).isFile(),
  );
  assertFreshCaptureLayout(layout, existsSync);
  const manifest = path.join(checkoutRoot, 'artifacts', 'release', 'music-variable', 'manifest.json');
  if (!existsSync(manifest) || !statSync(manifest).isFile())
    throw new Error('music-variable manifest is unavailable');

  mkdirSync(layout.captureRoot, { recursive: true });
  mkdirSync(layout.caseRoot, { recursive: true });
  assertFreshCaptureLayout(layout, existsSync);
  mkdirSync(layout.captureDirectory);

  const context = await chromium.launchPersistentContext(layout.profile, {
    channel: 'chrome',
    headless: false,
  });
  const page = context.pages()[0] ?? await context.newPage();
  let first!: CapturedRun;
  let second!: CapturedRun;
  try {
    await page.goto('http://127.0.0.1:5173/diagnostics');
    await page.getByLabel('Music duration').selectOption('10');
    first = await captureRun(page, layout, 1);
    second = await captureRun(page, layout, 2);
  } finally {
    await context.close();
  }

  expect(second.comparison).toEqual(first.comparison);
  expect(second.plan).toEqual(first.plan);
  expect(second.structure).toEqual(first.structure);
  expect(second.decoded).toEqual(first.decoded);
  assertMatchingWavSha256(first.wavSha256, second.wavSha256);

  const firstMetrics = path.join(layout.captureDirectory, first.metricsFile);
  const firstWav = path.join(layout.captureDirectory, first.wavFile);
  if (existsSync(layout.caseDirectory))
    throw new Error(`reference case already exists: ${layout.caseDirectory}`);
  runReferenceTool(checkoutRoot, [
    'build',
    '--manifest', manifest,
    '--metrics', firstMetrics,
    '--wav', firstWav,
    '--output-root', layout.caseRoot,
    '--case-id', layout.captureId,
  ]);
  runReferenceTool(checkoutRoot, [
    'verify',
    '--case', layout.caseDirectory,
    '--manifest', manifest,
    '--metrics', firstMetrics,
    '--wav', firstWav,
  ]);
  expect(readdirSync(layout.caseDirectory)).toEqual(['receipt.json']);

  writeJsonExclusive(path.join(layout.captureDirectory, 'repeat.json'), {
    schemaVersion: 1,
    captureId: layout.captureId,
    requestedDurationSeconds: 10,
    seed: 7,
    metadataEqual: true,
    planEqual: true,
    structureEqual: true,
    byteReproducible: true,
    runs: [first, second].map((run, index) => ({
      run: index + 1,
      metricsFile: run.metricsFile,
      wavFile: run.wavFile,
      wavSha256: run.wavSha256,
      plan: run.plan,
      structure: run.structure,
      decoded: run.decoded,
      health: run.health,
    })),
    receipt: path.relative(layout.captureDirectory, path.join(layout.caseDirectory, 'receipt.json'))
      .split(path.sep).join('/'),
  });
});
