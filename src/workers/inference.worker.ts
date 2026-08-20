/// <reference lib="webworker" />
import { OpfsArtifactStore, ensureArtifact } from '../runtime/model/artifact-cache';
import { OpfsFp16EmbeddingTable } from '../runtime/model/embedding-table';
import {
  parseConditionManifest,
  parseFlowManifest,
  parseMusicManifest,
  parseModelManifest,
  parseRvqStageManifest,
  parseVocoderManifest,
} from '../runtime/model/manifest';
import { createWebGpuDevice } from '../runtime/model/webgpu-device';
import { runConditionSmoke } from '../runtime/pipeline/condition-smoke';
import { runFlowSmoke } from '../runtime/pipeline/flow-generation';
import { runFixedFlowGeneration } from '../runtime/pipeline/flow-generation';
import { runGlobalSmoke } from '../runtime/pipeline/global-smoke';
import { runRvqSmoke } from '../runtime/pipeline/rvq-smoke';
import {
  analyticVocoderLatents,
  generateFixedVocoderWav,
} from '../runtime/pipeline/vocoder-generation';
import {
  areFiniteFp16,
  createFrameGenerator,
  EarlyAudioEndError,
  readConditionalGpuFp16,
} from '../runtime/pipeline/rvq-generation';
import {
  deterministicGaussianFp16,
  generateFiveSecondMusic,
  readExactGpuFp16,
} from '../runtime/pipeline/music-generation';
import type { WorkerRequest, WorkerResponse } from './protocol';

const send = (message: WorkerResponse, transfer?: Transferable[]) => self.postMessage(message, transfer ?? []);
const hashText = async (text: string) =>
  Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))))
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  void run(event.data).catch((error: unknown) =>
    send({
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
    }),
  );
};
async function run(request: WorkerRequest) {
  if (request.type === 'generate-music-5s') return runMusicGeneration(request);
  if (request.type === 'run-vocoder-smoke') return runVocoder(request.manifestUrl);
  if (request.type === 'run-flow-smoke') return runFlow(request.manifestUrl);
  if (request.type === 'run-condition-smoke') return runCondition(request.manifestUrl);
  if (request.type === 'generate-frames') return runFrameGeneration(request);
  if (request.type === 'run-rvq-smoke') return runRvq(request.manifestUrl);
  if (request.type !== 'run-global-smoke') return;
  send({
    type: 'progress',
    stage: 'manifest',
    detail: 'Reading release manifest',
  });
  let response: Response;
  try {
    response = await fetch(request.manifestUrl);
  } catch {
    throw new Error('Release manifest is unavailable');
  }
  if (!response.ok) throw new Error('Release manifest is unavailable');
  const manifestText = await response.text();
  const manifest = parseModelManifest(JSON.parse(manifestText));
  const cache = await OpfsArtifactStore.open(await hashText(manifestText));
  const base = new URL(request.manifestUrl, self.location.href);
  const artifacts = [
    manifest.graph,
    ...manifest.graph.externalData,
    manifest.reducedHead,
    ...manifest.reducedHead.externalData,
    ...manifest.embedding.shards,
  ];
  let artifactFetches = 0;
  for (const artifact of artifacts)
    await ensureArtifact(
      artifact,
      new URL(artifact.path, base),
      cache,
      ({ path, loaded, total }) =>
        send({
          type: 'progress',
          stage: 'artifact',
          detail: path,
          loaded,
          total,
        }),
      async (input, init) => {
        artifactFetches++;
        return fetch(input, init);
      },
    );
  send({
    type: 'progress',
    stage: 'adapter',
    detail: 'Requesting shader-f16 WebGPU device',
  });
  const { adapter, device } = await createWebGpuDevice(navigator.gpu, manifest.webgpu.requiredLimits);
  send({
    type: 'progress',
    stage: 'session',
    detail: 'Creating WebGPU decoder session',
  });
  const started = performance.now();
  const { createOrtSession } = await import('../runtime/model/ort-session');
  const ort = await import('onnxruntime-web/jspi');
  let graph: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  let head: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  try {
    graph = await createOrtSession(manifest.graph, cache, device);
    head = await createOrtSession(manifest.reducedHead, cache, device);
    const sessionCreateMs = performance.now() - started;
    const metrics = await runGlobalSmoke({
      ort,
      decoder: graph,
      head,
      embedding: await OpfsFp16EmbeddingTable.open(manifest.embedding, (path) => cache.openSyncFile(path)),
      embeddingTable: manifest.embedding,
      kvPairs: manifest.kvPairs,
    });
    send({
      type: 'result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        ...metrics,
        sessionCreateMs,
        artifactFetches,
        cacheReuseCount: artifacts.length - artifactFetches,
        status: 'passed',
      },
    });
  } finally {
    await head?.release();
    await graph?.release();
    device.destroy();
  }
}

async function runVocoder(manifestUrl: string) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading vocoder release manifest' });
  const release = await readManifest(manifestUrl, 'Vocoder release manifest is unavailable');
  const manifest = parseVocoderManifest(JSON.parse(release.text));
  const artifacts = [manifest.vocoder, ...manifest.vocoder.externalData];
  const artifactFetches = await cacheArtifacts(artifacts, release.base, release.cache);
  send({ type: 'progress', stage: 'adapter', detail: 'Requesting shader-f16 WebGPU device' });
  const { adapter, device } = await createWebGpuDevice(navigator.gpu, manifest.webgpu.requiredLimits);
  const { createOrtSession } = await import('../runtime/model/ort-session');
  const ort = await import('onnxruntime-web/jspi');
  let vocoder: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  try {
    send({ type: 'progress', stage: 'session', detail: 'Creating WebGPU vocoder session' });
    const sessionStarted = performance.now();
    vocoder = await createOrtSession(manifest.vocoder, release.cache, device);
    const sessionCreateMs = performance.now() - sessionStarted;
    const generationStarted = performance.now();
    const wav = await generateFixedVocoderWav(
      { ort, session: vocoder },
      analyticVocoderLatents(),
    );
    const generationMs = performance.now() - generationStarted;
    validateVocoderWav(wav);
    send({
      type: 'vocoder-result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        sessionCreateMs,
        generationMs,
        outputType: 'float32',
        shape: [1, 2, 220_160],
        finite: true,
        wavBytes: 880_684,
        sampleRate: 44_100,
        channels: 2,
        samples: 220_160,
        bitsPerSample: 16,
        artifactFetches,
        status: 'passed',
      },
    });
  } finally {
    await vocoder?.release();
    device.destroy();
  }
}

function validateVocoderWav(wav: ArrayBuffer) {
  const view = new DataView(wav);
  const text = (offset: number, length: number) =>
    String.fromCharCode(...new Uint8Array(wav, offset, length));
  if (
    wav.byteLength !== 880_684
    || text(0, 4) !== 'RIFF'
    || text(8, 4) !== 'WAVE'
    || view.getUint16(20, true) !== 1
    || view.getUint16(22, true) !== 2
    || view.getUint32(24, true) !== 44_100
    || view.getUint16(34, true) !== 16
    || view.getUint32(40, true) !== 880_640
  ) throw new Error('vocoder WAV structure is invalid');
}

async function runFlow(manifestUrl: string) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading flow release manifest' });
  const release = await readManifest(manifestUrl, 'Flow release manifest is unavailable');
  const manifest = parseFlowManifest(JSON.parse(release.text));
  const artifacts = [manifest.flow, ...manifest.flow.externalData];
  const artifactFetches = await cacheArtifacts(artifacts, release.base, release.cache);
  send({ type: 'progress', stage: 'adapter', detail: 'Requesting shader-f16 WebGPU device' });
  const { adapter, device } = await createWebGpuDevice(navigator.gpu, manifest.webgpu.requiredLimits);
  const { createOrtSession } = await import('../runtime/model/ort-session');
  const ort = await import('onnxruntime-web/jspi');
  let flow: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  try {
    send({ type: 'progress', stage: 'session', detail: 'Creating WebGPU flow transformer session' });
    const started = performance.now();
    flow = await createOrtSession(manifest.flow, release.cache, device);
    const sessionCreateMs = performance.now() - started;
    const metrics = await runFlowSmoke({ ort, session: flow });
    send({
      type: 'flow-result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        ...metrics,
        sessionCreateMs,
        artifactFetches,
        status: 'passed',
      },
    });
  } finally {
    await flow?.release();
    device.destroy();
  }
}

async function runCondition(manifestUrl: string) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading condition release manifest' });
  const release = await readManifest(manifestUrl, 'Condition release manifest is unavailable');
  const manifest = parseConditionManifest(JSON.parse(release.text));
  const artifacts = [manifest.conditionEncoder, ...manifest.conditionEncoder.externalData];
  const artifactFetches = await cacheArtifacts(artifacts, release.base, release.cache);
  send({ type: 'progress', stage: 'adapter', detail: 'Requesting shader-f16 WebGPU device' });
  const { adapter, device } = await createWebGpuDevice(navigator.gpu, manifest.webgpu.requiredLimits);
  const { createOrtSession } = await import('../runtime/model/ort-session');
  const ort = await import('onnxruntime-web/jspi');
  let conditionEncoder: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  try {
    send({ type: 'progress', stage: 'session', detail: 'Creating WebGPU condition encoder session' });
    const started = performance.now();
    conditionEncoder = await createOrtSession(manifest.conditionEncoder, release.cache, device);
    const sessionCreateMs = performance.now() - started;
    const metrics = await runConditionSmoke({ ort, session: conditionEncoder });
    send({
      type: 'condition-result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        ...metrics,
        sessionCreateMs,
        artifactFetches,
        status: 'passed',
      },
    });
  } finally {
    await conditionEncoder?.release();
    device.destroy();
  }
}

async function readManifest(url: string, unavailable: string) {
  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw new Error(unavailable);
  }
  if (!response.ok) throw new Error(unavailable);
  const text = await response.text();
  return {
    text,
    base: new URL(url, self.location.href),
    cache: await OpfsArtifactStore.open(await hashText(text)),
  };
}

async function cacheArtifacts(
  artifacts: readonly { path: string; bytes: number; sha256: string }[],
  base: URL,
  cache: OpfsArtifactStore,
) {
  let fetches = 0;
  for (const artifact of artifacts)
    await ensureArtifact(
      artifact,
      new URL(artifact.path, base),
      cache,
      ({ path, loaded, total }) =>
        send({ type: 'progress', stage: 'artifact', detail: path, loaded, total }),
      async (input, init) => {
        fetches++;
        return fetch(input, init);
      },
    );
  return fetches;
}

async function runFrameGeneration(request: Extract<WorkerRequest, { type: 'generate-frames' }>) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading Global and RVQ release manifests' });
  const globalRelease = await readManifest(request.globalManifestUrl, 'Global release manifest is unavailable');
  const rvqRelease = await readManifest(request.rvqManifestUrl, 'RVQ release manifest is unavailable');
  const globalManifest = parseModelManifest(JSON.parse(globalRelease.text));
  const rvqManifest = parseRvqStageManifest(JSON.parse(rvqRelease.text));
  const globalArtifacts = [
    globalManifest.graph,
    ...globalManifest.graph.externalData,
    globalManifest.reducedHead,
    ...globalManifest.reducedHead.externalData,
    ...globalManifest.embedding.shards,
  ];
  const rvqArtifacts = [
    rvqManifest.rvqDepth,
    ...rvqManifest.rvqDepth.externalData,
    rvqManifest.feedback,
    ...rvqManifest.feedback.externalData,
    ...rvqManifest.rvqEmbedding.shards,
  ];
  const artifactFetches =
    (await cacheArtifacts(globalArtifacts, globalRelease.base, globalRelease.cache)) +
    (await cacheArtifacts(rvqArtifacts, rvqRelease.base, rvqRelease.cache));
  send({ type: 'progress', stage: 'adapter', detail: 'Requesting shared shader-f16 WebGPU device' });
  const requiredLimits = { ...globalManifest.webgpu.requiredLimits };
  for (const [name, value] of Object.entries(rvqManifest.webgpu.requiredLimits))
    requiredLimits[name] = Math.max(requiredLimits[name] ?? 0, value);
  const { adapter, device } = await createWebGpuDevice(navigator.gpu, requiredLimits);
  const { createOrtSession } = await import('../runtime/model/ort-session');
  const ort = await import('onnxruntime-web/jspi');
  let decoder: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  let head: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  let rvqDepth: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  let feedback: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  try {
    send({ type: 'progress', stage: 'session', detail: 'Creating shared-device autoregressive sessions' });
    decoder = await createOrtSession(globalManifest.graph, globalRelease.cache, device);
    head = await createOrtSession(globalManifest.reducedHead, globalRelease.cache, device);
    rvqDepth = await createOrtSession(rvqManifest.rvqDepth, rvqRelease.cache, device);
    feedback = await createOrtSession(rvqManifest.feedback, rvqRelease.cache, device);
    const attemptedSeeds: number[] = [];
    let frames: Awaited<ReturnType<ReturnType<typeof createFrameGenerator>['generateFrames']>> | undefined;
    let cacheLengths: number[] = [];
    for (let attempt = 0; attempt < 2; attempt++) {
      const seed = request.seed + attempt;
      attemptedSeeds.push(seed);
      cacheLengths = [];
      try {
        frames = await createFrameGenerator({
          ort,
          decoder,
          head,
          rvqDepth,
          feedback,
          globalEmbedding: await OpfsFp16EmbeddingTable.open(
            globalManifest.embedding,
            (path) => globalRelease.cache.openSyncFile(path),
          ),
          rvqEmbedding: await OpfsFp16EmbeddingTable.open(
            rvqManifest.rvqEmbedding,
            (path) => rvqRelease.cache.openSyncFile(path),
          ),
          embeddingColumns: globalManifest.embedding.columns,
          kvPairs: globalManifest.kvPairs,
          readConditionalHidden: (tensor) => readConditionalGpuFp16(device, tensor),
          onCacheLength: (length) => cacheLengths.push(length),
        }).generateFrames({ maxFrames: request.maxFrames, seed, guidance: 1.5, topK: 50 });
        break;
      } catch (error) {
        if (!(error instanceof EarlyAudioEndError)) throw error;
        if (attempt === 1)
          throw new Error(`audio end sampled early for seeds ${attemptedSeeds.join(', ')}`);
        send({ type: 'progress', stage: 'session', detail: `${error.message}; retrying seed ${seed + 1}` });
      }
    }
    if (!frames) throw new Error(`audio end sampled early for seeds ${attemptedSeeds.join(', ')}`);
    const finiteHiddenGroups = frames.every((frame) => areFiniteFp16(frame.hiddenGroups));
    const codesInRange = frames.every(
      (frame) =>
        frame.semantic >= 0 &&
        frame.semantic < 16_384 &&
        frame.residual.every((code) => code >= 0 && code < 1_024),
    );
    send({
      type: 'frame-result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        frames: frames.length,
        attemptedSeeds,
        semanticDecisions: frames.length + 1,
        rvqCalls: (frames.length + 1) * 7,
        feedbackDecodes: frames.length,
        cacheLengths,
        finiteHiddenGroups,
        codesInRange,
        hiddenBytes: frames.reduce((total, frame) => total + frame.hiddenGroups.byteLength, 0),
        artifactFetches,
        status: 'passed',
      },
    });
  } finally {
    await feedback?.release();
    await rvqDepth?.release();
    await head?.release();
    await decoder?.release();
    device.destroy();
  }
}

async function runMusicGeneration(request: Extract<WorkerRequest, { type: 'generate-music-5s' }>) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading five-second music release manifest' });
  const release = await readManifest(request.manifestUrl, 'Music release manifest is unavailable');
  const manifestHash = await hashText(release.text);
  const manifest = parseMusicManifest(JSON.parse(release.text));
  const allArtifacts = [
    manifest.graph,
    ...manifest.graph.externalData,
    manifest.reducedHead,
    ...manifest.reducedHead.externalData,
    ...manifest.embedding.shards,
    manifest.rvqDepth,
    ...manifest.rvqDepth.externalData,
    manifest.feedback,
    ...manifest.feedback.externalData,
    ...manifest.rvqEmbedding.shards,
    manifest.conditionEncoder,
    ...manifest.conditionEncoder.externalData,
    manifest.flow,
    ...manifest.flow.externalData,
    manifest.vocoder,
    ...manifest.vocoder.externalData,
    ...manifest.tokenizerFiles,
    manifest.licenseFile,
  ];
  const artifacts = [...new Map(allArtifacts.map((artifact) => [artifact.path, artifact])).values()];
  const artifactFetches = await cacheArtifacts(artifacts, release.base, release.cache);
  const artifactBytes = artifacts.reduce((total, artifact) => total + artifact.bytes, 0);
  const { createOrtSession } = await import('../runtime/model/ort-session');
  const ort = await import('onnxruntime-web/jspi');
  const adapters: string[] = [];
  const sessionCreateMs = { autoregressive: 0, condition: 0, flow: 0, vocoder: 0 };
  const stageMs = { autoregressive: 0, condition: 0, flow: 0, vocoder: 0 };
  const inferenceMs = { autoregressive: 0, condition: 0, flow: 0, vocoder: 0 };
  const flowStepMs: number[] = [];
  const adapterName = (adapter: GPUAdapter) =>
    adapter.info.description || adapter.info.vendor || 'WebGPU adapter';

  const generated = await generateFiveSecondMusic(
    {
      async autoregressive(seed) {
        const stageStarted = performance.now();
        send({ type: 'progress', stage: 'autoregressive', detail: `Creating autoregressive sessions for seed ${seed}` });
        const { adapter, device } = await createWebGpuDevice(navigator.gpu, manifest.webgpu.requiredLimits);
        adapters.push(adapterName(adapter));
        let decoder: Awaited<ReturnType<typeof createOrtSession>> | undefined;
        let head: Awaited<ReturnType<typeof createOrtSession>> | undefined;
        let rvqDepth: Awaited<ReturnType<typeof createOrtSession>> | undefined;
        let feedback: Awaited<ReturnType<typeof createOrtSession>> | undefined;
        try {
          const sessionStarted = performance.now();
          decoder = await createOrtSession(manifest.graph, release.cache, device);
          head = await createOrtSession(manifest.reducedHead, release.cache, device);
          rvqDepth = await createOrtSession(manifest.rvqDepth, release.cache, device);
          feedback = await createOrtSession(manifest.feedback, release.cache, device);
          sessionCreateMs.autoregressive += performance.now() - sessionStarted;
          const generationStarted = performance.now();
          const frames = await createFrameGenerator({
            ort,
            decoder,
            head,
            rvqDepth,
            feedback,
            globalEmbedding: await OpfsFp16EmbeddingTable.open(
              manifest.embedding,
              (path) => release.cache.openSyncFile(path),
            ),
            rvqEmbedding: await OpfsFp16EmbeddingTable.open(
              manifest.rvqEmbedding,
              (path) => release.cache.openSyncFile(path),
            ),
            embeddingColumns: manifest.embedding.columns,
            kvPairs: manifest.kvPairs,
            readConditionalHidden: (tensor) => readConditionalGpuFp16(device, tensor),
          }).generateFrames({ maxFrames: 125, seed, guidance: 1.5, topK: 50 });
          inferenceMs.autoregressive += performance.now() - generationStarted;
          return frames;
        } finally {
          await feedback?.release();
          await rvqDepth?.release();
          await head?.release();
          await decoder?.release();
          device.destroy();
          stageMs.autoregressive += performance.now() - stageStarted;
        }
      },
      async condition(frameBits) {
        const stageStarted = performance.now();
        send({ type: 'progress', stage: 'condition', detail: 'Creating condition encoder session' });
        const { adapter, device } = await createWebGpuDevice(navigator.gpu, manifest.webgpu.requiredLimits);
        adapters.push(adapterName(adapter));
        let session: Awaited<ReturnType<typeof createOrtSession>> | undefined;
        const input = new ort.Tensor('float16', frameBits, [1, 125, 32_768]);
        let output: InstanceType<typeof ort.Tensor> | undefined;
        try {
          const sessionStarted = performance.now();
          session = await createOrtSession(manifest.conditionEncoder, release.cache, device);
          sessionCreateMs.condition = performance.now() - sessionStarted;
          const inferenceStarted = performance.now();
          const outputs = await session.run({ frame_hiddens: input });
          inferenceMs.condition = performance.now() - inferenceStarted;
          output = outputs.condition;
          if (!output) throw new Error('condition encoder did not return condition');
          return await readExactGpuFp16(device, output, [1, 430, 2048], 'condition');
        } finally {
          output?.dispose();
          input.dispose();
          await session?.release();
          device.destroy();
          stageMs.condition = performance.now() - stageStarted;
        }
      },
      async flow(conditionBits, seed, onStep) {
        const stageStarted = performance.now();
        send({ type: 'progress', stage: 'flow', detail: 'Creating flow transformer session' });
        const { adapter, device } = await createWebGpuDevice(navigator.gpu, manifest.webgpu.requiredLimits);
        adapters.push(adapterName(adapter));
        let session: Awaited<ReturnType<typeof createOrtSession>> | undefined;
        const condition = new ort.Tensor('float16', conditionBits, [1, 430, 2048]);
        let final: InstanceType<typeof ort.Tensor> | undefined;
        try {
          const sessionStarted = performance.now();
          session = await createOrtSession(manifest.flow, release.cache, device);
          sessionCreateMs.flow = performance.now() - sessionStarted;
          const initial = new ort.Tensor(
            'float16',
            deterministicGaussianFp16(seed, 128 * 430),
            [1, 128, 430],
          );
          let previous = performance.now();
          const inferenceStarted = performance.now();
          final = await runFixedFlowGeneration({ ort, session }, initial, condition, (completed) => {
            const now = performance.now();
            flowStepMs.push(now - previous);
            previous = now;
            onStep(completed);
          });
          inferenceMs.flow = performance.now() - inferenceStarted;
          return await readExactGpuFp16(device, final, [1, 128, 430], 'latents');
        } finally {
          final?.dispose();
          condition.dispose();
          await session?.release();
          device.destroy();
          stageMs.flow = performance.now() - stageStarted;
        }
      },
      async vocoder(latentBits) {
        const stageStarted = performance.now();
        send({ type: 'progress', stage: 'vocoder', detail: 'Creating vocoder session' });
        const { adapter, device } = await createWebGpuDevice(navigator.gpu, manifest.webgpu.requiredLimits);
        adapters.push(adapterName(adapter));
        let session: Awaited<ReturnType<typeof createOrtSession>> | undefined;
        try {
          const sessionStarted = performance.now();
          session = await createOrtSession(manifest.vocoder, release.cache, device);
          sessionCreateMs.vocoder = performance.now() - sessionStarted;
          const inferenceStarted = performance.now();
          const wav = await generateFixedVocoderWav({ ort, session }, latentBits);
          inferenceMs.vocoder = performance.now() - inferenceStarted;
          validateVocoderWav(wav);
          return wav;
        } finally {
          await session?.release();
          device.destroy();
          stageMs.vocoder = performance.now() - stageStarted;
        }
      },
    },
    request.seed,
    (progress) => {
      if (progress.stage === 'autoregressive')
        send({ type: 'progress', stage: 'autoregressive', detail: `Retained ${progress.retainedFrames} frames` });
      else if (progress.stage === 'flow')
        send({ type: 'progress', stage: 'flow', detail: `Completed flow step ${progress.completedSteps}/30` });
      else send({ type: 'progress', stage: progress.stage, detail: progress.stage === 'complete' ? 'Five-second music generation complete' : `Starting ${progress.stage} stage` });
    },
  );
  const result = {
    ...generated,
    adapters,
    wavBytes: generated.wav.byteLength,
    artifactBytes,
    artifactFetches,
    manifestHash,
    sessionCreateMs,
    stageMs,
    inferenceMs,
    flowStepMs,
    browser: navigator.userAgent,
    ortVersion: '1.30.0-dev.20260813-72e1c9c9b8',
    status: 'passed' as const,
  };
  send({ type: 'music-result', result }, [generated.wav]);
}

async function runRvq(manifestUrl: string) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading RVQ release manifest' });
  let response: Response;
  try {
    response = await fetch(manifestUrl);
  } catch {
    throw new Error('RVQ release manifest is unavailable');
  }
  if (!response.ok) throw new Error('RVQ release manifest is unavailable');
  const manifestText = await response.text();
  const manifest = parseRvqStageManifest(JSON.parse(manifestText));
  const cache = await OpfsArtifactStore.open(await hashText(manifestText));
  const base = new URL(manifestUrl, self.location.href);
  const artifacts = [
    manifest.rvqDepth,
    ...manifest.rvqDepth.externalData,
    manifest.feedback,
    ...manifest.feedback.externalData,
    ...manifest.rvqEmbedding.shards,
  ];
  let artifactFetches = 0;
  for (const artifact of artifacts)
    await ensureArtifact(
      artifact,
      new URL(artifact.path, base),
      cache,
      ({ path, loaded, total }) => send({ type: 'progress', stage: 'artifact', detail: path, loaded, total }),
      async (input, init) => {
        artifactFetches++;
        return fetch(input, init);
      },
    );
  send({ type: 'progress', stage: 'adapter', detail: 'Requesting shader-f16 WebGPU device' });
  const { adapter, device } = await createWebGpuDevice(navigator.gpu, manifest.webgpu.requiredLimits);
  const { createOrtSession } = await import('../runtime/model/ort-session');
  const ort = await import('onnxruntime-web/jspi');
  let rvqDepth: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  let feedback: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  try {
    send({ type: 'progress', stage: 'session', detail: 'Creating WebGPU RVQ and feedback sessions' });
    const started = performance.now();
    rvqDepth = await createOrtSession(manifest.rvqDepth, cache, device);
    feedback = await createOrtSession(manifest.feedback, cache, device);
    const sessionCreateMs = performance.now() - started;
    const metrics = await runRvqSmoke({
      ort,
      rvqDepth,
      feedback,
      embedding: await OpfsFp16EmbeddingTable.open(
        manifest.rvqEmbedding,
        (path) => cache.openSyncFile(path),
      ),
      embeddingTable: manifest.rvqEmbedding,
    });
    send({
      type: 'rvq-result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        ...metrics,
        sessionCreateMs,
        artifactFetches,
        status: 'passed',
      },
    });
  } finally {
    await feedback?.release();
    await rvqDepth?.release();
    device.destroy();
  }
}
