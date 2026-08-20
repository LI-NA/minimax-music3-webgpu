/// <reference lib="webworker" />
import { OpfsArtifactStore, ensureArtifact } from '../runtime/model/artifact-cache';
import { OpfsFp16EmbeddingTable } from '../runtime/model/embedding-table';
import { parseModelManifest, parseRvqStageManifest } from '../runtime/model/manifest';
import { createWebGpuDevice } from '../runtime/model/webgpu-device';
import { runGlobalSmoke } from '../runtime/pipeline/global-smoke';
import { runRvqSmoke } from '../runtime/pipeline/rvq-smoke';
import {
  areFiniteFp16,
  createFrameGenerator,
  EarlyAudioEndError,
  readConditionalGpuFp16,
} from '../runtime/pipeline/rvq-generation';
import type { WorkerRequest, WorkerResponse } from './protocol';

const send = (message: WorkerResponse) => self.postMessage(message);
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
