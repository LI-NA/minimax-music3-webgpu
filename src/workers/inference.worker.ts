/// <reference lib="webworker" />
import { OpfsArtifactStore, ensureArtifact } from '../runtime/model/artifact-cache';
import { OpfsFp16EmbeddingTable } from '../runtime/model/embedding-table';
import { parseModelManifest, parseRvqStageManifest } from '../runtime/model/manifest';
import { createWebGpuDevice } from '../runtime/model/webgpu-device';
import { runGlobalSmoke } from '../runtime/pipeline/global-smoke';
import { runRvqSmoke } from '../runtime/pipeline/rvq-smoke';
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
