/// <reference lib="webworker" />
import { OpfsArtifactStore, ensureArtifact } from '../runtime/model/artifact-cache';
import { OpfsFp16EmbeddingTable } from '../runtime/model/embedding-table';
import { parseModelManifest } from '../runtime/model/manifest';
import { createWebGpuDevice } from '../runtime/model/webgpu-device';
import { runGlobalSmoke } from '../runtime/pipeline/global-smoke';
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
