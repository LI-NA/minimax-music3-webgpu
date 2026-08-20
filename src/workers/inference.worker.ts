/// <reference lib="webworker" />
import { OpfsArtifactStore, ensureArtifact } from '../runtime/model/artifact-cache';
import { parseModelManifest } from '../runtime/model/manifest';
import { createOrtSession } from '../runtime/model/ort-session';
import { createWebGpuDevice } from '../runtime/model/webgpu-device';
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
  const response = await fetch(request.manifestUrl);
  if (!response.ok) throw new Error('Unable to read release manifest');
  const manifestText = await response.text();
  const manifest = parseModelManifest(JSON.parse(manifestText));
  const cache = await OpfsArtifactStore.open(await hashText(manifestText));
  const base = new URL(request.manifestUrl, self.location.href);
  const artifacts = [
    manifest.graph,
    ...manifest.graph.externalData,
    manifest.reducedHead,
    ...manifest.reducedHead.externalData,
  ];
  for (const artifact of artifacts)
    await ensureArtifact(artifact, new URL(artifact.path, base), cache, ({ path, loaded, total }) =>
      send({
        type: 'progress',
        stage: 'artifact',
        detail: path,
        loaded,
        total,
      }),
    );
  send({
    type: 'progress',
    stage: 'adapter',
    detail: 'Requesting shader-f16 WebGPU device',
  });
  const { adapter, device } = await createWebGpuDevice(navigator.gpu);
  send({
    type: 'progress',
    stage: 'session',
    detail: 'Creating WebGPU decoder session',
  });
  const graph = await createOrtSession(manifest.graph, cache, device);
  const head = await createOrtSession(manifest.reducedHead, cache, device);
  send({
    type: 'result',
    result: {
      adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
      graphInputs: graph.inputNames,
      graphOutputs: graph.outputNames,
      reducedHeadOutputs: head.outputNames,
      status: 'ready',
    },
  });
}
