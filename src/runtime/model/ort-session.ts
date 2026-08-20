import * as ort from 'onnxruntime-web/jspi';
import type { ArtifactStore } from './artifact-cache';
import type { OnnxGraphArtifact } from './manifest';

const wasmUrl = new URL('/ort/', self.location.origin).toString();

export async function createOrtSession(graph: OnnxGraphArtifact, cache: ArtifactStore, device: GPUDevice): Promise<ort.InferenceSession> {
  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = wasmUrl;
  ort.env.webgpu.device = device;
  const model = new Uint8Array(await (await cache.file(graph.path)).arrayBuffer());
  const externalData = await Promise.all(graph.externalData.map(async (item) => ({ path: item.onnxLocation, data: await cache.file(item.path) })));
  return ort.InferenceSession.create(model, {
    externalData,
    executionProviders: [{ name: 'webgpu', device, storageBufferCacheMode: 'disabled', uniformBufferCacheMode: 'disabled', queryResolveBufferCacheMode: 'disabled', defaultBufferCacheMode: 'disabled' }],
    executionMode: 'sequential',
    graphOptimizationLevel: 'all',
    enableMemPattern: false,
    enableGraphCapture: false,
    preferredOutputLocation: Object.fromEntries(graph.gpuOutputs.map((name) => [name, 'gpu-buffer'])),
    extra: { session: { disable_cpu_ep_fallback: '1', disable_prepacking: '1' } },
  });
}
