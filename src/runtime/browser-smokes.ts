import * as ort from 'onnxruntime-web/jspi';
import { localJspiWasmPaths } from './model/local-jspi-path';

async function device() {
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter || adapter.info.isFallbackAdapter || !adapter.features.has('shader-f16'))
    throw new Error('required WebGPU adapter unavailable');
  return adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
}
function configure(next: GPUDevice) {
  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = localJspiWasmPaths(location.origin);
  ort.env.webgpu.device = next;
}
export async function runExternalDataOpfsSmoke() {
  const root = await navigator.storage.getDirectory();
  const directory = await root.getDirectoryHandle('external-smoke', {
    create: true,
  });
  for (const path of ['external-add.onnx', 'external-add.bin']) {
    const data = await (await fetch(`/test-fixtures/${path}`)).arrayBuffer();
    const writable = await (await directory.getFileHandle(path, { create: true })).createWritable();
    await writable.write(data);
    await writable.close();
  }
  const model = await (await directory.getFileHandle('external-add.onnx')).getFile();
  const external = await (await directory.getFileHandle('external-add.bin')).getFile();
  Object.defineProperty(external, 'arrayBuffer', {
    value: () => {
      throw new Error('external File must be range-read');
    },
  });
  configure(await device());
  const session = await ort.InferenceSession.create(new Uint8Array(await model.arrayBuffer()), {
    externalData: [{ path: 'external-add.bin', data: external }],
    executionProviders: ['webgpu'],
    extra: { session: { disable_cpu_ep_fallback: '1' } },
  });
  const result = await session.run({
    input: new ort.Tensor('float32', new Float32Array([3]), [1]),
  });
  return (result.output.data as Float32Array)[0];
}
export async function runMatMulNBitsSmoke() {
  configure(await device());
  const model = new Uint8Array(
    await (await fetch('/test-fixtures/matmul-nbits.onnx')).arrayBuffer(),
  );
  const session = await ort.InferenceSession.create(model, {
    executionProviders: ['webgpu'],
    extra: { session: { disable_cpu_ep_fallback: '1' } },
  });
  const result = await session.run({
    input: new ort.Tensor('float16', new Uint16Array(128).fill(0x3c00), [1, 128]),
  });
  const data = result.output.data;
  if (!(data instanceof Uint16Array)) return Array.from(data as Float32Array);
  return Array.from(data).map((value) => {
    const exponent = (value >>> 10) & 31;
    const fraction = value & 1023;
    return (
      (value >>> 15 ? -1 : 1) *
      (exponent ? (1 + fraction / 1024) * 2 ** (exponent - 15) : fraction * 2 ** -24)
    );
  });
}
