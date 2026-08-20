import { beforeEach, describe, expect, it, vi } from 'vitest';

const create = vi.fn();

vi.mock('onnxruntime-web/jspi', () => ({
  env: { wasm: {}, webgpu: {} },
  InferenceSession: { create },
}));

describe('createOrtSession', () => {
  beforeEach(() => {
    create.mockReset().mockResolvedValue({});
    vi.stubGlobal('self', { location: { origin: 'http://127.0.0.1:5173' } });
  });

  it('uses the environment GPU device without passing a device through the WebGPU provider', async () => {
    const { createOrtSession } = await import('../../../src/runtime/model/ort-session');
    const device = {} as GPUDevice;
    await createOrtSession(
      { path: 'global.onnx', bytes: 1, sha256: 'a'.repeat(64), externalData: [], gpuOutputs: [] },
      { file: async () => new File([new Uint8Array([0])], 'global.onnx') } as never,
      device,
    );

    const options = create.mock.calls[0][1];
    expect(options.executionProviders).toEqual(['webgpu']);
    expect(options.executionProviders[0]).not.toHaveProperty('device');
  });
});
