import { beforeEach, describe, expect, it, vi } from 'vitest';

const create = vi.fn();
const webgpu = {} as {
  device?: GPUDevice;
  powerPreference?: 'low-power' | 'high-performance';
};

vi.mock('onnxruntime-web/jspi', () => ({
  env: { wasm: {}, webgpu },
  InferenceSession: { create },
}));

describe('createOrtSession', () => {
  beforeEach(() => {
    create.mockReset().mockResolvedValue({});
    delete webgpu.device;
    delete webgpu.powerPreference;
    vi.stubGlobal('self', { location: { origin: 'http://127.0.0.1:5173' } });
  });

  it('lets ORT own the WebGPU device without passing or assigning one', async () => {
    const { createOrtSession } = await import('../../../src/runtime/model/ort-session');
    await createOrtSession(
      { path: 'global.onnx', bytes: 1, sha256: 'a'.repeat(64), externalData: [], gpuOutputs: [] },
      { file: async () => new File([new Uint8Array([0])], 'global.onnx') } as never,
    );

    const options = create.mock.calls[0][1];
    expect(webgpu).not.toHaveProperty('device');
    expect(webgpu.powerPreference).toBe('high-performance');
    expect(options.executionProviders).toEqual(['webgpu']);
  });
});
