import { describe, expect, it, vi } from 'vitest';
import { inspectWebGpu } from '../../../src/runtime/model/webgpu-device';

describe('inspectWebGpu', () => {
  it('rejects an adapter without shader-f16', async () => {
    const adapter = {
      features: new Set<string>(),
      info: { isFallbackAdapter: false },
      limits: { maxStorageBufferBindingSize: 1_000_000_000 },
    };
    const gpu = { requestAdapter: vi.fn().mockResolvedValue(adapter) } as unknown as GPU;

    await expect(inspectWebGpu(gpu)).resolves.toEqual({
      supported: false,
      reason: 'shader-f16 is unavailable',
    });
  });
});
