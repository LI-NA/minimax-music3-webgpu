import { describe, expect, it, vi } from 'vitest';
import {
  inspectWebGpu,
  inspectWebGpuForRequirements,
} from '../../../src/runtime/model/webgpu-device';

describe('inspectWebGpu', () => {
  it('rejects an adapter without shader-f16', async () => {
    const adapter = {
      features: new Set<string>(),
      info: { isFallbackAdapter: false },
      limits: { maxStorageBufferBindingSize: 1_000_000_000 },
    };
    const gpu = {
      requestAdapter: vi.fn().mockResolvedValue(adapter),
    } as unknown as GPU;

    await expect(inspectWebGpu(gpu)).resolves.toEqual({
      supported: false,
      reason: 'shader-f16 is unavailable',
    });
  });

  it('returns a capable adapter without requesting a device', async () => {
    const requestDevice = vi.fn();
    const adapter = {
      features: new Set<string>(['shader-f16']),
      info: { isFallbackAdapter: false },
      limits: {
        maxStorageBufferBindingSize: 1_000_000_000,
        maxStorageBuffersPerShaderStage: 9,
      },
      requestDevice,
    };
    const requestAdapter = vi.fn().mockResolvedValue(adapter);
    const gpu = { requestAdapter } as unknown as GPU;

    await expect(inspectWebGpuForRequirements(gpu, {
      maxStorageBufferBindingSize: 1_000_000_000,
      maxStorageBuffersPerShaderStage: 9,
    })).resolves.toEqual({ supported: true, adapter });
    expect(requestAdapter).toHaveBeenCalledWith({ powerPreference: 'high-performance' });
    expect(requestDevice).not.toHaveBeenCalled();
  });

  it('rejects the adapter when any declared required limit is insufficient', async () => {
    const requestDevice = vi.fn();
    const adapter = {
      features: new Set<string>(['shader-f16']),
      info: { isFallbackAdapter: false },
      limits: {
        maxStorageBufferBindingSize: 1_000_000_000,
        maxStorageBuffersPerShaderStage: 8,
      },
      requestDevice,
    };
    const gpu = {
      requestAdapter: vi.fn().mockResolvedValue(adapter),
    } as unknown as GPU;

    await expect(inspectWebGpuForRequirements(gpu, {
      maxStorageBufferBindingSize: 1_000_000_000,
      maxStorageBuffersPerShaderStage: 9,
    })).resolves.toEqual({
      supported: false,
      reason: 'Adapter limits are insufficient',
    });
    expect(requestDevice).not.toHaveBeenCalled();
  });
});
