export type WebGpuCapability = { supported: true; adapter: GPUAdapter } | { supported: false; reason: string };

export async function inspectWebGpu(gpu: GPU | undefined): Promise<WebGpuCapability> {
  if (!gpu) return { supported: false, reason: 'WebGPU is unavailable' };
  const adapter = await gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) return { supported: false, reason: 'No WebGPU adapter was found' };
  if (adapter.info.isFallbackAdapter) return { supported: false, reason: 'A fallback adapter is not supported' };
  if (!adapter.features.has('shader-f16')) return { supported: false, reason: 'shader-f16 is unavailable' };
  return { supported: true, adapter };
}

export async function inspectWebGpuForRequirements(
  gpu: GPU | undefined,
  requiredLimits: Record<string, number>,
): Promise<WebGpuCapability> {
  const capability = await inspectWebGpu(gpu);
  if (!capability.supported) return capability;
  const limits = capability.adapter.limits as unknown as Record<string, number>;
  if (Object.entries(requiredLimits).some(([name, value]) => typeof limits[name] !== 'number' || limits[name] < value))
    return { supported: false, reason: 'Adapter limits are insufficient' };
  return capability;
}
