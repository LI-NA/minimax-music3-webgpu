export type WebGpuCapability =
  | { supported: true; adapter: GPUAdapter }
  | { supported: false; reason: string };

export async function inspectWebGpu(gpu: GPU | undefined): Promise<WebGpuCapability> {
  if (!gpu) return { supported: false, reason: 'WebGPU is unavailable' };
  const adapter = await gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) return { supported: false, reason: 'No WebGPU adapter was found' };
  if (adapter.info.isFallbackAdapter)
    return { supported: false, reason: 'A fallback adapter is not supported' };
  if (!adapter.features.has('shader-f16'))
    return { supported: false, reason: 'shader-f16 is unavailable' };
  return { supported: true, adapter };
}

export async function createWebGpuDevice(
  gpu: GPU,
  requiredLimits: Record<string, number> = {},
): Promise<{ adapter: GPUAdapter; device: GPUDevice }> {
  const capability = await inspectWebGpu(gpu);
  if (!capability.supported) throw new Error(capability.reason);
  const requiredFeatures: GPUFeatureName[] = ['shader-f16'];
  const limits = capability.adapter.limits as unknown as Record<string, number>;
  const required = Object.fromEntries(
    Object.entries(requiredLimits).filter(
      ([name, value]) => typeof limits[name] === 'number' && limits[name] >= value,
    ),
  );
  if (Object.keys(required).length !== Object.keys(requiredLimits).length)
    throw new Error('Adapter limits are insufficient');
  return {
    adapter: capability.adapter,
    device: await capability.adapter.requestDevice({
      requiredFeatures,
      requiredLimits: required,
    }),
  };
}
