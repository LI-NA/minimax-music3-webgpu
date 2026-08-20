import { useEffect, useState } from 'react';
import { inspectWebGpu, type WebGpuCapability } from '../runtime/model/webgpu-device';

type CapabilityState = WebGpuCapability | null;

export function App() {
  const [capability, setCapability] = useState<CapabilityState>(null);

  useEffect(() => {
    void inspectWebGpu(navigator.gpu).then(setCapability);
  }, []);

  const status = capability === null
    ? 'Checking WebGPU capability…'
    : capability.supported
      ? 'WebGPU with shader-f16 is available.'
      : capability.reason;

  return (
    <main className="app-shell">
      <section aria-labelledby="page-title" className="diagnostic-card">
        <p className="eyebrow">MiniMax Music 3</p>
        <h1 id="page-title">WebGPU feasibility diagnostic</h1>
        <p aria-live="polite" className={capability?.supported ? 'status supported' : 'status'}>
          {status}
        </p>
      </section>
    </main>
  );
}
