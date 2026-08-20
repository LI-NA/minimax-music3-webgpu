import { useEffect, useRef, useState } from 'react';
import { inspectWebGpu, type WebGpuCapability } from '../runtime/model/webgpu-device';
import type { GlobalSmokeResult, WorkerResponse } from '../workers/protocol';

type CapabilityState = WebGpuCapability | null;

export function App() {
  const [capability, setCapability] = useState<CapabilityState>(null);
  const [progress, setProgress] = useState('Awaiting diagnostic command');
  const [result, setResult] = useState<GlobalSmokeResult | null>(null);
  const worker = useRef<Worker | null>(null);

  useEffect(() => {
    void inspectWebGpu(navigator.gpu).then(setCapability);
  }, []);

  const cancel = () => {
    worker.current?.terminate();
    worker.current = null;
    setProgress('Diagnostic cancelled');
  };
  const run = () => {
    cancel();
    setResult(null);
    setProgress('Starting isolated runtime worker');
    const next = new Worker(new URL('../workers/inference.worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.current = next;
    next.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === 'progress') setProgress(`${data.stage}: ${data.detail}`);
      else if (data.type === 'result') {
        setResult(data.result);
        setProgress('Runtime ready');
      } else setProgress(`Error: ${data.message}`);
    };
    next.postMessage({
      type: 'run-global-smoke',
      manifestUrl: '/artifacts/manifest.json',
    });
  };

  const status =
    capability === null
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
        <div className="command-row">
          <button type="button" disabled={!capability?.supported} onClick={run}>
            Run runtime diagnostic
          </button>
          <button type="button" className="secondary" disabled={!worker.current} onClick={cancel}>
            Cancel worker
          </button>
        </div>
        <output aria-live="polite" className="progress">
          {progress}
        </output>
        {result && <pre className="result">{JSON.stringify(result, null, 2)}</pre>}
      </section>
    </main>
  );
}
