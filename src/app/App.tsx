import { useEffect, useRef, useState } from 'react';
import { inspectWebGpu, type WebGpuCapability } from '../runtime/model/webgpu-device';
import type { GlobalSmokeResult, RvqSmokeResult, WorkerResponse } from '../workers/protocol';

type CapabilityState = WebGpuCapability | null;

export function App() {
  const [capability, setCapability] = useState<CapabilityState>(null);
  const [progress, setProgress] = useState('Awaiting diagnostic command');
  const [result, setResult] = useState<GlobalSmokeResult | null>(null);
  const [rvqResult, setRvqResult] = useState<RvqSmokeResult | null>(null);
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
      } else if (data.type === 'error') setProgress(`Error: ${data.message}`);
    };
    next.postMessage({
      type: 'run-global-smoke',
      manifestUrl: 'http://127.0.0.1:5174/manifest.json',
    });
  };
  const runRvq = () => {
    cancel();
    setRvqResult(null);
    setProgress('Starting isolated RVQ runtime worker');
    const next = new Worker(new URL('../workers/inference.worker.ts', import.meta.url), { type: 'module' });
    worker.current = next;
    next.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === 'progress') setProgress(`${data.stage}: ${data.detail}`);
      else if (data.type === 'rvq-result') {
        setRvqResult(data.result);
        setProgress('RVQ runtime passed');
      } else if (data.type === 'error') setProgress(`Error: ${data.message}`);
    };
    next.postMessage({
      type: 'run-rvq-smoke',
      manifestUrl: 'http://127.0.0.1:5174/manifest.json',
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
            Run Global LLM smoke
          </button>
          <button type="button" disabled={!capability?.supported} onClick={runRvq}>
            Run RVQ depth smoke
          </button>
          <button type="button" className="secondary" disabled={!worker.current} onClick={cancel}>
            Cancel worker
          </button>
        </div>
        <output aria-live="polite" className="progress">
          {progress}
        </output>
        {result && (
          <pre data-testid="global-smoke-result" className="result">
            {`steps: ${result.stepMs.length - 1}\nfinite logits: ${result.finiteLogits ? 'yes' : 'no'}\nKV location: ${result.tensorLocations.every((location) => location === 'gpu-buffer') ? 'gpu-buffer' : 'non-GPU'}\n${JSON.stringify(result, null, 2)}`}
          </pre>
        )}
        {rvqResult && (
          <pre data-testid="rvq-smoke-result" className="result">
            {`lengths: ${rvqResult.lengths.join(', ')}\nfinite logits: ${rvqResult.finiteLogits ? 'yes' : 'no'}\nhidden location: ${rvqResult.hiddenLocations.every((location) => location === 'gpu-buffer') ? 'gpu-buffer' : 'non-GPU'}\nfeedback location: ${rvqResult.feedbackLocation}\n${JSON.stringify(rvqResult, null, 2)}`}
          </pre>
        )}
      </section>
    </main>
  );
}
