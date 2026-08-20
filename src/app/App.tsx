import { useEffect, useRef, useState } from 'react';
import { inspectWebGpu, type WebGpuCapability } from '../runtime/model/webgpu-device';
import type {
  ConditionSmokeResult,
  FlowSmokeResult,
  FrameGenerationResult,
  GlobalSmokeResult,
  RvqSmokeResult,
  WorkerResponse,
} from '../workers/protocol';

type CapabilityState = WebGpuCapability | null;

export function App() {
  const [capability, setCapability] = useState<CapabilityState>(null);
  const [progress, setProgress] = useState('Awaiting diagnostic command');
  const [result, setResult] = useState<GlobalSmokeResult | null>(null);
  const [rvqResult, setRvqResult] = useState<RvqSmokeResult | null>(null);
  const [frameResult, setFrameResult] = useState<FrameGenerationResult | null>(null);
  const [conditionResult, setConditionResult] = useState<ConditionSmokeResult | null>(null);
  const [flowResult, setFlowResult] = useState<FlowSmokeResult | null>(null);
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
  const generateFrames = () => {
    cancel();
    setFrameResult(null);
    const parsed = Number(new URLSearchParams(window.location.search).get('frames') ?? '2');
    const maxFrames = Number.isInteger(parsed) && parsed > 0 ? parsed : 2;
    setProgress(`Generating ${maxFrames} RVQ frames`);
    const next = new Worker(new URL('../workers/inference.worker.ts', import.meta.url), { type: 'module' });
    worker.current = next;
    next.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === 'progress') setProgress(`${data.stage}: ${data.detail}`);
      else if (data.type === 'frame-result') {
        setFrameResult(data.result);
        setProgress('RVQ frame generation passed');
      } else if (data.type === 'error') setProgress(`Error: ${data.message}`);
    };
    next.postMessage({
      type: 'generate-frames',
      globalManifestUrl: 'http://127.0.0.1:5174/manifest.json',
      rvqManifestUrl: 'http://127.0.0.1:5175/manifest.json',
      maxFrames,
      seed: 7,
    });
  };
  const runCondition = () => {
    cancel();
    setConditionResult(null);
    setProgress('Starting isolated condition runtime worker');
    const next = new Worker(new URL('../workers/inference.worker.ts', import.meta.url), { type: 'module' });
    worker.current = next;
    next.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === 'progress') setProgress(`${data.stage}: ${data.detail}`);
      else if (data.type === 'condition-result') {
        setConditionResult(data.result);
        setProgress('Condition encoder runtime passed');
      } else if (data.type === 'error') setProgress(`Error: ${data.message}`);
    };
    next.postMessage({
      type: 'run-condition-smoke',
      manifestUrl: 'http://127.0.0.1:5176/manifest.json',
    });
  };
  const runFlow = () => {
    cancel();
    setFlowResult(null);
    setProgress('Starting isolated flow runtime worker');
    const next = new Worker(new URL('../workers/inference.worker.ts', import.meta.url), { type: 'module' });
    worker.current = next;
    next.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === 'progress') setProgress(`${data.stage}: ${data.detail}`);
      else if (data.type === 'flow-result') {
        setFlowResult(data.result);
        setProgress('Flow transformer runtime passed');
      } else if (data.type === 'error') setProgress(`Error: ${data.message}`);
    };
    next.postMessage({
      type: 'run-flow-smoke',
      manifestUrl: 'http://127.0.0.1:5177/manifest.json',
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
          <button type="button" disabled={!capability?.supported} onClick={generateFrames}>
            Generate RVQ frames
          </button>
          <button type="button" disabled={!capability?.supported} onClick={runCondition}>
            Run condition encoder smoke
          </button>
          <button type="button" disabled={!capability?.supported} onClick={runFlow}>
            Run flow transformer smoke
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
        {frameResult && (
          <pre data-testid="frame-generation-result" className="result">
            {`frames: ${frameResult.frames}\nsemantic decisions: ${frameResult.semanticDecisions}\nRVQ calls: ${frameResult.rvqCalls}\nfeedback decodes: ${frameResult.feedbackDecodes}\ncache lengths: ${frameResult.cacheLengths.join(', ')}\nfinite hidden groups: ${frameResult.finiteHiddenGroups ? 'yes' : 'no'}\ncodes in range: ${frameResult.codesInRange ? 'yes' : 'no'}\n${JSON.stringify(frameResult, null, 2)}`}
          </pre>
        )}
        {conditionResult && (
          <pre data-testid="condition-smoke-result" className="result">
            {`shape: ${conditionResult.shape.join(', ')}\noutput location: ${conditionResult.outputLocation}\nfinite: ${conditionResult.finite ? 'yes' : 'no'}\n${JSON.stringify(conditionResult, null, 2)}`}
          </pre>
        )}
        {flowResult && (
          <pre data-testid="flow-smoke-result" className="result">
            {`one-step shape: ${flowResult.shape.join(', ')}\none-step location: ${flowResult.oneStepLocation}\none-step finite: ${flowResult.oneStepFinite ? 'yes' : 'no'}\nsteps: ${flowResult.stepMs.length}\nfinal location: ${flowResult.finalLocation}\nfinal finite: ${flowResult.finalFinite ? 'yes' : 'no'}\n${JSON.stringify(flowResult, null, 2)}`}
          </pre>
        )}
      </section>
    </main>
  );
}
