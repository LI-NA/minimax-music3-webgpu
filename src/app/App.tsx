import { useEffect, useReducer, useRef, useState } from 'react';
import {
  requestPersistentStorage,
} from '../runtime/model/artifact-cache-management';
import { inspectWebGpu, type WebGpuCapability } from '../runtime/model/webgpu-device';
import {
  cancelWorker,
  progressView,
  type ProgressView,
} from '../workers/music-progress';
import type {
  ConditionSmokeResult,
  FlowSmokeResult,
  FrameGenerationResult,
  GlobalSmokeResult,
  AnyMusicGenerationWorkerResult,
  RvqSmokeResult,
  VocoderSmokeResult,
  WorkerResponse,
} from '../workers/protocol';
import type { ArtifactCacheRequest, ArtifactOperation } from '../workers/protocol';
import {
  createMusicGenerationRequest,
} from '../workers/protocol';
import { FIXED_COMPARISON_CASE } from '../runtime/reference/fixed-comparison';
import {
  artifactCacheUiReducer,
  createArtifactCacheUiState,
  describeArtifactCacheStatus,
  deriveArtifactCacheControls,
  formatBytes,
  formatEta,
  formatRate,
  type ArtifactCacheRetryTarget,
  type ArtifactCacheUiOperation,
} from './artifact-cache-ui';

type CapabilityState = WebGpuCapability | null;
const PRODUCT_DURATIONS = Array.from({ length: 60 }, (_, index) => (index + 1) * 5);
const PRODUCT_MANIFEST_URL = 'http://127.0.0.1:5174/manifest.json';
const durationLabel = (durationSeconds: number) =>
  durationSeconds === 5 ? 'five-second' : `${durationSeconds}-second`;

export function createProductMusicRequest(durationSeconds: number) {
  return createMusicGenerationRequest({
    manifestUrl: PRODUCT_MANIFEST_URL,
    prompt: FIXED_COMPARISON_CASE.input.prompt,
    lyrics: FIXED_COMPARISON_CASE.input.lyrics,
    seed: 7,
    durationSeconds,
    sampling: {
      globalGuidance: 1.5,
      semanticTopK: 50,
      residualTopK: 50,
      temperature: 1,
      flowGuidance: 1.7,
      flowSteps: 30,
    },
  });
}

export function App() {
  const [capability, setCapability] = useState<CapabilityState>(null);
  const [progress, setProgress] = useState('Awaiting diagnostic command');
  const [result, setResult] = useState<GlobalSmokeResult | null>(null);
  const [rvqResult, setRvqResult] = useState<RvqSmokeResult | null>(null);
  const [frameResult, setFrameResult] = useState<FrameGenerationResult | null>(null);
  const [conditionResult, setConditionResult] = useState<ConditionSmokeResult | null>(null);
  const [flowResult, setFlowResult] = useState<FlowSmokeResult | null>(null);
  const [vocoderResult, setVocoderResult] = useState<VocoderSmokeResult | null>(null);
  const [musicResult, setMusicResult] = useState<AnyMusicGenerationWorkerResult | null>(null);
  const [musicProgress, setMusicProgress] = useState<ProgressView | null>(null);
  const [durationSeconds, setDurationSeconds] = useState(5);
  const [musicIsRunning, setMusicIsRunning] = useState(false);
  const [musicUrl, setMusicUrl] = useState<string | null>(null);
  const [cacheState, dispatchCache] = useReducer(
    artifactCacheUiReducer,
    null,
    createArtifactCacheUiState,
  );
  const musicUrlRef = useRef<string | null>(null);
  const worker = useRef<Worker | null>(null);
  const cacheWorker = useRef<Worker | null>(null);
  const musicRunning = useRef(false);

  const finishCacheWorker = (activeWorker: Worker) => {
    activeWorker.terminate();
    if (cacheWorker.current === activeWorker) cacheWorker.current = null;
  };

  const retryTarget = (
    operation: ArtifactCacheUiOperation,
    protocolOperation?: ArtifactOperation,
  ): ArtifactCacheRetryTarget => {
    if (protocolOperation === 'download-artifacts' || operation === 'download') return 'download';
    if (protocolOperation === 'delete-artifact-caches' || operation === 'delete') return 'delete';
    return 'inspect';
  };

  const runCacheWorker = (
    request: ArtifactCacheRequest,
    operation: Exclude<ArtifactCacheUiOperation, null | 'request-persistence'>,
  ) => {
    cacheWorker.current?.terminate();
    const next = new Worker(new URL('../workers/inference.worker.ts', import.meta.url), {
      type: 'module',
    });
    cacheWorker.current = next;
    next.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === 'progress') {
        if (operation === 'download' && data.stage === 'artifact') {
          dispatchCache({ type: 'progress-received', progress: data });
        }
        return;
      }
      if (
        data.type === 'artifact-cache-status'
        || data.type === 'artifact-download-complete'
        || data.type === 'artifact-cache-deleted'
      ) {
        const source = data.type === 'artifact-cache-status'
          ? 'inspect'
          : data.type === 'artifact-download-complete'
            ? 'download'
            : 'delete';
        dispatchCache({ type: 'status-received', source, status: data.status });
        finishCacheWorker(next);
        return;
      }
      if (data.type === 'error') {
        dispatchCache({
          type: 'operation-failed',
          error: {
            message: data.message,
            code: data.code,
            operation: data.operation,
            retryable: data.retryable === true,
            retryTarget: retryTarget(operation, data.operation),
          },
        });
        finishCacheWorker(next);
        if (operation === 'download' || operation === 'delete') inspectCache();
      }
    };
    next.postMessage(request);
  };

  const inspectCache = () => {
    dispatchCache({ type: 'operation-started', operation: 'inspect' });
    runCacheWorker({
      type: 'inspect-artifact-cache',
      manifestUrl: PRODUCT_MANIFEST_URL,
    }, 'inspect');
  };

  useEffect(() => {
    void inspectWebGpu(navigator.gpu).then(setCapability);
    inspectCache();
    return () => {
      cacheWorker.current?.terminate();
      if (musicUrlRef.current) URL.revokeObjectURL(musicUrlRef.current);
    };
  }, []);

  const cancel = () => {
    const wasMusicRunning = musicRunning.current;
    musicRunning.current = false;
    setMusicIsRunning(false);
    if (wasMusicRunning) {
      const cancelled = cancelWorker(worker.current, musicProgress ?? {
        status: 'running',
        text: 'Music generation running',
        indeterminate: true,
      });
      setMusicProgress(cancelled);
      setProgress(cancelled.text);
      if (musicUrlRef.current) URL.revokeObjectURL(musicUrlRef.current);
      musicUrlRef.current = null;
      setMusicUrl(null);
      setMusicResult(null);
    } else {
      worker.current?.terminate();
      setProgress('Diagnostic cancelled');
    }
    worker.current = null;
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
  const runVocoder = () => {
    cancel();
    setVocoderResult(null);
    setProgress('Starting isolated vocoder runtime worker');
    const next = new Worker(new URL('../workers/inference.worker.ts', import.meta.url), { type: 'module' });
    worker.current = next;
    next.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === 'progress') setProgress(`${data.stage}: ${data.detail}`);
      else if (data.type === 'vocoder-result') {
        setVocoderResult(data.result);
        setProgress('Vocoder runtime passed');
      } else if (data.type === 'error') setProgress(`Error: ${data.message}`);
    };
    next.postMessage({
      type: 'run-vocoder-smoke',
      manifestUrl: 'http://127.0.0.1:5178/manifest.json',
    });
  };
  const generateMusic = () => {
    cancel();
    setMusicResult(null);
    if (musicUrlRef.current) URL.revokeObjectURL(musicUrlRef.current);
    musicUrlRef.current = null;
    setMusicUrl(null);
    const request = createProductMusicRequest(durationSeconds);
    const starting = `Starting ${durationLabel(durationSeconds)} music generation`;
    setProgress(starting);
    setMusicProgress({
      status: 'running',
      text: starting,
      indeterminate: true,
    });
    const next = new Worker(new URL('../workers/inference.worker.ts', import.meta.url), { type: 'module' });
    worker.current = next;
    musicRunning.current = true;
    setMusicIsRunning(true);
    next.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === 'progress') {
        const view = progressView(data);
        setMusicProgress(view);
        setProgress(view.text);
      }
      else if (data.type === 'music-result') {
        const url = URL.createObjectURL(new Blob([data.result.wav], { type: 'audio/wav' }));
        musicUrlRef.current = url;
        setMusicUrl(url);
        setMusicResult(data.result);
        musicRunning.current = false;
        setMusicIsRunning(false);
        next.terminate();
        worker.current = null;
      } else if (data.type === 'error') {
        setProgress(`Error: ${data.message}`);
        setMusicProgress(null);
        musicRunning.current = false;
        setMusicIsRunning(false);
        next.terminate();
        worker.current = null;
      }
    };
    next.postMessage(request);
  };

  const downloadArtifacts = async () => {
    dispatchCache({ type: 'operation-started', operation: 'request-persistence' });
    const persistence = await requestPersistentStorage(navigator.storage);
    dispatchCache({ type: 'persistence-resolved', warning: persistence.warning });
    dispatchCache({ type: 'download-started' });
    runCacheWorker({
      type: 'download-artifacts',
      manifestUrl: PRODUCT_MANIFEST_URL,
    }, 'download');
  };

  const cancelArtifactDownload = () => {
    cacheWorker.current?.terminate();
    cacheWorker.current = null;
    dispatchCache({ type: 'download-cancelled' });
    inspectCache();
  };

  const deleteArtifactCaches = () => {
    if (!window.confirm('Delete all downloaded MiniMax Music 3 model files?')) return;
    dispatchCache({ type: 'operation-started', operation: 'delete' });
    runCacheWorker({
      type: 'delete-artifact-caches',
      manifestUrl: PRODUCT_MANIFEST_URL,
    }, 'delete');
  };

  const cacheControls = deriveArtifactCacheControls(cacheState, musicIsRunning);
  const downloadLabel = cacheControls.canRetry
    ? 'Retry'
    : cacheState.status?.state === 'partial'
      ? 'Resume'
      : 'Download';

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
        <section aria-labelledby="model-files-title" className="model-files">
          <h2 id="model-files-title">Model files</h2>
          <p aria-live="polite" className="cache-status">
            {describeArtifactCacheStatus(cacheState)}
          </p>
          {cacheState.persistenceWarning && (
            <p className="cache-warning">{cacheState.persistenceWarning}</p>
          )}
          {cacheState.lastError && (
            <p role="alert" className="cache-error">{cacheState.lastError.message}</p>
          )}
          {cacheState.notice && (
            <p aria-live="polite" className="cache-warning">{cacheState.notice}</p>
          )}
          <div className="cache-actions">
            <button
              type="button"
              disabled={!cacheControls.canDownload && !cacheControls.canRetry}
              onClick={() => void downloadArtifacts()}
            >
              {downloadLabel}
            </button>
            <button type="button" className="secondary" disabled={!cacheControls.canRefresh} onClick={inspectCache}>
              Refresh
            </button>
            <button type="button" className="secondary" disabled={!cacheControls.canDelete} onClick={deleteArtifactCaches}>
              Delete
            </button>
            <button type="button" className="secondary" disabled={!cacheControls.canCancel} onClick={cancelArtifactDownload}>
              Cancel
            </button>
          </div>
          {cacheState.operation === 'download' && !cacheState.downloadProgress && (
            <progress aria-label="Model download progress" />
          )}
          {cacheState.downloadProgress && (
            <div className="cache-progress">
              <p id="model-download-progress-label">
                Current file: {cacheState.downloadProgress.currentFile}
              </p>
              <progress
                aria-labelledby="model-download-progress-label"
                value={cacheState.downloadProgress.completedBytes}
                max={cacheState.downloadProgress.totalBytes}
              />
              <p>
                {formatBytes(cacheState.downloadProgress.completedBytes)} of{' '}
                {formatBytes(cacheState.downloadProgress.totalBytes)} verified
                {cacheState.downloadProgress.rate !== undefined
                  ? `, ${formatRate(cacheState.downloadProgress.rate)}`
                  : ''}
                {cacheState.downloadProgress.etaMs !== undefined
                  ? `, ETA ${formatEta(cacheState.downloadProgress.etaMs)}`
                  : ''}
              </p>
            </div>
          )}
        </section>
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
          <button type="button" disabled={!capability?.supported} onClick={runVocoder}>
            Run vocoder smoke
          </button>
          <label>
            Duration
            <select
              aria-label="Music duration"
              value={durationSeconds}
              onChange={(event) => setDurationSeconds(Number(event.target.value))}
            >
              {PRODUCT_DURATIONS.map((duration) => (
                <option key={duration} value={duration}>{duration} seconds</option>
              ))}
            </select>
          </label>
          <button
            type="button"
            disabled={!capability?.supported || !cacheControls.canGenerate}
            onClick={generateMusic}
          >
            Generate {durationLabel(durationSeconds)} music
          </button>
          <button type="button" className="secondary" disabled={!worker.current} onClick={cancel}>
            Cancel worker
          </button>
        </div>
        <output aria-live="polite" className="progress">
          {progress}
        </output>
        {musicProgress?.indeterminate && musicProgress.status === 'running' && (
          <progress data-testid="music-progress" aria-label="Music generation progress" />
        )}
        {musicProgress?.value !== undefined && musicProgress.max !== undefined && (
          <progress
            data-testid="music-progress"
            aria-label="Music generation progress"
            value={musicProgress.value}
            max={musicProgress.max}
          />
        )}
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
        {vocoderResult && (
          <pre data-testid="vocoder-smoke-result" className="result">
            {`waveform: ${vocoderResult.outputType} ${vocoderResult.shape.join(', ')}\nfinite: ${vocoderResult.finite ? 'yes' : 'no'}\nWAV bytes: ${vocoderResult.wavBytes}\naudio: ${vocoderResult.sampleRate} Hz, ${vocoderResult.channels} channels, ${vocoderResult.samples} samples, ${vocoderResult.bitsPerSample}-bit PCM\n${JSON.stringify(vocoderResult, null, 2)}`}
          </pre>
        )}
        {musicResult && musicUrl && (
          <section data-testid="music-generation-result" className="result">
            <audio data-testid="generated-audio" controls src={musicUrl} />
            <a
              data-testid="download-music"
              download={`minimax-music3-${musicResult.plan?.durationSeconds ?? durationSeconds}s.wav`}
              href={musicUrl}
            >
              Download WAV
            </a>
            <pre>{JSON.stringify({ ...musicResult, wav: undefined }, null, 2)}</pre>
          </section>
        )}
      </section>
    </main>
  );
}
