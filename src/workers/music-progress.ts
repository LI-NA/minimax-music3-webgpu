import type { MusicStage, WorkerProgress } from './protocol';

type SendProgress = (progress: WorkerProgress) => void;
type Now = () => number;

export interface ProgressView {
  status: 'idle' | 'running' | 'complete' | 'cancelled';
  text: string;
  indeterminate: boolean;
  value?: number;
  max?: number;
}

const seconds = (milliseconds: number) => `${(milliseconds / 1_000).toFixed(1)}s`;

export function formatProgress(progress: WorkerProgress) {
  if (progress.stage === 'artifact' && progress.completedBytes !== undefined && progress.totalBytes !== undefined) {
    const cache = progress.cacheHit ? ', cache hit' : '';
    return `${progress.detail}: ${progress.completedBytes}/${progress.totalBytes} bytes${cache}`;
  }
  if (progress.stage === 'autoregressive' && progress.completed !== undefined && progress.total !== undefined) {
    const speed = progress.rate === undefined ? '' : `, ${progress.rate.toFixed(2)} frames/s`;
    const eta = progress.etaMs === undefined ? '' : `, ${seconds(progress.etaMs)} remaining`;
    return `Autoregressive frames ${progress.completed}/${progress.total}${speed}${eta}`;
  }
  if (progress.stage === 'flow' && progress.completed !== undefined && progress.total !== undefined) {
    const speed = progress.stepMs === undefined ? '' : `, ${progress.stepMs.toFixed(0)} ms/step`;
    const eta = progress.etaMs === undefined ? '' : `, ${seconds(progress.etaMs)} remaining`;
    return `Flow steps ${progress.completed}/${progress.total}${speed}${eta}`;
  }
  if (progress.stage === 'complete' && progress.wavBytes !== undefined)
    return `Complete: ${progress.wavBytes} WAV bytes in ${seconds(progress.totalElapsedMs ?? 0)}`;
  return progress.detail;
}

export function progressView(progress: WorkerProgress): ProgressView {
  const completed = progress.completed ?? progress.completedBytes;
  const total = progress.total ?? progress.totalBytes;
  return {
    status: progress.stage === 'complete' ? 'complete' : 'running',
    text: formatProgress(progress),
    indeterminate: progress.activity === 'indeterminate',
    ...(completed !== undefined && total !== undefined ? { value: completed, max: total } : {}),
  };
}

export function initialProgressView(): ProgressView {
  return { status: 'idle', text: 'Awaiting diagnostic command', indeterminate: false };
}

export function cancelProgress(previous: ProgressView): ProgressView {
  void previous;
  return { status: 'cancelled', text: 'Music generation cancelled', indeterminate: false };
}

export function cancelWorker(
  worker: { terminate(): void } | null,
  previous: ProgressView,
) {
  worker?.terminate();
  return cancelProgress(previous);
}

function requireMonotonic(next: number, previous: number) {
  if (!Number.isInteger(next) || next < previous) throw new Error('progress counters must be monotonic integers');
}

export function createMusicProgressTracker(send: SendProgress, now: Now = () => performance.now()) {
  const totalStarted = now();
  let complete = false;
  let arCompleted = 0;
  let arStarted: number | undefined;
  let arPrevious: number | undefined;
  const arDurations: number[] = [];
  let flowCompleted = 0;
  let flowStarted: number | undefined;
  let flowPrevious: number | undefined;
  const flowDurations: number[] = [];

  const requireActive = () => {
    if (complete) throw new Error('music progress is already complete');
  };
  return {
    session(name: MusicStage) {
      requireActive();
      send({
        type: 'progress',
        stage: 'session',
        name,
        activity: 'indeterminate',
        detail: `Creating ${name} session`,
      });
    },
    beginAutoregressive() {
      requireActive();
      const current = now();
      arStarted ??= current;
      arPrevious ??= current;
    },
    autoregressive(retainedFrames: number) {
      requireActive();
      requireMonotonic(retainedFrames, arCompleted);
      const current = now();
      arStarted ??= current;
      const elapsedMs = current - arStarted;
      if (arPrevious !== undefined) arDurations.push(current - arPrevious);
      arPrevious = current;
      arCompleted = retainedFrames;
      const recent = arDurations.slice(-5);
      const rollingMs = recent.length
        ? recent.reduce((total, duration) => total + duration, 0) / recent.length
        : undefined;
      const rate = rollingMs !== undefined && rollingMs > 0 ? 1_000 / rollingMs : undefined;
      const stable = recent.length >= 3 && rate !== undefined && rate > 0;
      send({
        type: 'progress',
        stage: 'autoregressive',
        detail: `Retained frame ${retainedFrames} of 125`,
        completed: retainedFrames,
        total: 125,
        elapsedMs,
        ...(rate === undefined ? {} : { rate }),
        ...(stable ? { etaMs: ((125 - retainedFrames) / rate) * 1_000 } : {}),
      });
    },
    condition() {
      requireActive();
      send({ type: 'progress', stage: 'condition', detail: 'Encoding frame condition' });
    },
    beginFlow() {
      requireActive();
      const current = now();
      flowStarted ??= current;
      flowPrevious ??= current;
    },
    flow(completedSteps: number) {
      requireActive();
      requireMonotonic(completedSteps, flowCompleted);
      const current = now();
      flowStarted ??= current;
      if (flowPrevious !== undefined) flowDurations.push(current - flowPrevious);
      flowPrevious = current;
      flowCompleted = completedSteps;
      const recent = flowDurations.slice(-5);
      const stepMs = recent.length
        ? recent.reduce((total, duration) => total + duration, 0) / recent.length
        : undefined;
      send({
        type: 'progress',
        stage: 'flow',
        detail: `Flow step ${completedSteps} of 30`,
        completed: completedSteps,
        total: 30,
        elapsedMs: current - flowStarted,
        ...(stepMs === undefined ? {} : { stepMs }),
        ...(completedSteps >= 3 && stepMs !== undefined
          ? { etaMs: (30 - completedSteps) * stepMs }
          : {}),
      });
    },
    vocoder() {
      requireActive();
      send({ type: 'progress', stage: 'vocoder', detail: 'Synthesizing waveform' });
    },
    wav() {
      requireActive();
      send({ type: 'progress', stage: 'wav', detail: 'Encoding PCM WAV' });
    },
    complete(wavBytes: number) {
      requireActive();
      complete = true;
      send({
        type: 'progress',
        stage: 'complete',
        detail: 'Five-second music generation complete',
        wavBytes,
        totalElapsedMs: now() - totalStarted,
      });
    },
  };
}
