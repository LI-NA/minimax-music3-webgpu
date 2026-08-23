import {
  planDuration,
  planRetainedFrames,
  type DurationPlanRequest,
  type RetainedFramesPlan,
  type Termination,
} from '../runtime/pipeline/duration-plan';
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
  if (progress.stage === 'acoustic' && progress.completed !== undefined && progress.total !== undefined)
    return `Acoustic chunks ${progress.completed}/${progress.total}`;
  if (progress.stage === 'condition' && progress.completed !== undefined && progress.total !== undefined)
    return `Frame conditions ${progress.completed}/${progress.total}`;
  if (progress.stage === 'flow' && progress.completed !== undefined && progress.total !== undefined) {
    const speed = progress.stepMs === undefined ? '' : `, ${progress.stepMs.toFixed(0)} ms/step`;
    const eta = progress.etaMs === undefined ? '' : `, ${seconds(progress.etaMs)} remaining`;
    return `Flow steps ${progress.completed}/${progress.total}${speed}${eta}`;
  }
  if (progress.stage === 'vocoder' && progress.completed !== undefined && progress.total !== undefined)
    return `Vocoder channel runs ${progress.completed}/${progress.total}`;
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

export function cancelProgress(): ProgressView {
  return { status: 'cancelled', text: 'Music generation cancelled', indeterminate: false };
}

export function cancelWorker(worker: { terminate(): void } | null) {
  worker?.terminate();
  return cancelProgress();
}

function requireMonotonic(next: number, previous: number) {
  if (!Number.isInteger(next) || next < previous) throw new Error('progress counters must be monotonic integers');
}

function requireProgressCounter(next: number, previous: number, total: number) {
  requireMonotonic(next, previous);
  if (next > total) throw new Error('progress counters must not exceed their total');
}

export function createMusicProgressTracker(
  send: SendProgress,
  request: DurationPlanRequest = { durationSeconds: 5, promptTokens: 0 },
  now: Now = () => performance.now(),
) {
  const requestedPlan = planDuration(request);
  const flowSteps = request.flowSteps ?? 30;
  let retainedPlan: RetainedFramesPlan = requestedPlan;
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
  let acousticCompleted = 0;
  let conditionStarted = 0;
  let vocoderCompleted = 0;

  const requireActive = () => {
    if (complete) throw new Error('music progress is already complete');
  };
  return {
    setRetainedFrames(retainedFrames: number, termination: Termination) {
      requireActive();
      if (retainedFrames > requestedPlan.retainedFrames)
        throw new Error('Retained frames must not exceed requested frames');
      retainedPlan = planRetainedFrames({
        retainedFrames,
        promptTokens: request.promptTokens,
        termination,
        flowSteps,
      });
    },
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
      requireProgressCounter(retainedFrames, arCompleted, requestedPlan.retainedFrames);
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
        detail: `Retained frame ${retainedFrames} of ${requestedPlan.retainedFrames}`,
        completed: retainedFrames,
        total: requestedPlan.retainedFrames,
        elapsedMs,
        ...(rate === undefined ? {} : { rate }),
        ...(stable ? { etaMs: ((requestedPlan.retainedFrames - retainedFrames) / rate) * 1_000 } : {}),
      });
    },
    acoustic(completedChunks: number) {
      requireActive();
      requireProgressCounter(completedChunks, acousticCompleted, retainedPlan.chunks.length);
      acousticCompleted = completedChunks;
      send({
        type: 'progress',
        stage: 'acoustic',
        detail: `Acoustic chunk ${completedChunks} of ${retainedPlan.chunks.length}`,
        completed: completedChunks,
        total: retainedPlan.chunks.length,
      });
    },
    /**
     * Reported as the encode starts, so the display is not silent through it. A chunked run counts
     * the chunk it is starting; the fixed run has a single condition and counts nothing.
     */
    condition(startedChunks?: number) {
      requireActive();
      if (startedChunks === undefined) {
        send({ type: 'progress', stage: 'condition', detail: 'Encoding frame condition' });
        return;
      }
      requireProgressCounter(startedChunks, conditionStarted, retainedPlan.chunks.length);
      conditionStarted = startedChunks;
      send({
        type: 'progress',
        stage: 'condition',
        detail: `Frame condition ${startedChunks} of ${retainedPlan.chunks.length}`,
        completed: startedChunks,
        total: retainedPlan.chunks.length,
      });
    },
    startFlowChunk(completedSteps: number) {
      requireActive();
      requireProgressCounter(completedSteps, flowCompleted, retainedPlan.flowCalls);
      if (completedSteps !== flowCompleted) throw new Error('flow chunk must start at the current completed step');
      if (completedSteps % flowSteps !== 0) throw new Error('flow chunk must start at a flow-step boundary');
      const current = now();
      flowStarted ??= current;
      flowPrevious = current;
    },
    beginFlow() {
      requireActive();
      const current = now();
      flowStarted ??= current;
      flowPrevious ??= current;
    },
    flow(completedSteps: number) {
      requireActive();
      requireProgressCounter(completedSteps, flowCompleted, retainedPlan.flowCalls);
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
        detail: `Flow step ${completedSteps} of ${retainedPlan.flowCalls}`,
        completed: completedSteps,
        total: retainedPlan.flowCalls,
        elapsedMs: current - flowStarted,
        ...(stepMs === undefined ? {} : { stepMs }),
        ...(recent.length >= 3 && stepMs !== undefined
          ? { etaMs: (retainedPlan.flowCalls - completedSteps) * stepMs }
          : {}),
      });
    },
    vocoder(completedCalls?: number) {
      requireActive();
      if (completedCalls === undefined) {
        send({ type: 'progress', stage: 'vocoder', detail: 'Synthesizing waveform' });
        return;
      }
      requireProgressCounter(completedCalls, vocoderCompleted, retainedPlan.vocoderCalls);
      vocoderCompleted = completedCalls;
      send({
        type: 'progress',
        stage: 'vocoder',
        detail: `Vocoder channel run ${completedCalls} of ${retainedPlan.vocoderCalls}`,
        completed: completedCalls,
        total: retainedPlan.vocoderCalls,
      });
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
        detail: 'Music generation complete',
        wavBytes,
        totalElapsedMs: now() - totalStarted,
      });
    },
  };
}
