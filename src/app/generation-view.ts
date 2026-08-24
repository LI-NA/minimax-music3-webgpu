import { formatProgress } from '../workers/music-progress';
import type { MusicStage, WorkerProgress } from '../workers/protocol';

export const GENERATION_STAGE_COUNT = 5;
/** A run writes about a dozen rows, so this only bounds a worker that reports something unforeseen. */
const LOG_LINES = 40;

export type StageCounter = {
  completed: number;
  total: number;
  rate?: number;
  stepMs?: number;
};

/** One rendered log row. The key is what a counter rewrites in place instead of appending. */
export type GenerationLogRow = {
  key: string;
  line: string;
};

export type GenerationView = {
  stageIndex: number;
  stageFraction: number;
  indeterminate: boolean;
  etaMs?: number;
  frames?: StageCounter;
  acoustic?: StageCounter;
  flow?: StageCounter;
  vocoder?: StageCounter;
  log: GenerationLogRow[];
};

export function createGenerationView(): GenerationView {
  return { stageIndex: 0, stageFraction: 0, indeterminate: true, log: [] };
}

const SESSION_STAGE: Record<MusicStage, number> = {
  autoregressive: 0,
  acoustic: 1,
  condition: 1,
  flow: 2,
  vocoder: 3,
  wav: 4,
};

const carriesCounter = (progress: WorkerProgress) =>
  (progress.completed !== undefined && progress.total !== undefined) ||
  (progress.completedBytes !== undefined && progress.totalBytes !== undefined);

/**
 * A counting report keeps one row and rewrites it wherever that row already sits, however many
 * times the chunked pipeline re-enters its stage. Everything else happened once and is appended.
 * So a climbing number never pushes the run's history out of the log.
 */
function appendLog(view: GenerationView, progress: WorkerProgress): Pick<GenerationView, 'log'> {
  const row = {
    key: progress.stage === 'session' ? `session:${progress.name ?? ''}` : progress.stage,
    line: formatProgress(progress),
  };
  const at = carriesCounter(progress) ? view.log.findIndex((existing) => existing.key === row.key) : -1;
  if (at === -1) return { log: [...view.log, row].slice(-LOG_LINES) };
  return { log: view.log.map((existing, index) => (index === at ? row : existing)) };
}

function counter(progress: WorkerProgress): StageCounter | undefined {
  if (progress.completed === undefined || progress.total === undefined) return undefined;
  return {
    completed: progress.completed,
    total: progress.total,
    rate: progress.rate,
    stepMs: progress.stepMs,
  };
}

export function applyGenerationProgress(view: GenerationView, progress: WorkerProgress): GenerationView {
  const next: GenerationView = { ...view, ...appendLog(view, progress) };
  const measured = counter(progress);
  const fraction = measured && measured.total > 0 ? measured.completed / measured.total : undefined;
  // A chunked run does not walk the stages in order: it encodes one chunk's condition, runs that
  // chunk's flow steps, reports the chunk complete, and starts over. Taken literally those reports
  // drag the display back to an earlier stage once per chunk. So the stage index only moves
  // forward, and a report may drive the shared fraction, ETA and indeterminate flag only while its
  // own stage is the one on screen. Its own counter keeps updating either way.
  const onScreen = (stageIndex: number) => {
    next.stageIndex = Math.max(view.stageIndex, stageIndex);
    return next.stageIndex === stageIndex;
  };
  switch (progress.stage) {
    case 'session':
      if (onScreen(progress.name ? SESSION_STAGE[progress.name] : 0)) {
        next.stageFraction = next.stageIndex > view.stageIndex ? 0 : view.stageFraction;
        next.indeterminate = true;
      }
      break;
    case 'autoregressive':
      next.frames = measured ?? view.frames;
      if (onScreen(0)) {
        next.stageFraction = fraction ?? view.stageFraction;
        next.indeterminate = fraction === undefined;
        next.etaMs = progress.etaMs;
      }
      break;
    case 'acoustic':
      next.acoustic = measured ?? view.acoustic;
      if (onScreen(1)) {
        next.stageFraction = fraction ?? 0;
        next.indeterminate = fraction === undefined;
        next.etaMs = undefined;
      }
      break;
    case 'condition':
      // Condition counts the chunk it is starting, not one this stage has finished, so it names a
      // row in the log without claiming a share of the stage bar.
      if (onScreen(1)) {
        next.indeterminate = true;
        next.etaMs = undefined;
      }
      break;
    case 'flow':
      next.flow = measured ?? view.flow;
      if (onScreen(2)) {
        next.stageFraction = fraction ?? 0;
        next.indeterminate = fraction === undefined;
        next.etaMs = progress.etaMs;
      }
      break;
    case 'vocoder':
      next.vocoder = measured ?? view.vocoder;
      if (onScreen(3)) {
        next.stageFraction = fraction ?? 0;
        next.indeterminate = fraction === undefined;
        next.etaMs = undefined;
      }
      break;
    case 'wav':
    case 'complete':
      if (onScreen(4)) {
        next.stageFraction = progress.stage === 'complete' ? 1 : 0;
        next.indeterminate = progress.stage !== 'complete';
        next.etaMs = undefined;
      }
      break;
    default:
      break;
  }
  return next;
}

export function generationPercent(view: GenerationView): number {
  const fraction = Math.min(1, Math.max(0, view.stageFraction));
  return Math.min(99, Math.round(((view.stageIndex + fraction) / GENERATION_STAGE_COUNT) * 100));
}
