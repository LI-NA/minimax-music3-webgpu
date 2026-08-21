import { formatProgress } from '../workers/music-progress';
import type { MusicStage, WorkerProgress } from '../workers/protocol';

export const GENERATION_STAGE_COUNT = 5;
const LOG_LINES = 8;

export type StageCounter = {
  completed: number;
  total: number;
  rate?: number;
  stepMs?: number;
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
  log: string[];
  lastLogStage?: string;
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

function appendLog(
  view: GenerationView,
  progress: WorkerProgress,
): Pick<GenerationView, 'log' | 'lastLogStage'> {
  const line = formatProgress(progress);
  const key = progress.stage === 'session' ? `session:${progress.name ?? ''}` : progress.stage;
  const log = key === view.lastLogStage ? view.log.slice(0, -1) : view.log;
  return { log: [...log, line].slice(-LOG_LINES), lastLogStage: key };
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

export function applyGenerationProgress(
  view: GenerationView,
  progress: WorkerProgress,
): GenerationView {
  const next: GenerationView = { ...view, ...appendLog(view, progress) };
  const measured = counter(progress);
  const fraction = measured && measured.total > 0 ? measured.completed / measured.total : undefined;
  switch (progress.stage) {
    case 'session':
      next.stageIndex = Math.max(view.stageIndex, progress.name ? SESSION_STAGE[progress.name] : 0);
      next.stageFraction = next.stageIndex > view.stageIndex ? 0 : view.stageFraction;
      next.indeterminate = true;
      break;
    case 'autoregressive':
      next.stageIndex = 0;
      next.stageFraction = fraction ?? view.stageFraction;
      next.indeterminate = fraction === undefined;
      next.frames = measured ?? view.frames;
      next.etaMs = progress.etaMs;
      break;
    case 'acoustic':
    case 'condition':
      next.stageIndex = 1;
      next.stageFraction = progress.stage === 'acoustic' ? (fraction ?? 0) : view.stageFraction;
      next.indeterminate = fraction === undefined;
      next.acoustic = measured ?? view.acoustic;
      next.etaMs = undefined;
      break;
    case 'flow':
      next.stageIndex = 2;
      next.stageFraction = fraction ?? 0;
      next.indeterminate = fraction === undefined;
      next.flow = measured ?? view.flow;
      next.etaMs = progress.etaMs;
      break;
    case 'vocoder':
      next.stageIndex = 3;
      next.stageFraction = fraction ?? 0;
      next.indeterminate = fraction === undefined;
      next.vocoder = measured ?? view.vocoder;
      next.etaMs = undefined;
      break;
    case 'wav':
    case 'complete':
      next.stageIndex = 4;
      next.stageFraction = progress.stage === 'complete' ? 1 : 0;
      next.indeterminate = progress.stage !== 'complete';
      next.etaMs = undefined;
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
