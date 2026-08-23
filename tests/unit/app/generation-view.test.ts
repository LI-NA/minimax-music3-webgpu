import { describe, expect, it } from 'vitest';
import {
  applyGenerationProgress,
  createGenerationView,
  generationPercent,
  type GenerationView,
} from '../../../src/app/generation-view';
import type { WorkerProgress } from '../../../src/workers/protocol';

const session = (name: WorkerProgress['name']): WorkerProgress => ({
  type: 'progress',
  stage: 'session',
  name,
  activity: 'indeterminate',
  detail: `Creating ${name} session`,
});

const condition = (started: number, total: number): WorkerProgress => ({
  type: 'progress',
  stage: 'condition',
  detail: `Frame condition ${started} of ${total}`,
  completed: started,
  total,
});

const acoustic = (completed: number, total: number): WorkerProgress => ({
  type: 'progress',
  stage: 'acoustic',
  detail: `Acoustic chunk ${completed} of ${total}`,
  completed,
  total,
});

const flow = (completed: number, total: number, etaMs?: number): WorkerProgress => ({
  type: 'progress',
  stage: 'flow',
  detail: `Flow step ${completed} of ${total}`,
  completed,
  total,
  ...(etaMs === undefined ? {} : { etaMs }),
});

const apply = (view: GenerationView, reports: WorkerProgress[]) =>
  reports.reduce((current, report) => applyGenerationProgress(current, report), view);

/** The report order a chunked run produces: condition, that chunk's flow steps, chunk complete. */
function chunkedRun(chunks: number, stepsPerChunk: number): WorkerProgress[] {
  const totalSteps = chunks * stepsPerChunk;
  const reports: WorkerProgress[] = [session('condition'), session('flow')];
  for (let chunk = 0; chunk < chunks; chunk++) {
    reports.push(condition(chunk + 1, chunks));
    for (let step = 1; step <= stepsPerChunk; step++) reports.push(flow(chunk * stepsPerChunk + step, totalSteps, 500));
    reports.push(acoustic(chunk + 1, chunks));
  }
  return reports;
}

describe('generation view stages', () => {
  it('never steps back to an earlier stage while a chunked run interleaves its reports', () => {
    const stages: number[] = [];
    const percents: number[] = [];
    chunkedRun(4, 3).reduce((view, report) => {
      const next = applyGenerationProgress(view, report);
      stages.push(next.stageIndex);
      percents.push(generationPercent(next));
      return next;
    }, createGenerationView());

    expect(stages).toEqual([...stages].sort((left, right) => left - right));
    expect(percents).toEqual([...percents].sort((left, right) => left - right));
    expect(stages.at(-1)).toBe(2);
  });

  it('keeps the flow step fraction and ETA across a chunk boundary', () => {
    const midChunk = apply(createGenerationView(), [
      session('condition'),
      session('flow'),
      condition(1, 4),
      flow(3, 12, 4_500),
    ]);
    const boundary = apply(midChunk, [acoustic(1, 4), condition(2, 4)]);

    expect(midChunk).toMatchObject({ stageIndex: 2, stageFraction: 0.25, etaMs: 4_500, indeterminate: false });
    expect(boundary).toMatchObject({ stageIndex: 2, stageFraction: 0.25, etaMs: 4_500, indeterminate: false });
  });

  it('keeps counting chunks while the flow stage owns the display', () => {
    const view = apply(createGenerationView(), [
      session('condition'),
      session('flow'),
      condition(1, 4),
      flow(3, 12),
      acoustic(1, 4),
      condition(2, 4),
      flow(6, 12),
      acoustic(2, 4),
    ]);

    expect(view.acoustic).toEqual({ completed: 2, total: 4 });
    expect(view.flow).toEqual({ completed: 6, total: 12 });
  });

  it('still follows the stages of an unchunked run', () => {
    const view = apply(createGenerationView(), [
      session('autoregressive'),
      { type: 'progress', stage: 'autoregressive', detail: 'frame 125', completed: 125, total: 125 },
      session('condition'),
      condition(1, 1),
      session('flow'),
      flow(30, 30),
      session('vocoder'),
      { type: 'progress', stage: 'vocoder', detail: 'run 1', completed: 1, total: 1 },
      { type: 'progress', stage: 'wav', detail: 'Encoding PCM WAV' },
      { type: 'progress', stage: 'complete', detail: 'Music generation complete' },
    ]);

    expect(view).toMatchObject({ stageIndex: 4, stageFraction: 1, indeterminate: false });
    expect(generationPercent(view)).toBe(99);
  });
});

describe('generation view log', () => {
  const lines = (view: { log: { line: string }[] }) => view.log.map((row) => row.line);

  it('keeps one row per counter however often the chunked run re-enters its stage', () => {
    const view = apply(createGenerationView(), chunkedRun(7, 30));

    expect(lines(view)).toEqual([
      'Creating condition session',
      'Creating flow session',
      'Frame conditions 7/7',
      'Flow steps 210/210, 0.5s remaining',
      'Acoustic chunks 7/7',
    ]);
  });

  it('appends one-off events and never drops an earlier one', () => {
    const view = apply(createGenerationView(), [
      session('autoregressive'),
      { type: 'progress', stage: 'autoregressive', detail: 'frame 750', completed: 750, total: 750 },
      ...chunkedRun(74, 30),
      session('vocoder'),
      { type: 'progress', stage: 'vocoder', detail: 'run 148', completed: 148, total: 148 },
      { type: 'progress', stage: 'wav', detail: 'Encoding PCM WAV' },
    ]);

    expect(lines(view)).toEqual([
      'Creating autoregressive session',
      'Autoregressive frames 750/750',
      'Creating condition session',
      'Creating flow session',
      'Frame conditions 74/74',
      'Flow steps 2220/2220, 0.5s remaining',
      'Acoustic chunks 74/74',
      'Creating vocoder session',
      'Vocoder channel runs 148/148',
      'Encoding PCM WAV',
    ]);
  });

  it('holds a 300 second run to the same row count as a 10 second one', () => {
    const short = apply(createGenerationView(), chunkedRun(2, 30));
    const long = apply(createGenerationView(), chunkedRun(74, 30));

    expect(long.log).toHaveLength(short.log.length);
  });
});
