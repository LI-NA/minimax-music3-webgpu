import { describe, expect, it } from 'vitest';
import {
  cancelProgress,
  cancelWorker,
  createMusicProgressTracker,
  formatProgress,
  initialProgressView,
  progressView,
} from '../../../src/workers/music-progress';
import type { WorkerProgress } from '../../../src/workers/protocol';

describe('music generation progress', () => {
  it('keeps countable progress monotonic and omits unstable estimates', () => {
    let now = 1_000;
    const events: WorkerProgress[] = [];
    const tracker = createMusicProgressTracker(
      (event) => events.push(event),
      { durationSeconds: 10, promptTokens: 40 },
      () => now,
    );

    tracker.beginAutoregressive();
    now = 2_000;
    tracker.autoregressive(1);
    now = 3_000;
    tracker.autoregressive(2);
    now = 4_000;
    tracker.autoregressive(3);

    const ar = events.filter((event) => event.stage === 'autoregressive');
    expect(ar[0]).toMatchObject({ completed: 1, total: 250, elapsedMs: 1_000 });
    expect(ar[0]).not.toHaveProperty('etaMs');
    expect(ar[1]).not.toHaveProperty('etaMs');
    expect(ar[2]).toMatchObject({ completed: 3, total: 250, rate: 1, etaMs: 247_000 });
    expect(() => tracker.autoregressive(2)).toThrow(/monotonic/);
    expect(() => tracker.autoregressive(251)).toThrow(/total/);
  });

  it('derives acoustic, flow, and mono vocoder totals from actual retained frames', () => {
    let now = 0;
    const events: WorkerProgress[] = [];
    const tracker = createMusicProgressTracker(
      (event) => events.push(event),
      { durationSeconds: 10, promptTokens: 40 },
      () => now,
    );

    tracker.setRetainedFrames(201, 'natural-end');
    tracker.acoustic(1);
    tracker.condition();
    tracker.session('flow');
    tracker.beginFlow();
    now = 500;
    tracker.flow(31);
    now = 1_100;
    tracker.flow(32);
    now = 1_800;
    tracker.flow(33);
    tracker.vocoder(1);
    tracker.vocoder(4);
    tracker.wav();
    now = 2_000;
    tracker.complete(880_684);

    const flow = events.filter((event) => event.stage === 'flow');
    expect(flow[0]).not.toHaveProperty('etaMs');
    expect(flow[2]).toMatchObject({ completed: 33, total: 60, stepMs: 600, etaMs: 16_200 });
    expect(events.find((event) => event.stage === 'acoustic')).toMatchObject({
      completed: 1,
      total: 2,
    });
    expect(events.filter((event) => event.stage === 'vocoder')).toEqual([
      expect.objectContaining({ completed: 1, total: 4 }),
      expect.objectContaining({ completed: 4, total: 4 }),
    ]);
    expect(events.at(-1)).toMatchObject({
      stage: 'complete',
      detail: 'Music generation complete',
      wavBytes: 880_684,
      totalElapsedMs: 2_000,
    });
    expect(events.map((event) => event.stage)).toEqual([
      'acoustic',
      'condition',
      'session',
      'flow',
      'flow',
      'flow',
      'vocoder',
      'vocoder',
      'wav',
      'complete',
    ]);
    expect(() => tracker.flow(4)).toThrow(/complete/);
  });

  it('resets flow step timing between chunks without resetting cumulative progress', () => {
    let now = 0;
    const events: WorkerProgress[] = [];
    const tracker = createMusicProgressTracker(
      (event) => events.push(event),
      { durationSeconds: 10, promptTokens: 40 },
      () => now,
    );

    tracker.startFlowChunk(0);
    for (let completed = 1; completed <= 30; completed++) {
      now += 100;
      tracker.flow(completed);
    }
    now += 60_000;
    tracker.condition();
    tracker.startFlowChunk(30);
    now += 100;
    tracker.flow(31);

    const last = events.filter((event) => event.stage === 'flow').at(-1);
    expect(last).toMatchObject({
      completed: 31,
      total: 60,
      stepMs: 100,
      etaMs: 2_900,
    });
  });

  it('uses the requested flow step count for totals and chunk boundaries', () => {
    const events: WorkerProgress[] = [];
    const tracker = createMusicProgressTracker(
      (event) => events.push(event),
      { durationSeconds: 10, promptTokens: 40, flowSteps: 10 },
      () => 0,
    );

    tracker.startFlowChunk(0);
    tracker.flow(10);
    tracker.startFlowChunk(10);
    tracker.flow(11);

    expect(events.filter((event) => event.stage === 'flow').at(-1)).toMatchObject({
      completed: 11,
      total: 20,
    });

    tracker.flow(13);
    expect(() => tracker.startFlowChunk(13)).toThrow(/boundary/);
  });

  it('formats named indeterminate work, determinate stages, completion, and cancellation', () => {
    const session: WorkerProgress = {
      type: 'progress',
      stage: 'session',
      detail: 'Creating flow session',
      name: 'flow',
      activity: 'indeterminate',
    };
    const flow: WorkerProgress = {
      type: 'progress',
      stage: 'flow',
      detail: 'Flow step 4 of 30',
      completed: 4,
      total: 30,
      elapsedMs: 2_000,
    };

    expect(progressView(session)).toMatchObject({ indeterminate: true });
    expect(progressView(session).value).toBeUndefined();
    expect(progressView(flow)).toMatchObject({ indeterminate: false, value: 4, max: 30 });
    expect(formatProgress(flow)).toContain('4/30');
    expect(
      formatProgress({
        type: 'progress',
        stage: 'acoustic',
        detail: '',
        completed: 1,
        total: 2,
      }),
    ).toBe('Acoustic chunks 1/2');
    expect(
      formatProgress({
        type: 'progress',
        stage: 'vocoder',
        detail: '',
        completed: 3,
        total: 4,
      }),
    ).toBe('Vocoder channel runs 3/4');
    expect(
      formatProgress({
        type: 'progress',
        stage: 'complete',
        detail: 'Music generation complete',
        wavBytes: 1_417_260,
        totalElapsedMs: 2_000,
      }),
    ).toBe('Complete: 1417260 WAV bytes in 2.0s');
    expect(initialProgressView().status).toBe('idle');
    expect(cancelProgress()).toEqual({
      status: 'cancelled',
      text: 'Music generation cancelled',
      indeterminate: false,
    });
  });

  it('terminates the inference worker when music generation is cancelled', () => {
    let terminated = false;

    const cancelled = cancelWorker({
      terminate: () => {
        terminated = true;
      },
    });

    expect(terminated).toBe(true);
    expect(cancelled.status).toBe('cancelled');
    expect(cancelled.value).toBeUndefined();
  });

  it('reports aggregate artifact bytes and cache-hit state', () => {
    const event: WorkerProgress = {
      type: 'progress',
      stage: 'artifact',
      detail: 'flow.onnx.data',
      currentFile: 'flow.onnx.data',
      completedBytes: 90,
      totalBytes: 100,
      cacheHit: true,
    };

    expect(formatProgress(event)).toContain('90/100 bytes');
    expect(formatProgress(event)).toContain('cache hit');
  });
});
