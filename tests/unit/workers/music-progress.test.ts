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
    const tracker = createMusicProgressTracker((event) => events.push(event), () => now);

    tracker.beginAutoregressive();
    now = 2_000;
    tracker.autoregressive(1);
    now = 3_000;
    tracker.autoregressive(2);
    now = 4_000;
    tracker.autoregressive(3);

    const ar = events.filter((event) => event.stage === 'autoregressive');
    expect(ar[0]).toMatchObject({ completed: 1, total: 125, elapsedMs: 1_000 });
    expect(ar[0]).not.toHaveProperty('etaMs');
    expect(ar[1]).not.toHaveProperty('etaMs');
    expect(ar[2]).toMatchObject({ completed: 3, total: 125, rate: 1, etaMs: 122_000 });
    expect(() => tracker.autoregressive(2)).toThrow(/monotonic/);
  });

  it('reports flow rolling step time and completes only after WAV cleanup', () => {
    let now = 0;
    const events: WorkerProgress[] = [];
    const tracker = createMusicProgressTracker((event) => events.push(event), () => now);

    tracker.condition();
    tracker.session('flow');
    tracker.beginFlow();
    now = 500;
    tracker.flow(1);
    now = 1_100;
    tracker.flow(2);
    now = 1_800;
    tracker.flow(3);
    tracker.vocoder();
    tracker.wav();
    now = 2_000;
    tracker.complete(880_684);

    const flow = events.filter((event) => event.stage === 'flow');
    expect(flow[0]).not.toHaveProperty('etaMs');
    expect(flow[2]).toMatchObject({ completed: 3, total: 30, stepMs: 600, etaMs: 16_200 });
    expect(events.at(-1)).toMatchObject({
      stage: 'complete',
      wavBytes: 880_684,
      totalElapsedMs: 2_000,
    });
    expect(events.map((event) => event.stage)).toEqual([
      'condition',
      'session',
      'flow',
      'flow',
      'flow',
      'vocoder',
      'wav',
      'complete',
    ]);
    expect(() => tracker.flow(4)).toThrow(/complete/);
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
    expect(cancelProgress(initialProgressView())).toEqual({
      status: 'cancelled',
      text: 'Music generation cancelled',
      indeterminate: false,
    });
  });

  it('terminates the inference worker when music generation is cancelled', () => {
    let terminated = false;

    const cancelled = cancelWorker(
      { terminate: () => { terminated = true; } },
      { status: 'running', text: 'Flow steps 3/30', indeterminate: false, value: 3, max: 30 },
    );

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
