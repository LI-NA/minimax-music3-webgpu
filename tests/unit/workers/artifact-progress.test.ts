import { describe, expect, it } from 'vitest';
import { createArtifactProgressReporter } from '../../../src/workers/artifact-progress';

describe('artifact progress reporting', () => {
  it('derives rate from aggregate transferred bytes and ETA from remaining verified bytes', () => {
    const events: unknown[] = [];
    let now = 0;
    const reporter = createArtifactProgressReporter({
      totalBytes: 100,
      send: (event) => events.push(event),
      now: () => now,
    });

    reporter.report('a.bin', 10, 100, 0, 10);
    now = 1_000;
    reporter.report('a.bin', 50, 100, 0, 60);

    expect(events[0]).not.toHaveProperty('rate');
    expect(events[0]).not.toHaveProperty('etaMs');
    expect(events[1]).toEqual(
      expect.objectContaining({
        completedBytes: 50,
        rate: 50,
        etaMs: 1_000,
      }),
    );
  });

  it('averages aggregate throughput over the latest five seconds', () => {
    const events: Array<{ rate?: number; etaMs?: number }> = [];
    let now = 0;
    const reporter = createArtifactProgressReporter({
      totalBytes: 2_000,
      send: (event) => events.push(event),
      now: () => now,
    });

    for (const transferredBytes of [10, 110, 210, 310, 410, 510]) {
      reporter.report('a.bin', transferredBytes, 2_000, 0, transferredBytes);
      now += 1_000;
    }
    reporter.report('a.bin', 810, 2_000, 0, 810);

    expect(events.at(-1)?.rate).toBeCloseTo(140);
    expect(events.at(-1)?.etaMs).toBeCloseTo(8_500);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    'omits rate and ETA for non-finite transferred bytes (%s)',
    (transferredBytes) => {
      const events: unknown[] = [];
      let now = 0;
      const reporter = createArtifactProgressReporter({
        totalBytes: 100,
        send: (event) => events.push(event),
        now: () => now,
      });

      reporter.report('a.bin', 10, 100, 0, transferredBytes);
      now = 1_000;
      reporter.report('a.bin', 20, 100, 0, 20);

      expect(events).toHaveLength(2);
      for (const event of events) {
        expect(event).not.toHaveProperty('rate');
        expect(event).not.toHaveProperty('etaMs');
      }
    },
  );

  it('omits rate and ETA when derived transfer metrics are non-finite', () => {
    const events: unknown[] = [];
    let now = 0;
    const reporter = createArtifactProgressReporter({
      totalBytes: 100,
      send: (event) => events.push(event),
      now: () => now,
    });

    reporter.report('a.bin', 10, 100, 0, 1);
    now = 100;
    reporter.report('a.bin', 20, 100, 0, Number.MAX_VALUE);

    expect(events[1]).not.toHaveProperty('rate');
    expect(events[1]).not.toHaveProperty('etaMs');
  });

  it('reports the first callback immediately and coalesces a small-chunk download to its exact final total', () => {
    const events: unknown[] = [];
    const now = 0;
    const reporter = createArtifactProgressReporter({
      totalBytes: 160 * 1024 * 1000,
      send: (event) => events.push(event),
      now: () => now,
    });

    for (let loaded = 16 * 1024; loaded <= 160 * 1024 * 1000; loaded += 16 * 1024)
      reporter.report('weights.bin', loaded, 160 * 1024 * 1000, 0);
    reporter.complete('weights.bin', 160 * 1024 * 1000, 160 * 1024 * 1000, false);

    expect(events).toEqual([
      expect.objectContaining({ loaded: 16 * 1024, completedBytes: 16 * 1024, cacheHit: false }),
      expect.objectContaining({
        loaded: 160 * 1024 * 1000,
        completedBytes: 160 * 1024 * 1000,
        totalBytes: 160 * 1024 * 1000,
        cacheHit: false,
      }),
    ]);
  });

  it('emits the latest snapshot when callbacks cross the reporting interval', () => {
    const events: unknown[] = [];
    let now = 0;
    const reporter = createArtifactProgressReporter({
      totalBytes: 100,
      send: (event) => events.push(event),
      now: () => now,
    });

    reporter.report('a.bin', 10, 100, 0);
    now = 99;
    reporter.report('a.bin', 30, 100, 0);
    now = 100;
    reporter.report('a.bin', 40, 100, 0);

    expect(events).toEqual([
      expect.objectContaining({ loaded: 10, completedBytes: 10 }),
      expect.objectContaining({ loaded: 40, completedBytes: 40 }),
    ]);
  });

  it('keeps aggregate progress monotonic if an artifact download retries', () => {
    const events: unknown[] = [];
    let now = 0;
    const reporter = createArtifactProgressReporter({
      totalBytes: 100,
      send: (event) => events.push(event),
      now: () => now,
    });

    reporter.report('a.bin', 80, 100, 0);
    now = 100;
    reporter.report('a.bin', 10, 100, 0);

    expect(events).toEqual([
      expect.objectContaining({ loaded: 80, completedBytes: 80 }),
      expect.objectContaining({ loaded: 10, completedBytes: 80 }),
    ]);
  });

  it('keeps ETA positive when a fully transferred artifact restarts after verification fails', () => {
    const events: unknown[] = [];
    let now = 0;
    const reporter = createArtifactProgressReporter({
      totalBytes: 100,
      send: (event) => events.push(event),
      now: () => now,
    });

    reporter.report('a.bin', 100, 100, 0, 100);
    now = 1_000;
    reporter.report('a.bin', 10, 100, 0, 150);

    expect(events[1]).toEqual(
      expect.objectContaining({
        completedBytes: 100,
        rate: 50,
        etaMs: 1_800,
      }),
    );
  });

  it('holds the transfer baseline when an aggregate report falls behind', () => {
    const events: unknown[] = [];
    let now = 0;
    const reporter = createArtifactProgressReporter({
      totalBytes: 100,
      send: (event) => events.push(event),
      now: () => now,
    });

    reporter.report('a.bin', 10, 100, 0, 30);
    now = 500;
    reporter.report('a.bin', 20, 100, 0, 10);
    now = 1_000;
    reporter.report('a.bin', 40, 100, 0, 80);

    expect(events[1]).not.toHaveProperty('rate');
    expect(events[2]).toEqual(expect.objectContaining({ rate: 50, etaMs: 1_200 }));
  });

  it('emits a successful final after a callback already reached the file total', () => {
    const events: unknown[] = [];
    const reporter = createArtifactProgressReporter({
      totalBytes: 100,
      send: (event) => events.push(event),
      now: () => 0,
    });

    reporter.report('a.bin', 100, 100, 0);
    reporter.complete('a.bin', 100, 100, false);

    expect(events).toEqual([
      expect.objectContaining({
        currentFile: 'a.bin',
        loaded: 100,
        completedBytes: 100,
        cacheHit: false,
      }),
      expect.objectContaining({
        currentFile: 'a.bin',
        loaded: 100,
        completedBytes: 100,
        cacheHit: false,
      }),
    ]);
  });

  it('reports one exact cache-hit final and never schedules a stale failed-file event', () => {
    const events: unknown[] = [];
    let now = 0;
    const reporter = createArtifactProgressReporter({
      totalBytes: 200,
      send: (event) => events.push(event),
      now: () => now,
    });

    reporter.complete('cached.bin', 100, 100, true);
    expect(events[0]).not.toHaveProperty('rate');
    expect(events[0]).not.toHaveProperty('etaMs');
    reporter.report('failed.bin', 10, 100, 100);
    reporter.discard();
    now = 200;
    reporter.report('next.bin', 10, 100, 100);

    expect(events).toEqual([
      expect.objectContaining({
        currentFile: 'cached.bin',
        loaded: 100,
        completedBytes: 100,
        cacheHit: true,
      }),
      expect.objectContaining({
        currentFile: 'failed.bin',
        loaded: 10,
        completedBytes: 110,
        cacheHit: false,
      }),
      expect.objectContaining({
        currentFile: 'next.bin',
        loaded: 10,
        completedBytes: 110,
        cacheHit: false,
      }),
    ]);
  });
});
