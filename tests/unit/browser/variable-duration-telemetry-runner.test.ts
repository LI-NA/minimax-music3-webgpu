import { describe, expect, it } from 'vitest';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import {
  parseNvidiaSample,
  findGpuConflicts,
  inspectNamedGpuProcesses,
  ownedTreeKillCommand,
  summarizeTelemetry,
  type GpuSample,
} from '../../browser/variable-duration-telemetry-runner';
import * as telemetryRunner from '../../browser/variable-duration-telemetry-runner';

describe('variable-duration NVIDIA telemetry runner', () => {
  it('writes telemetry evidence exclusively and preserves the first capture', () => {
    const writeEvidence = (
      telemetryRunner as unknown as {
        writeEvidence?: (directory: string, samples: GpuSample[], result: Record<string, unknown>) => void;
      }
    ).writeEvidence;
    expect(typeof writeEvidence).toBe('function');
    const directory = mkdtempSync(path.join(tmpdir(), 'minimax-telemetry-'));
    try {
      const sample: GpuSample = {
        at: 'first',
        monotonicMs: 1,
        usedMiB: 2,
        totalMiB: 3,
        utilizationPercent: 4,
      };
      writeEvidence?.(directory, [sample], { marker: 'first' });
      expect(() => writeEvidence?.(directory, [sample], { marker: 'second' })).toThrow();
      expect(JSON.parse(readFileSync(path.join(directory, 'telemetry.json'), 'utf8'))).toEqual({
        marker: 'first',
      });
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it('clears the preflight timeout after success and failure settle', async () => {
    type RaceWithCancellableTimeout = <T>(
      operation: Promise<T>,
      timeoutMs: number,
      timeoutError: () => Error,
      schedule: (callback: () => void, milliseconds: number) => unknown,
      cancel: (handle: unknown) => void,
    ) => Promise<T>;
    const raceWithCancellableTimeout = (
      telemetryRunner as unknown as {
        raceWithCancellableTimeout?: RaceWithCancellableTimeout;
      }
    ).raceWithCancellableTimeout;
    expect(typeof raceWithCancellableTimeout).toBe('function');

    const handles: unknown[] = [];
    const cancelled: unknown[] = [];
    const schedule = (_callback: () => void, milliseconds: number) => {
      const handle = { milliseconds };
      handles.push(handle);
      return handle;
    };
    const cancel = (handle: unknown) => cancelled.push(handle);

    await expect(
      raceWithCancellableTimeout?.(Promise.resolve('clean'), 600_000, () => new Error('timed out'), schedule, cancel),
    ).resolves.toBe('clean');
    await expect(
      raceWithCancellableTimeout?.(
        Promise.reject(new Error('monitor failed')),
        600_000,
        () => new Error('timed out'),
        schedule,
        cancel,
      ),
    ).rejects.toThrow('monitor failed');
    expect(cancelled).toEqual(handles);
  });

  it('selects only the short or long browser qualification spec', () => {
    const selectBrowserGate = (
      telemetryRunner as unknown as {
        selectBrowserGate?: (value: string | undefined) => { spec: string };
      }
    ).selectBrowserGate;
    expect(typeof selectBrowserGate).toBe('function');
    expect(selectBrowserGate?.(undefined)).toEqual({
      spec: 'tests/browser/variable-duration.spec.ts',
    });
    expect(selectBrowserGate?.('long-duration')).toEqual({
      spec: 'tests/browser/long-duration.spec.ts',
    });
    expect(() => selectBrowserGate?.('reference-comparison')).toThrow('allowlist');
  });
  it('parses timestamped MiB samples and rejects malformed output', () => {
    expect(parseNvidiaSample('2026/08/21 20:00:00.123, 6834, 16376, 7', 12.5)).toEqual({
      at: '2026/08/21 20:00:00.123',
      monotonicMs: 12.5,
      usedMiB: 6_834,
      totalMiB: 16_376,
      utilizationPercent: 7,
    });
    expect(parseNvidiaSample('memory.used [MiB]', 0)).toBeUndefined();
    expect(parseNvidiaSample('2026/08/21 20:00:00.123, N/A', 0)).toBeUndefined();
  });

  it('records baseline, maximum, final, and effective cadence', () => {
    const samples: GpuSample[] = [
      { at: 'a', monotonicMs: 0, usedMiB: 1_704, totalMiB: 16_376, utilizationPercent: 0 },
      { at: 'b', monotonicMs: 180, usedMiB: 6_834, totalMiB: 16_376, utilizationPercent: 90 },
      { at: 'c', monotonicMs: 370, usedMiB: 2_000, totalMiB: 16_376, utilizationPercent: 5 },
      { at: 'd', monotonicMs: 550, usedMiB: 1_705, totalMiB: 16_376, utilizationPercent: 0 },
    ];
    expect(summarizeTelemetry(samples, 100, 0, 12_288, 512)).toEqual({
      requestedCadenceMs: 100,
      effectiveCadenceMs: { mean: 183.333, median: 180, maximum: 190 },
      baselineMiB: 1_704,
      physicalTotalMiB: 16_376,
      rawMaximumMiB: 6_834,
      rawFinalMiB: 1_705,
      maximumDeltaMiB: 5_130,
      incrementalLimitMiB: 12_288,
      incrementalExceeded: false,
      capacityReserveMiB: 512,
      capacityGuardMiB: 15_864,
      capacityGuardExceeded: false,
      baselineUtilizationPercent: 0,
      maximumUtilizationPercent: 90,
      finalUtilizationPercent: 0,
      sampleCount: 4,
    });
  });

  it('marks a sample strictly above the ceiling as exceeded', () => {
    const samples: GpuSample[] = [
      { at: 'a', monotonicMs: 0, usedMiB: 3_000, totalMiB: 20_000, utilizationPercent: 0 },
      { at: 'b', monotonicMs: 100, usedMiB: 15_288, totalMiB: 20_000, utilizationPercent: 0 },
      { at: 'c', monotonicMs: 200, usedMiB: 15_289, totalMiB: 20_000, utilizationPercent: 0 },
    ];
    const summary = summarizeTelemetry(samples, 100, 0, 12_288, 512);
    expect(summary.incrementalExceeded).toBe(true);
    expect(summary.maximumDeltaMiB).toBe(12_289);
  });

  it('keeps the physical-capacity guard separate from the incremental limit', () => {
    const samples: GpuSample[] = [
      { at: 'a', monotonicMs: 0, usedMiB: 15_000, totalMiB: 16_376, utilizationPercent: 0 },
      { at: 'b', monotonicMs: 100, usedMiB: 15_864, totalMiB: 16_376, utilizationPercent: 0 },
    ];
    const summary = summarizeTelemetry(samples, 100, 0, 12_288, 512);
    expect(summary.incrementalExceeded).toBe(false);
    expect(summary.capacityGuardExceeded).toBe(true);
  });

  it('uses the spawn-adjacent sample rather than an earlier preflight sample as baseline', () => {
    const samples: GpuSample[] = [
      { at: 'preflight', monotonicMs: 0, usedMiB: 6_000, totalMiB: 20_000, utilizationPercent: 80 },
      {
        at: 'baseline',
        monotonicMs: 100,
        usedMiB: 3_000,
        totalMiB: 20_000,
        utilizationPercent: 30,
      },
      { at: 'peak', monotonicMs: 200, usedMiB: 8_000, totalMiB: 20_000, utilizationPercent: 95 },
    ];
    const summary = summarizeTelemetry(samples, 100, 1, 12_288, 512);
    expect(summary.baselineMiB).toBe(3_000);
    expect(summary.rawMaximumMiB).toBe(8_000);
    expect(summary.maximumDeltaMiB).toBe(5_000);
  });

  it('recognizes only named high-memory preflight conflicts', () => {
    const listing = [
      'chrome.exe, 900',
      'msedge.exe, 900',
      'qemu-system-x86_64.exe, 700',
      'vmmemWSL, N/A',
      'vmware-vmx.exe, 512',
      'emulator.exe, 300',
    ].join('\n');
    expect(findGpuConflicts(listing)).toEqual(['qemu-system-x86_64.exe', 'vmware-vmx.exe']);
    expect(inspectNamedGpuProcesses(listing)).toEqual([
      { name: 'chrome.exe', usedMiB: 900 },
      { name: 'qemu-system-x86_64.exe', usedMiB: 700 },
      { name: 'vmmemWSL', usedMiB: null },
      { name: 'vmware-vmx.exe', usedMiB: 512 },
      { name: 'emulator.exe', usedMiB: 300 },
    ]);
  });

  it('targets exactly the runner-owned Windows PID tree', () => {
    expect(ownedTreeKillCommand(42, 'win32')).toEqual({
      command: 'taskkill.exe',
      args: ['/PID', '42', '/T', '/F'],
    });
    expect(() => ownedTreeKillCommand(0, 'win32')).toThrow('PID');
  });
});
