import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';
import {
  assertFreshQualificationCapture,
  resolveQualificationCapture,
  resolveQualificationProfile,
} from './qualification-paths';

export type GpuSample = {
  at: string;
  monotonicMs: number;
  usedMiB: number;
  totalMiB: number;
  utilizationPercent: number;
};

export function parseNvidiaSample(line: string, monotonicMs: number): GpuSample | undefined {
  const match = /^(.*),\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*$/.exec(line.trim());
  if (!match) return undefined;
  const usedMiB = Number(match[2]);
  const totalMiB = Number(match[3]);
  const utilizationPercent = Number(match[4]);
  if (!Number.isSafeInteger(usedMiB) || !Number.isSafeInteger(totalMiB) || !Number.isSafeInteger(utilizationPercent))
    return undefined;
  return { at: match[1].trim(), monotonicMs, usedMiB, totalMiB, utilizationPercent };
}

const vmConflictPattern = /^(?:qemu.*|vmmem(?:wsl)?|vmware-vmx(?:\.exe)?|emulator(?:\.exe)?|.*android.*)$/i;
const namedGpuProcessPattern =
  /^(?:chrome(?:\.exe)?|qemu.*|vmmem(?:wsl)?|vmware-vmx(?:\.exe)?|emulator(?:\.exe)?|.*android.*)$/i;

export function inspectNamedGpuProcesses(processListing: string) {
  return processListing
    .split(/\r?\n/)
    .map((line) => line.trim())
    .flatMap((line) => {
      const match = /^(.*),\s*(\d+|N\/A)\s*$/i.exec(line);
      if (!match) return [];
      const name = match[1].trim().replace(/^"|"$/g, '').split(/[\\/]/).at(-1) ?? '';
      if (!namedGpuProcessPattern.test(name)) return [];
      return [{ name, usedMiB: /^\d+$/.test(match[2]) ? Number(match[2]) : null }];
    });
}

export function findGpuConflicts(processListing: string) {
  return inspectNamedGpuProcesses(processListing)
    .filter(({ name, usedMiB }) => vmConflictPattern.test(name) && usedMiB !== null && usedMiB >= 512)
    .map(({ name }) => name);
}

export function ownedTreeKillCommand(pid: number, platform: NodeJS.Platform) {
  if (!Number.isSafeInteger(pid) || pid < 1) throw new Error('owned process PID must be positive');
  return platform === 'win32'
    ? { command: 'taskkill.exe', args: ['/PID', String(pid), '/T', '/F'] }
    : { command: 'kill', args: ['-TERM', String(-pid)] };
}

export async function raceWithCancellableTimeout<T>(
  operation: Promise<T>,
  timeoutMs: number,
  timeoutError: () => Error,
  schedule: (callback: () => void, milliseconds: number) => unknown = (callback, milliseconds) =>
    setTimeout(callback, milliseconds),
  cancel: (handle: unknown) => void = (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
) {
  let timeoutHandle: unknown;
  const timeout = new Promise<never>((_resolve, reject) => {
    timeoutHandle = schedule(() => reject(timeoutError()), timeoutMs);
  });
  try {
    return await Promise.race([operation, timeout]);
  } finally {
    if (timeoutHandle !== undefined) cancel(timeoutHandle);
  }
}

const rounded = (value: number) => Number(value.toFixed(3));

export function summarizeTelemetry(
  samples: readonly GpuSample[],
  requestedCadenceMs: number,
  baselineIndex: number,
  incrementalLimitMiB: number,
  capacityReserveMiB: number,
) {
  if (samples.length === 0) throw new Error('GPU telemetry did not produce any samples');
  if (!Number.isSafeInteger(baselineIndex) || baselineIndex < 0 || baselineIndex >= samples.length)
    throw new Error('GPU telemetry baseline index is invalid');
  const baseline = samples[baselineIndex];
  const measured = samples.slice(baselineIndex);
  const physicalTotalMiB = baseline.totalMiB;
  const capacityGuardMiB = physicalTotalMiB - capacityReserveMiB;
  const intervals = samples.slice(1).map((sample, index) => sample.monotonicMs - samples[index].monotonicMs);
  const sorted = [...intervals].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const median =
    sorted.length === 0 ? null : sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  return {
    requestedCadenceMs,
    effectiveCadenceMs: {
      mean:
        intervals.length === 0
          ? null
          : rounded(intervals.reduce((total, value) => total + value, 0) / intervals.length),
      median: median === null ? null : rounded(median),
      maximum: intervals.length === 0 ? null : rounded(Math.max(...intervals)),
    },
    baselineMiB: baseline.usedMiB,
    physicalTotalMiB,
    rawMaximumMiB: Math.max(...measured.map(({ usedMiB }) => usedMiB)),
    rawFinalMiB: samples.at(-1)!.usedMiB,
    maximumDeltaMiB: Math.max(...measured.map(({ usedMiB }) => usedMiB - baseline.usedMiB)),
    incrementalLimitMiB,
    incrementalExceeded: measured.some(({ usedMiB }) => usedMiB - baseline.usedMiB > incrementalLimitMiB),
    capacityReserveMiB,
    capacityGuardMiB,
    capacityGuardExceeded: measured.some(({ usedMiB }) => usedMiB >= capacityGuardMiB),
    baselineUtilizationPercent: baseline.utilizationPercent,
    maximumUtilizationPercent: Math.max(...measured.map(({ utilizationPercent }) => utilizationPercent)),
    finalUtilizationPercent: samples.at(-1)!.utilizationPercent,
    sampleCount: samples.length,
  };
}

function stopOwnedGate(gate: ChildProcess | undefined) {
  if (!gate?.pid || gate.exitCode !== null) return;
  if (process.platform === 'win32') {
    const kill = ownedTreeKillCommand(gate.pid, process.platform);
    spawnSync(kill.command, kill.args, {
      stdio: 'ignore',
      windowsHide: true,
    });
    return;
  }
  try {
    process.kill(-gate.pid, 'SIGTERM');
  } catch {
    gate.kill('SIGTERM');
  }
}

export function selectBrowserGate(value: string | undefined) {
  if (value === undefined || value === 'variable-duration') return { spec: 'tests/browser/variable-duration.spec.ts' };
  if (value === 'long-duration') return { spec: 'tests/browser/long-duration.spec.ts' };
  throw new Error('MINIMAX_VARIABLE_GATE must use the browser gate allowlist');
}

function spawnGate(captureDirectory: string, chromeProfile: string) {
  const { spec } = selectBrowserGate(process.env.MINIMAX_VARIABLE_GATE);
  const environment = {
    ...process.env,
    MINIMAX_RELEASE: 'music-variable',
    MINIMAX_VARIABLE_CAPTURE_DIR: captureDirectory,
    MINIMAX_VARIABLE_CHROME_PROFILE: chromeProfile,
  };
  if (process.platform === 'win32') {
    return spawn(
      process.env.ComSpec ?? 'cmd.exe',
      ['/d', '/s', '/c', `npx playwright test ${spec} --project=chrome --workers=1`],
      { env: environment, stdio: 'inherit', windowsHide: true },
    );
  }
  return spawn('npx', ['playwright', 'test', spec, '--project=chrome', '--workers=1'], {
    env: environment,
    stdio: 'inherit',
    detached: true,
  });
}

export function writeEvidence(
  captureDirectory: string,
  samples: readonly GpuSample[],
  result: Record<string, unknown>,
) {
  mkdirSync(captureDirectory, { recursive: true });
  const csv = [
    'nvidia_timestamp,host_monotonic_ms,memory_used_mib,memory_total_mib,gpu_utilization_percent',
    ...samples.map(
      (sample) =>
        `${sample.at},${sample.monotonicMs.toFixed(3)},${sample.usedMiB},${sample.totalMiB},${sample.utilizationPercent}`,
    ),
  ];
  writeFileSync(path.join(captureDirectory, 'gpu-samples.csv'), `${csv.join('\n')}\n`, {
    encoding: 'utf8',
    flag: 'wx',
  });
  writeFileSync(path.join(captureDirectory, 'telemetry.json'), `${JSON.stringify(result, null, 2)}\n`, {
    encoding: 'utf8',
    flag: 'wx',
  });
}

function positiveEnvironmentNumber(name: string, fallback: number) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be positive`);
  return value;
}

function materialGpuProcessListing() {
  const result = spawnSync(
    'nvidia-smi',
    ['--query-compute-apps=process_name,used_memory', '--format=csv,noheader,nounits'],
    {
      encoding: 'utf8',
      windowsHide: true,
    },
  );
  if (result.error || result.status !== 0)
    throw new Error(`process preflight failed: ${result.error?.message ?? result.stderr}`);
  return result.stdout;
}

export async function runTelemetryGate() {
  const checkoutRoot = path.resolve('.');
  const linkedWorktree = statSync(path.join(checkoutRoot, '.git')).isFile();
  const captureDirectory = resolveQualificationCapture(checkoutRoot, process.env.MINIMAX_VARIABLE_CAPTURE_DIR);
  assertFreshQualificationCapture(captureDirectory, existsSync);
  const chromeProfile = resolveQualificationProfile(
    checkoutRoot,
    process.env.MINIMAX_VARIABLE_CHROME_PROFILE,
    linkedWorktree,
  );
  const gpuIndex = Number(process.env.MINIMAX_NVIDIA_GPU_INDEX ?? '0');
  if (!Number.isSafeInteger(gpuIndex) || gpuIndex < 0)
    throw new Error('MINIMAX_NVIDIA_GPU_INDEX must be a non-negative integer');
  const requestedCadenceMs = 100;
  const incrementalLimitMiB = 12_288;
  const capacityReserveMiB = 512;
  const preflightTimeoutMs = positiveEnvironmentNumber('MINIMAX_PREFLIGHT_TIMEOUT_MS', 600_000);
  const samples: GpuSample[] = [];
  let gate: ChildProcess | undefined;
  let incrementalExceeded = false;
  let capacityGuardExceeded = false;
  let baselineIndex: number | undefined;
  let monitorFailure: string | undefined;
  let lineBuffer = '';
  let cleanStreak = 0;
  const profileIsLocked = () =>
    ['SingletonLock', 'SingletonCookie', 'SingletonSocket', 'lockfile'].some((name) =>
      existsSync(path.join(chromeProfile, name)),
    );
  const observedNamedGpuProcesses = new Map<string, number | null>();
  const readBlockingConflicts = () => {
    const listing = materialGpuProcessListing();
    for (const process of inspectNamedGpuProcesses(listing))
      observedNamedGpuProcesses.set(process.name, process.usedMiB);
    return [...findGpuConflicts(listing), ...(profileIsLocked() ? [`Chrome profile lock: ${chromeProfile}`] : [])];
  };
  let currentConflicts = readBlockingConflicts();
  const observedConflicts = new Set(currentConflicts);
  let lastProcessCheckMs = performance.now();
  let resolveClean: (() => void) | undefined;
  let rejectClean: ((error: Error) => void) | undefined;
  const cleanPreflight = new Promise<void>((resolve, reject) => {
    resolveClean = resolve;
    rejectClean = reject;
  });
  const monitor = spawn(
    'nvidia-smi',
    [
      `--id=${gpuIndex}`,
      '--query-gpu=timestamp,memory.used,memory.total,utilization.gpu',
      '--format=csv,noheader,nounits',
      '--loop-ms=100',
    ],
    { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true },
  );
  monitor.stdout.setEncoding('utf8');
  monitor.stderr.setEncoding('utf8');
  monitor.stdout.on('data', (chunk: string) => {
    lineBuffer += chunk;
    const lines = lineBuffer.split(/\r?\n/);
    lineBuffer = lines.pop() ?? '';
    for (const line of lines) {
      const sample = parseNvidiaSample(line, performance.now());
      if (!sample) continue;
      samples.push(sample);
      if (!capacityGuardExceeded && sample.usedMiB >= sample.totalMiB - capacityReserveMiB) {
        capacityGuardExceeded = true;
        stopOwnedGate(gate);
        resolveClean?.();
        resolveClean = undefined;
      }
      if (
        baselineIndex !== undefined &&
        !incrementalExceeded &&
        sample.usedMiB - samples[baselineIndex].usedMiB > incrementalLimitMiB
      ) {
        incrementalExceeded = true;
        stopOwnedGate(gate);
      }
      if (!gate && !capacityGuardExceeded) {
        if (sample.monotonicMs - lastProcessCheckMs >= 1_000) {
          try {
            currentConflicts = readBlockingConflicts();
            currentConflicts.forEach((name) => observedConflicts.add(name));
            lastProcessCheckMs = sample.monotonicMs;
          } catch (error) {
            rejectClean?.(error instanceof Error ? error : new Error(String(error)));
            rejectClean = undefined;
          }
        }
        const clean = currentConflicts.length === 0;
        cleanStreak = clean ? cleanStreak + 1 : 0;
        if (cleanStreak >= 3) {
          baselineIndex = samples.length - 1;
          resolveClean?.();
          resolveClean = undefined;
        }
      }
    }
  });
  monitor.stderr.on('data', (chunk: string) => {
    monitorFailure = `${monitorFailure ?? ''}${chunk}`.trim();
  });
  monitor.on('error', (error) => {
    monitorFailure = error.message;
    rejectClean?.(error);
    stopOwnedGate(gate);
  });
  monitor.on('exit', (code) => {
    if (code !== null && code !== 0 && !monitorFailure) monitorFailure = `nvidia-smi exited with code ${code}`;
    if (samples.length === 0) rejectClean?.(new Error(monitorFailure ?? 'nvidia-smi exited before its first sample'));
    if (gate && gate.exitCode === null) stopOwnedGate(gate);
  });

  let gateExitCode: number | null = null;
  let preflightFailure: string | undefined;
  try {
    try {
      await raceWithCancellableTimeout(cleanPreflight, preflightTimeoutMs, () => {
        const conflicts = currentConflicts.length ? `; conflicts: ${currentConflicts.join(', ')}` : '';
        return new Error(`clean GPU preflight timed out${conflicts}`);
      });
    } catch (error) {
      preflightFailure = error instanceof Error ? error.message : String(error);
    }
    if (!capacityGuardExceeded && !monitorFailure && !preflightFailure) {
      assertFreshQualificationCapture(captureDirectory, existsSync);
      gate = spawnGate(captureDirectory, chromeProfile);
      gateExitCode = await new Promise<number | null>((resolve, reject) => {
        gate!.once('error', reject);
        gate!.once('close', resolve);
      });
      await delay(2_000);
    }
  } finally {
    monitor.kill();
  }
  const summary = summarizeTelemetry(
    samples,
    requestedCadenceMs,
    baselineIndex ?? 0,
    incrementalLimitMiB,
    capacityReserveMiB,
  );
  const result = {
    gpuIndex,
    ...summary,
    cleanPreflight: {
      requiredConsecutiveSamples: 3,
      achievedConsecutiveSamples: cleanStreak,
      timeoutMs: preflightTimeoutMs,
      observedConflicts: [...observedConflicts],
      remainingConflicts: currentConflicts,
      observedNamedGpuProcesses: [...observedNamedGpuProcesses].map(([name, usedMiB]) => ({
        name,
        usedMiB,
      })),
    },
    gateExitCode,
    preflightFailure: preflightFailure || null,
    monitorFailure: monitorFailure || null,
    stoppedOwnedGate: Boolean(gate) && (incrementalExceeded || capacityGuardExceeded || Boolean(monitorFailure)),
  };
  writeEvidence(captureDirectory, samples, result);
  return incrementalExceeded || capacityGuardExceeded || monitorFailure || preflightFailure || gateExitCode !== 0
    ? 1
    : 0;
}

const invokedPath = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : undefined;
if (invokedPath === import.meta.url) {
  void runTelemetryGate()
    .then((exitCode) => {
      process.exitCode = exitCode;
    })
    .catch((error: unknown) => {
      process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
      process.exitCode = 1;
    });
}
