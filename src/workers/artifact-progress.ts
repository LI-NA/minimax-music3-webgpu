import type { WorkerProgress } from './protocol';

export type ArtifactProgress = WorkerProgress;

export interface ArtifactProgressReporterOptions {
  totalBytes: number;
  send(progress: ArtifactProgress): void;
  now?(): number;
}

const REPORT_INTERVAL_MS = 100;

export function createArtifactProgressReporter({
  totalBytes,
  send,
  now = () => performance.now(),
}: ArtifactProgressReporterOptions) {
  let lastReportedAt: number | undefined;
  let lastProgress: ArtifactProgress | undefined;
  let reportedCompletedBytes = 0;
  let transferredBaseline: { bytes: number; timestamp: number } | undefined;

  const emit = (
    progress: ArtifactProgress,
    timestamp: number,
    currentCompletedBytes: number,
    transferredBytes?: number,
  ) => {
    let next = progress;
    if (transferredBytes !== undefined && Number.isFinite(transferredBytes) && transferredBytes > 0) {
      if (transferredBaseline !== undefined) {
        const transferredDelta = transferredBytes - transferredBaseline.bytes;
        const elapsedMs = timestamp - transferredBaseline.timestamp;
        if (transferredDelta > 0 && elapsedMs > 0) {
          const rate = (transferredDelta * 1_000) / elapsedMs;
          const etaMs = ((totalBytes - currentCompletedBytes) * 1_000) / rate;
          if (Number.isFinite(rate) && rate > 0 && Number.isFinite(etaMs)) next = { ...progress, rate, etaMs };
        }
      }
      // A caller that reports less than it did before is behind, not slower. Holding the earlier
      // baseline lets the next larger reading measure across the whole span instead of losing the
      // rate for one report and then overstating it on the next.
      if (transferredBaseline === undefined || transferredBytes >= transferredBaseline.bytes)
        transferredBaseline = { bytes: transferredBytes, timestamp };
    }
    send(next);
    lastProgress = next;
    lastReportedAt = timestamp;
  };
  const progress = (path: string, loaded: number, total: number, completedBefore: number): ArtifactProgress => {
    reportedCompletedBytes = Math.max(reportedCompletedBytes, completedBefore + loaded);
    return {
      type: 'progress',
      stage: 'artifact',
      detail: path,
      loaded,
      total,
      currentFile: path,
      completedBytes: reportedCompletedBytes,
      totalBytes,
      cacheHit: false,
    };
  };

  return {
    report(path: string, loaded: number, total: number, completedBefore: number, transferredBytes?: number) {
      const next = progress(path, loaded, total, completedBefore);
      const timestamp = now();
      const currentCompletedBytes = Math.min(totalBytes, Math.max(0, completedBefore + loaded));
      if (
        lastReportedAt === undefined ||
        lastProgress?.currentFile !== next.currentFile ||
        timestamp - lastReportedAt >= REPORT_INTERVAL_MS
      )
        emit(next, timestamp, currentCompletedBytes, transferredBytes);
    },
    complete(path: string, total: number, completedBytes: number, cacheHit: boolean, transferredBytes?: number) {
      reportedCompletedBytes = Math.max(reportedCompletedBytes, completedBytes);
      const next: ArtifactProgress = {
        ...progress(path, total, total, completedBytes - total),
        completedBytes: reportedCompletedBytes,
        cacheHit,
      };
      const currentCompletedBytes = Math.min(totalBytes, Math.max(0, completedBytes));
      emit(next, now(), currentCompletedBytes, transferredBytes);
    },
    discard() {
      lastProgress = undefined;
      lastReportedAt = undefined;
    },
  };
}
