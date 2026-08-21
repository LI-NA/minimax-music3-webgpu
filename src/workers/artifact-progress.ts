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

  const emit = (progress: ArtifactProgress) => {
    send(progress);
    lastProgress = progress;
    lastReportedAt = now();
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
    report(path: string, loaded: number, total: number, completedBefore: number) {
      const next = progress(path, loaded, total, completedBefore);
      const timestamp = now();
      if (
        lastReportedAt === undefined
        || lastProgress?.currentFile !== next.currentFile
        || timestamp - lastReportedAt >= REPORT_INTERVAL_MS
      ) emit(next);
    },
    complete(path: string, total: number, completedBytes: number, cacheHit: boolean) {
      reportedCompletedBytes = Math.max(reportedCompletedBytes, completedBytes);
      const next: ArtifactProgress = {
        ...progress(path, total, total, completedBytes - total),
        completedBytes: reportedCompletedBytes,
        cacheHit,
      };
      emit(next);
    },
    discard() {
      lastProgress = undefined;
      lastReportedAt = undefined;
    },
  };
}
