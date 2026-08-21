import type {
  ArtifactCacheStatus,
  ArtifactErrorCode,
  ArtifactOperation,
  WorkerProgress,
} from '../workers/protocol';

export type ArtifactCacheUiOperation =
  | null
  | 'inspect'
  | 'request-persistence'
  | 'download'
  | 'delete';

export type ArtifactCacheRetryTarget = 'inspect' | 'download' | 'delete';

export interface ArtifactCacheUiError {
  message: string;
  code?: ArtifactErrorCode;
  operation?: ArtifactOperation;
  retryable: boolean;
  retryTarget: ArtifactCacheRetryTarget;
}

export interface ArtifactDownloadProgress {
  currentFile: string;
  completedBytes: number;
  totalBytes: number;
  rate?: number;
  etaMs?: number;
}

export interface ArtifactCacheUiState {
  status: ArtifactCacheStatus | null;
  operation: ArtifactCacheUiOperation;
  lastError: ArtifactCacheUiError | null;
  persistenceWarning: string | null;
  downloadProgress: ArtifactDownloadProgress | null;
  notice: string | null;
}

export interface ArtifactCacheControls {
  canDownload: boolean;
  canRetry: boolean;
  canRefresh: boolean;
  canDelete: boolean;
  canCancel: boolean;
  canGenerate: boolean;
}

export function createArtifactCacheUiState(
  status: ArtifactCacheStatus | null = null,
): ArtifactCacheUiState {
  return {
    status,
    operation: null,
    lastError: null,
    persistenceWarning: null,
    downloadProgress: null,
    notice: null,
  };
}

export function deriveArtifactCacheControls(
  state: ArtifactCacheUiState,
  musicRunning = false,
): ArtifactCacheControls {
  const idle = state.operation === null;
  const incomplete = state.status?.state === 'missing' || state.status?.state === 'partial';
  const downloadable = idle && incomplete && state.status?.sufficient === true && !musicRunning;
  const downloadError = state.lastError?.retryTarget === 'download' ? state.lastError : null;
  const canRetry = downloadable && downloadError?.retryable === true;
  return {
    canDownload: downloadable && downloadError === null,
    canRetry,
    canRefresh: idle,
    canDelete: idle && !musicRunning && (state.status?.projectCacheCount ?? 0) > 0,
    canCancel: state.operation === 'download',
    canGenerate: idle && state.status?.state === 'ready',
  };
}

export function artifactDownloadActionLabel(state: ArtifactCacheUiState): string {
  if (deriveArtifactCacheControls(state).canRetry) {
    return 'Retry Download';
  }
  return state.status?.state === 'partial'
    ? 'Resume Download'
    : 'Download Model';
}

export type ArtifactCacheUiAction =
  | { type: 'operation-started'; operation: Exclude<ArtifactCacheUiOperation, null> }
  | { type: 'persistence-resolved'; warning?: string }
  | { type: 'download-started' }
  | { type: 'progress-received'; progress: WorkerProgress }
  | { type: 'status-received'; status: ArtifactCacheStatus; source: 'inspect' | 'download' | 'delete' }
  | { type: 'operation-failed'; error: ArtifactCacheUiError }
  | { type: 'download-cancelled' };

export function artifactCacheUiReducer(
  state: ArtifactCacheUiState,
  action: ArtifactCacheUiAction,
): ArtifactCacheUiState {
  if (action.type === 'operation-started') {
    return {
      ...state,
      operation: action.operation,
      downloadProgress: action.operation === 'download' ? state.downloadProgress : null,
    };
  }
  if (action.type === 'persistence-resolved') {
    return { ...state, persistenceWarning: action.warning ?? null };
  }
  if (action.type === 'download-started') {
    return {
      ...state,
      operation: 'download',
      lastError: null,
      notice: null,
      downloadProgress: null,
    };
  }
  if (action.type === 'progress-received') {
    if (
      state.operation !== 'download'
      || action.progress.stage !== 'artifact'
      || action.progress.currentFile === undefined
      || action.progress.completedBytes === undefined
      || action.progress.totalBytes === undefined
    ) return state;
    return {
      ...state,
      downloadProgress: {
        currentFile: action.progress.currentFile,
        completedBytes: action.progress.completedBytes,
        totalBytes: action.progress.totalBytes,
        rate: action.progress.rate,
        etaMs: action.progress.etaMs,
      },
    };
  }
  if (action.type === 'status-received') {
    const preserveInterruption = action.source === 'inspect'
      && (
        (state.lastError !== null && state.lastError.retryTarget !== 'inspect')
        || state.notice !== null
      );
    return {
      ...state,
      status: action.status,
      operation: null,
      lastError: preserveInterruption ? state.lastError : null,
      notice: preserveInterruption ? state.notice : null,
      downloadProgress: null,
    };
  }
  if (action.type === 'operation-failed') {
    const preserveInterruption = action.error.retryTarget === 'inspect'
      && (
        (state.lastError !== null && state.lastError.retryTarget !== 'inspect')
        || state.notice !== null
      );
    return {
      ...state,
      operation: null,
      lastError: preserveInterruption ? state.lastError : action.error,
      notice: preserveInterruption ? state.notice : null,
      downloadProgress: null,
    };
  }
  return {
    ...state,
    operation: null,
    lastError: null,
    notice: 'Download cancelled. Partial model files were kept.',
    downloadProgress: null,
  };
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  const unit = Math.max(
    0,
    Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1),
  );
  const value = bytes / 1024 ** unit;
  return `${unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

export function formatRate(bytesPerSecond: number): string {
  return `${formatBytes(bytesPerSecond)}/s`;
}

export function formatEta(milliseconds: number): string {
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) return '0s';
  const seconds = Math.ceil(milliseconds / 1_000);
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return minutes > 0 ? `${minutes}m ${remainder}s` : `${seconds}s`;
}

export function describeArtifactCacheStatus(state: ArtifactCacheUiState): string {
  const status = state.status;
  if (!status) return state.operation === 'inspect'
    ? 'Inspecting model files.'
    : 'Model file status is unavailable.';

  const verified = `${formatBytes(status.completeArtifactBytes)} of ${formatBytes(status.totalArtifactBytes)} verified`;
  const cache = status.state === 'ready'
    ? `Model files are ready (${verified}).`
    : status.state === 'partial'
      ? `Model files are partially downloaded (${verified}).`
      : `Model files are not downloaded (${verified}).`;
  const requiredHeadroom = `${formatBytes(status.requiredHeadroomBytes)} required headroom`;
  const capacity = status.sufficient === undefined || status.availableBytes === undefined
    ? `Available storage is unavailable (${requiredHeadroom}).`
    : status.sufficient
      ? `Storage capacity is sufficient (${formatBytes(status.availableBytes)} available, ${requiredHeadroom}).`
      : `Storage capacity is insufficient (${formatBytes(status.availableBytes)} available, ${requiredHeadroom}).`;
  const persistence = status.persistence === 'persistent'
    ? 'Storage is persistent.'
    : status.persistence === 'best-effort'
      ? 'Storage is best-effort and may be evicted.'
      : 'Persistence status is unavailable.';
  return `${cache} ${capacity} ${persistence}`;
}
