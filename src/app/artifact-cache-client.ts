import { requestPersistentStorage } from '../runtime/model/artifact-cache-management';
import type { ArtifactCacheRequest, ArtifactOperation, WorkerResponse } from '../workers/protocol';
import type {
  ArtifactCacheRetryTarget,
  ArtifactCacheUiAction,
  ArtifactCacheUiError,
  ArtifactCacheUiOperation,
} from './artifact-cache-ui';

export type ArtifactCacheClientOperation = Exclude<ArtifactCacheUiOperation, null | 'request-persistence'>;

export interface ArtifactCacheClientOptions {
  manifestUrl: string;
  dispatch: (action: ArtifactCacheUiAction) => void;
  isMounted: () => boolean;
  createWorker: () => Worker;
}

export interface ArtifactCacheClient {
  inspect(): void;
  download(): Promise<void>;
  cancelDownload(): void;
  remove(): void;
  terminate(): void;
}

function retryTarget(
  operation: ArtifactCacheClientOperation,
  protocolOperation?: ArtifactOperation,
): ArtifactCacheRetryTarget {
  if (protocolOperation === 'download-artifacts' || operation === 'download') return 'download';
  if (protocolOperation === 'delete-artifact-caches' || operation === 'delete') return 'delete';
  return 'inspect';
}

function protocolOperationOf(operation: ArtifactCacheClientOperation): ArtifactOperation {
  if (operation === 'download') return 'download-artifacts';
  if (operation === 'delete') return 'delete-artifact-caches';
  return 'inspect-artifact-cache';
}

/**
 * Drives the inference worker for artifact cache inspection, download and deletion.
 * One worker runs at a time; a new request replaces the previous one.
 */
export function createArtifactCacheClient({
  manifestUrl,
  dispatch,
  isMounted,
  createWorker,
}: ArtifactCacheClientOptions): ArtifactCacheClient {
  let active: Worker | null = null;
  let persistenceGeneration = 0;

  const finish = (worker: Worker): boolean => {
    if (active !== worker) return false;
    active = null;
    worker.terminate();
    return true;
  };

  const runtimeError = (operation: ArtifactCacheClientOperation, error: unknown): ArtifactCacheUiError => ({
    message: error instanceof Error && error.message ? error.message : 'Model file worker failed',
    operation: protocolOperationOf(operation),
    retryable: true,
    retryTarget: retryTarget(operation),
  });

  const fail = (operation: ArtifactCacheClientOperation, error: ArtifactCacheUiError, worker?: Worker) => {
    if (worker && !finish(worker)) return;
    dispatch({ type: 'operation-failed', error });
    if (operation === 'download' || operation === 'delete') inspect();
  };

  const run = (request: ArtifactCacheRequest, operation: ArtifactCacheClientOperation) => {
    const previous = active;
    active = null;
    previous?.terminate();
    let next: Worker;
    try {
      next = createWorker();
    } catch (error) {
      fail(operation, runtimeError(operation, error));
      return;
    }
    active = next;
    next.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (active !== next) return;
      if (data.type === 'progress') {
        if (operation === 'download' && data.stage === 'artifact')
          dispatch({ type: 'progress-received', progress: data });
        return;
      }
      if (
        data.type === 'artifact-cache-status' ||
        data.type === 'artifact-download-complete' ||
        data.type === 'artifact-cache-deleted'
      ) {
        const source =
          data.type === 'artifact-cache-status'
            ? 'inspect'
            : data.type === 'artifact-download-complete'
              ? 'download'
              : 'delete';
        dispatch({ type: 'status-received', source, status: data.status });
        finish(next);
        return;
      }
      if (data.type === 'error') {
        fail(
          operation,
          {
            message: data.message,
            code: data.code,
            operation: data.operation,
            retryable: data.retryable === true,
            retryTarget: retryTarget(operation, data.operation),
          },
          next,
        );
      }
    };
    next.onerror = (event) => {
      if (active !== next) return;
      event.preventDefault();
      fail(operation, runtimeError(operation, event.error ?? new Error(event.message)), next);
    };
    try {
      next.postMessage(request);
    } catch (error) {
      fail(operation, runtimeError(operation, error), next);
    }
  };

  const inspect = () => {
    persistenceGeneration++;
    dispatch({ type: 'operation-started', operation: 'inspect' });
    run({ type: 'inspect-artifact-cache', manifestUrl }, 'inspect');
  };

  const download = async () => {
    const requestGeneration = ++persistenceGeneration;
    dispatch({ type: 'operation-started', operation: 'request-persistence' });
    const persistence = await requestPersistentStorage(navigator.storage);
    if (!isMounted() || persistenceGeneration !== requestGeneration) return;
    dispatch({ type: 'persistence-resolved', warning: persistence.warning });
    dispatch({ type: 'download-started' });
    run({ type: 'download-artifacts', manifestUrl }, 'download');
  };

  const cancelDownload = () => {
    const previous = active;
    active = null;
    previous?.terminate();
    dispatch({ type: 'download-cancelled' });
    inspect();
  };

  const remove = () => {
    persistenceGeneration++;
    dispatch({ type: 'operation-started', operation: 'delete' });
    run({ type: 'delete-artifact-caches', manifestUrl }, 'delete');
  };

  const terminate = () => {
    persistenceGeneration++;
    const previous = active;
    active = null;
    previous?.terminate();
  };

  return { inspect, download, cancelDownload, remove, terminate };
}
