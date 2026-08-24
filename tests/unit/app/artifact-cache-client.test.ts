import { afterEach, describe, expect, it, vi } from 'vitest';
import { createArtifactCacheClient } from '../../../src/app/artifact-cache-client';
import type { ArtifactCacheUiAction } from '../../../src/app/artifact-cache-ui';
import type { ArtifactCacheStatus, WorkerResponse } from '../../../src/workers/protocol';

const MANIFEST_URL = '/artifacts/music-variable/manifest.json';

const status = (): ArtifactCacheStatus => ({
  manifestHash: 'release',
  state: 'ready',
  artifactCount: 1,
  totalArtifactBytes: 1_024,
  completeArtifactCount: 1,
  completeArtifactBytes: 1_024,
  storedReferencedBytes: 1_024,
  additionalBytesNeeded: 0,
  largestPendingArtifactBytes: 0,
  projectCacheCount: 1,
  projectCacheBytes: 1_024,
  persistence: 'persistent',
  requiredHeadroomBytes: 0,
});

class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  readonly posted: unknown[] = [];
  terminated = false;

  constructor() {
    FakeWorker.instances.push(this);
  }
  postMessage(request: unknown) {
    this.posted.push(request);
  }
  terminate() {
    this.terminated = true;
  }
  respond(response: WorkerResponse) {
    this.onmessage?.({ data: response } as MessageEvent<WorkerResponse>);
  }
}

function harness() {
  FakeWorker.instances = [];
  const actions: ArtifactCacheUiAction[] = [];
  const client = createArtifactCacheClient({
    manifestUrl: MANIFEST_URL,
    dispatch: (action) => actions.push(action),
    isMounted: () => true,
    createWorker: () => new FakeWorker() as unknown as Worker,
  });
  return { actions, client, workers: FakeWorker.instances };
}

describe('artifact cache client', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reports an inspected status and releases the worker', () => {
    const { actions, client, workers } = harness();

    client.inspect();
    expect(workers[0].posted).toEqual([{ type: 'inspect-artifact-cache', manifestUrl: MANIFEST_URL }]);

    workers[0].respond({ type: 'artifact-cache-status', status: status() });

    expect(actions).toEqual([
      { type: 'operation-started', operation: 'inspect' },
      { type: 'status-received', source: 'inspect', status: status() },
    ]);
    expect(workers[0].terminated).toBe(true);
  });

  it('marks a failed deletion retryable and refreshes the status afterwards', () => {
    const { actions, client, workers } = harness();

    client.remove();
    workers[0].respond({
      type: 'error',
      message: 'Artifact cache deletion failed',
      code: 'cache-delete-failed',
      operation: 'delete-artifact-caches',
      retryable: true,
    });

    expect(actions).toEqual([
      { type: 'operation-started', operation: 'delete' },
      {
        type: 'operation-failed',
        error: {
          message: 'Artifact cache deletion failed',
          code: 'cache-delete-failed',
          operation: 'delete-artifact-caches',
          retryable: true,
          retryTarget: 'delete',
        },
      },
      { type: 'operation-started', operation: 'inspect' },
    ]);
    expect(workers[0].terminated).toBe(true);
    expect(workers[1].posted).toEqual([{ type: 'inspect-artifact-cache', manifestUrl: MANIFEST_URL }]);
  });

  it('waits for download cancellation before terminating the worker and refreshing status', async () => {
    const { actions, client, workers } = harness();
    vi.stubGlobal('navigator', {
      storage: {
        persist: vi.fn(async () => true),
        persisted: vi.fn(async () => true),
      },
    });

    await client.download();
    client.cancelDownload();

    expect(workers[0].posted).toEqual([
      { type: 'download-artifacts', manifestUrl: MANIFEST_URL },
      { type: 'cancel-artifact-download' },
    ]);
    expect(workers[0].terminated).toBe(false);
    expect(actions).not.toContainEqual({ type: 'download-cancelled' });

    client.cancelDownload();
    expect(workers[0].posted).toHaveLength(2);

    workers[0].respond({ type: 'artifact-download-cancelled' });

    expect(workers[0].terminated).toBe(true);
    expect(actions).toEqual([
      { type: 'operation-started', operation: 'request-persistence' },
      { type: 'persistence-resolved', warning: undefined },
      { type: 'download-started' },
      { type: 'download-cancelled' },
      { type: 'operation-started', operation: 'inspect' },
    ]);
    expect(workers).toHaveLength(2);
    expect(workers[1].posted).toEqual([{ type: 'inspect-artifact-cache', manifestUrl: MANIFEST_URL }]);
  });

  it('stops the active worker when the view is torn down', () => {
    const { client, workers } = harness();

    client.inspect();
    client.terminate();

    expect(workers[0].terminated).toBe(true);
  });
});
