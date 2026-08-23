import type { ArtifactStore } from './artifact-cache';
import type { ArtifactFile } from './manifest';

export type ArtifactCacheState = 'missing' | 'partial' | 'ready';

export interface ArtifactCacheInspection {
  state: ArtifactCacheState;
  artifactCount: number;
  totalArtifactBytes: number;
  completeArtifactCount: number;
  completeArtifactBytes: number;
  storedReferencedBytes: number;
  additionalBytesNeeded: number;
  largestPendingArtifactBytes: number;
}

export interface ArtifactCapacityAssessment {
  usageBytes?: number;
  quotaBytes?: number;
  availableBytes?: number;
  sufficient?: boolean;
  requiredHeadroomBytes: number;
}

export interface ProjectCacheUsage {
  cacheCount: number;
  storedBytes: number;
}

/**
 * Why the browser refused a durable grant. The UI turns the code into localised prose, so the
 * runtime never hands a user-facing English sentence to a Korean screen.
 */
export type PersistenceWarning = 'unsupported' | 'denied' | 'failed';

export type PersistenceRequestResult =
  { state: 'persistent'; warning?: never } | { state: 'best-effort'; warning: PersistenceWarning };

const PROJECT_CACHE_NAME = /^minimax-music3-[a-f0-9]{64}$/;

interface IterableDirectoryHandle extends FileSystemDirectoryHandle {
  entries(): AsyncIterableIterator<[string, FileSystemFileHandle | IterableDirectoryHandle]>;
}

const iterable = (directory: FileSystemDirectoryHandle) => directory as IterableDirectoryHandle;

const directorySize = async (directory: IterableDirectoryHandle): Promise<number> => {
  let storedBytes = 0;
  for await (const [, handle] of directory.entries()) {
    storedBytes += handle.kind === 'file' ? (await handle.getFile()).size : await directorySize(handle);
  }
  return storedBytes;
};

export async function inspectProjectArtifactCaches(root: FileSystemDirectoryHandle): Promise<ProjectCacheUsage> {
  let cacheCount = 0;
  let storedBytes = 0;
  for await (const [name, handle] of iterable(root).entries()) {
    if (handle.kind !== 'directory' || !PROJECT_CACHE_NAME.test(name)) continue;
    cacheCount++;
    storedBytes += await directorySize(handle);
  }
  return { cacheCount, storedBytes };
}

export async function deleteProjectArtifactCaches(root: FileSystemDirectoryHandle): Promise<void> {
  for await (const [name, handle] of iterable(root).entries()) {
    if (handle.kind !== 'directory' || !PROJECT_CACHE_NAME.test(name)) continue;
    try {
      await root.removeEntry(name, { recursive: true });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'NotFoundError') continue;
      throw new Error(`failed to delete artifact cache: ${name}`, { cause: error });
    }
  }
}

export async function withArtifactCacheMutationLock<T>(
  action: () => Promise<T>,
  locks: Pick<LockManager, 'request'> | undefined = typeof navigator === 'undefined' ? undefined : navigator.locks,
): Promise<T> {
  if (!locks) throw new Error('artifact cache mutation lock is unavailable');
  return locks.request('minimax-music3-artifact-cache', { mode: 'exclusive' }, action);
}

export async function withArtifactCacheReadLock<T>(
  action: () => Promise<T>,
  locks: Pick<LockManager, 'request'> | undefined = typeof navigator === 'undefined' ? undefined : navigator.locks,
): Promise<T> {
  if (!locks) throw new Error('artifact cache read lock is unavailable');
  return locks.request('minimax-music3-artifact-cache', { mode: 'shared' }, action);
}

export async function requestPersistentStorage(
  storage: Pick<StorageManager, 'persisted' | 'persist'> | undefined = typeof navigator === 'undefined'
    ? undefined
    : navigator.storage,
): Promise<PersistenceRequestResult> {
  if (!storage) return { state: 'best-effort', warning: 'unsupported' };
  try {
    const requested = storage.persist();
    const existing = storage.persisted();
    const [granted, alreadyPersistent] = await Promise.all([requested, existing]);
    if (granted || alreadyPersistent) return { state: 'persistent' };
    return { state: 'best-effort', warning: 'denied' };
  } catch {
    return { state: 'best-effort', warning: 'failed' };
  }
}

export async function inspectArtifactCache(
  artifacts: readonly ArtifactFile[],
  store: Pick<ArtifactStore, 'size' | 'isComplete'> | undefined,
): Promise<ArtifactCacheInspection> {
  const totalArtifactBytes = artifacts.reduce((sum, artifact) => sum + artifact.bytes, 0);
  if (store) {
    const entries = await Promise.all(
      artifacts.map(async (artifact) => {
        const size = await store.size(artifact.path);
        const complete = size === artifact.bytes && (await store.isComplete(artifact.path));
        return { artifact, size, complete };
      }),
    );
    const completeEntries = entries.filter((entry) => entry.complete);
    const pendingEntries = entries.filter((entry) => !entry.complete);
    return {
      state: pendingEntries.length === 0 ? 'ready' : 'partial',
      artifactCount: artifacts.length,
      totalArtifactBytes,
      completeArtifactCount: completeEntries.length,
      completeArtifactBytes: completeEntries.reduce((sum, entry) => sum + entry.artifact.bytes, 0),
      storedReferencedBytes: entries.reduce((sum, entry) => sum + entry.size, 0),
      additionalBytesNeeded: entries.reduce(
        (sum, entry) => sum + Math.max(entry.artifact.bytes - Math.min(entry.size, entry.artifact.bytes), 0),
        0,
      ),
      largestPendingArtifactBytes: Math.max(0, ...pendingEntries.map((entry) => entry.artifact.bytes)),
    };
  }
  return {
    state: 'missing',
    artifactCount: artifacts.length,
    totalArtifactBytes,
    completeArtifactCount: 0,
    completeArtifactBytes: 0,
    storedReferencedBytes: 0,
    additionalBytesNeeded: totalArtifactBytes,
    largestPendingArtifactBytes: Math.max(0, ...artifacts.map((artifact) => artifact.bytes)),
  };
}

export function assessArtifactCapacity(
  inspection: ArtifactCacheInspection,
  estimate?: Pick<StorageEstimate, 'usage' | 'quota'>,
): ArtifactCapacityAssessment {
  const requiredHeadroomBytes =
    inspection.state === 'ready' ? 0 : inspection.additionalBytesNeeded + inspection.largestPendingArtifactBytes;
  const usageBytes = estimate?.usage;
  const quotaBytes = estimate?.quota;
  if (
    usageBytes === undefined ||
    quotaBytes === undefined ||
    !Number.isSafeInteger(usageBytes) ||
    !Number.isSafeInteger(quotaBytes) ||
    usageBytes < 0 ||
    quotaBytes < usageBytes
  )
    return { requiredHeadroomBytes };
  const availableBytes = quotaBytes - usageBytes;
  return {
    usageBytes,
    quotaBytes,
    availableBytes,
    sufficient: availableBytes >= requiredHeadroomBytes,
    requiredHeadroomBytes,
  };
}
