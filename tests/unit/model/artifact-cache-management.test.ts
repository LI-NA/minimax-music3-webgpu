import { describe, expect, it, vi } from 'vitest';
import {
  assessArtifactCapacity,
  deleteProjectArtifactCaches,
  inspectArtifactCache,
  inspectProjectArtifactCaches,
  requestPersistentStorage,
  withArtifactCacheMutationLock,
  withArtifactCacheReadLock,
  type ArtifactCacheInspection,
} from '../../../src/runtime/model/artifact-cache-management';
import type { ArtifactStore } from '../../../src/runtime/model/artifact-cache';

const artifacts = [
  { path: 'a.bin', bytes: 10, sha256: 'a'.repeat(64) },
  { path: 'b.bin', bytes: 20, sha256: 'b'.repeat(64) },
];

const storeWith = (
  sizes: Readonly<Record<string, number>> = {},
  complete: readonly string[] = [],
): Pick<ArtifactStore, 'size' | 'isComplete'> => ({
  size: async (path) => sizes[path] ?? 0,
  isComplete: async (path) => complete.includes(path),
});

interface FakeDirectoryEntries {
  [name: string]: File | FakeDirectoryEntries;
}

const fakeDirectory = (entries: FakeDirectoryEntries): FileSystemDirectoryHandle => ({
  kind: 'directory',
  name: '',
  async *entries() {
    for (const [name, entry] of Object.entries(entries)) {
      if (entry instanceof File) {
        yield [name, {
          kind: 'file',
          name,
          getFile: async () => entry,
        } as unknown as FileSystemFileHandle];
      } else {
        yield [name, fakeDirectory(entry)];
      }
    }
  },
} as unknown as FileSystemDirectoryHandle);

describe('project artifact cache management', () => {
  const a = `minimax-music3-${'a'.repeat(64)}`;
  const b = `minimax-music3-${'b'.repeat(64)}`;
  const c = `minimax-music3-${'c'.repeat(64)}`;
  const uppercase = `minimax-music3-${'A'.repeat(64)}`;

  it('recursively totals files only in strictly matching direct cache directories', async () => {
    const root = fakeDirectory({
      [a]: {
        'artifact.bin': new File([new Uint8Array(5)], 'artifact.bin'),
        nested: {
          'artifact.complete': new File([new Uint8Array(1)], 'artifact.complete'),
        },
      },
      [b]: { 'model.bin': new File([new Uint8Array(8)], 'model.bin') },
      [uppercase]: { 'ignored.bin': new File([new Uint8Array(20)], 'ignored.bin') },
      'minimax-music3-short': {},
      notes: {},
      [a + '-extra']: {},
      [c]: new File([new Uint8Array(50)], c),
    });

    await expect(inspectProjectArtifactCaches(root)).resolves.toEqual({
      cacheCount: 2,
      storedBytes: 14,
    });
  });

  it('deletes only strictly matching direct cache directories', async () => {
    const root = fakeDirectory({
      [a]: {},
      [b]: {},
      [c]: new File([new Uint8Array(1)], c),
      [uppercase]: {},
      notes: {},
    });
    const removeEntry = vi.fn().mockResolvedValue(undefined);
    Object.assign(root, { removeEntry });

    await deleteProjectArtifactCaches(root);

    expect(removeEntry.mock.calls).toEqual([
      [a, { recursive: true }],
      [b, { recursive: true }],
    ]);
  });

  it('tolerates a cache disappearing during deletion', async () => {
    const root = fakeDirectory({ [a]: {} });
    Object.assign(root, {
      removeEntry: vi.fn().mockRejectedValue(new DOMException('gone', 'NotFoundError')),
    });

    await expect(deleteProjectArtifactCaches(root)).resolves.toBeUndefined();
  });

  it('reports the failed cache when deletion fails for another reason', async () => {
    const root = fakeDirectory({ [a]: {} });
    Object.assign(root, {
      removeEntry: vi.fn().mockRejectedValue(new DOMException('blocked', 'SecurityError')),
    });

    await expect(deleteProjectArtifactCaches(root)).rejects.toThrow(a);
  });
});

describe('withArtifactCacheMutationLock', () => {
  it('uses the exclusive project lock once and returns the action result', async () => {
    const action = vi.fn().mockResolvedValue('result');
    const request = vi.fn(async (_name, _options, callback) => callback());

    await expect(withArtifactCacheMutationLock(
      action,
      { request } as unknown as Pick<LockManager, 'request'>,
    )).resolves.toBe('result');
    expect(request).toHaveBeenCalledWith(
      'minimax-music3-artifact-cache',
      { mode: 'exclusive' },
      expect.any(Function),
    );
    expect(action).toHaveBeenCalledTimes(1);
  });

  it('fails closed before invoking the action when locks are unavailable', async () => {
    const action = vi.fn();
    vi.stubGlobal('navigator', {});
    try {
      await expect(withArtifactCacheMutationLock(action)).rejects.toThrow(/lock/i);
      expect(action).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('withArtifactCacheReadLock', () => {
  it('uses the shared project lock once and returns the action result', async () => {
    const action = vi.fn().mockResolvedValue('result');
    const request = vi.fn(async (_name, _options, callback) => callback());

    await expect(withArtifactCacheReadLock(
      action,
      { request } as unknown as Pick<LockManager, 'request'>,
    )).resolves.toBe('result');
    expect(request).toHaveBeenCalledWith(
      'minimax-music3-artifact-cache',
      { mode: 'shared' },
      expect.any(Function),
    );
    expect(action).toHaveBeenCalledTimes(1);
  });

  it('fails closed before invoking the action when locks are unavailable', async () => {
    const action = vi.fn();
    vi.stubGlobal('navigator', {});
    try {
      await expect(withArtifactCacheReadLock(action)).rejects.toThrow(/lock/i);
      expect(action).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('requestPersistentStorage', () => {
  it('does not request persistence when storage is already persistent', async () => {
    const storage = { persisted: vi.fn().mockResolvedValue(true), persist: vi.fn() };
    await expect(requestPersistentStorage(storage)).resolves.toEqual({ state: 'persistent' });
    expect(storage.persist).not.toHaveBeenCalled();
  });

  it('reports persistence when the request is granted', async () => {
    const storage = {
      persisted: vi.fn().mockResolvedValue(false),
      persist: vi.fn().mockResolvedValue(true),
    };
    await expect(requestPersistentStorage(storage)).resolves.toEqual({ state: 'persistent' });
  });

  it('reports best effort with a warning when persistence is denied', async () => {
    const storage = {
      persisted: vi.fn().mockResolvedValue(false),
      persist: vi.fn().mockResolvedValue(false),
    };
    await expect(requestPersistentStorage(storage)).resolves.toMatchObject({
      state: 'best-effort',
      warning: expect.any(String),
    });
  });

  it('reports best effort with a warning when persistence is unsupported', async () => {
    await expect(requestPersistentStorage(undefined)).resolves.toMatchObject({
      state: 'best-effort',
      warning: expect.any(String),
    });
  });

  it('reports best effort with a warning when the persistence request rejects', async () => {
    const storage = {
      persisted: vi.fn().mockResolvedValue(false),
      persist: vi.fn().mockRejectedValue(new Error('not allowed')),
    };
    await expect(requestPersistentStorage(storage)).resolves.toMatchObject({
      state: 'best-effort',
      warning: expect.any(String),
    });
  });
});

describe('inspectArtifactCache', () => {
  it('reports missing storage as needing every artifact', async () => {
    await expect(inspectArtifactCache(artifacts, undefined)).resolves.toEqual({
      state: 'missing',
      artifactCount: 2,
      totalArtifactBytes: 30,
      completeArtifactCount: 0,
      completeArtifactBytes: 0,
      storedReferencedBytes: 0,
      additionalBytesNeeded: 30,
      largestPendingArtifactBytes: 20,
    });
  });

  it('reports an existing empty directory as partial', async () => {
    await expect(inspectArtifactCache(artifacts, storeWith())).resolves.toEqual({
      state: 'partial',
      artifactCount: 2,
      totalArtifactBytes: 30,
      completeArtifactCount: 0,
      completeArtifactBytes: 0,
      storedReferencedBytes: 0,
      additionalBytesNeeded: 30,
      largestPendingArtifactBytes: 20,
    });
  });

  it('accounts for complete, partial, absent, unverified, and oversized artifacts', async () => {
    const mixed = [
      { path: 'complete.bin', bytes: 10, sha256: 'a'.repeat(64) },
      { path: 'partial.bin', bytes: 20, sha256: 'b'.repeat(64) },
      { path: 'absent.bin', bytes: 30, sha256: 'c'.repeat(64) },
      { path: 'unverified.bin', bytes: 40, sha256: 'd'.repeat(64) },
      { path: 'oversized.bin', bytes: 50, sha256: 'e'.repeat(64) },
    ];
    const store = storeWith(
      {
        'complete.bin': 10,
        'partial.bin': 7,
        'unverified.bin': 40,
        'oversized.bin': 60,
      },
      ['complete.bin', 'oversized.bin'],
    );

    await expect(inspectArtifactCache(mixed, store)).resolves.toEqual({
      state: 'partial',
      artifactCount: 5,
      totalArtifactBytes: 150,
      completeArtifactCount: 1,
      completeArtifactBytes: 10,
      storedReferencedBytes: 117,
      additionalBytesNeeded: 43,
      largestPendingArtifactBytes: 50,
    });
  });

  it('reports ready only when every expected-size artifact has a completion receipt', async () => {
    await expect(
      inspectArtifactCache(artifacts, storeWith({ 'a.bin': 10, 'b.bin': 20 }, ['a.bin', 'b.bin'])),
    ).resolves.toEqual({
      state: 'ready',
      artifactCount: 2,
      totalArtifactBytes: 30,
      completeArtifactCount: 2,
      completeArtifactBytes: 30,
      storedReferencedBytes: 30,
      additionalBytesNeeded: 0,
      largestPendingArtifactBytes: 0,
    });
  });
});

describe('assessArtifactCapacity', () => {
  const inspection: ArtifactCacheInspection = {
    state: 'partial',
    artifactCount: 5,
    totalArtifactBytes: 150,
    completeArtifactCount: 1,
    completeArtifactBytes: 10,
    storedReferencedBytes: 117,
    additionalBytesNeeded: 43,
    largestPendingArtifactBytes: 50,
  };

  it('accepts available capacity equal to the required headroom', () => {
    expect(assessArtifactCapacity(inspection, { usage: 7, quota: 100 })).toEqual({
      usageBytes: 7,
      quotaBytes: 100,
      availableBytes: 93,
      sufficient: true,
      requiredHeadroomBytes: 93,
    });
  });

  it('reports insufficient capacity below the required headroom', () => {
    expect(assessArtifactCapacity(inspection, { usage: 8, quota: 100 })).toMatchObject({
      availableBytes: 92,
      sufficient: false,
      requiredHeadroomBytes: 93,
    });
  });

  it.each([
    undefined,
    {},
    { usage: Number.NaN, quota: 100 },
    { usage: 1, quota: Number.POSITIVE_INFINITY },
    { usage: -1, quota: 100 },
    { usage: 101, quota: 100 },
    { usage: 1.5, quota: 100 },
    { usage: 1, quota: 100.5 },
    { usage: Number.MAX_SAFE_INTEGER + 1, quota: Number.MAX_SAFE_INTEGER + 2 },
    { usage: 1, quota: Number.MAX_SAFE_INTEGER + 1 },
  ])('omits unavailable capacity for an unavailable or malformed estimate', (estimate) => {
    expect(assessArtifactCapacity(inspection, estimate)).toEqual({
      requiredHeadroomBytes: 93,
    });
  });

  it('requires no headroom for a ready cache', () => {
    expect(assessArtifactCapacity({
      ...inspection,
      state: 'ready',
      additionalBytesNeeded: 0,
      largestPendingArtifactBytes: 0,
    }, { usage: 100, quota: 100 })).toEqual({
      usageBytes: 100,
      quotaBytes: 100,
      availableBytes: 0,
      sufficient: true,
      requiredHeadroomBytes: 0,
    });
  });
});
