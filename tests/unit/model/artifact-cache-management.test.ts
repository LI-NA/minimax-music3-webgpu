import { describe, expect, it } from 'vitest';
import {
  assessArtifactCapacity,
  inspectArtifactCache,
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
