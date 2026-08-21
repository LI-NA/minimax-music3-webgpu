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

export async function inspectArtifactCache(
  artifacts: readonly ArtifactFile[],
  store: Pick<ArtifactStore, 'size' | 'isComplete'> | undefined,
): Promise<ArtifactCacheInspection> {
  const totalArtifactBytes = artifacts.reduce((sum, artifact) => sum + artifact.bytes, 0);
  if (store) {
    const entries = await Promise.all(artifacts.map(async (artifact) => {
      const size = await store.size(artifact.path);
      const complete = size === artifact.bytes && await store.isComplete(artifact.path);
      return { artifact, size, complete };
    }));
    const completeEntries = entries.filter((entry) => entry.complete);
    const pendingEntries = entries.filter((entry) => !entry.complete);
    return {
      state: pendingEntries.length === 0 ? 'ready' : 'partial',
      artifactCount: artifacts.length,
      totalArtifactBytes,
      completeArtifactCount: completeEntries.length,
      completeArtifactBytes: completeEntries.reduce(
        (sum, entry) => sum + entry.artifact.bytes,
        0,
      ),
      storedReferencedBytes: entries.reduce((sum, entry) => sum + entry.size, 0),
      additionalBytesNeeded: entries.reduce(
        (sum, entry) => sum + Math.max(entry.artifact.bytes - Math.min(entry.size, entry.artifact.bytes), 0),
        0,
      ),
      largestPendingArtifactBytes: Math.max(
        0,
        ...pendingEntries.map((entry) => entry.artifact.bytes),
      ),
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
  const requiredHeadroomBytes = inspection.state === 'ready'
    ? 0
    : inspection.additionalBytesNeeded + inspection.largestPendingArtifactBytes;
  const usageBytes = estimate?.usage;
  const quotaBytes = estimate?.quota;
  if (
    usageBytes === undefined
    || quotaBytes === undefined
    || !Number.isSafeInteger(usageBytes)
    || !Number.isSafeInteger(quotaBytes)
    || usageBytes < 0
    || quotaBytes < usageBytes
  ) return { requiredHeadroomBytes };
  const availableBytes = quotaBytes - usageBytes;
  return {
    usageBytes,
    quotaBytes,
    availableBytes,
    sufficient: availableBytes >= requiredHeadroomBytes,
    requiredHeadroomBytes,
  };
}
