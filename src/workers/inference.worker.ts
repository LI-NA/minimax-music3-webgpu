/// <reference lib="webworker" />
import { OpfsArtifactStore, ensureArtifact } from '../runtime/model/artifact-cache';
import {
  assessArtifactCapacity,
  deleteProjectArtifactCaches,
  inspectArtifactCache,
  inspectProjectArtifactCaches,
  withArtifactCacheMutationLock,
  withArtifactCacheReadLock,
} from '../runtime/model/artifact-cache-management';
import { OpfsFp16EmbeddingTable } from '../runtime/model/embedding-table';
import {
  parseConditionManifest,
  parseFlowManifest,
  parseMusicManifest,
  parseMusicVariableManifest,
  parseModelManifest,
  parseRvqStageManifest,
  parseVocoderManifest,
} from '../runtime/model/manifest';
import { inspectWebGpuForRequirements } from '../runtime/model/webgpu-device';
import { runConditionSmoke } from '../runtime/pipeline/condition-smoke';
import { runFlowSmoke } from '../runtime/pipeline/flow-generation';
import { runChunkedFlowGeneration, runFixedFlowGeneration } from '../runtime/pipeline/flow-generation';
import { runGlobalSmoke } from '../runtime/pipeline/global-smoke';
import { runRvqSmoke } from '../runtime/pipeline/rvq-smoke';
import {
  analyticVocoderLatents,
  generateFixedVocoderWav,
  generateVariableVocoderWav,
} from '../runtime/pipeline/vocoder-generation';
import {
  areFiniteFp16,
  createFrameGenerator,
  EarlyAudioEndError,
  readConditionalGpuFp16,
} from '../runtime/pipeline/rvq-generation';
import {
  deterministicGaussianFp16,
  generateFiveSecondMusic,
  readExactGpuFp16,
} from '../runtime/pipeline/music-generation';
import {
  createFixedComparisonMetadata,
  FIXED_COMPARISON_CASE,
  PINNED_COMPARISON_ORT_VERSION,
} from '../runtime/reference/fixed-comparison';
import { createPromptTokenizer, preparePrompt } from '../runtime/pipeline/prompt-preparation';
import { createMusicProgressTracker } from './music-progress';
import { createArtifactProgressReporter } from './artifact-progress';
import {
  createMusicGenerationResultPlan,
  createResolvedMusicGenerationRequest,
  validateArtifactCacheRequest,
  validateMusicCapacityDiagnosticRequest,
  validateMusicGenerationRequest,
  type ArtifactCacheRequest,
  type ArtifactErrorCode,
  type ArtifactOperation,
  type MusicGenerationRequest,
  type WorkerRequest,
  type WorkerResponse,
} from './protocol';

const send = (message: WorkerResponse, transfer?: Transferable[]) => self.postMessage(message, transfer ?? []);
const hashText = async (text: string) =>
  Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))))
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
class ArtifactOperationError extends Error {
  constructor(
    message: string,
    readonly code: ArtifactErrorCode,
    readonly operation: ArtifactOperation,
    readonly retryable: boolean,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = 'ArtifactOperationError';
  }
}
function serializeWorkerError(error: unknown): Extract<WorkerResponse, { type: 'error' }> {
  const message = error instanceof Error ? error.message : String(error);
  if (!(error instanceof ArtifactOperationError)) return { type: 'error', message };
  return {
    type: 'error',
    message,
    code: error.code,
    operation: error.operation,
    retryable: error.retryable,
  };
}
type Failure = { error: unknown };
const failure = (error: unknown): Failure => ({ error });
async function settleSessionReleases(
  sessions: readonly ({ release(): Promise<void> } | undefined)[],
): Promise<Failure | undefined> {
  const results = await Promise.allSettled(
    sessions.filter((session): session is { release(): Promise<void> } => session !== undefined)
      .map((session) => session.release()),
  );
  const rejected = results.find((result): result is PromiseRejectedResult => result.status === 'rejected');
  return rejected ? failure(rejected.reason) : undefined;
}
function throwFirstFailure(...failures: readonly (Failure | undefined)[]) {
  const selected = failures.find((item) => item !== undefined);
  if (selected) throw selected.error;
}
type CreateOrtSession = (typeof import('../runtime/model/ort-session'))['createOrtSession'];
type OrtSession = Awaited<ReturnType<CreateOrtSession>>;
type OrtWebGpu = { device?: GPUDevice | Promise<GPUDevice> };
type OrtDevice = { binding: GPUDevice | Promise<GPUDevice>; device: GPUDevice };

async function capturePublishedOrtDevice(
  webgpu: OrtWebGpu,
  previous?: GPUDevice | Promise<GPUDevice>,
): Promise<OrtDevice | undefined> {
  const binding = webgpu.device;
  if (!binding || binding === previous) return undefined;
  return { binding, device: await binding };
}

function verifyOrtDevice(device: GPUDevice, requiredLimits: Readonly<Record<string, number>>) {
  if (!device.features.has('shader-f16')) throw new Error('ORT WebGPU device does not support shader-f16');
  const limits = device.limits as unknown as Record<string, number>;
  if (Object.entries(requiredLimits).some(
    ([name, value]) => typeof limits[name] !== 'number' || limits[name] < value,
  )) throw new Error('ORT WebGPU device limits are insufficient');
}

async function settlePublishedOrtDevice(
  webgpu: OrtWebGpu,
  handle: OrtDevice | undefined,
): Promise<Failure | undefined> {
  if (!handle) return undefined;
  let destroyFailure: Failure | undefined;
  try {
    handle.device.destroy();
  } catch (error) {
    destroyFailure = failure(error);
  }
  let clearFailure: Failure | undefined;
  try {
    const current = webgpu.device;
    if (current && await current === handle.device) delete webgpu.device;
  } catch (error) {
    clearFailure = failure(error);
  }
  return destroyFailure ?? clearFailure;
}

async function withOrtOwnedDevice<T>(
  requiredLimits: Readonly<Record<string, number>>,
  action: (createSession: CreateOrtSession, adapter: GPUAdapter) => Promise<T>,
): Promise<T> {
  const capability = await inspectWebGpuForRequirements(navigator.gpu, requiredLimits);
  if (!capability.supported) throw new Error(capability.reason);
  const { createOrtSession } = await import('../runtime/model/ort-session');
  const ort = await import('onnxruntime-web/jspi');
  const webgpu = ort.env.webgpu as unknown as OrtWebGpu;
  const previousBinding = webgpu.device;
  const sessions: OrtSession[] = [];
  let device: OrtDevice | undefined;
  let value: T | undefined;
  let primaryFailure: Failure | undefined;
  const createSession: CreateOrtSession = async (graph, cache) => {
    try {
      const session = await createOrtSession(graph, cache);
      sessions.push(session);
      if (!device) {
        device = await capturePublishedOrtDevice(webgpu, previousBinding);
        if (!device) throw new Error('ORT WebGPU session did not expose its device');
        verifyOrtDevice(device.device, requiredLimits);
      }
      return session;
    } catch (error) {
      if (!device) {
        try {
          device = await capturePublishedOrtDevice(webgpu, previousBinding);
        } catch {
          // The session creation failure remains primary.
        }
      }
      throw error;
    }
  };
  try {
    value = await action(createSession, capability.adapter);
  } catch (error) {
    primaryFailure = failure(error);
  }
  const releaseFailure = await settleSessionReleases([...sessions].reverse());
  const deviceFailure = await settlePublishedOrtDevice(webgpu, device);
  throwFirstFailure(primaryFailure, releaseFailure, deviceFailure);
  return value as T;
}

let workerRequestActive = false;
self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  if (workerRequestActive) {
    send({ type: 'error', message: 'Worker request already in progress' });
    return;
  }
  workerRequestActive = true;
  void runWorkerRequest(event.data).catch((error: unknown) => {
    send(serializeWorkerError(error));
  }).finally(() => {
    workerRequestActive = false;
  });
};
export async function runWorkerRequest(rawRequest: unknown) {
  if (typeof rawRequest !== 'object' || rawRequest === null)
    throw new Error('Worker request must be an object');
  if (
    rawRequest
    && typeof rawRequest === 'object'
    && (rawRequest as { type?: unknown }).type === 'generate-music'
  ) {
    const request = validateMusicGenerationRequest(rawRequest);
    return withArtifactCacheReadLock(() => runVariableMusicGeneration(request));
  }
  if (
    rawRequest
    && typeof rawRequest === 'object'
    && (rawRequest as { type?: unknown }).type === 'diagnose-music-capacity'
  ) {
    const request = validateMusicCapacityDiagnosticRequest(rawRequest);
    return withArtifactCacheMutationLock(() => runVariableMusicGeneration(request));
  }
  if (
    rawRequest
    && typeof rawRequest === 'object'
    && ['inspect-artifact-cache', 'download-artifacts', 'delete-artifact-caches']
      .includes(String((rawRequest as { type?: unknown }).type))
  ) {
    const request = validateArtifactCacheRequest(rawRequest);
    if (request.type === 'inspect-artifact-cache') return runArtifactCacheInspection(request);
    if (request.type === 'download-artifacts') return runArtifactDownload(request);
    return runArtifactCacheDeletion(request);
  }
  const request = rawRequest as WorkerRequest;
  if (request.type === 'generate-music-5s')
    return withArtifactCacheMutationLock(() => runMusicGeneration(request));
  if (request.type === 'run-vocoder-smoke')
    return withArtifactCacheMutationLock(() => runVocoder(request.manifestUrl));
  if (request.type === 'run-flow-smoke')
    return withArtifactCacheMutationLock(() => runFlow(request.manifestUrl));
  if (request.type === 'run-condition-smoke')
    return withArtifactCacheMutationLock(() => runCondition(request.manifestUrl));
  if (request.type === 'generate-frames')
    return withArtifactCacheMutationLock(() => runFrameGeneration(request));
  if (request.type === 'run-rvq-smoke')
    return withArtifactCacheMutationLock(() => runRvq(request.manifestUrl));
  if (request.type !== 'run-global-smoke') throw new Error('Unknown worker request type');
  return withArtifactCacheMutationLock(() => runGlobal(request.manifestUrl));
}

async function runGlobal(manifestUrl: string) {
  send({
    type: 'progress',
    stage: 'manifest',
    detail: 'Reading release manifest',
  });
  let response: Response;
  try {
    response = await fetch(manifestUrl);
  } catch {
    throw new Error('Release manifest is unavailable');
  }
  if (!response.ok) throw new Error('Release manifest is unavailable');
  const manifestText = await response.text();
  const manifest = parseModelManifest(JSON.parse(manifestText));
  const cache = await OpfsArtifactStore.open(await hashText(manifestText));
  const base = new URL(manifestUrl, self.location.href);
  const artifacts = [
    manifest.graph,
    ...manifest.graph.externalData,
    manifest.reducedHead,
    ...manifest.reducedHead.externalData,
    ...manifest.embedding.shards,
  ];
  const artifactFetches = await cacheArtifacts(artifacts, base, cache);
  send({
    type: 'progress',
    stage: 'adapter',
    detail: 'Checking shader-f16 WebGPU requirements',
  });
  send({
    type: 'progress',
    stage: 'session',
    detail: 'Creating WebGPU decoder session',
  });
  const started = performance.now();
  const ort = await import('onnxruntime-web/jspi');
  await withOrtOwnedDevice(manifest.webgpu.requiredLimits, async (createOrtSession, adapter) => {
    const graph = await createOrtSession(manifest.graph, cache);
    const head = await createOrtSession(manifest.reducedHead, cache);
    const sessionCreateMs = performance.now() - started;
    const metrics = await runGlobalSmoke({
      ort,
      decoder: graph,
      head,
      embedding: await OpfsFp16EmbeddingTable.open(manifest.embedding, (path) => cache.openSyncFile(path)),
      embeddingTable: manifest.embedding,
      kvPairs: manifest.kvPairs,
    });
    send({
      type: 'result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        ...metrics,
        sessionCreateMs,
        artifactFetches,
        cacheReuseCount: artifacts.length - artifactFetches,
        status: 'passed',
      },
    });
  });
}

async function runVocoder(manifestUrl: string) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading vocoder release manifest' });
  const release = await readManifest(manifestUrl, 'Vocoder release manifest is unavailable');
  const manifest = parseVocoderManifest(JSON.parse(release.text));
  const artifacts = [manifest.vocoder, ...manifest.vocoder.externalData];
  const artifactFetches = await cacheArtifacts(artifacts, release.base, release.cache);
  send({ type: 'progress', stage: 'adapter', detail: 'Checking shader-f16 WebGPU requirements' });
  const ort = await import('onnxruntime-web/jspi');
  await withOrtOwnedDevice(manifest.webgpu.requiredLimits, async (createOrtSession, adapter) => {
    send({ type: 'progress', stage: 'session', detail: 'Creating WebGPU vocoder session' });
    const sessionStarted = performance.now();
    const vocoder = await createOrtSession(manifest.vocoder, release.cache);
    const sessionCreateMs = performance.now() - sessionStarted;
    const generationStarted = performance.now();
    const wav = await generateFixedVocoderWav(
      { ort, session: vocoder },
      analyticVocoderLatents(),
    );
    const generationMs = performance.now() - generationStarted;
    validateVocoderWav(wav);
    send({
      type: 'vocoder-result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        sessionCreateMs,
        generationMs,
        outputType: 'float32',
        shape: [1, 2, 220_160],
        finite: true,
        wavBytes: 880_684,
        sampleRate: 44_100,
        channels: 2,
        samples: 220_160,
        bitsPerSample: 16,
        artifactFetches,
        status: 'passed',
      },
    });
  });
}

function validateVocoderWav(wav: ArrayBuffer) {
  const view = new DataView(wav);
  const text = (offset: number, length: number) =>
    String.fromCharCode(...new Uint8Array(wav, offset, length));
  if (
    wav.byteLength !== 880_684
    || text(0, 4) !== 'RIFF'
    || text(8, 4) !== 'WAVE'
    || view.getUint16(20, true) !== 1
    || view.getUint16(22, true) !== 2
    || view.getUint32(24, true) !== 44_100
    || view.getUint16(34, true) !== 16
    || view.getUint32(40, true) !== 880_640
  ) throw new Error('vocoder WAV structure is invalid');
}

async function runFlow(manifestUrl: string) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading flow release manifest' });
  const release = await readManifest(manifestUrl, 'Flow release manifest is unavailable');
  const manifest = parseFlowManifest(JSON.parse(release.text));
  const artifacts = [manifest.flow, ...manifest.flow.externalData];
  const artifactFetches = await cacheArtifacts(artifacts, release.base, release.cache);
  send({ type: 'progress', stage: 'adapter', detail: 'Checking shader-f16 WebGPU requirements' });
  const ort = await import('onnxruntime-web/jspi');
  await withOrtOwnedDevice(manifest.webgpu.requiredLimits, async (createOrtSession, adapter) => {
    send({ type: 'progress', stage: 'session', detail: 'Creating WebGPU flow transformer session' });
    const started = performance.now();
    const flow = await createOrtSession(manifest.flow, release.cache);
    const sessionCreateMs = performance.now() - started;
    const metrics = await runFlowSmoke({ ort, session: flow });
    send({
      type: 'flow-result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        ...metrics,
        sessionCreateMs,
        artifactFetches,
        status: 'passed',
      },
    });
  });
}

async function runCondition(manifestUrl: string) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading condition release manifest' });
  const release = await readManifest(manifestUrl, 'Condition release manifest is unavailable');
  const manifest = parseConditionManifest(JSON.parse(release.text));
  const artifacts = [manifest.conditionEncoder, ...manifest.conditionEncoder.externalData];
  const artifactFetches = await cacheArtifacts(artifacts, release.base, release.cache);
  send({ type: 'progress', stage: 'adapter', detail: 'Checking shader-f16 WebGPU requirements' });
  const ort = await import('onnxruntime-web/jspi');
  await withOrtOwnedDevice(manifest.webgpu.requiredLimits, async (createOrtSession, adapter) => {
    send({ type: 'progress', stage: 'session', detail: 'Creating WebGPU condition encoder session' });
    const started = performance.now();
    const conditionEncoder = await createOrtSession(manifest.conditionEncoder, release.cache);
    const sessionCreateMs = performance.now() - started;
    const metrics = await runConditionSmoke({ ort, session: conditionEncoder });
    send({
      type: 'condition-result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        ...metrics,
        sessionCreateMs,
        artifactFetches,
        status: 'passed',
      },
    });
  });
}

async function readManifest(url: string, unavailable: string) {
  const release = await fetchManifest(url, unavailable);
  return {
    ...release,
    cache: await OpfsArtifactStore.open(release.hash),
  };
}

async function fetchManifest(url: string, unavailable: string) {
  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw new Error(unavailable);
  }
  if (!response.ok) throw new Error(unavailable);
  const text = await response.text();
  return {
    text,
    hash: await hashText(text),
    base: new URL(url, self.location.href),
  };
}

function collectVariableMusicArtifacts(
  manifest: ReturnType<typeof parseMusicVariableManifest>,
) {
  const allArtifacts = [
    manifest.graph,
    ...manifest.graph.externalData,
    manifest.reducedHead,
    ...manifest.reducedHead.externalData,
    ...manifest.embedding.shards,
    manifest.rvqDepth,
    ...manifest.rvqDepth.externalData,
    manifest.feedback,
    ...manifest.feedback.externalData,
    ...manifest.rvqEmbedding.shards,
    manifest.conditionEncoder,
    ...manifest.conditionEncoder.externalData,
    manifest.flow,
    ...manifest.flow.externalData,
    manifest.vocoder,
    ...manifest.vocoder.externalData,
    ...manifest.tokenizerFiles,
    manifest.licenseFile,
  ];
  return [...new Map(allArtifacts.map((artifact) => [artifact.path, artifact])).values()];
}

function parseVariableMusicRelease(
  release: Awaited<ReturnType<typeof fetchManifest>>,
  operation: ArtifactOperation,
) {
  try {
    return parseMusicVariableManifest(JSON.parse(release.text));
  } catch (error) {
    throw new ArtifactOperationError(
      'Music release manifest is invalid',
      'manifest-invalid',
      operation,
      false,
      { cause: error },
    );
  }
}

async function readPersistenceState(storage: StorageManager | undefined) {
  if (!storage?.persisted) return 'unavailable' as const;
  try {
    return await storage.persisted() ? 'persistent' as const : 'best-effort' as const;
  } catch {
    return 'unavailable' as const;
  }
}

async function readStorageEstimate(storage: StorageManager | undefined) {
  if (!storage?.estimate) return undefined;
  try {
    return await storage.estimate();
  } catch {
    return undefined;
  }
}

async function inspectVariableArtifactStatus(
  release: Awaited<ReturnType<typeof fetchManifest>>,
  artifacts: ReturnType<typeof collectVariableMusicArtifacts>,
  opfsRoot?: FileSystemDirectoryHandle,
) {
  const storage = navigator.storage;
  const root = opfsRoot ?? await storage.getDirectory();
  const cache = await OpfsArtifactStore.openExisting(release.hash, root);
  const [inspection, project, persistence, estimate] = await Promise.all([
    inspectArtifactCache(artifacts, cache),
    inspectProjectArtifactCaches(root),
    readPersistenceState(storage),
    readStorageEstimate(storage),
  ]);
  return {
    manifestHash: release.hash,
    ...inspection,
    projectCacheCount: project.cacheCount,
    projectCacheBytes: project.storedBytes,
    persistence,
    ...assessArtifactCapacity(inspection, estimate),
  };
}

async function fetchVariableMusicRelease(manifestUrl: string, operation: ArtifactOperation) {
  try {
    return await fetchManifest(manifestUrl, 'Music release manifest is unavailable');
  } catch (error) {
    throw new ArtifactOperationError(
      'Music release manifest is unavailable',
      'manifest-unavailable',
      operation,
      true,
      { cause: error },
    );
  }
}

async function runArtifactCacheInspection(
  request: Extract<ArtifactCacheRequest, { type: 'inspect-artifact-cache' }>,
) {
  const operation = request.type;
  const release = await fetchVariableMusicRelease(request.manifestUrl, operation);
  const manifest = parseVariableMusicRelease(release, operation);
  try {
    const status = await inspectVariableArtifactStatus(
      release,
      collectVariableMusicArtifacts(manifest),
    );
    send({ type: 'artifact-cache-status', status });
  } catch (error) {
    if (error instanceof ArtifactOperationError) throw error;
    throw new ArtifactOperationError(
      'Artifact cache inspection failed',
      'cache-inspection-failed',
      operation,
      true,
      { cause: error },
    );
  }
}

function hasErrorName(error: unknown, name: string): boolean {
  if (!(error instanceof Error)) return false;
  if (error.name === name) return true;
  return hasErrorName(error.cause, name);
}

function operationFailureMessage(prefix: string, error: unknown) {
  const detail = error instanceof Error ? error.message : String(error);
  return detail ? `${prefix}: ${detail}` : prefix;
}

async function runArtifactDownload(
  request: Extract<ArtifactCacheRequest, { type: 'download-artifacts' }>,
) {
  const operation = request.type;
  try {
    await withArtifactCacheMutationLock(async () => {
      const release = await fetchVariableMusicRelease(request.manifestUrl, operation);
      const manifest = parseVariableMusicRelease(release, operation);
      const artifacts = collectVariableMusicArtifacts(manifest);
      const root = await navigator.storage.getDirectory();
      const status = await inspectVariableArtifactStatus(release, artifacts, root);
      if (status.sufficient === undefined) {
        throw new ArtifactOperationError(
          'Storage estimate is unavailable',
          'storage-estimate-unavailable',
          operation,
          true,
        );
      }
      if (!status.sufficient) {
        throw new ArtifactOperationError(
          'Storage quota is insufficient for model artifacts',
          'quota-insufficient',
          operation,
          true,
        );
      }
      const cache = await OpfsArtifactStore.open(release.hash, root);
      await cacheArtifacts(artifacts, release.base, cache);
      const completed = await inspectVariableArtifactStatus(release, artifacts, root);
      if (completed.state !== 'ready')
        throw new Error('Artifact cache is not ready after download');
      send({ type: 'artifact-download-complete', status: completed });
    });
  } catch (error) {
    if (error instanceof ArtifactOperationError) throw error;
    if (hasErrorName(error, 'QuotaExceededError')) {
      throw new ArtifactOperationError(
        operationFailureMessage(
          'Storage quota was exceeded while downloading model artifacts',
          error,
        ),
        'quota-exceeded',
        operation,
        true,
        { cause: error },
      );
    }
    throw new ArtifactOperationError(
      operationFailureMessage('Artifact download failed', error),
      'download-failed',
      operation,
      true,
      { cause: error },
    );
  }
}

async function runArtifactCacheDeletion(
  request: Extract<ArtifactCacheRequest, { type: 'delete-artifact-caches' }>,
) {
  const operation = request.type;
  const release = await fetchVariableMusicRelease(request.manifestUrl, operation);
  const manifest = parseVariableMusicRelease(release, operation);
  const artifacts = collectVariableMusicArtifacts(manifest);
  try {
    await withArtifactCacheMutationLock(async () => {
      const root = await navigator.storage.getDirectory();
      await deleteProjectArtifactCaches(root);
      const status = await inspectVariableArtifactStatus(release, artifacts, root);
      send({ type: 'artifact-cache-deleted', status });
    });
  } catch (error) {
    if (error instanceof ArtifactOperationError) throw error;
    throw new ArtifactOperationError(
      operationFailureMessage('Artifact cache deletion failed', error),
      'cache-delete-failed',
      operation,
      true,
      { cause: error },
    );
  }
}

async function cacheArtifacts(
  artifacts: readonly { path: string; bytes: number; sha256: string }[],
  base: URL,
  cache: OpfsArtifactStore,
) {
  let fetches = 0;
  let completedBytes = 0;
  let transferredBytes = 0;
  const totalBytes = artifacts.reduce((total, artifact) => total + artifact.bytes, 0);
  const reporter = createArtifactProgressReporter({ totalBytes, send });
  for (const artifact of artifacts) {
    let fetched = false;
    let artifactTransferredBytes = 0;
    try {
      await ensureArtifact(
        artifact,
        new URL(artifact.path, base),
        cache,
        ({ path, loaded, total, transferred }) => {
          artifactTransferredBytes = Math.max(artifactTransferredBytes, transferred);
          reporter.report(
            path,
            loaded,
            total,
            completedBytes,
            transferredBytes + artifactTransferredBytes,
          );
        },
        async (input, init) => {
          fetched = true;
          fetches++;
          return fetch(input, init);
        },
      );
    } catch (error) {
      reporter.discard();
      throw error;
    }
    transferredBytes += artifactTransferredBytes;
    completedBytes += artifact.bytes;
    reporter.complete(
      artifact.path,
      artifact.bytes,
      completedBytes,
      !fetched,
      transferredBytes,
    );
  }
  return fetches;
}

async function runFrameGeneration(request: Extract<WorkerRequest, { type: 'generate-frames' }>) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading Global and RVQ release manifests' });
  const globalRelease = await readManifest(request.globalManifestUrl, 'Global release manifest is unavailable');
  const rvqRelease = await readManifest(request.rvqManifestUrl, 'RVQ release manifest is unavailable');
  const globalManifest = parseModelManifest(JSON.parse(globalRelease.text));
  const rvqManifest = parseRvqStageManifest(JSON.parse(rvqRelease.text));
  const globalArtifacts = [
    globalManifest.graph,
    ...globalManifest.graph.externalData,
    globalManifest.reducedHead,
    ...globalManifest.reducedHead.externalData,
    ...globalManifest.embedding.shards,
  ];
  const rvqArtifacts = [
    rvqManifest.rvqDepth,
    ...rvqManifest.rvqDepth.externalData,
    rvqManifest.feedback,
    ...rvqManifest.feedback.externalData,
    ...rvqManifest.rvqEmbedding.shards,
  ];
  const artifactFetches =
    (await cacheArtifacts(globalArtifacts, globalRelease.base, globalRelease.cache)) +
    (await cacheArtifacts(rvqArtifacts, rvqRelease.base, rvqRelease.cache));
  send({ type: 'progress', stage: 'adapter', detail: 'Checking shared shader-f16 WebGPU requirements' });
  const requiredLimits = { ...globalManifest.webgpu.requiredLimits };
  for (const [name, value] of Object.entries(rvqManifest.webgpu.requiredLimits))
    requiredLimits[name] = Math.max(requiredLimits[name] ?? 0, value);
  const ort = await import('onnxruntime-web/jspi');
  await withOrtOwnedDevice(requiredLimits, async (createOrtSession, adapter) => {
    send({ type: 'progress', stage: 'session', detail: 'Creating shared-device autoregressive sessions' });
    const decoder = await createOrtSession(globalManifest.graph, globalRelease.cache);
    const head = await createOrtSession(globalManifest.reducedHead, globalRelease.cache);
    const rvqDepth = await createOrtSession(rvqManifest.rvqDepth, rvqRelease.cache);
    const feedback = await createOrtSession(rvqManifest.feedback, rvqRelease.cache);
    const attemptedSeeds: number[] = [];
    let frames: Awaited<ReturnType<ReturnType<typeof createFrameGenerator>['generateFrames']>> | undefined;
    let cacheLengths: number[] = [];
    for (let attempt = 0; attempt < 2; attempt++) {
      const seed = request.seed + attempt;
      attemptedSeeds.push(seed);
      cacheLengths = [];
      try {
        frames = await createFrameGenerator({
          ort,
          decoder,
          head,
          rvqDepth,
          feedback,
          globalEmbedding: await OpfsFp16EmbeddingTable.open(
            globalManifest.embedding,
            (path) => globalRelease.cache.openSyncFile(path),
          ),
          rvqEmbedding: await OpfsFp16EmbeddingTable.open(
            rvqManifest.rvqEmbedding,
            (path) => rvqRelease.cache.openSyncFile(path),
          ),
          embeddingColumns: globalManifest.embedding.columns,
          kvPairs: globalManifest.kvPairs,
          readConditionalHidden: readConditionalGpuFp16,
          onCacheLength: (length) => cacheLengths.push(length),
        }).generateFrames({
          maxFrames: request.maxFrames,
          seed,
          promptTokenRows: {
            conditional: FIXED_COMPARISON_CASE.input.tokenRows[0],
            unconditional: FIXED_COMPARISON_CASE.input.tokenRows[1],
          },
          guidance: 1.5,
          semanticTopK: 50,
          residualTopK: 50,
          temperature: 1,
        });
        break;
      } catch (error) {
        if (!(error instanceof EarlyAudioEndError)) throw error;
        if (attempt === 1)
          throw new Error(`audio end sampled early for seeds ${attemptedSeeds.join(', ')}`);
        send({ type: 'progress', stage: 'session', detail: `${error.message}; retrying seed ${seed + 1}` });
      }
    }
    if (!frames) throw new Error(`audio end sampled early for seeds ${attemptedSeeds.join(', ')}`);
    const finiteHiddenGroups = frames.every((frame) => areFiniteFp16(frame.hiddenGroups));
    const codesInRange = frames.every(
      (frame) =>
        frame.semantic >= 0 &&
        frame.semantic < 16_384 &&
        frame.residual.every((code) => code >= 0 && code < 1_024),
    );
    send({
      type: 'frame-result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        frames: frames.length,
        attemptedSeeds,
        semanticDecisions: frames.length + 1,
        rvqCalls: (frames.length + 1) * 7,
        feedbackDecodes: frames.length,
        cacheLengths,
        finiteHiddenGroups,
        codesInRange,
        hiddenBytes: frames.reduce((total, frame) => total + frame.hiddenGroups.byteLength, 0),
        artifactFetches,
        status: 'passed',
      },
    });
  });
}

async function prepareProductPrompt(
  request: MusicGenerationRequest,
  tokenizerFiles: readonly { path: string }[],
  cache: OpfsArtifactStore,
) {
  const tokenizerArtifact = tokenizerFiles.find(({ path }) => path.endsWith('/tokenizer.json'));
  const configArtifact = tokenizerFiles.find(({ path }) => path.endsWith('/tokenizer_config.json'));
  if (!tokenizerArtifact || !configArtifact)
    throw new Error('Music release tokenizer artifacts are unavailable');
  const tokenizerJson = await (await cache.file(tokenizerArtifact.path)).text();
  const tokenizerConfigJson = await (await cache.file(configArtifact.path)).text();
  const tokenizer = createPromptTokenizer(tokenizerJson, tokenizerConfigJson);
  const fixedPrepared = preparePrompt({
    prompt: FIXED_COMPARISON_CASE.input.prompt,
    lyrics: FIXED_COMPARISON_CASE.input.lyrics,
    requestedFrames: 250,
    tokenizer,
  });
  if (
    fixedPrepared.assembledPrompt !== FIXED_COMPARISON_CASE.input.assembledPrompt
    || JSON.stringify(fixedPrepared.tokenRows) !== JSON.stringify(FIXED_COMPARISON_CASE.input.tokenRows)
  ) throw new Error('Shipped tokenizer does not match the fixed prompt contract');
  return preparePrompt({
    prompt: request.prompt,
    lyrics: request.lyrics,
    requestedFrames: Math.floor(request.durationSeconds * 25),
    tokenizer,
  });
}

async function runVariableMusicGeneration(
  inputRequest: Extract<
    WorkerRequest,
    { type: 'generate-music' | 'diagnose-music-capacity' }
  >,
) {
  const capacityDiagnostic = inputRequest.type === 'diagnose-music-capacity';
  send({ type: 'progress', stage: 'manifest', detail: 'Reading variable music release manifest' });
  let release: Awaited<ReturnType<typeof readManifest>>;
  let manifest: ReturnType<typeof parseMusicVariableManifest>;
  let artifactFetches: number;
  if (capacityDiagnostic) {
    release = await readManifest(inputRequest.manifestUrl, 'Music release manifest is unavailable');
    manifest = parseMusicVariableManifest(JSON.parse(release.text));
    artifactFetches = await cacheArtifacts(
      collectVariableMusicArtifacts(manifest),
      release.base,
      release.cache,
    );
  } else {
    const fetched = await fetchVariableMusicRelease(inputRequest.manifestUrl, 'generate-music');
    manifest = parseVariableMusicRelease(fetched, 'generate-music');
    const artifacts = collectVariableMusicArtifacts(manifest);
    let cache: OpfsArtifactStore | undefined;
    try {
      const root = await navigator.storage.getDirectory();
      cache = await OpfsArtifactStore.openExisting(fetched.hash, root);
      const inspection = await inspectArtifactCache(artifacts, cache);
      if (inspection.state !== 'ready') {
        throw new ArtifactOperationError(
          'Model artifact cache is not ready',
          'cache-not-ready',
          'generate-music',
          true,
        );
      }
    } catch (error) {
      if (error instanceof ArtifactOperationError) throw error;
      throw new ArtifactOperationError(
        'Artifact cache inspection failed',
        'cache-inspection-failed',
        'generate-music',
        true,
        { cause: error },
      );
    }
    if (!cache) throw new Error('ready artifact cache is unavailable');
    release = { ...fetched, cache };
    artifactFetches = 0;
  }
  const manifestHash = release.hash;
  const artifacts = collectVariableMusicArtifacts(manifest);
  const artifactBytes = artifacts.reduce((total, artifact) => total + artifact.bytes, 0);
  const sampling = capacityDiagnostic
    ? {
        globalGuidance: 1.5,
        semanticTopK: 50,
        residualTopK: 50,
        temperature: 1,
        flowGuidance: 1.7,
        flowSteps: 30,
      }
    : inputRequest.sampling;
  const prepared = capacityDiagnostic
    ? {
        assembledPrompt: FIXED_COMPARISON_CASE.input.assembledPrompt,
        promptTokens: FIXED_COMPARISON_CASE.input.tokenRows[0].length,
        tokenRows: [
          [...FIXED_COMPARISON_CASE.input.tokenRows[0]],
          [...FIXED_COMPARISON_CASE.input.tokenRows[1]],
        ] as [number[], number[]],
      }
    : await prepareProductPrompt(inputRequest, manifest.tokenizerFiles, release.cache);
  const request = capacityDiagnostic
    ? inputRequest
    : createResolvedMusicGenerationRequest(inputRequest, prepared.promptTokens);
  const tracker = createMusicProgressTracker(send, {
    durationSeconds: request.durationSeconds,
    promptTokens: request.promptTokens,
    flowSteps: sampling.flowSteps,
  });
  const { createOrtSession } = await import('../runtime/model/ort-session');
  const ort = await import('onnxruntime-web/jspi');
  const adapters: string[] = [];
  const sessionCreateMs = { autoregressive: 0, condition: 0, flow: 0, vocoder: 0 };
  const stageMs = { autoregressive: 0, condition: 0, flow: 0, vocoder: 0 };
  const inferenceMs = { autoregressive: 0, condition: 0, flow: 0, vocoder: 0 };
  const flowStepMs: number[] = [];
  const adapterName = (adapter: GPUAdapter) =>
    adapter.info.description || adapter.info.vendor || 'WebGPU adapter';
  const webgpu = ort.env.webgpu as unknown as { device?: GPUDevice | Promise<GPUDevice> };
  type OrtDevice = { binding: GPUDevice | Promise<GPUDevice>; device: GPUDevice };
  const captureOrtDevice = async (): Promise<OrtDevice> => {
    const binding = webgpu.device;
    if (!binding) throw new Error('ORT WebGPU session did not expose its device');
    return { binding, device: await binding };
  };
  const captureNewOrtDevice = async (
    previous: GPUDevice | Promise<GPUDevice> | undefined,
  ): Promise<OrtDevice | undefined> => {
    const binding = webgpu.device;
    if (!binding || binding === previous) return undefined;
    return { binding, device: await binding };
  };
  const verifyOrtDevice = (handle: OrtDevice) => {
    if (!handle.device.features.has('shader-f16'))
      throw new Error('ORT WebGPU device does not support shader-f16');
    const limits = handle.device.limits as unknown as Record<string, number>;
    if (Object.entries(manifest.webgpu.requiredLimits).some(
      ([name, value]) => typeof limits[name] !== 'number' || limits[name] < value,
    )) throw new Error('ORT WebGPU device limits are insufficient');
  };
  const settleOrtDevice = async (handle: OrtDevice | undefined): Promise<Failure | undefined> => {
    if (!handle) return undefined;
    let destroyFailure: Failure | undefined;
    try {
      handle.device.destroy();
    } catch (error) {
      destroyFailure = failure(error);
    }
    let clearFailure: Failure | undefined;
    try {
      const current = webgpu.device;
      if (current && await current === handle.device) delete webgpu.device;
    } catch (error) {
      clearFailure = failure(error);
    }
    return destroyFailure ?? clearFailure;
  };
  send({ type: 'progress', stage: 'adapter', detail: 'Checking shader-f16 WebGPU requirements' });
  const capability = await inspectWebGpuForRequirements(navigator.gpu, manifest.webgpu.requiredLimits);
  if (!capability.supported) throw new Error(capability.reason);
  const adapter = capability.adapter;

  let generated: Awaited<ReturnType<ReturnType<typeof createFrameGenerator>['generateFrames']>> | undefined;
  {
    const stageStarted = performance.now();
    tracker.session('autoregressive');
    let decoder: Awaited<ReturnType<typeof createOrtSession>> | undefined;
    let head: Awaited<ReturnType<typeof createOrtSession>> | undefined;
    let rvqDepth: Awaited<ReturnType<typeof createOrtSession>> | undefined;
    let feedback: Awaited<ReturnType<typeof createOrtSession>> | undefined;
    let arDevice: OrtDevice | undefined;
    let primaryFailure: Failure | undefined;
    const previousDeviceBinding = webgpu.device;
    try {
      const sessionStarted = performance.now();
      decoder = await createOrtSession(manifest.graph, release.cache);
      arDevice = await captureOrtDevice();
      verifyOrtDevice(arDevice);
      adapters.push(adapterName(adapter));
      head = await createOrtSession(manifest.reducedHead, release.cache);
      rvqDepth = await createOrtSession(manifest.rvqDepth, release.cache);
      feedback = await createOrtSession(manifest.feedback, release.cache);
      sessionCreateMs.autoregressive = performance.now() - sessionStarted;
      tracker.beginAutoregressive();
      const inferenceStarted = performance.now();
      generated = await createFrameGenerator({
        ort,
        decoder,
        head,
        rvqDepth,
        feedback,
        globalEmbedding: await OpfsFp16EmbeddingTable.open(
          manifest.embedding,
          (path) => release.cache.openSyncFile(path),
        ),
        rvqEmbedding: await OpfsFp16EmbeddingTable.open(
          manifest.rvqEmbedding,
          (path) => release.cache.openSyncFile(path),
        ),
        embeddingColumns: manifest.embedding.columns,
        kvPairs: manifest.kvPairs,
        readConditionalHidden: readConditionalGpuFp16,
        onFrameRetained: (count) => tracker.autoregressive(count),
      }).generateFrames({
        maxFrames: request.requestedFrames,
        seed: request.seed,
        promptTokenRows: {
          conditional: prepared.tokenRows[0],
          unconditional: prepared.tokenRows[1],
        },
        guidance: sampling.globalGuidance,
        semanticTopK: sampling.semanticTopK,
        residualTopK: sampling.residualTopK,
        temperature: sampling.temperature,
        ...(capacityDiagnostic
          ? { audioEndPolicy: 'continue-for-capacity-diagnostic' as const }
          : {}),
      });
      inferenceMs.autoregressive = performance.now() - inferenceStarted;
    } catch (error) {
      primaryFailure = failure(error);
      if (!arDevice) {
        try {
          arDevice = await captureNewOrtDevice(previousDeviceBinding);
        } catch {
          // The original session creation failure remains primary.
        }
      }
    }
    const releaseFailure = await settleSessionReleases([feedback, rvqDepth, head, decoder]);
    const deviceFailure = await settleOrtDevice(arDevice);
    stageMs.autoregressive = performance.now() - stageStarted;
    throwFirstFailure(primaryFailure, releaseFailure, deviceFailure);
  }
  if (!generated) throw new Error('autoregressive generation did not return frames');
  if (capacityDiagnostic && !generated.capacityDiagnostic)
    throw new Error('capacity diagnostic metadata is missing');
  if (!capacityDiagnostic && generated.capacityDiagnostic)
    throw new Error('product generation returned capacity diagnostic metadata');

  tracker.setRetainedFrames(generated.plan.retainedFrames, generated.plan.termination);
  const resultPlan = createMusicGenerationResultPlan(
    request,
    generated.plan.retainedFrames,
    generated.plan.termination,
  );
  let condition: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  let flow: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  let vocoder: Awaited<ReturnType<typeof createOrtSession>> | undefined;
  let flowDevice: OrtDevice | undefined;
  let vocoderDevice: OrtDevice | undefined;
  let wav: ArrayBuffer | undefined;
  let latentChunks: Awaited<ReturnType<typeof runChunkedFlowGeneration>> = [];
  let acousticFailure: Failure | undefined;
  try {
    let flowFailure: Failure | undefined;
    const previousFlowDeviceBinding = webgpu.device;
    try {
      tracker.session('condition');
      let sessionStarted = performance.now();
      condition = await createOrtSession(manifest.conditionEncoder, release.cache);
      flowDevice = await captureOrtDevice();
      verifyOrtDevice(flowDevice);
      adapters.push(adapterName(adapter));
      sessionCreateMs.condition = performance.now() - sessionStarted;
      tracker.session('flow');
      sessionStarted = performance.now();
      flow = await createOrtSession(manifest.flow, release.cache);
      sessionCreateMs.flow = performance.now() - sessionStarted;

      const noiseValues = generated.plan.chunks.reduce(
        (total, chunk) => total + 128 * chunk.latentLength,
        0,
      );
      const noise = deterministicGaussianFp16(request.seed, noiseValues);
      let noiseOffset = 0;
      const initialLatents = generated.plan.chunks.map((chunk) => {
        const nextOffset = noiseOffset + 128 * chunk.latentLength;
        const values = noise.subarray(noiseOffset, nextOffset);
        noiseOffset = nextOffset;
        return values;
      });
      let previousStep = performance.now();
      latentChunks = await runChunkedFlowGeneration(
        { ort, conditionSession: condition, flowSession: flow },
        {
          plan: generated.plan,
          frameHiddens: generated.hiddenGroups,
          initialLatents,
          flowGuidance: sampling.flowGuidance,
          flowSteps: sampling.flowSteps,
          onConditionStart: () => tracker.condition(),
          onConditionComplete: ({ elapsedMs }) => {
            inferenceMs.condition += elapsedMs;
          },
          onChunkStart: (_chunkIndex, completedSteps) => {
            tracker.startFlowChunk(completedSteps);
            previousStep = performance.now();
          },
          onStep: (completedSteps) => {
            const now = performance.now();
            flowStepMs.push(now - previousStep);
            previousStep = now;
            tracker.flow(completedSteps);
          },
          onChunkComplete: ({ chunkIndex, elapsedMs }) => {
            inferenceMs.flow += elapsedMs;
            tracker.acoustic(chunkIndex + 1);
          },
        },
      );
      stageMs.condition = sessionCreateMs.condition + inferenceMs.condition;
      stageMs.flow = sessionCreateMs.flow + inferenceMs.flow;
    } catch (error) {
      flowFailure = failure(error);
      if (!flowDevice) {
        try {
          flowDevice = await captureNewOrtDevice(previousFlowDeviceBinding);
        } catch {
          // The original condition session failure remains primary.
        }
      }
    }
    const flowReleaseFailure = await settleSessionReleases([flow, condition]);
    flow = undefined;
    condition = undefined;
    const flowDeviceFailure = await settleOrtDevice(flowDevice);
    flowDevice = undefined;
    throwFirstFailure(flowFailure, flowReleaseFailure, flowDeviceFailure);

    tracker.session('vocoder');
    const vocoderStageStarted = performance.now();
    const sessionStarted = vocoderStageStarted;
    let vocoderFailure: Failure | undefined;
    const previousVocoderDeviceBinding = webgpu.device;
    try {
      vocoder = await createOrtSession(manifest.vocoder, release.cache);
      vocoderDevice = await captureOrtDevice();
      verifyOrtDevice(vocoderDevice);
      adapters.push(adapterName(adapter));
      sessionCreateMs.vocoder = performance.now() - sessionStarted;
      wav = await generateVariableVocoderWav(
        { ort, session: vocoder },
        generated.plan,
        latentChunks.map((chunk) => chunk.latentBits),
        ({ completedCalls, inferenceMs: channelInferenceMs }) => {
          inferenceMs.vocoder += channelInferenceMs;
          tracker.vocoder(completedCalls);
        },
        () => tracker.wav(),
      );
      stageMs.vocoder = performance.now() - vocoderStageStarted;
    } catch (error) {
      vocoderFailure = failure(error);
      if (!vocoderDevice) {
        try {
          vocoderDevice = await captureNewOrtDevice(previousVocoderDeviceBinding);
        } catch {
          // The original vocoder session failure remains primary.
        }
      }
    }
    const vocoderReleaseFailure = await settleSessionReleases([vocoder]);
    vocoder = undefined;
    const vocoderDeviceFailure = await settleOrtDevice(vocoderDevice);
    vocoderDevice = undefined;
    throwFirstFailure(vocoderFailure, vocoderReleaseFailure, vocoderDeviceFailure);
  } catch (error) {
    acousticFailure = failure(error);
  }
  const remainingReleaseFailure = await settleSessionReleases([vocoder, flow, condition]);
  const remainingVocoderDeviceFailure = await settleOrtDevice(vocoderDevice);
  const remainingFlowDeviceFailure = await settleOrtDevice(flowDevice);
  throwFirstFailure(
    acousticFailure,
    remainingReleaseFailure,
    remainingVocoderDeviceFailure,
    remainingFlowDeviceFailure,
  );
  if (!wav) throw new Error('vocoder generation did not return a WAV');
  tracker.complete(wav.byteLength);
  const comparison = capacityDiagnostic
    ? undefined
    : createFixedComparisonMetadata({
        input: {
          prompt: inputRequest.prompt,
          lyrics: inputRequest.lyrics,
          assembledPrompt: prepared.assembledPrompt,
          tokenRows: prepared.tokenRows,
        },
        sampling,
        seed: request.seed,
        durationSeconds: request.durationSeconds,
        plan: resultPlan,
        manifestHash,
        browser: navigator.userAgent,
        ortVersion: PINNED_COMPARISON_ORT_VERSION,
      });
  const metrics = {
    wav,
    adapters,
    attemptedSeeds: [request.seed],
    hiddenBytes: generated.hiddenGroups.byteLength,
    conditionBytes: generated.plan.chunks.reduce(
      (total, chunk) => total + chunk.latentLength * 2048 * 2,
      0,
    ),
    latentBytes: latentChunks.reduce((total, chunk) => total + chunk.latentBits.byteLength, 0),
    wavBytes: wav.byteLength,
    artifactBytes,
    artifactFetches,
    manifestHash,
    sessionCreateMs,
    stageMs,
    inferenceMs,
    flowStepMs,
    browser: navigator.userAgent,
    ortVersion: PINNED_COMPARISON_ORT_VERSION,
    status: 'passed' as const,
  };
  if (inputRequest.type === 'diagnose-music-capacity') {
    if (!generated.capacityDiagnostic) throw new Error('capacity diagnostic metadata is missing');
    send({
      type: 'music-result',
      result: {
        ...metrics,
        plan: resultPlan,
        capacityDiagnostic: generated.capacityDiagnostic,
      },
    }, [wav]);
  } else {
    send({
      type: 'music-result',
      result: {
        ...metrics,
        plan: resultPlan,
        effectiveInput: {
          prompt: inputRequest.prompt,
          lyrics: inputRequest.lyrics,
          assembledPrompt: prepared.assembledPrompt,
          tokenRows: prepared.tokenRows,
          promptTokens: prepared.promptTokens,
          seed: request.seed,
          durationSeconds: request.durationSeconds,
          sampling,
        },
        ...(comparison ? { comparison } : {}),
      },
    }, [wav]);
  }
}

async function runMusicGeneration(request: Extract<WorkerRequest, { type: 'generate-music-5s' }>) {
  const progressTracker = createMusicProgressTracker(send);
  send({ type: 'progress', stage: 'manifest', detail: 'Reading five-second music release manifest' });
  const release = await readManifest(request.manifestUrl, 'Music release manifest is unavailable');
  const manifestHash = await hashText(release.text);
  const manifest = parseMusicManifest(JSON.parse(release.text));
  const allArtifacts = [
    manifest.graph,
    ...manifest.graph.externalData,
    manifest.reducedHead,
    ...manifest.reducedHead.externalData,
    ...manifest.embedding.shards,
    manifest.rvqDepth,
    ...manifest.rvqDepth.externalData,
    manifest.feedback,
    ...manifest.feedback.externalData,
    ...manifest.rvqEmbedding.shards,
    manifest.conditionEncoder,
    ...manifest.conditionEncoder.externalData,
    manifest.flow,
    ...manifest.flow.externalData,
    manifest.vocoder,
    ...manifest.vocoder.externalData,
    ...manifest.tokenizerFiles,
    manifest.licenseFile,
  ];
  const artifacts = [...new Map(allArtifacts.map((artifact) => [artifact.path, artifact])).values()];
  const artifactFetches = await cacheArtifacts(artifacts, release.base, release.cache);
  const artifactBytes = artifacts.reduce((total, artifact) => total + artifact.bytes, 0);
  const ort = await import('onnxruntime-web/jspi');
  const adapters: string[] = [];
  const sessionCreateMs = { autoregressive: 0, condition: 0, flow: 0, vocoder: 0 };
  const stageMs = { autoregressive: 0, condition: 0, flow: 0, vocoder: 0 };
  const inferenceMs = { autoregressive: 0, condition: 0, flow: 0, vocoder: 0 };
  const flowStepMs: number[] = [];
  let reportedFrames = 0;
  const adapterName = (adapter: GPUAdapter) =>
    adapter.info.description || adapter.info.vendor || 'WebGPU adapter';

  const generated = await generateFiveSecondMusic(
    {
      async autoregressive(seed) {
        const stageStarted = performance.now();
        progressTracker.session('autoregressive');
        try {
          return await withOrtOwnedDevice(manifest.webgpu.requiredLimits, async (createOrtSession, adapter) => {
            adapters.push(adapterName(adapter));
            const sessionStarted = performance.now();
            const decoder = await createOrtSession(manifest.graph, release.cache);
            const head = await createOrtSession(manifest.reducedHead, release.cache);
            const rvqDepth = await createOrtSession(manifest.rvqDepth, release.cache);
            const feedback = await createOrtSession(manifest.feedback, release.cache);
            sessionCreateMs.autoregressive += performance.now() - sessionStarted;
            progressTracker.beginAutoregressive();
            const generationStarted = performance.now();
            const frames = await createFrameGenerator({
              ort,
              decoder,
              head,
              rvqDepth,
              feedback,
              globalEmbedding: await OpfsFp16EmbeddingTable.open(
                manifest.embedding,
                (path) => release.cache.openSyncFile(path),
              ),
              rvqEmbedding: await OpfsFp16EmbeddingTable.open(
                manifest.rvqEmbedding,
                (path) => release.cache.openSyncFile(path),
              ),
              embeddingColumns: manifest.embedding.columns,
              kvPairs: manifest.kvPairs,
              readConditionalHidden: readConditionalGpuFp16,
              onFrameRetained: (count) => {
                if (count > reportedFrames) {
                  reportedFrames = count;
                  progressTracker.autoregressive(count);
                }
              },
            }).generateFrames({
              maxFrames: 125,
              seed,
              promptTokenRows: {
                conditional: FIXED_COMPARISON_CASE.input.tokenRows[0],
                unconditional: FIXED_COMPARISON_CASE.input.tokenRows[1],
              },
              guidance: 1.5,
              semanticTopK: 50,
              residualTopK: 50,
              temperature: 1,
            });
            inferenceMs.autoregressive += performance.now() - generationStarted;
            return frames;
          });
        } finally {
          stageMs.autoregressive += performance.now() - stageStarted;
        }
      },
      async condition(frameBits) {
        const stageStarted = performance.now();
        progressTracker.session('condition');
        try {
          return await withOrtOwnedDevice(manifest.webgpu.requiredLimits, async (createOrtSession, adapter) => {
            adapters.push(adapterName(adapter));
            const sessionStarted = performance.now();
            const session = await createOrtSession(manifest.conditionEncoder, release.cache);
            sessionCreateMs.condition = performance.now() - sessionStarted;
            const input = new ort.Tensor('float16', frameBits, [1, 125, 32_768]);
            let output: InstanceType<typeof ort.Tensor> | undefined;
            try {
              progressTracker.condition();
              const inferenceStarted = performance.now();
              const outputs = await session.run({ frame_hiddens: input });
              inferenceMs.condition = performance.now() - inferenceStarted;
              output = outputs.condition;
              if (!output) throw new Error('condition encoder did not return condition');
              return await readExactGpuFp16(output, [1, 430, 2048], 'condition');
            } finally {
              output?.dispose();
              input.dispose();
            }
          });
        } finally {
          stageMs.condition = performance.now() - stageStarted;
        }
      },
      async flow(conditionBits, seed, onStep) {
        const stageStarted = performance.now();
        progressTracker.session('flow');
        try {
          return await withOrtOwnedDevice(manifest.webgpu.requiredLimits, async (createOrtSession, adapter) => {
            adapters.push(adapterName(adapter));
            const sessionStarted = performance.now();
            const session = await createOrtSession(manifest.flow, release.cache);
            sessionCreateMs.flow = performance.now() - sessionStarted;
            const condition = new ort.Tensor('float16', conditionBits, [1, 430, 2048]);
            let final: InstanceType<typeof ort.Tensor> | undefined;
            const initial = new ort.Tensor(
              'float16',
              deterministicGaussianFp16(seed, 128 * 430),
              [1, 128, 430],
            );
            try {
              let previous = performance.now();
              progressTracker.beginFlow();
              const inferenceStarted = performance.now();
              final = await runFixedFlowGeneration({ ort, session }, initial, condition, (completed) => {
                const now = performance.now();
                flowStepMs.push(now - previous);
                previous = now;
                onStep(completed);
              });
              inferenceMs.flow = performance.now() - inferenceStarted;
              return await readExactGpuFp16(final, [1, 128, 430], 'latents');
            } finally {
              final?.dispose();
              condition.dispose();
            }
          });
        } finally {
          stageMs.flow = performance.now() - stageStarted;
        }
      },
      async vocoder(latentBits) {
        const stageStarted = performance.now();
        progressTracker.session('vocoder');
        try {
          return await withOrtOwnedDevice(manifest.webgpu.requiredLimits, async (createOrtSession, adapter) => {
            adapters.push(adapterName(adapter));
            const sessionStarted = performance.now();
            const session = await createOrtSession(manifest.vocoder, release.cache);
            sessionCreateMs.vocoder = performance.now() - sessionStarted;
            progressTracker.vocoder();
            const inferenceStarted = performance.now();
            const wav = await generateFixedVocoderWav({ ort, session }, latentBits);
            inferenceMs.vocoder = performance.now() - inferenceStarted;
            validateVocoderWav(wav);
            return wav;
          });
        } finally {
          stageMs.vocoder = performance.now() - stageStarted;
        }
      },
    },
    request.seed,
    (progress) => {
      if (progress.stage === 'flow') progressTracker.flow(progress.completedSteps);
      else if (progress.stage === 'wav') progressTracker.wav();
      else if (progress.stage === 'complete') progressTracker.complete(880_684);
    },
  );
  const result = {
    ...generated,
    adapters,
    wavBytes: generated.wav.byteLength,
    artifactBytes,
    artifactFetches,
    manifestHash,
    sessionCreateMs,
    stageMs,
    inferenceMs,
    flowStepMs,
    browser: navigator.userAgent,
    ortVersion: PINNED_COMPARISON_ORT_VERSION,
    status: 'passed' as const,
  };
  send({ type: 'music-result', result }, [generated.wav]);
}

async function runRvq(manifestUrl: string) {
  send({ type: 'progress', stage: 'manifest', detail: 'Reading RVQ release manifest' });
  let response: Response;
  try {
    response = await fetch(manifestUrl);
  } catch {
    throw new Error('RVQ release manifest is unavailable');
  }
  if (!response.ok) throw new Error('RVQ release manifest is unavailable');
  const manifestText = await response.text();
  const manifest = parseRvqStageManifest(JSON.parse(manifestText));
  const cache = await OpfsArtifactStore.open(await hashText(manifestText));
  const base = new URL(manifestUrl, self.location.href);
  const artifacts = [
    manifest.rvqDepth,
    ...manifest.rvqDepth.externalData,
    manifest.feedback,
    ...manifest.feedback.externalData,
    ...manifest.rvqEmbedding.shards,
  ];
  const artifactFetches = await cacheArtifacts(artifacts, base, cache);
  send({ type: 'progress', stage: 'adapter', detail: 'Checking shader-f16 WebGPU requirements' });
  const ort = await import('onnxruntime-web/jspi');
  await withOrtOwnedDevice(manifest.webgpu.requiredLimits, async (createOrtSession, adapter) => {
    send({ type: 'progress', stage: 'session', detail: 'Creating WebGPU RVQ and feedback sessions' });
    const started = performance.now();
    const rvqDepth = await createOrtSession(manifest.rvqDepth, cache);
    const feedback = await createOrtSession(manifest.feedback, cache);
    const sessionCreateMs = performance.now() - started;
    const metrics = await runRvqSmoke({
      ort,
      rvqDepth,
      feedback,
      embedding: await OpfsFp16EmbeddingTable.open(
        manifest.rvqEmbedding,
        (path) => cache.openSyncFile(path),
      ),
      embeddingTable: manifest.rvqEmbedding,
    });
    send({
      type: 'rvq-result',
      result: {
        adapter: adapter.info.description || adapter.info.vendor || 'WebGPU adapter',
        ...metrics,
        sessionCreateMs,
        artifactFetches,
        status: 'passed',
      },
    });
  });
}
