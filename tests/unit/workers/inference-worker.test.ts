import { beforeEach, describe, expect, it, vi } from 'vitest';
import { planRetainedFrames } from '../../../src/runtime/pipeline/duration-plan';
import { FIXED_COMPARISON_CASE } from '../../../src/runtime/reference/fixed-comparison';

const state = vi.hoisted(() => ({
  events: [] as string[],
  messages: [] as { message: Record<string, unknown>; transfer: Transferable[] }[],
  failFlow: false,
  createFailure: undefined as string | undefined,
  publishedCreateFailure: undefined as string | undefined,
  releaseFailures: [] as string[],
  replaceBindingOnRelease: undefined as string | undefined,
  actualDeviceCount: 0,
  activeActualDevice: undefined as object | undefined,
  retainedPlan: undefined as ReturnType<typeof planRetainedFrames> | undefined,
  generateFrameOptions: undefined as Record<string, unknown> | undefined,
  flowRequest: undefined as Record<string, unknown> | undefined,
  preparedPrompt: undefined as
    | {
        assembledPrompt: string;
        promptTokens: number;
        tokenRows: [number[], number[]];
      }
    | undefined,
  invalidShippedTokenizer: false,
  promptPrepareCalls: 0,
  preflightGate: undefined as Promise<void> | undefined,
  ortWebgpu: {} as { device?: unknown },
  cacheState: 'ready' as 'missing' | 'partial' | 'ready',
  ensureError: undefined as Error | undefined,
  inspectionError: undefined as Error | undefined,
  deleteError: undefined as Error | undefined,
  storageEstimate: { usage: 10, quota: 100 } as { usage?: number; quota?: number } | undefined,
  storagePersisted: true as boolean | Error,
  projectCacheCount: 1,
  projectCacheBytes: 11,
  activeCacheLock: undefined as 'exclusive' | 'shared' | undefined,
  lockAtPreflight: [] as ('exclusive' | 'shared' | undefined)[],
  cacheOpenCalls: [] as { hash: string; root?: FileSystemDirectoryHandle }[],
  cacheOpenExistingCalls: [] as { hash: string; root?: FileSystemDirectoryHandle }[],
  inspectedArtifactPaths: [] as string[][],
}));
const opfsRoot = vi.hoisted(() => ({}) as FileSystemDirectoryHandle);

type TestArtifact = { path: string; bytes: number; sha256: string };
const sharedArtifact = { path: 'shared.bin', bytes: 1, sha256: 'a'.repeat(64) };
const graph = (path: string, externalData: TestArtifact[] = []) => ({
  path,
  bytes: 1,
  sha256: 'a'.repeat(64),
  externalData,
  gpuOutputs: [],
});
const variableManifest = {
  graph: graph('decoder', [sharedArtifact]),
  reducedHead: graph('head'),
  embedding: { columns: 32_768, shards: [] },
  kvPairs: [],
  rvqDepth: graph('rvq-depth'),
  rvqEmbedding: { shards: [] },
  feedback: graph('feedback'),
  conditionEncoder: graph('condition'),
  flow: graph('flow', [sharedArtifact]),
  vocoder: graph('vocoder'),
  tokenizerFiles: [
    { path: 'global/tokenizer/tokenizer.json', bytes: 1, sha256: 'a'.repeat(64) },
    { path: 'global/tokenizer/tokenizer_config.json', bytes: 1, sha256: 'a'.repeat(64) },
  ],
  licenseFile: { path: 'LICENSE', bytes: 1, sha256: 'a'.repeat(64) },
  acoustic: { flowSteps: 30 },
  webgpu: { requiredLimits: { maxStorageBufferBindingSize: 128 * 1024 * 1024 } },
};
const expectedVariableArtifactPaths = [
  'decoder',
  'shared.bin',
  'head',
  'rvq-depth',
  'feedback',
  'condition',
  'flow',
  'vocoder',
  'global/tokenizer/tokenizer.json',
  'global/tokenizer/tokenizer_config.json',
  'LICENSE',
];
const emptyObjectManifestHash = '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a';

vi.mock('../../../src/runtime/model/artifact-cache', () => ({
  OpfsArtifactStore: {
    open: vi.fn(async (hash: string, root?: FileSystemDirectoryHandle) => {
      state.cacheOpenCalls.push({ hash, root });
      state.events.push('cache:open');
      return {
        openSyncFile: vi.fn(),
        file: vi.fn(async (path: string) => ({
          text: vi.fn(async () => {
            state.events.push(`file:text:${path}`);
            return path.endsWith('tokenizer_config.json') ? '{"config":true}' : '{"tokenizer":true}';
          }),
        })),
      };
    }),
    openExisting: vi.fn(async (hash: string, root?: FileSystemDirectoryHandle) => {
      state.cacheOpenExistingCalls.push({ hash, root });
      state.events.push('cache:open-existing');
      if (state.cacheState === 'missing') return undefined;
      return {
        openSyncFile: vi.fn(),
        file: vi.fn(async (path: string) => ({
          text: vi.fn(async () => {
            state.events.push(`file:text:${path}`);
            return path.endsWith('tokenizer_config.json') ? '{"config":true}' : '{"tokenizer":true}';
          }),
        })),
      };
    }),
  },
  ensureArtifact: vi.fn(async (artifact: { path: string }) => {
    state.events.push(`artifact:${artifact.path}`);
    if (state.ensureError) throw state.ensureError;
    state.cacheState = 'ready';
  }),
}));
vi.mock('../../../src/runtime/model/artifact-cache-management', () => {
  let tail = Promise.resolve();
  const lock = async (mode: 'exclusive' | 'shared', action: () => Promise<unknown>) => {
    state.events.push(`cache:lock:${mode}:requested`);
    const previous = tail;
    let release!: () => void;
    tail = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    state.activeCacheLock = mode;
    state.events.push(`cache:lock:${mode}:start`);
    try {
      return await action();
    } finally {
      state.events.push(`cache:lock:${mode}:end`);
      state.activeCacheLock = undefined;
      release();
    }
  };
  return {
    inspectArtifactCache: vi.fn(async (artifacts: readonly { path: string; bytes: number }[], store: unknown) => {
      if (state.inspectionError) throw state.inspectionError;
      state.inspectedArtifactPaths.push(artifacts.map((artifact) => artifact.path));
      state.events.push(`cache:inspect:${artifacts.length}`);
      const cacheState = store === undefined ? 'missing' : state.cacheState;
      const ready = cacheState === 'ready';
      const totalArtifactBytes = artifacts.reduce((sum, artifact) => sum + artifact.bytes, 0);
      return {
        state: cacheState,
        artifactCount: artifacts.length,
        totalArtifactBytes,
        completeArtifactCount: ready ? artifacts.length : 0,
        completeArtifactBytes: ready ? totalArtifactBytes : 0,
        storedReferencedBytes: ready ? totalArtifactBytes : 0,
        additionalBytesNeeded: ready ? 0 : totalArtifactBytes,
        largestPendingArtifactBytes: ready ? 0 : Math.max(...artifacts.map(({ bytes }) => bytes)),
      };
    }),
    assessArtifactCapacity: vi.fn(
      (
        inspection: { additionalBytesNeeded: number; largestPendingArtifactBytes: number },
        estimate?: { usage?: number; quota?: number },
      ) => {
        const requiredHeadroomBytes = inspection.additionalBytesNeeded + inspection.largestPendingArtifactBytes;
        if (estimate?.usage === undefined || estimate.quota === undefined) return { requiredHeadroomBytes };
        const availableBytes = estimate.quota - estimate.usage;
        return {
          usageBytes: estimate.usage,
          quotaBytes: estimate.quota,
          availableBytes,
          sufficient: availableBytes >= requiredHeadroomBytes,
          requiredHeadroomBytes,
        };
      },
    ),
    inspectProjectArtifactCaches: vi.fn(async () => ({
      cacheCount: state.projectCacheCount,
      storedBytes: state.projectCacheBytes,
    })),
    deleteProjectArtifactCaches: vi.fn(async () => {
      state.events.push('cache:delete');
      if (state.deleteError) throw state.deleteError;
      state.cacheState = 'missing';
      state.projectCacheCount = 0;
      state.projectCacheBytes = 0;
    }),
    withArtifactCacheMutationLock: vi.fn((action: () => Promise<unknown>) => lock('exclusive', action)),
    withArtifactCacheReadLock: vi.fn((action: () => Promise<unknown>) => lock('shared', action)),
  };
});
vi.mock('../../../src/runtime/model/embedding-table', () => ({
  OpfsFp16EmbeddingTable: { open: vi.fn(async () => ({})) },
}));
vi.mock('../../../src/runtime/model/manifest', () => ({
  parseConditionManifest: vi.fn(() => variableManifest),
  parseFlowManifest: vi.fn(() => variableManifest),
  parseMusicManifest: vi.fn(() => variableManifest),
  parseMusicVariableManifest: vi.fn(() => variableManifest),
  parseModelManifest: vi.fn(() => variableManifest),
  parseRvqStageManifest: vi.fn(() => variableManifest),
  parseVocoderManifest: vi.fn(() => variableManifest),
}));
vi.mock('../../../src/runtime/model/webgpu-device', () => ({
  inspectWebGpuForRequirements: vi.fn(async () => {
    state.lockAtPreflight.push(state.activeCacheLock);
    state.events.push('adapter:preflight');
    await state.preflightGate;
    return { supported: true, adapter: { info: { description: 'synthetic adapter' } } };
  }),
}));
vi.mock('../../../src/runtime/model/ort-session', () => ({
  createOrtSession: vi.fn(async (artifact: { path: string }) => {
    state.events.push(`session:create:${artifact.path}`);
    if (state.createFailure === artifact.path) throw new Error(`synthetic ${artifact.path} creation failure`);
    if (state.ortWebgpu.device === undefined) {
      const name = ['ar', 'flow', 'vocoder', 'vocoder-2'][state.actualDeviceCount++];
      const device = {
        name,
        features: new Set(['shader-f16']),
        limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024 },
        destroy: () => state.events.push(`device:destroy:${name}`),
      };
      state.events.push(`device:create:${name}`);
      state.activeActualDevice = device;
    }
    state.ortWebgpu.device = Promise.resolve(state.activeActualDevice);
    if (state.publishedCreateFailure === artifact.path)
      throw new Error(`synthetic published ${artifact.path} creation failure`);
    return {
      run: vi.fn(async () => ({ condition: { dispose: vi.fn() } })),
      release: vi.fn(async () => {
        state.events.push(`session:release:${artifact.path}`);
        if (state.replaceBindingOnRelease === artifact.path) {
          const newer = {
            name: 'newer',
            features: new Set(['shader-f16']),
            limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024 },
            destroy: () => state.events.push('device:destroy:newer'),
          };
          state.events.push('device:create:newer');
          state.ortWebgpu.device = Promise.resolve(newer);
        }
        if (state.releaseFailures.includes(artifact.path))
          throw new Error(`synthetic ${artifact.path} release failure`);
      }),
    };
  }),
}));
vi.mock('../../../src/runtime/pipeline/rvq-generation', () => ({
  areFiniteFp16: vi.fn(),
  EarlyAudioEndError: class extends Error {},
  readConditionalGpuFp16: vi.fn(),
  createFrameGenerator: vi.fn((runtime) => ({
    generateFrames: vi.fn(async (options: Record<string, unknown>) => {
      state.generateFrameOptions = options;
      state.events.push('ar:run');
      const plan = state.retainedPlan!;
      runtime.onFrameRetained?.(1);
      runtime.onFrameRetained?.(plan.retainedFrames);
      const frames = Object.assign([], {
        hiddenGroups: new Uint16Array(plan.retainedFrames * 8 * 4096),
        termination: plan.termination,
        plan,
        ...(options.audioEndPolicy === 'continue-for-capacity-diagnostic'
          ? {
              capacityDiagnostic: {
                kind: 'continue-after-audio-end',
                suppressedAudioEnds: 1,
                firstAudioEndAtRetainedFrame: 1_743,
              },
            }
          : {}),
      });
      return frames;
    }),
  })),
}));
vi.mock('../../../src/runtime/pipeline/flow-generation', () => ({
  runFixedFlowGeneration: vi.fn(async (_runtime, _initial, _condition, onStep) => {
    for (let step = 1; step <= 30; step++) onStep?.(step);
    return { dispose: vi.fn() };
  }),
  runFlowSmoke: vi.fn(),
  runChunkedFlowGeneration: vi.fn(async (_runtime, request) => {
    state.flowRequest = request;
    for (let index = 0; index < request.plan.chunks.length; index++) {
      state.events.push(`flow:chunk:${index}`);
      request.onConditionStart?.(index);
      state.events.push(
        `condition-progress-before-encode:${state.messages.some(({ message }) => message.stage === 'condition')}`,
      );
      request.onConditionComplete?.({ chunkIndex: index, elapsedMs: 10 + index });
      request.onChunkStart?.(index, index * request.flowSteps);
      if (state.failFlow) throw new Error('synthetic flow failure');
      for (let step = 1; step <= request.flowSteps; step++) request.onStep?.(index * request.flowSteps + step);
      state.events.push(
        `acoustic-before-complete:${state.messages.some(
          ({ message }) => message.stage === 'acoustic' && message.completed === index + 1,
        )}`,
      );
      request.onChunkComplete?.({ chunkIndex: index, elapsedMs: 100 + index });
    }
    return request.plan.chunks.map((chunk: { latentLength: number }) => ({
      ...chunk,
      latentBits: new Uint16Array(128 * chunk.latentLength),
    }));
  }),
}));
vi.mock('../../../src/runtime/pipeline/prompt-preparation', () => ({
  createPromptTokenizer: vi.fn((tokenizerJson: string, tokenizerConfigJson: string) => {
    state.events.push(`tokenizer:create:${tokenizerJson}:${tokenizerConfigJson}`);
    return { encode: vi.fn() };
  }),
  preparePrompt: vi.fn(({ prompt, lyrics, requestedFrames }) => {
    state.promptPrepareCalls++;
    state.events.push(`prompt:prepare:${prompt}:${lyrics}:${requestedFrames}`);
    if (state.invalidShippedTokenizer && state.promptPrepareCalls === 1)
      return {
        assembledPrompt: 'mismatch',
        promptTokens: 3,
        tokenRows: [
          [1, 2, 3],
          [1, 2, 3],
        ],
      };
    return state.preparedPrompt!;
  }),
}));
vi.mock('../../../src/runtime/pipeline/vocoder-generation', () => ({
  analyticVocoderLatents: vi.fn(),
  generateFixedVocoderWav: vi.fn(async () => {
    const wav = new ArrayBuffer(880_684);
    const bytes = new Uint8Array(wav);
    bytes.set(new TextEncoder().encode('RIFF'), 0);
    bytes.set(new TextEncoder().encode('WAVE'), 8);
    const view = new DataView(wav);
    view.setUint16(20, 1, true);
    view.setUint16(22, 2, true);
    view.setUint32(24, 44_100, true);
    view.setUint16(34, 16, true);
    view.setUint32(40, 880_640, true);
    return wav;
  }),
  generateVariableVocoderWav: vi.fn(async (_runtime, plan, _chunks, observer, onWavStart) => {
    onWavStart?.();
    state.events.push(
      `wav-progress-before-allocation:${state.messages.some(({ message }) => message.stage === 'wav')}`,
    );
    let completedCalls = 0;
    for (let chunkIndex = 0; chunkIndex < plan.chunks.length; chunkIndex++) {
      for (const channel of ['left', 'right'] as const) {
        completedCalls++;
        state.events.push(`vocoder:${chunkIndex}:${channel}`);
        observer?.({
          chunkIndex,
          channel,
          completedCalls,
          totalCalls: plan.chunks.length * 2,
          inferenceMs: 2.5,
          pcmWriteMs: 0.5,
        });
      }
    }
    return new ArrayBuffer(plan.wavBytes);
  }),
}));
vi.mock('../../../src/runtime/pipeline/music-generation', () => ({
  deterministicGaussianFp16: vi.fn((_seed: number, length: number) => {
    state.events.push(`noise:${length}`);
    return new Uint16Array(length);
  }),
  generateFiveSecondMusic: vi.fn(async (runtime, seed, onProgress) => {
    const frames = await runtime.autoregressive(seed);
    const condition = await runtime.condition(frames.hiddenGroups);
    const latents = await runtime.flow(condition, seed, (completed: number) => {
      onProgress({ stage: 'flow', completedSteps: completed });
    });
    const wav = await runtime.vocoder(latents);
    onProgress({ stage: 'wav' });
    onProgress({ stage: 'complete' });
    return { wav, attemptedSeeds: [seed] };
  }),
  readExactGpuFp16: vi.fn(
    async (_tensor, shape: number[]) => new Uint16Array(shape.reduce((total, value) => total * value, 1)),
  ),
}));
vi.mock('../../../src/runtime/pipeline/condition-smoke', () => ({ runConditionSmoke: vi.fn() }));
vi.mock('../../../src/runtime/pipeline/global-smoke', () => ({
  runGlobalSmoke: vi.fn(async () => ({})),
}));
vi.mock('../../../src/runtime/pipeline/rvq-smoke', () => ({ runRvqSmoke: vi.fn() }));
vi.mock('onnxruntime-web/jspi', () => ({
  env: { webgpu: state.ortWebgpu },
  Tensor: class {
    dispose = vi.fn();
  },
}));

Object.defineProperty(globalThis, 'self', {
  configurable: true,
  value: {
    location: { href: 'http://worker.test/inference.js', origin: 'http://worker.test' },
    postMessage: (message: Record<string, unknown>, transfer: Transferable[] = []) => {
      state.messages.push({ message, transfer });
    },
  },
});
Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: {
    gpu: {},
    userAgent: 'Mozilla/5.0 Chrome/140.0.7339.81 Safari/537.36',
    storage: {
      getDirectory: vi.fn(async () => {
        state.events.push('storage:get-directory');
        return opfsRoot;
      }),
      persisted: vi.fn(async () => {
        if (state.storagePersisted instanceof Error) throw state.storagePersisted;
        return state.storagePersisted;
      }),
      estimate: vi.fn(async () => state.storageEstimate),
    },
    locks: { request: vi.fn() },
  },
});
vi.stubGlobal(
  'fetch',
  vi.fn(async () => new Response(JSON.stringify({}), { status: 200 })),
);

const { runWorkerRequest } = await import('../../../src/workers/inference.worker');

const musicRequest = (overrides: Record<string, unknown> = {}) => ({
  type: 'generate-music',
  manifestUrl: 'http://worker.test/music-variable/manifest.json',
  prompt: FIXED_COMPARISON_CASE.input.prompt,
  lyrics: FIXED_COMPARISON_CASE.input.lyrics,
  seed: 7,
  durationSeconds: 10,
  sampling: {
    globalGuidance: 1.5,
    semanticTopK: 50,
    residualTopK: 50,
    temperature: 1,
    flowGuidance: 1.7,
    flowSteps: 30,
  },
  ...overrides,
});

describe('variable inference worker lifecycle', () => {
  beforeEach(() => {
    vi.mocked(fetch).mockImplementation(async () => new Response(JSON.stringify({}), { status: 200 }));
    state.events.length = 0;
    state.messages.length = 0;
    state.failFlow = false;
    state.createFailure = undefined;
    state.publishedCreateFailure = undefined;
    state.releaseFailures.length = 0;
    state.replaceBindingOnRelease = undefined;
    state.actualDeviceCount = 0;
    state.activeActualDevice = undefined;
    state.generateFrameOptions = undefined;
    state.flowRequest = undefined;
    state.preparedPrompt = {
      assembledPrompt: FIXED_COMPARISON_CASE.input.assembledPrompt,
      promptTokens: 40,
      tokenRows: [[...FIXED_COMPARISON_CASE.input.tokenRows[0]], [...FIXED_COMPARISON_CASE.input.tokenRows[1]]],
    };
    state.invalidShippedTokenizer = false;
    state.promptPrepareCalls = 0;
    state.preflightGate = undefined;
    state.cacheState = 'ready';
    state.ensureError = undefined;
    state.inspectionError = undefined;
    state.deleteError = undefined;
    state.storageEstimate = { usage: 10, quota: 100 };
    state.storagePersisted = true;
    state.projectCacheCount = 1;
    state.projectCacheBytes = 11;
    state.activeCacheLock = undefined;
    state.lockAtPreflight.length = 0;
    state.cacheOpenCalls.length = 0;
    state.cacheOpenExistingCalls.length = 0;
    state.inspectedArtifactPaths.length = 0;
    delete state.ortWebgpu.device;
    state.retainedPlan = planRetainedFrames({
      retainedFrames: 201,
      promptTokens: 40,
      termination: 'natural-end',
    });
  });

  it('inspects the variable artifact cache without creating or downloading it', async () => {
    await runWorkerRequest({
      type: 'inspect-artifact-cache',
      manifestUrl: 'http://worker.test/music-variable/manifest.json',
    });

    expect(state.events).toContain('cache:open-existing');
    expect(state.events).not.toContain('cache:open');
    expect(state.events.some((event) => event.startsWith('artifact:'))).toBe(false);
    expect(state.cacheOpenExistingCalls).toEqual([{ hash: emptyObjectManifestHash, root: opfsRoot }]);
    expect(state.inspectedArtifactPaths).toEqual([expectedVariableArtifactPaths]);
    expect(state.messages.at(-1)?.message).toMatchObject({
      type: 'artifact-cache-status',
      status: {
        state: 'ready',
        projectCacheCount: 1,
        projectCacheBytes: 11,
        persistence: 'persistent',
        usageBytes: 10,
        quotaBytes: 100,
      },
    });
  });

  it('classifies unavailable and invalid manifests for artifact operations', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError('network unavailable'));
    await expect(
      runWorkerRequest({
        type: 'inspect-artifact-cache',
        manifestUrl: 'http://worker.test/music-variable/manifest.json',
      }),
    ).rejects.toMatchObject({
      code: 'manifest-unavailable',
      operation: 'inspect-artifact-cache',
      retryable: true,
      // A CORS block and a refused connection are indistinguishable to `fetch`, so the message
      // has to name the URL it tried and repeat what the browser said.
      message:
        'Music release manifest is unavailable: GET http://worker.test/music-variable/manifest.json failed (network unavailable)',
    });

    vi.mocked(fetch).mockResolvedValueOnce(new Response('{}', { status: 404 }));
    await expect(
      runWorkerRequest({
        type: 'inspect-artifact-cache',
        manifestUrl: 'http://worker.test/music-variable/manifest.json',
      }),
    ).rejects.toMatchObject({
      code: 'manifest-unavailable',
      retryable: true,
      message:
        'Music release manifest is unavailable: GET http://worker.test/music-variable/manifest.json returned 404',
    });

    const { parseMusicVariableManifest } = await import('../../../src/runtime/model/manifest');
    vi.mocked(parseMusicVariableManifest).mockImplementationOnce(() => {
      throw new Error('invalid manifest');
    });
    await expect(
      runWorkerRequest({
        type: 'inspect-artifact-cache',
        manifestUrl: 'http://worker.test/music-variable/manifest.json',
      }),
    ).rejects.toMatchObject({
      code: 'manifest-invalid',
      operation: 'inspect-artifact-cache',
      retryable: false,
    });
  });

  it('maps generic inspection, download, and partial delete failures', async () => {
    state.inspectionError = new Error('inspection failed');
    await expect(
      runWorkerRequest({
        type: 'inspect-artifact-cache',
        manifestUrl: 'http://worker.test/music-variable/manifest.json',
      }),
    ).rejects.toMatchObject({ code: 'cache-inspection-failed' });

    state.inspectionError = undefined;
    state.cacheState = 'missing';
    state.ensureError = new Error('artifact request failed: shared.bin');
    await expect(
      runWorkerRequest({
        type: 'download-artifacts',
        manifestUrl: 'http://worker.test/music-variable/manifest.json',
      }),
    ).rejects.toMatchObject({
      code: 'download-failed',
      message: expect.stringContaining('artifact request failed: shared.bin'),
    });

    state.ensureError = undefined;
    state.cacheState = 'ready';
    const failedCache = `minimax-music3-${'b'.repeat(64)}`;
    state.deleteError = new Error(`failed to delete artifact cache: ${failedCache}`);
    await expect(
      runWorkerRequest({
        type: 'delete-artifact-caches',
        manifestUrl: 'http://worker.test/music-variable/manifest.json',
      }),
    ).rejects.toMatchObject({
      code: 'cache-delete-failed',
      message: expect.stringContaining(failedCache),
    });
  });

  it('blocks artifact download before cache creation when storage estimate is unavailable', async () => {
    state.cacheState = 'missing';
    state.storageEstimate = undefined;

    await expect(
      runWorkerRequest({
        type: 'download-artifacts',
        manifestUrl: 'http://worker.test/music-variable/manifest.json',
      }),
    ).rejects.toMatchObject({
      code: 'storage-estimate-unavailable',
      operation: 'download-artifacts',
      retryable: true,
    });

    expect(state.events).not.toContain('cache:open');
    expect(state.events.some((event) => event.startsWith('artifact:'))).toBe(false);
  });

  it('blocks artifact download before transfer when storage capacity is insufficient', async () => {
    state.cacheState = 'missing';
    state.storageEstimate = { usage: 90, quota: 100 };

    await expect(
      runWorkerRequest({
        type: 'download-artifacts',
        manifestUrl: 'http://worker.test/music-variable/manifest.json',
      }),
    ).rejects.toMatchObject({
      code: 'quota-insufficient',
      operation: 'download-artifacts',
      retryable: true,
    });

    expect(state.events).not.toContain('cache:open');
    expect(state.events.some((event) => event.startsWith('artifact:'))).toBe(false);
  });

  it('downloads deduplicated artifacts and reports best-effort storage without requesting persistence', async () => {
    state.cacheState = 'missing';
    state.storagePersisted = false;

    await runWorkerRequest({
      type: 'download-artifacts',
      manifestUrl: 'http://worker.test/music-variable/manifest.json',
    });

    const downloads = state.events.filter((event) => event.startsWith('artifact:'));
    expect(downloads).toEqual(expectedVariableArtifactPaths.map((path) => `artifact:${path}`));
    expect(state.cacheOpenCalls).toEqual([{ hash: emptyObjectManifestHash, root: opfsRoot }]);
    expect(state.cacheOpenExistingCalls).toEqual([
      { hash: emptyObjectManifestHash, root: opfsRoot },
      { hash: emptyObjectManifestHash, root: opfsRoot },
    ]);
    expect(state.inspectedArtifactPaths).toEqual([expectedVariableArtifactPaths, expectedVariableArtifactPaths]);
    expect(state.events).not.toContain('storage:persist');
    expect(state.events.filter((event) => event === 'cache:lock:exclusive:start')).toHaveLength(1);
    expect(state.events.indexOf('cache:lock:exclusive:start')).toBeLessThan(state.events.indexOf('cache:open'));
    expect(state.events.indexOf('cache:open')).toBeLessThan(state.events.indexOf('cache:lock:exclusive:end'));
    expect(state.messages.at(-1)?.message).toMatchObject({
      type: 'artifact-download-complete',
      status: { state: 'ready', persistence: 'best-effort' },
    });
  });

  it('preserves persistent storage in the final download status', async () => {
    state.cacheState = 'missing';
    state.storagePersisted = true;

    await runWorkerRequest({
      type: 'download-artifacts',
      manifestUrl: 'http://worker.test/music-variable/manifest.json',
    });

    expect(state.messages.at(-1)?.message).toMatchObject({
      type: 'artifact-download-complete',
      status: { state: 'ready', persistence: 'persistent' },
    });
  });

  it('maps a nested quota failure during artifact download', async () => {
    state.cacheState = 'missing';
    state.ensureError = new Error('artifact write failed: shared.bin at 7 bytes', {
      cause: new DOMException('quota', 'QuotaExceededError'),
    });

    await expect(
      runWorkerRequest({
        type: 'download-artifacts',
        manifestUrl: 'http://worker.test/music-variable/manifest.json',
      }),
    ).rejects.toMatchObject({
      code: 'quota-exceeded',
      operation: 'download-artifacts',
      retryable: true,
      message: expect.stringContaining('shared.bin at 7 bytes'),
    });
  });

  it('deletes project caches and refreshes the active status without recreating it', async () => {
    await runWorkerRequest({
      type: 'delete-artifact-caches',
      manifestUrl: 'http://worker.test/music-variable/manifest.json',
    });

    expect(state.events.filter((event) => event === 'cache:lock:exclusive:start')).toHaveLength(1);
    expect(state.events.indexOf('cache:lock:exclusive:start')).toBeLessThan(state.events.indexOf('cache:delete'));
    expect(state.events.indexOf('cache:delete')).toBeLessThan(state.events.indexOf('cache:lock:exclusive:end'));
    expect(state.events).not.toContain('cache:open');
    expect(state.events.some((event) => event.startsWith('artifact:'))).toBe(false);
    expect(state.messages.at(-1)?.message).toMatchObject({
      type: 'artifact-cache-deleted',
      status: {
        state: 'missing',
        projectCacheCount: 0,
        projectCacheBytes: 0,
      },
    });
  });

  it('rejects a forged raw duration plan before reading the manifest', async () => {
    await expect(
      runWorkerRequest(
        musicRequest({
          promptTokens: 40,
          requestedFrames: 249,
        }),
      ),
    ).rejects.toThrow('Music generation request fields are invalid');

    expect(state.events).toEqual([]);
  });

  it.each(['missing', 'partial'] as const)(
    'rejects product generation when the artifact cache is %s before runtime setup',
    async (cacheState) => {
      state.cacheState = cacheState;

      await expect(runWorkerRequest(musicRequest())).rejects.toMatchObject({
        code: 'cache-not-ready',
        operation: 'generate-music',
        retryable: true,
      });

      expect(state.events.some((event) => event.startsWith('artifact:'))).toBe(false);
      expect(state.events.some((event) => event.startsWith('file:text:'))).toBe(false);
      expect(state.events).not.toContain('adapter:preflight');
    },
  );

  it('serializes structured artifact failures from worker messages', async () => {
    state.cacheState = 'missing';
    const onmessage = (self as unknown as { onmessage(event: MessageEvent<unknown>): void }).onmessage;

    onmessage({ data: musicRequest() } as MessageEvent);

    await vi.waitFor(() =>
      expect(state.messages.at(-1)?.message).toMatchObject({
        type: 'error',
        message: 'Model artifact cache is not ready',
        code: 'cache-not-ready',
        operation: 'generate-music',
        retryable: true,
      }),
    );
  });

  it('preserves actionable download context in serialized worker errors', async () => {
    state.cacheState = 'missing';
    state.ensureError = new Error('artifact write failed: shared.bin at 9 bytes');
    const onmessage = (self as unknown as { onmessage(event: MessageEvent<unknown>): void }).onmessage;

    onmessage({
      data: {
        type: 'download-artifacts',
        manifestUrl: 'http://worker.test/music-variable/manifest.json',
      },
    } as MessageEvent);

    await vi.waitFor(() =>
      expect(state.messages.at(-1)?.message).toMatchObject({
        type: 'error',
        code: 'download-failed',
        message: expect.stringContaining('shared.bin at 9 bytes'),
      }),
    );
  });

  it('rejects unknown and malformed worker requests', async () => {
    await expect(runWorkerRequest(null)).rejects.toThrow('Worker request must be an object');
    await expect(runWorkerRequest({ type: 'unknown' })).rejects.toThrow('Unknown worker request type');
    await expect(
      runWorkerRequest({
        type: 'inspect-artifact-cache',
        manifestUrl: '/manifest.json',
        extra: true,
      }),
    ).rejects.toThrow('Artifact cache request fields are invalid');
  });

  it('rejects a shipped tokenizer that does not reproduce the fixed prompt contract', async () => {
    state.invalidShippedTokenizer = true;

    await expect(runWorkerRequest(musicRequest())).rejects.toThrow(
      'Shipped tokenizer does not match the fixed prompt contract',
    );

    expect(state.events).not.toContain('adapter:preflight');
    expect(state.promptPrepareCalls).toBe(1);
  });

  it('uses and clears the actual ORT-owned device for a legacy smoke route', async () => {
    await runWorkerRequest({ type: 'run-global-smoke', manifestUrl: '/global/manifest.json' });

    expect(state.events.filter((event) => event === 'adapter:preflight')).toHaveLength(1);
    expect(state.events).toContain('device:create:ar');
    expect(state.events.indexOf('session:release:head')).toBeLessThan(state.events.indexOf('device:destroy:ar'));
    expect(state.events.indexOf('session:release:decoder')).toBeLessThan(state.events.indexOf('device:destroy:ar'));
    expect(state.ortWebgpu.device).toBeUndefined();
  });

  it('cleans a legacy device published before its first session rejects', async () => {
    state.publishedCreateFailure = 'decoder';

    await expect(
      runWorkerRequest({
        type: 'run-global-smoke',
        manifestUrl: '/global/manifest.json',
      }),
    ).rejects.toThrow('synthetic published decoder creation failure');

    expect(state.events.filter((event) => event === 'device:destroy:ar')).toHaveLength(1);
    expect(state.ortWebgpu.device).toBeUndefined();
    expect(state.events).not.toContain('session:create:head');
  });

  it('uses four sequential ORT-owned devices for fixed five-second generation', async () => {
    state.retainedPlan = planRetainedFrames({
      retainedFrames: 125,
      promptTokens: 40,
      termination: 'max-frames',
    });

    await runWorkerRequest({
      type: 'generate-music-5s',
      manifestUrl: '/music/manifest.json',
      seed: 7,
    });

    const creates = state.events.filter((event) => event.startsWith('device:create:'));
    const destroys = state.events.filter((event) => event.startsWith('device:destroy:'));
    expect(creates).toHaveLength(4);
    expect(destroys).toHaveLength(4);
    for (let index = 0; index < 3; index++) {
      expect(state.events.indexOf(destroys[index])).toBeLessThan(state.events.indexOf(creates[index + 1]));
    }
    expect(state.ortWebgpu.device).toBeUndefined();
    expect(state.messages.filter(({ message }) => message.type === 'music-result')).toHaveLength(1);
    expect(state.events.some((event) => event.startsWith('artifact:'))).toBe(true);
  });

  it('rejects a concurrent worker message without entering a second ORT lifecycle', async () => {
    let releasePreflight!: () => void;
    state.preflightGate = new Promise<void>((resolve) => {
      releasePreflight = resolve;
    });
    const onmessage = (self as unknown as { onmessage(event: MessageEvent<unknown>): void }).onmessage;
    onmessage({
      data: { type: 'run-global-smoke', manifestUrl: '/global/manifest.json' },
    } as MessageEvent);
    await vi.waitFor(() => expect(state.events).toContain('adapter:preflight'));

    onmessage({
      data: { type: 'run-global-smoke', manifestUrl: '/global/manifest.json' },
    } as MessageEvent);
    await vi.waitFor(() =>
      expect(
        state.messages.some(
          ({ message }) => message.type === 'error' && message.message === 'Worker request already in progress',
        ),
      ).toBe(true),
    );
    expect(state.events.filter((event) => event === 'adapter:preflight')).toHaveLength(1);

    releasePreflight();
    await vi.waitFor(() => expect(state.messages.some(({ message }) => message.type === 'result')).toBe(true));
    expect(state.ortWebgpu.device).toBeUndefined();
  });

  it('holds shared ownership for product generation while an exclusive delete waits', async () => {
    let releasePreflight!: () => void;
    state.preflightGate = new Promise<void>((resolve) => {
      releasePreflight = resolve;
    });
    const generation = runWorkerRequest(musicRequest());
    await vi.waitFor(() => expect(state.events).toContain('adapter:preflight'));

    expect(state.activeCacheLock).toBe('shared');
    const deletion = runWorkerRequest({
      type: 'delete-artifact-caches',
      manifestUrl: 'http://worker.test/music-variable/manifest.json',
    });
    await vi.waitFor(() => expect(state.events).toContain('cache:lock:exclusive:requested'));
    expect(state.events).not.toContain('cache:lock:exclusive:start');

    releasePreflight();
    await generation;
    await deletion;

    expect(state.events.indexOf('cache:lock:shared:end')).toBeLessThan(
      state.events.indexOf('cache:lock:exclusive:start'),
    );
  });

  it('holds exclusive ownership for the full legacy smoke lifecycle', async () => {
    let releasePreflight!: () => void;
    state.preflightGate = new Promise<void>((resolve) => {
      releasePreflight = resolve;
    });
    const generation = runWorkerRequest({
      type: 'run-global-smoke',
      manifestUrl: '/global/manifest.json',
    });
    await vi.waitFor(() => expect(state.events).toContain('adapter:preflight'));

    expect(state.activeCacheLock).toBe('exclusive');
    expect(state.lockAtPreflight).toEqual(['exclusive']);
    expect(state.events.filter((event) => event === 'cache:lock:exclusive:start')).toHaveLength(1);

    releasePreflight();
    await generation;
    expect(state.events.at(-1)).toBe('cache:lock:exclusive:end');
  });

  it('releases AR before one reused acoustic lifecycle and transfers the final WAV once', async () => {
    await runWorkerRequest(musicRequest());

    const flowDevice = state.events.indexOf('device:create:flow');
    expect(state.events.indexOf('session:release:decoder')).toBeLessThan(flowDevice);
    expect(state.events.indexOf('session:release:head')).toBeLessThan(flowDevice);
    expect(state.events.indexOf('session:release:rvq-depth')).toBeLessThan(flowDevice);
    expect(state.events.indexOf('session:release:feedback')).toBeLessThan(flowDevice);
    expect(state.events.indexOf('device:destroy:ar')).toBeLessThan(flowDevice);
    expect(state.events.filter((event) => event === 'session:create:condition')).toHaveLength(1);
    expect(state.events.filter((event) => event === 'session:create:flow')).toHaveLength(1);
    expect(state.events.filter((event) => event === 'session:create:vocoder')).toHaveLength(1);
    const vocoderSession = state.events.indexOf('session:create:vocoder');
    expect(state.events.indexOf('session:release:condition')).toBeLessThan(vocoderSession);
    expect(state.events.indexOf('session:release:flow')).toBeLessThan(vocoderSession);
    expect(state.events.indexOf('device:destroy:flow')).toBeLessThan(vocoderSession);
    expect(state.events.filter((event) => event.startsWith('noise:'))).toEqual([
      `noise:${state.retainedPlan!.chunks.reduce((total, chunk) => total + 128 * chunk.latentLength, 0)}`,
    ]);
    expect(state.events.filter((event) => event.startsWith('flow:chunk:'))).toEqual(['flow:chunk:0', 'flow:chunk:1']);
    expect(state.events.filter((event) => event.startsWith('condition-progress-before-encode:'))).toEqual([
      'condition-progress-before-encode:true',
      'condition-progress-before-encode:true',
    ]);
    expect(state.events.filter((event) => event.startsWith('acoustic-before-complete:'))).toEqual([
      'acoustic-before-complete:false',
      'acoustic-before-complete:false',
    ]);
    expect(state.events.filter((event) => event.startsWith('vocoder:'))).toEqual([
      'vocoder:0:left',
      'vocoder:0:right',
      'vocoder:1:left',
      'vocoder:1:right',
    ]);
    expect(state.events).toContain('wav-progress-before-allocation:true');
    const flowProgress = state.messages.map(({ message }) => message).filter((message) => message.stage === 'flow');
    expect(flowProgress.at(-1)).toMatchObject({ completed: 60, total: 60 });
    const arProgress = state.messages
      .map(({ message }) => message)
      .filter((message) => message.stage === 'autoregressive');
    expect(arProgress.at(-1)).toMatchObject({ completed: 201, total: 250 });
    const acousticProgress = state.messages
      .map(({ message }) => message)
      .filter((message) => message.stage === 'acoustic');
    expect(acousticProgress.map((message) => message.completed)).toEqual([1, 2]);
    const vocoderProgress = state.messages
      .map(({ message }) => message)
      .filter((message) => message.stage === 'vocoder' && message.completed !== undefined);
    expect(vocoderProgress.map((message) => message.completed)).toEqual([1, 2, 3, 4]);
    const results = state.messages.filter(({ message }) => message.type === 'music-result');
    expect(results).toHaveLength(1);
    expect(results[0].transfer).toHaveLength(1);
    expect(state.events.some((event) => event.startsWith('artifact:'))).toBe(false);
    expect(state.cacheOpenExistingCalls).toEqual([{ hash: emptyObjectManifestHash, root: opfsRoot }]);
    expect(results[0].message.result).toMatchObject({
      artifactFetches: 0,
      effectiveInput: {
        prompt: FIXED_COMPARISON_CASE.input.prompt,
        lyrics: FIXED_COMPARISON_CASE.input.lyrics,
        assembledPrompt: FIXED_COMPARISON_CASE.input.assembledPrompt,
        tokenRows: FIXED_COMPARISON_CASE.input.tokenRows,
        promptTokens: 40,
        seed: 7,
        durationSeconds: 10,
        sampling: {
          globalGuidance: 1.5,
          semanticTopK: 50,
          residualTopK: 50,
          temperature: 1,
          flowGuidance: 1.7,
          flowSteps: 30,
        },
      },
      plan: {
        retainedFrames: 201,
        termination: 'natural-end',
        chunkCount: 2,
        chunks: [
          {
            startFrame: 0,
            frameLength: 200,
            latentLength: 689,
            cropLeftLatents: 0,
            cropRightLatents: 258,
          },
          {
            startFrame: 100,
            frameLength: 101,
            latentLength: 347,
            cropLeftLatents: 86,
            cropRightLatents: 0,
          },
        ],
      },
    });
    expect(state.events).toContain('file:text:global/tokenizer/tokenizer.json');
    expect(state.events).toContain('file:text:global/tokenizer/tokenizer_config.json');
    expect(state.events).toContain('tokenizer:create:{"tokenizer":true}:{"config":true}');
    expect(state.events).toContain(
      `prompt:prepare:${FIXED_COMPARISON_CASE.input.prompt}:${FIXED_COMPARISON_CASE.input.lyrics}:250`,
    );
    expect(state.generateFrameOptions).toMatchObject({
      maxFrames: 250,
      seed: 7,
      promptTokenRows: {
        conditional: FIXED_COMPARISON_CASE.input.tokenRows[0],
        unconditional: FIXED_COMPARISON_CASE.input.tokenRows[1],
      },
      guidance: 1.5,
      semanticTopK: 50,
      residualTopK: 50,
      temperature: 1,
    });
    expect(state.flowRequest).toMatchObject({ flowGuidance: 1.7, flowSteps: 30 });
    const timings = results[0].message.result as {
      inferenceMs: Record<'condition' | 'flow' | 'vocoder', number>;
      sessionCreateMs: Record<'condition' | 'flow' | 'vocoder', number>;
      stageMs: Record<'condition' | 'flow' | 'vocoder', number>;
    };
    expect(timings.inferenceMs.condition).toBe(21);
    expect(timings.inferenceMs.flow).toBe(201);
    expect(timings.inferenceMs.vocoder).toBe(10);
    expect(timings.stageMs.condition).toBe(timings.sessionCreateMs.condition + timings.inferenceMs.condition);
    expect(timings.stageMs.flow).toBe(timings.sessionCreateMs.flow + timings.inferenceMs.flow);
    expect(timings.stageMs.vocoder).toBeGreaterThanOrEqual(timings.sessionCreateMs.vocoder);
    expect(state.events.slice(-3)).toEqual([
      'session:release:vocoder',
      'device:destroy:vocoder',
      'cache:lock:shared:end',
    ]);
    expect(state.actualDeviceCount).toBe(3);
    expect(state.events.filter((event) => event === 'adapter:preflight')).toHaveLength(1);
  });

  it('runs the full three-device lifecycle for a max-frame result', async () => {
    state.retainedPlan = planRetainedFrames({
      retainedFrames: 250,
      promptTokens: 40,
      termination: 'max-frames',
    });
    await runWorkerRequest(musicRequest());

    const result = state.messages.find(({ message }) => message.type === 'music-result')!.message.result;
    expect(result).toMatchObject({
      plan: {
        retainedFrames: 250,
        termination: 'max-frames',
        chunkCount: 2,
        chunks: [
          {
            startFrame: 0,
            frameLength: 200,
            latentLength: 689,
            cropLeftLatents: 0,
            cropRightLatents: 258,
          },
          {
            startFrame: 100,
            frameLength: 150,
            latentLength: 516,
            cropLeftLatents: 86,
            cropRightLatents: 0,
          },
        ],
      },
      comparison: {
        seed: 7,
        durationSeconds: 10,
        retainedFrames: 250,
        termination: 'max-frames',
        manifestHash: expect.stringMatching(/^[0-9a-f]{64}$/),
        browser: 'Mozilla/5.0 Chrome/140.0.7339.81 Safari/537.36',
        ortVersion: '1.30.0-dev.20260813-72e1c9c9b8',
        appVersion: '0.1.0-experimental',
        appRevision: expect.stringMatching(/^[0-9a-f]{64}$/),
      },
    });
    expect(state.events.filter((event) => event.startsWith('device:create:'))).toEqual([
      'device:create:ar',
      'device:create:flow',
      'device:create:vocoder',
    ]);
    expect(state.events.filter((event) => event.startsWith('device:destroy:'))).toEqual([
      'device:destroy:ar',
      'device:destroy:flow',
      'device:destroy:vocoder',
    ]);
  });

  it('keeps ordinary variable-duration results free of fixed-case metadata', async () => {
    const sampling = {
      globalGuidance: 2,
      semanticTopK: 64,
      residualTopK: 32,
      temperature: 0.8,
      flowGuidance: 2.2,
      flowSteps: 12,
    };
    await runWorkerRequest(musicRequest({ durationSeconds: 15.02, sampling }));

    const result = state.messages.find(({ message }) => message.type === 'music-result')!.message.result;
    expect(result).not.toHaveProperty('comparison');
    expect(result).not.toHaveProperty('capacityDiagnostic');
    expect(result).toMatchObject({ effectiveInput: { sampling } });
    expect(state.generateFrameOptions).toMatchObject({
      maxFrames: 375,
      guidance: 2,
      semanticTopK: 64,
      residualTopK: 32,
      temperature: 0.8,
    });
    expect(state.flowRequest).toMatchObject({ flowGuidance: 2.2, flowSteps: 12 });
    expect(state.events).toContain(
      `prompt:prepare:${FIXED_COMPARISON_CASE.input.prompt}:${FIXED_COMPARISON_CASE.input.lyrics}:375`,
    );
    expect(state.generateFrameOptions).not.toHaveProperty('audioEndPolicy');
  });

  it('runs the explicit capacity diagnostic with continuation metadata and no comparison receipt', async () => {
    state.retainedPlan = planRetainedFrames({
      retainedFrames: 7_500,
      promptTokens: 40,
      termination: 'max-frames',
    });

    await runWorkerRequest({
      type: 'diagnose-music-capacity',
      diagnostic: 'continue-after-audio-end',
      manifestUrl: 'http://worker.test/music-variable/manifest.json',
      seed: 7,
      durationSeconds: 300,
      promptTokens: 40,
      requestedFrames: 7_500,
    });

    expect(state.generateFrameOptions).toMatchObject({
      maxFrames: 7_500,
      seed: 7,
      promptTokenRows: {
        conditional: FIXED_COMPARISON_CASE.input.tokenRows[0],
        unconditional: FIXED_COMPARISON_CASE.input.tokenRows[1],
      },
      guidance: 1.5,
      semanticTopK: 50,
      residualTopK: 50,
      temperature: 1,
      audioEndPolicy: 'continue-for-capacity-diagnostic',
    });
    expect(state.flowRequest).toMatchObject({ flowGuidance: 1.7, flowSteps: 30 });
    expect(state.events.some((event) => event.startsWith('artifact:'))).toBe(true);
    expect(state.events.some((event) => event.startsWith('file:text:'))).toBe(false);
    const result = state.messages.find(({ message }) => message.type === 'music-result')!.message.result;
    expect(result).toMatchObject({
      plan: { retainedFrames: 7_500, termination: 'max-frames' },
      capacityDiagnostic: {
        kind: 'continue-after-audio-end',
        suppressedAudioEnds: 1,
        firstAudioEndAtRetainedFrame: 1_743,
      },
    });
    expect(result).not.toHaveProperty('comparison');
  });

  it('cleans every AR and acoustic resource when a chunk fails', async () => {
    state.failFlow = true;
    await expect(runWorkerRequest(musicRequest())).rejects.toThrow('synthetic flow failure');

    for (const name of ['decoder', 'head', 'rvq-depth', 'feedback', 'condition', 'flow'])
      expect(state.events.filter((event) => event === `session:release:${name}`)).toHaveLength(1);
    expect(state.events).not.toContain('session:create:vocoder');
    expect(state.events.filter((event) => event === 'device:destroy:ar')).toHaveLength(1);
    expect(state.events.filter((event) => event === 'device:destroy:flow')).toHaveLength(1);
    expect(state.ortWebgpu.device).toBeUndefined();
    expect(state.messages.some(({ message }) => message.type === 'music-result')).toBe(false);
  });

  it('releases an already-created condition session when flow creation fails', async () => {
    state.createFailure = 'flow';
    await expect(runWorkerRequest(musicRequest())).rejects.toThrow('synthetic flow creation failure');

    expect(state.events.filter((event) => event === 'session:release:condition')).toHaveLength(1);
    expect(state.events).not.toContain('session:release:flow');
    expect(state.events).not.toContain('session:create:vocoder');
    expect(state.events.filter((event) => event === 'device:destroy:flow')).toHaveLength(1);
    expect(state.ortWebgpu.device).toBeUndefined();
  });

  it('preserves a flow failure while settling every release and device cleanup', async () => {
    state.failFlow = true;
    state.releaseFailures.push('flow', 'condition');
    await expect(runWorkerRequest(musicRequest())).rejects.toThrow('synthetic flow failure');

    expect(state.events.filter((event) => event === 'session:release:flow')).toHaveLength(1);
    expect(state.events.filter((event) => event === 'session:release:condition')).toHaveLength(1);
    expect(state.events.filter((event) => event === 'device:destroy:flow')).toHaveLength(1);
    expect(state.ortWebgpu.device).toBeUndefined();
  });

  it('preserves partial AR creation failure through release rejection and device cleanup', async () => {
    state.createFailure = 'head';
    state.releaseFailures.push('decoder');
    await expect(runWorkerRequest(musicRequest())).rejects.toThrow('synthetic head creation failure');

    expect(state.events.filter((event) => event === 'session:release:decoder')).toHaveLength(1);
    expect(state.events).not.toContain('device:create:flow');
    expect(state.events.filter((event) => event === 'device:destroy:ar')).toHaveLength(1);
    expect(state.ortWebgpu.device).toBeUndefined();
  });

  it('destroys a device published before the first decoder session rejects', async () => {
    state.publishedCreateFailure = 'decoder';
    await expect(runWorkerRequest(musicRequest())).rejects.toThrow('synthetic published decoder creation failure');

    expect(state.events.filter((event) => event === 'device:destroy:ar')).toHaveLength(1);
    expect(state.ortWebgpu.device).toBeUndefined();
    expect(state.events).not.toContain('session:create:head');
    expect(state.events).not.toContain('device:create:flow');
  });

  it('destroys a device published before the first condition session rejects', async () => {
    state.publishedCreateFailure = 'condition';
    await expect(runWorkerRequest(musicRequest())).rejects.toThrow('synthetic published condition creation failure');

    expect(state.events.filter((event) => event === 'device:destroy:ar')).toHaveLength(1);
    expect(state.events.filter((event) => event === 'device:destroy:flow')).toHaveLength(1);
    expect(state.ortWebgpu.device).toBeUndefined();
    expect(state.events).not.toContain('session:create:flow');
    expect(state.events).not.toContain('session:create:vocoder');
  });

  it('destroys a device published before the first vocoder session rejects', async () => {
    state.publishedCreateFailure = 'vocoder';
    await expect(runWorkerRequest(musicRequest())).rejects.toThrow('synthetic published vocoder creation failure');

    expect(state.events.filter((event) => event === 'device:destroy:ar')).toHaveLength(1);
    expect(state.events.filter((event) => event === 'device:destroy:flow')).toHaveLength(1);
    expect(state.events.filter((event) => event === 'device:destroy:vocoder')).toHaveLength(1);
    expect(state.ortWebgpu.device).toBeUndefined();
    expect(state.messages.some(({ message }) => message.type === 'music-result')).toBe(false);
  });

  it('does not delete a newer ORT device binding while cleaning the captured device', async () => {
    state.failFlow = true;
    state.replaceBindingOnRelease = 'flow';
    await expect(runWorkerRequest(musicRequest())).rejects.toThrow('synthetic flow failure');

    expect(state.events.filter((event) => event === 'device:destroy:flow')).toHaveLength(1);
    expect(state.events).not.toContain('device:destroy:newer');
    await expect(state.ortWebgpu.device).resolves.toMatchObject({ name: 'newer' });
  });
});
