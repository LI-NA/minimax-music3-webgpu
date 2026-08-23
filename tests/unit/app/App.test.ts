import { describe, expect, it } from 'vitest';
import { createProductMusicRequest } from '../../../src/app/DiagnosticsApp';
import {
  artifactCacheNoticeMessage,
  artifactCacheUiReducer,
  artifactErrorMessage,
  artifactDownloadActionLabel,
  createArtifactCacheUiState,
  describeArtifactCacheStatus,
  deriveArtifactCacheControls,
  formatBytes,
  formatEta,
  formatRate,
} from '../../../src/app/artifact-cache-ui';
import { messages, type Language } from '../../../src/app/i18n';
import { persistenceWarningMessage } from '../../../src/app/artifact-cache-ui';
import type { PersistenceWarning } from '../../../src/runtime/model/artifact-cache-management';
import { FIXED_COMPARISON_CASE } from '../../../src/runtime/reference/fixed-comparison';
import type { ArtifactCacheStatus, ArtifactErrorCode } from '../../../src/workers/protocol';

const status = (overrides: Partial<ArtifactCacheStatus> = {}): ArtifactCacheStatus => ({
  manifestHash: 'release',
  state: 'missing',
  artifactCount: 4,
  totalArtifactBytes: 4_096,
  completeArtifactCount: 0,
  completeArtifactBytes: 0,
  storedReferencedBytes: 0,
  additionalBytesNeeded: 4_096,
  largestPendingArtifactBytes: 2_048,
  projectCacheCount: 0,
  projectCacheBytes: 0,
  persistence: 'best-effort',
  availableBytes: 20_000,
  sufficient: true,
  requiredHeadroomBytes: 6_144,
  ...overrides,
});

describe('model file controls', () => {
  it('offers a download only for an idle, incomplete cache with sufficient capacity', () => {
    const state = createArtifactCacheUiState(status());

    expect(deriveArtifactCacheControls(state)).toEqual({
      canDownload: true,
      canRetry: false,
      canRefresh: true,
      canDelete: false,
      canCancel: false,
      canGenerate: false,
    });
  });

  it('enables every action only in its allowed cache state', () => {
    const retryError = {
      message: 'Download failed',
      code: 'download-failed' as const,
      operation: 'download-artifacts' as const,
      retryable: true,
      retryTarget: 'download' as const,
    };
    const cases = [
      [
        createArtifactCacheUiState(status({ state: 'partial', projectCacheCount: 1 })),
        false,
        {
          canDownload: true,
          canRetry: false,
          canRefresh: true,
          canDelete: true,
          canCancel: false,
          canGenerate: false,
        },
      ],
      [
        { ...createArtifactCacheUiState(status({ state: 'partial' })), lastError: retryError },
        false,
        {
          canDownload: false,
          canRetry: true,
          canRefresh: true,
          canDelete: false,
          canCancel: false,
          canGenerate: false,
        },
      ],
      [
        {
          ...createArtifactCacheUiState(status({ state: 'partial' })),
          lastError: { ...retryError, retryable: false },
        },
        false,
        {
          canDownload: false,
          canRetry: false,
          canRefresh: true,
          canDelete: false,
          canCancel: false,
          canGenerate: false,
        },
      ],
      [
        createArtifactCacheUiState(status({ state: 'ready', completeArtifactCount: 4, completeArtifactBytes: 4_096 })),
        false,
        {
          canDownload: false,
          canRetry: false,
          canRefresh: true,
          canDelete: false,
          canCancel: false,
          canGenerate: true,
        },
      ],
      [
        { ...createArtifactCacheUiState(status()), operation: 'download' as const },
        false,
        {
          canDownload: false,
          canRetry: false,
          canRefresh: false,
          canDelete: false,
          canCancel: true,
          canGenerate: false,
        },
      ],
      [
        createArtifactCacheUiState(status({ sufficient: false })),
        false,
        {
          canDownload: false,
          canRetry: false,
          canRefresh: true,
          canDelete: false,
          canCancel: false,
          canGenerate: false,
        },
      ],
      [
        createArtifactCacheUiState(status({ state: 'partial', projectCacheCount: 1 })),
        true,
        {
          canDownload: false,
          canRetry: false,
          canRefresh: true,
          canDelete: false,
          canCancel: false,
          canGenerate: false,
        },
      ],
    ] as const;

    for (const [state, musicRunning, expected] of cases) {
      expect(deriveArtifactCacheControls(state, musicRunning)).toEqual(expected);
    }
  });

  it('provides the exact download action labels', () => {
    expect(artifactDownloadActionLabel(createArtifactCacheUiState(status()))).toBe('Download Model');
    expect(artifactDownloadActionLabel(createArtifactCacheUiState(status({ state: 'partial' })))).toBe(
      'Resume Download',
    );
    expect(
      artifactDownloadActionLabel({
        ...createArtifactCacheUiState(status({ state: 'partial' })),
        lastError: {
          message: 'Download failed',
          operation: 'download-artifacts',
          retryable: true,
          retryTarget: 'download',
        },
      }),
    ).toBe('Retry Download');
  });
});

describe('model file state transitions', () => {
  it('keeps a persistence warning while proceeding to a download', () => {
    const requesting = artifactCacheUiReducer(createArtifactCacheUiState(status()), {
      type: 'operation-started',
      operation: 'request-persistence',
    });
    const warned = artifactCacheUiReducer(requesting, {
      type: 'persistence-resolved',
      warning: 'denied',
    });
    const downloading = artifactCacheUiReducer(warned, { type: 'download-started' });

    expect(downloading).toMatchObject({
      operation: 'download',
      persistenceWarning: 'denied',
      lastError: null,
    });
  });

  it('adopts authoritative status and records artifact progress separately', () => {
    const downloading = artifactCacheUiReducer(createArtifactCacheUiState(status()), {
      type: 'download-started',
    });
    const progressed = artifactCacheUiReducer(downloading, {
      type: 'progress-received',
      progress: {
        type: 'progress',
        stage: 'artifact',
        detail: 'weights/model.onnx',
        currentFile: 'weights/model.onnx',
        completedBytes: 1_024,
        totalBytes: 4_096,
        rate: 512,
        etaMs: 6_000,
      },
    });
    const ready = status({
      state: 'ready',
      completeArtifactCount: 4,
      completeArtifactBytes: 4_096,
      storedReferencedBytes: 4_096,
      additionalBytesNeeded: 0,
      largestPendingArtifactBytes: 0,
    });
    const completed = artifactCacheUiReducer(progressed, {
      type: 'status-received',
      source: 'download',
      status: ready,
    });

    expect(progressed.downloadProgress).toEqual({
      currentFile: 'weights/model.onnx',
      completedBytes: 1_024,
      totalBytes: 4_096,
      rate: 512,
      etaMs: 6_000,
    });
    expect(completed).toMatchObject({ status: ready, operation: null, downloadProgress: null });
  });

  it('keeps the last known rate and ETA when a later report carries neither', () => {
    const downloading = artifactCacheUiReducer(createArtifactCacheUiState(status()), {
      type: 'download-started',
    });
    const measured = artifactCacheUiReducer(downloading, {
      type: 'progress-received',
      progress: {
        type: 'progress',
        stage: 'artifact',
        detail: 'weights/a.onnx',
        currentFile: 'weights/a.onnx',
        completedBytes: 1_024,
        totalBytes: 4_096,
        rate: 512,
        etaMs: 6_000,
      },
    });
    const unmeasured = artifactCacheUiReducer(measured, {
      type: 'progress-received',
      progress: {
        type: 'progress',
        stage: 'artifact',
        detail: 'weights/b.onnx',
        currentFile: 'weights/b.onnx',
        completedBytes: 2_048,
        totalBytes: 4_096,
      },
    });

    expect(unmeasured.downloadProgress).toEqual({
      currentFile: 'weights/b.onnx',
      completedBytes: 2_048,
      totalBytes: 4_096,
      rate: 512,
      etaMs: 6_000,
    });
  });

  it('retains a retryable download error after the immediate refresh', () => {
    const error = {
      message: 'Network interrupted',
      code: 'download-failed' as const,
      operation: 'download-artifacts' as const,
      retryable: true,
      retryTarget: 'download' as const,
    };
    const failed = artifactCacheUiReducer(
      { ...createArtifactCacheUiState(status()), operation: 'download' },
      { type: 'operation-failed', error },
    );
    const inspecting = artifactCacheUiReducer(failed, {
      type: 'operation-started',
      operation: 'inspect',
    });
    const refreshed = artifactCacheUiReducer(inspecting, {
      type: 'status-received',
      source: 'inspect',
      status: status({ state: 'partial', storedReferencedBytes: 512 }),
    });

    expect(refreshed.lastError).toEqual(error);
    expect(deriveArtifactCacheControls(refreshed).canRetry).toBe(true);
  });

  it('retains the cancellation notice after refreshing retained partial files', () => {
    const cancelled = artifactCacheUiReducer(
      { ...createArtifactCacheUiState(status()), operation: 'download' },
      { type: 'download-cancelled' },
    );
    const inspecting = artifactCacheUiReducer(cancelled, {
      type: 'operation-started',
      operation: 'inspect',
    });
    const refreshed = artifactCacheUiReducer(inspecting, {
      type: 'status-received',
      source: 'inspect',
      status: status({ state: 'partial', storedReferencedBytes: 512 }),
    });

    expect(refreshed.notice).toBe('download-cancelled');
    expect(refreshed.status?.state).toBe('partial');
  });

  it('preserves mutation errors and cancellation when the follow-up inspection fails', () => {
    const inspectError = {
      message: 'Inspection worker failed',
      operation: 'inspect-artifact-cache' as const,
      retryable: true,
      retryTarget: 'inspect' as const,
    };
    const interruption = (retryTarget: 'download' | 'delete') => ({
      message: `${retryTarget} failed`,
      operation: retryTarget === 'download' ? ('download-artifacts' as const) : ('delete-artifact-caches' as const),
      retryable: true,
      retryTarget,
    });
    for (const retryTarget of ['download', 'delete'] as const) {
      const error = interruption(retryTarget);
      const failed = artifactCacheUiReducer(createArtifactCacheUiState(status()), {
        type: 'operation-failed',
        error,
      });
      const inspecting = artifactCacheUiReducer(failed, {
        type: 'operation-started',
        operation: 'inspect',
      });

      expect(
        artifactCacheUiReducer(inspecting, {
          type: 'operation-failed',
          error: inspectError,
        }),
      ).toMatchObject({ operation: null, lastError: error, notice: null });
    }

    const cancelled = artifactCacheUiReducer(createArtifactCacheUiState(status()), {
      type: 'download-cancelled',
    });
    const inspecting = artifactCacheUiReducer(cancelled, {
      type: 'operation-started',
      operation: 'inspect',
    });
    expect(
      artifactCacheUiReducer(inspecting, {
        type: 'operation-failed',
        error: inspectError,
      }),
    ).toMatchObject({
      operation: null,
      lastError: null,
      notice: 'download-cancelled',
    });
  });
});

describe('model file presentation', () => {
  it('shows missing verified bytes, sufficient headroom, and the best-effort warning', () => {
    expect(describeArtifactCacheStatus(createArtifactCacheUiState(status()))).toBe(
      'Model files are not downloaded (0 B of 4.0 KiB verified). ' +
        'Storage capacity is sufficient (19.5 KiB available, 6.0 KiB required headroom). ' +
        'Storage is best-effort and may be evicted.',
    );
  });

  it('shows available versus required headroom when capacity is insufficient', () => {
    expect(
      describeArtifactCacheStatus(
        createArtifactCacheUiState(
          status({
            state: 'partial',
            completeArtifactBytes: 1_024,
            availableBytes: 5_120,
            sufficient: false,
          }),
        ),
      ),
    ).toBe(
      'Model files are partially downloaded (1.0 KiB of 4.0 KiB verified). ' +
        'Storage capacity is insufficient (5.0 KiB available, 6.0 KiB required headroom). ' +
        'Storage is best-effort and may be evicted.',
    );
  });

  it('shows required headroom and persistence when the estimate is unavailable', () => {
    expect(
      describeArtifactCacheStatus(
        createArtifactCacheUiState(
          status({
            state: 'partial',
            completeArtifactBytes: 1_024,
            availableBytes: undefined,
            sufficient: undefined,
            persistence: 'persistent',
          }),
        ),
      ),
    ).toBe(
      'Model files are partially downloaded (1.0 KiB of 4.0 KiB verified). ' +
        'Available storage is unavailable (6.0 KiB required headroom). ' +
        'Storage is persistent.',
    );
  });

  it('shows a ready cache and unavailable persistence status', () => {
    expect(
      describeArtifactCacheStatus(
        createArtifactCacheUiState(
          status({
            state: 'ready',
            completeArtifactCount: 4,
            completeArtifactBytes: 4_096,
            additionalBytesNeeded: 0,
            largestPendingArtifactBytes: 0,
            requiredHeadroomBytes: 0,
            persistence: 'unavailable',
          }),
        ),
      ),
    ).toBe(
      'Model files are ready (4.0 KiB of 4.0 KiB verified). ' +
        'Storage capacity is sufficient (19.5 KiB available, 0 B required headroom). ' +
        'Persistence status is unavailable.',
    );
  });

  it('formats byte counts, transfer rates, and ETA values compactly', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(0.5)).toBe('1 B');
    expect(formatBytes(-1)).toBe('0 B');
    expect(formatBytes(Number.NaN)).toBe('0 B');
    expect(formatBytes(Number.POSITIVE_INFINITY)).toBe('0 B');
    expect(formatBytes(1_536)).toBe('1.5 KiB');
    expect(formatRate(1_048_576)).toBe('1.0 MiB/s');
    expect(formatEta(6_000)).toBe('6s');
    expect(formatEta(65_000)).toBe('1m 5s');
  });
});

describe('music generation UI request', () => {
  it('uses the variable music route for the five-second product duration', () => {
    expect(createProductMusicRequest(5)).toEqual({
      type: 'generate-music',
      manifestUrl: '/artifacts/music-variable/manifest.json',
      prompt: FIXED_COMPARISON_CASE.input.prompt,
      lyrics: FIXED_COMPARISON_CASE.input.lyrics,
      seed: 7,
      durationSeconds: 5,
      sampling: {
        globalGuidance: 1.5,
        semanticTopK: 50,
        residualTopK: 50,
        temperature: 1,
        flowGuidance: 1.7,
        flowSteps: 30,
      },
    });
  });
});

describe('model file wording', () => {
  const languages: Language[] = ['ko', 'en'];
  const codes: ArtifactErrorCode[] = [
    'manifest-unavailable',
    'manifest-invalid',
    'storage-estimate-unavailable',
    'quota-insufficient',
    'cache-not-ready',
    'download-failed',
    'quota-exceeded',
    'cache-inspection-failed',
    'cache-delete-failed',
  ];
  const warnings: PersistenceWarning[] = ['unsupported', 'denied', 'failed'];

  it.each(languages)('words every failure code distinctly in %s', (language) => {
    const tr = messages[language];
    const worded = codes.map((code) => artifactErrorMessage(tr, code, tr.errWorker));

    expect(worded.every((text) => text.length > 0)).toBe(true);
    expect(new Set(worded).size).toBe(codes.length);
    expect(worded).not.toContain(tr.errWorker);
  });

  it.each(languages)('falls back to a worded failure when the worker sent no code in %s', (language) => {
    const tr = messages[language];

    expect(artifactErrorMessage(tr, undefined, tr.errWorker)).toBe(tr.errWorker);
    expect(artifactErrorMessage(tr, undefined, tr.errGeneration)).toBe(tr.errGeneration);
  });

  it.each(languages)('words every persistence warning and notice in %s', (language) => {
    const tr = messages[language];
    const worded = warnings.map((warning) => persistenceWarningMessage(tr, warning));

    expect(new Set(worded).size).toBe(warnings.length);
    expect(artifactCacheNoticeMessage(tr, 'download-cancelled')).toBe(tr.dlCancelled);
  });

  it('keeps the Korean catalog free of the English wording', () => {
    const shared = Object.entries(messages.ko).filter(
      ([key, value]) => typeof value === 'string' && value === messages.en[key as keyof typeof messages.en],
    );

    expect(shared.map(([key]) => key)).toEqual([
      'fineTab',
      'rawTab',
      'styleTag',
      'vocalTag',
      'arrTag',
      'lyricsTitle',
      'instrumental',
      'instBadge',
      'seedTitle',
      'inst',
      'dlUnknown',
      'flowStepsLabel',
      'kSeed',
    ]);
  });
});
