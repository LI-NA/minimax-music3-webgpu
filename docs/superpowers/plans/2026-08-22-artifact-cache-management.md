# Artifact Cache Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, resumable browser model-download lifecycle with cache inspection, advisory storage admission, progress, retry, project-only deletion, and a hard product-generation readiness guard.

**Architecture:** Keep artifact bytes in the existing manifest-scoped OPFS stores and reuse `ensureArtifact` for Range resume, hashing, receipts, and recoverable partial files. Add pure inspection and capacity functions around that store, expose three strict cache operations through the existing inference worker, and add a small cache controller to the current diagnostic App. Page load only inspects; only a user click requests persistent storage and starts the download.

**Tech Stack:** TypeScript, React 19, Dedicated Workers, OPFS, StorageManager, Web Locks, Vitest, Playwright, Vite.

---

## File map

- Create `src/runtime/model/artifact-cache-management.ts`: pure artifact-state arithmetic, storage estimate assessment, project-cache scan/delete, persistence helper, and the exclusive mutation lock.
- Modify `src/runtime/model/artifact-cache.ts`: non-creating `openExisting` and transferred-byte progress.
- Create `src/app/artifact-cache-ui.ts`: pure cache UI reducer, labels, action availability, and byte/rate formatting.
- Modify `src/workers/protocol.ts`: strict cache request, status, response, and structured error contracts.
- Modify `src/workers/artifact-progress.ts`: aggregate download rate and ETA.
- Modify `src/workers/inference.worker.ts`: manifest fetch without cache creation, cache operations, structured errors, and product readiness guard.
- Modify `src/app/App.tsx`: page-load inspection and explicit download, retry, refresh, cancel, and confirmed deletion controls.
- Modify `src/app/styles.css`: compact model-files section styling only.
- Add focused tests under `tests/unit/model`, `tests/unit/workers`, and `tests/unit/app`.
- Add `tests/browser/artifact-cache.spec.ts`: isolated tiny OPFS resume and deletion fixture, never the real release.
- Update `docs/development/variable-duration-generation.md`: user-visible cache lifecycle and advisory quota semantics.

### Task 1: Cache inspection and storage admission

**Files:**
- Create: `src/runtime/model/artifact-cache-management.ts`
- Create: `tests/unit/model/artifact-cache-management.test.ts`

- [ ] **Step 1: Write failing inspection and capacity tests**

Cover these exact cases with an injected `Pick<ArtifactStore, 'size' | 'isComplete'>`:

```ts
expect(await inspectArtifactCache(files, undefined)).toMatchObject({
  state: 'missing', artifactCount: 3, completeArtifactCount: 0,
  storedReferencedBytes: 0, additionalBytesNeeded: 60,
  largestPendingArtifactBytes: 30,
});

expect(await inspectArtifactCache(files, mixedStore)).toEqual({
  state: 'partial', artifactCount: 5, totalArtifactBytes: 150,
  completeArtifactCount: 1, completeArtifactBytes: 10,
  storedReferencedBytes: 125, additionalBytesNeeded: 65,
  largestPendingArtifactBytes: 50,
});

expect(assessArtifactCapacity(partial, { usage: 100, quota: 215 })).toMatchObject({
  availableBytes: 115, requiredHeadroomBytes: 115, sufficient: true,
});
expect(assessArtifactCapacity(partial, { usage: 101, quota: 215 }).sufficient).toBe(false);
expect(assessArtifactCapacity(partial, undefined).sufficient).toBeUndefined();
expect(assessArtifactCapacity(ready, undefined).requiredHeadroomBytes).toBe(0);
```

Also assert exact-size files without receipts are pending, oversized files contribute their actual stored bytes but zero net additional bytes, and malformed estimate values are treated as unavailable.

- [ ] **Step 2: Run the focused test and observe RED**

Run: `npx vitest run tests/unit/model/artifact-cache-management.test.ts`

Expected: FAIL because `artifact-cache-management` does not exist.

- [ ] **Step 3: Implement the pure contracts**

Export these exact public types and functions:

```ts
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
  requiredHeadroomBytes: number;
  sufficient?: boolean;
}

export async function inspectArtifactCache(
  artifacts: readonly ArtifactFile[],
  store: Pick<ArtifactStore, 'size' | 'isComplete'> | undefined,
): Promise<ArtifactCacheInspection>;

export function assessArtifactCapacity(
  inspection: ArtifactCacheInspection,
  estimate?: Pick<StorageEstimate, 'usage' | 'quota'>,
): ArtifactCapacityAssessment;
```

For each artifact, ready means exact expected size plus a completion receipt. Sum actual stored sizes for `storedReferencedBytes`. Sum `max(expected - min(stored, expected), 0)` for additional bytes. The largest pending value is the greatest expected size among non-ready artifacts. A missing store is `missing`; an existing but non-ready store is `partial`; every-ready is `ready`. Required headroom is zero when ready, otherwise additional bytes plus largest pending artifact.

- [ ] **Step 4: Run focused tests and commit**

Run: `npx vitest run tests/unit/model/artifact-cache-management.test.ts`

Expected: PASS.

Commit:

```powershell
git add -- src/runtime/model/artifact-cache-management.ts tests/unit/model/artifact-cache-management.test.ts
git diff --cached --check
git commit -m "feat(cache): inspect artifact storage"
```

### Task 2: OPFS cache discovery, deletion, persistence, and locking

**Files:**
- Modify: `src/runtime/model/artifact-cache.ts`
- Modify: `src/runtime/model/artifact-cache-management.ts`
- Modify: `tests/unit/model/artifact-cache.test.ts`
- Modify: `tests/unit/model/artifact-cache-management.test.ts`

- [ ] **Step 1: Add RED tests for non-creating open and strict deletion scope**

Test that `OpfsArtifactStore.openExisting(hash, root)` calls `getDirectoryHandle(name)` without `{ create: true }`, returns `undefined` only for `NotFoundError`, and propagates other failures.

Build a small fake directory tree and assert:

```ts
expect(await inspectProjectArtifactCaches(root)).toEqual({
  cacheCount: 2,
  storedBytes: 14,
});
await deleteProjectArtifactCaches(root);
expect(removed).toEqual([
  ['minimax-music3-' + 'a'.repeat(64), { recursive: true }],
  ['minimax-music3-' + 'b'.repeat(64), { recursive: true }],
]);
expect(unrelatedNames).toEqual(expect.arrayContaining([
  'notes', 'minimax-music3-short', 'minimax-music3-' + 'A'.repeat(64),
]));
```

Also cover recursive byte totals including receipt files, exact-name files being ignored, idempotent `NotFoundError`, and a non-NotFound deletion failure naming the failed cache.

Add lock tests that require the exact name `minimax-music3-artifact-cache`, `{ mode: 'exclusive' }`, one action invocation, returned value propagation, and fail-closed behavior when Web Locks are unavailable.

Add persistence tests for already persistent, granted, denied, unsupported, and rejected `persist()` calls. Denied and rejected states must be represented as best effort, not as capacity failures.

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```powershell
npx vitest run tests/unit/model/artifact-cache.test.ts tests/unit/model/artifact-cache-management.test.ts
```

Expected: FAIL for missing APIs.

- [ ] **Step 3: Implement the OPFS and storage helpers**

Add:

```ts
static async openExisting(
  manifestHash: string,
  opfsRoot?: FileSystemDirectoryHandle,
): Promise<OpfsArtifactStore | undefined>;

export interface ProjectCacheUsage {
  cacheCount: number;
  storedBytes: number;
}

export async function inspectProjectArtifactCaches(
  root: FileSystemDirectoryHandle,
): Promise<ProjectCacheUsage>;

export async function deleteProjectArtifactCaches(
  root: FileSystemDirectoryHandle,
): Promise<void>;

export async function withArtifactCacheMutationLock<T>(
  operation: () => Promise<T>,
  locks?: Pick<LockManager, 'request'>,
): Promise<T>;

export type PersistenceRequestResult =
  | { state: 'persistent'; warning?: never }
  | { state: 'best-effort'; warning: string };

export async function requestPersistentStorage(
  storage?: Pick<StorageManager, 'persisted' | 'persist'>,
): Promise<PersistenceRequestResult>;
```

Only direct root directories matching `^minimax-music3-[a-f0-9]{64}$` count. Recursively read file sizes but never file contents. Delete matching directories with `removeEntry(name, { recursive: true })`. Do not add rollback or retry. `withArtifactCacheMutationLock` must fail before calling the action if no lock manager is available.

Extend `ProgressSink` with `transferred: number`. Increment it only after successful network chunks, across the existing one hash-retry attempt. Existing destructuring callers remain compatible.

- [ ] **Step 4: Run focused tests and commit**

Run:

```powershell
npx vitest run tests/unit/model/artifact-cache.test.ts tests/unit/model/artifact-cache-management.test.ts
npx eslint src/runtime/model/artifact-cache.ts src/runtime/model/artifact-cache-management.ts tests/unit/model/artifact-cache.test.ts tests/unit/model/artifact-cache-management.test.ts
```

Expected: PASS.

Commit:

```powershell
git add -- src/runtime/model/artifact-cache.ts src/runtime/model/artifact-cache-management.ts tests/unit/model/artifact-cache.test.ts tests/unit/model/artifact-cache-management.test.ts
git diff --cached --check
git commit -m "feat(cache): manage OPFS model caches"
```

### Task 3: Cache protocol and download telemetry

**Files:**
- Modify: `src/workers/protocol.ts`
- Modify: `src/workers/artifact-progress.ts`
- Modify: `tests/unit/workers/protocol.test.ts`
- Modify: `tests/unit/workers/artifact-progress.test.ts`

- [ ] **Step 1: Write RED protocol and controlled-clock progress tests**

Test exact acceptance of these requests and rejection of missing, empty, or extra fields:

```ts
{ type: 'inspect-artifact-cache', manifestUrl: 'https://example.test/manifest.json' }
{ type: 'download-artifacts', manifestUrl: 'https://example.test/manifest.json' }
{ type: 'delete-artifact-caches', manifestUrl: 'https://example.test/manifest.json' }
```

For telemetry, report 10 transferred bytes at 0 ms, then 60 aggregate transferred bytes and 50 completed bytes at 1000 ms. Assert `rate: 50` and `etaMs: 1000` for a 100-byte total. Assert rate and ETA are absent at the first sample and on cache-hit-only completion.

- [ ] **Step 2: Run focused tests and observe RED**

Run:

```powershell
npx vitest run tests/unit/workers/protocol.test.ts tests/unit/workers/artifact-progress.test.ts
```

Expected: FAIL for missing cache protocol and telemetry fields.

- [ ] **Step 3: Add exact protocol contracts**

Add `ArtifactCacheRequest` to `WorkerRequest`, strict `validateArtifactCacheRequest(raw)`, `ArtifactCacheStatus`, the three success responses, and optional fields on the existing error response.

```ts
export type ArtifactOperation =
  | 'inspect-artifact-cache'
  | 'download-artifacts'
  | 'delete-artifact-caches'
  | 'generate-music';

export type ArtifactErrorCode =
  | 'manifest-unavailable'
  | 'manifest-invalid'
  | 'storage-estimate-unavailable'
  | 'quota-insufficient'
  | 'cache-not-ready'
  | 'download-failed'
  | 'quota-exceeded'
  | 'cache-inspection-failed'
  | 'cache-delete-failed';
```

`ArtifactCacheStatus` must contain the inspection and capacity fields from Task 1 plus `manifestHash`, project cache count/bytes, and persistence state `persistent | best-effort | unavailable`.

Update the reporter to accept the aggregate current-operation transferred bytes. After a positive elapsed interval, emit bytes per second and ETA from remaining verified bytes. Never report non-finite values.

- [ ] **Step 4: Run focused tests and commit**

Run:

```powershell
npx vitest run tests/unit/workers/protocol.test.ts tests/unit/workers/artifact-progress.test.ts
npx eslint src/workers/protocol.ts src/workers/artifact-progress.ts tests/unit/workers/protocol.test.ts tests/unit/workers/artifact-progress.test.ts
```

Expected: PASS.

Commit:

```powershell
git add -- src/workers/protocol.ts src/workers/artifact-progress.ts tests/unit/workers/protocol.test.ts tests/unit/workers/artifact-progress.test.ts
git diff --cached --check
git commit -m "feat(cache): report model download state"
```

### Task 4: Worker cache operations and product readiness guard

**Files:**
- Modify: `src/workers/inference.worker.ts`
- Modify: `tests/unit/workers/inference-worker.test.ts`

- [ ] **Step 1: Expand worker mocks and add RED lifecycle tests**

Make the existing mocked artifact store default to an exact ready inspection so current product lifecycle tests preserve their behavior. Add explicit tests for:

- inspect sends `artifact-cache-status`, never calls `OpfsArtifactStore.open`, and never calls `ensureArtifact`;
- unavailable or insufficient estimates block cache creation and network transfer with structured errors;
- download acquires one exclusive lock, caches the deduplicated variable artifact list, and sends a ready `artifact-download-complete`;
- a nested `QuotaExceededError` becomes `quota-exceeded`;
- delete removes project caches under the same exclusive lock and sends refreshed missing status;
- a missing or partial product cache throws `cache-not-ready` before tokenizer reads, adapter preflight, or artifact fetching;
- ready product generation performs zero artifact fetches;
- capacity diagnostic and fixed five-second routes retain their existing implicit downloads; and
- the `self.onmessage` catch preserves `code`, `operation`, and `retryable`.

- [ ] **Step 2: Run the worker tests and observe RED**

Run: `npx vitest run tests/unit/workers/inference-worker.test.ts`

Expected: FAIL because cache operations are not routed and product generation still downloads implicitly.

- [ ] **Step 3: Separate manifest fetch from cache creation**

Introduce a local `fetchManifest()` returning only `{ text, hash, base }`. Keep `readManifest()` for legacy routes by calling `fetchManifest()` and then `OpfsArtifactStore.open(hash)`. Add `collectVariableMusicArtifacts(manifest)` and use exactly the same deduplicated list for inspect, download, and product generation.

Implement a local `ArtifactOperationError` and a serializer that walks `Error.cause` to detect `QuotaExceededError`. Manifest fetch and parse must map separately to `manifest-unavailable` and `manifest-invalid`.

- [ ] **Step 4: Implement the three cache requests**

At the start of `runWorkerRequest`, strictly validate and route cache requests. Inspection must use `OpfsArtifactStore.openExisting` and must not create a directory. Build status from `inspectArtifactCache`, `inspectProjectArtifactCaches`, `navigator.storage.persisted()`, and `navigator.storage.estimate()`.

Download must refresh status under one exclusive mutation lock, reject unavailable or insufficient capacity before `OpfsArtifactStore.open`, run the existing sequential `cacheArtifacts`, re-inspect, require `ready`, and send `artifact-download-complete`.

Delete must validate the active manifest first, delete project caches under one exclusive mutation lock, re-inspect without creating, and send `artifact-cache-deleted`.

- [ ] **Step 5: Remove implicit product download only**

For `generate-music`, fetch and parse the variable release, call `openExisting`, inspect the exact artifact set, and throw `cache-not-ready` unless ready. Set `artifactFetches` to zero and continue through the existing tokenizer and three-device pipeline. Preserve the old create-and-download path for `diagnose-music-capacity`, `generate-music-5s`, and smoke routes.

- [ ] **Step 6: Run focused tests and commit**

Run:

```powershell
npx vitest run tests/unit/workers/inference-worker.test.ts tests/unit/workers/protocol.test.ts tests/unit/workers/artifact-progress.test.ts
npm run typecheck
npm run lint
```

Expected: PASS.

Commit:

```powershell
git add -- src/workers/inference.worker.ts tests/unit/workers/inference-worker.test.ts
git diff --cached --check
git commit -m "feat(cache): prepare model artifacts explicitly"
```

### Task 5: Minimal App cache lifecycle

**Files:**
- Create: `src/app/artifact-cache-ui.ts`
- Modify: `src/app/App.tsx`
- Modify: `src/app/styles.css`
- Modify: `tests/unit/app/App.test.ts`

- [ ] **Step 1: Write RED pure-controller tests**

Without adding jsdom or Testing Library, test an exported reducer and derived controls:

```ts
expect(cacheControls({ status: missingSufficient, operation: null, error: null })).toMatchObject({
  canDownload: true, canRetry: false, canRefresh: true, canDelete: false, canGenerate: false,
});
expect(cacheControls({ status: ready, operation: null, error: null }).canGenerate).toBe(true);
expect(cacheControls({ status: partial, operation: 'download', error: null })).toMatchObject({
  canCancel: true, canRefresh: false, canDelete: false,
});
expect(cacheReducer(state, { type: 'download-error', error: retryable })).toMatchObject({
  operation: null, error: retryable,
});
```

Test human-readable status text for missing, partial, ready, unavailable estimate, insufficient capacity, persistent, and best-effort states. Retain the existing product request test.

- [ ] **Step 2: Run App tests and observe RED**

Run: `npx vitest run tests/unit/app/App.test.ts`

Expected: FAIL because the cache UI module does not exist.

- [ ] **Step 3: Implement the small pure cache UI module**

Export `CacheUiState`, `CacheUiAction`, `cacheReducer`, `cacheControls`, `formatBytes`, `formatRate`, and `formatEta`. Keep the reducer limited to inspect, persistence, download, delete, terminal status, error, and cancellation transitions. The authoritative readiness always comes from `ArtifactCacheStatus.state`.

- [ ] **Step 4: Wire the current App without redesigning it**

Use a separate `cacheWorker` ref from the inference worker. On mount, send `inspect-artifact-cache`. Never send `download-artifacts` automatically.

Add one `Model files` section before the diagnostic command row. It must include:

- active cache state and verified bytes;
- available versus required advisory storage;
- persistent or best-effort status and warning;
- Download Model or Resume Download;
- Retry Download only after retryable download failure;
- Refresh Status;
- Remove Cached Model after `window.confirm`;
- Cancel Download only while downloading; and
- a separately labelled download progress bar with current file, bytes, rate, and ETA.

On Download click, call the Window-only `requestPersistentStorage(navigator.storage)` first. Continue on denial with a warning. Then send `download-artifacts`. On download failure or cancellation, terminate only the cache worker and immediately inspect again so retained partial bytes are visible. On successful terminal responses, terminate the cache worker and adopt the returned status.

Disable only the product Generate button unless WebGPU is supported, the active cache is ready, and no cache mutation is active. Keep diagnostic buttons and their existing behavior unchanged. The worker readiness guard remains authoritative.

- [ ] **Step 5: Run focused tests and commit**

Run:

```powershell
npx vitest run tests/unit/app/App.test.ts
npx eslint src/app/App.tsx src/app/artifact-cache-ui.ts src/app/styles.css tests/unit/app/App.test.ts
npm run typecheck
```

Expected: PASS. If ESLint does not accept CSS paths, run it only on the TypeScript paths and record that exact command.

Commit:

```powershell
git add -- src/app/App.tsx src/app/artifact-cache-ui.ts src/app/styles.css tests/unit/app/App.test.ts
git diff --cached --check
git commit -m "feat(ui): manage model downloads"
```

### Task 6: Isolated browser OPFS regression

**Files:**
- Create: `tests/browser/artifact-cache.spec.ts`

- [ ] **Step 1: Add the tiny browser fixture gate**

Use `page.evaluate()` to dynamically import the cache modules. Create one unique test cache with a deterministic unused 64-character lowercase hash. Use a tiny eight-byte artifact and an injected first response stream that writes three bytes then fails. Retry with a `206` response, assert the request is `Range: bytes=3-`, and assert final inspection is `ready`.

Create one unrelated root directory and one near-match cache name. Invoke project-cache deletion and assert only exact project-cache names are removed. Clean the exact test entries in `finally`. Do not open, enumerate, or remove the real release path from the repository filesystem.

- [ ] **Step 2: Collect and run the isolated gate**

Run:

```powershell
npx playwright test tests/browser/artifact-cache.spec.ts --project=chrome --list
npx playwright test tests/browser/artifact-cache.spec.ts --project=chrome --workers=1
```

Expected: one collected test and one PASS. This gate transfers only fixture bytes and does not run WebGPU inference.

- [ ] **Step 3: Commit**

```powershell
git add -- tests/browser/artifact-cache.spec.ts
git diff --cached --check
git commit -m "test(browser): verify resumable artifact cache"
```

### Task 7: Documentation and final verification

**Files:**
- Modify: `docs/development/variable-duration-generation.md`

- [ ] **Step 1: Document the exact user and platform contract**

State that page entry inspects only, Download Model is explicit, persistence is best effort unless granted, capacity is advisory rather than reserved, Retry resumes partial files, Remove Cached Model deletes only MiniMax Music 3 manifest caches, and product generation requires a ready active cache. Include the standard Storage and File System links from the design spec.

- [ ] **Step 2: Run one fresh complete verification pass**

Run:

```powershell
npm test
npm run lint
npm run typecheck
npm run build
uv run pytest -q tests/python
git diff --check
$unfinished = 'TO' + 'DO|FIX' + 'ME'
rg -n "$([char]0x2014)|$unfinished" src tests docs/development/variable-duration-generation.md
git status --short
```

Expected: all project checks pass, the scan has no prohibited em dash or unfinished marker in changed files, and status contains only the intended documentation change before its commit.

- [ ] **Step 3: Review artifact safety**

Record the active release manifest hash and size with read-only commands before and after the browser fixture. Confirm no tracked or ignored production artifact was removed or rewritten. Do not hash all multi-gigabyte shards again unless metadata indicates a change.

- [ ] **Step 4: Commit documentation**

```powershell
git add -- docs/development/variable-duration-generation.md
git diff --cached --check
git commit -m "docs: explain browser model storage"
```

- [ ] **Step 5: Request code review and verify clean handoff**

Use `superpowers:requesting-code-review` for a scoped review of cache correctness, quota semantics, destructive scope, protocol compatibility, and artifact safety. Address only findings that match the approved requirements, rerun the affected focused tests once, then confirm `git status --short` is clean.
