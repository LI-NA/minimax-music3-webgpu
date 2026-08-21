# Artifact Cache Management Design

## Status

Approved for implementation on 2026-08-22. The site presents model preparation before generation. A user explicitly starts the large download, while generation stays unavailable until the active manifest cache is ready.

## Goal

Provide a reliable browser-side lifecycle for the approximately 8.08 GB `music-variable` artifact set:

- inspect the active manifest cache without creating it;
- request persistent origin storage from the window;
- estimate whether the remaining artifacts fit before transfer;
- download, resume, verify, and report current progress, speed, and ETA;
- retry a failed download without discarding valid completed or partial files;
- remove every MiniMax Music 3 manifest cache without touching unrelated OPFS data; and
- prevent product generation until the active cache is verified ready.

## Web platform limits

`navigator.storage.estimate()` returns an implementation-defined estimate. It does not reserve quota or guarantee physical disk allocation. `navigator.storage.persist()` requests eviction protection and is exposed only on `Window`; it does not increase or reserve quota. The implementation therefore reports an advisory admission decision, handles `QuotaExceededError` during writes, and never tells the user that space has been reserved.

The implementation follows the WHATWG Storage and File System standards:

- https://storage.spec.whatwg.org/#storage-manager
- https://storage.spec.whatwg.org/#usage-and-quota
- https://fs.spec.whatwg.org/#dom-filesystemdirectoryhandle-removeentry

## Chosen architecture

Extend the existing inference worker instead of duplicating artifact I/O on the main thread or creating a second worker. The existing `ensureArtifact` path already owns range resume, streaming writes, SHA-256 verification, completion receipts, and progress coalescing.

The main window owns only the persistence request because `StorageManager.persist()` is not exposed to workers. Cache inspection, quota estimation, download, and deletion stay in the worker. All cache mutations use one exclusive Web Lock named `minimax-music3-artifact-cache`, so another tab cannot download while a deletion is in progress.

Alternatives rejected:

1. A main-thread cache manager would duplicate OPFS and progress logic and could race the inference worker.
2. A dedicated artifact worker would require cross-worker status and locking even though downloading and inference never need to run concurrently.

## Product flow

On page load:

1. Check `persisted()` without requesting permission.
2. Fetch and validate the active manifest.
3. Inspect the manifest-scoped OPFS directory without creating it.
4. Call `estimate()` and report cache, quota, and persistence status.
5. Show Download when the cache is not ready and Generate only when it is ready.

When the user starts model preparation:

1. The window calls `persist()` from the user action.
2. The worker refreshes cache and quota status.
3. If an estimate is unavailable or insufficient, no artifact transfer starts.
4. Persistence denial is a visible warning, not a download blocker.
5. The worker downloads artifacts sequentially under the exclusive cache lock.
6. A terminal ready result is emitted only after every artifact is verified or has a trusted completion receipt of the exact expected size.

Retry sends the same download request again. Existing valid completed files remain cache hits, and valid partial files resume with HTTP Range. The current one clean retry for a hash mismatch remains unchanged. There is no additional hidden network retry loop.

Cancellation continues to terminate the worker. Files completed before termination remain ready. A network or write error closes the current writer so its partial file can resume. The next inspection is authoritative.

## Capacity admission

For every referenced artifact, inspection records expected bytes, stored bytes, and whether the exact-size completion receipt exists. The cache summary includes:

- total artifact bytes and count;
- complete artifact bytes and count;
- currently stored referenced bytes;
- additional bytes needed to reach expected sizes;
- largest pending artifact size;
- project cache count and total stored project-cache bytes; and
- active cache state: `missing`, `partial`, or `ready`.

The advisory required headroom is:

```text
additional bytes needed + largest pending artifact size
```

The extra largest-artifact allowance covers one file-level safe-write or replacement at a time. Physical release files are capped at 128 MiB. A ready cache requires no additional headroom.

An estimate is sufficient only when `quota - usage` is at least the required headroom. All numbers remain exact bytes in the protocol. UI formatting may use GiB and MiB.

## Cache scope and deletion

Project caches are only direct OPFS-root directories whose names exactly match:

```text
^minimax-music3-[a-f0-9]{64}$
```

The delete operation recursively removes every matching directory and ignores all unrelated files, directories, and near-match names. Deletion is non-atomic by Web platform definition. A failure reports which operation failed, then a new inspection shows the remaining state. The UI asks for confirmation before sending the destructive request.

## Worker protocol

New requests:

```ts
type ArtifactCacheRequest =
  | { type: 'inspect-artifact-cache'; manifestUrl: string }
  | { type: 'download-artifacts'; manifestUrl: string }
  | { type: 'delete-artifact-caches'; manifestUrl: string };
```

The delete request carries the active manifest URL so the worker can return the refreshed active-cache status after deletion.

New successful responses:

```ts
type ArtifactCacheResponse =
  | { type: 'artifact-cache-status'; status: ArtifactCacheStatus }
  | { type: 'artifact-download-complete'; status: ArtifactCacheStatus }
  | { type: 'artifact-cache-deleted'; status: ArtifactCacheStatus };
```

`ArtifactCacheStatus` records manifest hash, cache state, artifact counts and bytes, project cache usage, persistence, quota, available bytes, required headroom, and the advisory `sufficient` decision.

Existing artifact progress events gain aggregate byte rate and ETA. They remain throttled to at most one intermediate update per 100 ms per active file, with an exact final event after successful verification.

Errors retain the existing `error` response and add optional structured fields:

```ts
type ArtifactErrorCode =
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

The response identifies the operation and whether retry can help. Quota insufficiency becomes retryable after the user frees space or removes stale project caches. Manifest invalidity is not retryable until deployment changes.

## Generation behavior

The product `generate-music` route fetches and validates the manifest, then inspects its cache. It fails with `cache-not-ready` instead of starting a large implicit download. Diagnostic and legacy smoke routes keep their existing implicit cache behavior so this feature does not broaden into a diagnostic rewrite.

## Existing-screen integration

This task adds only the minimum controls needed to exercise the cache lifecycle before the final WebUI redesign:

- storage persistence and estimated-capacity status;
- active cache ready, partial, or missing status;
- Download Model, Retry Download, Refresh Status, and Remove Cached Model actions;
- byte progress, current file, transfer rate, and ETA; and
- generation disabled until the active cache is ready.

It does not redesign the prompt, lyrics, sampler, result, or diagnostic layout.

## Testing

Unit tests cover:

- missing, partial, complete, exact-size-unverified, and oversized artifact inspection;
- exact additional-byte and headroom arithmetic;
- persistence granted, denied, and unsupported states;
- sufficient, insufficient, and unavailable estimates;
- strict project-cache name matching and unrelated OPFS preservation;
- recursive delete success and partial failure reporting;
- worker protocol validation and one-operation serialization;
- download progress rate and ETA;
- user-initiated retry preserving completed and partial artifacts;
- product generation rejecting a non-ready cache; and
- minimal App readiness, retry, confirmation, and removal behavior.

A small browser fixture gate covers interruption, a resumed Range request, ready state, and project-only deletion. It must not use or delete the real 8 GB cache.

## Acceptance criteria

- Page load performs inspection only and does not create a cache or start the model download.
- Download starts only after an explicit user action.
- No transfer starts when the current estimate is unavailable or insufficient.
- Persistence denial is clearly reported but does not block sufficient storage.
- Progress shows verified aggregate bytes, transfer rate, and ETA.
- Retry resumes reusable data and never re-downloads trusted completed files.
- Delete removes only exact MiniMax Music 3 cache directories.
- Product generation cannot implicitly download the model.
- Quota and write failures remain recoverable and never claim reserved space.
- Existing generation, converter, lint, typecheck, and build checks remain green.
