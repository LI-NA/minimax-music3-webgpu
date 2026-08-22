# Artifact hosting and manifest configuration

Date: 2026-08-22

## Problem

The application hard-codes `http://127.0.0.1:5174/manifest.json`. Three failures follow from
that single constant.

1. Every manifest request is cross-origin by construction. `tools/serve-artifacts.mjs`
   answers with `Access-Control-Allow-Origin: http://127.0.0.1:5173`, so a developer who
   runs `npm run dev` and opens `http://localhost:5173` is blocked by CORS. The worker
   discards the reason and reports only `Music release manifest is unavailable`.
2. The artifact server selects its release from `MINIMAX_RELEASE`, so an operator who
   forgets the variable silently serves `artifacts/release/global` to an application that
   requires `music-variable`.
3. A second Vite instance takes port 5174 when 5173 is occupied, and `localhost:5174` then
   answers with `index.html` instead of a manifest.

The constant also blocks the public release. A production build has no artifact server, and
GitHub Pages cannot host 8,083,535,909 bytes of model weights.

## Goals

- Make the manifest URL configurable per build, with a safe local default.
- Serve local release files same-origin so CORS cannot fail in development.
- Keep automatic cache inspection on load and manual, button-driven download.
- Deploy the public build to GitHub Pages with the manifest hosted on Hugging Face.
- Keep the diagnostics page out of the public bundle.

## Non-goals

- Changing the download, verification, or generation pipeline.
- Cross-origin isolation (COOP/COEP). The JSPI build does not need `SharedArrayBuffer`,
  and GitHub Pages cannot set response headers.

## Design

### Manifest URL resolution

`src/app/manifest-url.ts` owns URL selection.

```ts
const BUILD_DEFAULT =
  import.meta.env.VITE_MINIMAX_MANIFEST_URL ?? `${import.meta.env.BASE_URL}artifacts/music-variable/manifest.json`;

export function resolveManifestUrl(search = window.location.search): string {
  if (import.meta.env.DEV) {
    const override = new URLSearchParams(search).get('manifest');
    if (override) return override;
  }
  return BUILD_DEFAULT;
}

export const localReleaseManifestUrl = (release: string) =>
  `${import.meta.env.BASE_URL}artifacts/${release}/manifest.json`;
```

`import.meta.env.DEV` is replaced with a literal at build time, so the override branch is
removed from the production bundle by dead-code elimination. A public visitor cannot point
the application at an arbitrary manifest, which would be arbitrary model injection.

### Same-origin development serving

`tools/artifact-middleware.ts` holds the request handler extracted from
`tools/serve-artifacts.mjs`: realpath-based containment against the release root, byte-range
parsing, 416 with `Content-Range: bytes */size`, and stream-error containment. A Vite plugin
mounts it during `serve` only.

```
/artifacts/<release>/*  ->  artifacts/release/<release>/*
```

The release is a path segment rather than a port, so the primary-release indirection through
`MINIMAX_RELEASE` disappears. No CORS headers are needed because nothing is cross-origin.
The plugin registers `configureServer` only, so 8 GB of artifacts never reach `dist/`.

The middleware resolves the canonical release root lazily. A checkout without an `artifacts/`
directory must not crash the dev server.

`tools/serve-artifacts.mjs` is deleted along with `MINIMAX_APP_ORIGIN`,
`MINIMAX_ARTIFACT_PORT`, `MINIMAX_RELEASE`, and ports 5174 through 5178. Playwright drops
from six web servers to one.

### Download flow

No behavior change. `cache.inspect()` already runs on mount and download already requires a
button press. Only the diagnostic quality of the failure changes: `fetchManifest` in
`src/workers/inference.worker.ts` keeps the requested URL and the underlying reason, so a
CORS block, a 404, and a refused connection are distinguishable.

### Base path

`vite.config.ts` reads `base` from `MINIMAX_BASE`, defaulting to `/`. Runtime absolute paths
derive from `import.meta.env.BASE_URL`:

- `localJspiWasmPaths(origin, base)` builds `${base}ort/<name>?v=<patch>`. Without this the
  WASM 404s under a GitHub Pages project path.
- The dev middlewares mount at `${base}ort/` and `${base}artifacts`.

This keeps project pages and custom domains equivalent, so the deployment target can be
decided later without rework.

### Diagnostics separation

`diagnostics.html` plus `src/diagnostics.tsx` become a second entry point. Vite builds only
`index.html` unless another input is declared, so a root-level HTML file is served in
development and absent from the production build with no conditional configuration.

`src/main.tsx` loses its pathname and query sniffing, its `lazy` import, and its `Suspense`
wrapper. `DiagnosticsApp` already has a default export and imports its own stylesheet.

The previous sub-paths (`/diagnostics/condition`, `/diagnostics/flow`) were decorative:
`DiagnosticsApp` renders every panel and the specs click by button name. They collapse to
`/diagnostics.html`. The primary release becomes `?release=<name>`, defaulting to
`music-variable`.

### Public release

`.env.production` pins the manifest to a Hugging Face commit:

```
VITE_MINIMAX_MANIFEST_URL=https://huggingface.co/<owner>/<repo>/resolve/<sha>/manifest.json
```

Manifest entries are relative paths, so the artifact base follows the manifest URL. Hugging
Face answers `206 Partial Content` with `Accept-Ranges: bytes` and permissive CORS across the
redirect to its CDN, so the resume path in `src/runtime/model/artifact-cache.ts` works
unchanged. Per-file SHA-256 verification means the CDN does not have to be trusted.

`.github/workflows/pages.yml` builds and deploys with `pages: write` and `id-token: write`.
`tools/app-revision.ts` hashes source files rather than reading git, so it works in CI.

## Testing

- `manifest-url`: override honored in development, ignored without a query, build default
  falls back to the base-relative local path, `VITE_MINIMAX_MANIFEST_URL` wins.
- `artifact-middleware`: the existing containment, symlink-escape, range, empty-file, and
  stream-error cases, driven in-process instead of through a spawned server.
- `local-jspi-path`: base-prefixed asset URLs.
- Browser specs keep their assertions; only URLs change to same-origin equivalents.
- The production bundle is grepped to confirm the manifest override branch is absent.
