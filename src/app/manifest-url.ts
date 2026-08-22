/**
 * Base URL of the release the application loads. Production builds pin this to a hosted
 * manifest; development falls back to the same-origin release served by the Vite plugin.
 */
export const BUILD_DEFAULT_MANIFEST_URL =
  import.meta.env.VITE_MINIMAX_MANIFEST_URL ?? localReleaseManifestUrl('music-variable');

/** Same-origin URL of a converted release under `artifacts/release`, served in development only. */
export function localReleaseManifestUrl(release: string): string {
  return `${import.meta.env.BASE_URL}artifacts/${release}/manifest.json`;
}

/**
 * Primary release a diagnostics run loads. Gates select it with `?release=`, replacing the
 * environment variable that used to decide which release a fixed port happened to serve.
 */
export function selectedRelease(search: string = currentSearch(), fallback = 'music-variable'): string {
  return new URLSearchParams(search).get('release') ?? fallback;
}

// These modules are imported by unit tests that run outside a browser environment.
function currentSearch(): string {
  return typeof window === 'undefined' ? '' : window.location.search;
}

/**
 * Resolves the manifest the application loads.
 *
 * The `?manifest=` override exists so a developer can point the app at a mirror or a hosted
 * release without rebuilding. It is guarded by `import.meta.env.DEV`, which the bundler
 * replaces with a literal, so the branch is removed from production builds: on the public
 * site an arbitrary manifest would be arbitrary model injection.
 */
export function resolveManifestUrl(search: string = currentSearch()): string {
  if (import.meta.env.DEV) {
    const override = new URLSearchParams(search).get('manifest');
    if (override) return override;
  }
  return BUILD_DEFAULT_MANIFEST_URL;
}
