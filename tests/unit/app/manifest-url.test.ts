import { describe, expect, it } from 'vitest';
import {
  BUILD_DEFAULT_MANIFEST_URL,
  localReleaseManifestUrl,
  resolveManifestUrl,
  selectedRelease,
} from '../../../src/app/manifest-url';

describe('manifest URL resolution', () => {
  it('addresses local releases by path segment under the deployment base', () => {
    expect(localReleaseManifestUrl('music-variable')).toBe('/artifacts/music-variable/manifest.json');
    expect(localReleaseManifestUrl('rvq')).toBe('/artifacts/rvq/manifest.json');
  });

  it('falls back to the local release when no manifest is configured', () => {
    expect(BUILD_DEFAULT_MANIFEST_URL).toBe('/artifacts/music-variable/manifest.json');
    expect(resolveManifestUrl('')).toBe(BUILD_DEFAULT_MANIFEST_URL);
    expect(resolveManifestUrl('?release=global&frames=2')).toBe(BUILD_DEFAULT_MANIFEST_URL);
  });

  it('honours the development override so a mirror needs no rebuild', () => {
    expect(resolveManifestUrl('?manifest=https://example.test/release/manifest.json')).toBe(
      'https://example.test/release/manifest.json',
    );
  });

  it('selects the diagnostics release from the query rather than the environment', () => {
    expect(selectedRelease('?release=music-5s')).toBe('music-5s');
    expect(selectedRelease('?release=global&frames=2')).toBe('global');
    expect(selectedRelease('')).toBe('music-variable');
    expect(selectedRelease('?frames=2')).toBe('music-variable');
  });
});
