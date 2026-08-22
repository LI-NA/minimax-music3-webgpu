import { expect, test } from '@playwright/test';

test('resumes an interrupted artifact download in OPFS and deletes only project caches', async ({ page }) => {
  await page.goto('/src/runtime/model/artifact-cache.ts');

  const result = await page.evaluate(async () => {
    const cacheModuleUrl = '/src/runtime/model/artifact-cache.ts';
    const managementModuleUrl = '/src/runtime/model/artifact-cache-management.ts';
    const { ensureArtifact, OpfsArtifactStore } = await import(/* @vite-ignore */ cacheModuleUrl);
    const { deleteProjectArtifactCaches, inspectArtifactCache } = await import(/* @vite-ignore */ managementModuleUrl);
    const root = await navigator.storage.getDirectory();
    const iterableRoot = root as FileSystemDirectoryHandle & {
      entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
    };
    const artifactHash = '0123456789abcdef'.repeat(4);
    const extraProjectHash = 'fedcba9876543210'.repeat(4);
    const artifactCacheName = `minimax-music3-${artifactHash}`;
    const extraProjectCacheName = `minimax-music3-${extraProjectHash}`;
    const unrelatedName = 'artifact-cache-browser-unrelated';
    const nearMatchName = `${artifactCacheName}-extra`;
    const fixtureNames = [artifactCacheName, extraProjectCacheName, unrelatedName, nearMatchName];
    const ownedNames = new Set<string>();
    const projectCachePattern = /^minimax-music3-[a-f0-9]{64}$/;
    const listDirectories = async () => {
      const names: string[] = [];
      for await (const [name, handle] of iterableRoot.entries()) {
        if (handle.kind === 'directory') names.push(name);
      }
      return names.sort();
    };
    const removeTestOwnedDirectories = async () => {
      let cleanupFailed = false;
      let firstCleanupError: unknown;
      for (const name of ownedNames) {
        try {
          await root.removeEntry(name, { recursive: true });
        } catch (error) {
          if (!(error instanceof DOMException && error.name === 'NotFoundError') && !cleanupFailed) {
            cleanupFailed = true;
            firstCleanupError = error;
          }
        }
      }
      if (cleanupFailed) throw firstCleanupError;
    };

    try {
      const initialDirectories = await listDirectories();
      const initialProjectCaches = initialDirectories.filter((name) => projectCachePattern.test(name));
      if (initialProjectCaches.length !== 0) {
        throw new Error(`browser context is not isolated: ${initialProjectCaches.join(', ')}`);
      }
      if (fixtureNames.some((name) => initialDirectories.includes(name))) {
        throw new Error('browser context contains a test-owned directory before setup');
      }

      const manifestResponse = await fetch('http://127.0.0.1:5174/manifest.json');
      if (!manifestResponse.ok) throw new Error('active manifest is unavailable');
      const manifestText = await manifestResponse.text();
      const activeManifestHash = Array.from(
        new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(manifestText))),
        (byte) => byte.toString(16).padStart(2, '0'),
      ).join('');
      if ([artifactHash, extraProjectHash].includes(activeManifestHash)) {
        throw new Error('test artifact hash matches the active manifest hash');
      }

      const data = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
      const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)), (byte) =>
        byte.toString(16).padStart(2, '0'),
      ).join('');
      const artifact = { path: 'tiny.bin', bytes: data.byteLength, sha256 };
      const source = new URL('https://artifact-cache.test/tiny.bin');
      const store = await OpfsArtifactStore.open(artifactHash, root);
      ownedNames.add(artifactCacheName);
      let sentPartialChunk = false;
      const interruptedBody = new ReadableStream<Uint8Array>({
        pull(controller) {
          if (!sentPartialChunk) {
            sentPartialChunk = true;
            controller.enqueue(data.slice(0, 3));
          } else {
            controller.error(new Error('injected stream failure'));
          }
        },
      });

      let interruption = '';
      try {
        await ensureArtifact(
          artifact,
          source,
          store,
          () => {},
          async () => new Response(interruptedBody),
        );
      } catch (error) {
        interruption = error instanceof Error ? error.message : String(error);
      }
      const partialSize = await store.size(artifact.path);

      let requestedRange: string | null = null;
      await ensureArtifact(
        artifact,
        source,
        store,
        () => {},
        async (_input: URL, init?: RequestInit) => {
          requestedRange = new Headers(init?.headers).get('Range');
          if (requestedRange !== 'bytes=3-') {
            throw new Error(`unexpected retry range: ${requestedRange}`);
          }
          return new Response(data.slice(3), { status: 206 });
        },
      );
      const inspection = await inspectArtifactCache([artifact], store);

      await root.getDirectoryHandle(extraProjectCacheName, { create: true });
      ownedNames.add(extraProjectCacheName);
      await root.getDirectoryHandle(unrelatedName, { create: true });
      ownedNames.add(unrelatedName);
      await root.getDirectoryHandle(nearMatchName, { create: true });
      ownedNames.add(nearMatchName);
      const beforeDeletion = await listDirectories();
      const expectedProjectCaches = [artifactCacheName, extraProjectCacheName].sort();
      const actualProjectCaches = beforeDeletion.filter((name) => projectCachePattern.test(name));
      if (JSON.stringify(actualProjectCaches) !== JSON.stringify(expectedProjectCaches)) {
        throw new Error(`unexpected project caches before deletion: ${actualProjectCaches.join(', ')}`);
      }

      await deleteProjectArtifactCaches(root);
      const afterDeletion = await listDirectories();

      return {
        interruption,
        partialSize,
        requestedRange,
        inspection,
        projectCachesAfterDeletion: afterDeletion.filter((name) => projectCachePattern.test(name)),
        unrelatedRemains: afterDeletion.includes(unrelatedName),
        nearMatchRemains: afterDeletion.includes(nearMatchName),
      };
    } finally {
      await removeTestOwnedDirectories();
    }
  });

  expect(result.interruption).toContain('injected stream failure');
  expect(result.partialSize).toBe(3);
  expect(result.requestedRange).toBe('bytes=3-');
  expect(result.inspection.state).toBe('ready');
  expect(result.projectCachesAfterDeletion).toEqual([]);
  expect(result.unrelatedRemains).toBe(true);
  expect(result.nearMatchRemains).toBe(true);
});
