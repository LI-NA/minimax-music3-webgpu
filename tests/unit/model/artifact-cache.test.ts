import { describe, expect, it, vi } from 'vitest';
import {
  ensureArtifact,
  OpfsArtifactStore,
  type ArtifactStore,
  type ArtifactWriter,
} from '../../../src/runtime/model/artifact-cache';

const encoder = new TextEncoder();
const sha256 = async (value: Uint8Array) =>
  Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', value.slice())))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');

class MemoryStore implements ArtifactStore {
  readonly files = new Map<string, Uint8Array>();
  readonly complete = new Set<string>();
  writes: Uint8Array[] = [];
  streamCalls = 0;
  async size(path: string) {
    return this.files.get(path)?.byteLength ?? 0;
  }
  async stream(path: string, sink: (chunk: Uint8Array) => void | Promise<void>) {
    this.streamCalls++;
    const file = this.files.get(path);
    if (file) await sink(file);
  }
  async writer(path: string, append: boolean): Promise<ArtifactWriter> {
    const chunks = append && this.files.has(path) ? [this.files.get(path)!] : [];
    return {
      write: async (chunk) => {
        this.writes.push(chunk);
        chunks.push(chunk);
      },
      close: async () => {
        const contents = new Uint8Array(
          chunks.reduce((size, chunk) => size + chunk.byteLength, 0),
        );
        let offset = 0;
        for (const chunk of chunks) {
          contents.set(chunk, offset);
          offset += chunk.byteLength;
        }
        this.files.set(path, contents);
      },
    };
  }
  async remove(path: string) {
    this.files.delete(path);
    this.complete.delete(path);
  }
  async isComplete(path: string) {
    return this.complete.has(path);
  }
  async markComplete(path: string) {
    this.complete.add(path);
  }
  async file(path: string) {
    const bytes = this.files.get(path) ?? new Uint8Array();
    return new File([bytes.slice()], path);
  }
}

describe('OpfsArtifactStore', () => {
  const hash = 'a'.repeat(64);
  const cacheName = `minimax-music3-${hash}`;

  it('opens an existing cache without requesting creation', async () => {
    const cache = {} as FileSystemDirectoryHandle;
    const getDirectoryHandle = vi.fn().mockResolvedValue(cache);

    const store = await OpfsArtifactStore.openExisting(
      hash,
      { getDirectoryHandle } as unknown as FileSystemDirectoryHandle,
    );

    expect(store).toBeInstanceOf(OpfsArtifactStore);
    expect(getDirectoryHandle).toHaveBeenCalledWith(cacheName);
  });

  it('returns undefined only when an existing cache is absent', async () => {
    const getDirectoryHandle = vi.fn().mockRejectedValue(
      new DOMException('missing', 'NotFoundError'),
    );

    await expect(OpfsArtifactStore.openExisting(
      hash,
      { getDirectoryHandle } as unknown as FileSystemDirectoryHandle,
    )).resolves.toBeUndefined();
  });

  it('propagates errors other than a missing existing cache', async () => {
    const error = new DOMException('blocked', 'SecurityError');
    const getDirectoryHandle = vi.fn().mockRejectedValue(error);

    await expect(OpfsArtifactStore.openExisting(
      hash,
      { getDirectoryHandle } as unknown as FileSystemDirectoryHandle,
    )).rejects.toBe(error);
  });

  it('continues to create a cache when opening it normally', async () => {
    const getDirectoryHandle = vi.fn().mockResolvedValue({});

    await OpfsArtifactStore.open(
      hash,
      { getDirectoryHandle } as unknown as FileSystemDirectoryHandle,
    );

    expect(getDirectoryHandle).toHaveBeenCalledWith(cacheName, { create: true });
  });
});

describe('ensureArtifact', () => {
  it('reuses a complete expected-size file without reading or fetching it again', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    store.files.set('a.bin', data);
    store.complete.add('a.bin');

    const file = await ensureArtifact(
      { path: 'a.bin', bytes: 8, sha256: await sha256(data) },
      new URL('https://example.test/a.bin'),
      store,
      () => {},
      async () => {
        throw new Error('completed artifact must not be fetched');
      },
    );

    expect(await file.text()).toBe('abcdefgh');
    expect(store.streamCalls).toBe(0);
  });

  it('marks a valid full file complete without fetching it again', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    store.files.set('a.bin', data);
    let calls = 0;
    await ensureArtifact(
      { path: 'a.bin', bytes: 8, sha256: await sha256(data) },
      new URL('https://example.test/a.bin'),
      store,
      () => {},
      async () => {
        calls++;
        return new Response(data);
      },
    );
    expect(calls).toBe(0);
    expect(await store.isComplete('a.bin')).toBe(true);
    expect(store.streamCalls).toBe(1);
  });

  it('restarts a size-mismatched completed file instead of resuming it', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    store.files.set('a.bin', data.slice(0, 3));
    store.complete.add('a.bin');
    const ranges: string[] = [];

    const file = await ensureArtifact(
      { path: 'a.bin', bytes: 8, sha256: await sha256(data) },
      new URL('https://example.test/a.bin'),
      store,
      () => {},
      async (_url, init) => {
        ranges.push(new Headers(init?.headers).get('Range') ?? '');
        return new Response(data);
      },
    );

    expect(ranges).toEqual(['']);
    expect(await file.text()).toBe('abcdefgh');
    expect(await store.isComplete('a.bin')).toBe(true);
  });

  it('restarts an oversized existing file instead of requesting an invalid range', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    store.files.set('a.bin', encoder.encode('too-long-data'));
    const ranges: string[] = [];
    await ensureArtifact(
      { path: 'a.bin', bytes: 8, sha256: await sha256(data) },
      new URL('https://example.test/a.bin'),
      store,
      () => {},
      async (_url, init) => {
        ranges.push(new Headers(init?.headers).get('Range') ?? '');
        return new Response(data);
      },
    );
    expect(ranges).toEqual(['']);
  });
  it('resumes a partial file using a byte range', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    store.files.set('a.bin', data.slice(0, 3));
    const ranges: string[] = [];
    const source = new URL('https://example.test/a.bin');
    const file = await ensureArtifact(
      { path: 'a.bin', bytes: 8, sha256: await sha256(data) },
      source,
      store,
      () => {},
      async (_url, init) => {
        ranges.push(new Headers(init?.headers).get('Range') ?? '');
        return new Response(data.slice(3), { status: 206 });
      },
    );
    expect(ranges).toEqual(['bytes=3-']);
    expect(await file.text()).toBe('abcdefgh');
    expect(store.writes).toHaveLength(1);
  });

  it('reports only successfully written network bytes as cumulative transferred bytes', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    store.files.set('a.bin', data.slice(0, 3));
    const progress: Array<{ loaded: number; transferred: number }> = [];

    await ensureArtifact(
      { path: 'a.bin', bytes: 8, sha256: await sha256(data) },
      new URL('https://example.test/a.bin'),
      store,
      ({ loaded, transferred }) => progress.push({ loaded, transferred }),
      async () => new Response(data.slice(3), { status: 206 }),
    );

    expect(progress).toEqual([{ loaded: 8, transferred: 5 }]);
  });

  it('keeps transferred bytes cumulative across a hash retry', async () => {
    const data = encoder.encode('abcdefgh');
    const progress: Array<{ loaded: number; transferred: number }> = [];
    let calls = 0;

    await ensureArtifact(
      { path: 'a.bin', bytes: 8, sha256: await sha256(data) },
      new URL('https://example.test/a.bin'),
      new MemoryStore(),
      ({ loaded, transferred }) => progress.push({ loaded, transferred }),
      async () => new Response(calls++ === 0 ? encoder.encode('bad-data') : data),
    );

    expect(progress).toEqual([
      { loaded: 8, transferred: 8 },
      { loaded: 8, transferred: 16 },
    ]);
  });

  it('does not count a network chunk when writing it fails', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    store.writer = async () => ({
      write: async () => { throw new Error('disk full'); },
      close: async () => {},
    });
    const progress = vi.fn();

    await expect(ensureArtifact(
      { path: 'a.bin', bytes: 8, sha256: await sha256(data) },
      new URL('https://example.test/a.bin'),
      store,
      progress,
      async () => new Response(data),
    )).rejects.toThrow('disk full');
    expect(progress).not.toHaveBeenCalled();
  });

  it('restarts when a range request receives a full response', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    store.files.set('a.bin', encoder.encode('abc'));
    const file = await ensureArtifact(
      { path: 'a.bin', bytes: 8, sha256: await sha256(data) },
      new URL('https://example.test/a.bin'),
      store,
      () => {},
      async () => new Response(data),
    );
    expect(await file.text()).toBe('abcdefgh');
  });

  it('closes a partial writer after a read error and resumes it later', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    const artifact = { path: 'a.bin', bytes: 8, sha256: await sha256(data) };
    let sent = false;
    const broken = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (!sent) {
          sent = true;
          controller.enqueue(data.slice(0, 3));
        } else controller.error(new Error('network lost'));
      },
    });
    await expect(
      ensureArtifact(
        artifact,
        new URL('https://example.test/a.bin'),
        store,
        () => {},
        async () => new Response(broken),
      ),
    ).rejects.toThrow('network lost');
    expect(await store.size('a.bin')).toBe(3);
    const ranges: string[] = [];
    await ensureArtifact(
      artifact,
      new URL('https://example.test/a.bin'),
      store,
      () => {},
      async (_url, init) => {
        ranges.push(new Headers(init?.headers).get('Range') ?? '');
        return new Response(data.slice(3), { status: 206 });
      },
    );
    expect(ranges).toEqual(['bytes=3-']);
  });

  it('preserves the write failure when closing the failed stream also rejects', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    store.writer = async () => ({
      write: async () => {
        throw new Error('write failed');
      },
      close: async () => {
        throw new Error('close masked the write failure');
      },
    });

    await expect(
      ensureArtifact(
        { path: 'a.bin', bytes: data.byteLength, sha256: await sha256(data) },
        new URL('https://example.test/a.bin'),
        store,
        () => {},
        async () => new Response(data),
      ),
    ).rejects.toThrow('write failed');
  });

  it('retries only one hash-mismatched file and reuses a verified visit', async () => {
    const data = encoder.encode('abcdefgh');
    const store = new MemoryStore();
    let calls = 0;
    const artifact = { path: 'a.bin', bytes: 8, sha256: await sha256(data) };
    await ensureArtifact(
      artifact,
      new URL('https://example.test/a.bin'),
      store,
      () => {},
      async () => new Response(calls++ === 0 ? encoder.encode('bad-data') : data),
    );
    await ensureArtifact(
      artifact,
      new URL('https://example.test/a.bin'),
      store,
      () => {},
      async () => {
        calls++;
        return new Response(data);
      },
    );
    expect(calls).toBe(2);
  });
});
