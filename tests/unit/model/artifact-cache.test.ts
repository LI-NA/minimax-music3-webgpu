import { describe, expect, it } from 'vitest';
import {
  ensureArtifact,
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
  async size(path: string) {
    return this.files.get(path)?.byteLength ?? 0;
  }
  async stream(path: string, sink: (chunk: Uint8Array) => void | Promise<void>) {
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

describe('ensureArtifact', () => {
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
