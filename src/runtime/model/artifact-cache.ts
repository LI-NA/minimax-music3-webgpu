import { sha256 } from '@noble/hashes/sha2.js';
import { bytesToHex } from '@noble/hashes/utils.js';
import type { ArtifactFile } from './manifest';

export type ProgressSink = (progress: {
  path: string;
  loaded: number;
  total: number;
  transferred: number;
}) => void;
export interface ArtifactWriter {
  write(data: Uint8Array): Promise<void>;
  close(): Promise<void>;
}
export interface ArtifactStore {
  size(path: string): Promise<number>;
  stream(path: string, sink: (chunk: Uint8Array) => void | Promise<void>): Promise<void>;
  writer(path: string, append: boolean): Promise<ArtifactWriter>;
  remove(path: string): Promise<void>;
  isComplete(path: string): Promise<boolean>;
  markComplete(path: string): Promise<void>;
  file(path: string): Promise<File>;
}
export interface OpfsSyncFileHandle {
  read(buffer: Uint8Array, options: { at: number }): number;
  close(): void;
}
export type Fetcher = (input: URL, init?: RequestInit) => Promise<Response>;

export async function ensureArtifact(
  file: ArtifactFile,
  source: URL,
  store: ArtifactStore,
  onProgress: ProgressSink,
  fetcher: Fetcher = fetch,
): Promise<File> {
  let existingSize = await store.size(file.path);
  const complete = await store.isComplete(file.path);
  if (complete && existingSize === file.bytes) return store.file(file.path);
  if (complete) {
    await store.remove(file.path);
    existingSize = 0;
  }
  const digestStored = async () => {
    const hash = sha256.create();
    await store.stream(file.path, async (chunk) => {
      hash.update(chunk);
    });
    return bytesToHex(hash.digest());
  };
  if (existingSize === file.bytes && (await digestStored()) === file.sha256) {
    await store.markComplete(file.path);
    return store.file(file.path);
  }
  if (existingSize >= file.bytes) {
    await store.remove(file.path);
    existingSize = 0;
  }
  let transferred = 0;
  for (let attempt = 0; attempt < 2; attempt++) {
    const partialSize = attempt === 0 ? existingSize : 0;
    const response = await fetcher(
      source,
      partialSize ? { headers: { Range: `bytes=${partialSize}-` } } : undefined,
    );
    if (!response.ok) throw new Error(`artifact request failed: ${file.path}`);
    const append = partialSize > 0 && response.status === 206;
    const hash = sha256.create();
    if (append)
      await store.stream(file.path, async (chunk) => {
        hash.update(chunk);
      });
    const writer = await store.writer(file.path, append);
    let loaded = append ? partialSize : 0;
    const reader = response.body?.getReader();
    if (!reader) throw new Error(`artifact response has no body: ${file.path}`);
    let failed = false;
    try {
      for (;;) {
        const next = await reader.read();
        if (next.done) break;
        try {
          await writer.write(next.value);
        } catch (error) {
          const detail = error instanceof Error ? error.message : String(error);
          throw new Error(`artifact write failed: ${file.path} at ${loaded} bytes: ${detail}`, {
            cause: error,
          });
        }
        hash.update(next.value);
        loaded += next.value.byteLength;
        transferred += next.value.byteLength;
        onProgress({ path: file.path, loaded, total: file.bytes, transferred });
      }
    } catch (error) {
      failed = true;
      throw error;
    } finally {
      if (failed) await writer.close().catch(() => undefined);
      else await writer.close();
    }
    if (loaded === file.bytes && bytesToHex(hash.digest()) === file.sha256) {
      await store.markComplete(file.path);
      return store.file(file.path);
    }
    await store.remove(file.path);
  }
  throw new Error(`artifact verification failed: ${file.path}`);
}

export class OpfsArtifactStore implements ArtifactStore {
  constructor(private readonly root: FileSystemDirectoryHandle) {}
  static async open(
    manifestHash: string,
    opfsRoot?: FileSystemDirectoryHandle,
  ): Promise<OpfsArtifactStore> {
    const root = opfsRoot ?? await navigator.storage.getDirectory();
    return new OpfsArtifactStore(
      await root.getDirectoryHandle(`minimax-music3-${manifestHash}`, {
        create: true,
      }),
    );
  }
  static async openExisting(
    manifestHash: string,
    opfsRoot?: FileSystemDirectoryHandle,
  ): Promise<OpfsArtifactStore | undefined> {
    const root = opfsRoot ?? await navigator.storage.getDirectory();
    try {
      return new OpfsArtifactStore(
        await root.getDirectoryHandle(`minimax-music3-${manifestHash}`),
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === 'NotFoundError') return undefined;
      throw error;
    }
  }
  private async handle(path: string, create = false) {
    const parts = path.split('/');
    let directory = this.root;
    for (const part of parts.slice(0, -1))
      directory = await directory.getDirectoryHandle(part, { create });
    return directory.getFileHandle(parts.at(-1)!, { create });
  }
  async size(path: string) {
    try {
      return (await (await this.handle(path)).getFile()).size;
    } catch (error) {
      if (error instanceof DOMException && error.name === 'NotFoundError') return 0;
      throw error;
    }
  }
  async stream(path: string, sink: (chunk: Uint8Array) => void | Promise<void>) {
    const reader = (await (await this.handle(path)).getFile()).stream().getReader();
    for (;;) {
      const next = await reader.read();
      if (next.done) return;
      await sink(next.value);
    }
  }
  async writer(path: string, append: boolean) {
    const writable = await (
      await this.handle(path, true)
    ).createWritable({ keepExistingData: append });
    if (append) await writable.seek(await this.size(path));
    return {
      write: (data: Uint8Array) => writable.write(data.slice()),
      close: () => writable.close(),
    };
  }
  async remove(path: string) {
    const removeEntry = async (entryPath: string) => {
      try {
        const parts = entryPath.split('/');
        let directory = this.root;
        for (const part of parts.slice(0, -1))
          directory = await directory.getDirectoryHandle(part);
        await directory.removeEntry(parts.at(-1)!);
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'NotFoundError')) throw error;
      }
    };
    await removeEntry(path);
    await removeEntry(`${path}.complete`);
  }
  async isComplete(path: string) {
    return (await this.size(`${path}.complete`)) > 0;
  }
  async markComplete(path: string) {
    const writer = await this.writer(`${path}.complete`, false);
    await writer.write(new Uint8Array([1]));
    await writer.close();
  }
  async file(path: string) {
    return (await this.handle(path)).getFile();
  }
  async openSyncFile(path: string): Promise<OpfsSyncFileHandle> {
    const handle = (await this.handle(path)) as FileSystemFileHandle & {
      createSyncAccessHandle(): Promise<OpfsSyncFileHandle>;
    };
    return handle.createSyncAccessHandle();
  }
}
