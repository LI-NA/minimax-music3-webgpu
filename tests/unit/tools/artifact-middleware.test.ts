import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { createArtifactMiddleware, type ArtifactMiddlewareOptions } from '../../../tools/artifact-middleware';

const root = mkdtempSync(join(tmpdir(), 'minimax-artifacts-'));
const servers: Server[] = [];

async function serve(options: ArtifactMiddlewareOptions) {
  const middleware = createArtifactMiddleware(options);
  const server = createServer((request, response) =>
    middleware(request, response, () => {
      response.writeHead(501);
      response.end();
    }),
  );
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  return `http://127.0.0.1:${typeof address === 'object' && address ? address.port : 0}`;
}

let origin: string;

beforeAll(async () => {
  writeFileSync(join(root, 'sample.bin'), Buffer.from([1, 2, 3, 4]));
  writeFileSync(join(root, 'empty.bin'), Buffer.alloc(0));
  origin = await serve({ releaseRoot: root });
});

afterAll(async () => {
  await Promise.all(servers.map((server) => new Promise((resolve) => server.close(resolve))));
});

describe('artifact middleware', () => {
  it('serves a contained single byte range and rejects traversal', async () => {
    const range = await fetch(`${origin}/sample.bin`, { headers: { Range: 'bytes=1-2' } });
    expect(range.status).toBe(206);
    expect(range.headers.get('accept-ranges')).toBe('bytes');
    expect(range.headers.get('content-range')).toBe('bytes 1-2/4');
    expect([...new Uint8Array(await range.arrayBuffer())]).toEqual([2, 3]);
    expect((await fetch(`${origin}/..%2fsample.bin`)).status).toBe(404);
  });

  it('serves an empty file without constructing an invalid byte range', async () => {
    const full = await fetch(`${origin}/empty.bin`);
    expect(full.status).toBe(200);
    expect(full.headers.get('content-length')).toBe('0');
    expect((await full.arrayBuffer()).byteLength).toBe(0);

    const head = await fetch(`${origin}/empty.bin`, { method: 'HEAD' });
    expect(head.status).toBe(200);
    expect(head.headers.get('content-length')).toBe('0');

    const range = await fetch(`${origin}/empty.bin`, { headers: { Range: 'bytes=0-' } });
    expect(range.status).toBe(416);
    expect(range.headers.get('content-range')).toBe('bytes */0');
  });

  it('survives a file stream error after accepting a request', async () => {
    const failing = await serve({
      releaseRoot: root,
      createFileStream: () =>
        new Readable({
          read() {
            this.destroy(new Error('expected stream failure'));
          },
        }),
    });

    let streamFailed = false;
    try {
      await (await fetch(`${failing}/sample.bin`)).arrayBuffer();
    } catch {
      streamFailed = true;
    }
    expect(streamFailed).toBe(true);

    const followup = await fetch(`${failing}/sample.bin`, { method: 'HEAD' });
    expect(followup.status).toBe(200);
  });

  it('rejects a symlink that resolves outside the release root', async (context) => {
    const outside = join(mkdtempSync(join(tmpdir(), 'minimax-artifacts-outside-')), 'secret.bin');
    const link = join(root, 'escape.bin');
    writeFileSync(outside, Buffer.from([9, 8, 7]));
    try {
      symlinkSync(outside, link, 'file');
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code === 'EPERM' || code === 'EACCES') {
        context.skip();
        return;
      }
      throw error;
    }
    expect((await fetch(`${origin}/escape.bin`)).status).toBe(404);
  });

  it('selects a release by path segment without changing range or containment behavior', async () => {
    const releaseRoot = join(mkdtempSync(join(tmpdir(), 'minimax-releases-')), 'release');
    mkdirSync(join(releaseRoot, 'music-variable'), { recursive: true });
    writeFileSync(join(releaseRoot, 'music-variable', 'manifest.json'), Buffer.from([5, 6, 7, 8]));
    const releases = await serve({ releaseRoot });

    const response = await fetch(`${releases}/music-variable/manifest.json`, { headers: { Range: 'bytes=1-2' } });
    expect(response.status).toBe(206);
    expect([...new Uint8Array(await response.arrayBuffer())]).toEqual([6, 7]);
    expect((await fetch(`${releases}/music-variable/..%2f..%2fmanifest.json`)).status).toBe(404);
  });

  it('starts without a release root and picks it up once it appears', async () => {
    const checkout = mkdtempSync(join(tmpdir(), 'minimax-empty-checkout-'));
    const releaseRoot = join(checkout, 'artifacts', 'release');
    const late = await serve({ releaseRoot });
    expect((await fetch(`${late}/music-variable/manifest.json`)).status).toBe(404);

    mkdirSync(join(releaseRoot, 'music-variable'), { recursive: true });
    writeFileSync(join(releaseRoot, 'music-variable', 'manifest.json'), Buffer.from([1]));
    expect((await fetch(`${late}/music-variable/manifest.json`)).status).toBe(200);
  });

  it('passes non-read methods to the next handler', async () => {
    expect((await fetch(`${origin}/sample.bin`, { method: 'POST' })).status).toBe(501);
  });
});

afterAll(() => rmSync(root, { recursive: true, force: true }));
