import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';

const root = mkdtempSync(join(tmpdir(), 'minimax-artifacts-'));
let server: ReturnType<typeof spawn>;
let port: number;

async function readyPort(child: ReturnType<typeof spawn>) {
  return new Promise<number>((resolve, reject) => {
    let output = '';
    const cleanup = () => {
      child.stdout!.off('data', onData);
      child.off('error', onError);
      child.off('exit', onExit);
    };
    const onData = (data: unknown) => {
      output += String(data);
      const match = /MiniMax artifact server port: (\d+)/.exec(output);
      if (!match) return;
      cleanup();
      resolve(Number(match[1]));
    };
    const onError = (error: Error) => {
      cleanup();
      reject(error);
    };
    const onExit = (code: number | null) => {
      cleanup();
      reject(new Error(`artifact server exited with ${code}: ${output}`));
    };
    child.stdout!.on('data', onData);
    child.on('error', onError);
    child.on('exit', onExit);
  });
}

async function stopServer(child: ReturnType<typeof spawn> | undefined) {
  if (!child || child.exitCode !== null) return;
  child.kill();
  await once(child, 'exit');
}

beforeAll(async () => {
  writeFileSync(join(root, 'sample.bin'), Buffer.from([1, 2, 3, 4]));
  writeFileSync(join(root, 'empty.bin'), Buffer.alloc(0));
  server = spawn('node', ['tools/serve-artifacts.mjs'], {
    cwd: process.cwd(),
    env: { ...process.env, MINIMAX_RELEASE_ROOT: root, MINIMAX_ARTIFACT_PORT: '0' },
  });
  port = await readyPort(server);
});
afterAll(() => stopServer(server));

describe('artifact range server', () => {
  it('serves a contained single byte range and rejects traversal', async () => {
    const range = await fetch(`http://127.0.0.1:${port}/sample.bin`, { headers: { Range: 'bytes=1-2' } });
    expect(range.status).toBe(206);
    expect(range.headers.get('accept-ranges')).toBe('bytes');
    expect(range.headers.get('content-range')).toBe('bytes 1-2/4');
    expect([...new Uint8Array(await range.arrayBuffer())]).toEqual([2, 3]);
    expect((await fetch(`http://127.0.0.1:${port}/..%2fsample.bin`)).status).toBe(404);
  });

  it('serves an empty file without constructing an invalid byte range', async () => {
    const full = await fetch(`http://127.0.0.1:${port}/empty.bin`);
    expect(full.status).toBe(200);
    expect(full.headers.get('content-length')).toBe('0');
    expect((await full.arrayBuffer()).byteLength).toBe(0);

    const head = await fetch(`http://127.0.0.1:${port}/empty.bin`, { method: 'HEAD' });
    expect(head.status).toBe(200);
    expect(head.headers.get('content-length')).toBe('0');

    const range = await fetch(`http://127.0.0.1:${port}/empty.bin`, {
      headers: { Range: 'bytes=0-' },
    });
    expect(range.status).toBe(416);
    expect(range.headers.get('content-range')).toBe('bytes */0');
  });

  it('survives a file stream error after accepting a request', async () => {
    const moduleUrl = new URL('../../../tools/serve-artifacts.mjs', import.meta.url).href;
    const script = `
      import { Readable } from 'node:stream';
      import { createArtifactServer } from ${JSON.stringify(moduleUrl)};
      const server = createArtifactServer({
        releaseRoot: process.env.MINIMAX_RELEASE_ROOT,
        createFileStream: () => new Readable({
          read() { this.destroy(new Error('expected stream failure')); },
        }),
      });
      server.listen(0, '127.0.0.1', () => {
        const address = server.address();
        process.stdout.write('MiniMax artifact server port: ' + address.port + '\\n');
      });
    `;
    const failingServer = spawn('node', ['--input-type=module', '--eval', script], {
      env: { ...process.env, MINIMAX_RELEASE_ROOT: root },
    });
    const failingPort = await readyPort(failingServer);
    try {
      let streamFailed = false;
      try {
        const response = await fetch(`http://127.0.0.1:${failingPort}/sample.bin`);
        await response.arrayBuffer();
      } catch {
        streamFailed = true;
      }
      expect(streamFailed).toBe(true);

      const followup = await fetch(`http://127.0.0.1:${failingPort}/sample.bin`, { method: 'HEAD' });
      expect(followup.status).toBe(200);
      expect(failingServer.exitCode).toBeNull();
    } finally {
      await stopServer(failingServer);
    }
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
    expect((await fetch(`http://127.0.0.1:${port}/escape.bin`)).status).toBe(404);
  });

  it('selects the music-variable release without changing range or containment behavior', async () => {
    const checkout = mkdtempSync(join(tmpdir(), 'minimax-variable-release-'));
    const release = join(checkout, 'artifacts', 'release', 'music-variable');
    mkdirSync(release, { recursive: true });
    writeFileSync(join(release, 'manifest.json'), Buffer.from([5, 6, 7, 8]));
    const variableServer = spawn('node', [join(process.cwd(), 'tools', 'serve-artifacts.mjs')], {
      cwd: checkout,
      env: {
        ...process.env,
        MINIMAX_RELEASE: 'music-variable',
        MINIMAX_ARTIFACT_PORT: '0',
      },
    });
    const variablePort = await readyPort(variableServer);
    try {
      const response = await fetch(`http://127.0.0.1:${variablePort}/manifest.json`, {
        headers: { Range: 'bytes=1-2' },
      });
      expect(response.status).toBe(206);
      expect([...new Uint8Array(await response.arrayBuffer())]).toEqual([6, 7]);
      expect((await fetch(`http://127.0.0.1:${variablePort}/..%2fmanifest.json`)).status).toBe(404);
    } finally {
      await stopServer(variableServer);
    }
  });
});
