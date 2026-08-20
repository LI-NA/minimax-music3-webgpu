import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';

const root = mkdtempSync(join(tmpdir(), 'minimax-artifacts-'));
const port = 20_000 + Math.floor(Math.random() * 20_000);
const server = spawn('node', ['tools/serve-artifacts.mjs'], {
  cwd: process.cwd(),
  env: { ...process.env, MINIMAX_RELEASE_ROOT: root, MINIMAX_ARTIFACT_PORT: String(port) },
});

beforeAll(async () => {
  writeFileSync(join(root, 'sample.bin'), Buffer.from([1, 2, 3, 4]));
  await new Promise<void>((resolve) => server.stdout.once('data', () => resolve()));
});
afterAll(() => server.kill());

describe('artifact range server', () => {
  it('serves a contained single byte range and rejects traversal', async () => {
    const range = await fetch(`http://127.0.0.1:${port}/sample.bin`, { headers: { Range: 'bytes=1-2' } });
    expect(range.status).toBe(206);
    expect(range.headers.get('accept-ranges')).toBe('bytes');
    expect(range.headers.get('content-range')).toBe('bytes 1-2/4');
    expect([...new Uint8Array(await range.arrayBuffer())]).toEqual([2, 3]);
    expect((await fetch(`http://127.0.0.1:${port}/..%2fsample.bin`)).status).toBe(404);
  });
});
