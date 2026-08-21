import { createReadStream, existsSync, realpathSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const origin = 'http://127.0.0.1:5173';
const port = Number(process.env.MINIMAX_ARTIFACT_PORT ?? 5174);
const requestedRelease = process.env.MINIMAX_RELEASE;
const releaseName = ['global-one-layer', 'rvq', 'condition', 'flow', 'vocoder', 'music-5s', 'music-variable'].includes(requestedRelease) ? requestedRelease : 'global';
const defaultRoot = resolve(process.env.MINIMAX_RELEASE_ROOT ?? `artifacts/release/${releaseName}`);

function reply(response, status, headers = {}) {
  response.writeHead(status, { 'Access-Control-Allow-Origin': origin, ...headers });
  response.end();
}

function range(value, size) {
  if (!value?.startsWith('bytes=') || value.includes(',')) return undefined;
  const match = /^bytes=(\d*)-(\d*)$/.exec(value);
  if (!match) return undefined;
  if (size === 0) return undefined;
  const start = match[1] === '' ? undefined : Number(match[1]);
  const end = match[2] === '' ? undefined : Number(match[2]);
  if ((start !== undefined && !Number.isSafeInteger(start)) || (end !== undefined && !Number.isSafeInteger(end))) return undefined;
  if (start === undefined) {
    if (end === undefined || end < 1) return undefined;
    return { start: Math.max(0, size - end), end: size - 1 };
  }
  if (start >= size || (end !== undefined && (end < start || end >= size))) return undefined;
  return { start, end: end ?? size - 1 };
}

export function createArtifactServer({ releaseRoot = defaultRoot, createFileStream = createReadStream } = {}) {
  const root = resolve(releaseRoot);
  const canonicalRoot = realpathSync(root);

  function artifactPath(pathname) {
    const relative = decodeURIComponent(pathname).replace(/^\/+/, '');
    const file = resolve(root, relative);
    if (!(file === root || file.startsWith(`${root}${sep}`)) || !existsSync(file)) return undefined;
    try {
      const canonicalFile = realpathSync(file);
      return canonicalFile === canonicalRoot || canonicalFile.startsWith(`${canonicalRoot}${sep}`)
        ? canonicalFile
        : undefined;
    } catch {
      return undefined;
    }
  }

  return createServer((request, response) => {
    if (!request.url || !['GET', 'HEAD'].includes(request.method ?? '')) return reply(response, 405);
    let file;
    try {
      file = artifactPath(new URL(request.url, `http://${request.headers.host}`).pathname);
    } catch {
      return reply(response, 400);
    }
    if (!file || !existsSync(file) || !statSync(file).isFile()) return reply(response, 404);
    const size = statSync(file).size;
    const requested = request.headers.range;
    const selected = requested ? range(requested, size) : { start: 0, end: size - 1 };
    if (!selected) return reply(response, 416, { 'Accept-Ranges': 'bytes', 'Content-Range': `bytes */${size}` });
    const length = selected.end - selected.start + 1;
    const partial = Boolean(requested);
    const headers = {
      'Accept-Ranges': 'bytes',
      'Content-Length': String(length),
      ...(partial ? { 'Content-Range': `bytes ${selected.start}-${selected.end}/${size}` } : {}),
    };
    response.writeHead(partial ? 206 : 200, { 'Access-Control-Allow-Origin': origin, ...headers });
    if (request.method === 'HEAD' || size === 0) return response.end();
    const stream = createFileStream(file, selected);
    stream.once('error', () => response.destroy());
    response.once('close', () => stream.destroy());
    stream.pipe(response);
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const server = createArtifactServer();
  server.listen(port, '127.0.0.1', () => {
    const address = server.address();
    const actualPort = typeof address === 'object' && address ? address.port : port;
    process.stdout.write(`MiniMax artifact server: ${defaultRoot}\nMiniMax artifact server port: ${actualPort}\n`);
  });
}
