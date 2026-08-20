import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { resolve, sep } from 'node:path';

const origin = 'http://127.0.0.1:5173';
const port = Number(process.env.MINIMAX_ARTIFACT_PORT ?? 5174);
const requestedRelease = process.env.MINIMAX_RELEASE;
const releaseName = ['global-one-layer', 'rvq', 'condition', 'flow'].includes(requestedRelease) ? requestedRelease : 'global';
const root = resolve(process.env.MINIMAX_RELEASE_ROOT ?? `artifacts/release/${releaseName}`);

function reply(response, status, headers = {}) {
  response.writeHead(status, { 'Access-Control-Allow-Origin': origin, ...headers });
  response.end();
}

function artifactPath(pathname) {
  const relative = decodeURIComponent(pathname).replace(/^\/+/, '');
  const file = resolve(root, relative);
  return file === root || file.startsWith(`${root}${sep}`) ? file : undefined;
}

function range(value, size) {
  if (!value?.startsWith('bytes=') || value.includes(',')) return undefined;
  const match = /^bytes=(\d*)-(\d*)$/.exec(value);
  if (!match) return undefined;
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

createServer((request, response) => {
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
  if (request.method === 'HEAD') return response.end();
  createReadStream(file, selected).pipe(response);
}).listen(port, '127.0.0.1', () => {
  process.stdout.write(`MiniMax artifact server: ${root}\n`);
});
