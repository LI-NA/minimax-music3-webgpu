import { createReadStream, existsSync, realpathSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { resolve, sep } from 'node:path';
import type { Readable } from 'node:stream';

export type FileStreamFactory = (path: string, options: { start: number; end: number }) => Readable;

export interface ArtifactMiddlewareOptions {
  releaseRoot?: string;
  createFileStream?: FileStreamFactory;
}

interface ByteRange {
  start: number;
  end: number;
}

function parseRange(value: string, size: number): ByteRange | undefined {
  if (!value.startsWith('bytes=') || value.includes(',')) return undefined;
  const match = /^bytes=(\d*)-(\d*)$/.exec(value);
  if (!match) return undefined;
  if (size === 0) return undefined;
  const start = match[1] === '' ? undefined : Number(match[1]);
  const end = match[2] === '' ? undefined : Number(match[2]);
  if ((start !== undefined && !Number.isSafeInteger(start)) || (end !== undefined && !Number.isSafeInteger(end)))
    return undefined;
  if (start === undefined) {
    if (end === undefined || end < 1) return undefined;
    return { start: Math.max(0, size - end), end: size - 1 };
  }
  if (start >= size || (end !== undefined && (end < start || end >= size))) return undefined;
  return { start, end: end ?? size - 1 };
}

/**
 * Serves release artifacts with byte ranges from a single root, containment-checked against
 * symlinks and traversal. The release is the first path segment, so one mount replaces the
 * per-release artifact servers and their ports.
 */
export function createArtifactMiddleware({
  releaseRoot = 'artifacts/release',
  createFileStream = createReadStream,
}: ArtifactMiddlewareOptions = {}) {
  const root = resolve(releaseRoot);
  let resolvedRoot: string | undefined;

  // The release root is generated output and may not exist yet, so it is resolved per request
  // until it does. A missing directory must not stop the dev server from starting.
  const canonicalRoot = () => {
    if (resolvedRoot !== undefined) return resolvedRoot;
    try {
      resolvedRoot = realpathSync(root);
    } catch {
      return undefined;
    }
    return resolvedRoot;
  };

  const artifactPath = (pathname: string) => {
    const base = canonicalRoot();
    if (base === undefined) return undefined;
    const relative = decodeURIComponent(pathname).replace(/^\/+/, '');
    const file = resolve(root, relative);
    if (!(file === root || file.startsWith(`${root}${sep}`)) || !existsSync(file)) return undefined;
    try {
      const canonicalFile = realpathSync(file);
      return canonicalFile === base || canonicalFile.startsWith(`${base}${sep}`) ? canonicalFile : undefined;
    } catch {
      return undefined;
    }
  };

  return function artifactMiddleware(request: IncomingMessage, response: ServerResponse, next: () => void) {
    if (!request.url) return next();
    if (!['GET', 'HEAD'].includes(request.method ?? '')) return next();

    let file: string | undefined;
    try {
      file = artifactPath(new URL(request.url, 'http://localhost').pathname);
    } catch {
      response.writeHead(400);
      return response.end();
    }
    if (!file || !existsSync(file) || !statSync(file).isFile()) {
      response.writeHead(404);
      return response.end();
    }

    const size = statSync(file).size;
    const requested = request.headers.range;
    const selected = requested ? parseRange(requested, size) : { start: 0, end: size - 1 };
    if (!selected) {
      response.writeHead(416, { 'Accept-Ranges': 'bytes', 'Content-Range': `bytes */${size}` });
      return response.end();
    }

    const length = selected.end - selected.start + 1;
    const partial = Boolean(requested);
    response.writeHead(partial ? 206 : 200, {
      'Accept-Ranges': 'bytes',
      'Content-Length': String(length),
      ...(partial ? { 'Content-Range': `bytes ${selected.start}-${selected.end}/${size}` } : {}),
    });
    if (request.method === 'HEAD' || size === 0) return response.end();

    const stream = createFileStream(file, selected);
    stream.once('error', () => response.destroy());
    response.once('close', () => stream.destroy());
    stream.pipe(response);
  };
}
