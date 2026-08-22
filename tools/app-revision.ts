import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';

const checkoutRoot = new URL('../', import.meta.url);

export const APP_VERSION = (
  JSON.parse(readFileSync(new URL('package.json', checkoutRoot), 'utf8')) as { version: string }
).version;

function applicationSources(directory: URL, prefix = 'src'): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = `${prefix}/${entry.name}`;
    if (entry.isDirectory()) return applicationSources(new URL(`${entry.name}/`, directory), path);
    return /(?:\.tsx?|\.css)$/.test(entry.name) ? [path] : [];
  });
}

export const appRevisionInputs = [
  'index.html',
  'package.json',
  'package-lock.json',
  ...applicationSources(new URL('src/', checkoutRoot)),
  'tests/fixtures/prompt-contract.json',
  'tools/app-revision.ts',
  'tools/reference/fixed_case.json',
  'vite.config.ts',
].sort();

export function workingSourceRevision() {
  const digest = createHash('sha256');
  for (const path of appRevisionInputs) {
    digest.update(path);
    digest.update('\0');
    digest.update(readFileSync(new URL(path, checkoutRoot)));
    digest.update('\0');
  }
  return digest.digest('hex');
}
