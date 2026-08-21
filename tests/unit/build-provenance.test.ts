import { readdirSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { appRevisionInputs } from '../../tools/app-revision';

function sourceFiles(directory: URL, prefix = 'src'): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = `${prefix}/${entry.name}`;
    if (entry.isDirectory()) return sourceFiles(new URL(`${entry.name}/`, directory), path);
    return /(?:\.tsx?|\.css)$/.test(entry.name) ? [path] : [];
  });
}

describe('working-source application revision', () => {
  it('deterministically includes every application source and the HTML entry point', () => {
    const expectedSources = sourceFiles(new URL('../../src/', import.meta.url)).sort();
    const revisionSources = appRevisionInputs.filter((path) => path.startsWith('src/'));

    expect(appRevisionInputs).toEqual([...appRevisionInputs].sort());
    expect(revisionSources).toEqual(expectedSources);
    expect(appRevisionInputs).toContain('index.html');
    expect(appRevisionInputs).toContain('tests/fixtures/prompt-contract.json');
    expect(appRevisionInputs).toContain('tools/app-revision.ts');
    expect(appRevisionInputs).toContain('tools/reference/fixed_case.json');
  });
});
