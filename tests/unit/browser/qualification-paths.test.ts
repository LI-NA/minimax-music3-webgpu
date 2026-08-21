import { existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, unlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

describe('qualification capture and Chrome profile paths', () => {
  it('requires a fresh nonempty child capture', async () => {
    const paths = await import('../../browser/qualification-paths').catch(() => undefined);
    expect(paths).toBeDefined();
    const checkout = path.resolve('synthetic-checkout');
    const expected = path.join(
      checkout,
      'artifacts',
      'diagnostics',
      'variable-duration',
      'run-01',
    );
    expect(paths?.resolveQualificationCapture(checkout, expected)).toBe(expected);
    expect(() => paths?.resolveQualificationCapture(
      checkout,
      path.join(checkout, 'artifacts', 'diagnostics', 'variable-duration'),
    )).toThrow('nonempty child');
    expect(() => paths?.resolveQualificationCapture(checkout, path.resolve('outside')))
      .toThrow('contained');
    expect(() => paths?.assertFreshQualificationCapture(expected, () => false)).not.toThrow();
    expect(() => paths?.assertFreshQualificationCapture(expected, () => true)).toThrow('already exists');
  });

  it('rejects a capture below an existing link that escapes its allowed root', async ({ skip }) => {
    const paths = await import('../../browser/qualification-paths').catch(() => undefined);
    expect(paths).toBeDefined();
    const sandbox = mkdtempSync(path.join(tmpdir(), 'minimax-qualification-paths-'));
    const checkout = path.join(sandbox, 'checkout');
    const captureRoot = path.join(
      checkout,
      'artifacts',
      'diagnostics',
      'variable-duration',
    );
    const outside = path.join(sandbox, 'outside');
    const escape = path.join(captureRoot, 'escape');
    let linked = false;

    try {
      mkdirSync(captureRoot, { recursive: true });
      mkdirSync(outside);
      try {
        symlinkSync(outside, escape, process.platform === 'win32' ? 'junction' : 'dir');
        linked = true;
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code === 'EPERM' || code === 'EACCES') {
          skip();
          return;
        }
        throw error;
      }

      expect(() => paths?.resolveQualificationCapture(
        checkout,
        path.join(escape, 'run-01'),
      )).toThrow('contained');
    } finally {
      if (linked && existsSync(escape)) unlinkSync(escape);
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('uses browser-profiles only in the primary checkout', async () => {
    const paths = await import('../../browser/qualification-paths').catch(() => undefined);
    expect(paths).toBeDefined();
    const checkout = path.resolve('synthetic-checkout');
    expect(paths?.resolveQualificationProfile(checkout, undefined, false)).toBe(path.join(
      checkout,
      'artifacts',
      'browser-profiles',
      'variable-duration',
      'task11',
    ));
    const explicit = path.join(checkout, 'artifacts', 'browser-profiles', 'local', 'gate');
    expect(paths?.resolveQualificationProfile(checkout, explicit, false)).toBe(explicit);
    expect(() => paths?.resolveQualificationProfile(
      checkout,
      path.join(checkout, 'artifacts', 'worktree-profiles', 'feature', 'gate'),
      false,
    )).toThrow('browser-profiles');
  });

  it('requires an explicit worktree-namespaced profile in linked worktrees', async () => {
    const paths = await import('../../browser/qualification-paths').catch(() => undefined);
    expect(paths).toBeDefined();
    const checkout = path.resolve('synthetic-checkout');
    expect(() => paths?.resolveQualificationProfile(checkout, undefined, true))
      .toThrow('explicit');
    const explicit = path.join(
      checkout,
      'artifacts',
      'worktree-profiles',
      'feature-name',
      'long-duration',
    );
    expect(paths?.resolveQualificationProfile(checkout, explicit, true)).toBe(explicit);
    expect(() => paths?.resolveQualificationProfile(
      checkout,
      path.join(checkout, 'artifacts', 'browser-profiles', 'variable-duration', 'task11'),
      true,
    )).toThrow('worktree-profiles');
    expect(() => paths?.resolveQualificationProfile(
      checkout,
      path.join(checkout, 'artifacts', 'worktree-profiles'),
      true,
    )).toThrow('nonempty child');
  });
});
