import { existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, unlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  analyzeCanonicalPcm16Wav,
  assertMatchingWavSha256,
  assertFreshCaptureLayout,
  resolveCaptureLayout,
} from '../../browser/reference-capture-helpers';

function wav(frames: readonly (readonly [number, number])[]) {
  const dataBytes = frames.length * 4;
  const bytes = new Uint8Array(44 + dataBytes);
  const view = new DataView(bytes.buffer);
  const text = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index++) bytes[offset + index] = value.charCodeAt(index);
  };
  text(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  text(8, 'WAVEfmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, 44_100, true);
  view.setUint32(28, 176_400, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, dataBytes, true);
  frames.forEach(([left, right], index) => {
    view.setInt16(44 + index * 4, left, true);
    view.setInt16(46 + index * 4, right, true);
  });
  return bytes;
}

describe('reference capture paths', () => {
  it('resolves a new capture, case, and profile only below their artifact roots', () => {
    const root = path.resolve('synthetic-checkout');
    const layout = resolveCaptureLayout(root, 'ten-seconds-20260821');

    expect(layout.captureDirectory).toBe(path.join(root, 'artifacts', 'reference', 'captures', 'ten-seconds-20260821'));
    expect(layout.caseDirectory).toBe(path.join(root, 'artifacts', 'reference', 'cases', 'ten-seconds-20260821'));
    expect(layout.profile).toBe(path.join(root, 'artifacts', 'browser-profiles', 'variable-duration', 'task11'));
    expect(() => assertFreshCaptureLayout(layout, () => false)).not.toThrow();
  });

  it.each(['../escape', 'Uppercase', '', 'a'.repeat(65)])('rejects unsafe capture id %j', (id) => {
    expect(() => resolveCaptureLayout(path.resolve('synthetic-checkout'), id)).toThrow('capture id');
  });

  it('rejects an external profile and any existing capture or case', () => {
    const root = path.resolve('synthetic-checkout');
    expect(() => resolveCaptureLayout(root, 'valid', path.resolve('outside-profile'))).toThrow('profile');

    const layout = resolveCaptureLayout(root, 'valid');
    expect(() => assertFreshCaptureLayout(layout, (candidate) => candidate === layout.captureDirectory)).toThrow(
      'already exists',
    );
    expect(() => assertFreshCaptureLayout(layout, (candidate) => candidate === layout.caseDirectory)).toThrow(
      'already exists',
    );
  });

  it('rejects a profile below an existing link that escapes its allowed root', ({ skip }) => {
    const sandbox = mkdtempSync(path.join(tmpdir(), 'minimax-reference-paths-'));
    const checkout = path.join(sandbox, 'checkout');
    const profileRoot = path.join(checkout, 'artifacts', 'browser-profiles');
    const outside = path.join(sandbox, 'outside');
    const escape = path.join(profileRoot, 'escape');
    let linked = false;

    try {
      mkdirSync(profileRoot, { recursive: true });
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

      expect(() => resolveCaptureLayout(checkout, 'valid', path.join(escape, 'profile'))).toThrow('profile');
    } finally {
      if (linked && existsSync(escape)) unlinkSync(escape);
      rmSync(sandbox, { recursive: true, force: true });
    }
  });

  it('requires an explicit worktree profile for linked checkouts', () => {
    const root = path.resolve('synthetic-checkout');
    expect(() => resolveCaptureLayout(root, 'valid', undefined, true)).toThrow('explicit');
    const profile = path.join(root, 'artifacts', 'worktree-profiles', 'feature-name', 'reference');
    const layout = resolveCaptureLayout(root, 'valid', profile, true);
    expect(layout.profileRoot).toBe(path.join(root, 'artifacts', 'worktree-profiles'));
    expect(layout.profile).toBe(profile);
    expect(() =>
      resolveCaptureLayout(
        root,
        'valid',
        path.join(root, 'artifacts', 'browser-profiles', 'variable-duration', 'task11'),
        true,
      ),
    ).toThrow('worktree-profiles');
  });
});

describe('reference capture WAV inspection', () => {
  it('requires both fixed runs to produce the same WAV bytes', () => {
    expect(() => assertMatchingWavSha256('a'.repeat(64), 'a'.repeat(64))).not.toThrow();
    expect(() => assertMatchingWavSha256('a'.repeat(64), 'b'.repeat(64))).toThrow('not byte reproducible');
  });

  it('reports canonical stereo structure and simple health evidence', () => {
    expect(
      analyzeCanonicalPcm16Wav(
        wav([
          [1, 2],
          [2, 4],
          [3, 6],
        ]),
      ),
    ).toEqual({
      structure: {
        riff: 'RIFF',
        wave: 'WAVE',
        format: 1,
        channels: 2,
        sampleRate: 44_100,
        byteRate: 176_400,
        blockAlign: 4,
        bitsPerSample: 16,
        riffSize: 48,
        dataBytes: 12,
        samplesPerChannel: 3,
        wavBytes: 56,
      },
      health: {
        stereoDiffers: true,
        longestConstantFrameRun: 1,
        finalSecondDelta: 6,
      },
    });
  });

  it('rejects extra data and a non-canonical sample rate', () => {
    const extra = new Uint8Array([...wav([[1, 2]]), 0]);
    expect(() => analyzeCanonicalPcm16Wav(extra)).toThrow('canonical');
    const wrongRate = wav([[1, 2]]);
    new DataView(wrongRate.buffer).setUint32(24, 48_000, true);
    expect(() => analyzeCanonicalPcm16Wav(wrongRate)).toThrow('canonical');
  });
});
