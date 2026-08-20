import { describe, expect, it } from 'vitest';

import { encodeStereoPcm16Wav } from '../../../src/runtime/audio/wav';

describe('encodeStereoPcm16Wav', () => {
  it('writes the exact fixed five-second stereo container size and header', () => {
    const samples = 220_160;
    const wav = encodeStereoPcm16Wav([
      new Float32Array(samples),
      new Float32Array(samples),
    ]);
    const view = new DataView(wav);

    expect(wav.byteLength).toBe(880_684);
    expect(ascii(view, 0, 4)).toBe('RIFF');
    expect(view.getUint32(4, true)).toBe(880_676);
    expect(ascii(view, 8, 4)).toBe('WAVE');
    expect(ascii(view, 12, 4)).toBe('fmt ');
    expect(view.getUint16(20, true)).toBe(1);
    expect(view.getUint16(22, true)).toBe(2);
    expect(view.getUint32(24, true)).toBe(44_100);
    expect(view.getUint32(28, true)).toBe(176_400);
    expect(view.getUint16(32, true)).toBe(4);
    expect(view.getUint16(34, true)).toBe(16);
    expect(ascii(view, 36, 4)).toBe('data');
    expect(view.getUint32(40, true)).toBe(880_640);
  });

  it('clamps, scales, and interleaves signed PCM16 samples', () => {
    const wav = encodeStereoPcm16Wav([
      Float32Array.of(-2, -0.5, 1),
      Float32Array.of(2, 0.5, 0),
    ]);
    const view = new DataView(wav);

    expect(Array.from({ length: 6 }, (_, index) => view.getInt16(44 + index * 2, true))).toEqual([
      -32_768,
      32_767,
      -16_384,
      16_384,
      32_767,
      0,
    ]);
  });

  it('rejects unequal channels and non-finite samples', () => {
    expect(() =>
      encodeStereoPcm16Wav([Float32Array.of(0), Float32Array.of(0, 1)]),
    ).toThrow('channel lengths must match');
    expect(() =>
      encodeStereoPcm16Wav([Float32Array.of(Number.NaN), Float32Array.of(0)]),
    ).toThrow('samples must be finite');
  });
});

function ascii(view: DataView, offset: number, length: number): string {
  return String.fromCharCode(
    ...Array.from({ length }, (_, index) => view.getUint8(offset + index)),
  );
}
