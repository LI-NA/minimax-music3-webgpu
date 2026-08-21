import { describe, expect, it, vi } from 'vitest';

import { readGpuFp16Bits } from '../../../src/runtime/model/fp16-readback';

describe('GPU FP16 readback', () => {
  it('uses the ORT downloader and preserves native Float16Array bits exactly', async () => {
    const expected = new Uint16Array([0x3c00, 0xbc00, 0x3555, 0x0001]);
    const decoded = new Float16Array(expected.buffer.slice(0));
    const getData = vi.fn(async () => decoded);

    const result = await readGpuFp16Bits({
      type: 'float16',
      location: 'gpu-buffer',
      dims: [1, 2, 2],
      getData,
    } as never, [1, 2, 2], 'fixture');

    expect(getData).toHaveBeenCalledOnce();
    expect(result).toBeInstanceOf(Uint16Array);
    expect(Array.from(result)).toEqual(Array.from(expected));
    expect(result.buffer).not.toBe(decoded.buffer);
  });

  it('clones legacy Uint16Array data after the ORT downloader completes', async () => {
    const downloaded = new Uint16Array([0x0000, 0x3c00]);
    const result = await readGpuFp16Bits({
      type: 'float16',
      location: 'gpu-buffer',
      dims: [1, 2],
      getData: async () => downloaded,
    } as never, [1, 2], 'fixture');

    downloaded.fill(0);
    expect(Array.from(result)).toEqual([0x0000, 0x3c00]);
  });

  it('rejects unrelated downloaded storage types', async () => {
    await expect(readGpuFp16Bits({
      type: 'float16',
      location: 'gpu-buffer',
      dims: [1],
      getData: async () => new Float32Array([1]),
    } as never, [1], 'fixture')).rejects.toThrow('fixture downloader did not return FP16 storage');
  });
});
