import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import { patchOrtWebGpuConvTransposeCoordinates } from '../../../src/runtime/model/ort-wasm-patch';

const brokenCast = new TextEncoder().encode('dy_element_t(');
const fixedCast = new TextEncoder().encode('f32         (');

function count(source: Uint8Array, pattern: Uint8Array) {
  let matches = 0;
  for (let offset = 0; offset <= source.length - pattern.length; offset++) {
    if (pattern.every((value, index) => source[offset + index] === value)) matches += 1;
  }
  return matches;
}

describe('pinned ORT WebGPU runtime patch', () => {
  it('uses FP32 for all eight ConvTranspose coordinate casts without changing WASM length', () => {
    const source = readFileSync('node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jspi.wasm');

    const patched = patchOrtWebGpuConvTransposeCoordinates(source);

    expect(count(source, brokenCast)).toBe(8);
    expect(count(patched, brokenCast)).toBe(0);
    expect(count(patched, fixedCast)).toBe(8);
    expect(patched.byteLength).toBe(source.byteLength);
    expect(createHash('sha256').update(patched).digest('hex')).toBe(
      '0569a267c57da3947fefc95934a7eee1426188cba11997be556515d482347534',
    );
  });

  it('fails closed when the pinned source bytes differ', () => {
    const source = Uint8Array.from(readFileSync('node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jspi.wasm'));
    source[0] ^= 1;

    expect(() => patchOrtWebGpuConvTransposeCoordinates(source)).toThrow('differs from the pinned runtime');
  });
});
