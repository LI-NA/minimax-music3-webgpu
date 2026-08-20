import { describe, expect, it } from 'vitest';
import { localJspiAssetName } from '../../../src/runtime/model/local-jspi-path';

describe('localJspiAssetName', () => {
  it('matches an allowlisted JSPI module when Vite appends an import query', () => {
    expect(localJspiAssetName('/ort/ort-wasm-simd-threaded.jspi.mjs?import')).toBe(
      'ort-wasm-simd-threaded.jspi.mjs',
    );
  });

  it('does not claim non-ORT requests', () => {
    expect(localJspiAssetName('/ort/unknown.wasm')).toBeUndefined();
  });
});
