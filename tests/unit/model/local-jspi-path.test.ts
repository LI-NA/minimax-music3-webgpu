import { describe, expect, it } from 'vitest';
import { localJspiAssetName, localJspiWasmPaths } from '../../../src/runtime/model/local-jspi-path';

describe('localJspiAssetName', () => {
  it('matches an allowlisted JSPI module when Vite appends an import query', () => {
    expect(localJspiAssetName('/ort/ort-wasm-simd-threaded.jspi.mjs?import')).toBe('ort-wasm-simd-threaded.jspi.mjs');
  });

  it('does not claim non-ORT requests', () => {
    expect(localJspiAssetName('/ort/unknown.wasm')).toBeUndefined();
  });

  it('uses patch-versioned absolute URLs for both JSPI runtime files', () => {
    expect(localJspiWasmPaths('https://example.test', '/')).toEqual({
      mjs: 'https://example.test/ort/ort-wasm-simd-threaded.jspi.mjs?v=0569a267',
      wasm: 'https://example.test/ort/ort-wasm-simd-threaded.jspi.wasm?v=0569a267',
    });
  });

  it('keeps the runtime files under a deployment base path', () => {
    expect(localJspiWasmPaths('https://example.test', '/minimax-music3-webgpu/')).toEqual({
      mjs: 'https://example.test/minimax-music3-webgpu/ort/ort-wasm-simd-threaded.jspi.mjs?v=0569a267',
      wasm: 'https://example.test/minimax-music3-webgpu/ort/ort-wasm-simd-threaded.jspi.wasm?v=0569a267',
    });
  });
});
