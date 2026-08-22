const localJspiFiles = new Set(['ort-wasm-simd-threaded.jspi.wasm', 'ort-wasm-simd-threaded.jspi.mjs']);
const patchVersion = '0569a267';

export function localJspiAssetName(url: string | undefined): string | undefined {
  if (!url) return undefined;
  const name = new URL(url, 'http://localhost').pathname.split('/').at(-1);
  return name && localJspiFiles.has(name) ? name : undefined;
}

export const localJspiFilesList = [...localJspiFiles];

export function localJspiWasmPaths(origin: string) {
  const path = (name: string) => new URL(`/ort/${name}?v=${patchVersion}`, origin).toString();
  return {
    mjs: path('ort-wasm-simd-threaded.jspi.mjs'),
    wasm: path('ort-wasm-simd-threaded.jspi.wasm'),
  };
}
