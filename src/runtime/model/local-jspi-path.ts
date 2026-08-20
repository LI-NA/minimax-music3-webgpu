const localJspiFiles = new Set(['ort-wasm-simd-threaded.jspi.wasm', 'ort-wasm-simd-threaded.jspi.mjs']);

export function localJspiAssetName(url: string | undefined): string | undefined {
  if (!url) return undefined;
  const name = new URL(url, 'http://localhost').pathname.split('/').at(-1);
  return name && localJspiFiles.has(name) ? name : undefined;
}

export const localJspiFilesList = [...localJspiFiles];
