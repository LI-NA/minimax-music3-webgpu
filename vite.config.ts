import { readFileSync } from 'node:fs';
import { defineConfig, type Plugin } from 'vite';
import { localJspiAssetName, localJspiFilesList } from './src/runtime/model/local-jspi-path.ts';
import { patchOrtWebGpuConvTransposeCoordinates } from './src/runtime/model/ort-wasm-patch.ts';

const readLocalJspiAsset = (name: string) => {
  const source = readFileSync(
    new URL(`./node_modules/onnxruntime-web/dist/${name}`, import.meta.url),
  );
  return name.endsWith('.wasm') ? patchOrtWebGpuConvTransposeCoordinates(source) : source;
};

function localJspiWasm(): Plugin {
  return {
    name: 'local-jspi-wasm',
    configureServer(server) {
      server.middlewares.use('/ort/', (request, response, next) => {
        const name = localJspiAssetName(request.url);
        if (!name) return next();
        response.setHeader(
          'Content-Type',
          name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript',
        );
        response.end(readLocalJspiAsset(name));
      });
    },
    generateBundle() {
      for (const name of localJspiFilesList)
        this.emitFile({
          type: 'asset',
          fileName: `ort/${name}`,
          source: readLocalJspiAsset(name),
        });
    },
  };
}

export default defineConfig({ plugins: [localJspiWasm()] });
