import { readFileSync } from 'node:fs';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig, type Plugin } from 'vite';
import { localJspiAssetName, localJspiFilesList } from './src/runtime/model/local-jspi-path.ts';
import { patchOrtWebGpuConvTransposeCoordinates } from './src/runtime/model/ort-wasm-patch.ts';
import { APP_VERSION, workingSourceRevision } from './tools/app-revision.ts';
import { createArtifactMiddleware } from './tools/artifact-middleware.ts';

// GitHub Pages project sites live under `/<repo>/`. Everything that builds an absolute runtime
// URL derives it from `import.meta.env.BASE_URL`, so the deployment shape is a build input.
const base = process.env.MINIMAX_BASE ?? '/';

export const appBuildDefines = {
  __MINIMAX_APP_VERSION__: JSON.stringify(APP_VERSION),
  __MINIMAX_APP_REVISION__: JSON.stringify(workingSourceRevision()),
};

const readLocalJspiAsset = (name: string) => {
  const source = readFileSync(new URL(`./node_modules/onnxruntime-web/dist/${name}`, import.meta.url));
  return name.endsWith('.wasm') ? patchOrtWebGpuConvTransposeCoordinates(source) : source;
};

function localJspiWasm(): Plugin {
  return {
    name: 'local-jspi-wasm',
    configureServer(server) {
      server.middlewares.use(`${base}ort/`, (request, response, next) => {
        const name = localJspiAssetName(request.url);
        if (!name) return next();
        response.setHeader('Content-Type', name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
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

/**
 * Serves converted releases same-origin during development. Registering only `configureServer`
 * keeps the multi-gigabyte release tree out of `dist/`, and same-origin means no CORS to
 * misconfigure.
 */
function localArtifacts(): Plugin {
  return {
    name: 'local-artifacts',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(`${base}artifacts`, createArtifactMiddleware());
    },
  };
}

export default defineConfig({
  base,
  define: appBuildDefines,
  // Lazy dependency discovery would re-optimize and reload the page when the
  // dynamically imported ONNX Runtime entry loads mid-session, which aborts the
  // long headed browser gates. Every dev-server dependency must be listed here.
  optimizeDeps: {
    noDiscovery: true,
    include: [
      '@huggingface/tokenizers',
      '@noble/hashes/sha2.js',
      '@noble/hashes/utils.js',
      'onnxruntime-web/jspi',
      'react',
      'react-dom/client',
      'react/jsx-dev-runtime',
    ],
  },
  plugins: [tailwindcss(), localJspiWasm(), localArtifacts()],
});
