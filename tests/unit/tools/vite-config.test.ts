import { describe, expect, it } from 'vitest';
import config from '../../../vite.config';

describe('Vite configuration', () => {
  it('prebundles the dynamically imported ONNX Runtime entry point', () => {
    if (typeof config === 'function') throw new Error('expected an object Vite configuration');
    expect(config.optimizeDeps).toMatchObject({
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
    });
  });
});
