import { describe, expect, it } from 'vitest';
import { parseModelManifest } from '../../../src/runtime/model/manifest';

const sha = 'a'.repeat(64);
const manifest = {
  schemaVersion: 1,
  graph: { path: 'graphs/global.onnx', bytes: 16, sha256: sha, externalData: [{ path: 'graphs/weights.bin', bytes: 8, sha256: sha, onnxLocation: 'weights.bin' }], gpuOutputs: ['hidden'] },
  reducedHead: { path: 'graphs/head.onnx', bytes: 16, sha256: sha, externalData: [], gpuOutputs: ['semantic_logits'] },
  embedding: { rows: 4, columns: 4, rowBytes: 8, shards: [{ path: 'embedding/0.fp16', bytes: 32, sha256: sha, rowStart: 0, rowCount: 4 }] },
  kvPairs: [{ pastInput: 'past.0', presentOutput: 'present.0' }],
};

describe('parseModelManifest', () => {
  it('parses both graph artifacts and their GPU output contracts', () => {
    const parsed = parseModelManifest(manifest);
    expect(parsed.graph.path).toBe('graphs/global.onnx');
    expect(parsed.reducedHead.gpuOutputs).toEqual(['semantic_logits']);
  });

  it('rejects a traversal artifact path', () => {
    expect(() => parseModelManifest({ ...manifest, graph: { ...manifest.graph, path: '../global.onnx' } })).toThrow('path');
  });

  it('rejects an embedding range that does not cover its declared rows', () => {
    expect(() => parseModelManifest({ ...manifest, embedding: { ...manifest.embedding, shards: [{ ...manifest.embedding.shards[0], rowCount: 3 }] } })).toThrow('embedding');
  });
});
