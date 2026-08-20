import { describe, expect, it } from 'vitest';
import { parseModelManifest, parseRvqStageManifest } from '../../../src/runtime/model/manifest';

const sha = 'a'.repeat(64);
const manifest = {
  schemaVersion: 1,
  graph: {
    path: 'graphs/global.onnx',
    bytes: 16,
    sha256: sha,
    externalData: [
      {
        path: 'graphs/weights.bin',
        bytes: 8,
        sha256: sha,
        onnxLocation: 'weights.bin',
      },
    ],
    gpuOutputs: ['hidden'],
  },
  reducedHead: {
    path: 'graphs/head.onnx',
    bytes: 16,
    sha256: sha,
    externalData: [],
    gpuOutputs: ['semantic_logits'],
  },
  embedding: {
    rows: 4,
    columns: 4,
    rowBytes: 8,
    shards: [
      {
        path: 'embedding/0.fp16',
        bytes: 32,
        sha256: sha,
        rowStart: 0,
        rowCount: 4,
      },
    ],
  },
  kvPairs: [{ pastInput: 'past.0', presentOutput: 'present.0' }],
  webgpu: {
    requiredFeatures: ['shader-f16'],
    requiredLimits: { maxStorageBufferBindingSize: 128 },
  },
};

describe('parseModelManifest', () => {
  it('parses both graph artifacts and their GPU output contracts', () => {
    const parsed = parseModelManifest(manifest);
    expect(parsed.graph.path).toBe('graphs/global.onnx');
    expect(parsed.reducedHead.gpuOutputs).toEqual(['semantic_logits']);
    expect(parsed.webgpu.requiredLimits).toEqual({ maxStorageBufferBindingSize: 128 });
  });

  it('requires the schema WebGPU feature contract', () => {
    expect(() =>
      parseModelManifest({ ...manifest, webgpu: { requiredFeatures: [], requiredLimits: {} } }),
    ).toThrow('requiredFeatures');
  });

  it('rejects a traversal artifact path', () => {
    expect(() =>
      parseModelManifest({
        ...manifest,
        graph: { ...manifest.graph, path: '../global.onnx' },
      }),
    ).toThrow('path');
  });

  it('rejects an embedding range that does not cover its declared rows', () => {
    expect(() =>
      parseModelManifest({
        ...manifest,
        embedding: {
          ...manifest.embedding,
          shards: [{ ...manifest.embedding.shards[0], rowCount: 3 }],
        },
      }),
    ).toThrow('embedding');
  });
});

describe('parseRvqStageManifest', () => {
  const rvq = {
    schemaVersion: 1,
    rvqDepth: { ...manifest.graph, gpuOutputs: ['depth_hidden'] },
    feedback: { ...manifest.reducedHead, gpuOutputs: ['inputs_embeds'] },
    rvqEmbedding: {
      ...manifest.embedding,
      rows: 7,
      shards: [{ ...manifest.embedding.shards[0], bytes: 56, rowCount: 7 }],
    },
    webgpu: manifest.webgpu,
  };

  it('parses the fixed depth, feedback, and residual row-table contract', () => {
    const parsed = parseRvqStageManifest(rvq);
    expect(parsed.rvqDepth.gpuOutputs).toEqual(['depth_hidden']);
    expect(parsed.feedback.gpuOutputs).toEqual(['inputs_embeds']);
    expect(parsed.rvqEmbedding.rows).toBe(7);
  });

  it('requires the GPU-resident hidden and feedback outputs', () => {
    expect(() => parseRvqStageManifest({ ...rvq, feedback: { ...rvq.feedback, gpuOutputs: [] } }))
      .toThrow('inputs_embeds');
  });
});
