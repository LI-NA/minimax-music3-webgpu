import { describe, expect, it } from 'vitest';
import {
  parseConditionManifest,
  parseFlowManifest,
  parseModelManifest,
  parseMusicManifest,
  parseRvqStageManifest,
  parseVocoderManifest,
} from '../../../src/runtime/model/manifest';

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
    gpuOutputs: ['last_state'],
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
    expect(parsed.reducedHead.gpuOutputs).toEqual(['last_state']);
    expect(parsed.webgpu.requiredLimits).toEqual({ maxStorageBufferBindingSize: 128 });
  });

  it('requires the gathered two-lane last state to remain GPU-resident', () => {
    expect(() =>
      parseModelManifest({
        ...manifest,
        reducedHead: { ...manifest.reducedHead, gpuOutputs: ['semantic_logits'] },
      }),
    ).toThrow('last_state');
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

describe('parseConditionManifest', () => {
  const condition = {
    schemaVersion: 1,
    conditionEncoder: { ...manifest.graph, gpuOutputs: ['condition'] },
    webgpu: manifest.webgpu,
  };

  it('parses the standalone fixed condition graph contract', () => {
    const parsed = parseConditionManifest(condition);
    expect(parsed.conditionEncoder.path).toBe('graphs/global.onnx');
    expect(parsed.conditionEncoder.gpuOutputs).toEqual(['condition']);
  });

  it('requires the condition output to stay GPU-resident', () => {
    expect(() =>
      parseConditionManifest({
        ...condition,
        conditionEncoder: { ...condition.conditionEncoder, gpuOutputs: [] },
      }),
    ).toThrow('condition');
  });
});

describe('parseFlowManifest', () => {
  const flow = {
    schemaVersion: 1,
    flow: { ...manifest.graph, gpuOutputs: ['next_latents'] },
    slice: {
      semanticFrames: 125,
      latentLength: 430,
      flowSteps: 30,
      flowGuidance: 1.7,
    },
    quantization: {
      bits: 4,
      blockSize: 128,
      accuracyLevel: 4,
      symmetric: true,
    },
    webgpu: {
      requiredFeatures: ['shader-f16'],
      requiredLimits: {
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxStorageBuffersPerShaderStage: 9,
      },
    },
  };

  it('parses the exact fixed q4 flow slice and GPU output contract', () => {
    const parsed = parseFlowManifest(flow);

    expect(parsed.flow.gpuOutputs).toEqual(['next_latents']);
    expect(parsed.slice).toEqual(flow.slice);
    expect(parsed.webgpu.requiredLimits.maxStorageBufferBindingSize).toBe(128 * 1024 * 1024);
    expect(parsed.webgpu.requiredLimits.maxStorageBuffersPerShaderStage).toBe(9);
  });

  it('rejects a changed slice, q4 contract, GPU output, or insufficient buffer limits', () => {
    expect(() => parseFlowManifest({ ...flow, slice: { ...flow.slice, latentLength: 431 } }))
      .toThrow('slice');
    expect(() => parseFlowManifest({
      ...flow,
      quantization: { ...flow.quantization, blockSize: 64 },
    })).toThrow('quantization');
    expect(() => parseFlowManifest({
      ...flow,
      flow: { ...flow.flow, gpuOutputs: [] },
    })).toThrow('next_latents');
    expect(() => parseFlowManifest({
      ...flow,
      webgpu: {
        ...flow.webgpu,
        requiredLimits: { ...flow.webgpu.requiredLimits, maxStorageBuffersPerShaderStage: 8 },
      },
    })).toThrow('requiredLimits');
  });
});

describe('parseVocoderManifest', () => {
  const vocoder = {
    schemaVersion: 1,
    vocoder: { ...manifest.graph, gpuOutputs: [] },
    slice: {
      latentChannels: 128,
      latentLength: 430,
      outputSamples: 220_160,
      sampleRate: 44_100,
      channels: 2,
    },
    precision: {
      convolution: 'float16',
      fp32Snakes: ['blocks.0.snake1', 'blocks.1.snake1'],
    },
    webgpu: {
      requiredFeatures: ['shader-f16'],
      requiredLimits: { maxStorageBufferBindingSize: 128 * 1024 * 1024 },
    },
  };

  it('parses the exact fixed mixed-precision vocoder contract', () => {
    const parsed = parseVocoderManifest(vocoder);

    expect(parsed.vocoder.gpuOutputs).toEqual([]);
    expect(parsed.slice).toEqual(vocoder.slice);
    expect(parsed.precision).toEqual(vocoder.precision);
    expect(parsed.webgpu.requiredLimits.maxStorageBufferBindingSize).toBe(128 * 1024 * 1024);
  });

  it('rejects changed audio, precision, output location, or buffer contracts', () => {
    expect(() => parseVocoderManifest({
      ...vocoder,
      slice: { ...vocoder.slice, outputSamples: 220_161 },
    })).toThrow('slice');
    expect(() => parseVocoderManifest({
      ...vocoder,
      precision: { ...vocoder.precision, fp32Snakes: ['blocks.0.snake1'] },
    })).toThrow('precision');
    expect(() => parseVocoderManifest({
      ...vocoder,
      vocoder: { ...vocoder.vocoder, gpuOutputs: ['waveform'] },
    })).toThrow('CPU output');
    expect(() => parseVocoderManifest({
      ...vocoder,
      webgpu: {
        ...vocoder.webgpu,
        requiredLimits: { maxStorageBufferBindingSize: 64 * 1024 * 1024 },
      },
    })).toThrow('requiredLimits');
  });
});

describe('parseMusicManifest', () => {
  const combined = {
    ...manifest,
    model: {
      id: 'MiniMaxAI/MiniMax-Music3',
      revision: 'fbdf52fbaaca799592917417eb05f1899f1255ec',
      diffusersRevision: '3681e65996b4d2589219720101a6acbfd25073f8',
    },
    rvqDepth: { ...manifest.graph, gpuOutputs: ['depth_hidden'] },
    feedback: { ...manifest.reducedHead, gpuOutputs: ['inputs_embeds'] },
    rvqEmbedding: manifest.embedding,
    conditionEncoder: { ...manifest.graph, gpuOutputs: ['condition'] },
    flow: { ...manifest.graph, gpuOutputs: ['next_latents'] },
    vocoder: { ...manifest.graph, gpuOutputs: [] },
    tokenizerFiles: [{ path: 'global/tokenizer.json', bytes: 2, sha256: sha }],
    licenseFile: { path: 'global/LICENSE', bytes: 7, sha256: sha },
    slice: {
      semanticFrames: 125,
      latentLength: 430,
      outputSamples: 220_160,
      sampleRate: 44_100,
      channels: 2,
      flowSteps: 30,
      globalGuidance: 1.5,
      flowGuidance: 1.7,
    },
    quantization: { bits: 4, blockSize: 128, accuracyLevel: 4, symmetric: true },
    precision: {
      convolution: 'float16',
      fp32Snakes: ['blocks.0.snake1', 'blocks.1.snake1'],
    },
    webgpu: {
      requiredFeatures: ['shader-f16'],
      requiredLimits: {
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxStorageBuffersPerShaderStage: 9,
      },
    },
  };

  it('parses every fixed stage and release support artifact from one namespace', () => {
    const parsed = parseMusicManifest(combined);

    expect(parsed.model.revision).toBe(combined.model.revision);
    expect(parsed.conditionEncoder.gpuOutputs).toEqual(['condition']);
    expect(parsed.flow.gpuOutputs).toEqual(['next_latents']);
    expect(parsed.vocoder.gpuOutputs).toEqual([]);
    expect(parsed.slice.outputSamples).toBe(220_160);
    expect(parsed.tokenizerFiles[0].path).toBe('global/tokenizer.json');
  });

  it('rejects a substitute checkpoint or changed fixed five-second shape', () => {
    expect(() => parseMusicManifest({ ...combined, model: { ...combined.model, revision: 'other' } }))
      .toThrow('revision');
    expect(() => parseMusicManifest({
      ...combined,
      slice: { ...combined.slice, semanticFrames: 124 },
    })).toThrow('slice');
  });
});
