import { describe, expect, it, vi } from 'vitest';
import promptContract from '../../fixtures/prompt-contract.json';
import {
  areFiniteFp16,
  createFrameGenerator,
  readConditionalGpuFp16,
  type GenerateFrameOptions,
} from '../../../src/runtime/pipeline/rvq-generation';

const frameOptions = (
  maxFrames: number,
  seed = 7,
  overrides: Partial<GenerateFrameOptions> = {},
): GenerateFrameOptions => ({
  maxFrames,
  seed,
  promptTokenRows: {
    conditional: promptContract.conditional,
    unconditional: promptContract.unconditional,
  },
  guidance: 1.5,
  semanticTopK: 50,
  residualTopK: 50,
  temperature: 1,
  ...overrides,
});

class FakeTensor {
  location: 'cpu' | 'gpu-buffer';
  gpuBuffer = { size: 64 };
  disposed = false;
  disposeCalls = 0;
  constructor(
    public readonly type: string,
    public readonly data: Float32Array | Float16Array | Uint16Array | Int32Array,
    public readonly dims: readonly number[],
    location: 'cpu' | 'gpu-buffer' = 'cpu',
    public readonly conditional?: number,
  ) {
    this.location = location;
  }
  async getData() { return this.data; }
  dispose() {
    this.disposed = true;
    this.disposeCalls++;
  }
}

type FailureBoundary = 'prefill' | 'head' | 'head-partial' | 'depth' | 'feedback' | 'decoder';

function failureRuntime(boundary: FailureBoundary) {
  let decoderCalls = 0;
  const feeds: Partial<Record<FailureBoundary, Record<string, FakeTensor>>> = {};
  let cacheTensor: FakeTensor | undefined;
  let headOutputs: FakeTensor[] = [];
  const globalEmbeddingDispose = vi.fn();
  const rvqEmbeddingDispose = vi.fn();
  const generator = createFrameGenerator({
    ort: { Tensor: FakeTensor } as never,
    decoder: {
      run: async (input: Record<string, FakeTensor>) => {
        const call = decoderCalls++;
        if (call === 0) {
          feeds.prefill = input;
          if (boundary === 'prefill') throw new Error('prefill failure');
          cacheTensor = new FakeTensor('float16', new Uint16Array(), [2, 8, 40, 128], 'gpu-buffer');
          return {
            hidden_states: new FakeTensor('float16', new Uint16Array(), [2, 1, 4096], 'gpu-buffer'),
            present: cacheTensor,
          };
        }
        feeds.decoder = input;
        if (boundary === 'decoder') throw new Error('decoder failure');
        throw new Error('unexpected decoder call');
      },
    } as never,
    head: {
      run: async (input: Record<string, FakeTensor>) => {
        feeds.head = input;
        if (boundary === 'head') throw new Error('head failure');
        const semantic = new Float16Array(2 * 16_384).fill(-100);
        semantic[100] = semantic[16_384 + 100] = 10;
        const semanticLogits = new FakeTensor('float32', semantic, [2, 16_384]);
        const lastState = new FakeTensor('float16', new Uint16Array(), [2, 4096], 'gpu-buffer', 0);
        if (boundary === 'head-partial') {
          headOutputs = [semanticLogits, lastState];
          return { semantic_logits: semanticLogits, last_state: lastState };
        }
        return {
          semantic_logits: semanticLogits,
          end_logit: new FakeTensor('float16', new Float16Array([-100, -100]), [2, 1]),
          last_state: lastState,
        };
      },
    } as never,
    rvqDepth: {
      run: async (input: Record<string, FakeTensor>) => {
        feeds.depth = input;
        if (boundary === 'depth') throw new Error('depth failure');
        const logits = new Float16Array(2 * 7 * 1_024).fill(-100);
        logits[0] = logits[7 * 1_024] = 10;
        return {
          depth_hidden: new FakeTensor('float16', new Uint16Array(), [2, 4096], 'gpu-buffer', 1),
          depth_logits: new FakeTensor('float32', logits, [2, 7, 1_024]),
        };
      },
    } as never,
    feedback: {
      run: async (input: Record<string, FakeTensor>) => {
        feeds.feedback = input;
        if (boundary === 'feedback') throw new Error('feedback failure');
        return { inputs_embeds: new FakeTensor('float16', new Uint16Array(), [2, 1, 4096], 'gpu-buffer') };
      },
    } as never,
    globalEmbedding: {
      lookup: (ids: readonly number[]) => new Uint16Array(ids.length * 4096),
      dispose: globalEmbeddingDispose,
    },
    rvqEmbedding: {
      lookup: (ids: readonly number[]) => new Uint16Array(ids.length * 4096),
      dispose: rvqEmbeddingDispose,
    },
    embeddingColumns: 4096,
    kvPairs: [{ pastInput: 'past', presentOutput: 'present' }],
    readConditionalHidden: async () => new Uint16Array(4096),
  });
  return {
    generator,
    feeds,
    cache: () => cacheTensor,
    headOutputs: () => headOutputs,
    globalEmbeddingDispose,
    rvqEmbeddingDispose,
  };
}

function runtime(decisions: number, onFrameRetained?: (count: number) => void, endDecision?: number) {
  let decoderCalls = 0;
  let headCalls = 0;
  let depthCalls = 0;
  let feedbackCalls = 0;
  const depthFeeds: Record<string, FakeTensor>[] = [];
  const decoderFeeds: Record<string, FakeTensor>[] = [];
  const cacheLengths: number[] = [];
  const decoder = {
    run: async (feeds: Record<string, FakeTensor>) => {
      decoderFeeds.push(feeds);
      const cacheLength = (feeds.total_seq_len.data as Int32Array)[0];
      decoderCalls++;
      return {
        hidden_states: new FakeTensor('float16', new Uint16Array(), [2, 1, 4096], 'gpu-buffer'),
        present: new FakeTensor('float16', new Uint16Array(), [2, 8, cacheLength, 128], 'gpu-buffer'),
      };
    },
  };
  const head = {
    run: async () => {
      const decision = headCalls++;
      const semantic = new Float16Array(2 * 16_384).fill(-100);
      if (decision === endDecision) {
        for (let code = 100 + decision; code < 104 + decision; code++)
          semantic[code] = semantic[16_384 + code] = 90;
      } else {
        semantic[100 + decision] = semantic[16_384 + 100 + decision] = 10;
      }
      return {
        semantic_logits: new FakeTensor('float32', semantic, [2, 16_384]),
        end_logit: new FakeTensor(
          'float16',
          new Float16Array(decision === endDecision ? [100, 100] : [-100, -100]),
          [2, 1],
        ),
        last_state: new FakeTensor('float16', new Uint16Array(), [2, 4096], 'gpu-buffer', decision),
      };
    },
  };
  const rvqDepth = {
    run: async (feeds: Record<string, FakeTensor>) => {
      depthFeeds.push(feeds);
      const call = depthCalls++;
      const depth = call % 7;
      const logits = new Float16Array(2 * 7 * 1_024).fill(-100);
      logits[depth * 1_024 + depth] = 10;
      logits[7 * 1_024 + depth * 1_024 + depth] = 10;
      return {
        depth_hidden: new FakeTensor('float16', new Uint16Array(), [2, 4096], 'gpu-buffer', depth + 1),
        depth_logits: new FakeTensor('float32', logits, [2, 7, 1_024]),
      };
    },
  };
  const feedback = {
    run: async () => {
      feedbackCalls++;
      return { inputs_embeds: new FakeTensor('float16', new Uint16Array(), [2, 1, 4096], 'gpu-buffer') };
    },
  };
  const generator = createFrameGenerator({
    ort: { Tensor: FakeTensor } as never,
    decoder: decoder as never,
    head: head as never,
    rvqDepth: rvqDepth as never,
    feedback: feedback as never,
    globalEmbedding: {
      lookup: (ids: readonly number[]) => Uint16Array.from(ids.flatMap((id) => Array(4096).fill(id))),
      dispose() {},
    },
    rvqEmbedding: { lookup: (ids: readonly number[]) => Uint16Array.from(ids.flatMap((id) => Array(4096).fill(id))), dispose() {} },
    embeddingColumns: 4096,
    kvPairs: [{ pastInput: 'past', presentOutput: 'present' }],
    readConditionalHidden: async (tensor) =>
      new Uint16Array(4096).fill(0x3c00 + (tensor as unknown as FakeTensor).conditional!),
    onCacheLength: (length) => cacheLengths.push(length),
    onFrameRetained,
  });
  return {
    generator,
    counts: () => ({ decoderCalls, headCalls, depthCalls, feedbackCalls }),
    depthFeeds,
    decoderFeeds,
    cacheLengths,
    decisions,
  };
}

describe('RVQ frame generation', () => {
  it('disposes every prefill input once when decoder prefill rejects', async () => {
    const fixture = failureRuntime('prefill');

    await expect(fixture.generator.generateFrames(frameOptions(1))).rejects.toThrow('prefill failure');

    expect(Object.values(fixture.feeds.prefill!).map((tensor) => tensor.disposeCalls)).toEqual([1, 1, 1, 1]);
    expect(fixture.globalEmbeddingDispose).toHaveBeenCalledOnce();
    expect(fixture.rvqEmbeddingDispose).toHaveBeenCalledOnce();
  });

  it('disposes the head hidden input once and cleans the outer cache when head rejects', async () => {
    const fixture = failureRuntime('head');

    await expect(fixture.generator.generateFrames(frameOptions(1))).rejects.toThrow('head failure');

    expect(fixture.feeds.head!.hidden_states.disposeCalls).toBe(1);
    expect(fixture.cache()!.disposeCalls).toBe(1);
    expect(fixture.globalEmbeddingDispose).toHaveBeenCalledOnce();
    expect(fixture.rvqEmbeddingDispose).toHaveBeenCalledOnce();
  });

  it('disposes each partially returned head output once when output validation fails', async () => {
    const fixture = failureRuntime('head-partial');

    await expect(fixture.generator.generateFrames(frameOptions(1))).rejects.toThrow('reduced head outputs are incomplete');

    expect(fixture.feeds.head!.hidden_states.disposeCalls).toBe(1);
    expect(fixture.headOutputs().map((tensor) => tensor.disposeCalls)).toEqual([1, 1]);
    expect(fixture.cache()!.disposeCalls).toBe(1);
    expect(fixture.globalEmbeddingDispose).toHaveBeenCalledOnce();
    expect(fixture.rvqEmbeddingDispose).toHaveBeenCalledOnce();
  });

  it('disposes every depth input once and cleans the outer cache when RVQ depth rejects', async () => {
    const fixture = failureRuntime('depth');

    await expect(fixture.generator.generateFrames(frameOptions(1))).rejects.toThrow('depth failure');

    expect(Object.values(fixture.feeds.depth!).map((tensor) => tensor.disposeCalls)).toEqual([1, 1, 1, 1]);
    expect(fixture.cache()!.disposeCalls).toBe(1);
    expect(fixture.globalEmbeddingDispose).toHaveBeenCalledOnce();
    expect(fixture.rvqEmbeddingDispose).toHaveBeenCalledOnce();
  });

  it('disposes every feedback input once and cleans the outer cache when feedback rejects', async () => {
    const fixture = failureRuntime('feedback');

    await expect(fixture.generator.generateFrames(frameOptions(1))).rejects.toThrow('feedback failure');

    expect(Object.values(fixture.feeds.feedback!).map((tensor) => tensor.disposeCalls)).toEqual([1, 1]);
    expect(fixture.cache()!.disposeCalls).toBe(1);
    expect(fixture.globalEmbeddingDispose).toHaveBeenCalledOnce();
    expect(fixture.rvqEmbeddingDispose).toHaveBeenCalledOnce();
  });

  it('disposes every decoder input once and cleans the outer cache when incremental decode rejects', async () => {
    const fixture = failureRuntime('decoder');

    await expect(fixture.generator.generateFrames(frameOptions(1))).rejects.toThrow('decoder failure');

    expect(Object.values(fixture.feeds.decoder!).map((tensor) => tensor.disposeCalls)).toEqual([1, 1, 1, 1]);
    expect(fixture.cache()!.disposeCalls).toBe(1);
    expect(fixture.globalEmbeddingDispose).toHaveBeenCalledOnce();
    expect(fixture.rvqEmbeddingDispose).toHaveBeenCalledOnce();
  });

  it('discards warmup, emits two real frames, and keeps the exact eight conditional hidden groups', async () => {
    const fixture = runtime(2);
    const frames = await fixture.generator.generateFrames(frameOptions(2));

    expect(frames.map((frame) => [frame.semantic, ...frame.residual])).toEqual([
      [101, 0, 1, 2, 3, 4, 5, 6],
      [102, 0, 1, 2, 3, 4, 5, 6],
    ]);
    expect(fixture.counts()).toEqual({ decoderCalls: 3, headCalls: 3, depthCalls: 21, feedbackCalls: 2 });
    expect(fixture.cacheLengths).toEqual([40, 41, 42]);
    expect(frames[0].hiddenGroups).toHaveLength(8 * 4096);
    expect(frames[0].hiddenGroups).toBeInstanceOf(Uint16Array);
    expect(frames[0].hiddenGroups[0]).toBe(0x3c01);
    expect(frames[0].hiddenGroups[4096]).toBe(0x3c01);
    expect(frames[0].hiddenGroups[7 * 4096]).toBe(0x3c07);
    expect(frames.hiddenGroups.byteLength).toBe(131_072);
    expect(frames[0].hiddenGroups.buffer).toBe(frames.hiddenGroups.buffer);
    expect(frames[1].hiddenGroups.byteOffset).toBe(frames.hiddenGroups.byteOffset + 8 * 4096 * 2);
    expect(frames.termination).toBe('max-frames');
    expect(frames.plan).toMatchObject({
      retainedFrames: 2,
      termination: 'max-frames',
      semanticDecisions: 3,
      rvqCalls: 21,
      feedbackCalls: 2,
    });
  });

  it('samples ORT Float16Array CPU logits without treating their values as raw half bits', async () => {
    const fixture = runtime(2);
    const frames = await fixture.generator.generateFrames(frameOptions(2));

    expect(frames.map((frame) => frame.semantic)).toEqual([101, 102]);
  });

  it('runs 126 semantic decisions, 882 depth calls, and 125 feedback decodes for 125 retained frames', async () => {
    const fixture = runtime(125);
    const frames = await fixture.generator.generateFrames(frameOptions(125, 11));

    expect(frames).toHaveLength(125);
    expect(fixture.counts()).toEqual({ decoderCalls: 126, headCalls: 126, depthCalls: 882, feedbackCalls: 125 });
    expect(frames.reduce((bytes, frame) => bytes + frame.hiddenGroups.byteLength, 0)).toBe(8_192_000);
  });

  it('reports each retained frame after its hidden groups are ready', async () => {
    const retained: number[] = [];
    const fixture = runtime(2, (count) => retained.push(count));

    await fixture.generator.generateFrames(frameOptions(2, 11));

    expect(retained).toEqual([1, 2]);
  });

  it('uses caller prompt rows and their common length for prefill and cache bookkeeping', async () => {
    const fixture = runtime(1);
    const promptTokenRows = {
      conditional: [11, 12, 13],
      unconditional: [21, 22, 23],
    };

    await fixture.generator.generateFrames(frameOptions(1, 7, { promptTokenRows }));

    const prefill = fixture.decoderFeeds[0];
    expect(prefill.inputs_embeds.dims).toEqual([2, 3, 4096]);
    expect(Array.from(prefill.inputs_embeds.data.slice(0, 3 * 4096).filter((_, index) => index % 4096 === 0))).toEqual([11, 12, 13]);
    expect(Array.from(prefill.inputs_embeds.data.slice(3 * 4096).filter((_, index) => index % 4096 === 0))).toEqual([21, 22, 23]);
    expect(prefill.seqlens_k.data).toEqual(new Int32Array([2, 2]));
    expect(prefill.total_seq_len.data).toEqual(new Int32Array([3]));
    expect(fixture.cacheLengths).toEqual([3, 4]);
  });

  it.each([
    ['seed', { seed: -1 }, 'seed must be a uint32 integer'],
    ['guidance', { guidance: Number.NaN }, 'guidance must be finite and non-negative'],
    ['semantic top-k', { semanticTopK: 16_386 }, 'semanticTopK must be an integer between 1 and 16385'],
    ['residual top-k', { residualTopK: 0 }, 'residualTopK must be an integer between 1 and 1024'],
    ['temperature', { temperature: 0 }, 'temperature must be finite and greater than zero'],
    [
      'prompt row lengths',
      { promptTokenRows: { conditional: [1, 2], unconditional: [1] } },
      'prompt token rows must have the same positive length',
    ],
  ])('rejects invalid %s before inference', async (_name, overrides, message) => {
    const fixture = runtime(1);
    await expect(fixture.generator.generateFrames(frameOptions(1, 7, overrides as Partial<GenerateFrameOptions>)))
      .rejects.toThrow(message as string);
    expect(fixture.counts().decoderCalls).toBe(0);
  });

  it('returns retained frames with natural-end counts when audio end is sampled after warmup', async () => {
    const fixture = runtime(2, undefined, 3);

    const frames = await fixture.generator.generateFrames(frameOptions(125));

    expect(frames).toHaveLength(2);
    expect(frames.termination).toBe('natural-end');
    expect(frames).not.toHaveProperty('capacityDiagnostic');
    expect(frames.hiddenGroups.byteLength).toBe(131_072);
    expect(frames.hiddenGroups.buffer.byteLength).toBe(8_192_000);
    expect(frames.plan).toMatchObject({
      retainedFrames: 2,
      termination: 'natural-end',
      semanticDecisions: 4,
      rvqCalls: 21,
      feedbackCalls: 3,
    });
    expect(fixture.counts()).toEqual({ decoderCalls: 4, headCalls: 4, depthCalls: 21, feedbackCalls: 3 });
    expect(fixture.cacheLengths).toEqual([40, 41, 42, 43]);
  });

  it('uses one extra draw to continue only in capacity diagnostics and records the suppressed end', async () => {
    const fixture = runtime(4, undefined, 3);

    const frames = await fixture.generator.generateFrames(frameOptions(4, 7, {
      audioEndPolicy: 'continue-for-capacity-diagnostic',
    }));

    expect(frames).toHaveLength(4);
    expect(frames.slice(0, 2).map((frame) => frame.semantic)).toEqual([101, 102]);
    expect(frames[2].semantic).toBe(104);
    expect(frames.termination).toBe('max-frames');
    expect(frames.capacityDiagnostic).toEqual({
      kind: 'continue-after-audio-end',
      suppressedAudioEnds: 1,
      firstAudioEndAtRetainedFrame: 2,
    });
    expect(fixture.counts()).toEqual({
      decoderCalls: 5,
      headCalls: 5,
      depthCalls: 35,
      feedbackCalls: 4,
    });
  });

  it('uses residual embedding offsets and repeats sampled codes across both CFG lanes', async () => {
    const fixture = runtime(2);
    await fixture.generator.generateFrames(frameOptions(2, 13));
    const seventhDepth = fixture.depthFeeds[6].residual_embeddings.data as Uint16Array;

    expect(seventhDepth[0]).toBe(0);
    expect(seventhDepth[4096]).toBe(1025);
    expect(seventhDepth[5 * 4096]).toBe(5125);
    expect(seventhDepth[6 * 4096]).toBe(0);
  });
});

describe('FP16 hidden readback', () => {
  it('downloads both lanes through ORT and keeps the raw conditional half bits', async () => {
    const mapped = new Uint16Array(2 * 4096);
    mapped.set([0x3c00, 0xbc00, 0x3555, 0x0001]);
    mapped.fill(0x3c00, 4096);
    const getData = vi.fn(async () => new Float16Array(mapped.buffer.slice(0)));

    const result = await readConditionalGpuFp16(
      { type: 'float16', location: 'gpu-buffer', dims: [2, 4096], getData } as never,
    );

    expect(result).toBeInstanceOf(Uint16Array);
    expect(result).toHaveLength(4096);
    expect(Array.from(result.slice(0, 4))).toEqual([0x3c00, 0xbc00, 0x3555, 0x0001]);
    expect(getData).toHaveBeenCalledOnce();
  });

  it('rejects FP16 infinity and NaN bit patterns in retained hidden groups', () => {
    expect(areFiniteFp16(new Uint16Array([0x0000, 0x3c00, 0x7bff]))).toBe(true);
    expect(areFiniteFp16(new Uint16Array([0x7c00]))).toBe(false);
    expect(areFiniteFp16(new Uint16Array([0x7e00]))).toBe(false);
  });
});
