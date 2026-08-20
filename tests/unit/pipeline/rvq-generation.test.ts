import { describe, expect, it } from 'vitest';
import { createFrameGenerator } from '../../../src/runtime/pipeline/rvq-generation';

class FakeTensor {
  location: 'cpu' | 'gpu-buffer';
  gpuBuffer = { size: 64 };
  disposed = false;
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
  dispose() { this.disposed = true; }
}

function runtime(decisions: number) {
  let decoderCalls = 0;
  let headCalls = 0;
  let depthCalls = 0;
  let feedbackCalls = 0;
  const depthFeeds: Record<string, FakeTensor>[] = [];
  const cacheLengths: number[] = [];
  const decoder = {
    run: async () => {
      const cacheLength = 40 + decoderCalls++;
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
      semantic[100 + decision] = semantic[16_384 + 100 + decision] = 10;
      return {
        semantic_logits: new FakeTensor('float32', semantic, [2, 16_384]),
        end_logit: new FakeTensor('float16', new Float16Array([-100, -100]), [2, 1]),
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
    globalEmbedding: { lookup: (ids: readonly number[]) => new Uint16Array(ids.length * 4096), dispose() {} },
    rvqEmbedding: { lookup: (ids: readonly number[]) => Uint16Array.from(ids.flatMap((id) => Array(4096).fill(id))), dispose() {} },
    embeddingColumns: 4096,
    kvPairs: [{ pastInput: 'past', presentOutput: 'present' }],
    readConditionalHidden: async (tensor) => new Float32Array(4096).fill((tensor as unknown as FakeTensor).conditional!),
    onCacheLength: (length) => cacheLengths.push(length),
  });
  return {
    generator,
    counts: () => ({ decoderCalls, headCalls, depthCalls, feedbackCalls }),
    depthFeeds,
    cacheLengths,
    decisions,
  };
}

describe('RVQ frame generation', () => {
  it('discards warmup, emits two real frames, and keeps the exact eight conditional hidden groups', async () => {
    const fixture = runtime(2);
    const frames = await fixture.generator.generateFrames({ maxFrames: 2, seed: 7, guidance: 1.5, topK: 50 });

    expect(frames.map((frame) => [frame.semantic, ...frame.residual])).toEqual([
      [101, 0, 1, 2, 3, 4, 5, 6],
      [102, 0, 1, 2, 3, 4, 5, 6],
    ]);
    expect(fixture.counts()).toEqual({ decoderCalls: 3, headCalls: 3, depthCalls: 21, feedbackCalls: 2 });
    expect(fixture.cacheLengths).toEqual([40, 41, 42]);
    expect(frames[0].hiddenGroups).toHaveLength(8 * 4096);
    expect(frames[0].hiddenGroups[0]).toBe(1);
    expect(frames[0].hiddenGroups[4096]).toBe(1);
    expect(frames[0].hiddenGroups[7 * 4096]).toBe(7);
  });

  it('samples ORT Float16Array CPU logits without treating their values as raw half bits', async () => {
    const fixture = runtime(2);
    const frames = await fixture.generator.generateFrames({ maxFrames: 2, seed: 7, guidance: 1.5, topK: 50 });

    expect(frames.map((frame) => frame.semantic)).toEqual([101, 102]);
  });

  it('runs 126 semantic decisions, 882 depth calls, and 125 feedback decodes for 125 retained frames', async () => {
    const fixture = runtime(125);
    const frames = await fixture.generator.generateFrames({ maxFrames: 125, seed: 11, guidance: 1.5, topK: 50 });

    expect(frames).toHaveLength(125);
    expect(fixture.counts()).toEqual({ decoderCalls: 126, headCalls: 126, depthCalls: 882, feedbackCalls: 125 });
    expect(frames.reduce((bytes, frame) => bytes + frame.hiddenGroups.byteLength, 0)).toBe(16_384_000);
  });

  it('uses residual embedding offsets and repeats sampled codes across both CFG lanes', async () => {
    const fixture = runtime(2);
    await fixture.generator.generateFrames({ maxFrames: 2, seed: 13, guidance: 1.5, topK: 50 });
    const seventhDepth = fixture.depthFeeds[6].residual_embeddings.data as Uint16Array;

    expect(seventhDepth[0]).toBe(0);
    expect(seventhDepth[4096]).toBe(1025);
    expect(seventhDepth[5 * 4096]).toBe(5125);
    expect(seventhDepth[6 * 4096]).toBe(0);
  });
});
