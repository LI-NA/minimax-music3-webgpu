import { describe, expect, it } from 'vitest';
import { runGlobalSmoke } from '../../../src/runtime/pipeline/global-smoke';

const tensors: { disposed: number }[] = [];
function tensor(dims: number[], location: 'gpu-buffer' | 'cpu' = 'gpu-buffer') {
  const value = {
    dims,
    location,
    gpuBuffer: { size: 64 },
    getData: async () => new Float32Array([1]),
    dispose: () => value.disposed++,
    disposed: 0,
  };
  tensors.push(value);
  return value;
}

describe('runGlobalSmoke', () => {
  it('prefills the exact two-lane contract then advances GPU KV cache from 40 to 50', async () => {
    let calls = 0;
    const feeds: Record<string, { dims: readonly number[]; data: unknown }> [] = [];
    const decoder = {
      inputNames: [],
      outputNames: [],
      run: async (nextFeeds: Record<string, { dims: readonly number[]; data: unknown }>) => {
        feeds.push(nextFeeds);
        const length = 40 + calls++;
        return { hidden_states: tensor([2, 1, 4]), present: tensor([2, 8, length, 128]) };
      },
    };
    const head = {
      inputNames: ['hidden_states'],
      outputNames: [],
      run: async () => ({ semantic_logits: tensor([2, 4]), end_logit: tensor([2, 1]) }),
    };
    try {
      const result = await runGlobalSmoke({
        ort: {
          Tensor: class {
            constructor(
              public readonly type: string,
              public readonly data: unknown,
              public readonly dims: readonly number[],
            ) {}
            dispose() {}
          },
        } as never,
        decoder: decoder as never,
        head: head as never,
        embedding: { lookup: (ids) => new Uint16Array(ids.length * 4), dispose() {} },
        embeddingTable: { columns: 4 },
        kvPairs: [{ pastInput: 'past', presentOutput: 'present' }],
      });
      expect(result.cacheLengths).toEqual([40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]);
      expect(result.stepMs).toHaveLength(11);
      expect(result.tensorLocations).toContain('gpu-buffer');
      expect(result.finiteLogits).toBe(true);
      expect(tensors.every((item) => item.disposed > 0)).toBe(true);
      expect(feeds.every((entry) => !('attention_mask' in entry))).toBe(true);
      expect(feeds.map((entry) => Array.from(entry.seqlens_k.data as Int32Array))).toEqual(
        [[39, 39], [40, 40], [41, 41], [42, 42], [43, 43], [44, 44], [45, 45], [46, 46], [47, 47], [48, 48], [49, 49]],
      );
      expect(feeds.map((entry) => Array.from(entry.total_seq_len.data as Int32Array))).toEqual(
        [[40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50]],
      );
    } finally {
      tensors.length = 0;
    }
  });
});
