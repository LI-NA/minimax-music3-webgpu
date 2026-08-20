import { describe, expect, it } from 'vitest';
import { runConditionSmoke } from '../../../src/runtime/pipeline/condition-smoke';

describe('runConditionSmoke', () => {
  it('runs the fixed analytic FP16 input and downloads one finite GPU output', async () => {
    let input: { type: string; data: Uint16Array; dims: readonly number[] } | undefined;
    let inputDisposed = 0;
    let outputDisposed = 0;
    const output = {
      dims: [1, 430, 2048],
      location: 'gpu-buffer',
      getData: async () => new Uint16Array([0x0000, 0x3c00, 0xbc00]),
      dispose: () => outputDisposed++,
    };

    const result = await runConditionSmoke({
      ort: {
        Tensor: class {
          readonly type: string;
          readonly data: Uint16Array;
          readonly dims: readonly number[];
          constructor(type: string, data: Uint16Array, dims: readonly number[]) {
            this.type = type;
            this.data = data;
            this.dims = dims;
            input = { type, data, dims };
          }
          dispose() {
            inputDisposed++;
          }
        },
      } as never,
      session: {
        run: async () => ({ condition: output }),
      } as never,
    });

    expect(input?.type).toBe('float16');
    expect(input?.dims).toEqual([1, 125, 32768]);
    expect(input?.data.slice(0, 4)).toEqual(new Uint16Array([0x2800, 0xa800, 0x2800, 0xa800]));
    expect(result.shape).toEqual([1, 430, 2048]);
    expect(result.outputLocation).toBe('gpu-buffer');
    expect(result.finite).toBe(true);
    expect(inputDisposed).toBe(1);
    expect(outputDisposed).toBe(1);
  });
});
