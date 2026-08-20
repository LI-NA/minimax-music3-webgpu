import { describe, expect, it } from 'vitest';
import scheduleFixture from '../../fixtures/flow-schedule.json';
import {
  areFiniteFlowValues,
  exactFlowSchedule,
  runFlowSmoke,
  runFixedFlowGeneration,
} from '../../../src/runtime/pipeline/flow-generation';

class FakeTensor {
  location: 'cpu' | 'gpu-buffer';
  disposed = false;
  getDataCalls = 0;

  constructor(
    public readonly type: string,
    public readonly data: Uint16Array | Float32Array,
    public readonly dims: readonly number[],
    location: 'cpu' | 'gpu-buffer' = 'cpu',
  ) {
    this.location = location;
  }

  dispose() { this.disposed = true; }
  async getData() {
    this.getDataCalls++;
    return this.data;
  }
}

const bits = (values: Float32Array) => Array.from(new Uint32Array(values.buffer));

describe('fixed flow schedule', () => {
  it('preserves the pinned scheduler float32 bit patterns', () => {
    const schedule = exactFlowSchedule();

    expect(bits(schedule.timesteps)).toEqual(scheduleFixture.timestep_f32_bits);
    expect(bits(schedule.dts)).toEqual(scheduleFixture.dt_f32_bits);
  });
});

describe('flow output finite checks', () => {
  it('accepts raw FP16 bits and decoded Float16Array or Float32Array values', () => {
    const Float16 = (globalThis as unknown as {
      Float16Array: { from(values: readonly number[]): ArrayLike<number> };
    }).Float16Array;

    expect(areFiniteFlowValues(new Uint16Array([0x0000, 0x3c00, 0xbc00]))).toBe(true);
    expect(areFiniteFlowValues(new Uint16Array([0x7c00]))).toBe(false);
    expect(areFiniteFlowValues(Float16.from([0, 1, -1]))).toBe(true);
    expect(areFiniteFlowValues(Float16.from([Number.POSITIVE_INFINITY]))).toBe(false);
    expect(areFiniteFlowValues(new Float32Array([0, 1, -1]))).toBe(true);
    expect(areFiniteFlowValues(new Float32Array([Number.NaN]))).toBe(false);
  });

  it('rejects unrelated tensor data types', () => {
    expect(() => areFiniteFlowValues(new Int32Array([1]))).toThrow('float16');
  });
});

describe('fixed flow generation', () => {
  it('chains 30 GPU outputs without latent readback and disposes intermediate tensors', async () => {
    const feeds: Record<string, FakeTensor>[] = [];
    const outputs: FakeTensor[] = [];
    const initial = new FakeTensor('float16', new Uint16Array(), [1, 128, 430]);
    const condition = new FakeTensor('float16', new Uint16Array(), [1, 430, 2048]);
    const session = {
      run: async (nextFeeds: Record<string, FakeTensor>) => {
        feeds.push(nextFeeds);
        const output = new FakeTensor('float16', new Uint16Array(), [1, 128, 430], 'gpu-buffer');
        outputs.push(output);
        return { next_latents: output };
      },
    };
    const completed: number[] = [];

    const result = await runFixedFlowGeneration(
      { ort: { Tensor: FakeTensor } as never, session: session as never },
      initial as never,
      condition as never,
      (step) => completed.push(step),
    );

    expect(feeds).toHaveLength(30);
    expect(completed).toEqual(Array.from({ length: 30 }, (_, index) => index + 1));
    expect(result).toBe(outputs[29]);
    expect(initial.disposed).toBe(true);
    expect(outputs.slice(0, -1).every((tensor) => tensor.disposed)).toBe(true);
    expect(outputs[29].disposed).toBe(false);
    expect(condition.disposed).toBe(false);
    expect(feeds[0].latents.location).toBe('cpu');
    expect(feeds.slice(1).every((feed) => feed.latents.location === 'gpu-buffer')).toBe(true);
    expect(feeds.every((feed) => feed.condition === condition)).toBe(true);
    expect(feeds.every((feed) => feed.timestep.disposed && feed.dt.disposed)).toBe(true);
  });

  it('rejects a CPU flow output before the next step', async () => {
    const initial = new FakeTensor('float16', new Uint16Array(), [1, 128, 430]);
    const condition = new FakeTensor('float16', new Uint16Array(), [1, 430, 2048]);
    const session = {
      run: async () => ({
        next_latents: new FakeTensor('float16', new Uint16Array(), [1, 128, 430]),
      }),
    };

    await expect(runFixedFlowGeneration(
      { ort: { Tensor: FakeTensor } as never, session: session as never },
      initial as never,
      condition as never,
    )).rejects.toThrow('next_latents must be a GPU-resident float16 tensor');
  });
});

describe('flow browser smoke', () => {
  it('gates one exact GPU step before a 30-step chain and downloads only both gate outputs', async () => {
    const outputs: FakeTensor[] = [];
    const feeds: Record<string, FakeTensor>[] = [];
    const session = {
      run: async (nextFeeds: Record<string, FakeTensor>) => {
        feeds.push(nextFeeds);
        const output = new FakeTensor(
          'float16',
          new Uint16Array([0x0000, 0x3c00, 0xbc00]),
          [1, 128, 430],
          'gpu-buffer',
        );
        outputs.push(output);
        return { next_latents: output };
      },
    };

    const result = await runFlowSmoke({
      ort: { Tensor: FakeTensor } as never,
      session: session as never,
    });

    expect(feeds).toHaveLength(31);
    expect(result.oneStepLocation).toBe('gpu-buffer');
    expect(result.finalLocation).toBe('gpu-buffer');
    expect(result.shape).toEqual([1, 128, 430]);
    expect(result.oneStepFinite).toBe(true);
    expect(result.finalFinite).toBe(true);
    expect(result.stepMs).toHaveLength(30);
    expect(outputs.filter((tensor) => tensor.getDataCalls > 0)).toEqual([outputs[0], outputs[30]]);
    expect(outputs.every((tensor) => tensor.disposed)).toBe(true);
    expect(Array.from((feeds[0].latents.data as Uint16Array).slice(0, 4)))
      .toEqual([0x2c00, 0xac00, 0x2c00, 0xac00]);
    expect(Array.from((feeds[0].condition.data as Uint16Array).slice(0, 4)))
      .toEqual([0x2800, 0xa800, 0x2800, 0xa800]);
    expect(feeds[0].latents.disposed).toBe(true);
    expect(feeds[0].condition.disposed).toBe(true);
    expect(feeds[1].latents.disposed).toBe(true);
    expect(feeds[1].condition.disposed).toBe(true);
  });
});
