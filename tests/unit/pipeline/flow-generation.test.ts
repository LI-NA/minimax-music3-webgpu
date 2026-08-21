import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import scheduleFixture from '../../fixtures/flow-schedule.json';
import {
  areFiniteFlowValues,
  exactFlowSchedule,
  runChunkedFlowGeneration,
  runFlowSmoke,
  runFixedFlowGeneration,
} from '../../../src/runtime/pipeline/flow-generation';
import { planDuration } from '../../../src/runtime/pipeline/duration-plan';

class FakeTensor {
  location: 'cpu' | 'gpu-buffer';
  disposed = false;
  getDataCalls = 0;

  constructor(
    public readonly type: string,
    public readonly data: Uint16Array | Float32Array | BigInt64Array,
    public readonly dims: readonly number[],
    location: 'cpu' | 'gpu-buffer' = 'cpu',
    private readonly onGetData?: () => void,
  ) {
    this.location = location;
  }

  dispose() { this.disposed = true; }
  async getData() {
    this.getDataCalls++;
    this.onGetData?.();
    return this.data;
  }
}

const bits = (values: Float32Array) => Array.from(new Uint32Array(values.buffer));
const fixedCase = JSON.parse(readFileSync(
  new URL('../../../tools/reference/fixed_case.json', import.meta.url),
  'utf8',
)) as {
  timestepF32Bits: number[];
  dtF32Bits: number[];
  flowScheduleRevision: string;
};

describe('fixed flow schedule', () => {
  it('preserves the pinned scheduler float32 bit patterns', () => {
    const schedule = exactFlowSchedule();

    expect(bits(schedule.timesteps)).toEqual(scheduleFixture.timestep_f32_bits);
    expect(bits(schedule.dts)).toEqual(scheduleFixture.dt_f32_bits);
  });

  it('binds the browser schedule bits to the cloud receipt revision', () => {
    const schedule = exactFlowSchedule();
    const timestepBits = bits(schedule.timesteps);
    const dtBits = bits(schedule.dts);
    const allBits = [...timestepBits, ...dtBits];
    const raw = Buffer.alloc(allBits.length * 4);
    allBits.forEach((value, index) => raw.writeUInt32LE(value, index * 4));

    expect(timestepBits).toEqual(fixedCase.timestepF32Bits);
    expect(dtBits).toEqual(fixedCase.dtF32Bits);
    expect(`sha256:${createHash('sha256').update(raw).digest('hex')}`)
      .toBe(fixedCase.flowScheduleRevision);
  });

  it('matches the pinned numpy linspace schedule for any positive step count', () => {
    const schedule = exactFlowSchedule(7);

    expect(bits(schedule.timesteps)).toEqual([
      0, 1041385764, 1049774372, 1054567862, 1058162980, 1060559726, 1062956471,
    ]);
    expect(bits(schedule.dts)).toEqual([
      1041385764, 1041385764, 1041385764, 1041385764, 1041385768, 1041385764, 1041385764,
    ]);
    expect(Array.from(exactFlowSchedule(1).timesteps)).toEqual([0]);
    expect(Array.from(exactFlowSchedule(1).dts)).toEqual([1]);
  });

  it.each([0, -1, 1.5, Number.NaN, Number.MAX_SAFE_INTEGER + 1])(
    'rejects invalid step count %s',
    (stepCount) => {
      expect(() => exactFlowSchedule(stepCount)).toThrow('positive safe integer');
    },
  );
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
  it.each([0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    'rejects invalid flow step count %s before running the graph',
    async (stepCount) => {
      const initial = new FakeTensor('float16', new Uint16Array(), [1, 128, 430]);
      const condition = new FakeTensor('float16', new Uint16Array(), [1, 430, 2048]);
      let calls = 0;

      await expect(runFixedFlowGeneration(
        {
          ort: { Tensor: FakeTensor } as never,
          session: { run: async () => { calls++; return {}; } } as never,
        },
        initial as never,
        condition as never,
        undefined,
        1.7,
        stepCount,
      )).rejects.toThrow('positive safe integer');
      expect(calls).toBe(0);
    },
  );

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
    expect(feeds.every((feed) => feed.guidance.data[0] === 0x3ecd)).toBe(true);
    expect(feeds.every((feed) => feed.timestep.disposed && feed.dt.disposed)).toBe(true);
    expect(feeds.every((feed) => feed.guidance.disposed)).toBe(true);
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

describe('official chunked flow generation', () => {
  it('runs the literal ten-second continuation contract through maximum-window sessions', async () => {
    let now = 0;
    const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => now);
    const maxLatents = 689;
    const overlapLatents = 172;
    const conditionWidth = 2048;
    const plan = planDuration({ durationSeconds: 10, promptTokens: 40 });
    const frameHiddens = new Uint16Array(250 * 32_768);
    frameHiddens[0] = 0x1111;
    frameHiddens[100 * 32_768] = 0x2222;
    const initialLatents = plan.chunks.map(({ latentLength }, chunkIndex) => {
      const values = new Uint16Array(128 * latentLength);
      for (let channel = 0; channel < 128; channel++)
        for (let index = 0; index < latentLength; index++)
          values[channel * latentLength + index] = chunkIndex === 0
            ? 0x0400 + index
            : 0x2400 + index;
      return values;
    });
    const conditionFeeds: Record<string, FakeTensor>[] = [];
    const conditionOutputs: FakeTensor[] = [];
    const conditionSession = {
      run: async (feeds: Record<string, FakeTensor>) => {
        conditionFeeds.push(feeds);
        const chunkIndex = conditionFeeds.length - 1;
        const values = new Uint16Array(maxLatents * conditionWidth);
        for (let latent = 0; latent < maxLatents; latent++)
          values.fill(0x1000 * (chunkIndex + 1) + latent, latent * conditionWidth, (latent + 1) * conditionWidth);
        now += 5;
        const output = new FakeTensor(
          'float16', values, [1, maxLatents, conditionWidth], 'gpu-buffer', () => { now += 2; },
        );
        conditionOutputs.push(output);
        return { condition: output };
      },
    };
    const flowFeeds: Record<string, FakeTensor>[] = [];
    const flowOutputs: FakeTensor[] = [];
    const completedSteps: number[] = [];
    const ordering: string[] = [];
    const flowSession = {
      run: async (feeds: Record<string, FakeTensor>) => {
        now += 1;
        flowFeeds.push(feeds);
        const chunkIndex = Math.floor((flowFeeds.length - 1) / 7);
        const step = (flowFeeds.length - 1) % 7;
        ordering.push(`flow:${chunkIndex}:${step}`);
        const values = new Uint16Array(128 * maxLatents);
        for (let channel = 0; channel < 128; channel++)
          for (let index = 0; index < maxLatents; index++)
            values[channel * maxLatents + index] = chunkIndex === 0
              ? 0x4000 + index
              : 0x6000 + step;
        const output = new FakeTensor(
          'float16', values, [1, 128, maxLatents], 'gpu-buffer',
          step === 6 ? () => { now += 3; } : undefined,
        );
        flowOutputs.push(output);
        return { next_latents: output };
      },
    };

    const result = await runChunkedFlowGeneration(
      {
        ort: { Tensor: FakeTensor } as never,
        conditionSession: conditionSession as never,
        flowSession: flowSession as never,
      },
      {
        plan,
        frameHiddens,
        initialLatents,
        flowGuidance: 1.25,
        flowSteps: 7,
        onConditionStart: (chunkIndex) => ordering.push(`condition-start:${chunkIndex}`),
        onConditionComplete: ({ chunkIndex, elapsedMs }) =>
          ordering.push(`condition:${chunkIndex}:${elapsedMs}`),
        onChunkStart: (chunkIndex, globalCompletedSteps) => {
          expect(conditionOutputs[chunkIndex].getDataCalls).toBe(1);
          expect(conditionOutputs[chunkIndex].disposed).toBe(true);
          ordering.push(`start:${chunkIndex}:${globalCompletedSteps}`);
        },
        onChunkComplete: ({ chunkIndex, elapsedMs }) =>
          ordering.push(`complete:${chunkIndex}:${elapsedMs}`),
        onStep: (step) => completedSteps.push(step),
      },
    );

    expect(result.map(({ startFrame }) => startFrame)).toEqual([0, 100]);
    expect(result.map(({ frameLength }) => frameLength)).toEqual([200, 150]);
    expect(result.map(({ latentLength }) => latentLength)).toEqual([689, 516]);
    expect(result.map(({ cropLeftLatents, cropRightLatents }) => [cropLeftLatents, cropRightLatents]))
      .toEqual([[0, 258], [86, 0]]);
    expect(conditionFeeds).toHaveLength(2);
    expect(conditionFeeds.map((feed) => feed.frame_hiddens.data[0])).toEqual([0x1111, 0x2222]);
    expect(conditionFeeds[1].frame_hiddens.data[150 * 32_768]).toBe(0);
    expect(conditionFeeds[1].nearest_index.data[515]).toBe(149n);
    expect(conditionFeeds[1].nearest_index.data[516]).toBe(0n);
    expect(flowFeeds).toHaveLength(14);
    expect(ordering).toEqual([
      'condition-start:0',
      'condition:0:7',
      'start:0:0',
      ...Array.from({ length: 7 }, (_, step) => `flow:0:${step}`),
      'complete:0:10',
      'condition-start:1',
      'condition:1:7',
      'start:1:7',
      ...Array.from({ length: 7 }, (_, step) => `flow:1:${step}`),
      'complete:1:10',
    ]);
    expect(completedSteps).toEqual(Array.from({ length: 14 }, (_, index) => index + 1));
    expect(flowOutputs.filter((tensor) => tensor.getDataCalls > 0)).toEqual([flowOutputs[6], flowOutputs[13]]);
    expect(conditionOutputs.every((tensor) => tensor.getDataCalls === 1 && tensor.disposed)).toBe(true);

    expect(flowFeeds.every((feed) => feed.guidance.data[0] === 0x3d00)).toBe(true);
    const secondChunkFeeds = flowFeeds.slice(7);
    const expectedCarry = Array.from({ length: overlapLatents }, (_, index) => 0x4000 + 345 + index);
    const expectedNoisePrompt = Array.from({ length: overlapLatents }, (_, index) => 0x2400 + index);
    expect(secondChunkFeeds.every((feed) => feed.overlap_enabled.data[0] === 0x3c00)).toBe(true);
    expect(secondChunkFeeds.every((feed) =>
      Array.from((feed.previous_latent.data as Uint16Array).slice(0, overlapLatents)).join(',')
      === expectedCarry.join(','),
    )).toBe(true);
    expect(secondChunkFeeds.every((feed) =>
      Array.from((feed.noise_prompt.data as Uint16Array).slice(0, overlapLatents)).join(',')
      === expectedNoisePrompt.join(','),
    )).toBe(true);
    expect(secondChunkFeeds.every((feed) => {
      const condition = feed.condition.data as Uint16Array;
      return condition[0] === 0x1000 + 345
        && condition[(overlapLatents - 1) * conditionWidth] === 0x1000 + 516
        && condition[516 * conditionWidth] === 0;
    })).toBe(true);
    expect(secondChunkFeeds.every((feed) => {
      const mask = feed.active_latent_mask.data as Uint16Array;
      const bias = feed.key_attention_bias.data as Uint16Array;
      return mask[515] === 0x3c00 && mask[516] === 0
        && bias[516] === 0 && bias[517] === 0xfbff;
    })).toBe(true);
    expect(flowFeeds.slice(1, 7).every((feed) => feed.latents.location === 'gpu-buffer')).toBe(true);
    expect(flowFeeds.slice(8).every((feed) => feed.latents.location === 'gpu-buffer')).toBe(true);
    const secondInitial = flowFeeds[7].latents.data as Uint16Array;
    expect(secondInitial[515]).toBe(0x2400 + 515);
    expect(secondInitial[516]).toBe(0);

    for (let channel = 0; channel < 128; channel++)
      expect(Array.from(result[1].latentBits.slice(
        channel * 516,
        channel * 516 + overlapLatents,
      ))).toEqual(expectedCarry);
    expect(result[1].latentBits[overlapLatents]).toBe(0x6000 + 6);
    expect(result.map(({ latentBits }) => latentBits.length)).toEqual([128 * 689, 128 * 516]);
    nowSpy.mockRestore();
  });

  it('does not announce a chunk when condition encoding fails', async () => {
    const plan = planDuration({ durationSeconds: 10, promptTokens: 40 });
    let chunkStarts = 0;
    const conditionStarts: number[] = [];
    let conditionCompletions = 0;
    let chunkCompletions = 0;
    let flowCalls = 0;

    await expect(runChunkedFlowGeneration(
      {
        ort: { Tensor: FakeTensor } as never,
        conditionSession: {
          run: async () => { throw new Error('condition failed'); },
        } as never,
        flowSession: {
          run: async () => {
            flowCalls++;
            return {};
          },
        } as never,
      },
      {
        plan,
        frameHiddens: new Uint16Array(plan.retainedFrames * 32_768),
        initialLatents: plan.chunks.map(({ latentLength }) => new Uint16Array(128 * latentLength)),
        flowGuidance: 1.7,
        flowSteps: 30,
        onConditionStart: (chunkIndex) => conditionStarts.push(chunkIndex),
        onConditionComplete: () => conditionCompletions++,
        onChunkStart: () => chunkStarts++,
        onChunkComplete: () => chunkCompletions++,
      },
    )).rejects.toThrow('condition failed');

    expect(chunkStarts).toBe(0);
    expect(conditionStarts).toEqual([0]);
    expect(conditionCompletions).toBe(0);
    expect(chunkCompletions).toBe(0);
    expect(flowCalls).toBe(0);
  });

  it('does not complete a chunk when final latent readback fails', async () => {
    const plan = planDuration({ durationSeconds: 5, promptTokens: 40 });
    let flowCalls = 0;
    let conditionCompletions = 0;
    let chunkCompletions = 0;

    await expect(runChunkedFlowGeneration(
      {
        ort: { Tensor: FakeTensor } as never,
        conditionSession: {
          run: async () => ({
            condition: new FakeTensor(
              'float16',
              new Uint16Array(689 * 2048),
              [1, 689, 2048],
              'gpu-buffer',
            ),
          }),
        } as never,
        flowSession: {
          run: async () => {
            flowCalls++;
            return {
              next_latents: new FakeTensor(
                'float16',
                new Uint16Array(),
                [1, 128, 689],
                'gpu-buffer',
              ),
            };
          },
        } as never,
      },
      {
        plan,
        frameHiddens: new Uint16Array(plan.retainedFrames * 32_768),
        initialLatents: [new Uint16Array(128 * plan.chunks[0].latentLength)],
        flowGuidance: 1.7,
        flowSteps: 30,
        onConditionComplete: () => conditionCompletions++,
        onChunkComplete: () => chunkCompletions++,
      },
    )).rejects.toThrow('final latents downloader returned 0 values');

    expect(flowCalls).toBe(30);
    expect(conditionCompletions).toBe(1);
    expect(chunkCompletions).toBe(0);
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
