import { describe, expect, it, vi } from 'vitest';
import { planRetainedFrames } from '../../../src/runtime/pipeline/duration-plan';
import type { GeneratedFrame, GeneratedFrames } from '../../../src/runtime/pipeline/rvq-generation';
import { EarlyAudioEndError } from '../../../src/runtime/pipeline/rvq-generation';
import {
  deterministicGaussianFp16,
  flattenFrameHiddens,
  generateFiveSecondMusic,
  readExactGpuFp16,
} from '../../../src/runtime/pipeline/music-generation';

function frames(): GeneratedFrame[] {
  return Array.from({ length: 125 }, (_, frame) => ({
    semantic: frame,
    residual: [0, 1, 2, 3, 4, 5, 6],
    hiddenGroups: new Uint16Array(8 * 4096).fill(frame + 1),
  }));
}

describe('five-second music generation data contracts', () => {
  it('flattens 125 retained frame groups in frame-major order as exactly 8,192,000 FP16 bytes', () => {
    const flattened = flattenFrameHiddens(frames());

    expect(flattened).toBeInstanceOf(Uint16Array);
    expect(flattened.byteLength).toBe(8_192_000);
    expect(flattened[0]).toBe(1);
    expect(flattened[8 * 4096]).toBe(2);
    expect(flattened.at(-1)).toBe(125);
  });

  it('exposes a generated flat hidden prefix without allocating a second copy', () => {
    const storage = new Uint16Array(4 * 8 * 4096);
    const hiddenGroups = storage.subarray(0, 3 * 8 * 4096);
    const generated = Object.assign(
      Array.from({ length: 3 }, (_, frame) => ({
        semantic: frame,
        residual: [0, 1, 2, 3, 4, 5, 6] as const,
        hiddenGroups: hiddenGroups.subarray(frame * 8 * 4096, (frame + 1) * 8 * 4096),
      })),
      {
        hiddenGroups,
        termination: 'natural-end' as const,
        plan: planRetainedFrames({ retainedFrames: 3, promptTokens: 40, termination: 'natural-end' }),
      },
    ) as GeneratedFrames;

    const flattened = flattenFrameHiddens(generated);

    expect(flattened).toBe(hiddenGroups);
    expect(flattened.buffer).toBe(storage.buffer);
    expect(flattened.byteLength).toBe(196_608);
  });

  it('retains 300 seconds in one 491,520,000-byte hidden buffer with planned AR counts', () => {
    const retainedFrames = 7_500;
    const valuesPerFrame = 8 * 4096;
    const storage = new Uint16Array(retainedFrames * valuesPerFrame);
    storage[0] = 0x3c00;
    storage[storage.length - 1] = 0xbc00;
    const plan = planRetainedFrames({ retainedFrames, promptTokens: 40, termination: 'max-frames' });
    const generated = Object.assign(
      Array.from({ length: retainedFrames }, (_, frame) => ({
        semantic: frame % 16_384,
        residual: [0, 1, 2, 3, 4, 5, 6] as const,
        hiddenGroups: storage.subarray(frame * valuesPerFrame, (frame + 1) * valuesPerFrame),
      })),
      { hiddenGroups: storage, termination: 'max-frames' as const, plan },
    ) as GeneratedFrames;

    const flattened = flattenFrameHiddens(generated);

    expect(generated).toHaveLength(7_500);
    expect(flattened.byteLength).toBe(491_520_000);
    expect(flattened).toBe(storage);
    expect(flattened.buffer).toBe(storage.buffer);
    expect([flattened[0], flattened.at(-1)]).toEqual([0x3c00, 0xbc00]);
    expect({
      semanticDecisions: plan.semanticDecisions,
      depthCalls: plan.rvqCalls,
      feedbackCalls: plan.feedbackCalls,
      cacheReports: plan.semanticDecisions,
    }).toEqual({
      semanticDecisions: 7_501,
      depthCalls: 52_507,
      feedbackCalls: 7_500,
      cacheReports: 7_501,
    });
  });

  it('creates deterministic seeded Gaussian FP16 noise rather than an analytic alternating pattern', () => {
    const first = deterministicGaussianFp16(7, 128 * 430);
    const repeat = deterministicGaussianFp16(7, 128 * 430);
    const nextSeed = deterministicGaussianFp16(8, 128 * 430);

    expect(repeat).toEqual(first);
    expect(nextSeed).not.toEqual(first);
    expect(Array.from(first.slice(0, 8))).toEqual([
      0x308b, 0x2b74, 0xbaea, 0xc136, 0xbc07, 0x396f, 0x2c89, 0x3c79,
    ]);
    expect(new Set(first.slice(0, 32)).size).toBeGreaterThan(8);
    expect(first.some((value) => (value & 0x8000) === 0)).toBe(true);
    expect(first.some((value) => (value & 0x8000) !== 0)).toBe(true);
    expect(first.every((value) => (value & 0x7c00) !== 0x7c00)).toBe(true);
  });

  it('retries early audio end once with the next seed then runs condition, flow, and vocoder in order', async () => {
    const calls: string[] = [];
    const retained = frames();
    const condition = new Uint16Array(430 * 2048).fill(0x3c00);
    const latents = new Uint16Array(128 * 430).fill(0xbc00);
    const wav = new ArrayBuffer(880_684);
    const result = await generateFiveSecondMusic({
      autoregressive: async (seed) => {
        calls.push(`ar:${seed}`);
        if (seed === 7) throw new EarlyAudioEndError(seed, 2);
        return retained;
      },
      condition: async (bits) => {
        calls.push(`condition:${bits.byteLength}`);
        return condition;
      },
      flow: async (bits, seed, onStep) => {
        calls.push(`flow:${bits.byteLength}:${seed}`);
        for (let step = 1; step <= 30; step++) onStep(step);
        return latents;
      },
      vocoder: async (bits) => {
        calls.push(`vocoder:${bits.byteLength}`);
        return wav;
      },
    }, 7);

    expect(calls).toEqual([
      'ar:7', 'ar:8', 'condition:8192000', 'flow:1761280:8', 'vocoder:110080',
    ]);
    expect(result.wav).toBe(wav);
    expect(result.attemptedSeeds).toEqual([7, 8]);
    expect(result.retainedFrames).toBe(125);
    expect(result.termination).toBe('max-frames');
    expect(result.hiddenBytes).toBe(8_192_000);
    expect(result.conditionBytes).toBe(1_761_280);
    expect(result.latentBytes).toBe(110_080);
  });

  it('retries retained natural-end frames before fixed 125-frame acoustic stages', async () => {
    const hiddenGroups = new Uint16Array(2 * 8 * 4096).fill(0x3c00);
    const naturalFrames = Object.assign(
      Array.from({ length: 2 }, (_, frame) => ({
        semantic: frame,
        residual: [0, 1, 2, 3, 4, 5, 6] as const,
        hiddenGroups: hiddenGroups.subarray(frame * 8 * 4096, (frame + 1) * 8 * 4096),
      })),
      {
        hiddenGroups,
        termination: 'natural-end' as const,
        plan: planRetainedFrames({ retainedFrames: 2, promptTokens: 40, termination: 'natural-end' }),
      },
    ) as GeneratedFrames;
    const attemptedSeeds: number[] = [];
    const retained = frames();

    const result = await generateFiveSecondMusic({
      autoregressive: async (seed) => {
        attemptedSeeds.push(seed);
        return seed === 7 ? naturalFrames : retained;
      },
      condition: async (bits) => {
        expect(bits.byteLength).toBe(8_192_000);
        return new Uint16Array(430 * 2048);
      },
      flow: async () => new Uint16Array(128 * 430),
      vocoder: async () => new ArrayBuffer(880_684),
    }, 7);

    expect(attemptedSeeds).toEqual([7, 8]);
    expect(result.attemptedSeeds).toEqual([7, 8]);
    expect(result.retainedFrames).toBe(125);
    expect(result.termination).toBe('max-frames');
    expect(result.hiddenBytes).toBe(8_192_000);
  });
});

describe('exact GPU FP16 readback', () => {
  it('downloads raw bits for only the required tensor shape through ORT', async () => {
    const mapped = new Uint16Array([0x3c00, 0xbc00, 0x3555, 0x0001]);
    const getData = vi.fn(async () => new Float16Array(mapped.buffer.slice(0)));

    const result = await readExactGpuFp16(
      { type: 'float16', location: 'gpu-buffer', dims: [1, 2, 2], getData } as never,
      [1, 2, 2],
      'fixture',
    );

    expect(Array.from(result)).toEqual([0x3c00, 0xbc00, 0x3555, 0x0001]);
    expect(getData).toHaveBeenCalledOnce();
  });
});
