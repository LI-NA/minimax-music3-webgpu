import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GeneratedFrame } from '../../../src/runtime/pipeline/rvq-generation';
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
    expect(result.hiddenBytes).toBe(8_192_000);
    expect(result.conditionBytes).toBe(1_761_280);
    expect(result.latentBytes).toBe(110_080);
  });
});

describe('exact GPU FP16 readback', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('copies raw bits for only the required tensor shape before unmapping', async () => {
    vi.stubGlobal('GPUBufferUsage', { COPY_DST: 1, MAP_READ: 2 });
    vi.stubGlobal('GPUMapMode', { READ: 1 });
    const mapped = new Uint16Array([0x3c00, 0xbc00, 0x3555, 0x0001]);
    const staging = {
      mapState: 'mapped',
      mapAsync: async () => undefined,
      getMappedRange: () => mapped.buffer,
      unmap: () => mapped.fill(0),
      destroy: vi.fn(),
    };
    const copyBufferToBuffer = vi.fn();
    const device = {
      createBuffer: () => staging,
      createCommandEncoder: () => ({ copyBufferToBuffer, finish: () => ({}) }),
      queue: { submit: vi.fn() },
    };

    const result = await readExactGpuFp16(
      device as never,
      { type: 'float16', location: 'gpu-buffer', dims: [1, 2, 2], gpuBuffer: {} } as never,
      [1, 2, 2],
      'fixture',
    );

    expect(Array.from(result)).toEqual([0x3c00, 0xbc00, 0x3555, 0x0001]);
    expect(copyBufferToBuffer).toHaveBeenCalledWith({}, 0, staging, 0, 8);
  });
});
