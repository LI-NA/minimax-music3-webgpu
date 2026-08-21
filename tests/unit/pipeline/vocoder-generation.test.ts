import { describe, expect, it, vi } from 'vitest';

import {
  analyticVocoderLatents,
  generateFixedVocoderWav,
  generateVariableVocoderWav,
} from '../../../src/runtime/pipeline/vocoder-generation';
import { planDuration } from '../../../src/runtime/pipeline/duration-plan';

const latentValues = 128 * 430;
const waveformSamples = 220_160;

class FakeInputTensor {
  readonly location = 'cpu';
  disposed = false;

  constructor(
    public readonly type: string,
    public readonly data: Uint16Array,
    public readonly dims: readonly number[],
  ) {}

  dispose() { this.disposed = true; }
}

describe('fixed vocoder generation', () => {
  it('creates the exact deterministic analytic FP16 latent bit pattern', () => {
    const first = analyticVocoderLatents();
    const second = analyticVocoderLatents();

    expect(first).toHaveLength(latentValues);
    expect(first.slice(0, 6)).toEqual(
      new Uint16Array([0x2800, 0xa800, 0x3000, 0xb000, 0x3400, 0xb400]),
    );
    expect(second).toEqual(first);
    expect(second).not.toBe(first);
  });

  it('runs one CPU FP16 latent tensor and returns the canonical stereo PCM16 WAV', async () => {
    const latentBits = new Uint16Array(latentValues);
    latentBits.set([0x0000, 0x3c00, 0xbc00]);
    const waveform = new Float32Array(2 * waveformSamples);
    waveform.set([-1, -0.5, 0.25]);
    waveform.set([1, 0.5, -0.25], waveformSamples);
    const inputs: FakeInputTensor[] = [];
    let runs = 0;
    let outputDisposed = false;
    const output = {
      type: 'float32',
      dims: [1, 2, waveformSamples],
      location: 'cpu',
      getData: async () => waveform,
      dispose: () => { outputDisposed = true; },
    };

    const wav = await generateFixedVocoderWav(
      {
        ort: {
          Tensor: class extends FakeInputTensor {
            constructor(type: string, data: Uint16Array, dims: readonly number[]) {
              super(type, data, dims);
              inputs.push(this);
            }
          },
        } as never,
        session: {
          run: async (feeds: Record<string, FakeInputTensor>) => {
            runs++;
            expect(feeds).toEqual({ latents: inputs[0] });
            return { waveform: output };
          },
        } as never,
      },
      latentBits,
    );

    const view = new DataView(wav);
    expect(runs).toBe(1);
    expect(inputs[0].type).toBe('float16');
    expect(inputs[0].data).toBe(latentBits);
    expect(inputs[0].dims).toEqual([1, 128, 430]);
    expect(inputs[0].location).toBe('cpu');
    expect(inputs[0].disposed).toBe(true);
    expect(outputDisposed).toBe(true);
    expect(wav.byteLength).toBe(880_684);
    expect(view.getUint32(24, true)).toBe(44_100);
    expect(Array.from({ length: 6 }, (_, index) => view.getInt16(44 + index * 2, true))).toEqual([
      -32_768, 32_767, -16_384, 16_384, 8_192, -8_192,
    ]);
  });

  it('rejects a latent bit buffer that cannot fill the exact fixed tensor', async () => {
    let tensors = 0;
    let runs = 0;

    await expect(generateFixedVocoderWav(
      {
        ort: {
          Tensor: class extends FakeInputTensor {
            constructor(type: string, data: Uint16Array, dims: readonly number[]) {
              super(type, data, dims);
              tensors++;
            }
          },
        } as never,
        session: {
          run: async () => {
            runs++;
            return {};
          },
        } as never,
      },
      new Uint16Array(latentValues - 1),
    )).rejects.toThrow(`latents must contain exactly ${latentValues} float16 values`);

    expect(tensors).toBe(0);
    expect(runs).toBe(0);
  });

  it('rejects a non-float32 ORT output and disposes both tensors', async () => {
    const inputs: FakeInputTensor[] = [];
    let outputDisposed = false;
    const output = {
      type: 'float16',
      dims: [1, 2, waveformSamples],
      location: 'cpu',
      getData: async () => new Float32Array(2 * waveformSamples),
      dispose: () => { outputDisposed = true; },
    };

    await expect(generateFixedVocoderWav(
      {
        ort: {
          Tensor: class extends FakeInputTensor {
            constructor(type: string, data: Uint16Array, dims: readonly number[]) {
              super(type, data, dims);
              inputs.push(this);
            }
          },
        } as never,
        session: { run: async () => ({ waveform: output }) } as never,
      },
      new Uint16Array(latentValues),
    )).rejects.toThrow('vocoder waveform must be float32');

    expect(inputs[0].disposed).toBe(true);
    expect(outputDisposed).toBe(true);
  });

  it('rejects an unexpected waveform shape before reading its data', async () => {
    const inputs: FakeInputTensor[] = [];
    let outputDisposed = false;
    let dataReads = 0;
    const output = {
      type: 'float32',
      dims: [1, waveformSamples, 2],
      location: 'cpu',
      getData: async () => {
        dataReads++;
        return new Float32Array();
      },
      dispose: () => { outputDisposed = true; },
    };

    await expect(generateFixedVocoderWav(
      {
        ort: {
          Tensor: class extends FakeInputTensor {
            constructor(type: string, data: Uint16Array, dims: readonly number[]) {
              super(type, data, dims);
              inputs.push(this);
            }
          },
        } as never,
        session: { run: async () => ({ waveform: output }) } as never,
      },
      new Uint16Array(latentValues),
    )).rejects.toThrow('vocoder waveform must have shape [1,2,220160]');

    expect(dataReads).toBe(0);
    expect(inputs[0].disposed).toBe(true);
    expect(outputDisposed).toBe(true);
  });

  it('rejects ORT waveform data that is not a Float32Array', async () => {
    const inputs: FakeInputTensor[] = [];
    let outputDisposed = false;
    const output = {
      type: 'float32',
      dims: [1, 2, waveformSamples],
      location: 'cpu',
      getData: async () => new Uint16Array(2 * waveformSamples),
      dispose: () => { outputDisposed = true; },
    };

    await expect(generateFixedVocoderWav(
      {
        ort: {
          Tensor: class extends FakeInputTensor {
            constructor(type: string, data: Uint16Array, dims: readonly number[]) {
              super(type, data, dims);
              inputs.push(this);
            }
          },
        } as never,
        session: { run: async () => ({ waveform: output }) } as never,
      },
      new Uint16Array(latentValues),
    )).rejects.toThrow('vocoder waveform data must be a Float32Array');

    expect(inputs[0].disposed).toBe(true);
    expect(outputDisposed).toBe(true);
  });

  it('rejects non-finite waveform samples and disposes both tensors', async () => {
    const inputs: FakeInputTensor[] = [];
    let outputDisposed = false;
    const waveform = new Float32Array(2 * waveformSamples);
    waveform[17] = Number.NaN;
    const output = {
      type: 'float32',
      dims: [1, 2, waveformSamples],
      location: 'cpu',
      getData: async () => waveform,
      dispose: () => { outputDisposed = true; },
    };

    await expect(generateFixedVocoderWav(
      {
        ort: {
          Tensor: class extends FakeInputTensor {
            constructor(type: string, data: Uint16Array, dims: readonly number[]) {
              super(type, data, dims);
              inputs.push(this);
            }
          },
        } as never,
        session: { run: async () => ({ waveform: output }) } as never,
      },
      new Uint16Array(latentValues),
    )).rejects.toThrow('vocoder waveform samples must be finite');

    expect(inputs[0].disposed).toBe(true);
    expect(outputDisposed).toBe(true);
  });

  it('rejects a float32 waveform buffer whose length contradicts its shape', async () => {
    const inputs: FakeInputTensor[] = [];
    let outputDisposed = false;
    const output = {
      type: 'float32',
      dims: [1, 2, waveformSamples],
      location: 'cpu',
      getData: async () => new Float32Array(2 * waveformSamples - 1),
      dispose: () => { outputDisposed = true; },
    };

    await expect(generateFixedVocoderWav(
      {
        ort: {
          Tensor: class extends FakeInputTensor {
            constructor(type: string, data: Uint16Array, dims: readonly number[]) {
              super(type, data, dims);
              inputs.push(this);
            }
          },
        } as never,
        session: { run: async () => ({ waveform: output }) } as never,
      },
      new Uint16Array(latentValues),
    )).rejects.toThrow(`vocoder waveform must contain exactly ${2 * waveformSamples} samples`);

    expect(inputs[0].disposed).toBe(true);
    expect(outputDisposed).toBe(true);
  });
});

describe('variable vocoder generation', () => {
  it('matches the fixed deterministic stereo wrapper for the five-second plan', async () => {
    const plan = planDuration({ durationSeconds: 5, promptTokens: 0 });
    const latentBits = analyticVocoderLatents();
    const left = Float32Array.from(
      { length: waveformSamples },
      (_, index) => ((index % 17) - 8) / 8,
    );
    const right = Float32Array.from(left, (sample) => -sample);
    const fixedWaveform = new Float32Array(2 * waveformSamples);
    fixedWaveform.set(left);
    fixedWaveform.set(right, waveformSamples);
    const makeOutput = (waveform: Float32Array, dims: readonly number[]) => ({
      type: 'float32', dims, location: 'cpu', getData: async () => waveform, dispose() {},
    });
    const ortRuntime = { Tensor: FakeInputTensor } as never;
    const fixed = await generateFixedVocoderWav(
      {
        ort: ortRuntime,
        session: { run: async () => ({
          waveform: makeOutput(fixedWaveform, [1, 2, waveformSamples]),
        }) } as never,
      },
      latentBits,
    );
    let monoCall = 0;
    const variable = await generateVariableVocoderWav(
      {
        ort: ortRuntime,
        session: { run: async () => {
          const waveform = monoCall++ === 0 ? left : right;
          return { waveform: makeOutput(waveform, [1, 1, waveformSamples]) };
        } } as never,
      },
      plan,
      [latentBits],
    );

    expect(variable).toEqual(fixed);
    expect(monoCall).toBe(2);
  });

  it('runs one symbolic mono session left then right per chunk and crops directly into the final WAV', async () => {
    const plan = planDuration({ durationSeconds: 10, promptTokens: 0 });
    const latentChunks = plan.chunks.map(({ latentLength }, chunkIndex) => {
      const values = new Uint16Array(128 * latentLength);
      values.fill(0x3000 + chunkIndex, 0, 64 * latentLength);
      values.fill(0xb000 + chunkIndex, 64 * latentLength);
      return values;
    });
    const inputs: FakeInputTensor[] = [];
    const calls: Array<{ bits: Uint16Array; dims: readonly number[] }> = [];
    let outputDisposals = 0;
    const completed: unknown[] = [];
    const observerDisposalCounts: number[] = [];
    const ordering: string[] = [];
    let clock = 0;
    const now = vi.spyOn(performance, 'now').mockImplementation(() => clock += 5);
    const session = {
      run: async ({ latents }: Record<string, FakeInputTensor>) => {
        const callIndex = calls.length;
        calls.push({ bits: latents.data, dims: latents.dims });
        const chunkIndex = Math.floor(callIndex / 2);
        const channel = callIndex % 2;
        const sampleCount = 512 * plan.chunks[chunkIndex].latentLength;
        const waveform = new Float32Array(sampleCount).fill(
          chunkIndex === 0 ? (channel === 0 ? 0.25 : -0.25) : (channel === 0 ? 0.5 : -0.5),
        );
        if (chunkIndex === 0 && channel === 0) {
          waveform[plan.chunks[0].samplesPerChannel - 1] = 0.125;
          waveform[plan.chunks[0].samplesPerChannel] = 0.75;
        }
        if (chunkIndex === 1 && channel === 0) waveform[0] = -0.75;
        return { waveform: {
          type: 'float32',
          dims: [1, 1, sampleCount],
          location: 'cpu',
          getData: async () => waveform,
          dispose: () => { outputDisposals++; },
        } };
      },
    };

    const wav = await generateVariableVocoderWav(
      {
        ort: { Tensor: class extends FakeInputTensor {
          constructor(type: string, data: Uint16Array, dims: readonly number[]) {
            super(type, data, dims);
            inputs.push(this);
          }
        } } as never,
        session: session as never,
      },
      plan,
      latentChunks,
      (progress) => {
        completed.push(progress);
        observerDisposalCounts.push(outputDisposals);
        ordering.push(`${progress.chunkIndex}:${progress.channel}`);
      },
      () => ordering.push('wav-start'),
    );
    now.mockRestore();
    const view = new DataView(wav);
    const secondChunkStart = plan.chunks[0].samplesPerChannel;

    expect(wav.byteLength).toBe(1_763_372);
    expect(plan.samplesPerChannel).toBe(440_832);
    expect(calls.map(({ dims }) => dims)).toEqual([
      [1, 64, 689], [1, 64, 689], [1, 64, 516], [1, 64, 516],
    ]);
    expect(calls.map(({ bits }) => bits[0])).toEqual([0x3000, 0xb000, 0x3001, 0xb001]);
    expect(inputs.every(({ disposed }) => disposed)).toBe(true);
    expect(outputDisposals).toBe(4);
    expect(observerDisposalCounts).toEqual([1, 2, 3, 4]);
    expect(completed).toEqual([
      { chunkIndex: 0, channel: 'left', completedCalls: 1, totalCalls: 4, inferenceMs: 5, pcmWriteMs: 5 },
      { chunkIndex: 0, channel: 'right', completedCalls: 2, totalCalls: 4, inferenceMs: 5, pcmWriteMs: 5 },
      { chunkIndex: 1, channel: 'left', completedCalls: 3, totalCalls: 4, inferenceMs: 5, pcmWriteMs: 5 },
      { chunkIndex: 1, channel: 'right', completedCalls: 4, totalCalls: 4, inferenceMs: 5, pcmWriteMs: 5 },
    ]);
    expect(ordering).toEqual(['wav-start', '0:left', '0:right', '1:left', '1:right']);
    expect(view.getInt16(44, true)).toBe(8_192);
    expect(view.getInt16(46, true)).toBe(-8_192);
    expect(view.getInt16(44 + (secondChunkStart - 1) * 4, true)).toBe(4_096);
    expect(view.getInt16(44 + secondChunkStart * 4, true)).toBe(16_384);
    expect(view.getInt16(46 + secondChunkStart * 4, true)).toBe(-16_384);
  });

  it('reports completed channels in order without reporting a failed mono call', async () => {
    const plan = planDuration({ durationSeconds: 5, promptTokens: 0 });
    const completed: unknown[] = [];
    let calls = 0;
    const validWaveform = new Float32Array(waveformSamples);

    await expect(generateVariableVocoderWav(
      {
        ort: { Tensor: FakeInputTensor } as never,
        session: { run: async () => {
          const call = calls++;
          return { waveform: {
            type: call === 0 ? 'float32' : 'float16',
            dims: [1, 1, waveformSamples],
            location: 'cpu',
            getData: async () => validWaveform,
            dispose() {},
          } };
        } } as never,
      },
      plan,
      [new Uint16Array(latentValues)],
      (progress) => completed.push(progress),
    )).rejects.toThrow('vocoder waveform must be float32');

    expect(calls).toBe(2);
    expect(completed).toEqual([
      expect.objectContaining({
        chunkIndex: 0,
        channel: 'left',
        completedCalls: 1,
        totalCalls: 2,
      }),
    ]);
  });

  it('rejects chunks that do not contain the exact active FP16 latent shape before inference', async () => {
    const plan = planDuration({ durationSeconds: 6, promptTokens: 0 });
    let runs = 0;

    await expect(generateVariableVocoderWav(
      {
        ort: { Tensor: FakeInputTensor } as never,
        session: { run: async () => { runs++; return {}; } } as never,
      },
      plan,
      [new Uint16Array(128 * plan.chunks[0].latentLength - 1)],
    )).rejects.toThrow('vocoder chunk 0 latents must contain exactly 66048 float16 values');
    expect(runs).toBe(0);
  });

  it.each([
    ['float16', [1, 1, 264_192], new Float32Array(264_192), 'vocoder waveform must be float32'],
    ['float32', [1, 2, 264_192], new Float32Array(264_192), 'vocoder waveform must have shape [1,1,264192]'],
    ['float32', [1, 1, 264_192], new Float32Array(264_192).fill(Number.NaN), 'vocoder waveform samples must be finite'],
  ])('rejects invalid symbolic mono output: %s %j', async (type, dims, data, message) => {
    const plan = planDuration({ durationSeconds: 6, promptTokens: 0 });
    let outputDisposed = false;
    const inputs: FakeInputTensor[] = [];

    await expect(generateVariableVocoderWav(
      {
        ort: { Tensor: class extends FakeInputTensor {
          constructor(tensorType: string, bits: Uint16Array, shape: readonly number[]) {
            super(tensorType, bits, shape);
            inputs.push(this);
          }
        } } as never,
        session: { run: async () => ({ waveform: {
          type,
          dims,
          location: 'cpu',
          getData: async () => data,
          dispose: () => { outputDisposed = true; },
        } }) } as never,
      },
      plan,
      [new Uint16Array(128 * plan.chunks[0].latentLength)],
    )).rejects.toThrow(message);
    expect(inputs[0].disposed).toBe(true);
    expect(outputDisposed).toBe(true);
  });
});
