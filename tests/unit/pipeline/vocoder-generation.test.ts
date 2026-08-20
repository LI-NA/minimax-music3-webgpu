import { describe, expect, it } from 'vitest';

import {
  analyticVocoderLatents,
  generateFixedVocoderWav,
} from '../../../src/runtime/pipeline/vocoder-generation';

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
