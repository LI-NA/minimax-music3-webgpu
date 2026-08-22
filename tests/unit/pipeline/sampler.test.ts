import { describe, expect, it, vi } from 'vitest';
import { sampleResidual, sampleSemantic, sampleSemanticExcludingAudioEnd } from '../../../src/runtime/pipeline/sampler';

const sampling = (
  overrides: Partial<{
    guidance: number;
    topK: number;
    temperature: number;
    draw: () => number;
  }> = {},
) => ({
  guidance: 1.5,
  topK: 50,
  temperature: 1,
  draw: () => 0.5,
  ...overrides,
});

describe('semantic sampler', () => {
  it('restricts guidance to the conditional top 50 before sampling semantic rows plus audio end', () => {
    const conditional = new Float32Array(16_385).fill(-100);
    const unconditional = new Float32Array(16_385).fill(-100);
    for (let index = 0; index < 50; index++) conditional[index] = 50 - index;
    conditional[16_384] = 0;
    unconditional[16_384] = 1_000;
    expect(sampleSemantic(conditional, unconditional, sampling({ draw: () => 0.999 }))).toBeLessThan(50);
  });

  it('never lets masked semantic rows re-enter after extreme guidance', () => {
    const conditional = new Float32Array(16_385);
    const unconditional = new Float32Array(16_385);
    conditional[0] = 10;
    unconditional[0] = 1e9;

    expect(
      sampleSemantic(
        conditional,
        unconditional,
        sampling({
          guidance: 3,
          topK: 1,
          draw: () => 0,
        }),
      ),
    ).toBe(0);
  });

  it('keeps every conditional candidate tied at the semantic preselection kth threshold', () => {
    const conditional = new Float32Array(16_385).fill(-100);
    const unconditional = new Float32Array(16_385).fill(-100);
    for (let index = 0; index < 60; index++) conditional[index] = unconditional[index] = 5;
    for (let index = 0; index < 60; index++)
      expect(sampleSemantic(conditional, unconditional, sampling({ draw: () => (index + 0.5) / 60 }))).toBe(index);
  });

  it('excludes audio end only from the explicit capacity-diagnostic resample', () => {
    const conditional = new Float32Array(16_385).fill(-100);
    const unconditional = new Float32Array(16_385).fill(-100);
    conditional[123] = unconditional[123] = 90;
    conditional[16_384] = unconditional[16_384] = 100;
    const options = sampling({ draw: () => 0.999_999 });

    expect(sampleSemantic(conditional, unconditional, options)).toBe(16_384);
    expect(sampleSemanticExcludingAudioEnd(conditional, unconditional, options)).toBe(123);
  });

  it('consumes exactly one provided draw for each normal and exclusion sample', () => {
    const conditional = new Float32Array(16_385).fill(-100);
    const unconditional = new Float32Array(16_385).fill(-100);
    conditional[123] = unconditional[123] = 90;
    conditional[16_384] = unconditional[16_384] = 100;
    const normalDraw = vi.fn(() => 0.999_999);
    const exclusionDraw = vi.fn(() => 0.5);

    sampleSemantic(conditional, unconditional, sampling({ draw: normalDraw }));
    sampleSemanticExcludingAudioEnd(conditional, unconditional, sampling({ draw: exclusionDraw }));

    expect(normalDraw).toHaveBeenCalledOnce();
    expect(exclusionDraw).toHaveBeenCalledOnce();
  });
});

describe('residual sampler', () => {
  it('keeps every candidate tied at the kth final top-k threshold', () => {
    const conditional = new Float32Array(1_024).fill(-100);
    const unconditional = new Float32Array(1_024).fill(-100);
    for (let index = 0; index < 60; index++) conditional[index] = unconditional[index] = 5;
    for (let index = 0; index < 60; index++)
      expect(sampleResidual(conditional, unconditional, sampling({ draw: () => (index + 0.5) / 60 }))).toBe(index);
  });

  it('uses float32 guidance multiplication before storing guided logits', () => {
    const conditional = new Float32Array(1_024).fill(-100);
    const unconditional = new Float32Array(1_024).fill(-100);
    conditional.set([2.833_797_693_252_563_5, 13.935_146_331_787_11]);
    unconditional.set([-2.788_192_033_767_7, -5.052_731_990_814_209]);
    expect(
      sampleResidual(conditional, unconditional, sampling({ topK: 2, draw: () => 1.889_640_444_119_322_6e-8 })),
    ).toBe(1);
  });

  it('rounds exponential weights to float32', () => {
    const conditional = new Float32Array(1_024).fill(-100);
    const unconditional = new Float32Array(1_024).fill(-100);
    conditional.set([-5.953_262_805_938_721, -3.805_346_727_371_216]);
    unconditional.set([-4.551_779_270_172_119, -9.676_701_545_715_332]);
    expect(
      sampleResidual(conditional, unconditional, sampling({ topK: 2, draw: () => 0.003_065_924_160_182_476 })),
    ).toBe(1);
  });

  it('rounds cumulative probability additions to float32', () => {
    const conditional = new Float32Array(1_024).fill(-100);
    const unconditional = new Float32Array(1_024).fill(-100);
    conditional.set([9.054_932_594_299_316, -12.189_710_617_065_43]);
    unconditional.set([9.299_930_572_509_766, -10.067_653_656_005_86]);
    expect(sampleResidual(conditional, unconditional, sampling({ topK: 2, draw: () => 0.999_999_97 }))).toBe(0);
  });

  it('rounds target multiplication to float32', () => {
    const conditional = new Float32Array(1_024).fill(-100);
    const unconditional = new Float32Array(1_024).fill(-100);
    conditional.set([-5.161_171_913_146_973, -7.631_301_879_882_812_5]);
    unconditional.set([10.720_241_546_630_86, 6.320_966_243_743_8965]);
    expect(sampleResidual(conditional, unconditional, sampling({ topK: 2, draw: () => 0.818_401_634_693_145_8 }))).toBe(
      1,
    );
  });

  it('rounds inverse-CDF subtraction to float32', () => {
    const conditional = new Float32Array(1_024).fill(-100);
    const unconditional = new Float32Array(1_024).fill(-100);
    conditional.set([3.549_213_171_005_249, 6.778_862_476_348_877, -1.302_111_148_834_228_5]);
    unconditional.set([1.891_952_872_276_306_2, -11.212_660_789_489_746, 14.040_442_466_735_84]);
    expect(sampleResidual(conditional, unconditional, sampling({ topK: 3, draw: () => 0.999_999_95 }))).toBe(2);
  });

  it('walks inverse-CDF probabilities in original vocabulary order', () => {
    const conditional = new Float32Array(1_024).fill(-100);
    const unconditional = new Float32Array(1_024).fill(-100);
    conditional[4] = unconditional[4] = 0;
    conditional[900] = unconditional[900] = 1;
    expect(sampleResidual(conditional, unconditional, sampling({ topK: 2, draw: () => 0.2 }))).toBe(4);
  });

  it('divides guided logits by temperature using float32 semantics', () => {
    const conditional = new Float32Array(1_024).fill(-100);
    const unconditional = new Float32Array(1_024).fill(-100);
    conditional.set([0, 1]);
    unconditional.set([0, 1]);

    expect(sampleResidual(conditional, unconditional, sampling({ topK: 2, temperature: 1, draw: () => 0.1 }))).toBe(0);
    expect(sampleResidual(conditional, unconditional, sampling({ topK: 2, temperature: 0.1, draw: () => 0.1 }))).toBe(
      1,
    );
  });

  it.each([0, -1, Number.NaN, Infinity])('rejects invalid temperature %s', (temperature) => {
    const logits = new Float32Array(1_024);
    expect(() => sampleResidual(logits, logits, sampling({ temperature }))).toThrow(
      'temperature must be finite and greater than zero',
    );
  });
});
