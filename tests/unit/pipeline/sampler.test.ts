import { describe, expect, it } from 'vitest';
import { sampleResidual, sampleSemantic } from '../../../src/runtime/pipeline/sampler';

describe('semantic sampler', () => {
  it('restricts guidance to the conditional top 50 before sampling semantic rows plus audio end', () => {
    const conditional = new Float32Array(16_385).fill(-100);
    const unconditional = new Float32Array(16_385).fill(-100);
    for (let index = 0; index < 50; index++) conditional[index] = 50 - index;
    conditional[16_384] = 0;
    unconditional[16_384] = 1_000;

    expect(
      sampleSemantic(conditional, unconditional, { guidance: 1.5, topK: 50, draw: () => 0.999 }),
    ).toBeLessThan(50);
  });

  it('uses stable index ordering for tied logits and deterministic injected draws', () => {
    const conditional = new Float32Array(16_385).fill(-100);
    const unconditional = new Float32Array(16_385).fill(-100);
    conditional[7] = conditional[3] = 4;
    unconditional[7] = unconditional[3] = 4;

    expect(sampleSemantic(conditional, unconditional, { guidance: 1.5, topK: 2, draw: () => 0 })).toBe(3);
    expect(sampleSemantic(conditional, unconditional, { guidance: 1.5, topK: 2, draw: () => 0.75 })).toBe(7);
  });
});

describe('residual sampler', () => {
  it('applies two-lane guidance and top-k 50 to the 1024 residual rows', () => {
    const conditional = new Float32Array(1_024).fill(-100);
    const unconditional = new Float32Array(1_024).fill(-100);
    conditional[5] = 2;
    unconditional[5] = 0;
    conditional[8] = 0;
    unconditional[8] = 3;

    expect(sampleResidual(conditional, unconditional, { guidance: 1.5, topK: 50, draw: () => 0.5 })).toBe(5);
  });
});
