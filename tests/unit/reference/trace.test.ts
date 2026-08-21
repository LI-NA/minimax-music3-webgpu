import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import Ajv from 'ajv';
import { describe, expect, it } from 'vitest';
import { parseReferenceTrace } from '../../../src/runtime/reference/trace';

const sha = 'a'.repeat(64);
const manifestSha = '5c295ebfb4b7849d317cf0abd3dd8bfc9da3b58dc74de12a3523c07f28d4500e';
const validate = new Ajv({ allErrors: true }).compile(JSON.parse(readFileSync(
  resolve('tests/fixtures/reference/first-transition.schema.json'), 'utf8',
)));
const provenance = {
  model: { id: 'MiniMaxAI/MiniMax-Music3', revision: 'fbdf52fbaaca799592917417eb05f1899f1255ec' },
  diffusersRevision: '3681e65996b4d2589219720101a6acbfd25073f8',
  python: '3.12.10', torch: '2.7.1+cu128', transformers: '4.52.4', cudaRuntime: '12.8',
  driver: '570.133.20', gpu: 'NVIDIA GeForce RTX 5090', generatorDevice: 'cuda',
  combinedManifestSha256: manifestSha, sourceReceiptSha256: sha,
};
const input = {
  rawPrompt: 'Global\nbpm is 96\nWarm female vocal', lyrics: '[verse]\nHello\n[chorus]\nStay\nTogether\n[bridge][solo]',
  assembledText: '<|im_start|><|caption_start|>Global\nbpm is 96\nWarm female vocal<|caption_end|><|lyrics_start|>[start]\n[verse]\nHello\n[chorus]\nStay\nTogether\n[bridge][solo]<|lyrics_end|><|im_end|><|audio_start|>',
  tokenRows: [[151644,151671,11646,198,65,5187,374,220,24,21,198,95275,8778,25407,151672,151673,28463,921,58,4450,921,9707,198,58,6150,355,921,38102,198,80987,198,58,13709,1457,82,10011,60,151674,151645,151669], [151644,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151654,151645,151669]],
};
const parameters = {
  seed: 7, audioDuration: 5, retainedFrames: 125, globalGuidance: 1.5,
  semanticTopK: 50, residualTopK: 50, flowGuidance: 1.7, flowSteps: 30,
};
const decision = { kind: 'forced-sampled-codes', semanticCode: 8, residualCodes: [0, 1, 2, 3, 4, 5, 6] };
const checkpoint = (name: string, dtype: string, shape: number[], value = 0) => ({ name, dtype, shape, values: Array.from({ length: shape.reduce((size, dimension) => size * dimension, 1) }, () => value) });
const receipt = (name: string, dtype: string, shape: number[], bytes: number) => ({ name, path: `tensors/${name}.bin`, dtype, shape, bytes, sha256: sha });
const firstTransition = {
  schemaVersion: 2, scope: 'first-transition', provenance, input, parameters, decisions: [decision],
  checkpoints: [
    checkpoint('semantic-topk-ids', 'int32', [2, 50]), checkpoint('semantic-topk-logits', 'float32', [2, 50]),
    checkpoint('residual-topk-ids', 'int32', [7, 2, 50]), checkpoint('residual-topk-logits', 'float32', [7, 2, 50]),
    checkpoint('conditional-frame-hidden', 'bfloat16', [8, 4096]), checkpoint('feedback', 'bfloat16', [2, 1, 4096]),
    { ...checkpoint('cache-lengths', 'int32', [2]), values: [40, 41] },
  ],
  tensorReceipts: [], termination: 'fixed-length-no-early-end',
};
const fullFiveSecond = {
  ...firstTransition,
  scope: 'full-5s',
  decisions: Array.from({ length: 126 }, () => decision),
  checkpoints: [],
  tensorReceipts: [
    receipt('frame-handoff', 'bfloat16', [1, 125, 32768], 8192000), receipt('condition', 'bfloat16', [1, 430, 2048], 1761280), receipt('initial-flow-noise', 'bfloat16', [1, 128, 430], 110080), receipt('flow-step-1', 'bfloat16', [1, 128, 430], 110080), receipt('flow-step-15', 'bfloat16', [1, 128, 430], 110080), receipt('flow-step-30', 'bfloat16', [1, 128, 430], 110080), receipt('final-latent', 'bfloat16', [1, 128, 430], 110080), receipt('waveform', 'float32', [1, 2, 220160], 1761280), receipt('wav', 'uint8', [880684], 880684),
  ],
  flowNoise: { kind: 'gaussian-flow-noise', receiptName: 'initial-flow-noise' },
};

function expectAgreement(value: unknown) {
  expect(validate(value), JSON.stringify(validate.errors)).toBe(true);
  expect(parseReferenceTrace(value)).toEqual(value);
}

function expectRejection(value: unknown) {
  expect(validate(value)).toBe(false);
  expect(() => parseReferenceTrace(value)).toThrow();
}

describe('reference trace contract', () => {
  it('accepts committed first-transition and complete five-second trace scopes', () => {
    expectAgreement(firstTransition);
    expectAgreement(fullFiveSecond);
  });

  it('rejects incomplete required provenance in both validators', () => {
    expectRejection({ ...firstTransition, provenance: { ...provenance, torch: undefined } });
  });

  it('rejects invalid token and sampled-code ranges in both validators', () => {
    expectRejection({ ...firstTransition, input: { ...input, tokenRows: [[-1], [1]] } });
    expectRejection({ ...firstTransition, decisions: [{ ...decision, semanticCode: 16_384 }] });
    expectRejection({ ...firstTransition, decisions: [{ ...decision, residualCodes: [...decision.residualCodes.slice(0, 6), 1_024] }] });
  });

  it('rejects the wrong fixed prompt, token rows, checkpoint shape, and receipt contract', () => {
    expectRejection({ ...firstTransition, input: { ...input, rawPrompt: 'other' } });
    expectRejection({ ...firstTransition, input: { ...input, tokenRows: input.tokenRows.map((row) => [...row]) .map((row, index) => index ? row : [...row, 1]) } });
    expectRejection({ ...firstTransition, checkpoints: [{ ...firstTransition.checkpoints[0], shape: [1, 100] }, ...firstTransition.checkpoints.slice(1)] });
    expectRejection({ ...fullFiveSecond, tensorReceipts: [{ ...fullFiveSecond.tensorReceipts[0], dtype: 'float32' }, ...fullFiveSecond.tensorReceipts.slice(1)] });
  });

  it('requires portable relative tensor receipt paths in both validators', () => {
    for (const path of ['/outside.bin', 'C:/outside.bin', 'tensors\\outside.bin', 'tensors//outside.bin', 'tensors/outside.bin/', 'tensors/./outside.bin', 'tensors/../outside.bin', 'tensors/a\nb', 'tensors/a\rb', 'tensors/a\tb', 'tensors/a\0b', 'tensors/a\u007fb', 'tensors/a\u2028b', 'tensors/a\u2029b']) {
      expectRejection({
        ...fullFiveSecond,
        tensorReceipts: [{ ...fullFiveSecond.tensorReceipts[0], path }, ...fullFiveSecond.tensorReceipts.slice(1)],
      });
    }
  });

  it('requires exactly one first transition with seven residual codes', () => {
    expectRejection({ ...firstTransition, decisions: [] });
    expectRejection({ ...firstTransition, decisions: [{ ...decision, residualCodes: decision.residualCodes.slice(0, 6) }] });
  });

  it('requires the discarded start plus all 125 retained frame decisions without an early end', () => {
    expectRejection({ ...fullFiveSecond, decisions: fullFiveSecond.decisions.slice(0, 125) });
    expectRejection({ ...fullFiveSecond, termination: 'early-end' });
  });

  it('requires finite structured checkpoints and every full tensor receipt', () => {
    const logits = firstTransition.checkpoints[1];
    expectRejection({
      ...firstTransition,
      checkpoints: [
        firstTransition.checkpoints[0],
        { ...logits, values: [Infinity, ...logits.values.slice(1)] },
        ...firstTransition.checkpoints.slice(2),
      ],
    });
    expectRejection({ ...fullFiveSecond, tensorReceipts: fullFiveSecond.tensorReceipts.slice(0, -1) });
  });

  it('rejects finite values outside the declared float dtype range', () => {
    const logits = firstTransition.checkpoints[1];
    const hidden = firstTransition.checkpoints[4];
    expectRejection({
      ...firstTransition,
      checkpoints: [
        firstTransition.checkpoints[0],
        { ...logits, values: [3.402_823_47e38, ...logits.values.slice(1)] },
        ...firstTransition.checkpoints.slice(2),
      ],
    });
    expectRejection({
      ...firstTransition,
      checkpoints: [
        ...firstTransition.checkpoints.slice(0, 4),
        { ...hidden, values: [3.389_531_39e38, ...hidden.values.slice(1)] },
        ...firstTransition.checkpoints.slice(5),
      ],
    });
  });

  it('rejects duplicate, extra, and scope-inappropriate boundaries in both validators', () => {
    expectRejection({ ...firstTransition, checkpoints: [firstTransition.checkpoints[0], firstTransition.checkpoints[0], ...firstTransition.checkpoints.slice(2)] });
    expectRejection({ ...firstTransition, checkpoints: [firstTransition.checkpoints[0], { ...firstTransition.checkpoints[1], name: 'extra' }, ...firstTransition.checkpoints.slice(2)] });
    expectRejection({ ...firstTransition, checkpoints: [{ ...firstTransition.checkpoints[0], extra: true }, ...firstTransition.checkpoints.slice(1)] });
    expectRejection({ ...firstTransition, checkpoints: [...firstTransition.checkpoints.slice(0, -1), { ...firstTransition.checkpoints.at(-1), values: [40, 42] }] });
    expectRejection({ ...firstTransition, flowNoise: { kind: 'gaussian-flow-noise', receiptName: 'initial-flow-noise' } });
    expectRejection({ ...fullFiveSecond, checkpoints: [firstTransition.checkpoints[0]] });
    expectRejection({ ...fullFiveSecond, tensorReceipts: [fullFiveSecond.tensorReceipts[0], fullFiveSecond.tensorReceipts[0], ...fullFiveSecond.tensorReceipts.slice(2)] });
    expectRejection({ ...fullFiveSecond, tensorReceipts: [fullFiveSecond.tensorReceipts[0], { ...fullFiveSecond.tensorReceipts[1], name: 'extra' }, ...fullFiveSecond.tensorReceipts.slice(2)] });
    expectRejection({ ...fullFiveSecond, tensorReceipts: [{ ...fullFiveSecond.tensorReceipts[0], extra: true }, ...fullFiveSecond.tensorReceipts.slice(1)] });
  });

  it('keeps forced sampled codes distinct from Gaussian flow noise', () => {
    expectRejection({ ...firstTransition, decisions: [{ ...decision, kind: 'gaussian-flow-noise' }] });
    expectRejection({ ...fullFiveSecond, flowNoise: { kind: 'uniform-draws', receiptName: 'initial-flow-noise' } });
  });
});
