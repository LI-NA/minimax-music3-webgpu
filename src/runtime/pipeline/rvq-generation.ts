import type * as ort from 'onnxruntime-web/jspi';
import type { KvPairSpec } from '../model/manifest';
import { readGpuFp16Bits } from '../model/fp16-readback';
import { GpuKvState } from './gpu-kv-state';
import { planRetainedFrames, type RetainedFramesPlan, type Termination } from './duration-plan';
import { createDeterministicDraw, sampleResidual, sampleSemantic, sampleSemanticExcludingAudioEnd } from './sampler';

const batchSize = 2;
const hiddenSize = 4096;
const audioCodeOffset = 151_675;
const residualVocabulary = 1_024;

export interface GeneratedFrame {
  semantic: number;
  residual: readonly [number, number, number, number, number, number, number];
  hiddenGroups: Uint16Array;
}

export type GeneratedFrames = readonly GeneratedFrame[] & {
  hiddenGroups: Uint16Array;
  termination: Termination;
  plan: RetainedFramesPlan;
  capacityDiagnostic?: CapacityDiagnosticMetadata;
};

export type CapacityDiagnosticMetadata = {
  kind: 'continue-after-audio-end';
  suppressedAudioEnds: number;
  firstAudioEndAtRetainedFrame: number | null;
};

export interface GenerateFrameOptions {
  maxFrames: number;
  seed: number;
  promptTokenRows: {
    conditional: readonly number[];
    unconditional: readonly number[];
  };
  guidance: number;
  semanticTopK: number;
  residualTopK: number;
  temperature: number;
  audioEndPolicy?: 'continue-for-capacity-diagnostic';
}

type Embedding = { lookup(ids: readonly number[]): Uint16Array; dispose(): void };
type HiddenReader = (tensor: ort.Tensor) => Promise<Uint16Array>;

export interface FrameGenerationRuntime {
  ort: typeof ort;
  decoder: ort.InferenceSession;
  head: ort.InferenceSession;
  rvqDepth: ort.InferenceSession;
  feedback: ort.InferenceSession;
  globalEmbedding: Embedding;
  rvqEmbedding: Embedding;
  embeddingColumns: number;
  kvPairs: readonly KvPairSpec[];
  readConditionalHidden: HiddenReader;
  onCacheLength?: (length: number) => void;
  onFrameRetained?: (count: number) => void;
}

export class EarlyAudioEndError extends Error {
  constructor(
    public readonly seed: number,
    public readonly retainedFrames: number,
  ) {
    super(`audio end sampled after ${retainedFrames} retained frames with seed ${seed}`);
  }
}

const fp16 = (value: number) => {
  const sign = value >>> 15 ? -1 : 1;
  const exponent = (value >>> 10) & 31;
  const fraction = value & 1023;
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
};

async function floatData(tensor: ort.Tensor) {
  const data = await tensor.getData();
  if (data instanceof Float32Array) return data;
  if (data instanceof Float16Array) return Float32Array.from(data);
  if (data instanceof Uint16Array) return Float32Array.from(data, fp16);
  throw new Error(`expected float logits, received ${data.constructor.name}`);
}

function initialCaches(runtime: FrameGenerationRuntime) {
  return Object.fromEntries(
    runtime.kvPairs.map(({ pastInput }) => [
      pastInput,
      new runtime.ort.Tensor('float16', new Uint16Array(), [batchSize, 8, 0, 128]),
    ]),
  );
}

function decoderLengths(runtime: FrameGenerationRuntime, totalLength: number) {
  return {
    seqlens_k: new runtime.ort.Tensor('int32', new Int32Array(batchSize).fill(totalLength - 1), [batchSize]),
    total_seq_len: new runtime.ort.Tensor('int32', new Int32Array([totalLength]), []),
  };
}

function reportCacheLength(runtime: FrameGenerationRuntime, outputs: Record<string, ort.Tensor>) {
  const length = outputs[runtime.kvPairs[0].presentOutput]?.dims[2];
  if (length === undefined) throw new Error('decoder cache length is missing');
  runtime.onCacheLength?.(length);
}

async function sampleGlobal(
  runtime: FrameGenerationRuntime,
  hidden: ort.Tensor,
  draw: () => number,
  options: GenerateFrameOptions,
) {
  let outputs: Record<string, ort.Tensor>;
  try {
    outputs = await runtime.head.run({ hidden_states: hidden });
  } finally {
    hidden.dispose();
  }
  const semantic = outputs.semantic_logits;
  const end = outputs.end_logit;
  const lastState = outputs.last_state;
  const disposeOutputs = () => new Set([semantic, end, lastState]).forEach((tensor) => tensor?.dispose());
  if (!semantic || !end || !lastState) {
    disposeOutputs();
    throw new Error('reduced head outputs are incomplete');
  }
  if (lastState.location !== 'gpu-buffer') {
    disposeOutputs();
    throw new Error('Global last_state is not GPU-resident');
  }
  try {
    const semanticValues = await floatData(semantic);
    const endValues = await floatData(end);
    const conditional = new Float32Array(16_385);
    const unconditional = new Float32Array(16_385);
    conditional.set(semanticValues.subarray(0, 16_384));
    unconditional.set(semanticValues.subarray(16_384, 32_768));
    conditional[16_384] = endValues[0];
    unconditional[16_384] = endValues[1];
    const sampling = {
      guidance: options.guidance,
      topK: options.semanticTopK,
      temperature: options.temperature,
      draw,
    };
    const sampled = sampleSemantic(conditional, unconditional, sampling);
    const suppressedAudioEnd = sampled === 16_384 && options.audioEndPolicy === 'continue-for-capacity-diagnostic';
    return {
      code: suppressedAudioEnd ? sampleSemanticExcludingAudioEnd(conditional, unconditional, sampling) : sampled,
      lastState,
      suppressedAudioEnd,
    };
  } catch (error) {
    lastState.dispose();
    throw error;
  } finally {
    semantic.dispose();
    end.dispose();
  }
}

function residualInputs(runtime: FrameGenerationRuntime, codes: readonly number[]) {
  const result = new Uint16Array(batchSize * 6 * hiddenSize);
  if (!codes.length) return result;
  const ids = Array.from({ length: batchSize }, () =>
    codes.map((code, index) => code + index * residualVocabulary),
  ).flat();
  const rows = runtime.rvqEmbedding.lookup(ids);
  for (let lane = 0; lane < batchSize; lane++)
    for (let position = 0; position < codes.length; position++) {
      const source = (lane * codes.length + position) * hiddenSize;
      const target = (lane * 6 + position) * hiddenSize;
      result.set(rows.subarray(source, source + hiddenSize), target);
    }
  return result;
}

function feedbackResiduals(runtime: FrameGenerationRuntime, codes: readonly number[]) {
  const ids = Array.from({ length: batchSize }, () =>
    codes.map((code, index) => code + index * residualVocabulary),
  ).flat();
  return runtime.rvqEmbedding.lookup(ids);
}

export function createFrameGenerator(runtime: FrameGenerationRuntime) {
  if (runtime.embeddingColumns !== hiddenSize) throw new Error('embedding width must be 4096');
  return {
    async generateFrames(options: GenerateFrameOptions): Promise<GeneratedFrames> {
      if (!Number.isInteger(options.maxFrames) || options.maxFrames < 1)
        throw new Error('maxFrames must be a positive integer');
      if (!Number.isInteger(options.seed) || options.seed < 0 || options.seed > 0xffff_ffff)
        throw new Error('seed must be a uint32 integer');
      if (!Number.isFinite(options.guidance) || options.guidance < 0)
        throw new Error('guidance must be finite and non-negative');
      if (!Number.isInteger(options.semanticTopK) || options.semanticTopK < 1 || options.semanticTopK > 16_385)
        throw new Error('semanticTopK must be an integer between 1 and 16385');
      if (!Number.isInteger(options.residualTopK) || options.residualTopK < 1 || options.residualTopK > 1_024)
        throw new Error('residualTopK must be an integer between 1 and 1024');
      if (!Number.isFinite(options.temperature) || options.temperature <= 0)
        throw new Error('temperature must be finite and greater than zero');
      const { conditional, unconditional } = options.promptTokenRows ?? {};
      if (
        !Array.isArray(conditional) ||
        !Array.isArray(unconditional) ||
        conditional.length === 0 ||
        conditional.length !== unconditional.length
      )
        throw new Error('prompt token rows must have the same positive length');
      if (conditional.length > 5_000) throw new Error('prompt token rows must not exceed 5000 tokens');
      if ([...conditional, ...unconditional].some((token) => !Number.isSafeInteger(token) || token < 0))
        throw new Error('prompt token rows must contain non-negative integers');
      const promptLength = conditional.length;
      if (options.audioEndPolicy !== undefined && options.audioEndPolicy !== 'continue-for-capacity-diagnostic')
        throw new Error('audioEndPolicy is invalid');
      planRetainedFrames({
        retainedFrames: options.maxFrames,
        promptTokens: promptLength,
        termination: 'max-frames',
      });
      const draw = createDeterministicDraw(options.seed);
      const cache = new GpuKvState(runtime.kvPairs);
      const frames: GeneratedFrame[] = [];
      const hiddenGroups = new Uint16Array(options.maxFrames * 8 * hiddenSize);
      let termination: Termination = 'max-frames';
      let suppressedAudioEnds = 0;
      let firstAudioEndAtRetainedFrame: number | null = null;
      const recordSuppression = (suppressed: boolean) => {
        if (!suppressed) return;
        firstAudioEndAtRetainedFrame ??= frames.length;
        suppressedAudioEnds++;
      };
      let lastState: ort.Tensor | undefined;
      try {
        const prompt = [...conditional, ...unconditional];
        const promptEmbedding = new runtime.ort.Tensor('float16', runtime.globalEmbedding.lookup(prompt), [
          batchSize,
          promptLength,
          hiddenSize,
        ]);
        const emptyCaches = initialCaches(runtime);
        const lengths = decoderLengths(runtime, promptLength);
        let prefill: Record<string, ort.Tensor>;
        try {
          prefill = await runtime.decoder.run({
            inputs_embeds: promptEmbedding,
            ...lengths,
            ...emptyCaches,
          });
        } finally {
          promptEmbedding.dispose();
          lengths.seqlens_k.dispose();
          lengths.total_seq_len.dispose();
          Object.values(emptyCaches).forEach((tensor) => tensor.dispose());
        }
        cache.advance(prefill);
        reportCacheLength(runtime, prefill);
        const initial = await sampleGlobal(runtime, prefill.hidden_states, draw, options);
        lastState = initial.lastState;
        let semanticCode = initial.code;
        recordSuppression(initial.suppressedAudioEnd);

        for (let frameIndex = 0; frameIndex <= options.maxFrames; frameIndex++) {
          if (semanticCode === 16_384) {
            if (!frames.length) throw new EarlyAudioEndError(options.seed, 0);
            termination = 'natural-end';
            break;
          }
          const retain = frameIndex > 0;
          const frameOffset = frames.length * 8 * hiddenSize;
          const groups = retain ? hiddenGroups.subarray(frameOffset, frameOffset + 8 * hiddenSize) : undefined;
          if (groups) groups.set(await runtime.readConditionalHidden(lastState), 0);
          const semanticRows = runtime.globalEmbedding.lookup([
            audioCodeOffset + semanticCode,
            audioCodeOffset + semanticCode,
          ]);
          const semanticEmbedding = new runtime.ort.Tensor('float16', semanticRows, [batchSize, hiddenSize]);
          const residual: number[] = [];
          try {
            for (let depth = 0; depth < 7; depth++) {
              const residualEmbedding = new runtime.ort.Tensor('float16', residualInputs(runtime, residual), [
                batchSize,
                6,
                hiddenSize,
              ]);
              const depthIndex = new runtime.ort.Tensor('int32', new Int32Array([depth]), []);
              let outputs: Record<string, ort.Tensor>;
              try {
                outputs = await runtime.rvqDepth.run({
                  global_last_hidden: lastState,
                  semantic_embedding: semanticEmbedding,
                  residual_embeddings: residualEmbedding,
                  depth_index: depthIndex,
                });
              } finally {
                residualEmbedding.dispose();
                depthIndex.dispose();
              }
              const hidden = outputs.depth_hidden;
              const logits = outputs.depth_logits;
              if (!hidden || !logits) throw new Error('RVQ depth outputs are incomplete');
              try {
                const values = await floatData(logits);
                const laneSize = 7 * residualVocabulary;
                const start = depth * residualVocabulary;
                residual.push(
                  sampleResidual(
                    values.slice(start, start + residualVocabulary),
                    values.slice(laneSize + start, laneSize + start + residualVocabulary),
                    {
                      guidance: options.guidance,
                      topK: options.residualTopK,
                      temperature: options.temperature,
                      draw,
                    },
                  ),
                );
                if (groups) groups.set(await runtime.readConditionalHidden(hidden), (depth + 1) * hiddenSize);
              } finally {
                hidden.dispose();
                logits.dispose();
              }
            }
          } finally {
            semanticEmbedding.dispose();
          }
          if (groups)
            frames.push({
              semantic: semanticCode,
              residual: residual as unknown as GeneratedFrame['residual'],
              hiddenGroups: groups,
            });
          if (groups) runtime.onFrameRetained?.(frames.length);
          if (frames.length === options.maxFrames) break;

          const feedbackSemantic = new runtime.ort.Tensor('float16', semanticRows, [batchSize, 1, hiddenSize]);
          const feedbackResidual = new runtime.ort.Tensor('float16', feedbackResiduals(runtime, residual), [
            batchSize,
            7,
            hiddenSize,
          ]);
          let feedbackOutputs: Record<string, ort.Tensor>;
          try {
            feedbackOutputs = await runtime.feedback.run({
              semantic_rows: feedbackSemantic,
              residual_rows: feedbackResidual,
            });
          } finally {
            feedbackSemantic.dispose();
            feedbackResidual.dispose();
          }
          const nextEmbedding = feedbackOutputs.inputs_embeds;
          if (!nextEmbedding || nextEmbedding.location !== 'gpu-buffer')
            throw new Error('feedback inputs_embeds is not GPU-resident');
          lastState.dispose();
          lastState = undefined;
          const totalLength = promptLength + frameIndex + 1;
          const nextLengths = decoderLengths(runtime, totalLength);
          let decoded: Record<string, ort.Tensor>;
          try {
            decoded = await runtime.decoder.run({
              inputs_embeds: nextEmbedding,
              ...nextLengths,
              ...cache.inputs(),
            });
          } finally {
            nextEmbedding.dispose();
            nextLengths.seqlens_k.dispose();
            nextLengths.total_seq_len.dispose();
          }
          cache.advance(decoded);
          reportCacheLength(runtime, decoded);
          const next = await sampleGlobal(runtime, decoded.hidden_states, draw, options);
          lastState = next.lastState;
          semanticCode = next.code;
          recordSuppression(next.suppressedAudioEnd);
        }
        const usedHiddenGroups = hiddenGroups.subarray(0, frames.length * 8 * hiddenSize);
        return Object.assign(frames, {
          hiddenGroups: usedHiddenGroups,
          termination,
          plan: planRetainedFrames({
            retainedFrames: frames.length,
            promptTokens: promptLength,
            termination,
          }),
          ...(options.audioEndPolicy === 'continue-for-capacity-diagnostic'
            ? {
                capacityDiagnostic: {
                  kind: 'continue-after-audio-end' as const,
                  suppressedAudioEnds,
                  firstAudioEndAtRetainedFrame,
                },
              }
            : {}),
        });
      } finally {
        lastState?.dispose();
        cache.dispose();
        runtime.globalEmbedding.dispose();
        runtime.rvqEmbedding.dispose();
      }
    },
  };
}

export async function readConditionalGpuFp16(tensor: ort.Tensor) {
  const bothLanes = await readGpuFp16Bits(tensor, [batchSize, hiddenSize], 'hidden');
  return bothLanes.slice(0, hiddenSize);
}

export function areFiniteFp16(values: Uint16Array) {
  return values.every((value) => (value & 0x7c00) !== 0x7c00);
}
