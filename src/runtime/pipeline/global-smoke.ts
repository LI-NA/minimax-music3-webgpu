import type * as ort from 'onnxruntime-web/jspi';
import promptContract from '../../../tests/fixtures/prompt-contract.json';
import type { Fp16EmbeddingTable, KvPairSpec } from '../model/manifest';
import { GpuKvState } from './gpu-kv-state';

const promptLength = 40;
const batchSize = 2;
const diagnosticEmbeddingId = 151675;

export interface GlobalSmokeMetrics {
  sessionCreateMs: number;
  stepMs: readonly number[];
  cacheLengths: readonly number[];
  tensorLocations: readonly string[];
  finiteLogits: boolean;
  ownedTensorBytes: number;
}
export interface GlobalSmokeRuntime {
  ort: typeof ort;
  decoder: ort.InferenceSession;
  head: ort.InferenceSession;
  embedding: { lookup(ids: readonly number[]): Uint16Array; dispose(): void };
  embeddingTable: Pick<Fp16EmbeddingTable, 'columns'>;
  kvPairs: readonly KvPairSpec[];
}

const now = () => performance.now();
const tensorData = async (tensor: ort.Tensor) => tensor.getData();
const finite = (data: ort.Tensor['data']) =>
  data instanceof Uint16Array
    ? data.every((value) => (value & 0x7c00) !== 0x7c00)
    : Array.from(data as Float32Array).every(Number.isFinite);

function embeddings(runtime: GlobalSmokeRuntime, ids: readonly number[], length: number) {
  return new runtime.ort.Tensor('float16', runtime.embedding.lookup(ids), [batchSize, length, runtime.embeddingTable.columns]);
}
function initialCaches(runtime: GlobalSmokeRuntime): Record<string, ort.Tensor> {
  return Object.fromEntries(
    runtime.kvPairs.map(({ pastInput }) => [
      pastInput,
      new runtime.ort.Tensor('float16', new Uint16Array(), [batchSize, 8, 0, 128]),
    ]),
  );
}
function decoderFeeds(
  runtime: GlobalSmokeRuntime,
  ids: readonly number[],
  length: number,
  totalLength: number,
  caches: Record<string, ort.Tensor>,
) {
  return {
    inputs_embeds: embeddings(runtime, ids, length),
    seqlens_k: new runtime.ort.Tensor('int32', new Int32Array(batchSize).fill(totalLength - 1), [batchSize]),
    total_seq_len: new runtime.ort.Tensor('int32', new Int32Array([totalLength]), []),
    ...caches,
  };
}
function cacheLength(outputs: Record<string, ort.Tensor>, pairs: readonly KvPairSpec[]) {
  const lengths = pairs.map((pair) => outputs[pair.presentOutput].dims[2]);
  if (lengths.some((length) => length === undefined || length !== lengths[0]))
    throw new Error('decoder KV cache lengths disagree');
  return lengths[0]!;
}
async function reducedLogits(
  runtime: GlobalSmokeRuntime,
  hidden: ort.Tensor,
): Promise<{ finite: boolean; locations: string[] }> {
  const input = runtime.head.inputNames[0];
  const outputs = await runtime.head.run({ [input]: hidden });
  try {
    const values = await Promise.all(Object.values(outputs).map(tensorData));
    return {
      finite: values.every(finite),
      locations: Object.values(outputs).map((tensor) => tensor.location),
    };
  } finally {
    Object.values(outputs).forEach((tensor) => tensor.dispose());
  }
}
async function runDecoder(
  runtime: GlobalSmokeRuntime,
  feeds: Record<string, ort.Tensor>,
): Promise<Record<string, ort.Tensor>> {
  const ownedInputs = Object.entries(feeds).filter(
    ([name]) => !runtime.kvPairs.some((pair) => pair.pastInput === name),
  );
  try {
    return await runtime.decoder.run(feeds);
  } finally {
    ownedInputs.forEach(([, tensor]) => tensor.dispose());
  }
}

export async function runGlobalSmoke(runtime: GlobalSmokeRuntime): Promise<GlobalSmokeMetrics> {
  const cache = new GpuKvState(runtime.kvPairs);
  const stepMs: number[] = [];
  const cacheLengths: number[] = [];
  const locations: string[] = [];
  let finiteLogits = true;
  let ownedTensorBytes = 0;
  try {
    const prompt = promptContract;
    if (prompt.conditional.length !== promptLength || prompt.unconditional.length !== promptLength)
      throw new Error('Prompt contract must contain exactly 40 tokens per lane');
    const initialCacheInputs = initialCaches(runtime);
    const prefill = decoderFeeds(
      runtime,
      [...prompt.conditional, ...prompt.unconditional],
      promptLength,
      promptLength,
      initialCacheInputs,
    );
    const firstStarted = now();
    const first = await runDecoder(runtime, prefill);
    stepMs.push(now() - firstStarted);
    const hidden = first.hidden_states;
    if (!hidden || hidden.location !== 'gpu-buffer') throw new Error('decoder hidden output is not GPU-resident');
    try {
      const logits = await reducedLogits(runtime, hidden);
      finiteLogits &&= logits.finite;
      locations.push(hidden.location, ...logits.locations);
    } finally {
      hidden.dispose();
    }
    cache.advance(first);
    Object.values(initialCacheInputs).forEach((tensor) => tensor.dispose());
    cacheLengths.push(cacheLength(first, runtime.kvPairs));
    ownedTensorBytes = Math.max(ownedTensorBytes, cache.ownedBytes());
    for (let step = 0; step < 10; step++) {
      const started = now();
      const outputs = await runDecoder(
        runtime,
        decoderFeeds(
          runtime,
          [diagnosticEmbeddingId, diagnosticEmbeddingId],
          1,
          promptLength + step + 1,
          cache.inputs(),
        ),
      );
      stepMs.push(now() - started);
      const hidden = outputs.hidden_states;
      if (!hidden || hidden.location !== 'gpu-buffer') throw new Error('decoder hidden output is not GPU-resident');
      try {
        const logits = await reducedLogits(runtime, hidden);
        finiteLogits &&= logits.finite;
        locations.push(hidden.location, ...logits.locations);
      } finally {
        hidden.dispose();
      }
      cache.advance(outputs);
      cacheLengths.push(cacheLength(outputs, runtime.kvPairs));
      ownedTensorBytes = Math.max(ownedTensorBytes, cache.ownedBytes());
    }
    if (!finiteLogits) throw new Error('reduced logits are not finite');
    return { sessionCreateMs: 0, stepMs, cacheLengths, tensorLocations: locations, finiteLogits, ownedTensorBytes };
  } finally {
    cache.dispose();
    runtime.embedding.dispose();
  }
}
