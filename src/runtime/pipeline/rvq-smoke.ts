import type * as ort from 'onnxruntime-web/jspi';
import type { Fp16EmbeddingTable } from '../model/manifest';

export interface RvqSmokeRuntime {
  ort: typeof ort;
  rvqDepth: ort.InferenceSession;
  feedback: ort.InferenceSession;
  embedding: { lookup(ids: readonly number[]): Uint16Array; dispose(): void };
  embeddingTable: Pick<Fp16EmbeddingTable, 'columns'>;
}

export interface RvqSmokeMetrics {
  lengths: readonly number[];
  stepMs: readonly number[];
  finiteLogits: boolean;
  hiddenLocations: readonly string[];
  feedbackLocation: string;
}

const finiteFloat32 = (data: ort.Tensor['data']) =>
  Array.from(data as Float32Array).every(Number.isFinite);

export async function runRvqSmoke(runtime: RvqSmokeRuntime): Promise<RvqSmokeMetrics> {
  const columns = runtime.embeddingTable.columns;
  const globalHidden = new runtime.ort.Tensor('float16', new Uint16Array(2 * columns), [2, columns]);
  const semantic = new runtime.ort.Tensor('float16', new Uint16Array(2 * columns), [2, columns]);
  const residualIds = [0, 1024, 2048, 3072, 4096, 5120];
  const rows = runtime.embedding.lookup([...residualIds, ...residualIds]);
  const lengths: number[] = [];
  const stepMs: number[] = [];
  const hiddenLocations: string[] = [];
  let finiteLogits = true;
  try {
    for (let depthIndex = 0; depthIndex < 7; depthIndex++) {
      const residualData = new Uint16Array(2 * 6 * columns);
      for (let lane = 0; lane < 2; lane++)
        for (let position = 0; position < depthIndex; position++) {
          const source = (lane * 6 + position) * columns;
          residualData.set(rows.subarray(source, source + columns), source);
        }
      const residuals = new runtime.ort.Tensor('float16', residualData, [2, 6, columns]);
      const index = new runtime.ort.Tensor('int32', new Int32Array([depthIndex]), []);
      const started = performance.now();
      const outputs = await runtime.rvqDepth.run({
        global_last_hidden: globalHidden,
        semantic_embedding: semantic,
        residual_embeddings: residuals,
        depth_index: index,
      });
      stepMs.push(performance.now() - started);
      residuals.dispose();
      index.dispose();
      const hidden = outputs.depth_hidden;
      const logits = outputs.depth_logits;
      if (!hidden || !logits) throw new Error('RVQ depth outputs are incomplete');
      hiddenLocations.push(hidden.location);
      finiteLogits &&= finiteFloat32(await logits.getData());
      hidden.dispose();
      logits.dispose();
      lengths.push(depthIndex + 2);
    }
    const feedbackRows = runtime.embedding.lookup([
      0, 1024, 2048, 3072, 4096, 5120, 6144,
      0, 1024, 2048, 3072, 4096, 5120, 6144,
    ]);
    const semanticRows = new runtime.ort.Tensor('float16', new Uint16Array(2 * columns), [2, 1, columns]);
    const residualRows = new runtime.ort.Tensor('float16', feedbackRows, [2, 7, columns]);
    const feedbackOutputs = await runtime.feedback.run({ semantic_rows: semanticRows, residual_rows: residualRows });
    semanticRows.dispose();
    residualRows.dispose();
    const feedback = feedbackOutputs.inputs_embeds;
    if (!feedback) throw new Error('feedback output is missing');
    const feedbackLocation = feedback.location;
    feedback.dispose();
    if (!finiteLogits) throw new Error('RVQ logits are not finite');
    if (hiddenLocations.some((location) => location !== 'gpu-buffer'))
      throw new Error('RVQ hidden output is not GPU-resident');
    if (feedbackLocation !== 'gpu-buffer') throw new Error('feedback output is not GPU-resident');
    return { lengths, stepMs, finiteLogits, hiddenLocations, feedbackLocation };
  } finally {
    globalHidden.dispose();
    semantic.dispose();
    runtime.embedding.dispose();
  }
}
