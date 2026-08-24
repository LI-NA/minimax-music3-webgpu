export interface ArtifactFile {
  path: string;
  bytes: number;
  sha256: string;
}
export interface ExternalDataArtifact extends ArtifactFile {
  onnxLocation: string;
}
export interface OnnxGraphArtifact extends ArtifactFile {
  externalData: readonly ExternalDataArtifact[];
  gpuOutputs: readonly string[];
}
export interface EmbeddingShard extends ArtifactFile {
  rowStart: number;
  rowCount: number;
}
export interface Fp16EmbeddingTable {
  rows: number;
  columns: number;
  rowBytes: number;
  shards: readonly EmbeddingShard[];
}
export interface KvPairSpec {
  pastInput: string;
  presentOutput: string;
}
export interface ModelManifest {
  schemaVersion: 1;
  graph: OnnxGraphArtifact;
  reducedHead: OnnxGraphArtifact;
  embedding: Fp16EmbeddingTable;
  kvPairs: readonly KvPairSpec[];
  webgpu: {
    requiredFeatures: readonly ['shader-f16'];
    requiredLimits: Readonly<Record<string, number>>;
  };
}
export interface RvqStageManifest {
  schemaVersion: 1;
  rvqDepth: OnnxGraphArtifact;
  rvqEmbedding: Fp16EmbeddingTable;
  feedback: OnnxGraphArtifact;
  webgpu: ModelManifest['webgpu'];
}
export interface ConditionManifest {
  schemaVersion: 1;
  conditionEncoder: OnnxGraphArtifact;
  webgpu: ModelManifest['webgpu'];
}
export interface FlowManifest {
  schemaVersion: 1;
  flow: OnnxGraphArtifact;
  slice: {
    semanticFrames: 125;
    latentLength: 430;
    flowSteps: 30;
    flowGuidance: 1.7;
  };
  quantization: {
    bits: 4;
    blockSize: 32;
    accuracyLevel: 4;
    symmetric: true;
  };
  precision: {
    float16Weights: readonly ['time_proj.weight', 'proj_out.weight'];
  };
  webgpu: ModelManifest['webgpu'];
}
export interface VocoderManifest {
  schemaVersion: 1;
  vocoder: OnnxGraphArtifact;
  slice: {
    latentChannels: 128;
    latentLength: 430;
    outputSamples: 220160;
    sampleRate: 44100;
    channels: 2;
  };
  precision: {
    convolution: 'float16';
    fp32Snakes: readonly ['blocks.0.snake1', 'blocks.1.snake1'];
  };
  webgpu: ModelManifest['webgpu'];
}
export interface TensorContract {
  name: string;
  dtype: string;
  shape: readonly (number | string)[];
  maxShape?: readonly number[];
}
export interface ContractGraphArtifact extends OnnxGraphArtifact {
  inputs: readonly TensorContract[];
}
export interface MusicVariableManifest extends ModelManifest {
  model: {
    id: 'MiniMaxAI/MiniMax-Music3';
    revision: 'fbdf52fbaaca799592917417eb05f1899f1255ec';
    diffusersRevision: '3681e65996b4d2589219720101a6acbfd25073f8';
  };
  rvqDepth: OnnxGraphArtifact;
  rvqEmbedding: Fp16EmbeddingTable;
  feedback: OnnxGraphArtifact;
  conditionEncoder: ContractGraphArtifact;
  flow: ContractGraphArtifact;
  vocoder: ContractGraphArtifact & { outputs: readonly TensorContract[] };
  tokenizerFiles: readonly ArtifactFile[];
  licenseFile: ArtifactFile;
  quantization: FlowManifest['quantization'];
  precision: VocoderManifest['precision'] & {
    flowFp16Weights: readonly ['time_proj.weight', 'proj_out.weight'];
  };
  acoustic: {
    maxSemanticFrames: 200;
    windowFrames: 200;
    hopFrames: 100;
    overlapLatents: 172;
    leftCrop: 86;
    rightCrop: 258;
    samplesPerLatent: 512;
    maxLatentLength: 689;
    flowSteps: 30;
    flowGuidance: 1.7;
  };
}

const SHA = /^[a-f0-9]{64}$/;
const safePath = (value: unknown, label: string): string => {
  if (
    typeof value !== 'string' ||
    !value ||
    value.startsWith('/') ||
    value.includes('\\') ||
    value.split('/').some((part) => !part || part === '.' || part === '..')
  )
    throw new Error(`${label} path is invalid`);
  return value;
};
const object = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as Record<string, unknown>;
};
const integer = (value: unknown, label: string): number => {
  if (!Number.isSafeInteger(value) || (value as number) < 0) throw new Error(`${label} must be a non-negative integer`);
  return value as number;
};
const textList = (value: unknown, label: string): readonly string[] => {
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string' || !item))
    throw new Error(`${label} must be strings`);
  return value;
};
const artifact = (value: unknown, label: string): ArtifactFile => {
  const item = object(value, label);
  const sha256 = item.sha256;
  if (typeof sha256 !== 'string' || !SHA.test(sha256)) throw new Error(`${label} sha256 is invalid`);
  return {
    path: safePath(item.path, label),
    bytes: integer(item.bytes, label),
    sha256,
  };
};
const graph = (value: unknown, label: string): OnnxGraphArtifact => {
  const item = object(value, label);
  const base = artifact(item, label);
  if (!Array.isArray(item.externalData)) throw new Error(`${label} externalData is invalid`);
  return {
    ...base,
    externalData: item.externalData.map((entry) => {
      const ext = object(entry, `${label} externalData`);
      return {
        ...artifact(ext, `${label} externalData`),
        onnxLocation: safePath(ext.onnxLocation, `${label} externalData`),
      };
    }),
    gpuOutputs: textList(item.gpuOutputs, `${label} gpuOutputs`),
  };
};

const tensorContract = (value: unknown, label: string): TensorContract => {
  const item = object(value, label);
  if (typeof item.name !== 'string' || !item.name || typeof item.dtype !== 'string' || !item.dtype)
    throw new Error(`${label} is invalid`);
  if (
    !Array.isArray(item.shape) ||
    item.shape.some(
      (part) =>
        !(typeof part === 'string' && part.length > 0) && !(Number.isSafeInteger(part) && (part as number) >= 0),
    )
  )
    throw new Error(`${label} shape is invalid`);
  if (
    item.maxShape !== undefined &&
    (!Array.isArray(item.maxShape) || item.maxShape.some((part) => !Number.isSafeInteger(part) || part < 0))
  )
    throw new Error(`${label} maxShape is invalid`);
  return {
    name: item.name,
    dtype: item.dtype,
    shape: item.shape as (number | string)[],
    ...(item.maxShape === undefined ? {} : { maxShape: item.maxShape as number[] }),
  };
};

const contractGraph = (value: unknown, label: string): ContractGraphArtifact => {
  const item = object(value, label);
  if (!Array.isArray(item.inputs)) throw new Error(`${label} inputs are invalid`);
  return {
    ...graph(value, label),
    inputs: item.inputs.map((entry) => tensorContract(entry, `${label} input`)),
  };
};

const exactContracts = (actual: readonly TensorContract[], expected: readonly TensorContract[]) =>
  JSON.stringify(actual) === JSON.stringify(expected);

const embeddingTable = (value: unknown, label: string): Fp16EmbeddingTable => {
  const embeddingValue = object(value, label);
  const shards = embeddingValue.shards;
  if (!Array.isArray(shards) || !shards.length) throw new Error(`${label} shards are invalid`);
  const embedding = {
    rows: integer(embeddingValue.rows, `${label} rows`),
    columns: integer(embeddingValue.columns, `${label} columns`),
    rowBytes: integer(embeddingValue.rowBytes, `${label} rowBytes`),
    shards: shards.map((entry) => {
      const item = object(entry, `${label} shard`);
      return {
        ...artifact(item, `${label} shard`),
        rowStart: integer(item.rowStart, `${label} shard rowStart`),
        rowCount: integer(item.rowCount, `${label} shard rowCount`),
      };
    }),
  };
  if (
    !embedding.rows ||
    !embedding.columns ||
    embedding.rowBytes !== embedding.columns * 2 ||
    embedding.shards.some(
      (shard, index) =>
        shard.rowCount === 0 ||
        shard.bytes !== shard.rowCount * embedding.rowBytes ||
        shard.rowStart !==
          (index === 0 ? 0 : embedding.shards[index - 1].rowStart + embedding.shards[index - 1].rowCount),
    ) ||
    embedding.shards.at(-1)!.rowStart + embedding.shards.at(-1)!.rowCount !== embedding.rows
  )
    throw new Error(`${label} ranges are invalid`);
  return embedding;
};

const webgpuContract = (value: unknown): ModelManifest['webgpu'] => {
  const webgpu = object(value, 'webgpu');
  if (
    !Array.isArray(webgpu.requiredFeatures) ||
    webgpu.requiredFeatures.length !== 1 ||
    webgpu.requiredFeatures[0] !== 'shader-f16'
  )
    throw new Error('webgpu requiredFeatures must be [shader-f16]');
  const requiredLimits = object(webgpu.requiredLimits, 'webgpu requiredLimits');
  for (const [name, value] of Object.entries(requiredLimits)) {
    if (!name || typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0)
      throw new Error('webgpu requiredLimits are invalid');
  }
  return {
    requiredFeatures: ['shader-f16'],
    requiredLimits: requiredLimits as Record<string, number>,
  };
};

export function parseModelManifest(value: unknown): ModelManifest {
  const root = object(value, 'manifest');
  if (root.schemaVersion !== 1) throw new Error('schemaVersion must be 1');
  const embedding = embeddingTable(root.embedding, 'embedding');
  if (
    !Array.isArray(root.kvPairs) ||
    root.kvPairs.some((pair) => {
      const item = object(pair, 'kv pair');
      return (
        typeof item.pastInput !== 'string' ||
        !item.pastInput ||
        typeof item.presentOutput !== 'string' ||
        !item.presentOutput
      );
    })
  )
    throw new Error('kvPairs are invalid');
  const webgpu = webgpuContract(root.webgpu);
  const reducedHead = graph(root.reducedHead, 'reducedHead');
  if (!reducedHead.gpuOutputs.includes('last_state')) throw new Error('reducedHead must keep last_state at gpu-buffer');
  return {
    schemaVersion: 1,
    graph: graph(root.graph, 'graph'),
    reducedHead,
    embedding,
    kvPairs: root.kvPairs as KvPairSpec[],
    webgpu: {
      requiredFeatures: ['shader-f16'],
      requiredLimits: webgpu.requiredLimits,
    },
  };
}

export function parseRvqStageManifest(value: unknown): RvqStageManifest {
  const root = object(value, 'manifest');
  if (root.schemaVersion !== 1) throw new Error('schemaVersion must be 1');
  const rvqDepth = graph(root.rvqDepth, 'rvqDepth');
  const feedback = graph(root.feedback, 'feedback');
  if (!rvqDepth.gpuOutputs.includes('depth_hidden')) throw new Error('rvqDepth must keep depth_hidden at gpu-buffer');
  if (!feedback.gpuOutputs.includes('inputs_embeds')) throw new Error('feedback must keep inputs_embeds at gpu-buffer');
  return {
    schemaVersion: 1,
    rvqDepth,
    rvqEmbedding: embeddingTable(root.rvqEmbedding, 'rvqEmbedding'),
    feedback,
    webgpu: webgpuContract(root.webgpu),
  };
}

export function parseConditionManifest(value: unknown): ConditionManifest {
  const root = object(value, 'manifest');
  if (root.schemaVersion !== 1) throw new Error('schemaVersion must be 1');
  const conditionEncoder = graph(root.conditionEncoder, 'conditionEncoder');
  if (!conditionEncoder.gpuOutputs.includes('condition'))
    throw new Error('conditionEncoder must keep condition at gpu-buffer');
  return {
    schemaVersion: 1,
    conditionEncoder,
    webgpu: webgpuContract(root.webgpu),
  };
}

export function parseFlowManifest(value: unknown): FlowManifest {
  const root = object(value, 'manifest');
  if (root.schemaVersion !== 1) throw new Error('schemaVersion must be 1');
  const flow = graph(root.flow, 'flow');
  if (flow.gpuOutputs.length !== 1 || flow.gpuOutputs[0] !== 'next_latents')
    throw new Error('flow must keep next_latents at gpu-buffer');
  const slice = object(root.slice, 'flow slice');
  if (
    slice.semanticFrames !== 125 ||
    slice.latentLength !== 430 ||
    slice.flowSteps !== 30 ||
    slice.flowGuidance !== 1.7
  )
    throw new Error('flow slice does not match the fixed contract');
  const quantization = object(root.quantization, 'flow quantization');
  if (
    quantization.bits !== 4 ||
    quantization.blockSize !== 32 ||
    quantization.accuracyLevel !== 4 ||
    quantization.symmetric !== true
  )
    throw new Error('flow quantization does not match the q4 contract');
  const precision = object(root.precision, 'flow precision');
  if (
    !Array.isArray(precision.float16Weights) ||
    precision.float16Weights.length !== 2 ||
    precision.float16Weights[0] !== 'time_proj.weight' ||
    precision.float16Weights[1] !== 'proj_out.weight'
  )
    throw new Error('flow precision does not match the selective float16 contract');
  const webgpu = webgpuContract(root.webgpu);
  if (
    webgpu.requiredLimits.maxStorageBufferBindingSize !== 128 * 1024 * 1024 ||
    (webgpu.requiredLimits.maxStorageBuffersPerShaderStage ?? 0) < 9
  )
    throw new Error('flow webgpu requiredLimits are invalid');
  return {
    schemaVersion: 1,
    flow,
    slice: {
      semanticFrames: 125,
      latentLength: 430,
      flowSteps: 30,
      flowGuidance: 1.7,
    },
    quantization: {
      bits: 4,
      blockSize: 32,
      accuracyLevel: 4,
      symmetric: true,
    },
    precision: { float16Weights: ['time_proj.weight', 'proj_out.weight'] },
    webgpu,
  };
}

export function parseVocoderManifest(value: unknown): VocoderManifest {
  const root = object(value, 'manifest');
  if (root.schemaVersion !== 1) throw new Error('schemaVersion must be 1');
  const vocoder = graph(root.vocoder, 'vocoder');
  if (vocoder.gpuOutputs.length !== 0) throw new Error('vocoder waveform must remain a CPU output');
  const slice = object(root.slice, 'vocoder slice');
  if (
    slice.latentChannels !== 128 ||
    slice.latentLength !== 430 ||
    slice.outputSamples !== 220_160 ||
    slice.sampleRate !== 44_100 ||
    slice.channels !== 2
  )
    throw new Error('vocoder slice does not match the fixed contract');
  const precision = object(root.precision, 'vocoder precision');
  if (
    precision.convolution !== 'float16' ||
    !Array.isArray(precision.fp32Snakes) ||
    precision.fp32Snakes.length !== 2 ||
    precision.fp32Snakes[0] !== 'blocks.0.snake1' ||
    precision.fp32Snakes[1] !== 'blocks.1.snake1'
  )
    throw new Error('vocoder precision does not match the mixed contract');
  const webgpu = webgpuContract(root.webgpu);
  if (webgpu.requiredLimits.maxStorageBufferBindingSize !== 128 * 1024 * 1024)
    throw new Error('vocoder webgpu requiredLimits are invalid');
  return {
    schemaVersion: 1,
    vocoder,
    slice: {
      latentChannels: 128,
      latentLength: 430,
      outputSamples: 220_160,
      sampleRate: 44_100,
      channels: 2,
    },
    precision: {
      convolution: 'float16',
      fp32Snakes: ['blocks.0.snake1', 'blocks.1.snake1'],
    },
    webgpu,
  };
}

export function parseMusicVariableManifest(value: unknown): MusicVariableManifest {
  const root = object(value, 'manifest');
  const model = object(root.model, 'model');
  if (model.id !== 'MiniMaxAI/MiniMax-Music3') throw new Error('model id is invalid');
  if (model.revision !== 'fbdf52fbaaca799592917417eb05f1899f1255ec') throw new Error('model revision is invalid');
  if (model.diffusersRevision !== '3681e65996b4d2589219720101a6acbfd25073f8')
    throw new Error('Diffusers revision is invalid');
  const global = parseModelManifest(root);
  const rvq = parseRvqStageManifest(root);
  const webgpu = webgpuContract(root.webgpu);
  if (
    webgpu.requiredLimits.maxStorageBufferBindingSize !== 128 * 1024 * 1024 ||
    (webgpu.requiredLimits.maxStorageBuffersPerShaderStage ?? 0) < 9
  )
    throw new Error('music variable webgpu requiredLimits are invalid');
  const quantization = object(root.quantization, 'music variable quantization');
  if (
    quantization.bits !== 4 ||
    quantization.blockSize !== 32 ||
    quantization.accuracyLevel !== 4 ||
    quantization.symmetric !== true
  )
    throw new Error('music variable quantization is invalid');
  const precision = object(root.precision, 'music variable precision');
  if (
    precision.convolution !== 'float16' ||
    !Array.isArray(precision.fp32Snakes) ||
    precision.fp32Snakes.length !== 2 ||
    precision.fp32Snakes[0] !== 'blocks.0.snake1' ||
    precision.fp32Snakes[1] !== 'blocks.1.snake1' ||
    !Array.isArray(precision.flowFp16Weights) ||
    precision.flowFp16Weights.length !== 2 ||
    precision.flowFp16Weights[0] !== 'time_proj.weight' ||
    precision.flowFp16Weights[1] !== 'proj_out.weight'
  )
    throw new Error('music variable precision is invalid');
  const acoustic = object(root.acoustic, 'music variable acoustic');
  const expectedAcoustic = {
    maxSemanticFrames: 200,
    windowFrames: 200,
    hopFrames: 100,
    overlapLatents: 172,
    leftCrop: 86,
    rightCrop: 258,
    samplesPerLatent: 512,
    maxLatentLength: 689,
    flowSteps: 30,
    flowGuidance: 1.7,
  } as const;
  if (Object.entries(expectedAcoustic).some(([name, expected]) => acoustic[name] !== expected))
    throw new Error('music variable acoustic contract is invalid');
  const conditionEncoder = contractGraph(root.conditionEncoder, 'conditionEncoder');
  const flow = contractGraph(root.flow, 'flow');
  const vocoderValue = object(root.vocoder, 'vocoder');
  const vocoder = contractGraph(root.vocoder, 'vocoder');
  if (!Array.isArray(vocoderValue.outputs)) throw new Error('vocoder outputs are invalid');
  const vocoderOutputs = vocoderValue.outputs.map((entry) => tensorContract(entry, 'vocoder output'));
  const expectedCondition: TensorContract[] = [
    { name: 'frame_hiddens', dtype: 'float16', shape: [1, 200, 32768] },
    { name: 'nearest_index', dtype: 'int64', shape: [689] },
    { name: 'active_latent_mask', dtype: 'float16', shape: [1, 689, 1] },
  ];
  const expectedFlow: TensorContract[] = [
    { name: 'latents', dtype: 'float16', shape: [1, 128, 689] },
    { name: 'condition', dtype: 'float16', shape: [1, 689, 2048] },
    { name: 'timestep', dtype: 'float16', shape: [1] },
    { name: 'dt', dtype: 'float32', shape: [1] },
    { name: 'active_latent_mask', dtype: 'float16', shape: [1, 689, 1] },
    { name: 'key_attention_bias', dtype: 'float16', shape: [1, 1, 1, 690] },
    { name: 'noise_prompt', dtype: 'float16', shape: [1, 128, 172] },
    { name: 'previous_latent', dtype: 'float16', shape: [1, 128, 172] },
    { name: 'overlap_enabled', dtype: 'float16', shape: [1] },
    { name: 'guidance', dtype: 'float16', shape: [1] },
  ];
  const expectedVocoderInputs: TensorContract[] = [
    { name: 'latents', dtype: 'float16', shape: [1, 64, 'L'], maxShape: [1, 64, 689] },
  ];
  const expectedVocoderOutputs: TensorContract[] = [
    { name: 'waveform', dtype: 'float32', shape: [1, 1, '512L'], maxShape: [1, 1, 352768] },
  ];
  if (!exactContracts(conditionEncoder.inputs, expectedCondition))
    throw new Error('condition inputs do not match the variable contract');
  if (!exactContracts(flow.inputs, expectedFlow)) throw new Error('flow inputs do not match the variable contract');
  if (!exactContracts(vocoder.inputs, expectedVocoderInputs) || !exactContracts(vocoderOutputs, expectedVocoderOutputs))
    throw new Error('vocoder symbolic mono contract is invalid');
  if (!conditionEncoder.gpuOutputs.includes('condition'))
    throw new Error('conditionEncoder must keep condition at gpu-buffer');
  if (flow.gpuOutputs.length !== 1 || flow.gpuOutputs[0] !== 'next_latents')
    throw new Error('flow must keep next_latents at gpu-buffer');
  if (vocoder.gpuOutputs.length) throw new Error('vocoder waveform must remain a CPU output');
  if (!Array.isArray(root.tokenizerFiles) || !root.tokenizerFiles.length) throw new Error('tokenizerFiles are invalid');
  const tokenizerFiles = root.tokenizerFiles.map((item) => artifact(item, 'tokenizer file'));
  const licenseFile = artifact(root.licenseFile, 'license file');
  const allArtifacts = [
    global.graph,
    global.reducedHead,
    rvq.rvqDepth,
    rvq.feedback,
    conditionEncoder,
    flow,
    vocoder,
    ...global.graph.externalData,
    ...global.reducedHead.externalData,
    ...rvq.rvqDepth.externalData,
    ...rvq.feedback.externalData,
    ...conditionEncoder.externalData,
    ...flow.externalData,
    ...vocoder.externalData,
    ...global.embedding.shards,
    ...rvq.rvqEmbedding.shards,
    ...tokenizerFiles,
    licenseFile,
  ];
  const artifactMetadata = new Map<string, ArtifactFile>();
  for (const item of allArtifacts) {
    const previous = artifactMetadata.get(item.path);
    if (previous && (previous.bytes !== item.bytes || previous.sha256 !== item.sha256))
      throw new Error('music variable duplicate artifact path has conflicting metadata');
    artifactMetadata.set(item.path, item);
  }
  if (allArtifacts.some((item) => item.bytes > 128 * 1024 * 1024))
    throw new Error('music variable release artifact exceeds 128 MiB');
  return {
    ...global,
    model: model as MusicVariableManifest['model'],
    rvqDepth: rvq.rvqDepth,
    rvqEmbedding: rvq.rvqEmbedding,
    feedback: rvq.feedback,
    conditionEncoder,
    flow,
    vocoder: { ...vocoder, outputs: vocoderOutputs },
    tokenizerFiles,
    licenseFile,
    acoustic: expectedAcoustic,
    quantization: { bits: 4, blockSize: 32, accuracyLevel: 4, symmetric: true },
    precision: {
      convolution: 'float16',
      fp32Snakes: ['blocks.0.snake1', 'blocks.1.snake1'],
      flowFp16Weights: ['time_proj.weight', 'proj_out.weight'],
    },
  };
}
