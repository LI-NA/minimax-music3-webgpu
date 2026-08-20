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
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error(`${label} must be an object`);
  return value as Record<string, unknown>;
};
const integer = (value: unknown, label: string): number => {
  if (!Number.isSafeInteger(value) || (value as number) < 0)
    throw new Error(`${label} must be a non-negative integer`);
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
  if (typeof sha256 !== 'string' || !SHA.test(sha256))
    throw new Error(`${label} sha256 is invalid`);
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

export function parseModelManifest(value: unknown): ModelManifest {
  const root = object(value, 'manifest');
  if (root.schemaVersion !== 1) throw new Error('schemaVersion must be 1');
  const embeddingValue = object(root.embedding, 'embedding');
  const shards = embeddingValue.shards;
  if (!Array.isArray(shards) || !shards.length) throw new Error('embedding shards are invalid');
  const embedding = {
    rows: integer(embeddingValue.rows, 'embedding rows'),
    columns: integer(embeddingValue.columns, 'embedding columns'),
    rowBytes: integer(embeddingValue.rowBytes, 'embedding rowBytes'),
    shards: shards.map((entry) => {
      const item = object(entry, 'embedding shard');
      return {
        ...artifact(item, 'embedding shard'),
        rowStart: integer(item.rowStart, 'embedding shard rowStart'),
        rowCount: integer(item.rowCount, 'embedding shard rowCount'),
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
          (index === 0
            ? 0
            : embedding.shards[index - 1].rowStart + embedding.shards[index - 1].rowCount),
    ) ||
    embedding.shards.at(-1)!.rowStart + embedding.shards.at(-1)!.rowCount !== embedding.rows
  )
    throw new Error('embedding ranges are invalid');
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
  const webgpu = object(root.webgpu, 'webgpu');
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
    schemaVersion: 1,
    graph: graph(root.graph, 'graph'),
    reducedHead: graph(root.reducedHead, 'reducedHead'),
    embedding,
    kvPairs: root.kvPairs as KvPairSpec[],
    webgpu: {
      requiredFeatures: ['shader-f16'],
      requiredLimits: requiredLimits as Record<string, number>,
    },
  };
}
