type Scope = 'first-transition' | 'full-5s';
type RecordValue = Record<string, unknown>;

export interface TensorReceipt {
  name: string;
  path: string;
  dtype: 'float32' | 'float16' | 'bfloat16' | 'int32' | 'uint8';
  shape: readonly number[];
  bytes: number;
  sha256: string;
}
export interface ReferenceTrace {
  schemaVersion: 2;
  scope: Scope;
  provenance: {
    model: { id: 'MiniMaxAI/MiniMax-Music3'; revision: 'fbdf52fbaaca799592917417eb05f1899f1255ec' };
    diffusersRevision: '3681e65996b4d2589219720101a6acbfd25073f8';
    python: string;
    torch: string;
    transformers: string;
    cudaRuntime: string;
    driver: string;
    gpu: string;
    generatorDevice: string;
    combinedManifestSha256: string;
    sourceReceiptSha256: string;
  };
  input: {
    rawPrompt: string;
    lyrics: string;
    assembledText: string;
    tokenRows: readonly (readonly number[])[];
  };
  parameters: {
    seed: 7;
    audioDuration: 5;
    retainedFrames: 125;
    globalGuidance: 1.5;
    semanticTopK: 50;
    residualTopK: 50;
    flowGuidance: 1.7;
    flowSteps: 30;
  };
  decisions: readonly {
    kind: 'forced-sampled-codes';
    semanticCode: number;
    residualCodes: readonly number[];
  }[];
  flowNoise?: { kind: 'gaussian-flow-noise'; receiptName: 'initial-flow-noise' };
  checkpoints: readonly {
    name: string;
    dtype: 'float32' | 'float16' | 'bfloat16' | 'int32';
    shape: readonly number[];
    values: readonly number[];
  }[];
  tensorReceipts: readonly TensorReceipt[];
  termination: 'fixed-length-no-early-end';
}

const SHA256 = /^[a-f0-9]{64}$/;
const REQUIRED_CHECKPOINTS: Record<string, readonly [string, readonly number[]]> = {
  'semantic-topk-ids': ['int32', [2, 50]],
  'semantic-topk-logits': ['float32', [2, 50]],
  'residual-topk-ids': ['int32', [7, 2, 50]],
  'residual-topk-logits': ['float32', [7, 2, 50]],
  'conditional-frame-hidden': ['bfloat16', [8, 4096]],
  feedback: ['bfloat16', [2, 1, 4096]],
  'cache-lengths': ['int32', [2]],
};
const RECEIPTS: Record<string, readonly [TensorReceipt['dtype'], readonly number[], number]> = {
  'frame-handoff': ['bfloat16', [1, 125, 32768], 8192000],
  condition: ['bfloat16', [1, 430, 2048], 1761280],
  'initial-flow-noise': ['bfloat16', [1, 128, 430], 110080],
  'flow-step-1': ['bfloat16', [1, 128, 430], 110080],
  'flow-step-15': ['bfloat16', [1, 128, 430], 110080],
  'flow-step-30': ['bfloat16', [1, 128, 430], 110080],
  'final-latent': ['bfloat16', [1, 128, 430], 110080],
  waveform: ['float32', [1, 2, 220160], 1761280],
  wav: ['uint8', [880684], 880684],
};
const same = (left: readonly number[], right: readonly number[]) =>
  left.length === right.length && left.every((value, index) => value === right[index]);
const object = (value: unknown, label: string, keys?: readonly string[]): RecordValue => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  const parsed = value as RecordValue;
  if (keys && Object.keys(parsed).some((key) => !keys.includes(key)))
    throw new Error(`${label} has unexpected properties`);
  return parsed;
};
const text = (value: unknown, label: string): string => {
  if (typeof value !== 'string' || !value) throw new Error(`${label} must be non-empty text`);
  return value;
};
const finite = (value: unknown, label: string): number => {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label} must be finite`);
  return value;
};
const integer = (value: unknown, label: string, maximum?: number): number => {
  if (!Number.isSafeInteger(value) || (value as number) < 0 || (maximum !== undefined && (value as number) > maximum))
    throw new Error(`${label} is out of range`);
  return value as number;
};
const sha256 = (value: unknown, label: string): string => {
  const parsed = text(value, label);
  if (!SHA256.test(parsed)) throw new Error(`${label} is invalid`);
  return parsed;
};
const shape = (value: unknown, label: string): number[] => {
  if (!Array.isArray(value) || !value.length) throw new Error(`${label} must be a non-empty array`);
  return value.map((dimension) => {
    const parsed = integer(dimension, label);
    if (!parsed) throw new Error(`${label} must be positive`);
    return parsed;
  });
};
const relativePath = (value: unknown, label: string): string => {
  const parsed = text(value, label);
  const hasControlCharacter = [...parsed].some((character) => {
    const code = character.charCodeAt(0);
    return code < 32 || code === 127 || code === 0x2028 || code === 0x2029;
  });
  if (
    hasControlCharacter ||
    parsed.startsWith('/') ||
    parsed.includes('\\') ||
    parsed.includes(':') ||
    parsed.endsWith('/') ||
    parsed.split('/').some((part) => !part || part === '.' || part === '..')
  )
    throw new Error(`${label} must be a relative path`);
  return parsed;
};

function parseReceipt(value: unknown): TensorReceipt {
  const receipt = object(value, 'tensor receipt', ['name', 'path', 'dtype', 'shape', 'bytes', 'sha256']);
  const dtype = receipt.dtype;
  if (dtype !== 'float32' && dtype !== 'float16' && dtype !== 'bfloat16' && dtype !== 'int32' && dtype !== 'uint8')
    throw new Error('tensor receipt dtype is invalid');
  const parsedShape = shape(receipt.shape, 'tensor receipt shape');
  const bytes = integer(receipt.bytes, 'tensor receipt bytes');
  if (!bytes) throw new Error('tensor receipt bytes must be positive');
  return {
    name: text(receipt.name, 'tensor receipt name'),
    path: relativePath(receipt.path, 'tensor receipt path'),
    dtype,
    shape: parsedShape,
    bytes,
    sha256: sha256(receipt.sha256, 'tensor receipt sha256'),
  };
}

export function parseReferenceTrace(value: unknown): ReferenceTrace {
  const root = object(value, 'trace', [
    'schemaVersion',
    'scope',
    'provenance',
    'input',
    'parameters',
    'decisions',
    'flowNoise',
    'checkpoints',
    'tensorReceipts',
    'termination',
  ]);
  if (root.schemaVersion !== 2) throw new Error('schemaVersion must be 2');
  const scope = root.scope;
  if (scope !== 'first-transition' && scope !== 'full-5s') throw new Error('scope is invalid');
  const provenance = object(root.provenance, 'provenance', [
    'model',
    'diffusersRevision',
    'python',
    'torch',
    'transformers',
    'cudaRuntime',
    'driver',
    'gpu',
    'generatorDevice',
    'combinedManifestSha256',
    'sourceReceiptSha256',
  ]);
  const model = object(provenance.model, 'provenance model', ['id', 'revision']);
  if (model.id !== 'MiniMaxAI/MiniMax-Music3' || model.revision !== 'fbdf52fbaaca799592917417eb05f1899f1255ec')
    throw new Error('provenance model is invalid');
  if (provenance.diffusersRevision !== '3681e65996b4d2589219720101a6acbfd25073f8')
    throw new Error('provenance Diffusers revision is invalid');
  const input = object(root.input, 'input', ['rawPrompt', 'lyrics', 'assembledText', 'tokenRows']);
  const expectedPrompt = 'Global\nbpm is 96\nWarm female vocal';
  const expectedLyrics = '[verse]\nHello\n[chorus]\nStay\nTogether\n[bridge][solo]';
  const expectedText =
    '<|im_start|><|caption_start|>Global\nbpm is 96\nWarm female vocal<|caption_end|><|lyrics_start|>[start]\n[verse]\nHello\n[chorus]\nStay\nTogether\n[bridge][solo]<|lyrics_end|><|im_end|><|audio_start|>';
  if (
    input.rawPrompt !== expectedPrompt ||
    input.lyrics !== expectedLyrics ||
    input.assembledText !== expectedText ||
    provenance.combinedManifestSha256 !== '5c295ebfb4b7849d317cf0abd3dd8bfc9da3b58dc74de12a3523c07f28d4500e'
  )
    throw new Error('trace is not the committed fixed case');
  if (
    !Array.isArray(input.tokenRows) ||
    input.tokenRows.length !== 2 ||
    input.tokenRows.some((row) => !Array.isArray(row) || !row.length)
  )
    throw new Error('input token rows are invalid');
  const tokenRows = input.tokenRows.map((row) =>
    (row as unknown[]).map((token) => integer(token, 'input token', 151_935)),
  );
  const tokens = [
    [
      151644, 151671, 11646, 198, 65, 5187, 374, 220, 24, 21, 198, 95275, 8778, 25407, 151672, 151673, 28463, 921, 58,
      4450, 921, 9707, 198, 58, 6150, 355, 921, 38102, 198, 80987, 198, 58, 13709, 1457, 82, 10011, 60, 151674, 151645,
      151669,
    ],
    [151644, ...Array(37).fill(151654), 151645, 151669],
  ];
  if (!same(tokenRows[0], tokens[0]) || !same(tokenRows[1], tokens[1]))
    throw new Error('input token rows are not the committed fixed case');
  const parameters = object(root.parameters, 'parameters', [
    'seed',
    'audioDuration',
    'retainedFrames',
    'globalGuidance',
    'semanticTopK',
    'residualTopK',
    'flowGuidance',
    'flowSteps',
  ]);
  if (
    parameters.seed !== 7 ||
    parameters.audioDuration !== 5 ||
    parameters.retainedFrames !== 125 ||
    parameters.globalGuidance !== 1.5 ||
    parameters.semanticTopK !== 50 ||
    parameters.residualTopK !== 50 ||
    parameters.flowGuidance !== 1.7 ||
    parameters.flowSteps !== 30
  )
    throw new Error('parameters do not match the fixed five-second contract');
  if (!Array.isArray(root.decisions)) throw new Error('decisions must be an array');
  const decisions = root.decisions.map((value) => {
    const decision = object(value, 'decision', ['kind', 'semanticCode', 'residualCodes']);
    if (
      decision.kind !== 'forced-sampled-codes' ||
      !Array.isArray(decision.residualCodes) ||
      decision.residualCodes.length !== 7
    )
      throw new Error('decision must contain seven forced sampled residual codes');
    return {
      kind: 'forced-sampled-codes' as const,
      semanticCode: integer(decision.semanticCode, 'semantic code', 16_383),
      residualCodes: decision.residualCodes.map((code) => integer(code, 'residual code', 1_023)),
    };
  });
  if (decisions.length !== (scope === 'first-transition' ? 1 : 126))
    throw new Error('decision count is inconsistent with scope');
  let parsedFlowNoise: ReferenceTrace['flowNoise'];
  if (scope === 'full-5s') {
    const flowNoise = object(root.flowNoise, 'flow noise', ['kind', 'receiptName']);
    if (flowNoise.kind !== 'gaussian-flow-noise' || flowNoise.receiptName !== 'initial-flow-noise')
      throw new Error('flow noise must reference the Gaussian initial receipt');
    parsedFlowNoise = { kind: 'gaussian-flow-noise', receiptName: 'initial-flow-noise' };
  } else if (root.flowNoise !== undefined) throw new Error('first-transition does not include flow noise');
  if (!Array.isArray(root.checkpoints) || !Array.isArray(root.tensorReceipts))
    throw new Error('checkpoints and tensor receipts must be arrays');
  const checkpoints = root.checkpoints.map((value) => {
    const checkpoint = object(value, 'checkpoint', ['name', 'dtype', 'shape', 'values']);
    const parsedShape = shape(checkpoint.shape, 'checkpoint shape');
    if (!Array.isArray(checkpoint.values)) throw new Error('checkpoint values must be an array');
    const values = checkpoint.values.map((entry) => finite(entry, 'checkpoint value'));
    if (parsedShape.reduce((size, dimension) => size * dimension, 1) !== values.length)
      throw new Error('checkpoint shape does not match values');
    const name = text(checkpoint.name, 'checkpoint name');
    const contract = REQUIRED_CHECKPOINTS[name];
    if (
      !contract ||
      checkpoint.dtype !== contract[0] ||
      !same(parsedShape, contract[1]) ||
      (name === 'cache-lengths' && (values[0] !== 40 || values[1] !== 41))
    )
      throw new Error('checkpoint contract is invalid');
    const bound =
      checkpoint.dtype === 'float16'
        ? 65_504
        : checkpoint.dtype === 'bfloat16'
          ? 3.389_531_389_251_535_5e38
          : 3.402_823_466_385_288_6e38;
    if (
      checkpoint.dtype === 'int32' &&
      values.some(
        (entry) =>
          !Number.isSafeInteger(entry) ||
          entry < 0 ||
          (name.includes('semantic') && entry > 16384) ||
          (name.includes('residual') && entry > 1023),
      )
    )
      throw new Error('checkpoint integer values are invalid');
    if (checkpoint.dtype !== 'int32' && values.some((entry) => Math.abs(entry) > bound))
      throw new Error('checkpoint values exceed dtype range');
    return {
      name,
      dtype: checkpoint.dtype as 'float32' | 'float16' | 'bfloat16' | 'int32',
      shape: parsedShape,
      values,
    };
  });
  const tensorReceipts = root.tensorReceipts.map(parseReceipt);
  if (scope === 'first-transition') {
    if (
      checkpoints.length !== Object.keys(REQUIRED_CHECKPOINTS).length ||
      Object.keys(REQUIRED_CHECKPOINTS).some(
        (name) => checkpoints.filter((checkpoint) => checkpoint.name === name).length !== 1,
      ) ||
      tensorReceipts.length
    )
      throw new Error('first-transition checkpoints are incomplete');
  } else if (
    checkpoints.length ||
    tensorReceipts.length !== Object.keys(RECEIPTS).length ||
    Object.entries(RECEIPTS).some(
      ([name, contract]) =>
        tensorReceipts.filter((receipt) => receipt.name === name).length !== 1 ||
        !tensorReceipts.some(
          (receipt) =>
            receipt.name === name &&
            receipt.dtype === contract[0] &&
            same(receipt.shape, contract[1]) &&
            receipt.bytes === contract[2],
        ),
    )
  ) {
    throw new Error('full trace tensor receipts are incomplete');
  }
  if (root.termination !== 'fixed-length-no-early-end') throw new Error('trace must not end early');
  return {
    schemaVersion: 2,
    scope,
    provenance: {
      model: {
        id: 'MiniMaxAI/MiniMax-Music3',
        revision: 'fbdf52fbaaca799592917417eb05f1899f1255ec',
      },
      diffusersRevision: '3681e65996b4d2589219720101a6acbfd25073f8',
      python: text(provenance.python, 'provenance python'),
      torch: text(provenance.torch, 'provenance torch'),
      transformers: text(provenance.transformers, 'provenance transformers'),
      cudaRuntime: text(provenance.cudaRuntime, 'provenance cudaRuntime'),
      driver: text(provenance.driver, 'provenance driver'),
      gpu: text(provenance.gpu, 'provenance gpu'),
      generatorDevice: text(provenance.generatorDevice, 'provenance generatorDevice'),
      combinedManifestSha256: sha256(provenance.combinedManifestSha256, 'provenance combinedManifestSha256'),
      sourceReceiptSha256: sha256(provenance.sourceReceiptSha256, 'provenance sourceReceiptSha256'),
    },
    input: {
      rawPrompt: text(input.rawPrompt, 'input rawPrompt'),
      lyrics: text(input.lyrics, 'input lyrics'),
      assembledText: text(input.assembledText, 'input assembledText'),
      tokenRows,
    },
    parameters: {
      seed: 7,
      audioDuration: 5,
      retainedFrames: 125,
      globalGuidance: 1.5,
      semanticTopK: 50,
      residualTopK: 50,
      flowGuidance: 1.7,
      flowSteps: 30,
    },
    decisions,
    flowNoise: parsedFlowNoise,
    checkpoints,
    tensorReceipts,
    termination: 'fixed-length-no-early-end',
  };
}
