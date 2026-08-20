export type WorkerRequest =
  | { type: 'run-global-smoke'; manifestUrl: string }
  | { type: 'run-rvq-smoke'; manifestUrl: string }
  | { type: 'run-condition-smoke'; manifestUrl: string }
  | { type: 'run-flow-smoke'; manifestUrl: string }
  | { type: 'run-vocoder-smoke'; manifestUrl: string }
  | {
      type: 'generate-frames';
      globalManifestUrl: string;
      rvqManifestUrl: string;
      maxFrames: number;
      seed: number;
    }
  | { type: 'generate-music-5s'; manifestUrl: string; seed: number };
export type MusicStage = 'autoregressive' | 'condition' | 'flow' | 'vocoder' | 'wav';
export type WorkerProgress = {
  type: 'progress';
  stage: 'manifest' | 'artifact' | 'adapter' | 'session' | MusicStage | 'complete';
  detail: string;
  loaded?: number;
  total?: number;
  name?: MusicStage;
  activity?: 'indeterminate';
  currentFile?: string;
  completedBytes?: number;
  totalBytes?: number;
  cacheHit?: boolean;
  completed?: number;
  elapsedMs?: number;
  rate?: number;
  stepMs?: number;
  etaMs?: number;
  wavBytes?: number;
  totalElapsedMs?: number;
};
export type GlobalSmokeResult = {
  adapter: string;
  sessionCreateMs: number;
  stepMs: readonly number[];
  cacheLengths: readonly number[];
  tensorLocations: readonly string[];
  finiteLogits: boolean;
  ownedTensorBytes: number;
  artifactFetches: number;
  cacheReuseCount: number;
  status: 'passed';
};
export type RvqSmokeResult = {
  adapter: string;
  sessionCreateMs: number;
  lengths: readonly number[];
  stepMs: readonly number[];
  finiteLogits: boolean;
  hiddenLocations: readonly string[];
  feedbackLocation: string;
  artifactFetches: number;
  status: 'passed';
};
export type FrameGenerationResult = {
  adapter: string;
  frames: number;
  attemptedSeeds: readonly number[];
  semanticDecisions: number;
  rvqCalls: number;
  feedbackDecodes: number;
  cacheLengths: readonly number[];
  finiteHiddenGroups: boolean;
  codesInRange: boolean;
  hiddenBytes: number;
  artifactFetches: number;
  status: 'passed';
};
export type ConditionSmokeResult = {
  adapter: string;
  sessionCreateMs: number;
  elapsedMs: number;
  shape: readonly number[];
  outputLocation: string;
  finite: boolean;
  artifactFetches: number;
  status: 'passed';
};
export type FlowSmokeResult = {
  adapter: string;
  sessionCreateMs: number;
  oneStepMs: number;
  generationMs: number;
  stepMs: readonly number[];
  shape: readonly number[];
  oneStepLocation: string;
  finalLocation: string;
  oneStepFinite: boolean;
  finalFinite: boolean;
  artifactFetches: number;
  status: 'passed';
};
export type VocoderSmokeResult = {
  adapter: string;
  sessionCreateMs: number;
  generationMs: number;
  outputType: 'float32';
  shape: readonly [1, 2, 220160];
  finite: true;
  wavBytes: 880684;
  sampleRate: 44100;
  channels: 2;
  samples: 220160;
  bitsPerSample: 16;
  artifactFetches: number;
  status: 'passed';
};
export type MusicGenerationWorkerResult = {
  wav: ArrayBuffer;
  adapters: readonly string[];
  attemptedSeeds: readonly number[];
  hiddenBytes: number;
  conditionBytes: number;
  latentBytes: number;
  wavBytes: number;
  artifactBytes: number;
  artifactFetches: number;
  manifestHash: string;
  sessionCreateMs: Readonly<Record<'autoregressive' | 'condition' | 'flow' | 'vocoder', number>>;
  stageMs: Readonly<Record<'autoregressive' | 'condition' | 'flow' | 'vocoder', number>>;
  inferenceMs: Readonly<Record<'autoregressive' | 'condition' | 'flow' | 'vocoder', number>>;
  flowStepMs: readonly number[];
  browser: string;
  ortVersion: string;
  status: 'passed';
};
export type WorkerResponse =
  | WorkerProgress
  | { type: 'result'; result: GlobalSmokeResult }
  | { type: 'rvq-result'; result: RvqSmokeResult }
  | { type: 'frame-result'; result: FrameGenerationResult }
  | { type: 'condition-result'; result: ConditionSmokeResult }
  | { type: 'flow-result'; result: FlowSmokeResult }
  | { type: 'vocoder-result'; result: VocoderSmokeResult }
  | { type: 'music-result'; result: MusicGenerationWorkerResult }
  | { type: 'error'; message: string };
