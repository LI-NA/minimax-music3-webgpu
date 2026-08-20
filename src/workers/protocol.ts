export type WorkerRequest =
  | { type: 'run-global-smoke'; manifestUrl: string }
  | { type: 'run-rvq-smoke'; manifestUrl: string }
  | { type: 'run-condition-smoke'; manifestUrl: string }
  | {
      type: 'generate-frames';
      globalManifestUrl: string;
      rvqManifestUrl: string;
      maxFrames: number;
      seed: number;
    };
export type WorkerProgress = {
  type: 'progress';
  stage: 'manifest' | 'artifact' | 'adapter' | 'session';
  detail: string;
  loaded?: number;
  total?: number;
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
export type WorkerResponse =
  | WorkerProgress
  | { type: 'result'; result: GlobalSmokeResult }
  | { type: 'rvq-result'; result: RvqSmokeResult }
  | { type: 'frame-result'; result: FrameGenerationResult }
  | { type: 'condition-result'; result: ConditionSmokeResult }
  | { type: 'error'; message: string };
