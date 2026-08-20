export type WorkerRequest = { type: 'run-global-smoke'; manifestUrl: string };
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
export type WorkerResponse =
  | WorkerProgress
  | { type: 'result'; result: GlobalSmokeResult }
  | { type: 'error'; message: string };
