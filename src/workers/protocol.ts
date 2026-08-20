export type WorkerRequest = { type: 'run-global-smoke'; manifestUrl: string };
export type WorkerProgress = { type: 'progress'; stage: 'manifest' | 'artifact' | 'adapter' | 'session'; detail: string; loaded?: number; total?: number };
export type GlobalSmokeResult = { adapter: string; graphInputs: readonly string[]; graphOutputs: readonly string[]; reducedHeadOutputs: readonly string[]; status: 'ready' };
export type WorkerResponse = WorkerProgress | { type: 'result'; result: GlobalSmokeResult } | { type: 'error'; message: string };
