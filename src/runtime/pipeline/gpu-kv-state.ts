import type * as ort from 'onnxruntime-web/jspi';
import type { KvPairSpec } from '../model/manifest';
type GpuTensor = ort.Tensor & { location: 'gpu-buffer'; dispose(): void };
export class GpuKvState {
  private values = new Map<string, GpuTensor>();
  constructor(private readonly pairs: readonly KvPairSpec[]) {}
  inputs(): Record<string, GpuTensor> {
    return Object.fromEntries(this.pairs.flatMap((pair) => this.values.has(pair.pastInput) ? [[pair.pastInput, this.values.get(pair.pastInput)!]] : []));
  }
  advance(outputs: Record<string, ort.Tensor>) {
    const next = new Map<string, GpuTensor>();
    for (const pair of this.pairs) {
      const output = outputs[pair.presentOutput] as GpuTensor | undefined;
      if (!output || output.location !== 'gpu-buffer') throw new Error(`KV output ${pair.presentOutput} is not GPU-resident`);
      next.set(pair.pastInput, output);
    }
    this.values.forEach((tensor) => tensor.dispose());
    this.values = next;
  }
  dispose() { this.values.forEach((tensor) => tensor.dispose()); this.values.clear(); }
}
