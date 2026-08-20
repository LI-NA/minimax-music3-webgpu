import type { Fp16EmbeddingTable } from './manifest';
export interface SyncFileHandle { read(buffer: Uint8Array, options: { at: number }): number; close(): void; }
export class OpfsFp16EmbeddingTable {
  private constructor(private readonly table: Fp16EmbeddingTable, private readonly handles: readonly SyncFileHandle[]) {}
  static async open(table: Fp16EmbeddingTable, open: (path: string) => Promise<SyncFileHandle>): Promise<OpfsFp16EmbeddingTable> { const handles: SyncFileHandle[] = []; try { for (const shard of table.shards) handles.push(await open(shard.path)); return new OpfsFp16EmbeddingTable(table, handles); } catch (error) { handles.forEach((handle) => handle.close()); throw error; } }
  lookup(ids: readonly number[]): Uint16Array { const result = new Uint16Array(ids.length * this.table.columns); ids.forEach((id, resultIndex) => { if (!Number.isInteger(id) || id < 0 || id >= this.table.rows) throw new Error(`embedding id ${id} is out of range`); const index = this.table.shards.findIndex((shard) => id >= shard.rowStart && id < shard.rowStart + shard.rowCount); const shard = this.table.shards[index]; const target = new Uint8Array(result.buffer, result.byteOffset + resultIndex * this.table.rowBytes, this.table.rowBytes); const read = this.handles[index].read(target, { at: (id - shard.rowStart) * this.table.rowBytes }); if (read !== this.table.rowBytes) throw new Error(`embedding short read for id ${id}`); }); return result; }
  dispose() { this.handles.forEach((handle) => handle.close()); }
}
