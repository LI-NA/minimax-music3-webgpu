import { describe, expect, it } from 'vitest';
import { OpfsFp16EmbeddingTable, type SyncFileHandle } from '../../../src/runtime/model/embedding-table';

function bytes(values: number[]) { return new Uint8Array(new Uint16Array(values).buffer); }

describe('OpfsFp16EmbeddingTable', () => {
  it('looks up duplicated ids across a shard boundary in input order', async () => {
    const files = new Map([['a', bytes([0, 1, 2, 3, 4, 5, 6, 7])], ['b', bytes([8, 9, 10, 11, 12, 13, 14, 15])]]);
    const table = await OpfsFp16EmbeddingTable.open({ rows: 4, columns: 4, rowBytes: 8, shards: [{ path: 'a', bytes: 16, sha256: 'a'.repeat(64), rowStart: 0, rowCount: 2 }, { path: 'b', bytes: 16, sha256: 'b'.repeat(64), rowStart: 2, rowCount: 2 }] }, async (path) => ({ read(buffer, options) { const source = files.get(path)!; buffer.set(source.slice(options.at, options.at + buffer.length)); return buffer.length; }, close() {} }));
    expect(Array.from(table.lookup([1, 2, 1]))).toEqual([4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7]);
  });

  it('rejects invalid IDs and short reads, then closes opened handles', async () => {
    let closed = 0;
    const make = async (): Promise<SyncFileHandle> => ({ read: () => 0, close: () => { closed++; } });
    const table = await OpfsFp16EmbeddingTable.open({ rows: 1, columns: 4, rowBytes: 8, shards: [{ path: 'a', bytes: 8, sha256: 'a'.repeat(64), rowStart: 0, rowCount: 1 }] }, make);
    expect(() => table.lookup([1])).toThrow('out of range'); expect(() => table.lookup([0])).toThrow('short read'); table.dispose();
    expect(closed).toBe(1);
  });

  it('closes an already-open shard when a later shard cannot open', async () => {
    let closed = 0; let calls = 0;
    const table = { rows: 2, columns: 4, rowBytes: 8, shards: [{ path: 'a', bytes: 8, sha256: 'a'.repeat(64), rowStart: 0, rowCount: 1 }, { path: 'b', bytes: 8, sha256: 'b'.repeat(64), rowStart: 1, rowCount: 1 }] };
    await expect(OpfsFp16EmbeddingTable.open(table, async () => { if (calls++ === 1) throw new Error('second shard failed'); return { read: () => 8, close: () => { closed++; } }; })).rejects.toThrow('second shard failed');
    expect(closed).toBe(1);
  });
});
