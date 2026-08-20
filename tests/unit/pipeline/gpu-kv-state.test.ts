import { describe, expect, it } from 'vitest';
import { GpuKvState } from '../../../src/runtime/pipeline/gpu-kv-state';

function tensor(id: string) { return { location: 'gpu-buffer', dispose: () => disposed.push(id) }; }
const disposed: string[] = [];
describe('GpuKvState', () => {
  it('keeps GPU KV outputs through their dependent run and disposes superseded values afterwards', async () => {
    disposed.length = 0; const state = new GpuKvState([{ pastInput: 'past', presentOutput: 'present' }]);
    state.advance({ present: tensor('first') } as never); const feeds = state.inputs();
    expect(feeds.past).toBeTruthy(); expect(disposed).toEqual([]);
    state.advance({ present: tensor('second') } as never); expect(disposed).toEqual(['first']); state.dispose(); expect(disposed).toEqual(['first', 'second']);
  });
});
