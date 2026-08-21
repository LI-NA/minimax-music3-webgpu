import { describe, expect, it } from 'vitest';
import { createProductMusicRequest } from '../../../src/app/App';
import { FIXED_COMPARISON_CASE } from '../../../src/runtime/reference/fixed-comparison';

describe('music generation UI request', () => {
  it('uses the variable music route for the five-second product duration', () => {
    expect(createProductMusicRequest(5)).toEqual({
      type: 'generate-music',
      manifestUrl: 'http://127.0.0.1:5174/manifest.json',
      prompt: FIXED_COMPARISON_CASE.input.prompt,
      lyrics: FIXED_COMPARISON_CASE.input.lyrics,
      seed: 7,
      durationSeconds: 5,
      sampling: {
        globalGuidance: 1.5,
        semanticTopK: 50,
        residualTopK: 50,
        temperature: 1,
        flowGuidance: 1.7,
        flowSteps: 30,
      },
    });
  });
});
