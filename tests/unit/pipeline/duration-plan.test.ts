import fixture from '../../fixtures/duration-plans.json';
import { describe, expect, it } from 'vitest';
import {
  planDuration,
  planRetainedFrames,
  type DurationPlanRequest,
  type Termination,
} from '../../../src/runtime/pipeline/duration-plan';

describe('duration planning', () => {
  it('matches the hand-derived official plans', () => {
    for (const entry of fixture.cases) {
      expect(JSON.stringify(planDuration({
        durationSeconds: entry.durationSeconds,
        promptTokens: entry.promptTokens,
      }))).toBe(JSON.stringify(entry.plan));
    }
  });

  it('plans actual retained frames independently of the requested duration', () => {
    for (const entry of fixture.retainedFrameCases) {
      expect(JSON.stringify(planRetainedFrames({ ...entry, termination: entry.termination as Termination }))).toBe(JSON.stringify(entry.plan));
    }
  });

  it('exposes max-frame counts separately from a natural end', () => {
    expect(planRetainedFrames({ retainedFrames: 201, promptTokens: 40, termination: 'max-frames' })).toMatchObject({
      termination: 'max-frames', semanticDecisions: 202, rvqCalls: 1414, feedbackCalls: 201,
    });
  });

  it('uses an explicit flow step count while preserving the 30-step default', () => {
    expect(planRetainedFrames({
      retainedFrames: 201,
      promptTokens: 40,
      termination: 'max-frames',
      flowSteps: 12,
    }).flowCalls).toBe(24);
    expect(planDuration({ durationSeconds: 10, promptTokens: 40, flowSteps: 12 }).flowCalls).toBe(24);
    expect(planDuration({ durationSeconds: 10, promptTokens: 40 }).flowCalls).toBe(60);
  });

  it('rejects flow step counts that are not positive safe integers', () => {
    for (const flowSteps of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, Number.POSITIVE_INFINITY]) {
      expect(() => planDuration({ durationSeconds: 5, promptTokens: 0, flowSteps } as never)).toThrow('positive safe integer');
      expect(() => planRetainedFrames({
        retainedFrames: 1,
        promptTokens: 0,
        termination: 'max-frames',
        flowSteps,
      } as never)).toThrow('positive safe integer');
    }
  });

  it('floors arbitrary durations to frames and derives exact output sizes', () => {
    const diagnosticFlagRemoved: 'allowDiagnosticDuration' extends keyof DurationPlanRequest ? false : true = true;
    expect(diagnosticFlagRemoved).toBe(true);
    expect(planDuration({ durationSeconds: 5.04, promptTokens: 0 })).toMatchObject({
      retainedFrames: 126,
      samplesPerChannel: 222_208,
      wavBytes: 888_876,
    });
    expect(planDuration({ durationSeconds: 6, promptTokens: 0 })).toMatchObject({
      retainedFrames: 150,
      samplesPerChannel: 264_192,
      wavBytes: 1_056_812,
    });
    expect(planDuration({ durationSeconds: 10.5, promptTokens: 0 })).toMatchObject({
      retainedFrames: 262,
      samplesPerChannel: 462_336,
      wavBytes: 1_849_388,
    });
  });

  it('rejects invalid durations and durations shorter than one frame', () => {
    for (const durationSeconds of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, 300.01]) {
      expect(() => planDuration({ durationSeconds, promptTokens: 0 })).toThrow('greater than zero and at most 300');
    }
    expect(() => planDuration({ durationSeconds: 0.039, promptTokens: 0 })).toThrow('at least one frame');
    expect(planDuration({ durationSeconds: 0.04, promptTokens: 0 }).retainedFrames).toBe(1);
  });

  it('rejects requests that exceed the language-model context', () => {
    expect(() => planDuration({ durationSeconds: 300, promptTokens: 2741 })).toThrow('10240');
    expect(planRetainedFrames({ retainedFrames: 7500, promptTokens: 2740, termination: 'max-frames' }).retainedFrames).toBe(7500);
    expect(() => planRetainedFrames({ retainedFrames: 7500, promptTokens: 2741, termination: 'max-frames' })).toThrow('10240');
  });

  it('accepts every product duration step', () => {
    for (let durationSeconds = 5; durationSeconds <= 300; durationSeconds += 5) {
      expect(planDuration({ durationSeconds, promptTokens: 0 }).retainedFrames).toBe(durationSeconds * 25);
    }
  });

  it('rejects non-integer, negative, and non-boolean runtime inputs', () => {
    expect(() => planDuration({ durationSeconds: true, promptTokens: 0 } as never)).toThrow('number');
    for (const request of [
      { durationSeconds: 5, promptTokens: -1 },
      { durationSeconds: 5, promptTokens: true },
    ]) {
      expect(() => planDuration(request as never)).toThrow('integers');
    }
    for (const request of [
      { retainedFrames: 1.5, promptTokens: 0, termination: 'max-frames' },
      { retainedFrames: true, promptTokens: 0, termination: 'max-frames' },
      { retainedFrames: -1, promptTokens: 0, termination: 'max-frames' },
      { retainedFrames: 1, promptTokens: 0.5, termination: 'max-frames' },
      { retainedFrames: 1, promptTokens: true, termination: 'max-frames' },
    ]) {
      expect(() => planRetainedFrames(request as never)).toThrow('integers');
    }
    expect(() => planRetainedFrames({ retainedFrames: 0, promptTokens: 0, termination: 'max-frames' })).toThrow('between 1 and 7500');
    expect(() => planRetainedFrames({ retainedFrames: 7501, promptTokens: 0, termination: 'max-frames' })).toThrow('between 1 and 7500');
    expect(() => planRetainedFrames({ retainedFrames: 1, promptTokens: 0 } as never)).toThrow('termination');
    expect(() => planRetainedFrames({ retainedFrames: 1, promptTokens: 0, termination: 1 } as never)).toThrow('termination');
    expect(() => planRetainedFrames({ retainedFrames: 1, promptTokens: 0, termination: 'other' } as never)).toThrow('termination');
  });
});
