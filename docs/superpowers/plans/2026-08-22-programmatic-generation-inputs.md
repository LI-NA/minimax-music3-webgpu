# Programmatic generation inputs implementation plan

> Execute in the current primary checkout without Git mutations. Preserve the dirty working tree and every artifact generation. Use one RED, GREEN, review cycle per behavior boundary.

**Status:** Completed. The active release passed headed Chrome 6 and 10-second raw-input generation with zero completed-artifact fetches. Release and archive identifiers are recorded in the development documentation.

**Spec:** `docs/superpowers/specs/2026-08-22-programmatic-generation-inputs-design.md`

**Goal:** Replace product hardcoded generation values with one validated raw-input worker path, while leaving UI work and checkpoint structure constants out of scope.

## Task 1: Add exact browser prompt preparation

**Files:**

- Add the pinned lightweight tokenizer dependency.
- Create `src/runtime/pipeline/prompt-preparation.ts`.
- Create focused unit tests and update the fixed tokenizer oracle.

**RED:** Fixed raw prompt and lyrics do not yet reproduce the existing assembled string and two 40-token rows through a browser tokenizer.

**GREEN:** Port the pinned caption and lyric preprocessing exactly, load tokenizer JSON through an injectable adapter, build both CFG rows, and validate token and context limits.

## Task 2: Define the strict product request

**Files:**

- Modify `src/workers/protocol.ts` and protocol tests.
- Update non-UI callers and browser fixtures that construct product requests.

**RED:** The product request currently trusts caller-provided `promptTokens`, contains no raw prompt or lyrics, and cannot carry sampling settings.

**GREEN:** Require raw prompt, lyrics, uint32 seed, any valid duration through 300 seconds, and all six sampling fields. Derive frames with `floor(durationSeconds * 25)`, reject unknown fields, and remove derived product fields from the transport. Keep dedicated diagnostics fixed.

## Task 3: Generalize AR prompt and sampling inputs

**Files:**

- Modify `src/runtime/pipeline/sampler.ts` and tests.
- Modify `src/runtime/pipeline/rvq-generation.ts` and tests.
- Modify dependent plan/progress code only where dynamic token count is consumed.

**RED:** Dynamic token rows, separate semantic and residual top-k, arbitrary guidance, and temperature do not affect generation.

**GREEN:** Use actual prompt rows and length for prefill and cache bookkeeping. Apply requested guidance, top-k values, and temperature with float32 product semantics. Preserve diagnostic defaults explicitly at their call sites.

## Task 4: Generalize flow guidance and steps

**Files:**

- Modify `tools/converter/src/minimax_music3_webgpu/flow_transformer.py` and Python tests.
- Modify `src/runtime/pipeline/flow-generation.ts` and unit tests.
- Modify graph and manifest validation only as required by the new `guidance` input.

**RED:** The graph contains the literal 1.7 and the runtime rejects any schedule length other than 30.

**GREEN:** Add the scalar graph input, generate the official schedule for any positive step count, preserve the frozen 30-step bits, pass guidance to every flow call, and report dynamic totals.

## Task 5: Integrate the worker and reproducibility metadata

**Files:**

- Modify `src/workers/inference.worker.ts` and lifecycle tests.
- Modify `src/workers/music-progress.ts` and tests.
- Modify `src/runtime/reference/fixed-comparison.ts` and tests.
- Add a general effective-input result contract.

**RED:** Product generation still uses fixture rows and hardcoded sampler values, and fixed comparison metadata is selected only by seed and duration.

**GREEN:** Prepare prompt tokens from verified OPFS files, derive the plan after tokenization, forward every setting, echo effective inputs, and require an exact fixed-case match before comparison metadata is emitted. Preserve all session and device cleanup behavior.

## Task 6: Rebuild and validate the product release

**Files:**

- Use the existing `build-music-variable` pipeline and canonical artifacts directory.
- Do not modify or delete earlier generations.

**Preflight:** Verify source receipts, free disk, converter lock, target release graph, and absence of a target Chrome lock. Do not download source files again.

**Gate:** Build into unique staging, validate graph inputs and external ranges, promote atomically, and verify the new manifest and receipt. If promotion cannot proceed, preserve the old release and report the lock or validation failure.

## Task 7: Verify the non-UI path

Run one fresh verification pass after implementation:

- focused tokenizer, protocol, sampler, AR, flow, worker, and fixed-comparison tests;
- `uv run pytest -q tests/python`;
- `npm test`;
- `npm run lint`;
- `npm run typecheck`;
- `npm run build`;
- `git diff --check` and a scoped final diff review;
- one programmatic request with the frozen settings through the worker or headed browser harness, without adding UI controls.

Do not commit, stage, branch, create a worktree, delete artifacts, or claim a new browser quality result without the matching runtime gate.
