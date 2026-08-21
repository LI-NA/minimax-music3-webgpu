# Variable-duration generation and reference parity implementation plan

> The 5-second product increment in this historical plan is superseded by the approved programmatic input contract dated 2026-08-22. Product duration now uses `floor(durationSeconds * 25)` for any finite positive value through 300 seconds.

> Execute this plan in the current checkout without Git mutations. The project instructions override workflow defaults that would create a worktree or commit. Use RED, GREEN, and focused review for every behavior change.

**Spec:** `docs/superpowers/specs/2026-08-21-variable-duration-and-reference-parity-design.md`

**Goal:** Extend the verified five-second WebGPU pipeline through the official long-song algorithm, qualify lengths up to five minutes where the 12 GiB incremental project budget permits, and create a reproducible WebGPU receipt plus cloud-service comparison workflow.

## Global constraints

- Preserve the current five-second release and every existing artifact generation.
- Do not delete, overwrite, or relocate artifact directories.
- Keep CPU execution-provider fallback disabled in browser gates.
- Keep every initializer and expected storage-buffer binding at or below 134,217,728 bytes.
- Do not duplicate full Global, RVQ, or flow weights per duration.
- Do not add custom WGSL, a second runtime, or unmeasured performance profiles.
- Run converter downloads and builds only from the main checkout, one at a time.
- Do not begin a high-VRAM browser gate while another process materially occupies GPU memory.
- Do not claim a duration or hardware tier without its specified measured gate.
- Do not perform Git mutations unless the user explicitly requests them in a later turn.

## Task 1: Persist artifact and test-discovery policy

**Files:**

- Modify: `AGENTS.override.md`
- Modify: `pyproject.toml`
- Create: `docs/development/artifact-worktree-policy.md`

**RED:** Add a scoped pytest collection check proving the configured suite does not collect nested ignored source checkouts under `artifacts/`.

**GREEN:**

- Add the approved canonical-artifacts and worktree-junction instructions to `AGENTS.override.md` and the development document.
- Configure pytest to collect the project suite without descending into `artifacts/`.
- Preserve `.gitignore` and `.kopiaignore`, which already ignore the correct root.

**Verify:**

- `uv run pytest --collect-only -q`
- `uv run pytest -q tests/python`
- Review the final diff for destructive artifact instructions.

## Task 2: Correct sampler parity and define trace schemas

**Files:**

- Modify: `src/runtime/pipeline/sampler.ts`
- Modify: `tests/unit/pipeline/sampler.test.ts`
- Create: `src/runtime/reference/trace.ts`
- Create: `tests/unit/reference/trace.test.ts`
- Create: `tests/fixtures/reference/first-transition.schema.json`

**RED:**

- Prove a 60-way kth-threshold tie retains all 60 candidates when top-k is 50.
- Prove guidance, masking, exponentiation, and cumulative probability are rounded through float32 product semantics.
- Prove a reference trace rejects missing provenance, wrong shapes, non-finite numeric checkpoints, and inconsistent decision counts.

**GREEN:**

- Match pinned Diffusers threshold top-k behavior.
- Keep deterministic injected draws for product tests, but do not claim PyTorch RNG equality.
- Define a versioned trace metadata contract for fixed prompt, lyrics, revisions, parameters, code decisions, noise, checkpoints, and receipts.

**Verify:** Focused Vitest, then full Vitest, lint, and typecheck.

## Task 3: Implement exact duration and chunk planning

**Files:**

- Create: `src/runtime/pipeline/duration-plan.ts`
- Create: `tests/unit/pipeline/duration-plan.test.ts`
- Create: `tools/converter/src/minimax_music3_webgpu/duration_plan.py`
- Create: `tests/python/test_duration_plan.py`

**RED:** Hand-derived tables for 5, 6, 10, 30, 60, and 300 seconds must fail before implementation. Include chunk starts, frame lengths, latent lengths, crops, output samples, flow calls, vocoder calls, and WAV bytes.

**GREEN:** Implement the pinned formulas. Product requests accept 5-second increments from 5 through 300. Internal six-second diagnostics are allowed. Validate the 10,240-token context limit.

**Cross-language gate:** Serialize both planners for a literal case table and require exact equality.

## Task 4: Prove maximum-window graph equivalence

**Files:**

- Modify: `tools/converter/src/minimax_music3_webgpu/condition_encoder.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/flow_transformer.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/vocoder.py`
- Modify: `tests/python/test_condition_encoder.py`
- Modify: `tests/python/test_flow_transformer.py`
- Modify: `tests/python/test_vocoder.py`

**RED:**

- A maximum condition graph must match fixed graphs for 125, 150, 175, and 200 active frames.
- A one-block maximum flow graph must match fixed active outputs at latent lengths 430, 516, 602, and 689.
- A symbolic-length mono vocoder must match each channel of the current fixed-length stereo wrapper at latent lengths 430, 516, 602, and 689.

**GREEN:** Add only the maximum-shape condition and flow inputs and masking required by the spec. Publish a symbolic temporal mono vocoder because static zero-padding changes the non-causal convolutional model math.

**Stop condition:** If maximum flow equivalence fails after one root-cause diagnosis and one minimal correction, stop that route and implement four fixed tail graphs sharing external weight shards. Do not duplicate q4 weights.

**Verify:** Desktop ORT numeric oracles, ONNX checker, operator allow-list, external ranges, and binding-size calculation.

## Task 5: Build and publish the variable acoustic release

**Files:**

- Modify: `tools/converter/src/minimax_music3_webgpu/acoustic_manifest.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/manifest.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/cli.py`
- Modify: corresponding Python tests

**RED:** A manifest parser and release test require the official window, hop, overlap, crops, sample hop, max frames, max latent length, active-mask inputs, and symbolic-length mono vocoder contract.

**GREEN:**

- Build the new acoustic graphs into a unique staging generation.
- Reuse unchanged model generations and external weights by hash.
- Publish the graph set, manifest, and receipt atomically.
- Keep the old `music-5s` release intact.

**Large-run gate:** Before conversion, verify free disk, source receipts, no converter process, and no Chrome process holding the target. Record sizes and hashes after publication.

## Task 6: Generalize runtime protocol, progress, and result contracts

**Files:**

- Modify: `src/workers/protocol.ts`
- Modify: `src/workers/music-progress.ts`
- Modify: `src/app/App.tsx`
- Modify: corresponding unit tests

**RED:**

- Variable duration requests reject unsupported steps and context overflow.
- Progress totals reflect requested frames, chunks, all flow steps, and all mono vocoder calls.
- ETA appears only after stable samples.
- Detailed completion text remains visible.
- Cancellation clears incomplete state.

**GREEN:** Replace fixed five-second request and tuple fields with duration-derived contracts. Keep the UI diagnostic and minimal.

## Task 7: Generalize AR retention and natural early end

**Files:**

- Modify: `src/runtime/pipeline/rvq-generation.ts`
- Modify: `src/runtime/pipeline/music-generation.ts`
- Modify: corresponding unit tests

**RED:**

- A natural audio-end after retained frames returns those frames instead of throwing and retrying.
- A no-end 300-second synthetic run writes exactly 491,520,000 hidden bytes into one flat buffer without a second full copy.
- Decision, depth, feedback, and cache counts match the plan.

**GREEN:** Write retained groups directly into a preallocated flat FP16 buffer, expose actual retained frame count, and preserve current five-second semantics.

## Task 8: Implement official multi-chunk flow orchestration

**Files:**

- Modify: `src/runtime/pipeline/flow-generation.ts`
- Create or modify: focused flow chunk runtime helpers
- Modify: corresponding unit tests

**RED:** Compare TypeScript overlap replacement, noise prompt, carry selection, final restoration, and crop metadata with literal pinned-source fixtures for the ten-second two-chunk case.

**GREEN:** Reuse one condition and flow session, keep all 30 intermediate steps on GPU, download one final latent per chunk, restore the exact overlap, and carry the specified latent and condition intervals.

## Task 9: Implement sequential mono vocoder and direct WAV assembly

**Files:**

- Modify: `src/runtime/pipeline/vocoder-generation.ts`
- Modify: `src/runtime/audio/wav.ts`
- Modify: corresponding unit tests

**RED:**

- Sequential mono output matches deterministic stereo wrapper output.
- Ten-second crop and stitch produces 440,832 samples per channel and a 1,763,372-byte WAV.
- Five-minute assembly allocates one final PCM16 WAV buffer and returns 52,989,996 bytes.

**GREEN:** Run one mono session twice per chunk, crop before writing, and write PCM16 directly to final channel-interleaved positions.

## Task 10: Wire variable generation in the worker

**Files:**

- Modify: `src/workers/inference.worker.ts`
- Modify: `src/runtime/model/manifest.ts`
- Modify: `tools/serve-artifacts.mjs`
- Modify: integration unit tests

**RED:** A synthetic two-chunk lifecycle requires AR release before acoustic device creation, session reuse within acoustic stages, dynamic progress, final WAV transfer, and cleanup on failure.

**GREEN:** Wire the new release and duration contract without changing CPU fallback, JSPI patching, or artifact receipt semantics.

## Task 11: Gate six and ten seconds in Chrome

**Files:**

- Create: `tests/browser/variable-duration.spec.ts`
- Modify: browser launch scripts only as required
- Record ignored metrics under `artifacts/diagnostics/variable-duration/`

**Six-second gate:** Verify exact counts and bytes, finite and varying stereo audio, no constant tail, `AudioContext` decode, zero warm fetches, and at most 12,288 MiB project growth above the spawn-adjacent GPU baseline.

**Ten-second gate:** Verify exact two-chunk plan, 60 flow steps, four vocoder calls, 440,832 samples per channel, 1,763,372 WAV bytes, progress boundaries, cancellation behavior, and the same health and memory conditions.

Run branded headed Chrome with one worker and a persistent namespaced profile. Start only after GPU preflight is clean.

## Task 12: Create cloud comparison case tooling

**Files:**

- Create: `tools/reference/reference_case.py`
- Create: focused pure-Python tests
- Create: `docs/development/reference-comparison.md`

**RED:** A WebGPU case receipt requires every user-facing input, seed and sampler parameter, model/runtime provenance, exact WAV metadata, atomic publication, and hash verification.

**GREEN:** Create a pure-Python offline tool that combines a WebGPU manifest, generation metrics, and WAV into `artifacts/reference/<case-id>/`. It must not import Torch or Diffusers, download files, load model weights, or run the original model. Document the exact fields to enter into a cloud service and how to attach the returned cloud WAV.

**Limitation:** Cloud services may not expose seed or internal sampler controls and may use a different model revision. Record unavailable controls explicitly and treat the result as structural and manual audio comparison rather than tensor parity.

## Task 13: Record the WebGPU comparison case and cloud handoff

**Files:**

- Add: comparison metadata to the variable-duration browser result
- Add: a focused browser spec for deterministic case reproduction
- Create: the cloud input sheet and receipt
- Update: `docs/development/reference-comparison.md`

**RED:** A case with a changed prompt, lyrics, seed, duration, sampler parameter, manifest hash, or WAV hash must fail receipt verification.

**GREEN:** Generate the fixed WebGPU case, verify deterministic repetition within WebGPU, persist the receipt, and present the exact cloud inputs to the user. Keep this as diagnostic tooling, not product UI.

**Gate:** After the user returns a cloud WAV, attach it to the case and report structure plus a manual listening comparison. Do not claim isolated quantization or provider tolerances.

## Task 14: Run staged long-duration qualification

**Files:**

- Extend: `tests/browser/variable-duration.spec.ts`
- Create: ignored telemetry runner and evidence below `artifacts/diagnostics/variable-duration/`
- Update: runtime requirement documentation

Run 30, 60, 120, and 300-second headed persistent-profile gates in order. Reuse completed artifacts. At every run record stage timings, progress events, requested and effective telemetry cadence, baseline, peak, final GPU use, output structure, and failure boundary.

Stop later gates if an earlier gate loses the device, exceeds 12,288 MiB above its spawn-adjacent baseline, approaches within 512 MiB of physical adapter capacity, or exposes a correctness defect. Do not narrow the goal silently. Diagnose the first blocker and document the highest verified duration.

If 300 seconds exceeds the memory ceiling, run only the specified batch-one sequential CFG experiment with RED/GREEN graph and runtime contracts. Retain it only when measured evidence qualifies it.

**Measured outcome:** The archived pre-programmatic release reached the requested maximum through 60 seconds. Its 120-second seed-7 request ended naturally at frame 1,743. A separate capacity diagnostic processed all 7,500 frames after suppressing 83 audio-end decisions, reached 13,892 MiB total from a 3,168 MiB baseline, and stayed within the 12,288 MiB incremental budget. This qualifies that archived full-pipeline workload, not native five-minute music quality. The active programmatic-input release has separate 6 and 10-second evidence.

## Task 15: Final documentation and completion audit

**Files:**

- Update: `README.md`
- Update: `docs/development/webgpu-runtime-requirements.md`
- Update: `docs/development/five-second-generation.md` or replace with a variable-duration guide
- Update: reference comparison documentation

Document:

- verified duration ceiling;
- measured hardware tiers and uncertainty;
- retained and rejected optimizations;
- warm and cold behavior;
- progress and cancellation semantics;
- artifact policy;
- WebGPU-versus-cloud comparison results and limitations; and
- exact reproduction commands.

Run fresh Python, Vitest, lint, typecheck, production build, artifact manifest validation, all relevant headed Chrome gates, diff check, and a requirement-by-requirement audit against the spec. Keep the goal active if any required evidence is missing.
