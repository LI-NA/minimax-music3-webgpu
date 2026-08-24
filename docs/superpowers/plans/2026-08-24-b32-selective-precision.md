# B32 Selective Precision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a MiniMax Music 3 WebGPU release using symmetric Q4 block size 32 with Flow `time_proj.weight` and `proj_out.weight` retained in FP16.

**Architecture:** Keep the existing converter and runtime layout, centralize the Python Q4 contract, make Flow precision exceptions explicit and validated, and isolate Global work by precision profile. Preserve B128 work with a hashed receipt and rely on the existing atomic release promotion for active release archives.

**Tech Stack:** Python 3.12, ONNX 1.22, ONNX Runtime 1.28, ONNX Runtime GenAI 0.15.2, TypeScript 5.9, ONNX Runtime Web WebGPU, Playwright.

**Spec:** `docs/superpowers/specs/2026-08-24-b32-selective-precision-design.md`

## Global Constraints

- Do not commit, stage, create branches, or create worktrees.
- Run converter builds only from the main checkout, one at a time.
- Keep every existing release archive and never remove archived generations automatically.
- Keep every referenced artifact at or below 134,217,728 bytes.
- Use Q4 block size 32, accuracy level 4, and symmetric quantization.
- Keep only `time_proj.weight` and `proj_out.weight` as Flow FP16 linear weights.

---

### Task 1: Define the B32 converter contract

**Files:**
- Modify: `tools/converter/src/minimax_music3_webgpu/constants.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/global_decoder.py`
- Test: `tests/python/test_global_decoder.py`

**Interfaces:**
- Produces: `Q4_BLOCK_SIZE`, `Q4_ACCURACY_LEVEL`, and precision-specific Global work paths.

- [ ] Change the builder test to require `block_size=32` and the validator fixture to emit block size 32.
- [ ] Run the targeted Global tests and confirm they fail because production still emits or requires 128.
- [ ] Add shared Q4 constants, use them in builder arguments and validation, and name work directories with `q4-b32`.
- [ ] Run the targeted Global tests and confirm they pass.

### Task 2: Export and validate selective Flow precision

**Files:**
- Modify: `tools/converter/src/minimax_music3_webgpu/flow_transformer.py`
- Test: `tests/python/test_flow_transformer.py`

**Interfaces:**
- Produces: `FLOW_FP16_LINEAR_WEIGHTS = ("time_proj.weight", "proj_out.weight")` and a graph containing 219 Q4 nodes for the full model.

- [ ] Change Flow tests to require B32 packed shapes, attributes, 9 Q4 nodes in one block, and FP16 `MatMul` nodes for both exception weights.
- [ ] Run the targeted Flow tests and confirm the expected B128 and Q4 `proj_out` failures.
- [ ] Parameterize Q4 packing with block size 32, export `proj_out.weight` through FP16 `MatMul`, and validate the exact Q4 and FP16 sets.
- [ ] Update maximum binding calculations for the FP16 exception.
- [ ] Run the targeted Flow tests and confirm they pass.

### Task 3: Publish and parse truthful precision metadata

**Files:**
- Modify: `tools/converter/src/minimax_music3_webgpu/manifest.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/acoustic_manifest.py`
- Modify: `src/runtime/model/manifest.ts`
- Test: `tests/python/test_acoustic_manifest.py`
- Test: `tests/python/test_global_decoder.py`
- Test: `tests/unit/model/manifest.test.ts`

**Interfaces:**
- Produces: Q4 block-size-32 manifest metadata and `precision.flowFp16Weights` containing the exact two weight names.

- [ ] Change Python and TypeScript tests to require the new manifest contract.
- [ ] Run the targeted tests and confirm they reject the current B128 metadata.
- [ ] Emit, validate, parse, and return the exact B32 and Flow FP16 metadata.
- [ ] Run the targeted Python and TypeScript tests and confirm they pass.

### Task 4: Prove the B32 WebGPU operator path

**Files:**
- Modify: `tools/generate-browser-fixtures.py`
- Modify: its existing Python or browser assertions when required by the new contract.

**Interfaces:**
- Produces: a deterministic B32 symmetric `MatMulNBits` browser fixture.

- [ ] Change the fixture contract test to require block size 32.
- [ ] Run the fixture test and confirm it fails against the current B128 generator.
- [ ] Generate the B32 fixture with hand-derived scales and expected output.
- [ ] Run its CPU oracle and headed Chrome WebGPU smoke with CPU fallback disabled.

### Task 5: Preserve B128 work and build Global B32

**Files:**
- Move: `artifacts/work/global-builder-1`
- Move: `artifacts/work/global-builder-36`
- Move: `artifacts/work/global-packed-1`
- Move: `artifacts/work/global-packed-36`
- Create: `artifacts/archive/b128-work/<generation>/receipt.json`

**Interfaces:**
- Consumes: the passing B32 converter tests and operator smoke.
- Produces: archived B128 work and new B32 `global-one-layer` and `global` releases.

- [ ] Hash every B128 work file, verify every move target remains under `artifacts/archive/b128-work/<generation>`, move the four directories, and verify the receipt.
- [ ] Run `build-global` for one layer and validate its graph and manifest.
- [ ] Run the one-layer headed Chrome Global gate.
- [ ] Run `build-global` for 36 layers and validate its graph and manifest.
- [ ] Run the full headed Chrome Global gate.

### Task 6: Build and validate the combined release

**Files:**
- Replace through existing atomic promotion: `artifacts/release/music-variable`
- Archive through existing promotion: `artifacts/archive/music-variable/<generation>`

**Interfaces:**
- Consumes: the new B32 Global release and pinned acoustic sources.
- Produces: a complete B32 plus selective FP16 `music-variable` release and receipt.

- [ ] Run the large-build preflight and confirm source hashes, disk, profile, lock, and process checks pass.
- [ ] Run `build-music-variable` and confirm the Flow fingerprint forces a rebuild.
- [ ] Validate 219 Flow Q4 nodes, both FP16 exception nodes, all external ranges, and all manifest hashes.
- [ ] Run the real Flow 30-step product gate.
- [ ] Run the combined headed Chrome 6-second and 10-second variable-duration gate.

### Task 7: Final verification and handoff

**Files:**
- Modify measured sizes, hashes, and precision wording in relevant development documentation.

**Interfaces:**
- Produces: fresh verification evidence and an uncommitted working tree ready for user listening tests.

- [ ] Run all Python tests using a workspace-local pytest base directory.
- [ ] Run `npm run test`, `npm run lint`, `npm run typecheck`, and `npm run build`.
- [ ] Validate the active release receipt, manifest file hashes, artifact count, byte total, file-size ceiling, and source revisions.
- [ ] Review the complete Git diff and confirm no unrelated files or commits were created.
- [ ] Report the new release path, exact contract, test evidence, archive locations, and any skipped or failed gate.
