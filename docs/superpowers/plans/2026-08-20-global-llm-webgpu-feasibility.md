# Reduced Global LLM WebGPU Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the exact MiniMax-Music3 Global LLM into a reduced q4 browser artifact, load it in Windows Chrome through ONNX Runtime WebGPU, and complete prompt prefill plus ten GPU-resident cached decode steps.

**Architecture:** A `uv`-managed Python tool pins and selectively downloads only the official Global LLM source files, converts the prompt embedding into row-aligned FP16 files, builds a q4 decoder without its full embedding or LM head, exports an exact reduced FP16 head, and emits a hashed release manifest. A dedicated browser Worker owns OPFS, an explicit WebGPU device, the ONNX Runtime JSPI build, and the GPU KV tensor chain. React provides only a diagnostic surface for this feasibility milestone.

**Tech Stack:** Python 3.11+, uv, PyTorch, Safetensors, ONNX, ONNX Runtime GenAI model builder, ONNX Runtime desktop, React 19, TypeScript, Vite, Vitest, Playwright, ONNX Runtime Web JSPI, WebGPU, OPFS

**Spec:** `docs/superpowers/specs/2026-08-20-minimax-music3-webgpu-design.md`

## Global Constraints

- Use the exact `MiniMaxAI/MiniMax-Music3` checkpoint at revision `fbdf52fbaaca799592917417eb05f1899f1255ec`. Do not substitute or distill the model.
- Follow Diffusers commit `3681e65996b4d2589219720101a6acbfd25073f8` for the inference contract.
- The product runtime is browser-only. Python, PyTorch, CUDA, and ONNX tooling are development dependencies.
- Phase 1 targets Windows desktop Chromium and a practical peak GPU allocation at or below 12 GB on the RTX 4080 development machine.
- Keep all source weights, conversion intermediates, generated fixtures, and release artifacts under `artifacts/`. That directory is already present in both `.gitignore` and `.kopiaignore` through commit `506955f`.
- Do not download `qwen_7B/**`, `dav.pth`, `flowmatching_vae.pth`, or acoustic-stage weights during this plan.
- Use q4 symmetric `MatMulNBits` with block size 128 and FP16 activations for the Global decoder. Keep the exact reduced head in FP16.
- Keep every physical artifact file at or below 128 MiB. An ONNX initializer may not cross an external-data file boundary.
- Store the 200,000 by 4,096 prompt embedding as FP16 row shards. Gather only requested rows from OPFS inside the inference Worker.
- Pin `onnxruntime-web` to `1.30.0-dev.20260813-72e1c9c9b8` and import `onnxruntime-web/jspi` until a stable release contains ONNX Runtime commit `010a8f0`.
- Do not enable ORT proxy mode, CPU execution-provider fallback, graph capture, custom WGSL, or a split decoder before a measured failure requires one.
- The feasibility gate requires one real reduced decoder session, a 40-token batch-two prefill, ten cached decode steps, finite reduced logits, and all KV tensors remaining at `gpu-buffer` location.
- Until the RVQ stage is implemented, each cached diagnostic step feeds the exact semantic embedding row for token 151,675 to both batch lanes. This validates the decoder and cache path without claiming a complete generation loop.
- Do not perform automated audio analysis. This plan stops before the acoustic stage.
- Follow Conventional Commits as a practical guideline. Do not use emojis or em dashes in project files.

## File Structure

### Web application

- `package.json`: pinned browser dependencies and project commands.
- `package-lock.json`: exact npm dependency lock.
- `index.html`: Vite entry document.
- `tsconfig.json`: TypeScript project references.
- `tsconfig.app.json`: browser and Worker compiler settings.
- `tsconfig.node.json`: Vite and Playwright compiler settings.
- `vite.config.ts`: application build and test configuration.
- `playwright.config.ts`: branded Chrome WebGPU test configuration.
- `src/main.tsx`: React bootstrap.
- `src/app/App.tsx`: feasibility diagnostic screen.
- `src/app/styles.css`: minimal readable diagnostic layout.
- `src/runtime/model/manifest.ts`: release-manifest types and validation.
- `src/runtime/model/artifact-cache.ts`: resumable verified OPFS artifact storage.
- `src/runtime/model/embedding-table.ts`: row lookup from FP16 OPFS shards.
- `src/runtime/model/webgpu-device.ts`: adapter limit checks and device creation.
- `src/runtime/model/ort-session.ts`: JSPI ORT initialization and session creation.
- `src/runtime/pipeline/gpu-kv-state.ts`: ownership and disposal of GPU KV tensors.
- `src/workers/protocol.ts`: typed diagnostic request, progress, result, and error messages.
- `src/workers/inference.worker.ts`: exclusive owner of OPFS, WebGPU, ORT, and decoder state.

### Conversion tooling

- `pyproject.toml`: converter package, dependency pins, commands, and pytest settings.
- `uv.lock`: exact Python dependency lock.
- `tools/converter/src/minimax_music3_webgpu/__init__.py`: converter package marker.
- `tools/converter/src/minimax_music3_webgpu/constants.py`: official revisions, token IDs, dimensions, and artifact limits.
- `tools/converter/src/minimax_music3_webgpu/paths.py`: validated artifact-directory layout.
- `tools/converter/src/minimax_music3_webgpu/source.py`: selective Hugging Face download plan and receipt.
- `tools/converter/src/minimax_music3_webgpu/embedding.py`: FP16 row-shard exporter.
- `tools/converter/src/minimax_music3_webgpu/external_data.py`: streaming ONNX external-data repacker.
- `tools/converter/src/minimax_music3_webgpu/reduced_head.py`: exact semantic and end-token head exporter.
- `tools/converter/src/minimax_music3_webgpu/global_decoder.py`: ORT GenAI q4 builder wrapper and graph checks.
- `tools/converter/src/minimax_music3_webgpu/manifest.py`: SHA-256 release-manifest emitter.
- `tools/converter/src/minimax_music3_webgpu/cli.py`: `music3-convert` command entry point.

### Tests and developer tools

- `tests/python/test_paths.py`: artifact-root containment tests.
- `tests/python/test_source.py`: exact selective-download tests.
- `tests/python/test_embedding.py`: row-shard round-trip tests.
- `tests/python/test_external_data.py`: external-data size and byte-preservation tests.
- `tests/python/test_reduced_head.py`: exact row-selection tests.
- `tests/python/test_global_decoder.py`: builder command and graph-invariant tests.
- `tests/python/test_converter_smoke.py`: tiny Qwen3 end-to-end conversion smoke.
- `tests/fixtures/prompt-contract.json`: exact 40-token conditional and unconditional prefill fixture.
- `tests/unit/model/manifest.test.ts`: browser manifest validation tests.
- `tests/unit/model/artifact-cache.test.ts`: resume, hash, and retry tests.
- `tests/unit/model/embedding-table.test.ts`: cross-shard OPFS row lookup tests.
- `tests/unit/model/webgpu-device.test.ts`: adapter capability tests.
- `tests/unit/pipeline/gpu-kv-state.test.ts`: GPU tensor ownership tests.
- `tests/browser/external-data-opfs.spec.ts`: JSPI `File` range-loading smoke.
- `tests/browser/matmul-nbits.spec.ts`: q4 WebGPU known-answer smoke.
- `tests/browser/global-decoder.spec.ts`: real checkpoint feasibility test.
- `tools/serve-artifacts.mjs`: development-only static artifact server with byte ranges.
- `docs/development/conversion.md`: reproducible download and conversion commands.
- `docs/development/global-llm-feasibility.md`: measured gate result and diagnostics.

---

### Task 1: Create the tested browser and converter foundation

**Files:**
- Create: `package.json`
- Create: `package-lock.json`
- Create: `index.html`
- Create: `tsconfig.json`
- Create: `tsconfig.app.json`
- Create: `tsconfig.node.json`
- Create: `vite.config.ts`
- Create: `playwright.config.ts`
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `tools/converter/src/minimax_music3_webgpu/__init__.py`
- Create: `tools/converter/src/minimax_music3_webgpu/cli.py`
- Create: `src/main.tsx`
- Create: `src/app/App.tsx`
- Create: `src/app/styles.css`
- Create: `src/runtime/model/webgpu-device.ts`
- Create: `tests/unit/model/webgpu-device.test.ts`
- Modify: `.gitignore`
- Modify: `.kopiaignore`

**Interfaces:**
- Produces: `inspectWebGpu(gpu: GPU | undefined): Promise<WebGpuCapability>`.
- Produces: npm commands `dev`, `build`, `lint`, `typecheck`, `test`, and `test:browser`.
- Produces: the `music3-convert` Python entry point used by every later converter task.

- [ ] **Step 1: Create the pinned project configuration and ignore rules**

Use React `19.2.8`, Vite `8.2.2`, Vitest `4.1.11`, Playwright `1.62.1`, `@webgpu/types`, `@noble/hashes` `2.3.0`, ESLint, TypeScript, and `onnxruntime-web` `1.30.0-dev.20260813-72e1c9c9b8`. Add `node_modules/`, `.venv/`, `dist/`, `coverage/`, `playwright-report/`, and `test-results/` to `.gitignore`. Add the reproducible large dependency directories `node_modules/` and `.venv/` to `.kopiaignore` before installing dependencies.

Set Python to `>=3.11,<3.14` and add `huggingface-hub==1.27.0`, `safetensors==0.8.0`, `torch==2.13.0`, `transformers==5.15.0`, `onnx==1.22.0`, `onnxscript==0.7.1`, `onnxruntime==1.28.0`, `onnxruntime-genai==0.15.2`, `numpy`, and `pytest`. Register `music3-convert = "minimax_music3_webgpu.cli:main"`, configure the package root as `tools/converter/src`, and create an `argparse` entry point whose `--help` exits successfully. These versions are the newest required-compatible releases admitted by the repository owner's global uv rule that excludes packages published within the last seven days.

- [ ] **Step 2: Install and lock dependencies**

Run: `npm install`

Run: `uv sync`

Expected: `package-lock.json` and `uv.lock` are created without dependency-resolution errors.

- [ ] **Step 3: Write the WebGPU capability test**

```ts
import { describe, expect, it, vi } from 'vitest';
import { inspectWebGpu } from '../../../src/runtime/model/webgpu-device';

describe('inspectWebGpu', () => {
  it('rejects an adapter without shader-f16', async () => {
    const adapter = {
      features: new Set<string>(),
      info: { isFallbackAdapter: false },
      limits: { maxStorageBufferBindingSize: 1_000_000_000 },
    };
    const gpu = { requestAdapter: vi.fn().mockResolvedValue(adapter) } as unknown as GPU;

    await expect(inspectWebGpu(gpu)).resolves.toEqual({
      supported: false,
      reason: 'shader-f16 is unavailable',
    });
  });
});
```

- [ ] **Step 4: Run the test and verify the missing module failure**

Run: `npm test -- tests/unit/model/webgpu-device.test.ts`

Expected: FAIL because `src/runtime/model/webgpu-device.ts` does not exist.

- [ ] **Step 5: Implement the minimal React application and WebGPU capability contract**

The diagnostic page displays capability status and no generation controls. Implement the capability contract as:

```ts
export type WebGpuCapability =
  | { supported: true; adapter: GPUAdapter }
  | { supported: false; reason: string };

export async function inspectWebGpu(gpu: GPU | undefined): Promise<WebGpuCapability> {
  if (!gpu) return { supported: false, reason: 'WebGPU is unavailable' };
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { supported: false, reason: 'No WebGPU adapter was found' };
  if (adapter.info.isFallbackAdapter) return { supported: false, reason: 'A fallback adapter is not supported' };
  if (!adapter.features.has('shader-f16')) return { supported: false, reason: 'shader-f16 is unavailable' };
  return { supported: true, adapter };
}
```

- [ ] **Step 6: Run the foundation checks**

Run: `npm test -- tests/unit/model/webgpu-device.test.ts`

Expected: PASS.

Run: `npm run typecheck && npm run lint && npm run build`

Expected: all commands exit 0 and Vite writes `dist/`.

Run: `uv run music3-convert --help`

Expected: exit 0 and display command help.

- [ ] **Step 7: Commit the foundation**

```powershell
git add package.json package-lock.json index.html tsconfig.json tsconfig.app.json tsconfig.node.json vite.config.ts playwright.config.ts pyproject.toml uv.lock tools/converter/src src tests/unit/model/webgpu-device.test.ts .gitignore .kopiaignore
git commit -m "feat: scaffold WebGPU feasibility app"
```

### Task 2: Add the pinned selective source downloader

**Files:**
- Create: `tools/converter/src/minimax_music3_webgpu/constants.py`
- Create: `tools/converter/src/minimax_music3_webgpu/paths.py`
- Create: `tools/converter/src/minimax_music3_webgpu/source.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/cli.py`
- Create: `tests/python/test_paths.py`
- Create: `tests/python/test_source.py`

**Interfaces:**
- Produces: `ArtifactPaths.from_root(root: Path, repository_root: Path = Path.cwd()) -> ArtifactPaths` with `source`, `work`, `release`, and `receipts` children.
- Produces: `global_source_patterns() -> tuple[str, ...]`.
- Produces: `download_global_source(paths: ArtifactPaths) -> SourceReceipt`.
- Produces: `music3-convert download-global --artifacts-dir artifacts`.

- [ ] **Step 1: Write tests for containment and the exact download set**

```python
from minimax_music3_webgpu.source import global_source_patterns


def test_global_source_patterns_exclude_legacy_and_acoustic_weights() -> None:
    patterns = global_source_patterns()
    assert patterns == (
        "LICENSE",
        "modular_model_index.json",
        "language_model/*",
        "tokenizer/*",
    )
    assert all("qwen_7B" not in pattern for pattern in patterns)
    assert all("transformer" not in pattern for pattern in patterns)
```

`test_paths.py` must also prove that an artifact root resolving outside the repository is rejected and that every generated path remains below the chosen `artifacts/` root.

- [ ] **Step 2: Run the downloader tests and verify they fail**

Run: `uv run pytest tests/python/test_paths.py tests/python/test_source.py -q`

Expected: FAIL because the converter modules do not exist.

- [ ] **Step 3: Implement constants, paths, and the selective download call**

Use these constants verbatim:

```python
MODEL_ID = "MiniMaxAI/MiniMax-Music3"
MODEL_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
DIFFUSERS_REVISION = "3681e65996b4d2589219720101a6acbfd25073f8"
ARTIFACT_FILE_LIMIT = 128 * 1024 * 1024
HIDDEN_SIZE = 4096
VOCAB_SIZE = 200_000
AUDIO_END_TOKEN_ID = 151_670
SEMANTIC_TOKEN_START = 151_675
SEMANTIC_TOKEN_COUNT = 16_384
```

Call `huggingface_hub.snapshot_download` with the pinned revision, exact allow patterns, `local_dir=paths.source`, and `cache_dir=paths.root / "hf-cache"`. Write `receipts/source-global.json` atomically with repository ID, revision, selected relative paths, sizes, and SHA-256 values.

- [ ] **Step 4: Expose the command and pass the unit tests**

Run: `uv run pytest tests/python/test_paths.py tests/python/test_source.py -q`

Expected: PASS without downloading network data because the Hub call is replaced by a test double.

- [ ] **Step 5: Verify the command help and commit**

Run: `uv run music3-convert --help`

Expected: exit 0 and list `download-global`.

```powershell
git add tools/converter/src tests/python
git commit -m "feat(converter): add selective model downloader"
```

### Task 3: Add exact embedding, reduced-head, and external-data packing tools

**Files:**
- Create: `tools/converter/src/minimax_music3_webgpu/embedding.py`
- Create: `tools/converter/src/minimax_music3_webgpu/external_data.py`
- Create: `tools/converter/src/minimax_music3_webgpu/reduced_head.py`
- Create: `tests/python/test_embedding.py`
- Create: `tests/python/test_external_data.py`
- Create: `tests/python/test_reduced_head.py`

**Interfaces:**
- Produces: `shard_fp16_rows(rows: np.ndarray, output_dir: Path, max_file_bytes: int) -> EmbeddingTableReceipt`.
- Produces: `export_embedding_table(source_shard: Path, output_dir: Path) -> EmbeddingTableReceipt`.
- Produces: `repack_external_data(model_path: Path, output_dir: Path, max_file_bytes: int) -> RepackedModel`.
- Produces: `export_reduced_head(source_shard: Path, output_dir: Path) -> RepackedModel`.
- Consumes: constants and artifact paths from Task 2.

- [ ] **Step 1: Write the row-shard round-trip test**

```python
import numpy as np

from minimax_music3_webgpu.embedding import shard_fp16_rows


def test_shard_fp16_rows_preserves_order_and_boundary(tmp_path) -> None:
    rows = np.arange(17 * 8, dtype=np.float16).reshape(17, 8)
    receipt = shard_fp16_rows(rows, tmp_path, max_file_bytes=8 * 8 * 2)
    assert [item.row_count for item in receipt.shards] == [8, 8, 1]
    restored = np.concatenate(
        [np.fromfile(item.path, dtype=np.float16).reshape(item.row_count, 8) for item in receipt.shards]
    )
    np.testing.assert_array_equal(restored, rows)
```

Add external-data tests that use three synthetic ONNX initializers, verify no tensor crosses a file boundary, verify every copied byte, and reject one initializer larger than the limit. Add reduced-head tests asserting semantic rows `151675:168059` and end row `151670` are selected in their original order.

- [ ] **Step 2: Run all three tests and verify missing implementations**

Run: `uv run pytest tests/python/test_embedding.py tests/python/test_external_data.py tests/python/test_reduced_head.py -q`

Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement row-aligned FP16 embedding export**

Read `model.embed_tokens.weight` through `safetensors.safe_open(..., framework="pt", device="cpu")`. Convert at most 16,384 rows at a time from BF16 to FP16, write twelve 128 MiB files and one final partial file, and record `rowStart`, `rowCount`, `columns=4096`, `rowBytes=8192`, file size, and SHA-256 for each shard.

- [ ] **Step 4: Implement streaming external-data repacking**

Load ONNX protobuf with `load_external_data=False`. For each initializer, read its existing `location`, `offset`, and `length`, select a target shard that can hold the entire tensor, stream-copy its bytes, and replace the initializer metadata with the new relative `location`, `offset`, and `length`. Fail before writing the final graph if an initializer exceeds 128 MiB. Save atomically and reload metadata to validate every range.

- [ ] **Step 5: Implement the exact reduced head**

Export one ONNX graph accepting `hidden_states` FP16 with shape `[batch, sequence, 4096]`. Gather the final sequence position, then return `semantic_logits` with shape `[batch, 16384]` from the exact semantic rows and `end_logit` with shape `[batch, 1]` from token row 151,670. Store the 128 MiB semantic matrix and 8 KiB end matrix as separate initializers so neither exceeds the artifact limit.

- [ ] **Step 6: Run the converter tests and commit**

Run: `uv run pytest tests/python/test_embedding.py tests/python/test_external_data.py tests/python/test_reduced_head.py -q`

Expected: PASS.

```powershell
git add tools/converter/src/minimax_music3_webgpu tests/python
git commit -m "feat(converter): add sharded model packing"
```

### Task 4: Build and validate the reduced q4 Global decoder

**Files:**
- Create: `tools/converter/src/minimax_music3_webgpu/global_decoder.py`
- Create: `tools/converter/src/minimax_music3_webgpu/manifest.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/cli.py`
- Create: `tests/python/test_global_decoder.py`
- Create: `tests/python/test_converter_smoke.py`
- Create: `tests/fixtures/prompt-contract.json`

**Interfaces:**
- Produces: `build_global_decoder(paths: ArtifactPaths, num_hidden_layers: int = 36) -> GlobalDecoderReceipt`.
- Produces: `validate_global_decoder(model_path: Path) -> GraphReport`.
- Produces: `emit_global_release(paths: ArtifactPaths) -> Path`.
- Produces: `music3-convert build-global --artifacts-dir artifacts --layers {1,36}`.
- Consumes: the pinned source layout and packers from Tasks 2 and 3.

- [ ] **Step 1: Write graph-invariant tests**

```python
from minimax_music3_webgpu.global_decoder import builder_arguments


def test_builder_arguments_target_webgpu_q4(tmp_path) -> None:
    args = builder_arguments(tmp_path / "language_model", tmp_path / "output", tmp_path / "cache")
    joined = " ".join(args)
    assert "-p int4" in joined
    assert "-e webgpu" in joined
    assert "exclude_embeds=true" in args
    assert "exclude_lm_head=true" in args
    assert "include_hidden_states=true" not in args
    assert "block_size=128" in args
    assert "accuracy_level=4" in args
    assert "is_symmetric=true" in args
    assert "fuse_qk_norm_gqa=true" in args
```

Graph validation tests parameterize the expected layer count. A 36-layer build requires 36 `GroupQueryAttention` nodes, 72 past inputs, and 72 present outputs, while a one-layer diagnostic build requires one attention node and two cache inputs and outputs. Every build requires an `inputs_embeds` input, q4 `MatMulNBits` nodes with block size 128, a hidden-state output, no token embedding initializer, no full LM-head initializer, no CPU-only fallback partition in a WebGPU optimization report, and no external initializer over 128 MiB.

- [ ] **Step 2: Run the decoder tests and verify they fail**

Run: `uv run pytest tests/python/test_global_decoder.py -q`

Expected: FAIL because `global_decoder.py` does not exist.

- [ ] **Step 3: Implement the ORT GenAI builder wrapper**

Invoke the builder through the active Python interpreter:

```text
python -m onnxruntime_genai.models.builder
  -i <source>/language_model
  -o <work>/global-builder
  -p int4
  -e webgpu
  -c <work>/ortgenai-cache
  --extra_options
  exclude_embeds=true
  exclude_lm_head=true
  filename=global_decoder.onnx
  block_size=128
  accuracy_level=4
  is_symmetric=true
  op_types_to_quantize=MatMul
  fuse_qk_norm_gqa=true
```

In ORT GenAI 0.15.2, `exclude_lm_head=true` already exposes hidden states instead of logits and is mutually exclusive with `include_hidden_states=true`.

Capture stdout, stderr, package versions, arguments, elapsed time, and exit code in the conversion receipt. When `num_hidden_layers` is 1, add `num_hidden_layers=1` to the extra options and publish a separate `global-one-layer` diagnostic manifest. Repack external data through Task 3 and run every graph invariant before publishing a release manifest. Resolve `model.embed_tokens.weight` and `lm_head.weight` through the pinned safetensors index, invoke the Task 3 embedding and reduced-head exporters, and place the decoder, reduced head, embedding shards, tokenizer files, and license below the selected `global-one-layer` or `global` release root. `build-global` is the orchestration command; it may not depend on a separate unpublished packing command.

- [ ] **Step 4: Implement a tiny Qwen3 conversion smoke**

Create a deterministic local `Qwen3ForCausalLM` fixture with vocabulary 128, hidden size 64, two layers, four attention heads, two KV heads, head size 16, and intermediate size 128. Save it with `save_pretrained`. Keep `fuse_qk_norm_gqa=true` as the production default and validate that fused WebGPU graph structurally. For the desktop CPU execution smoke only, build the same q4 graph with `fuse_qk_norm_gqa=false`, because ORT 1.28 CPU does not support the fused GQA Q/K normalization-weight inputs. Confirm the two variants retain identical inputs, outputs, q4 node count, attention count, and cache contract. Create an ONNX Runtime desktop session for the unfused diagnostic graph, perform prefill and two cached steps, and assert hidden and cache shapes are finite and grow by one token per step.

- [ ] **Step 5: Emit the versioned global release manifest**

The JSON manifest must contain schema version 1, model and Diffusers revisions, quantization fields, WebGPU feature and limit requirements, decoder and reduced-head graph metadata, exact ONNX external locations for both graphs, embedding-shard row ranges, tokenizer files, license file, GPU-output names, KV input/output pairs, and SHA-256 for every referenced file. Every cache path is relative to its release root. Reject duplicate paths and missing referenced files. Emit this complete manifest for both the one-layer diagnostic release and the full release.

Add `tests/fixtures/prompt-contract.json` with the exact conditional IDs:

```json
[151644,151671,11646,198,65,5187,374,220,24,21,198,95275,8778,25407,151672,151673,28463,921,58,4450,921,9707,198,58,6150,355,921,38102,198,80987,198,58,13709,1457,82,10011,60,151674,151645,151669]
```

The unconditional row retains the first and last two IDs and replaces indices 1 through 37 with token 151,654. Verify both arrays have length 40 before writing the release fixture.

- [ ] **Step 6: Run the unit and tiny conversion tests**

Run: `uv run pytest tests/python/test_global_decoder.py -q`

Expected: PASS.

Run: `uv run pytest tests/python/test_converter_smoke.py -m converter_smoke -q`

Expected: PASS with no MiniMax checkpoint download.

- [ ] **Step 7: Commit the reduced decoder tooling**

```powershell
git add tools/converter/src/minimax_music3_webgpu tests/python tests/fixtures/prompt-contract.json
git commit -m "feat(converter): build reduced q4 decoder"
```

### Task 5: Add OPFS, JSPI, and WebGPU diagnostic runtime

**Files:**
- Create: `src/runtime/model/manifest.ts`
- Create: `src/runtime/model/artifact-cache.ts`
- Create: `src/runtime/model/embedding-table.ts`
- Modify: `src/runtime/model/webgpu-device.ts`
- Create: `src/runtime/model/ort-session.ts`
- Create: `src/runtime/pipeline/gpu-kv-state.ts`
- Create: `src/workers/protocol.ts`
- Create: `src/workers/inference.worker.ts`
- Modify: `src/app/App.tsx`
- Create: `tests/unit/model/manifest.test.ts`
- Create: `tests/unit/model/artifact-cache.test.ts`
- Create: `tests/unit/model/embedding-table.test.ts`
- Create: `tests/unit/pipeline/gpu-kv-state.test.ts`
- Create: `tests/browser/external-data-opfs.spec.ts`
- Create: `tests/browser/matmul-nbits.spec.ts`

**Interfaces:**
- Produces: `parseModelManifest(value: unknown) -> ModelManifest`.
- Produces: `ensureArtifact(file: ArtifactFile, source: URL, store: ArtifactStore, onProgress: ProgressSink) -> Promise<File>`.
- Produces: `OpfsFp16EmbeddingTable.lookup(ids: readonly number[]) -> Uint16Array`.
- Produces: `createOrtSession(graph: OnnxGraphArtifact, cache: ArtifactStore, device: GPUDevice) -> Promise<ort.InferenceSession>`.
- Produces: `GpuKvState.advance(outputs: Record<string, ort.Tensor>) -> void` and `GpuKvState.dispose() -> void`.
- Produces: Worker request `run-global-smoke` and result `GlobalSmokeResult`.

Use these shared manifest types:

```ts
export interface ArtifactFile {
  path: string;
  bytes: number;
  sha256: string;
}

export interface ExternalDataArtifact extends ArtifactFile {
  onnxLocation: string;
}

export interface OnnxGraphArtifact extends ArtifactFile {
  externalData: readonly ExternalDataArtifact[];
  gpuOutputs: readonly string[];
}

export interface EmbeddingShard extends ArtifactFile {
  rowStart: number;
  rowCount: number;
}

export interface Fp16EmbeddingTable {
  rows: number;
  columns: number;
  rowBytes: number;
  shards: readonly EmbeddingShard[];
}

export interface KvPairSpec {
  pastInput: string;
  presentOutput: string;
}
```

- [ ] **Step 1: Write manifest, embedding, cache, and KV ownership tests**

```ts
it('looks up duplicated ids across a shard boundary in input order', async () => {
  const table = await makeEmbeddingTable({ rowsPerShard: 2, columns: 4 });
  const result = table.lookup([1, 2, 1]);
  expect(Array.from(result)).toEqual([
    4, 5, 6, 7,
    8, 9, 10, 11,
    4, 5, 6, 7,
  ]);
});
```

The cache tests must cover a completed file, HTTP range resume from an existing partial file, a server that ignores `Range`, a SHA-256 mismatch, one retry of only the bad file, and zero fetches on a verified second visit. The embedding tests must also cover an invalid ID, a short sync-access-handle read, and handle closure. The KV tests must assert `gpu-buffer` location and disposal only after the next run consumes the previous state.

- [ ] **Step 2: Run the unit tests and verify missing implementations**

Run: `npm test -- tests/unit/model tests/unit/pipeline/gpu-kv-state.test.ts`

Expected: FAIL because the runtime modules do not exist.

- [ ] **Step 3: Implement manifest parsing and resumable OPFS storage**

Use one versioned OPFS directory selected by the manifest hash. Stream fetch bodies into `FileSystemWritableFileStream`, resume with `Range: bytes=<existing>-`, hash incrementally, and atomically mark completion only after size and SHA-256 match. Never create an IndexedDB or service-worker copy.

- [ ] **Step 4: Implement worker-only embedding lookup**

Open each embedding shard with `createSyncAccessHandle()`. Calculate `rowOffset = (id - rowStart) * rowBytes`, read exactly one 8 KiB row into its destination slice, preserve duplicate and input ordering, and close every handle in `dispose()` and on partial construction failure.

- [ ] **Step 5: Implement explicit WebGPU and JSPI ORT session ownership**

Inside the inference Worker, request a high-performance non-fallback adapter with `shader-f16`, then request one device using manifest limits. Import `onnxruntime-web/jspi`, set `ort.env.wasm.proxy = false`, `ort.env.wasm.numThreads = 1`, and create sessions with WebGPU only, sequential execution, graph optimization, graph capture disabled, CPU fallback disabled, cache modes conservative, GPU preferred outputs, and external data passed as OPFS `File` objects without calling `arrayBuffer()`.

- [ ] **Step 6: Implement worker protocol and diagnostic UI**

The UI starts one Worker, shows adapter and artifact progress, runs the smoke on explicit user action, supports cancellation through Worker termination, and renders the structured result. `postMessage` may carry JSON data only. It may not transfer GPU resources or ORT tensors.

- [ ] **Step 7: Add the JSPI external-data browser smoke**

Generate a tiny external-data ONNX fixture during test setup, cache it in OPFS, override the returned `File` object's `arrayBuffer()` to throw, and create a JSPI WebGPU session. PASS proves initializer-range reads do not materialize the complete file.

- [ ] **Step 8: Add the q4 known-answer browser smoke**

Generate a small symmetric `MatMulNBits` model with bits 4, block size 128, FP16 input and scales, no zero points, no `g_idx`, and accuracy level 4. Compare its WebGPU output with a frozen FP32 calculation using absolute and relative tolerance `0.02`.

- [ ] **Step 9: Run the browser-runtime checks and commit**

Run: `npm test -- tests/unit/model tests/unit/pipeline/gpu-kv-state.test.ts`

Run: `npm run typecheck && npm run lint && npm run build`

Run: `npm run test:browser -- tests/browser/external-data-opfs.spec.ts tests/browser/matmul-nbits.spec.ts`

Expected: all commands pass in branded Chrome with one test worker.

```powershell
git add src tests package.json package-lock.json playwright.config.ts
git commit -m "feat(runtime): add WebGPU model loader"
```

### Task 6: Run the exact-checkpoint Global LLM feasibility gate

**Files:**
- Create: `tools/serve-artifacts.mjs`
- Create: `tests/browser/global-decoder.spec.ts`
- Create: `docs/development/conversion.md`
- Create: `docs/development/global-llm-feasibility.md`
- Modify: `README.md`

**Interfaces:**
- Produces: a local byte-range artifact endpoint for files below `artifacts/release/global/`.
- Produces: `GlobalSmokeResult` containing adapter info, session-create time, ten per-step timings, cache lengths, tensor locations, finite-logit status, owned tensor bytes, and cache-reuse counts.
- Produces: a documented PASS or FAIL gate tied to exact source and converted-manifest hashes.

- [ ] **Step 1: Write the real decoder test before artifacts exist**

```ts
test('prefills and performs ten GPU-resident cached decodes', async ({ page }) => {
  await page.goto('/diagnostics/global');
  await page.getByRole('button', { name: 'Run Global LLM smoke' }).click();
  const result = page.getByTestId('global-smoke-result');
  await expect(result).toContainText('steps: 10', { timeout: 30 * 60_000 });
  await expect(result).toContainText('finite logits: yes');
  await expect(result).toContainText('KV location: gpu-buffer');
});
```

- [ ] **Step 2: Run the test and verify the missing-artifact failure**

Run: `npm run test:browser -- tests/browser/global-decoder.spec.ts`

Expected: FAIL with a user-visible missing `artifacts/release/global/manifest.json` diagnostic.

- [ ] **Step 3: Implement the byte-range development artifact server**

Serve only files resolved below `artifacts/release/global/`, support `HEAD` and one `Range` request, set `Accept-Ranges: bytes`, reject path traversal, and use `Access-Control-Allow-Origin` only for the configured local Vite origin.

- [ ] **Step 4: Download only the exact Global source files**

Run: `uv run music3-convert download-global --artifacts-dir artifacts`

Expected: approximately 17.2 GB decimal of language-model weights plus tokenizer, configuration, and license files under `artifacts/source/`. Interrupted runs reuse Hugging Face Xet cache and the pinned receipt.

- [ ] **Step 5: Convert and test one real checkpoint layer**

Run: `uv run music3-convert build-global --artifacts-dir artifacts --layers 1`

Run the same headed diagnostic against `artifacts/release/global-one-layer/manifest.json`.

Expected: one real Qwen3 layer prefills, performs ten cached decodes, keeps KV tensors at `gpu-buffer`, and returns finite reduced logits without CPU fallback.

- [ ] **Step 6: Convert and validate the full exact Global release**

Run: `uv run music3-convert build-global --artifacts-dir artifacts --layers 36`

Expected: thirteen prompt-embedding row files, a q4 decoder graph with external files no larger than 128 MiB, an FP16 reduced head, tokenizer files, license, receipts, and a complete hashed manifest. The decoder core session must remain below the 4 GiB model limit, and the reduced head remains a separate session.

- [ ] **Step 7: Run the real headed Chrome smoke with GPU telemetry**

Start the Vite and artifact servers, then run:

Run: `npm run test:browser -- tests/browser/global-decoder.spec.ts`

In parallel, sample `nvidia-smi --query-gpu=timestamp,memory.used --format=csv` from before session creation through Worker disposal. Record baseline, peak, and delta in the feasibility document.

Expected: JSPI range loading passes, the real session creates without CPU fallback, the exact 40-token batch-two fixture prefills, 36 KV pairs remain `gpu-buffer`, cache length advances from 40 to 50 across ten deterministic embedding-row decodes, reduced logits remain finite, superseded tensors are disposed, peak allocation stays at or below 12 GB, and a page reload performs zero completed-artifact downloads.

- [ ] **Step 8: Record the gate result and update development documentation**

`docs/development/global-llm-feasibility.md` must record exact source revision, manifest hash, artifact sizes, Chrome and ORT versions, adapter limits, session and step timings, GPU baseline and peak, test command, and complete PASS or FAIL evidence. A failure record includes the first unsupported operator or allocation error and does not claim feasibility.

- [ ] **Step 9: Run the complete milestone verification**

Run: `uv run pytest tests/python -q`

Run: `npm test`

Run: `npm run typecheck && npm run lint && npm run build`

Run: `npm run test:browser -- tests/browser/external-data-opfs.spec.ts tests/browser/matmul-nbits.spec.ts tests/browser/global-decoder.spec.ts`

Expected: every command passes for a PASS gate. If the real gate fails, all non-real-model checks still pass and the feasibility document truthfully records the blocker.

- [ ] **Step 10: Commit the measured feasibility result**

```powershell
git add tools/serve-artifacts.mjs tests/browser/global-decoder.spec.ts docs/development README.md
git commit -m "test: record Global LLM WebGPU feasibility"
```

## Milestone handoff

After a PASS result, write the next implementation plan for RVQ depth generation, feedback embedding, one flow chunk, vocoder decode, and the first 10-second WAV. Carry forward the exact prompt and sampler fixtures, GPU-resident KV chain, artifact manifest, and OPFS cache from this plan.

If session creation fails specifically because the reduced decoder still exceeds a browser allocation limit, revise the design and next plan to split the 36 layers into two 18-layer decoder graphs. If an unsupported WebGPU operator is the measured blocker, isolate that operator in a micrograph before deciding whether a minimal custom WGSL kernel is justified. Do not implement either fallback without the corresponding measured failure.
