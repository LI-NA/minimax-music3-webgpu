# Five-Second Acoustic Vertical Slice Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` task by task. Every behavior change starts with a failing product-facing test. Do not start a later model stage before the preceding real Chrome gate passes.

**Goal:** Generate one approximately five-second vocal and accompaniment WAV in desktop Chrome with the exact MiniMax-Music3 checkpoint, entirely through WebGPU after automatic local artifact download.

**Architecture:** Extend the passing q4 Global LLM pipeline with one FP16 RVQ depth session, an OPFS residual embedding table, a small feedback graph, a fixed-shape condition graph, a fixed-length q4 flow graph, and a mixed-precision FP16 vocoder. TypeScript owns sampling and the 30-step loop. One inference Worker loads and releases stage groups so total device usage stays below 12 GB. The first slice uses the committed prompt fixture and one deterministic seed, then returns a PCM16 stereo WAV for manual listening.

**Tech stack:** Python 3.11+, uv, pinned Diffusers source, PyTorch, Safetensors, ONNX, ONNX Runtime desktop, React, TypeScript, Vite, Vitest, Playwright, ONNX Runtime Web JSPI, WebGPU, OPFS

**Specs:**

- `docs/superpowers/specs/2026-08-20-minimax-music3-webgpu-design.md`
- `docs/superpowers/specs/2026-08-21-five-second-acoustic-slice.md`

## Fixed contract

- Checkpoint: `MiniMaxAI/MiniMax-Music3` at `fbdf52fbaaca799592917417eb05f1899f1255ec`
- Diffusers reference: `3681e65996b4d2589219720101a6acbfd25073f8`
- Prompt: existing committed 40-token conditional and unconditional fixture
- Semantic frames: 125
- Latent length: 430
- Flow steps: 30
- Output: FP32 or FP16 waveform converted to PCM16 stereo, 44,100 Hz, 220,160 samples per channel
- Global and RVQ guidance: 1.5, top-k 50
- Flow guidance: 1.7
- Large linear quantization: symmetric q4, block size 128, accuracy level 4
- Physical artifact limit: 128 MiB
- Browser: branded desktop Chrome, WebGPU only, CPU execution-provider fallback disabled
- Peak total GPU usage: at or below 12 GB on the RTX 4080 test machine

All model source, conversion work, releases, persistent Chrome profiles, and generated audio stay under `artifacts/`. The directory is already ignored by Git and Kopia.

Do not download or use `qwen_7B/**`, `dav.pth`, `flowmatching_vae.pth`, a distilled checkpoint, or a substitute model. Do not add custom WGSL, layer splitting, multiple durations, overlap stitching, waveform scoring, or UI polish before a measured blocker requires it.

## Target release

`artifacts/release/music-5s/manifest.json` extends the passing Global release with:

```text
rvqDepth
rvqEmbedding
feedback
condition
flow
vocoder
slice:
  semanticFrames: 125
  latentLength: 430
  outputSamples: 220160
  sampleRate: 44100
  channels: 2
  flowSteps: 30
  globalGuidance: 1.5
  flowGuidance: 1.7
```

Each graph entry carries its ONNX graph, external-data locations, SHA-256 values, byte sizes, and GPU output names. The final product profile starts from this combined manifest so it does not duplicate the earlier diagnostic release in OPFS.

## Task 1: Pin the reference implementation and download only acoustic source

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tools/converter/src/minimax_music3_webgpu/constants.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/paths.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/cli.py`
- Create: `tools/converter/src/minimax_music3_webgpu/acoustic_source.py`
- Create: `tests/python/test_acoustic_source.py`

**Interfaces:**

- `music3-convert download-acoustic --artifacts-dir artifacts`
- `download_acoustic_source(paths: ArtifactPaths) -> AcousticSourceReceipt`
- Receipt: `artifacts/receipts/source-acoustic.json`

- [ ] Write a failing allow-list test that requires exactly the condition encoder, RVQ depth decoder, scheduler, two flow shards plus index/config, and vocoder config/weights.
- [ ] Reject every legacy file and any source path outside the validated artifact root.
- [ ] Pin Diffusers to the approved Git commit in the uv environment. Do not use a floating branch or PyPI version for the reference oracle.
- [ ] Implement the selective Hugging Face download with the same atomic receipt and containment rules as `download-global`.
- [ ] Record relative path, exact byte size, and SHA-256 for every selected file. Expected weight download is approximately 11.34 GB decimal.
- [ ] Run `uv run pytest tests/python/test_acoustic_source.py tests/python/test_paths.py -q`.
- [ ] Run `uv run music3-convert download-acoustic --artifacts-dir artifacts` and verify the pinned receipt.

Expected source patterns:

```text
condition_encoder/config.json
condition_encoder/diffusion_pytorch_model.safetensors
rvq_depth_decoder/config.json
rvq_depth_decoder/diffusion_pytorch_model.safetensors
scheduler/scheduler_config.json
transformer/config.json
transformer/diffusion_pytorch_model.safetensors.index.json
transformer/diffusion_pytorch_model-00001-of-00002.safetensors
transformer/diffusion_pytorch_model-00002-of-00002.safetensors
vocoder/config.json
vocoder/diffusion_pytorch_model.safetensors
```

Commit after the real receipt is verified:

```text
feat(converter): add acoustic source downloader
```

## Task 2: Export the exact RVQ depth and feedback stage

**Files:**

- Create: `tools/converter/src/minimax_music3_webgpu/rvq_depth.py`
- Create: `tests/python/test_rvq_depth.py`
- Create: `tests/fixtures/rvq-contract.npz` or an equivalent compact JSON fixture
- Modify: `tools/converter/src/minimax_music3_webgpu/cli.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/manifest.py`
- Modify: `src/runtime/model/manifest.ts`
- Create: `tests/browser/rvq-depth.spec.ts`

**Graph contract:**

```text
rvq-depth.onnx
  global_last_hidden    float16 [2,4096]
  semantic_embedding    float16 [2,4096]
  residual_embeddings  float16 [2,6,4096]
  depth_index           int64   []
  -> depth_hidden       float16 [2,4096]
  -> depth_logits       float32 [2,7,1024]

feedback.onnx
  semantic_rows  float16 [2,1,4096]
  residual_rows  float16 [2,7,4096]
  -> inputs_embeds float16 [2,1,4096]
```

The fixed eight-position RVQ graph avoids runtime `Shape`, `Range`, and `Trilu`. It constructs `[Global hidden, semantic embedding, c1..c6]`; unused residual positions are zero padded. Fixed causal attention guarantees that future zero positions cannot affect the current row. The graph computes all seven small heads, casts their logits to FP32 like the reference implementation, and TypeScript consumes the head selected by `depth_index`.

- [ ] Write failing tests for the exact four-layer topology, 16 heads, 4,096 hidden size, 6,144 feed-forward size, seven 1,024-row heads, and the original safetensor key mapping.
- [ ] Add a fixed-shape causal-mask oracle for valid lengths 2 through 8. Compare the zero-padded graph's current hidden state and selected logits with the pinned Diffusers prefix module on a small deterministic model before loading the real checkpoint.
- [ ] Export `audio_embeddings.weight` as a separate FP16 row-major table `[7168,4096]`, exactly 56 MiB, with the existing row-shard receipt format.
- [ ] Export the remaining 46 RVQ tensors in FP16 with every initializer below 128 MiB. Use explicit constant shapes, four expanded MatMul and Softmax causal-attention blocks, nine RMSNorm applications, four gated MLPs, and all seven output heads.
- [ ] Export the exact feedback expression `(semantic + sum(residuals)) / sqrt(8)` and declare its output at `gpu-buffer`.
- [ ] Validate every graph with ONNX checker and desktop ORT. Assert finite outputs for all seven current indices.
- [ ] Add a real headed Chrome gate that creates the RVQ and feedback sessions with CPU fallback disabled, runs valid lengths 2 through 8, and keeps hidden and feedback outputs GPU resident.
- [ ] Record session time, step time, and VRAM delta before integration with the Global decoder.

Do not q4 the RVQ stage unless the combined real browser gate exceeds 12 GB or session creation proves an aggregate runtime limit.

Commit:

```text
feat(converter): export RVQ depth stage
```

## Task 3: Replace the diagnostic row with real semantic and RVQ frame generation

**Files:**

- Create: `src/runtime/pipeline/sampler.ts`
- Create: `src/runtime/pipeline/rvq-generation.ts`
- Create: `tests/unit/pipeline/sampler.test.ts`
- Create: `tests/unit/pipeline/rvq-generation.test.ts`
- Modify: `src/workers/inference.worker.ts`
- Modify: `src/workers/protocol.ts`
- Create: `tests/browser/autoregressive-frames.spec.ts`

**Interfaces:**

```ts
interface GeneratedFrame {
  semantic: number;
  residual: readonly [number, number, number, number, number, number, number];
  hiddenGroups: Float32Array;
}

generateFrames(options: {
  maxFrames: number;
  seed: number;
  guidance: 1.5;
  topK: 50;
}): Promise<readonly GeneratedFrame[]>
```

- [ ] Write a failing sampler test for two-lane classifier-free guidance, the 16,384 semantic rows plus end row, stable top-k ordering, residual top-k 50, and deterministic injected random draws.
- [ ] Freeze a small Python reference fixture for the first semantic decision, seven residual decisions, feedback row, and cache growth. Do not require JavaScript and PyTorch RNG streams to match globally.
- [ ] Implement the official audio-start transition, semantic sampling, seven depth calls, residual OPFS lookup, feedback graph, and next Global cached decode.
- [ ] Expose the reduced head's existing gathered `last_state [2,4096]` as a GPU output so the prefill path enters RVQ without downloading the full Global hidden tensor.
- [ ] Download only the small logits used by sampling. Keep Global KV, Global hidden handoff, RVQ hidden outputs, and feedback at GPU buffer locations until their consumers finish.
- [ ] Retain only the conditional lane's eight hidden groups on CPU for the later condition encoder. The complete FP16 125-frame tensor is 8,192,000 bytes.
- [ ] First run two real frames in headed Chrome. Assert semantic and residual ranges, finite hidden groups, correct Global cache growth, and no CPU fallback.
- [ ] Then run the exact fixed loop: 126 semantic samples, 882 RVQ depth calls, 125 Global feedback decodes, discard the warm-up result, and retain 125 emitted frames. If the end token appears early, retry once with the next seed and record both seeds.
- [ ] Measure combined Global, head, RVQ, feedback, KV, and activation peak. Stop if total device usage exceeds 12 GB.

Commit:

```text
feat(runtime): generate RVQ audio frames
```

## Task 4: Export the fixed 125-frame condition encoder

**Files:**

- Create: `tools/converter/src/minimax_music3_webgpu/condition_encoder.py`
- Create: `tests/python/test_condition_encoder.py`
- Create: `tests/browser/condition-encoder.spec.ts`

**Graph contract:**

```text
condition-125.onnx
  frame_hiddens float16 [1,125,32768]
  -> condition  float16 [1,430,2048]
```

- [ ] Write a failing exact-key test for the eight layer weights, scalar scale, and Conv1D projection.
- [ ] Freeze the softmax of `layer_weight_logits` during conversion.
- [ ] Replace runtime nearest-neighbor resize with a converter-generated 430-index Gather table whose output is verified against the pinned Diffusers oracle.
- [ ] Use Mul plus ReduceSum rather than Einsum. Keep the fixed graph free of runtime Shape and Resize operations.
- [ ] Compare the real graph with the reference module on deterministic frame hiddens using realistic tolerance for FP16 conversion.
- [ ] Run a headed Chrome WebGPU-only session and assert output shape `[1,430,2048]`, finite data, and GPU buffer location.

Commit:

```text
feat(converter): export condition encoder
```

## Task 5: Export and gate the fixed q4 flow step

**Files:**

- Create: `tools/converter/src/minimax_music3_webgpu/flow_transformer.py`
- Create: `tests/python/test_flow_transformer.py`
- Create: `tests/fixtures/flow-contract.npz` or an equivalent compact fixture
- Modify: `tools/converter/src/minimax_music3_webgpu/cli.py`
- Create: `src/runtime/pipeline/flow-generation.ts`
- Create: `tests/unit/pipeline/flow-generation.test.ts`
- Create: `tests/browser/flow-step.spec.ts`

**Graph contract:**

```text
flow-step-430-q4.onnx
  latents    float16 [1,128,430]
  condition  float16 [1,430,2048]
  timestep   float16 [1]
  dt         float32 [1]
  -> next_latents float16 [1,128,430]
```

The graph constructs conditional and zero-condition lanes, runs the exact 36-layer transformer once at batch two, applies `uncond + 1.7 * (cond - uncond)`, performs the Euler update in FP32, and returns FP16 latent state at `gpu-buffer`.

- [ ] Write failing tests for all 441 source keys, 36 blocks, 32 heads, 2,048 hidden size, 8,192 feed-forward size, 128 latent channels, and exact fixed RoPE tables.
- [ ] Export and run a one-block FP16 fixture first. Require an ONNX graph without runtime Shape, Range, ConstantOfShape, or CPU fallback.
- [ ] Compare one deterministic reference step against the pinned Diffusers transformer and scheduler before quantization.
- [ ] Freeze all 30 float32 scheduler `dt` bit patterns from the pinned scheduler. Do not replace them with a repeated mathematical `1/30`, because its finite-precision sequence is not identical.
- [ ] Quantize all 220 large Linear MatMuls to symmetric q4 MatMulNBits with block size 128. Keep convolution, normalization, bias, Fourier, and scalar weights in FP16.
- [ ] Run export, quantization, and final 128 MiB repacking in separate processes so their peak allocations do not overlap. Never materialize the full 9.73 GB source or a 4.5 GB ONNX initializer blob in one Python object.
- [ ] Validate the full 36-layer graph, initializer byte ranges, and expected approximate 1.18 GiB q4 tensor payload.
- [ ] Run a real headed Chrome one-step gate with CPU fallback disabled and latent output at `gpu-buffer`.
- [ ] Chain the same session for 30 exact scheduler steps without CPU latent readback. Assert finite final latent data and record per-step time and VRAM peak.

Do not split conditional and unconditional passes or add custom WGSL unless the measured batch-two gate fails.

Commit:

```text
feat(converter): export q4 flow stage
```

## Task 6: Export the fixed vocoder and encode WAV

**Files:**

- Create: `tools/converter/src/minimax_music3_webgpu/vocoder.py`
- Create: `tests/python/test_vocoder.py`
- Create: `src/runtime/audio/wav.ts`
- Create: `tests/unit/audio/wav.test.ts`
- Create: `tests/browser/vocoder.spec.ts`

**Graph contract:**

```text
vocoder-430.onnx
  latents float16 [1,128,430]
  -> waveform float32 [1,2,220160]
```

- [ ] Write failing tests for the four upsample strides `(8,8,4,2)`, 12 residual units, dilation pattern `(1,3,9)`, exact output length, and every source key.
- [ ] Fold each weight-normalized convolution with `weight = g * v / norm(v)` in FP32 before converting the ordinary Conv or ConvTranspose weight to FP16.
- [ ] Keep convolution weights and activations FP16. Inspect every Snake alpha during conversion and run only modules with `abs(alpha) < 2^-14` in FP32, then cast back to FP16. At the pinned checkpoint those modules must be exactly `blocks.0.snake1` and `blocks.1.snake1`. Preserve the original Reciprocal, Sin, Pow, Mul, and Add order.
- [ ] Export the fixed graph at opset 18 with `torch.onnx.export(..., dynamo=False)`. Remove only reshapes proven redundant by the fixed three-dimensional input contract. Express stereo fold and unfold with fixed Split and Concat operations.
- [ ] Compare the real graph output to the pinned Diffusers vocoder on deterministic latent input. Assert exact shape and finite values.
- [ ] Run focused headed Chrome micrographs for mixed-precision Snake and ConvTranspose before the full vocoder.
- [ ] Require 27 Conv, four ConvTranspose, 29 Sin, 29 Reciprocal, 29 Pow, two Split, and two Concat nodes, with no Reshape, Shape, or ReduceL2 nodes. Keep every initializer and activation binding below 128 MiB. The expected initializer payload is approximately 108.3 MB.
- [ ] Run the full vocoder with CPU fallback disabled, cast the final Tanh result to FP32, download the waveform once, clamp to `[-1,1]`, interleave stereo channels, and emit a canonical PCM16 WAV header.
- [ ] Unit-test RIFF size, format fields, sample rate, channel count, sample count, clipping, and little-endian sample order.

Commit:

```text
feat(converter): export five-second vocoder
```

## Task 7: Assemble the combined release and one-click browser generation

**Files:**

- Create: `tools/converter/src/minimax_music3_webgpu/acoustic_manifest.py`
- Modify: `tools/converter/src/minimax_music3_webgpu/cli.py`
- Modify: `tools/serve-artifacts.mjs`
- Modify: `src/runtime/model/manifest.ts`
- Create: `src/runtime/pipeline/music-generation.ts`
- Modify: `src/workers/inference.worker.ts`
- Modify: `src/workers/protocol.ts`
- Modify: `src/app/App.tsx`
- Modify: `src/app/styles.css`
- Create: `tests/python/test_acoustic_manifest.py`
- Create: `tests/unit/pipeline/music-generation.test.ts`
- Create: `tests/browser/music-generation.spec.ts`
- Create: `docs/development/acoustic-conversion.md`
- Create: `docs/development/five-second-generation.md`

**Commands:**

```powershell
uv run music3-convert build-music-5s --artifacts-dir artifacts
$env:MINIMAX_RELEASE = 'music-5s'
$env:MINIMAX_CHROME_PROFILE = 'artifacts/music-browser-profile'
npx playwright test tests/browser/music-generation.spec.ts --project=chrome
```

- [ ] Write a failing manifest assembly test that requires all Global and acoustic graphs, exact slice constants, source revisions, license, tokenizer, external-data locations, file hashes, and no file above 128 MiB.
- [ ] Build the release transactionally in a fresh staging directory and promote it only after every referenced file rehashes successfully.
- [ ] Extend the Worker protocol with generation progress, cancellation by Worker termination, and a transferable final WAV ArrayBuffer.
- [ ] Implement explicit stage release boundaries. Autoregressive sessions must be gone before condition and flow load; flow must be gone before vocoder load.
- [ ] Add one minimal Generate button, progress text, an audio element, and a WAV download link. Use the committed prompt fixture for this first gate. Do not add editable prompt controls yet.
- [ ] Run the real 125-frame generation, fixed condition pass, 30 flow steps, vocoder, and WAV assembly in headed Chrome.
- [ ] Save the Playwright download to `artifacts/generated/minimax-music3-5s.wav`.
- [ ] Repeat once with the same persistent profile and assert zero completed-artifact fetches.
- [ ] Sample `nvidia-smi` every 0.5 seconds across all stage transitions. Record baseline, each stage peak, total peak, session times, frame rate, flow times, vocoder time, artifact bytes, manifest hash, Chrome version, and ORT version.
- [ ] Require total device usage at or below 12 GB. On a measured failure, stop and document the first exact allocation or unsupported operator before changing architecture.

Commit implementation and the measured result separately:

```text
feat(runtime): generate five-second music
docs: record acoustic WebGPU gate
```

## Task 8: Final verification and user handoff

- [ ] Run `uv run pytest -q`.
- [ ] Run `npm test`.
- [ ] Run `npm run lint`.
- [ ] Run `npm run typecheck`.
- [ ] Run `npm run build`.
- [ ] Run existing external-data, q4, Global, RVQ, condition, flow, and vocoder browser gates in branded Chrome with one worker.
- [ ] Run the final music-generation browser gate from a persistent profile.
- [ ] Check the final Git diff and working tree.
- [ ] Provide the generated WAV to the user and ask them to play it once and report whether the vocals and accompaniment sound correct.

Do not inspect the waveform for musical quality, run audio classifiers, compare it with a reference song, or add further UI before the user listening check.
