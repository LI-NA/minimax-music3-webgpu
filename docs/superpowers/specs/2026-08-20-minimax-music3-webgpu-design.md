# MiniMax Music 3 WebGPU Phase 1 Design

Date: 2026-08-20

Status: Approved architecture, awaiting written-spec review

## 1. Summary

This project will port the exact `MiniMaxAI/MiniMax-Music3` checkpoint to a local browser runtime. Phase 1 ends when a Windows desktop Chromium browser can automatically download and cache converted model artifacts, accept a music description and lyrics, and generate a short WAV file containing intelligible vocals and accompaniment.

The product runtime is browser-only. Python, PyTorch, CUDA, and ONNX conversion tools are development dependencies, not user requirements.

## 2. Confirmed requirements

- Use the exact official MiniMax Music 3 checkpoint. Do not substitute or distill another model.
- Direct weight quantization is allowed. Quantized output does not need to match BF16 output numerically.
- Target Windows desktop Chrome and Edge in phase 1.
- Use WebGPU for model inference.
- Keep practical peak GPU allocation at or below 12 GB on the RTX 4080 development machine.
- Generate a 10 to 30 second result with both vocals and accompaniment.
- Accept a music description and user-provided lyrics with section tags.
- Download converted artifacts automatically and persist them in OPFS.
- Provide download progress, generation progress, cancellation, playback, and WAV download.
- Keep the project noncommercial and experimental.
- Validate final audio through one real browser generation and user listening.

## 3. Explicit non-goals

Phase 1 will not include:

- five-minute full-song generation;
- Firefox, Safari, mobile, or integrated-GPU support;
- a Python, CUDA, or Node inference server;
- an Electron or native desktop wrapper;
- a polished Suno-like production interface;
- automatic prompt rewriting with another language model;
- reference-audio upload, voice cloning, or cover generation;
- automated music-quality scoring or signal analysis;
- BF16 waveform or seed-for-seed output comparison;
- real-time generation guarantees;
- service workers or a second model cache; or
- a general-purpose custom WebGPU machine-learning runtime.

## 4. Source model

The browser port is based on the official Diffusers implementation and official converted checkpoint layout.

The generation pipeline contains:

- an 8B Qwen3-based Global LLM;
- a 0.6B RVQ depth decoder;
- a condition encoder;
- a 2.4B flow-matching transformer; and
- a 123M DAC-style vocoder.

The official Hugging Face repository contains both legacy and Diffusers artifacts and totals about 57.35 GB. Only browser-converted artifacts derived from the Diffusers components will be downloaded by the application.

## 5. Architecture decision

### 5.1 Runtime boundary

ONNX Runtime Web with the WebGPU execution provider is the primary tensor runtime. TypeScript owns all stateful and stochastic control flow. A single dedicated Web Worker owns ONNX Runtime and WebGPU resources so the UI remains responsive and GPU buffers do not cross incompatible proxy boundaries.

Custom WGSL is a fallback for a confirmed missing operator or memory-loading blocker. It is not part of the initial implementation.

```text
Browser UI
  -> model manifest and OPFS cache
  -> dedicated inference Worker
     -> prompt preparation and tokenization
     -> Global LLM and RVQ frame generation
     -> condition encoding and flow matching
     -> vocoder, chunk stitching, and WAV encoding
  -> audio playback and file download
```

### 5.2 Application modules

The implementation will keep the following boundaries:

- `app`: minimal React UI and user-visible state.
- `runtime/model`: manifest validation, downloads, OPFS storage, and model versions.
- `runtime/pipeline`: TypeScript generation state machine and model-stage lifecycle.
- `runtime/audio`: chunk stitching and WAV serialization.
- `workers`: the dedicated inference Worker and typed message protocol.
- `tools/converter`: `uv`-managed Python conversion and quantization commands.
- `tests`: unit, converter-smoke, and browser-smoke coverage.

The project will remain a single web application rather than a monorepo.

## 6. Model execution design

### 6.1 Input preparation

TypeScript will reproduce the checkpoint contract for caption cleanup, lyric normalization, special-token prompt assembly, and conditional/unconditional token IDs. The official Qwen tokenizer vocabulary is packaged with the browser artifacts.

The UI will preserve lyric section tags such as `[verse]`, `[chorus]`, and `[bridge]`. It will explain that a tag must be on its own line.

### 6.2 Autoregressive stage

The Global LLM first processes the conditional and unconditional prompts as a batch of two and initializes a shared GPU KV cache. At 25 frames per second it then:

1. predicts the semantic audio token with classifier-free guidance;
2. samples from the allowed semantic vocabulary and end token;
3. runs seven RVQ depth steps for the remaining codebooks;
4. constructs the audio-frame feedback embedding; and
5. retains the eight hidden-state groups used by the acoustic stage.

Sampling, vocabulary masks, top-k selection, end-token handling, and seeded random-number generation remain in TypeScript. ONNX graphs perform only deterministic tensor computation.

The exported graph set is:

- a temporary full prompt-embedding graph;
- a q4 Global LLM decoder core with shared past and present KV buffers;
- an exact reduced LM head containing only the 16,384 semantic rows and end-token row used by inference;
- a small audio-feedback embedding graph;
- an FP16 RVQ depth graph; and
- tokenizer data.

Reducing the LM head is exact because the official inference code masks every other output row before sampling. It does not change the probability distribution.

The full prompt-embedding session is released after prefill. The smaller audio-feedback embedding remains available during frame generation.

### 6.3 Acoustic stage

The condition encoder mixes the eight hidden-state groups and resamples them from 25 semantic frames per second to the Flow-VAE latent rate.

Semantic frames are processed in 200-frame windows with a 100-frame hop. Each window runs 30 Euler flow-matching steps with guidance scale 1.7. Conditional and unconditional passes may be batched only if measurement confirms the combined activation peak remains below 12 GB. Sequential passes are the safe default.

For a 30-second request without an early end token, the stage processes 750 semantic frames across seven overlapping chunks. Shorter output uses fewer chunks.

After denoising, the FP16 vocoder decodes each latent chunk to native 44.1 kHz stereo audio. The runtime applies the official 86-frame left crop and 258-frame right crop where applicable, concatenates the chunks, clamps samples to `[-1, 1]`, and serializes a 16-bit stereo WAV file.

## 7. ONNX model-size strategy

ONNX Runtime Web documents a 4 GB WebAssembly memory limit for a single large model. A naive q4 export of the complete 8.584B-parameter language model would exceed it before graph overhead.

The initial export therefore removes the full input embedding and the unused output rows from the repeated decoder graph. The expected q4 decoder-core artifact is approximately 3.65 to 3.8 GB, including block scales and FP16 normalization weights. Creating that session in Chromium is the first go or no-go test.

If the reduced decoder still cannot create a WebGPU session, the only planned fallback is to split its 36 layers into two sequential decoder graphs. The project will not implement this fallback before the single reduced graph is tested.

The initial quality-oriented conversion target is:

| Component | Representation | Estimated artifact size |
| --- | ---: | ---: |
| Temporary full prompt embedding | FP16 | 1.64 GB |
| Global decoder core and reduced head | q4 weights, FP16 norms and cache | 3.65 to 3.8 GB |
| Audio-feedback embeddings | FP16 | 0.19 GB |
| RVQ depth decoder | FP16 | 1.29 GB |
| Flow transformer | q8 linear weights, FP16 supporting paths | 2.45 to 2.6 GB |
| Condition encoder | FP16 | 0.05 GB |
| Vocoder | FP16 | 0.108 GB |
| Tokenizer, graphs, and metadata | Mixed | 0.02 to 0.15 GB |

The expected OPFS cache is approximately 9.4 to 9.8 GB. Selective additional q4 conversion is deferred until a complete song has been generated and the user has identified a real memory or quality need.

## 8. GPU-memory strategy

All model stages will not be resident at once.

1. Load the temporary prompt embedding and Global LLM resources.
2. Release the temporary embedding after prefill.
3. Generate semantic frames and frame hidden states.
4. Release all autoregressive sessions and KV buffers.
5. Load the condition encoder and flow transformer.
6. Process one acoustic chunk at a time.
7. Release the flow stage.
8. Load the vocoder, decode chunks, then release it.

For 30 seconds, the retained FP16 frame-conditioning tensor is about 49 MB. A typical prompt plus 750-frame batch-two KV cache is estimated around 0.3 GB. Current estimates place the autoregressive peak between 6.5 and 9.5 GB and the flow-stage peak between 4 and 6 GB. These estimates include uncertainty from q4 prepacking and ONNX Runtime buffer pooling, so measured browser allocation decides acceptance.

The Worker will use GPU-resident past and present buffers and conservative ONNX Runtime Web buffer-cache settings. Dynamic decoder length means graph capture is not part of phase 1.

## 9. Artifact download and cache

The browser artifact release contains only converted artifacts, the model license, and a versioned manifest. It does not contain the legacy SGLang checkpoint layout or duplicate original weights.

The manifest records:

- artifact version and source checkpoint revision;
- graph and external-data shard paths;
- byte sizes and SHA-256 hashes;
- tensor representation;
- required WebGPU features and limits; and
- the MiniMax model-license revision.

External-data shards will be at most 128 MiB. The downloader streams directly into OPFS, tracks completed shards, verifies hashes, and retries only incomplete or invalid shards. It will not create an IndexedDB or service-worker copy.

On first use, the app requests persistent storage and confirms that enough quota is available. A manifest hash selects the cache version. The UI provides one explicit action to remove the project's cached model artifacts.

Local conversion inputs and generated artifacts will live under a dedicated project artifact directory. Before any large checkpoint download, that directory will be added to both `.gitignore` and `.kopiaignore`.

## 10. Minimal user interface

Phase 1 has one generation screen:

- model readiness and download progress;
- a music-description field;
- a lyrics field with a compact section-tag example;
- a short duration selector limited to 10, 20, or 30 seconds;
- Generate and Cancel actions;
- current stage and progress;
- a standard browser audio player; and
- a Download WAV action.

The interface will identify the model as `MiniMax-Music3`, link its license, and remind users to provide lyrics they have the right to use. It will mark downloaded output as AI-generated in the surrounding UI. No prompt library, history database, account system, or advanced editor is included.

## 11. Error handling

The application will stop early with a clear user-visible message for:

- a non-Chromium or unavailable WebGPU environment;
- missing `shader-f16` or insufficient adapter limits;
- insufficient persistent-storage quota;
- a failed or hash-mismatched model shard;
- an unsupported ONNX operator or unexpected CPU fallback;
- WebGPU out-of-memory or device loss;
- an invalid empty prompt or lyrics input; and
- cancellation.

Generation failures release all sessions and GPU tensors before allowing a retry. The app will not silently fall back to CPU inference because this model cannot fit the intended browser CPU path.

## 12. Privacy, safety, and licensing

Inference data remains local. The static host serves application files and converted model artifacts only. The app does not upload prompts, lyrics, weights, intermediate tensors, or audio.

The source repository will distinguish Apache-2.0 project code from model artifacts governed by the MiniMax-Music3 Community License. Every redistributed browser-artifact release must include the unmodified model license and required notices.

The first-run model notice will link the Acceptable Use Policy and explain lawful lyrics use and AI-output disclosure. Phase 1 will not add a separate moderation model or remote moderation service.

## 13. Verification

Automated verification is intentionally narrow.

### Unit checks

- Prompt assembly and lyric normalization preserve the checkpoint contract.
- The seeded sampler applies the official masks, guidance, and top-k rules.
- Chunk starts, overlap state, crop lengths, and final WAV headers are correct.
- Manifest parsing, OPFS version selection, interrupted downloads, and hash failures behave correctly.
- Worker cancellation releases sessions and tensors.

### Technical smoke checks

- Each converted ONNX graph creates a WebGPU session and returns finite tensors with the expected shapes.
- The reduced q4 Global LLM performs prefill and at least ten cached decode steps without CPU-visible KV copies.
- A single flow chunk runs first for one step and then for all 30 steps.
- The vocoder produces a non-empty stereo waveform and the browser can decode the resulting WAV container.

### Final product check

Run one 10 to 30 second generation in Chrome on the development machine. The user plays the generated WAV and confirms that it contains recognizable sung lyrics and accompaniment without an immediately obvious severe defect.

No automated audio analysis, quality score, or BF16 comparison is required.

## 14. Implementation sequence

The later implementation plan will use the following milestone order:

1. Prove the reduced q4 Global LLM session can load below the 4 GB runtime limit and perform cached decoding.
2. Add RVQ depth generation and audio-frame feedback for a few semantic frames.
3. Run one flow-transformer chunk and vocoder decode.
4. Complete a 10-second end-to-end browser generation.
5. Add 20 to 30 second chunk stitching, OPFS download and cache, and the minimal UI.
6. Measure the 12 GB GPU ceiling and perform the approved manual listening check.

The first milestone is the principal feasibility gate. UI polish and deeper optimization do not start until it passes.

## 15. Acceptance criteria

Phase 1 is complete when all of the following are true:

- A static site opens in supported Windows Chromium without a local inference server.
- It detects an appropriate WebGPU adapter and explains unsupported environments.
- It automatically downloads, verifies, and persists the converted model artifacts.
- A second visit reuses the OPFS cache without downloading complete artifacts again.
- The full browser pipeline stays at or below 12 GB practical GPU allocation on the RTX 4080 development machine.
- A music description and tagged lyrics produce a 10 to 30 second 44.1 kHz stereo WAV.
- The browser can play and download the WAV.
- The user manually confirms recognizable vocals and accompaniment.
- No prompt, lyrics, or generated audio is sent to an inference backend.
- Relevant automated checks, lint, type checking, and the browser smoke test pass.

## 16. Primary references

- [MiniMax Music 3 model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
- [MiniMax Music 3 official repository](https://github.com/MiniMax-AI/MiniMax-Music3)
- [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE)
- [Diffusers MiniMax Music 3 pipeline documentation](https://github.com/huggingface/diffusers/blob/3681e65996b4d2589219720101a6acbfd25073f8/docs/source/en/api/pipelines/minimax_music3.md)
- [Diffusers autoregressive pipeline source](https://github.com/huggingface/diffusers/blob/3681e65996b4d2589219720101a6acbfd25073f8/src/diffusers/modular_pipelines/minimax_music3/encoders.py)
- [Diffusers flow pipeline source](https://github.com/huggingface/diffusers/blob/3681e65996b4d2589219720101a6acbfd25073f8/src/diffusers/modular_pipelines/minimax_music3/denoise.py)
- [Diffusers vocoder stitching source](https://github.com/huggingface/diffusers/blob/3681e65996b4d2589219720101a6acbfd25073f8/src/diffusers/modular_pipelines/minimax_music3/decoders.py)
- [ONNX Runtime Web large-model guidance](https://onnxruntime.ai/docs/tutorials/web/large-models.html)
- [ONNX Runtime WebGPU operator support](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)
- [ONNX Runtime WebGPU GPU-tensor guidance](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [WebGPU specification and default limits](https://www.w3.org/TR/webgpu/)
