# Variable-duration WebGPU generation and reference parity design

> The 5-second product increment in this historical design is superseded by the approved programmatic input contract dated 2026-08-22. Product duration now uses `floor(durationSeconds * 25)` for any finite positive value through 300 seconds.

**Date:** 2026-08-21

## Status

Implemented and measured on the development RTX 4080. This specification extends the completed five-second vertical slice. It does not reopen rejected cache or prepacking experiments.

The archived pre-programmatic release reached its requested maximum through the 60-second seed-7 gate. A 120-second request ended naturally at retained frame 1,743 and produced about 69.799 seconds of valid audio. A separate 300-second capacity diagnostic completed all 7,500 frames by suppressing 83 audio-end decisions, first at frame 1,743. That diagnostic qualifies the archived runtime workload and memory architecture, not native five-minute music quality. The active programmatic-input release has separate 6 and 10-second browser evidence.

## Objective

Make the current MiniMax-Music3 WebGPU port practical before full UI work by:

1. retaining only measured VRAM and speed optimizations;
2. documenting hardware guidance that is supported by measurements;
3. supporting any finite positive requested maximum through 300 seconds at 25-frame-per-second resolution, subject to staged browser qualification;
4. preserving truthful progress, rate, ETA, and cancellation behavior for variable workloads; and
5. preparing a fixed-input original-model comparison that separates quantization loss from WebGPU implementation error.

## Non-goals

- A polished production UI.
- Custom WebGPU kernels or a second inference runtime.
- A high-memory performance profile without a measured benefit.
- Exact waveform equality between ordinary PyTorch and browser runs that merely share a seed.
- Advertising native five-minute music or quality from a capacity diagnostic that suppresses audio-end.
- Subjective audio scoring during the implementation gates. Structural and numeric audio checks remain required.

## Proven baseline

The fixed five-second path is the retained baseline:

- 125 semantic frames, 430 acoustic latents, 220,160 stereo samples at 44.1 kHz.
- Total-system GPU peak of 6,834 to 6,835 MiB on the development system.
- 10 GiB short-workload recommendation and a 12,288 MiB incremental project budget above the pre-run baseline.
- Sequential stage lifetimes, disabled memory patterns, disabled graph capture, disabled prepacking, and bucketed storage-buffer caching.
- Manifest-scoped OPFS completion receipts.
- Progress events for artifact work, sessions, autoregressive frames, flow steps, vocoder, WAV assembly, completion, and cancellation.

The rejected `simple` storage cache and decoder prepacking experiments remain rejected. Neither becomes a product profile.

## Artifact ownership

The main checkout's `artifacts/` directory is canonical.

- Existing physical copies are preserved unless the user explicitly authorizes deletion.
- A future linked worktree reads canonical artifacts through one whole-directory Windows junction. It does not copy the artifact tree.
- Downloads and converter builds run only from the main checkout, one at a time.
- Every linked worktree uses a distinct persistent Chrome profile below `artifacts/worktree-profiles/<worktree-name>/<suite>`.
- A worktree junction is verified against the canonical target and removed non-recursively before worktree removal.
- Release publication stays generation-based and atomic. A failed build cannot replace a valid graph, manifest, or referenced shard.

## User duration contract

The initial product contract accepts requested durations in five-second increments:

```text
5, 10, 15, ..., 300 seconds
```

The semantic target is:

```text
requestedFrames = durationSeconds * 25
```

Natural audio-end is valid. The result may be shorter than the requested maximum and must retain all completed frames. It must not discard a long generation and silently retry another seed.

The language-model context limit is 10,240 tokens. The runtime rejects a request when:

```text
promptTokens + requestedFrames > 10240
```

The fixed comparison fixture continues to use 40 prompt tokens, so five minutes uses 7,540 total tokens.

## Official acoustic chunk contract

The pinned Diffusers pipeline is the authority.

For `F` retained semantic frames:

```text
latentLength(F) = floor(F * 441 / 128)
```

Chunk planning is:

```text
one chunk when F <= 200
start_i = 100 * i otherwise
frames_i = min(200, F - start_i)
latent_i = floor(frames_i * 441 / 128)
```

Continuation state uses:

- a 172-latent overlap;
- previous latent carry from `[L - 344:L - 172)`;
- previous condition carry from the same interval;
- per-step prefix replacement
  `((1 - (1 - 1e-6) * t) * noisePrompt) + (t * previousLatent)`;
- exact previous-latent restoration over the overlap after step 30;
- decoder crops of 86 latents on each non-first chunk's left and 258 latents on each non-last chunk's right; and
- 512 waveform samples per retained latent.

For a five-minute no-end run with the 40-token fixture, the required counts are:

- 7,500 retained frames;
- 7,501 semantic decisions;
- 52,507 RVQ depth calls;
- 7,500 feedback decodes;
- 74 acoustic chunks;
- 2,220 flow steps;
- 148 mono vocoder calls;
- 13,247,488 samples per channel; and
- a 52,989,996-byte PCM16 stereo WAV.

## Maximum-window acoustic graphs

The preferred release contains one 200-frame acoustic window rather than duplicating q4 flow weights for every tail length.

### Condition encoder

The graph consumes:

- zero-padded frame hiddens `[1,200,32768]`;
- runtime nearest-neighbor indices for 689 latent positions; and
- an active-latent mask.

It returns `[1,689,2048]` with inactive positions zeroed. For every active position, the output must match a fixed-shape pinned Diffusers condition encoder for frame counts 125, 150, 175, and 200 within the existing real-weight tolerance.

### Flow step

The graph keeps the fixed maximum latent length 689 and sequence length 690. Runtime inputs include:

- latents `[1,128,689]`;
- condition `[1,689,2048]`;
- timestep and Euler delta;
- an active-latent mask;
- a compact key-attention bias that excludes inactive positions;
- overlap noise prompt `[1,128,172]`;
- previous overlap latent `[1,128,172]`; and
- an overlap-enabled scalar or mask.

The overlap replacement occurs before the transformer on every step. Inactive latent positions are zero in the returned tensor. The runtime restores the previous overlap after the final step when it downloads the completed chunk.

The maximum-window graph is accepted only if one-block and real-weight diagnostic comparisons prove that active outputs match the existing fixed-shape graph at latent lengths 430, 516, 602, and 689. If that equivalence fails, the only fallback is four fixed tail graphs sharing the same external q4 weight shards. Duplicating the 1.263 GB flow weights is forbidden.

### Mono vocoder

The vocoder exports the shared convolutional backbone with a symbolic temporal axis:

```text
float16 [1,64,L] -> float32 [1,1,512L], 1 <= L <= 689
```

The runtime feeds the exact active latent length and runs left and right channel inputs sequentially through the same session. There are no cross-batch operations in the source graph, so sequential mono execution must match the existing fixed-length stereo-as-batch-two wrapper within the current vocoder tolerance.

A zero-padded static maximum input is forbidden. Vocoder convolutions contain biases and non-causal receptive fields, so the padded tail becomes nonzero inside the network and changes active samples. The dynamic graph preserves fixed-length source math for every tail while sharing one weight set. The largest expected full-window activation remains 67,731,456 bytes, below the 128 MiB binding limit.

## Runtime memory and lifecycle

Autoregressive generation runs once for the requested maximum and is released before acoustic sessions are created.

- Retained frame hiddens are written directly into one flat FP16 buffer. The runtime must not first keep per-frame arrays and then allocate a second full copy.
- Condition and flow sessions are reused across acoustic chunks.
- Completed flow latents are downloaded once per chunk. Intermediate Euler latents stay on the GPU.
- One mono vocoder session is reused for both channels and all chunks.
- PCM16 samples are written directly into the final WAV buffer. The runtime does not first concatenate a full float32 stereo song.
- Stage resources are disposed in `finally` paths.

At five minutes, the exact autoregressive old-plus-new KV payload is 4,447,010,816 bytes. The pre-measurement total-system projection of about 10,982 MiB was low. The real capacity diagnostic reached 13,892 MiB total from a 3,168 MiB baseline, a 10,724 MiB increase.

The 300-second increase remained below the 12,288 MiB budget, so the sequential batch-one CFG fallback was not triggered. No speculative memory profile is added.

## Progress and cancellation

Progress remains event-driven and truthful.

- Artifact progress reports aggregate completed and total bytes, current file, and cache reuse.
- Session creation remains indeterminate.
- AR progress reports retained frames over the requested maximum, elapsed time, rolling frames per second, and ETA only after at least three samples.
- Acoustic progress reports chunk index and total chunks.
- Flow progress reports completed steps over `chunks * 30`, rolling step time, elapsed time, and ETA.
- Vocoder progress reports completed channel runs over `chunks * 2`.
- WAV assembly and completion report byte count and total elapsed time.
- Cancellation terminates the worker during AR or acoustic work and revokes incomplete result state.

The detailed completion status must not be immediately overwritten by a generic status line.

## Hardware guidance

Documentation distinguishes measurement from recommendation:

- 8 GiB: not recommended for the variable runtime; the short variable gate reached 8,502 MiB total.
- 10 GiB: unverified short-duration target.
- 12 GiB: unverified recommendation for the qualified product workload through one minute.
- 16 GiB: tested recommendation for the complete five-minute capacity workload.
- Certified minimum: unknown until a lower-memory adapter is tested.

No high-memory speed tier is published. Existing experiments showed no benefit.

Before a high-VRAM browser run, preflight checks look for Android emulators, `qemu`, `vmmem`, `vmware-vmx`, or another materially allocating GPU process. The run waits without terminating user processes when that occupancy would invalidate the measurement.

## Fixed-input cloud comparison

### Reproduction contract

The first comparison case freezes:

- model `MiniMaxAI/MiniMax-Music3` at revision `fbdf52fbaaca799592917417eb05f1899f1255ec`;
- Diffusers commit `3681e65996b4d2589219720101a6acbfd25073f8`;
- prompt `Global\nbpm is 96\nWarm female vocal`;
- lyrics `[verse]\nHello\n[chorus]\nStay\nTogether\n[bridge][solo]`;
- seed 7;
- ten requested seconds, with the actual retained frames and termination recorded from the WebGPU run;
- Global guidance 1.5, top-k 50;
- flow guidance 1.7 and 30 Euler steps; and
- exact browser, ORT, model, Diffusers, manifest, and WAV receipt metadata.

The original BF16 model requires CUDA, so this project does not install a local CUDA reference environment or run the original model. The WebGPU run writes a contained comparison receipt with every user-facing input and internal parameter. The user submits the same prompt, lyrics, requested duration, and seed to a cloud service when those controls are available, then returns the cloud WAV for comparison.

### Comparison limits

The browser and cloud service may differ in categorical RNG, Gaussian RNG, top-k tie handling, arithmetic precision, generator stream consumption, hidden defaults, and model revision. A cloud service may not expose seed or advanced sampler controls. Therefore the two outputs are structural and qualitative audio comparisons, not tensor parity or an isolated quantization measurement.

The browser sampler is corrected to match the source contract where practical:

- float32 guidance and probability arithmetic;
- threshold top-k that retains all values tied at the kth threshold; and
- no hidden seed retry in comparison mode.

It does not reimplement PyTorch RNG or claim that the cloud service reproduces the same latent trace.

### Comparison receipt

The WebGPU run records:

- user-facing prompt and lyrics;
- assembled prompt text and token rows when available;
- seed, requested duration, actual retained frames, and termination;
- Global guidance, top-k, temperature, flow guidance, and Euler-step contract;
- model, Diffusers, browser, ORT, and release-manifest revisions;
- WAV sample rate, channels, samples per channel, bytes, and SHA-256; and
- the exact input and output file hashes needed to identify the case.

The cloud input sheet identifies which fields can be entered directly and which internal fields are recorded only for audit. The returned cloud WAV is attached to the same case for structural checks and one manual listening comparison.

Comparison cases live below `artifacts/reference/<case-id>/`. Git may track a small parameter template, but generated WAV files and receipts remain artifact data.

### What remains measurable locally

The existing local component oracles still validate non-q4 condition and vocoder math and one-layer flow behavior. WebGPU output is also required to be deterministic for one fixed case and to pass the structural audio checks.

The cloud result cannot cleanly separate quantization from provider, revision, or sampling differences. Documentation must state this limitation and must not freeze a BF16-to-q4 numeric tolerance from the cloud audio pair.

## Staged acceptance

### Gate A: graph equivalence

- Maximum condition outputs match fixed source shapes for 125, 150, 175, and 200 frames.
- Maximum flow active outputs match fixed graphs for latent lengths 430, 516, 602, and 689.
- Dynamic sequential mono vocoder matches the existing fixed-length stereo wrapper for latent lengths 430, 516, 602, and 689.
- Every initializer and expected activation binding is at most 128 MiB.
- Fallback-disabled headed Chrome creates and executes the maximum condition and flow graphs plus the dynamic mono vocoder graph.

### Gate B: six seconds

- 150 retained frames, 516 latents, and 264,192 samples per channel.
- Exact 1,056,812-byte PCM16 stereo WAV.
- Finite, varying, non-constant-tail audio.
- Warm artifact fetches equal zero.
- Project increase above the spawn-adjacent baseline at or below 12,288 MiB.

### Gate C: ten seconds

- Chunk starts `[0,100]`, frame lengths `[200,150]`, latent lengths `[689,516]`.
- One exact 172-latent continuation overlap.
- 60 flow steps and four mono vocoder runs.
- 440,832 samples per channel and a 1,763,372-byte WAV.
- Pinned Python and TypeScript chunk plans, carry, blend, restoration, crops, and stitching agree exactly.
- Browser audio passes finite, stereo, no-constant-tail, and decode checks.

### Gate D: longer durations

Run headed persistent-profile diagnostics at 30, 60, 120, and 300 seconds. Stop on:

- project GPU-memory increase above 12,288 MiB;
- raw GPU use within 512 MiB of physical adapter capacity;
- device loss;
- worker or renderer termination;
- CPU execution-provider fallback;
- non-finite output;
- missing patched JSPI WASM; or
- unexpected completed-artifact fetch.

Late-AR and multi-chunk-flow cancellation are duration-independent and were exercised in the six and ten-second headed gate before long qualification.

### Gate E: cloud comparison handoff

- The fixed WebGPU prompt, lyrics, seed, duration, sampler parameters, provenance, manifest hash, and WAV receipt are exact.
- Two WebGPU runs with the fixed case are deterministic and produce structurally healthy audio.
- The cloud input sheet lists every directly enterable value and identifies unavailable controls.
- A returned cloud WAV can be attached and verified without changing the WebGPU receipt.
- Comparison reporting is limited to structure and manual listening. It does not claim tensor parity or isolated q4/provider tolerances.

## Completion rule

This goal is complete only when the implemented duration contract, staged browser evidence, hardware documentation, progress interaction, artifact policy, and fixed-input comparison workflow all have current authoritative evidence. The final report must distinguish the highest native max-frame product result from the longest capacity-only workload and must not turn forced continuation into a native five-minute claim.
