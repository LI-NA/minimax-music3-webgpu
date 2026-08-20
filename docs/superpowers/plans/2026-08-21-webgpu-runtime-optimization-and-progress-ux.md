# WebGPU Runtime Optimization and Progress UX Plan

**Status:** Complete on 2026-08-21. Measured results and the shipped configuration are recorded in `docs/development/webgpu-runtime-requirements.md`.

**Goal:** Make the exact MiniMax-Music3 WebGPU pipeline practical before building the final UI. Establish a defensible minimum and recommended VRAM requirement, reduce avoidable device memory, use measured headroom for safe speed improvements, and expose useful progress without expanding the product scope.

## Entry conditions

- The exact five-second pipeline produces a non-empty 44.1 kHz stereo WAV in branded Chrome with CPU fallback disabled.
- Every stage has a reproducible standalone browser gate and a combined gate.
- The current ORT version, Chrome version, artifact hashes, stage timings, and 100 ms `nvidia-smi` samples are recorded.
- The manual listening check remains deferred. Container validity and browser decodability are sufficient at this point.

## Measurement hygiene

Before every memory or performance comparison:

1. Record all GPU processes and total dedicated memory with `nvidia-smi`.
2. If an Android VM or another unrelated process holds at least 512 MiB of dedicated GPU memory, wait for it to exit instead of treating the contaminated result as a baseline. Do not terminate the user's process automatically.
3. Use one headed branded Chrome worker, the same persistent profile state, the same artifacts, prompt fixture, seed, and five-second contract.
4. Report total observed device memory, baseline, delta, and sampling interval. Do not claim an isolated component peak from total-system telemetry.
5. Change one variable at a time and keep it only when the same gate demonstrates a material improvement without changing model semantics or output contracts.

## Task 1: Establish the baseline and VRAM tiers

- Measure clean-stage and combined peaks for autoregressive generation, condition plus flow, and vocoder.
- Record session creation time, first-run shader compilation, warm-run time, per-frame AR speed, each of 30 flow steps, vocoder time, artifact download bytes, and OPFS cache reuse.
- Calculate owned GPU tensor bytes where the runtime can do so exactly, while keeping `nvidia-smi` as the acceptance measurement.
- Define three values from evidence:
  - measured minimum for the fixed five-second gate;
  - recommended VRAM with a practical safety reserve for Windows and Chromium;
  - the existing 12 GiB acceptance ceiling.
- Do not infer a low-end requirement by subtracting unrelated process memory. Run a clean measurement.

## Task 2: Low-memory profile

Start with the existing staged session lifetime and explicit tensor disposal. Test only the smallest plausible changes:

1. Verify that autoregressive sessions and GPU tensors are released before condition and flow, and that flow is released before vocoder.
2. Compare the current ORT buffer-cache and memory-pattern settings with one conservative lower-memory setting per fixed graph.
3. If the measured flow peak is the limiting stage, compare batch-two CFG with two sequential batch-one passes only once. Keep the split only if it materially lowers the required VRAM and still completes at a usable speed.
4. Check for accidental CPU and GPU copies at stage handoffs. Retain only the required 8,192,000-byte FP16 frame hidden tensor and the final 110,080-byte latent handoff.

Do not add model distillation, new quantization formats, custom kernels, or graph partitioning unless a measured blocker proves the current exact q4 and FP16 layout cannot meet the requirement.

## Task 3: Headroom performance profile

If the clean baseline leaves meaningful VRAM headroom below 12 GiB:

- Compare cold and warm runs before tuning.
- Test a bounded set of ORT settings that may reuse fixed-shape buffers or compiled pipelines.
- Preserve the proven batch-two flow graph when it fits. Do not duplicate model sessions merely to consume spare VRAM.
- Consider retaining the next stage only when overlap removes measurable load time and the combined peak remains safely below 12 GiB.
- Keep an optimization only if it produces a repeatable end-to-end or dominant-stage speed improvement and all fallback-disabled gates still pass.

## Task 4: General WebGPU performance

- Use stage timing to focus on the dominant bottleneck, expected to be flow attention or vocoder ConvTranspose.
- Inspect browser and ORT diagnostics before changing graph structure.
- Allow at most three measured hypotheses for one unexplained performance problem, then reconsider the approach instead of accumulating patches.
- Custom WGSL, unsupported runtime forks, speculative operator rewrites, and architecture changes remain out of scope unless the existing path is blocked rather than merely slow.

## Task 5: Progress and interaction contract

Add a small runtime-facing progress contract before final UI work:

- artifact download: completed bytes, total bytes, current file, and cache hit state;
- session loading: named stage and indeterminate activity when ORT exposes no granular percentage;
- autoregressive generation: retained frames out of 125, rolling frames per second, elapsed time, and estimated remaining time after enough samples exist;
- condition: named single step;
- flow: completed steps out of 30, rolling step time, elapsed time, and estimated remaining time;
- vocoder and WAV: named single steps;
- completion: WAV byte length and total elapsed time;
- cancellation: terminate the inference Worker and report a cancelled state.

ETA must be optional and must not display until a stable estimate is available. Do not simulate percentages for opaque ORT session creation. Unit-test progress ordering, monotonic counters, ETA omission, cancellation, and final completion. The follow-up UI only needs to prove that these events can drive clear status text, a progress indicator for countable stages, and cancellation.

## Acceptance and handoff

- A low-memory configuration and a faster configuration are documented only if measurements justify both. Otherwise ship one recommended configuration.
- The minimum and recommended VRAM figures cite clean measured peaks, reserve assumptions, Chrome and ORT versions, and the five-second fixed workload.
- The combined fallback-disabled Chrome gate still emits a structurally valid WAV.
- Relevant Python, unit, lint, typecheck, build, and browser gates pass.
- The final diff contains no speculative optimization that lacks a measured benefit.
- Generate one fresh WAV after optimization. Then ask the user to listen once and confirm whether vocals and accompaniment sound normal.

## Completion summary

- One recommended configuration was retained. Separate memory and performance profiles were rejected by measurements.
- The measured total-system peak was 6,835 MiB before optimization and 6,834 MiB after it. The recommendation is 10 GiB or more, while 8 GiB remains an unverified compatibility target.
- Exact-size cache mode raised the peak to 12,074 MiB and was reverted.
- Decoder prepacking did not improve AR inference or stage time and was reverted.
- Manifest-scoped completed-artifact receipts reduced warm cache validation from 76,282.1 to 223.5 ms on average and were retained.
- The progress and cancellation contract was implemented without inventing percentages for opaque session creation.
- The final listening check remains deferred until the user returns.
