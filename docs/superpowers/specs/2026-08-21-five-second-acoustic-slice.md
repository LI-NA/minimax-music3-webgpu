# Five-Second Acoustic Vertical Slice

**Status:** Approved implementation scope

**Builds on:** `docs/superpowers/specs/2026-08-20-minimax-music3-webgpu-design.md`

## Outcome

The next milestone produces one real 44.1 kHz stereo WAV in desktop Chromium from the exact `MiniMaxAI/MiniMax-Music3` checkpoint. The clip is fixed at approximately five seconds. It must contain the model's generated vocals and accompaniment, not a placeholder tone or a substituted model.

Success is a downloadable WAV that the user can play and judge. Automated audio-quality analysis is out of scope.

## Fixed model contract

- Checkpoint revision: `fbdf52fbaaca799592917417eb05f1899f1255ec`
- Diffusers reference revision: `3681e65996b4d2589219720101a6acbfd25073f8`
- Semantic frames: 125 at 25 frames per second
- RVQ codebooks: one semantic code and seven residual codes per frame
- Flow latent length: 430
- Flow steps: 30 Euler updates
- Flow guidance: 1.7
- Output: one stereo tensor with 220,160 samples per channel
- Native sample rate: 44,100 Hz
- Nominal duration: 4.992 seconds

Five seconds is the first safe fixed shape. At eight seconds, the largest vocoder activation is 135,462,912 bytes and exceeds the current 128 MiB storage-buffer binding contract. At five seconds, the same activation is 84,541,440 bytes.

## Autoregressive stage

The existing q4 Global LLM, reduced head, OPFS embedding table, WebGPU device, and GPU KV chain remain authoritative. The diagnostic fixed embedding row is replaced with the official generation loop:

1. Apply classifier-free guidance 1.5 to semantic logits.
2. Sample from the 16,384 semantic rows plus the audio-end row with top-k 50.
3. Run seven RVQ depth steps with classifier-free guidance 1.5 and top-k 50.
4. Build the next Global LLM input from the semantic embedding and seven residual embeddings, divided by `sqrt(8)`.
5. Retain the Global hidden state and seven depth hidden states for each emitted frame.

The first loop transition after the audio-start token does not emit a frame. The fixed slice retains exactly 125 later frames. If the audio-end token appears before frame 125 during the first WAV gate, the runner may retry once with the next deterministic seed. Dynamic acoustic shapes are not part of this milestone.

## Acoustic stage

The condition encoder combines the eight hidden-state groups and produces FP16 condition tensor `[1,430,2048]`.

The flow transformer uses symmetric q4 `MatMulNBits` for large linear weights and FP16 for remaining weights and activations. The browser runs one fixed-length batch-two classifier-free-guidance graph for 30 Euler steps. q8 is not used because the pinned WebGPU kernel path has only been proven for q4.

The mixed-precision FP16 vocoder folds weight normalization during conversion and maps fixed latent tensor `[1,128,430]` to FP32 stereo waveform `[1,2,220160]`. Snake activations whose alpha is below the FP16 normal threshold run their original formula in FP32, then cast back to FP16. The final Tanh result is cast to FP32 once for waveform download. One chunk needs no overlap carry, crop, or stitching.

## Runtime lifetime

One inference Worker owns OPFS, ONNX Runtime, and the WebGPU device.

1. Load Global LLM, reduced head, RVQ depth, and feedback stages.
2. Generate 125 complete RVQ frames and retain their eight hidden-state groups.
3. Release autoregressive sessions, KV tensors, and embedding handles.
4. Load the condition and q4 flow sessions, run 30 updates, then release them.
5. Load the FP16 vocoder, download the final waveform, release it, and encode PCM16 WAV.

No GPU resource crosses the Worker boundary. The main thread receives progress JSON and the final transferable WAV `ArrayBuffer` only.

## Storage and hardware

All source weights, intermediate files, converted releases, Chrome profiles, and generated WAV files remain below `artifacts/`, which is excluded by both Git and Kopia.

Every physical release file remains at or below 128 MiB. All ONNX sessions use WebGPU only with CPU execution-provider fallback disabled. The measured total GPU usage must remain at or below 12 GB on the RTX 4080 development machine.

## Acceptance

- Exact checkpoint and pinned Diffusers contract only
- 125 semantic frames with all seven residual codes
- GPU-resident Global KV and feedback chain
- Finite condition and flow tensors
- One flow step and all 30 steps pass in headed Chrome
- Vocoder returns 220,160 finite samples per channel
- Browser produces a valid non-empty PCM16 stereo WAV
- Repeated execution fetches zero completed artifacts
- Measured total GPU usage stays at or below 12 GB
- User receives the WAV and is asked only to play it and report whether it sounds correct

UI polish, multiple durations, chunk overlap, automated audio analysis, custom WGSL, and broad browser support remain deferred.
