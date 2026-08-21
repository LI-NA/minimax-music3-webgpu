# Cloud reference comparison

CUDA is required to run the original MiniMax Music 3 model, so this repository does not provide a local original-model execution path. The current workflow packages one frozen WebGPU case for comparison with a song generated separately by a cloud service. The tool uses only the Python standard library. It does not import Torch or Diffusers, load weights, access a model cache, or download anything.

Full tensor parity, forced sampled-code replay, Gaussian-noise replay, and a real first-transition fixture are deferred. No first-transition fixture has been produced or inferred from synthetic data.

## Captured WebGPU case

The fixed case was generated twice in headed Chrome from the same warm profile. Both runs reached 250 retained frames, produced canonical 1,763,372-byte WAV files, and were bit-identical with SHA-256 `bcbbf15a5e43f45c7c6713b06432a458a012fc23bc0fe63badfb10ef8c7905f8`. This capture belongs to the archived pre-programmatic manifest, not the current active release.

- Capture: `artifacts/reference/captures/webgpu-fixed-10s-20260821-02`
- Verified receipt: `artifacts/reference/cases/webgpu-fixed-10s-20260821-02/receipt.json`
- Receipt SHA-256: `704493099f2defd1d65f4178457645aa1e74e7d93fc7ff0dfd42a302a2783f50`
- Manifest SHA-256: `12fe28c91c2474f6a4a6ed02fe6f087c0c3b310cbc421138757dacf9072a6e0a`
- Archived release: `artifacts/archive/music-variable-pre-programmatic-20260822-0208`
- Chrome: 151.0.0.0
- ONNX Runtime Web: `1.30.0-dev.20260813-72e1c9c9b8`
- Application revision: `d4c4a2b1a2fd360e42808d008b7005dab66c19ea519b0fbded92e264fbe587b6`

The WebGPU side of the handoff is complete. The next comparison step is for the user to enter the values below in the selected cloud service, record unavailable controls, and return the unconverted cloud WAV. Until that file exists, no original-versus-WebGPU quality or structure conclusion is claimed.

## Frozen comparison case

Enter these values in the cloud service when the controls are available:

- Prompt, with actual line breaks:

  ```text
  Global
  bpm is 96
  Warm female vocal
  ```

- Lyrics:

  ```text
  [verse]
  Hello
  [chorus]
  Stay
  Together
  [bridge][solo]
  ```

- Seed: 7.
- Requested duration: 10 seconds.
- Global guidance: 1.5.
- Semantic top-k: 50.
- Residual top-k: 50.
- Temperature: 1.0.
- Flow guidance: 1.7.
- Flow steps: 30.
- Sampler revision: `mulberry32-guided-threshold-topk-f32-box-muller-fp16-v1`.
- Flow schedule revision: `sha256:b1c658b8efb8382eac3a9d4115d1566ad11abd8a91b223c48d73699a0140bd92`.
- Model revision: `fbdf52fbaaca799592917417eb05f1899f1255ec`.
- Diffusers revision: `3681e65996b4d2589219720101a6acbfd25073f8`.

The exact float32 timestep and dt bit arrays are stored once in `tools/reference/fixed_case.json`. The Python tool validates the fixture hash when it starts.

Many hosted interfaces do not expose the seed, guidance, top-k, temperature, scheduler revision, or exact model revision. Record which controls were unavailable. A cloud result without those controls remains useful for structural and manual listening comparison, but it is not deterministic parity evidence.

Download the returned cloud WAV without converting it. Keep it with a note containing the cloud service name, generation time, displayed model name, and every unavailable advanced control. This cloud WAV can later be attached beside the WebGPU receipt and WebGPU WAV for duration, structure, and manual audio comparison.

## WebGPU inputs

The tool accepts a Task 5 variable-duration manifest containing the exact `acoustic` constants, the worker's flat `comparison` JSON, and the canonical WebGPU WAV. Save `result.comparison` directly as the metrics file. No field renaming or flattening step is required. The worker emits this object only for the fixed ten-second, seed 7 request. The frozen max-frame case uses 250 retained frames. A natural end must be explicit and may retain 1 through 249 frames.

```json
{
  "prompt": "Global\nbpm is 96\nWarm female vocal",
  "lyrics": "[verse]\nHello\n[chorus]\nStay\nTogether\n[bridge][solo]",
  "assembledPrompt": "<|im_start|><|caption_start|>Global\nbpm is 96\nWarm female vocal<|caption_end|><|lyrics_start|>[start]\n[verse]\nHello\n[chorus]\nStay\nTogether\n[bridge][solo]<|lyrics_end|><|im_end|><|audio_start|>",
  "tokenIds": [
    [151644, 151671, 11646, 198, 65, 5187, 374, 220, 24, 21, 198, 95275, 8778, 25407, 151672, 151673, 28463, 921, 58, 4450, 921, 9707, 198, 58, 6150, 355, 921, 38102, 198, 80987, 198, 58, 13709, 1457, 82, 10011, 60, 151674, 151645, 151669],
    [151644, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151654, 151645, 151669]
  ],
  "seed": 7,
  "durationSeconds": 10,
  "retainedFrames": 250,
  "termination": "max-frames",
  "globalGuidance": 1.5,
  "semanticTopK": 50,
  "residualTopK": 50,
  "temperature": 1.0,
  "samplerRevision": "mulberry32-guided-threshold-topk-f32-box-muller-fp16-v1",
  "flowGuidance": 1.7,
  "flowSteps": 30,
  "flowScheduleRevision": "sha256:b1c658b8efb8382eac3a9d4115d1566ad11abd8a91b223c48d73699a0140bd92",
  "manifestHash": "<64 lowercase hexadecimal characters from this run>",
  "browser": "Mozilla/5.0 ... Chrome/140.0.7339.81 ...",
  "ortVersion": "1.30.0-dev.20260813-72e1c9c9b8",
  "appVersion": "0.1.0-experimental",
  "appRevision": "<64 lowercase hexadecimal working-source fingerprint>"
}
```

All 21 root fields are required. The exact field list and static comparison values are pinned in `tools/reference/fixed_case.json`, while the token rows come from `tests/fixtures/prompt-contract.json`. The receipt always records the fixed prompt token count of 40, and the duration planner always receives that value. The worker's exact `navigator.userAgent` string is the `browser` input. It must contain a four-part Chrome version, which the receipt extracts into structured browser provenance. ONNX Runtime Web must match the pinned package version. `appRevision` is a deterministic SHA-256 of `index.html`, every TypeScript, TSX, and CSS file under `src`, the package and lock files, Vite configuration, the revision tool, and the fixed prompt, schedule, and comparison fixtures. It is not a Git revision, so it remains accurate with a dirty read-only working tree. `manifestHash` must exactly match the SHA-256 computed from the supplied manifest file. A missing or stale worker hash stops publication.

The WAV must be canonical PCM16 stereo at 44,100 Hz with the standard 44-byte header and no extra or truncated bytes. The tool calls the project Python duration planner with the actual retained frame count, then requires the exact planned samples and byte length. For 250 retained frames this is 440,832 samples per channel and 1,763,372 total bytes.

## Build and verify

Build a contained atomic receipt only after an explicit WebGPU generation:

```powershell
.\.venv\Scripts\python.exe tools/reference/reference_case.py build `
  --manifest path/to/variable-duration-manifest.json `
  --metrics path/to/exported-result-comparison.json `
  --wav path/to/webgpu.wav `
  --output-root path/to/comparison-cases `
  --case-id webgpu-ten-seconds
```

The output case contains only `receipt.json`. Publication uses a private staging directory and an atomic rename. Existing cases are never replaced, case IDs cannot escape the selected output root, and inputs or receipts cannot be symlinks.

The receipt records the exact input and token rows, retained-frame and termination result, sampler contract, model and runtime provenance, SHA-256 hashes for all three attachments, and canonical WAV metadata.

Verify the receipt whenever attaching or moving its source files:

```powershell
.\.venv\Scripts\python.exe tools/reference/reference_case.py verify `
  --case path/to/comparison-cases/webgpu-ten-seconds `
  --manifest path/to/variable-duration-manifest.json `
  --metrics path/to/exported-result-comparison.json `
  --wav path/to/webgpu.wav
```

Later comparison can inspect both WAV structures and perform a manual listening comparison. Full internal tensor parity remains deferred.
