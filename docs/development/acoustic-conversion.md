# Five-second acoustic release assembly

The browser generation gate uses one combined release at `artifacts/release/music-5s`. Build it only from the exact standalone `global`, `rvq`, `condition`, `flow`, and `vocoder` releases:

```powershell
uv run music3-convert build-music-5s --artifacts-dir artifacts
```

The assembler reads and validates every standalone manifest, copies referenced files into a fresh staging directory under a release-specific prefix, rehashes the staged files, then atomically promotes the staging directory. An existing combined release remains in place if validation fails. Diagnostic releases such as `global-one-layer` and `rvq-pre-inline` are not inputs.

The fixed source contract is:

- Model: `MiniMaxAI/MiniMax-Music3`
- Model revision: `fbdf52fbaaca799592917417eb05f1899f1255ec`
- Diffusers revision: `3681e65996b4d2589219720101a6acbfd25073f8`
- Global and flow quantization: symmetric q4, block size 128, accuracy level 4
- Slice: 125 retained semantic frames, latent length 430, 30 flow steps, 220,160 stereo samples at 44,100 Hz
- Guidance: Global 1.5, flow 1.7
- WebGPU: `shader-f16`, CPU execution-provider fallback disabled at session creation

The assembled release contains 80 referenced artifacts totaling 8,083,469,618 bytes. Including `manifest.json`, the directory contains 81 files totaling 8,083,502,086 bytes. Its largest artifact is exactly 134,217,728 bytes. The measured manifest SHA-256 is `5c295ebfb4b7849d317cf0abd3dd8bfc9da3b58dc74de12a3523c07f28d4500e`.

Each combined manifest path is prefixed by its standalone stage while each ONNX external-data `onnxLocation` remains unchanged. Tokenizer files, the license, Global KV names, both embedding tables, graph GPU-output contracts, source revisions, file sizes, and SHA-256 values are validated before promotion.

All source weights, standalone releases, combined artifacts, browser profiles, and generated audio remain under `artifacts/`.
