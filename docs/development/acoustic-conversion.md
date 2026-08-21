# Acoustic release assembly

## Variable-duration release

The current browser path uses `artifacts/release/music-variable`:

```powershell
uv run music3-convert build-music-variable --artifacts-dir artifacts
```

The command verifies the pinned source receipt, disk space, exclusive build lock, target-profile lock, repository or target-profile Chrome conflicts, and unrelated converter processes before staging. Unrelated Chrome processes are allowed. A fresh release builds the maximum condition and flow graphs plus the symbolic-length mono vocoder inside one private generation. When a verified prior release has the current exporter fingerprint and exact runtime contract, its condition, flow, and vocoder graphs are full-checked at the prior location, copied into staging, checked again there, and their exporters are skipped. Global and RVQ artifacts remain SHA-verified hardlink reuse. A one-time legacy flow upgrade must still run the existing coupled flow graph and shard exporter, after which matching prior shards are SHA-reused through hardlinks when supported. The command validates every ONNX contract and external range, then atomically promotes the complete release and receipt. A failed build preserves the previous generation or its recovery backup.

Every successful replacement archives the verified prior release tree and its coherent receipt under `artifacts/archive/music-variable/<generation>/`. These recovery archives are never deleted automatically.

The published release contains 80 referenced artifacts totaling 8,083,501,198 bytes. Including `manifest.json`, the directory contains 81 files totaling 8,083,535,909 bytes. The manifest SHA-256 is `730293c66360cc9a413446311d2fd7957b547423d38bf7f81b80d2d331f96232`, and the release receipt SHA-256 is `347855db3edc63b330d61b6da492295faaed6fb571d02d57e5f8b3d347320470`.

This release supports the official acoustic windows up to 200 semantic frames and 689 latents. Flow guidance is a runtime FP16 graph input, while flow step count is controlled by the runtime schedule. The mono vocoder temporal axis is symbolic because a static zero-padded tail changes the source network's active output through biased non-causal convolutions.

## Legacy five-second release

The browser generation gate uses one combined release at `artifacts/release/music-5s`. Build it only from the exact standalone `global`, `rvq`, `condition`, `flow`, and `vocoder` releases:

```powershell
uv run music3-convert build-music-5s --artifacts-dir artifacts
```

The assembler reads and validates every standalone manifest, copies referenced files into a fresh staging directory under a release-specific prefix, rehashes the staged files, then atomically promotes the staging directory. An existing combined release remains in place if validation fails. Diagnostic releases such as `global-one-layer` and `rvq-pre-inline` are not inputs.

Every successful replacement first preserves and verifies the prior release under a unique `artifacts/archive/music-5s/<generation>/release` directory. This legacy release has no receipt, and its archives are never deleted automatically.

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
