# Global LLM conversion

The Global LLM source is the official `MiniMaxAI/MiniMax-Music3` checkpoint at revision `fbdf52fbaaca799592917417eb05f1899f1255ec`.

The active conversion contract is symmetric Q4 with block size 32 and accuracy level 4. Global work directories include the `q4-b32` precision profile. The variable-duration Flow graph uses the same Q4 contract except that `time_proj.weight` and `proj_out.weight` remain FP16 `MatMul` weights.

```powershell
uv run music3-convert download-global --artifacts-dir artifacts
uv run music3-convert build-global --artifacts-dir artifacts --layers 1
uv run music3-convert build-global --artifacts-dir artifacts --layers 36
uv run music3-convert build-music-variable --artifacts-dir artifacts
```

On a host where the 36-layer ORT GenAI builder exceeds the system commit limit, `requantize_global_decoder_from_template` can stream the official BF16 weights into a preserved B128 graph topology. It quantizes one source matrix at a time and writes bounded B32 external-data shards without retaining the complete quantized model in memory. The template supplies topology only. All quantized values are regenerated from the pinned official source weights, which are converted from BF16 to FP16 before quantization exactly like the Flow converter, so the result is not bit-identical to the full ORT GenAI builder output. The resulting graph must pass `validate_global_decoder` before release emission.

The downloader intentionally selects only `LICENSE`, `modular_model_index.json`, `language_model/*`, and `tokenizer/*`. It does not download the legacy Qwen checkpoint or acoustic-stage weights.

Run a headed browser gate against the selected release with:

```powershell
npx playwright test tests/browser/global-decoder.spec.ts --project=chrome
```

The gate opens `/diagnostics.html?release=global`. Change the `release` query to `global-one-layer`
to exercise the reduced conversion. The dev server serves `artifacts/release/<release>` same-origin,
so no separate artifact server or environment variable selects the release.

The real-model test launches branded Chrome with a persistent profile at `artifacts/browser-profile`. This keeps multi-gigabyte OPFS artifacts across test processes and exercises download resume and cache reuse. Set `MINIMAX_CHROME_PROFILE` to an alternate ignored profile path when isolation is required.

`artifacts/` is excluded from Git and Kopia. The release manifest carries file hashes, the exact source revision, WebGPU requirements, and the artifacts used by the browser cache.

Before replacing a release, the converter preserves the prior release and receipt under `artifacts/archive/`. The B128 Global work directories are separately retained under `artifacts/archive/b128-work/` with a file-by-file SHA-256 receipt.
