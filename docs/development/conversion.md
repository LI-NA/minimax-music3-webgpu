# Global LLM conversion

The Global LLM source is the official `MiniMaxAI/MiniMax-Music3` checkpoint at revision `fbdf52fbaaca799592917417eb05f1899f1255ec`.

```powershell
uv run music3-convert download-global --artifacts-dir artifacts
uv run music3-convert build-global --artifacts-dir artifacts --layers 1
uv run music3-convert build-global --artifacts-dir artifacts --layers 36
```

The downloader intentionally selects only `LICENSE`, `modular_model_index.json`, `language_model/*`, and `tokenizer/*`. It does not download the legacy Qwen checkpoint or acoustic-stage weights.

Run a headed browser gate against the selected release with:

```powershell
$env:MINIMAX_RELEASE = 'global-one-layer' # or global
npx playwright test tests/browser/global-decoder.spec.ts --project=chrome
```

The real-model test launches branded Chrome with a persistent profile at `artifacts/browser-profile`. This keeps multi-gigabyte OPFS artifacts across test processes and exercises download resume and cache reuse. Set `MINIMAX_CHROME_PROFILE` to an alternate ignored profile path when isolation is required.

`artifacts/` is excluded from Git and Kopia. The release manifest carries file hashes, the exact source revision, WebGPU requirements, and the artifacts used by the browser cache.
