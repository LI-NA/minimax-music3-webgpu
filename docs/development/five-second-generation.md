# Five-second browser generation gate

Run the exact combined release in headed branded Chrome with one worker and a persistent profile:

```powershell
$env:MINIMAX_RELEASE = 'music-5s'
$env:MINIMAX_MUSIC_CHROME_PROFILE = 'artifacts/music-browser-profile'
npx playwright test tests/browser/music-generation.spec.ts --project=chrome --workers=1
```

The worker runs the stages sequentially. Autoregressive and RVQ sessions share one device, retain 125 frames as 8,192,000 bytes of raw FP16, then release every session and destroy that device. Condition uses a new device and returns 1,761,280 raw FP16 bytes. Flow uses another new device, deterministic standard-normal seed 7 latents, and 30 GPU steps before returning 110,080 raw FP16 bytes. Vocoder uses a final new device and produces the WAV before its session and device are released. Only the final WAV `ArrayBuffer` crosses back to the UI. No GPU tensor crosses a stage boundary.

## Measured result

The exact gate passed on 2026-08-21 with Chrome 151.0.7922.170 and ONNX Runtime Web `1.30.0-dev.20260813-72e1c9c9b8`.

- Seed attempts: `7` only
- Referenced artifact bytes: 8,083,469,618
- First persistent-profile gate: passed in 8.3 minutes, including initial OPFS fill and two generations
- Pre-optimization warm metric gate: passed in 4.1 minutes, including two generations and repeated full-cache hashing
- Post-optimization warm benchmark: three generations completed in 35.5, 28.2, and 28.1 seconds of browser wall time
- Final progress-aware combined gate: passed in 1.5 minutes, including two zero-fetch generations
- Completed-artifact fetches on warm runs: 0
- Download: `artifacts/generated/minimax-music3-5s.wav`
- WAV: 880,684 bytes, PCM16, 44,100 Hz, 2 channels, 220,160 samples per channel
- Corrected WAV SHA-256: `c16731b5597fcf343fc1cf05c842ea559c5406bf95678bff75c88767e4102800`
- Browser decoding: `AudioContext.decodeAudioData` returned 44,100 Hz, 2 channels, and 220,160 samples

## Audio correctness regression

The first generated file was structurally valid but not usable audio. Its two channels were identical, and samples 65,563 through 220,117 were the exact PCM value `+100`. Two independent runtime defects caused this result:

1. Condition and flow stage outputs were copied from GPU buffers before ONNX Runtime flushed its internal command encoder. The product handoff received zero FP16 values even though ONNX Runtime's downloader returned completed nonzero tensors.
2. The pinned JSPI WebGPU `ConvTranspose` shader used FP16 for spatial coordinates. Large vocoder coordinates rounded incorrectly and then overflowed above 65,504, making most of the output bias-only.

GPU FP16 stage handoffs now use `Tensor.getData()` as the ONNX Runtime flush and download boundary while preserving native Float16 storage as raw `Uint16Array` bits. Vite also serves a fail-closed patch of the pinned JSPI WASM that changes only the eight embedded `ConvTranspose` coordinate casts to FP32. The source size and SHA-256, match count, and patched SHA-256 are all fixed contracts. Runtime URLs include the patch hash prefix so a browser cannot reuse the old fixed-name WASM from cache.

The corrected headed Chrome gate passed twice from the persistent profile. The saved WAV has a longest constant stereo-frame run of 2 samples, 220,047 of 220,160 frames differ between channels, and the final second has nonzero variation with normalized RMS `0.2225`. The test now rejects a one-second constant run, a non-varying final second, and identical stereo channels. These are narrow regressions for the observed defect, not general music-quality scoring.

During this corrected gate, total-system dedicated GPU memory rose from 2,138 MiB to an observed maximum of 7,269 MiB. This remains below the 12,288 MiB acceptance limit. The earlier clean three-run measurements remain the basis for hardware guidance because the background baseline differed.

The post-optimization three-run mean timings were:

| Stage | Session creation | Inference or generation | End-to-end stage |
| --- | ---: | ---: | ---: |
| Autoregressive and RVQ | 4,542.2 ms | 15,202.9 ms | 19,817.4 ms |
| Condition | 62.4 ms | 19.3 ms | 252.2 ms |
| Flow | 1,448.1 ms | 5,799.5 ms | 7,331.2 ms |
| Vocoder and WAV | 207.8 ms | 161.9 ms | 527.1 ms |

Autoregressive inference retained 125 frames at about 8.22 frames per second. The worker averaged 29,414.4 ms from manifest through result.

Before the cache optimization, each warm run read and hashed all 8.08 GB again. The three measured cache phases averaged 76,282.1 ms. Completed artifacts now use the manifest-scoped verification receipt written after their first successful SHA-256 check. The post-optimization phases took 238.4, 214.9, and 217.1 ms with zero fetches, reducing mean browser wall time from 104,956 to 30,587 ms without changing model execution.

## GPU memory

Total-system dedicated GPU memory was sampled through `nvidia-smi` across the post-optimization three-run benchmark. The monitor requested 100 ms samples and recorded 700 samples at an effective 180.64 ms cadence.

- Baseline: 1,704 MiB
- Observed total-system peak: 6,834 MiB
- Observed increase from baseline: 5,130 MiB
- Final after Chrome closed: 1,704 MiB
- Limit: 12,288 MiB

The combined autoregressive segment reached the 6,834 MiB peak. A later condition/flow segment reached about 3,323 to 3,332 MiB, but its internal boundary was not externally labelled, so separate combined condition and flow peaks are not claimed. The combined vocoder peak was also not separately attributable. Standalone measurements provide supporting context only: condition was 1,947 to 2,188 MiB, flow was 1,695 to 3,252 MiB, and vocoder was 1,695 to 2,751 MiB. These are total-system samples, not isolated process allocations.

ONNX Runtime emitted non-fatal constant-folding warnings for vocoder Snake reciprocal nodes because no CPU kernel was available. The WebGPU-only vocoder still completed with CPU fallback disabled.

VRAM requirements, rejected runtime experiments, the retained cache optimization, and progress behavior are recorded in [WebGPU runtime requirements and optimization](webgpu-runtime-requirements.md).

This gate validates artifact reuse, stage cleanup, exact dimensions, WAV structure, browser decodability, and the narrow constant-tail regression above. It does not compare against a reference song, run an audio classifier, or perform the deferred listening check.
