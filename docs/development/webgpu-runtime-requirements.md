# WebGPU runtime requirements and optimization

This document applies to the fixed five-second MiniMax-Music3 browser workload. It covers 125 semantic frames, latent length 430, 30 flow steps, and a 44.1 kHz stereo PCM16 WAV. Longer songs and other model revisions require separate measurements.

## Recommended configuration

Use one runtime configuration:

- Chromium with a non-fallback WebGPU adapter and `shader-f16`
- ONNX Runtime Web `1.30.0-dev.20260813-72e1c9c9b8`
- `executionProviders: ['webgpu']` with CPU fallback disabled
- sequential execution and all graph optimizations
- memory patterns and graph capture disabled
- prepacking disabled
- the default bucketed storage-buffer cache
- one stage device at a time, with every session released and the device destroyed before the next stage

Separate low-memory and faster profiles are not justified by the measurements. The default configuration used less memory and was at least as fast as both tested alternatives.

## VRAM guidance

The clean three-run baseline used headed Chrome 151.0.7922.170 on a 16 GiB RTX 4080. Total dedicated GPU memory was sampled through `nvidia-smi`, including Windows and Chromium background usage.

| Tier | Guidance | Evidence |
| --- | --- | --- |
| Measured workload footprint | 6,835 MiB total-system peak | Three baseline runs reached the same peak from a 1,705 MiB baseline. The retained optimization reached 6,834 MiB. |
| Experimental next target | 8 GiB physical VRAM | The measured total fits, but no 8 GiB adapter has been tested. Do not describe this tier as supported yet. |
| Recommended | 10 GiB or more physical VRAM | The measured 6.67 GiB peak plus at least 2 GiB of practical Windows and Chromium reserve rounds up to this tier. |
| Verified acceptance budget | 12 GiB | Every combined run remained well below the project's 12,288 MiB limit. |

The current evidence supports a 10 GiB recommendation, not a certified hardware minimum. An 8 GiB device is the next useful compatibility test. Before generation, close or wait for Android emulators and other applications that materially consume GPU memory.

The WebGPU adapter must also expose a 128 MiB storage-buffer binding and at least nine storage buffers per shader stage. The combined release stores 8,083,469,618 referenced artifact bytes in the browser profile.

## Memory ownership

Autoregressive generation is the limiting stage. Exact application-owned GPU tensor payloads are small relative to ONNX Runtime weights, activations, and buffer buckets:

| Stage | Exact peak application-owned GPU bytes |
| --- | ---: |
| Autoregressive and RVQ | 97,058,816 |
| Condition output and readback staging | 3,522,560 |
| Flow current tensor and readback staging | 220,160 |
| Vocoder | 0 |

The autoregressive figure includes the old and new KV sets at the final decode boundary. Final live KV data is 48,660,480 bytes. Stage handoffs remain on CPU as raw FP16: 8,192,000 frame-hidden bytes, 1,761,280 condition bytes, and 110,080 latent bytes. No GPU tensor crosses a device boundary.

Standalone total-system measurements provide supporting context only. Condition rose from 1,947 to 2,188 MiB, flow from 1,695 to 3,252 MiB, and vocoder from 1,695 to 2,751 MiB. The combined autoregressive stage remained the stable peak, so splitting flow classifier-free guidance would add complexity without reducing the requirement.

## Measured optimization decisions

The pre-change warm baseline used the same persistent profile, seed 7, one headed Chrome worker, and zero artifact fetches across three runs. It averaged 104,956 ms of browser wall time. Autoregressive inference averaged 15,077.2 ms with a 0.16% coefficient of variation, and flow inference averaged 5,795.4 ms with a 0.62% coefficient of variation.

| Candidate | Result | Decision |
| --- | --- | --- |
| Simple storage-buffer cache | GPU peak increased from 6,835 to 12,074 MiB and AR inference slowed by 2.38% to 2.79% | Rejected and reverted |
| Decoder prepacking | GPU peak was 6,836 MiB, AR inference slowed by 0.46%, and AR stage time slowed by 0.79% | Rejected and reverted |
| Manifest-scoped completed-artifact receipts | Warm artifact validation fell from 76,282.1 to 223.5 ms on average | Retained |

The retained cache change trusts a completion receipt only inside the directory keyed by the complete manifest text hash and only when the artifact still has the declared byte length. Initial downloads and marker-less files remain SHA-256 verified. A size-mismatched completed artifact and its receipt are removed before a clean download, while ordinary partial downloads remain resumable.

After this change, three warm generations averaged 30,587 ms of browser wall time, 70.86% faster than the baseline. Worker time from manifest through result averaged 29,414.4 ms. The three artifact phases took 238.4, 214.9, and 217.1 ms, fetched zero files, and each produced an 880,684-byte WAV. AR inference changed by only 0.83%, flow inference by 0.07%, and the GPU peak changed from 6,835 to 6,834 MiB.

No further graph split, custom WGSL, new quantization, duplicated sessions, or stage overlap is warranted for this milestone. The dominant inference costs are approximately 15.2 seconds for autoregressive generation and 5.8 seconds for flow, but the fixed workload is already usable after removing the 76-second cache scan.

## Progress and cancellation

The worker now exposes only progress that can be measured honestly:

- artifact bytes, current file, and cache-hit state;
- named session creation as indeterminate activity;
- autoregressive frames from 1 through 125, rolling frames per second, elapsed time, and ETA after three samples;
- the condition step;
- flow steps from 1 through 30, rolling milliseconds per step, elapsed time, and ETA after three steps;
- vocoder synthesis, WAV encoding, final WAV bytes, and total elapsed time;
- cancellation by terminating the inference worker.

ONNX Runtime does not expose granular session-compilation progress, so the UI does not invent a percentage for that work. Results are published only after stage sessions and devices have been cleaned up.

## Reproduction and limits

Run the combined fallback-disabled gate with one worker and the persistent browser profile:

```powershell
$env:MINIMAX_RELEASE = 'music-5s'
$env:MINIMAX_MUSIC_CHROME_PROFILE = 'artifacts/music-browser-profile'
npx playwright test tests/browser/music-generation.spec.ts --project=chrome --workers=1
```

The measurements use model revision `fbdf52fbaaca799592917417eb05f1899f1255ec`, Diffusers revision `3681e65996b4d2589219720101a6acbfd25073f8`, and combined manifest SHA-256 `5c295ebfb4b7849d317cf0abd3dd8bfc9da3b58dc74de12a3523c07f28d4500e`.

This milestone validates execution, dimensions, finite outputs, WAV structure, browser decoding, cache reuse, progress, and cleanup. It does not certify an 8 GiB device, longer music, non-Chromium browsers, or subjective audio quality.

The final progress-aware headed Chrome gate passed in 1.5 minutes for two warm generations. It displayed live autoregressive progress, fetched zero artifacts on the repeated run, decoded the WAV through `AudioContext`, and saved an 880,684-byte file with SHA-256 `3093550ac43137d53d65e4ee72006e5574004703730af9059f6ada902697d0fb`.
