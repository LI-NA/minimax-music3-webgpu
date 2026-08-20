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
- Warm metric gate: passed in 4.1 minutes, including two generations
- Completed-artifact fetches on warm runs: 0
- Download: `artifacts/generated/minimax-music3-5s.wav`
- WAV: 880,684 bytes, PCM16, 44,100 Hz, 2 channels, 220,160 samples per channel
- Browser decoding: `AudioContext.decodeAudioData` returned 44,100 Hz, 2 channels, and 220,160 samples

The persisted warm-run timings were:

| Stage | Session creation | Inference or generation | End-to-end stage |
| --- | ---: | ---: | ---: |
| Autoregressive and RVQ | 5,755.4 ms | 15,404.2 ms | 21,254.7 ms |
| Condition | 80.7 ms | 27.0 ms | 274.1 ms |
| Flow | 1,887.7 ms | 5,885.8 ms | 7,868.1 ms |
| Vocoder and WAV | 317.6 ms | 167.9 ms | 658.4 ms |

Autoregressive inference retained 125 frames at 8.11 frames per second. The first flow step took 348.9 ms. The remaining 29 steps took 188.1 to 191.9 ms each, and all 30 steps averaged 196.2 ms.

## GPU memory

Total-system dedicated GPU memory was sampled through `nvidia-smi`. The monitor requested a 100 ms sleep, but command overhead produced an effective cadence of about 170 ms, with 2,930 samples across the 8.3-minute gate and short pre/post windows.

- Baseline: 1,695 MiB
- Observed total-system peak: 6,822 MiB
- Observed increase from baseline: 5,127 MiB
- Final after Chrome closed: 1,695 MiB
- Limit: 12,288 MiB

The combined autoregressive segment reached the 6,822 MiB peak. A later condition/flow segment reached 3,312 MiB, but its internal boundary was not externally labelled, so separate combined condition and flow peaks are not claimed. The combined vocoder peak was also not separately attributable. Standalone measurements provide supporting context only: condition was 1,947 to 2,188 MiB, flow was 1,695 to 3,252 MiB, and vocoder was 1,695 to 2,751 MiB. These are total-system samples, not isolated process allocations.

ONNX Runtime emitted non-fatal constant-folding warnings for vocoder Snake reciprocal nodes because no CPU kernel was available. The WebGPU-only vocoder still completed with CPU fallback disabled.

This gate validates artifact reuse, stage cleanup, exact dimensions, WAV structure, and browser decodability. It does not inspect waveform quality, compare against a reference song, run an audio classifier, or perform the deferred listening check.
