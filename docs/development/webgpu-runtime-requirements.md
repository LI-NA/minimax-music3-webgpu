# WebGPU runtime requirements and optimization

This document covers the measured MiniMax-Music3 browser workloads from the fixed five-second baseline through the five-minute capacity diagnostic. Product duration is a requested maximum. A sampled audio-end may return a shorter valid song.

## Recommended configuration

Use one runtime configuration:

- Chromium with a non-fallback WebGPU adapter and `shader-f16`
- ONNX Runtime Web `1.30.0-dev.20260813-72e1c9c9b8`
- the project-served JSPI WASM with patched SHA-256 `0569a267c57da3947fefc95934a7eee1426188cba11997be556515d482347534`
- `executionProviders: ['webgpu']` with CPU fallback disabled
- sequential execution and all graph optimizations
- memory patterns and graph capture disabled
- prepacking disabled
- the default bucketed storage-buffer cache
- sequential device groups for autoregressive, combined condition and flow, then vocoder work, with each group's sessions released before its device is destroyed

Separate low-memory and faster profiles are not justified by the measurements. The default configuration used less memory and was at least as fast as both tested alternatives.

## VRAM guidance

All gates used headed Chrome 151 on a 16 GiB RTX 4080. `nvidia-smi` sampled total dedicated GPU memory, including Windows and Chromium background use. The 12,288 MiB limit is the project increase above the sample taken immediately before Chrome starts. It is not a 12 GiB cap on total device use. A separate physical guard stops at 512 MiB below adapter capacity.

| Release evidence     | Workload                  | Outcome                                          | Baseline MiB | Raw peak MiB | Increase MiB |
| -------------------- | ------------------------- | ------------------------------------------------ | -----------: | -----------: | -----------: |
| Active `730293...`   | Combined 6 s and 10 s     | Both maxima reached with raw programmatic inputs |        2,987 |        8,800 |        5,813 |
| Archived `12fe28...` | Fixed 5 s warm baseline   | Maximum reached                                  |        1,704 |        6,834 |        5,130 |
| Archived `12fe28...` | Combined 6 s and 10 s     | Both maxima reached                              |        2,977 |        8,502 |        5,525 |
| Archived `12fe28...` | 30 s                      | Maximum reached                                  |        2,898 |        8,600 |        5,702 |
| Archived `12fe28...` | 60 s                      | Maximum reached                                  |        3,128 |        9,872 |        6,744 |
| Archived `12fe28...` | 120 s request             | Natural end at 1,743 frames, about 69.8 s        |        2,975 |       10,188 |        7,213 |
| Archived `12fe28...` | 300 s capacity diagnostic | Full workload, synthetic after native audio-end  |        3,168 |       13,892 |       10,724 |

Practical physical-memory recommendations are:

| Tier   | Guidance                                                                                                                                                                    |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8 GiB  | Not recommended. The active short variable gate reached 8,800 MiB total, and no 8 GiB adapter has been tested.                                                              |
| 10 GiB | Reasonable short-generation target based on the active 6 and 10-second peak of 8,800 MiB, but not a certified minimum.                                                      |
| 12 GiB | Recommended for the measured product workload through one minute, with more than 2 GiB between the 60-second peak and physical capacity. No 12 GiB adapter has been tested. |
| 16 GiB | Tested recommendation for the complete five-minute capacity workload. The measured raw peak was 13,892 MiB and retained the 512 MiB physical reserve.                       |

Adapter limits, driver behavior, and Chromium allocation can differ even when nominal VRAM matches. Only the RTX 4080 is verified. Before a long generation, wait for Android emulators, virtual machines, or other applications that materially consume GPU memory. Ordinary baseline use is included in the spawn-adjacent measurement and is not subtracted twice.

The WebGPU adapter must expose a 128 MiB storage-buffer binding and at least nine storage buffers per shader stage. The active variable release stores 8,083,501,198 referenced artifact bytes in the browser profile.

## Memory ownership

Autoregressive generation is the limiting stage and grows with retained duration. The five-second application-owned GPU tensor payloads are small relative to ONNX Runtime weights, activations, and buffer buckets:

| Stage                         | Exact peak application-owned GPU bytes |
| ----------------------------- | -------------------------------------: |
| Autoregressive and RVQ        |                             97,058,816 |
| Condition output              |                              1,761,280 |
| Flow current and next tensors |                                220,160 |
| Vocoder                       |                                      0 |

The five-second autoregressive figure includes the old and new KV sets at the final decode boundary. Final live KV data is 48,660,480 bytes. At 300 seconds, the exact old-plus-new KV payload reaches 4,447,010,816 bytes and the flat FP16 retained-hidden handoff reaches 491,520,000 bytes. The actual capacity measurement supersedes the earlier 10,982 MiB analytical total projection: raw total-system use reached 13,892 MiB.

Stage handoffs remain on CPU as raw FP16 and no GPU tensor crosses a device boundary. Variable generation writes every retained hidden group directly into one flat allocation, reuses the condition and flow sessions across chunks, reuses one mono vocoder session, and writes PCM16 directly into the final WAV. It does not allocate a second full hidden copy or a concatenated float32 song.

ONNX Runtime may allocate internal readback buffers while downloading tensors. Those buffers are not application-owned and are represented only in the measured total-system GPU peak.

Standalone total-system measurements provide supporting context only. Condition rose from 1,947 to 2,188 MiB, flow from 1,695 to 3,252 MiB, and vocoder from 1,695 to 2,751 MiB. The combined autoregressive stage remained the stable peak at every duration. The 300-second run dropped from the AR peak to about 4.7 GiB during flow, so splitting flow classifier-free guidance would add complexity without reducing the requirement.

## Measured optimization decisions

The pre-change warm baseline used the same persistent profile, seed 7, one headed Chrome worker, and zero artifact fetches across three runs. It averaged 104,956 ms of browser wall time. Autoregressive inference averaged 15,077.2 ms with a 0.16% coefficient of variation, and flow inference averaged 5,795.4 ms with a 0.62% coefficient of variation.

| Candidate                                                                                                                      | Result                                                                                                      | Decision              |
| ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | --------------------- |
| Simple storage-buffer cache                                                                                                    | GPU peak increased from 6,835 to 12,074 MiB and AR inference slowed by 2.38% to 2.79%                       | Rejected and reverted |
| Decoder prepacking                                                                                                             | GPU peak was 6,836 MiB, AR inference slowed by 0.46%, and AR stage time slowed by 0.79%                     | Rejected and reverted |
| Manifest-scoped completed-artifact receipts                                                                                    | Warm artifact validation fell from 76,282.1 to 223.5 ms on average                                          | Retained              |
| Sequential stage devices, flat FP16 retained hiddens, reused acoustic sessions, symbolic mono vocoder, and direct PCM16 output | Completed the 300-second capacity workload with a 10,724 MiB project increase                               | Retained              |
| Artifact progress coalescing                                                                                                   | Replaced per-16-KiB message floods with immediate, at-most-100-ms intermediate, and exact completion events | Retained              |
| Buffered artifact writes and overlapped transfers                                                                              | Cold download of the variable release rose from 5-10 MB/s to a sustained 93-152 MiB/s                       | Retained              |

The download rate was pinned by write granularity rather than by disk or server speed. Chrome hands a network response body to a reader in units of roughly 8 KiB, set by the transport and not by the server: the dev server's `/ort/` handler writes a 16 MB body in a single call and it still arrives that way, and pausing the reader for 500 ms still yields a 16 KiB chunk. A same-origin release is the worst case, because a remote CDN has enough latency to make each delivery larger, and Hugging Face measured a 16 KiB median against 8 KiB for localhost. Since every `FileSystemWritableFileStream.write` is an asynchronous round trip through a swap file, one write per delivered chunk meant one round trip per 8 KiB.

Measured in a worker against the same release, 32 MiB per run: per-chunk `createWritable` reached 19.9 MB/s, 1 MiB batches reached 109.4 MB/s, and `createSyncAccessHandle` reached 90.0 MB/s per chunk. Batching alone recovers the transfer's own limit, so the write API is unchanged. Four concurrent transfers then cover the per-file gaps, which matters most against a hosted release where one HTTPS stream rarely saturates a link.

The retained cache change trusts a completion receipt only inside the directory keyed by the complete manifest text hash and only when the artifact still has the declared byte length. Initial downloads and marker-less files remain SHA-256 verified. A size-mismatched completed artifact and its receipt are removed before a clean download, while ordinary partial downloads remain resumable.

After this change, three warm generations averaged 30,587 ms of browser wall time, 70.86% faster than the baseline. Worker time from manifest through result averaged 29,414.4 ms. The three artifact phases took 238.4, 214.9, and 217.1 ms, fetched zero files, and each produced an 880,684-byte WAV. AR inference changed by only 0.83%, flow inference by 0.07%, and the GPU peak changed from 6,835 to 6,834 MiB.

No further graph split, custom WGSL, new quantization, duplicated sessions, or stage overlap is warranted for this milestone. The dominant inference costs are approximately 15.2 seconds for autoregressive generation and 5.8 seconds for flow, but the fixed workload is already usable after removing the 76-second cache scan.

## Progress and cancellation

The worker now exposes only progress that can be measured honestly:

- artifact bytes, current file, cache-hit state, and a distinct verified-completion event, with intermediate download callbacks coalesced to at most one update per 100 ms;
- named session creation as indeterminate activity;
- autoregressive retained frames over the requested maximum, rolling frames per second, elapsed time, and ETA after three samples;
- condition completion for each acoustic chunk;
- flow steps over `chunkCount * 30`, rolling milliseconds per step, elapsed time, and ETA after three steps;
- vocoder channel calls over `chunkCount * 2`;
- WAV allocation and encoding, final WAV bytes, and total elapsed time;
- cancellation by terminating the inference worker.

ONNX Runtime does not expose granular session-compilation progress, so the UI does not invent a percentage for that work. Results are published only after stage sessions and devices have been cleaned up.

## Reproduction and limits

Run the short variable-duration fallback-disabled gate with one worker and a persistent browser profile:

```powershell
$env:MINIMAX_VARIABLE_CHROME_PROFILE = 'artifacts/browser-profiles/variable-duration/local'
npx playwright test tests/browser/variable-duration.spec.ts --project=chrome --workers=1
```

The measurements use model revision `fbdf52fbaaca799592917417eb05f1899f1255ec` and Diffusers revision `3681e65996b4d2589219720101a6acbfd25073f8`. The long-duration table uses archived variable manifest SHA-256 `12fe28c91c2474f6a4a6ed02fe6f087c0c3b310cbc421138757dacf9072a6e0a`. The active programmatic-input gate uses manifest SHA-256 `730293c66360cc9a413446311d2fd7957b547423d38bf7f81b80d2d331f96232`.

The pinned ONNX Runtime package has an FP16 coordinate defect in its WebGPU `ConvTranspose` shader. Vite patches the eight coordinate casts to FP32 while serving and bundling the JSPI WASM. The patch fails closed against the original byte length, original SHA-256, replacement count, and final SHA-256. Session setup uses patch-versioned absolute JSPI URLs, and the browser gate verifies that no automatically emitted unpatched WASM URL is requested.

The normal product path reached its requested maximum through the qualified 60-second case. The seed-7 120-second request ended naturally after about 69.8 seconds. The five-minute capacity diagnostic completed the full workload only by suppressing 83 audio-end decisions beginning at retained frame 1,743. That diagnostic validates memory, lifecycle, chunking, progress, vocoder, and WAV assembly capacity. It does not certify native five-minute music or subjective quality.

Detailed duration evidence is in [Variable-duration browser generation](variable-duration-generation.md). The original fixed five-second benchmark remains documented in [Five-second browser generation](five-second-generation.md).

The corrected progress-aware headed Chrome gate passed in 1.4 minutes for two warm generations. It displayed live autoregressive progress, fetched zero artifacts on the repeated run, decoded the WAV through a 44.1 kHz `AudioContext`, and saved an 880,684-byte file with SHA-256 `c16731b5597fcf343fc1cf05c842ea559c5406bf95678bff75c88767e4102800`.
