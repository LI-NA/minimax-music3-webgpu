# Variable-duration browser generation

The `music-variable` release accepts any finite positive requested maximum through 300 seconds. It derives `floor(durationSeconds * 25)` semantic frames and requires at least one frame, so the effective duration resolution is 0.04 seconds. A normal product request respects the model's audio-end token, so the returned song may be shorter than the requested maximum. It is not retried with another seed and its completed prefix is preserved.

The release uses the official long-song chunk contract:

- up to 200 semantic frames per acoustic window;
- 100-frame chunk stride;
- 172 latent continuation overlap, with an 86-latent left crop on non-first chunks;
- one maximum condition graph and one maximum flow graph;
- one symbolic-length mono vocoder graph, run once per channel; and
- direct PCM16 writes into the final 44.1 kHz stereo WAV.

Condition and flow sessions are reused across chunks. Autoregressive, acoustic, and vocoder device groups are released sequentially. Intermediate Euler latents remain on the GPU, while cross-device handoffs use exact raw FP16 data.

## Model download and cache

Page load inspects the project cache but does not start a download. The user must select **Download Model**. The Window requests persistent storage before starting, but denial leaves storage best-effort and does not block the download when the reported capacity is sufficient. [`navigator.storage.estimate()`](https://storage.spec.whatwg.org/) is advisory: it neither reserves space nor guarantees that a later write will succeed. The worker blocks transfer when the estimate is unavailable or insufficient, and it also handles an actual `QuotaExceededError` from the file write.

Downloads resume with HTTP Range requests and validate each completed file by SHA-256 before writing its receipt. **Retry Download** reuses both verified complete files and resumable partial files. **Cancel** stops the current transfer and keeps partial data for a later retry. **Remove Cached Model** deletes only caches named by the exact project manifest, preserving unrelated data in the origin private file system (OPFS). See the standard [Storage](https://storage.spec.whatwg.org/) and [File System](https://fs.spec.whatwg.org/) specifications for the underlying browser APIs.

Product generation requires the active release cache to be ready. Diagnostic routes retain their legacy cache behavior. Download progress reports verified bytes, the current file, transfer rate, and ETA. This technical demo performs downloads only while its page and worker are active. It does not install a background service worker or provide an automatic retry engine.

## Active release qualification

The active programmatic-input release has manifest SHA-256 `730293c66360cc9a413446311d2fd7957b547423d38bf7f81b80d2d331f96232`. A headed Chrome gate passed 6 and 10-second requests using the raw product request contract, returned 150 and 250 retained frames, produced 1,056,812-byte and 1,763,372-byte WAV files, and fetched zero completed artifacts. Total dedicated GPU memory rose from 2,987 MiB to 8,800 MiB, an increase of 5,813 MiB.

## Archived long-duration qualification

The remaining measurements in this document were captured from the archived pre-programmatic release with manifest SHA-256 `12fe28c91c2474f6a4a6ed02fe6f087c0c3b310cbc421138757dacf9072a6e0a`. That release is preserved at `artifacts/archive/music-variable-pre-programmatic-20260822-0208`. Its model graphs and weights provide historical duration and capacity evidence, but these runs have not been repeated against the active manifest.

All archived measurements below used headed Chrome 151, one persistent profile, CPU execution-provider fallback disabled, seed 7, and zero completed-artifact fetches. GPU numbers are total dedicated memory from `nvidia-smi`. The project budget is the increase from the sample taken immediately before Chrome starts, not 12 GiB total device use.

| Requested maximum | Product result | Retained frames | Chunks | WAV bytes | Baseline MiB | Raw peak MiB | Increase MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 s | Maximum reached | 150 | 1 | 1,056,812 | 2,977 | 8,502 | 5,525 |
| 10 s | Maximum reached | 250 | 2 | 1,763,372 | 2,977 | 8,502 | 5,525 |
| 30 s | Maximum reached | 750 | 7 | 5,296,172 | 2,898 | 8,600 | 5,702 |
| 60 s | Maximum reached | 1,500 | 14 | 10,596,396 | 3,128 | 9,872 | 6,744 |
| 120 s | Natural audio-end | 1,743 | 17 | 12,312,620 | 2,975 | 10,188 | 7,213 |

The 120-second request returned 3,078,144 samples per channel, or about 69.799 seconds. Its gate exited nonzero only because the qualification expected the full requested maximum. The WAV itself was complete, finite, decodable, stereo, and structurally valid. This is normal product behavior, not a crash or memory failure.

The current seed-7 evidence therefore proves normal maximum-frame product generation through 60 seconds and proves honest natural termination on a longer request. It does not prove that every prompt and seed will reach 60 seconds, because audio-end is sampled by the model.

## Five-minute capacity diagnostic

A separate diagnostic request tested the full 300-second computation without changing product behavior. It sampled normally until audio-end. When audio-end was selected, it excluded only that token, consumed one additional random draw, and continued. The original prefix remained unchanged through the first audio-end at retained frame 1,743. Everything after that boundary is a synthetic capacity continuation and is not evidence of native five-minute music quality.

The diagnostic passed in headed Chrome:

- 7,500 retained frames and `max-frames` termination;
- 7,501 semantic decisions, 52,507 RVQ calls, and 7,500 feedback calls;
- 74 acoustic chunks, 2,220 flow calls, and 148 mono vocoder calls;
- 83 suppressed audio-end decisions, first at frame 1,743;
- 13,247,488 samples per channel;
- canonical 52,989,996-byte PCM16 stereo WAV;
- WAV SHA-256 `fb7527d567c29ca6ac8d303635d45520d2fd8e6673ee3b92db126b3da7f1d0a3`;
- zero artifact fetches;
- baseline 3,168 MiB, raw peak 13,892 MiB, increase 10,724 MiB, and final 2,875 MiB; and
- no device loss, CPU fallback, telemetry failure, incremental-budget breach, or physical-capacity breach.

The requested telemetry cadence was 100 ms and the effective mean was 108.76 ms over 20,269 samples. The incremental limit was 12,288 MiB. The separate physical guard was 15,864 MiB on the 16,376 MiB adapter.

This result qualifies the runtime and memory architecture for the complete five-minute workload on the tested 16 GiB RTX 4080. It does not advertise native five-minute output for this prompt and seed. Product generation continues to respect audio-end.

## Progress and cancellation

The worker reports only measurable progress:

- artifact bytes, current file, cache state, and an exact completion event, with download callbacks coalesced to at most one intermediate update per 100 ms;
- indeterminate named session creation;
- retained AR frames, rolling frames per second, elapsed time, and ETA after three samples;
- acoustic chunk boundaries and condition timing;
- flow steps over the exact chunk-derived total, rolling step time, elapsed time, and ETA;
- completed vocoder channel calls over the exact chunk-derived total;
- WAV allocation and byte count; and
- completion only after session and device cleanup.

Cancellation terminates the inference worker. Browser gates cover cancellation during late AR and during the second flow chunk, and reject any leaked WAV or completion event.

## Programmatic input

The product worker accepts every intentional generation setting without relying on UI defaults:

```ts
worker.postMessage({
  type: 'generate-music',
  manifestUrl: 'http://127.0.0.1:5174/manifest.json',
  prompt: 'Warm female vocal, 96 BPM',
  lyrics: '[verse]\nHello',
  seed: 7,
  durationSeconds: 10.5,
  sampling: {
    globalGuidance: 1.5,
    semanticTopK: 50,
    residualTopK: 50,
    temperature: 1,
    flowGuidance: 1.7,
    flowSteps: 30,
  },
});
```

All fields are required. The worker loads the verified release tokenizer from OPFS, prepares the conditional and unconditional prompt rows, derives prompt and frame counts, and echoes the validated effective input in the result. Callers cannot provide token IDs, frame counts, cache lengths, latent tensors, chunk geometry, model dimensions, or checkpoint constants.

## Reproduction

The variable release is selected with `MINIMAX_RELEASE=music-variable`. The short headed gate is `tests/browser/variable-duration.spec.ts`. Long product and capacity measurements use `tests/browser/long-duration.spec.ts` through `tests/browser/variable-duration-telemetry-runner.ts` with a persistent profile and a unique ignored capture directory.

Capacity mode is diagnostic-only and must be explicitly selected. It is intentionally separate from the normal request protocol and must not be exposed as an exact-duration music option.
