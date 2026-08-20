# Global LLM WebGPU feasibility gate

## Contract

- Checkpoint: `MiniMaxAI/MiniMax-Music3` at `fbdf52fbaaca799592917417eb05f1899f1255ec`
- Browser runtime: headed Windows Chrome with ONNX Runtime Web JSPI and WebGPU only
- Decoder input: batch-two, 40-token conditional and unconditional committed fixture
- Decode: ten cached steps using embedding row `151675` for both lanes
- Acceptance: cache lengths 40 through 50, GPU-resident KV tensors, finite reduced logits, and no artifact fetches on a repeated smoke

## Measurement record

The one-layer and full-release commands record the following results here when each headed Chrome gate finishes:

| Field | One layer | Full 36 layers |
| --- | --- | --- |
| Release manifest SHA-256 | `451e7fc51f20dbf2dbc8f6fc4616a9a81615914c167fbd5d2fb3d3cd8b30f51c` | Pending |
| Chrome and ONNX Runtime version | Chrome 151.0.7922.138, ORT Web 1.30.0-dev.20260813-72e1c9c9b8 | Pending |
| Adapter and declared limits | NVIDIA, `shader-f16`, 128 MiB binding, 9 storage buffers per shader stage | Pending |
| Session creation time | 956.3 ms | Pending |
| Prefill and ten cached-step timings | 267.3 ms, then 74.5, 1.7, 2.4, 2.7, 2.0, 1.5, 1.8, 2.8, 1.6, 1.7 ms | Pending |
| GPU baseline, peak, delta | 4,407 MiB, 4,655 MiB, 248 MiB | Pending |
| Cache reuse fetches | 0 | Pending |
| Result | PASS | Pending |

## One-layer gate, 2026-08-21

The referenced release is 1,886,202,902 bytes, or 1.757 GiB, across 23 hashed files. Every referenced artifact is at or below 128 MiB. The initial headed run exposed six attention-mask bookkeeping operations without WebGPU kernels. The converter now removes only that subgraph and exposes its exact results as `seqlens_k` and `total_seq_len` INT32 inputs. The browser computes those values from the unpadded two-lane sequence length. Model weights and GroupQueryAttention math are unchanged.

The first error was:

```text
This session contains graph nodes that are assigned to the default CPU EP, but fallback to CPU EP has been explicitly disabled by the user.
```

The second blocker was isolated to passing the same custom `GPUDevice` both through `ort.env.webgpu.device` and through the execution-provider object. That route creates an ORT WebGPU instance without the timed-wait feature needed by asynchronous pipeline creation. The runtime now assigns the requested device once through `ort.env.webgpu.device` and creates sessions with literal `executionProviders: ['webgpu']`.

The final headed Chrome run passed with `session.disable_cpu_ep_fallback=1`. It completed the exact 40-token conditional and unconditional prefill, advanced cache length from 40 through 50 over ten cached steps, kept every decoder hidden and KV tensor at `gpu-buffer`, produced finite reduced logits, and fetched zero artifacts during the repeated run. Reduced logits are intentionally downloaded to CPU for TypeScript sampling and are not counted as KV tensors.

The verification command was:

```powershell
$env:MINIMAX_RELEASE = 'global-one-layer'
npx playwright test tests/browser/global-decoder.spec.ts --project=chrome
```

Playwright reported `1 passed (1.4m)`. Most elapsed time was OPFS integrity verification across two workers. Session creation and inference timings are reported separately above. `nvidia-smi` sampled total device usage every 0.5 to 2 seconds. Those numbers include Windows and other applications, so the 248 MiB delta is the useful one-layer diagnostic measurement.

An unsuccessful gate must record the first allocation or unsupported-operator error and must not be described as feasible.
