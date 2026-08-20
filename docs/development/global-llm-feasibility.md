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
| Release manifest SHA-256 | `afe8964d5c7d30f84116fd45a9a8ae5b21ab8c1ec8f9f039c24b7b706f9b566a` | Pending |
| Chrome and ONNX Runtime version | Pending | Pending |
| Adapter and declared limits | Pending | Pending |
| Session creation time | Pending | Pending |
| Ten cached-step timings | Pending | Pending |
| GPU baseline, peak, delta | Pending | Pending |
| Cache reuse fetches | Pending | Pending |
| Result | Pending rewritten-graph gate | Pending |

## Initial one-layer diagnostic, 2026-08-21

The initial generated one-layer release was 1.757 GiB, with every physical artifact at or below 128 MiB. Headed Chrome reached decoder session initialization after the browser cached the graph and embedding shards. ONNX Runtime rejected the original builder graph before execution because mask bookkeeping nodes remained assigned to the default CPU execution provider while `session.disable_cpu_ep_fallback=1` was enabled.

The first error was:

```text
This session contains graph nodes that are assigned to the default CPU EP, but fallback to CPU EP has been explicitly disabled by the user.
```

This diagnostic was not a reason to permit CPU fallback. The converter now removes only the attention-mask bookkeeping subgraph, exposes `seqlens_k` and `total_seq_len` as explicit GQA inputs, and rejects converted graphs that retain the mask path. The rewritten one-layer artifact must still pass the headed Chrome gate before feasibility is claimed.

An unsuccessful gate must record the first allocation or unsupported-operator error and must not be described as feasible.
