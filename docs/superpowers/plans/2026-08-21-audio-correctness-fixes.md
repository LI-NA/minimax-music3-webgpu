# Audio Correctness Fixes

**Status:** Implemented and verified

## Goal

Restore the fixed five-second WebGPU output without changing the MiniMax Music 3 checkpoint, sampling contract, or acoustic graph math. Fix the two proven runtime defects and check adjacent paths for the same failure mode.

## Confirmed defects

1. Product code copies GPU-only condition and flow tensors before ONNX Runtime flushes its internal command encoder. The copied FP16 buffers can therefore contain zeros even though `Tensor.getData()` returns the completed values.
2. The pinned JSPI WebGPU runtime performs `ConvTranspose` spatial coordinate calculations in FP16. Coordinates above 65,504 overflow, causing the final vocoder layers to become bias-only for most of the clip.

## Implementation

1. Add one exact FP16 readback helper that validates type, residency, and shape, then uses ONNX Runtime `getData()` to flush and download the tensor. Reinterpret native `Float16Array` storage as raw `Uint16Array` bits without numeric conversion. Use it for condition, final flow latent, and RVQ hidden readback.
2. Patch the pinned JSPI WASM bytes only while Vite serves or bundles the runtime. Replace the eight `dy_element_t(` coordinate casts in the embedded `ConvTranspose` shader template with same-length FP32 casts. Fail closed if the pinned artifact no longer has the exact expected pattern count. Do not modify `node_modules` or the ONNX graph.
3. Add regressions for exact FP16 bit preservation, runtime flush ordering, and the patched WASM contract.
4. Extend the real five-second browser gate to reject a long constant tail and zero late-window AC energy. This is a defect regression, not general audio-quality scoring.

## Adjacent-path audit

- The RVQ conditional hidden reader used the same direct GPU copy pattern. Its current call order happened to flush through CPU logits first, but it now uses the shared safe helper so later reuse cannot reintroduce the race.
- Other current GPU downloads already use ONNX Runtime `getData()` and require no change.
- The pinned runtime has related tensor-type coordinate code in `Resize` and large pooling paths, but none of the seven fixed music graphs contains those operators. `GroupQueryAttention` converts a causal length in one exceptional branch, but the fixed maximum length is only 165. No unrequested runtime patch was added for unreachable cases.

## Verification

- Focused Vitest regressions pass.
- Python converter tests remain green because the model graph is unchanged.
- Typecheck, lint, and production build pass.
- A headed Chrome run produces a valid 880,684-byte WAV, has a varying final second, and does not repeat the previous multi-second constant tail.
- CPU fallback remains disabled and measured total GPU usage remains below 12 GiB.
