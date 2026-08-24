# B32 Selective Precision Design

## Goal

Replace the active symmetric Q4 block-size-128 Global decoder and Flow transformer weights with block-size-32 weights, while keeping the Flow `time_proj.weight` and `proj_out.weight` matrices in FP16. Preserve every active B128 release and its receipt before replacement.

## Precision contract

- Global decoder large `MatMul` weights use symmetric Q4, block size 32, accuracy level 4.
- The prompt embedding table and reduced output head remain FP16.
- Flow transformer large linear weights use symmetric Q4, block size 32, accuracy level 4.
- Flow `time_proj.weight` and `proj_out.weight` use FP16 `MatMul` nodes.
- RVQ, condition encoder, and vocoder precision remain unchanged.
- The combined manifest records the Q4 contract and both Flow FP16 weight names.

The scope intentionally does not promote additional first, last, attention, or MLP layers. Those candidates require activation or listening evidence that is not currently available.

## Conversion behavior

The quantization constants are shared by the Python converter and its validators. Global B32 work uses precision-specific work directories so a new conversion cannot copy unreferenced B128 shards into the new standalone release. When the full ORT GenAI builder exceeds the system commit limit, a low-memory path uses a preserved B128 graph as topology and regenerates every Q4 value from the pinned official BF16 source, one matrix at a time. Flow conversion excludes the two declared FP16 weights from the expected `MatMulNBits` node set and validates their FP16 `MatMul` nodes and initializer shapes.

Changing the Flow converter changes its build fingerprint, so `build-music-variable` rebuilds Flow while continuing to reuse verified Global, RVQ, condition, and vocoder artifacts where their contracts are unchanged. A legacy release without component fingerprints is rebuilt once before later generations can use component reuse.

## B128 preservation

Before conversion, the existing B128 Global builder and packed work directories move into one generation directory under `artifacts/archive/b128-work/`. A JSON receipt records every moved file's relative path, size, and SHA-256.

Active Global and `music-variable` releases remain in place until the existing promotion code validates and archives them during successful replacement. This preserves atomic rollback. No archive generation is removed automatically.

## Runtime contract

The TypeScript manifest parser accepts only Q4 block size 32 and requires the exact Flow FP16 weight list. A new manifest hash creates a distinct OPFS cache, so existing B128 browser caches remain isolated.

## Verification

1. Python unit tests prove B32 graph attributes, Q4 node counts, FP16 exception nodes, initializer types, manifest metadata, archive safety, and fingerprint invalidation.
2. TypeScript unit tests prove the browser parser accepts the new contract and rejects B128 or a missing FP16 exception.
3. A generated B32 `MatMulNBits` fixture runs in headed Chrome with CPU fallback disabled.
4. One-layer and full Global releases pass their existing headed Chrome gates.
5. Static validation proves 219 Flow Q4 nodes plus the two FP16 linear weights.
6. The combined variable-duration 6-second and 10-second headed Chrome gate remains deferred while browser testing is paused.
7. Fresh tests, lint, typecheck, build, manifest integrity, release receipt, file limits, and final diff are checked before handoff.

Subjective quality is not claimed by these gates. The user will perform the final listening test against the preserved B128 release.
