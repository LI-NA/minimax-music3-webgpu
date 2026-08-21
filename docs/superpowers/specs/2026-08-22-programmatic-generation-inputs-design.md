# Programmatic generation inputs design

**Date:** 2026-08-22

## Status

Implemented and validated in headed Chrome for 6 and 10-second requests. This design adds a complete non-UI product input path. UI defaults and presentation remain out of scope.

## Objective

Let a caller provide every value that intentionally changes music generation while deriving internal correctness values and preserving checkpoint structure constants.

The product request is:

```ts
type MusicGenerationInput = {
  manifestUrl: string;
  prompt: string;
  lyrics: string;
  seed: number;
  durationSeconds: number;
  sampling: {
    globalGuidance: number;
    semanticTopK: number;
    residualTopK: number;
    temperature: number;
    flowGuidance: number;
    flowSteps: number;
  };
};
```

Every field is required. A future UI may supply defaults, but the programmatic API does not silently invent them.

## Input validation

- `manifestUrl` is a non-empty string.
- `prompt` and `lyrics` are non-empty strings, matching the pinned Diffusers input contract.
- `seed` is an integer from 0 through 4,294,967,295. The runtime must not silently wrap a negative or oversized value.
- `durationSeconds` is a finite positive number no greater than 300. The runtime derives `floor(durationSeconds * 25)` frames and requires at least one frame, matching the pinned model's 25-frame-per-second contract.
- `globalGuidance` and `flowGuidance` are finite, non-negative numbers. Flow guidance must remain representable as finite FP16 because the graph consumes FP16 guidance.
- `semanticTopK` is an integer from 1 through 16,385.
- `residualTopK` is an integer from 1 through 1,024.
- `temperature` is finite and greater than zero.
- `flowSteps` is a positive safe integer.

## Prompt preparation

The worker prepares prompt tokens only after the manifest and tokenizer artifacts have passed their existing byte and SHA-256 checks in OPFS.

The runtime reproduces the pinned Diffusers preprocessing:

1. Clean the caption using the source special-tag and accepted Markdown rules.
2. Normalize lyric structure tags and prepend `[start]\n`.
3. Assemble the exact checkpoint string:

   ```text
   <|im_start|><|caption_start|>{caption}<|caption_end|><|lyrics_start|>{lyrics}<|lyrics_end|><|im_end|><|audio_start|>
   ```

4. Encode it using the release's `tokenizer.json` and `tokenizer_config.json`.
5. Copy the conditional row and replace positions `1:-2` with audio-CFG token 151654 to create the unconditional row.

The caller cannot provide token IDs, `promptTokens`, `requestedFrames`, chunk plans, latent noise, or cache lengths. The worker derives them and checks:

```text
promptTokens <= 5000
promptTokens + requestedFrames <= 10240
```

The fixed reference prompt must reproduce the existing 40-token fixture exactly before the tokenizer path is accepted.

## Autoregressive sampling

The product generator consumes dynamic prompt rows and their common length. Prefill shapes, sequence lengths, KV cache totals, and retained-frame plans use that actual length.

The sampler applies the requested values as follows:

- `globalGuidance` controls conditional and unconditional interpolation for semantic and residual decisions.
- `semanticTopK` controls semantic conditional candidate restriction and the guided semantic distribution.
- `residualTopK` controls residual distributions.
- `temperature` divides the finite guided logits before exponentiation. Temperature 1 preserves the existing reference behavior.

The fixed five-second smoke, frame diagnostic, and five-minute capacity diagnostic keep their explicit fixture and reference sampling values. They do not become product defaults.

## Flow sampling

The maximum-window flow graph replaces its embedded FP16 guidance constant with a required FP16 scalar input named `guidance`.

The runtime generates a schedule for the requested positive `flowSteps` using the pinned formula:

```text
sourceSigmas = linspace(1, 1 / flowSteps, flowSteps), cast to float32
timesteps = 1 - sourceSigmas, cast to float32
dts = diff(concat(timesteps, [1])), cast to float32
```

The 30-step schedule must remain bit-identical to the frozen reference fixture. Progress totals become `chunkCount * flowSteps`.

The release manifest may retain 1.7 and 30 as the reference recipe values for future defaults and comparison metadata, but the product runtime must use the request values.

## Result metadata

The result echoes one validated effective-input record containing:

- raw prompt and lyrics;
- assembled prompt and both token rows;
- actual prompt token count;
- seed and requested duration;
- every sampling value; and
- termination and retained-frame information already present in the result plan.

Fixed comparison metadata is emitted only when raw text, assembled text, token rows, seed, duration, and all sampling parameters match the frozen comparison case.

## Artifact compatibility

- Existing artifacts and generations are never deleted or overwritten.
- The music-variable flow graph is rebuilt into a unique staging generation and promoted atomically.
- Unchanged Global, RVQ, condition, and vocoder artifacts are reused by the existing release builder.
- The release validator requires the new guidance input before publishing.
- Existing fixed releases remain available for their smoke and diagnostic routes.

## Non-goals

- Building or styling the Web UI.
- Exposing model dimensions, vocabulary offsets, sample rate, RVQ depth, chunk overlap, crop sizes, scheduler family, or checkpoint revisions as user inputs.
- Changing the product audio-end policy.
- Accepting caller-provided token IDs or latent tensors.
- Adding another inference runtime or custom WebGPU kernels.

## Acceptance criteria

1. The fixed raw prompt and lyrics encode to the exact existing 40-token conditional and unconditional rows.
2. A changed prompt changes the actual AR prefill rows and dynamic prompt length.
3. The public request rejects unknown, missing, derived, or invalid fields.
4. All six sampling values reach the stage that consumes them.
5. Temperature 1 and flow settings 1.7/30 preserve the frozen reference math.
6. A non-default flow step count produces the exact cross-language schedule and matching progress totals.
7. Fixed comparison metadata is absent when any input differs.
8. Existing diagnostics retain their fixed contracts.
9. The rebuilt music-variable graph validates, references intact bounded shards, and preserves every previous artifact generation.
10. Unit, Python, lint, typecheck, build, and one non-UI integration path pass.
