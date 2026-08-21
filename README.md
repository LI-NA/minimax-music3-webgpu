# MiniMax Music 3 WebGPU

MiniMax Music 3 WebGPU is an experimental community project that aims to run the exact `MiniMaxAI/MiniMax-Music3` checkpoint locally in a desktop Chromium browser through WebGPU.

The exact checkpoint now passes end-to-end variable-duration browser gates. Headed Chrome runs the q4 Global LLM, RVQ, condition encoder, q4 flow transformer, and mixed-precision vocoder sequentially with CPU fallback disabled, then emits and decodes a 44.1 kHz stereo WAV. This is a runtime milestone, not the final prompt and lyrics UI.

The active programmatic-input release contains 8,083,501,198 referenced artifact bytes and is cached in OPFS. It passed 6 and 10-second generation with raw prompt, lyrics, seed, duration, and sampling inputs. The archived pre-programmatic release reached the requested maximum at 30 and 60 seconds, returned a valid natural end for a 120-second request, and completed a separate five-minute capacity-only diagnostic. Those longer measurements remain historical evidence for that archived manifest and are not presented as fresh qualification of the active release.

The current physical VRAM guidance is 10 GiB for short generation, 12 GiB for the measured one-minute workload, and 16 GiB for the tested five-minute capacity workload. The one-minute and five-minute figures come from the archived pre-programmatic manifest. These are recommendations from one RTX 4080 system, not certified hardware minimums. The 12,288 MiB development budget means project-attributable growth above the pre-run baseline, not 12 GiB total device use.

## Phase 1 goal

Phase 1 will grow the validated runtime into a minimal local music generator that:

- runs entirely in the browser after downloading model artifacts;
- uses the official MiniMax Music 3 checkpoint without distillation or model substitution;
- accepts a music description and tagged lyrics;
- accepts any positive requested maximum through 300 seconds at 25-frame-per-second resolution and preserves a shorter natural audio-end result;
- plays and saves the result as a 44.1 kHz stereo WAV file;
- targets Windows desktop Chromium with a measured 12 GiB incremental GPU-memory budget; and
- caches approximately 8.08 GB of converted model artifacts in browser storage.

The first version prioritizes reliable local generation over UI polish, broad browser support, forced exact-duration output, or automated audio-quality analysis.

## Proposed architecture

The browser runtime will use ONNX Runtime Web with its WebGPU execution provider. TypeScript will control sampling, model-stage transitions, chunking, progress, cancellation, and WAV output. Python and CUDA may be used to convert the official checkpoint, but they will not be end-user runtime requirements.

Large model stages will be loaded and released separately to stay below the GPU-memory target. Custom WGSL kernels will be added only when a confirmed ONNX Runtime Web limitation blocks the end-to-end path.

See the [approved phase 1 design](docs/superpowers/specs/2026-08-20-minimax-music3-webgpu-design.md) for the full requirements and technical decisions.

For the exact Global LLM conversion and headed diagnostic commands, see [development conversion notes](docs/development/conversion.md) and the [feasibility record](docs/development/global-llm-feasibility.md). The original fixed-slice gate is documented in [five-second browser generation](docs/development/five-second-generation.md). Current duration evidence is in [variable-duration browser generation](docs/development/variable-duration-generation.md), with hardware guidance and retained optimizations in [WebGPU runtime requirements](docs/development/webgpu-runtime-requirements.md).

## Model and license

MiniMax Music 3 is distributed as open weights under the custom [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE). It is not an OSI-licensed open-source model. Converted browser artifacts must retain that license and its notices.

The application will identify generated audio as AI-generated and will not send prompts, lyrics, model weights, or generated audio to an inference server.

## Current environment

The initial development and browser-validation machine has:

- Windows with WSL2 available;
- an NVIDIA GeForce RTX 4080 with 16 GB VRAM;
- approximately 12 GB of practical WebGPU memory headroom;
- 64 GB system RAM;
- Chrome 151 or newer;
- Node.js and npm; and
- Python managed through `uv`.

CUDA is available through the installed NVIDIA driver on Windows and WSL2. A standalone CUDA toolkit is not currently required.
