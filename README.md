# MiniMax Music 3 WebGPU

MiniMax Music 3 WebGPU is an experimental community project that aims to run the exact `MiniMaxAI/MiniMax-Music3` checkpoint locally in a desktop Chromium browser through WebGPU.

The project is implementing the first browser feasibility gate. It converts the exact Global LLM into a reduced q4 release, caches release files in OPFS, and validates a 40-token prefill plus ten GPU-resident cached decode steps in headed Chrome before acoustic-stage work begins.

## Phase 1 goal

Phase 1 will provide a minimal local music generator that:

- runs entirely in the browser after downloading model artifacts;
- uses the official MiniMax Music 3 checkpoint without distillation or model substitution;
- accepts a music description and tagged lyrics;
- generates a 10 to 30 second song with vocals and accompaniment;
- plays and saves the result as a 44.1 kHz stereo WAV file;
- targets Windows desktop Chromium and a practical 12 GB GPU-memory ceiling; and
- caches approximately 10 GB of converted model artifacts in browser storage.

The first version prioritizes one complete generation over UI polish, broad browser support, full-song generation, or automated audio-quality analysis.

## Proposed architecture

The browser runtime will use ONNX Runtime Web with its WebGPU execution provider. TypeScript will control sampling, model-stage transitions, chunking, progress, cancellation, and WAV output. Python and CUDA may be used to convert the official checkpoint, but they will not be end-user runtime requirements.

Large model stages will be loaded and released separately to stay below the GPU-memory target. Custom WGSL kernels will be added only when a confirmed ONNX Runtime Web limitation blocks the end-to-end path.

See the [approved phase 1 design](docs/superpowers/specs/2026-08-20-minimax-music3-webgpu-design.md) for the full requirements and technical decisions.

For the exact Global LLM conversion and headed diagnostic commands, see [development conversion notes](docs/development/conversion.md) and the [feasibility record](docs/development/global-llm-feasibility.md).

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
