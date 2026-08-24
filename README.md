<div align="center">
  <h1>MiniMax Music 3 WebGPU</h1>
  <p><strong>A browser-based local music generation demo for MiniMax Music 3</strong></p>

[![Model: MiniMax-Music3](https://img.shields.io/badge/Model-MiniMax--Music3-blue?logo=huggingface&logoColor=white)](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
[![Chrome ≥151](https://img.shields.io/badge/Chrome-%E2%89%A5151-4285F4?logo=googlechrome&logoColor=white)](https://www.google.com/chrome/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-brightgreen)](LICENSE)
[![Pages build](https://github.com/LI-NA/minimax-music3-webgpu/actions/workflows/pages.yml/badge.svg)](https://github.com/LI-NA/minimax-music3-webgpu/actions/workflows/pages.yml)

  <p>
    English ·
    <a href="README.ko.md">한국어</a>
  </p>
</div>

## About

MiniMax Music 3 WebGPU is a local music generation demo that runs the official [`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3) checkpoint, quantized to mixed precision, on WebGPU in your browser.

After the one-time model download, every generation happens entirely inside the browser. Prompts, lyrics, and generated audio are never sent to any server.

> MiniMax Music 3 WebGPU is a community project unaffiliated with MiniMax and is neither sponsored nor endorsed by them.

## System requirements

| Item             | Requirement                                                                 |
| ---------------- | --------------------------------------------------------------------------- |
| Browser          | Desktop Chromium (Chrome 151 or later) with WebGPU and `shader-f16` support |
| GPU memory       | See the table below                                                         |
| Storage          | About 8 GB of browser storage for the model cache                           |
| Operating system | Tested on Windows                                                           |

The VRAM guidance was measured on a single GPU with 16 GB of VRAM and is for reference only.

| Physical VRAM | Guidance                                           |
| ------------- | -------------------------------------------------- |
| 8 GB          | Not recommended                                    |
| 10 GB         | Suitable for short generations (around 10 seconds) |
| 12 GB         | Recommended for workloads up to one minute         |
| 16 GB         | Tested up to the five-minute capacity workload     |

## How to use

The demo is available at <https://li-na.github.io/minimax-music3-webgpu/>.

1. On first run you need to download about 8 GB of model files. They are cached in the browser, so you do not have to download them again.
2. Then enter a prompt describing the music, and adjust the sampling values in the advanced controls if needed.
3. Once a track is generated it is saved as WAV and starts playing automatically. Generated tracks are stored in the browser, so you can revisit or download them at any time.

Use of the generated audio is subject to the model license in [License](#license).

## Running locally

Node.js and npm are required.

```bash
npm install
npm run dev
```

The dev server serves converted releases from `artifacts/release/<release>` at `/artifacts/<release>/`, same-origin with the application. No separate artifact server or origin configuration is needed. See [Model conversion](#model-conversion) for how to build a converted release yourself.

- The app loads the `artifacts/release/music-variable` release by default. In development builds, `?manifest=<url>` points it at a different release or a hosted mirror.
- `/diagnostics.html` is a separate entry point that runs per-stage tests. Its target release is selected with `?release=<name>`, and the page is not part of the production build.

## Development

### Scripts

| Command                                   | Description                                                    |
| ----------------------------------------- | -------------------------------------------------------------- |
| `npm run dev`                             | Start the dev server                                           |
| `npm run build`                           | Type-check, then build for production                          |
| `npm run lint` / `npm run typecheck`      | ESLint check / TypeScript type check                           |
| `npm run test`                            | Vitest unit tests                                              |
| `npm run test:browser`                    | Playwright browser tests (requires the model conversion below) |
| `npm run format` / `npm run format:check` | Apply / check Prettier formatting                              |

### Model conversion

Converting the checkpoint requires Python 3.11 or later and [`uv`](https://docs.astral.sh/uv/). The results are stored under `artifacts/`, which is excluded from Git.

```bash
uv sync
uv run music3-convert download-global --artifacts-dir artifacts
uv run music3-convert build-global --artifacts-dir artifacts --layers 36
```

See the [conversion notes](docs/development/conversion.md) for the full conversion pipeline and per-stage commands.

## AI involvement

This project was developed with the help of AI. To see how the AI worked, check the documents under `docs`. If anything is wrong or could be improved, please open an issue or a PR anytime.

## License

The code in this repository is licensed under [MIT](LICENSE).

The MiniMax Music 3 model is distributed as open weights under the [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE), which is not an OSI open-source license. The converted browser artifacts retain that license and its notices. Use of the demo and of generated audio is subject to the license terms, including its Acceptable Use Policy, and generated audio distributed publicly must be clearly disclosed as AI-generated.

MiniMax Music 3 and related assets are the property of MiniMax.
