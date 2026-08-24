<div align="center">
  <h1>MiniMax Music 3 WebGPU</h1>
  <p><strong>브라우저 기반 MiniMax Music 3 로컬 음악 생성 데모</strong></p>

[![Model: MiniMax-Music3](https://img.shields.io/badge/Model-MiniMax--Music3-blue?logo=huggingface&logoColor=white)](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
[![Chrome ≥151](https://img.shields.io/badge/Chrome-%E2%89%A5151-4285F4?logo=googlechrome&logoColor=white)](https://www.google.com/chrome/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-brightgreen)](LICENSE)
[![Pages build](https://github.com/LI-NA/minimax-music3-webgpu/actions/workflows/pages.yml/badge.svg)](https://github.com/LI-NA/minimax-music3-webgpu/actions/workflows/pages.yml)

  <p>
    <a href="README.md">English</a> ·
    한국어
  </p>
</div>

## 프로젝트 소개

MiniMax Music 3 WebGPU는 공식 [`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3) 체크포인트를 혼합 정밀도로 양자화하여 브라우저의 WebGPU 위에서 실행하는 로컬 음악 생성 데모입니다.

한 번 모델을 다운로드하면 그 이후의 모든 생성은 브라우저 안에서만 이루어집니다. 프롬프트, 가사, 생성된 오디오는 어떤 서버로도 전송되지 않습니다.

> MiniMax Music 3 WebGPU는 MiniMax와 무관한 커뮤니티 프로젝트로, 공식적인 후원이나 인증을 받지 않았습니다.

## 시스템 요구 사항

| 항목       | 요구 사항                                                      |
| ---------- | -------------------------------------------------------------- |
| 브라우저   | 데스크톱 Chromium(Chrome 151 이상), WebGPU와 `shader-f16` 지원 |
| GPU 메모리 | 아래 표 참고                                                   |
| 저장 공간  | 모델 캐시용 약 8 GB의 브라우저 저장 공간                       |
| 운영체제   | Windows에서 테스트됨                                           |

VRAM 권장치는 16GB VRAM을 가진 GPU 한 대에서 측정한 값으로, 단순 참고용입니다.

| 물리 VRAM | 안내                           |
| --------- | ------------------------------ |
| 8 GB      | 권장하지 않음                  |
| 10 GB     | 짧은 생성(10초 내외)에 적합    |
| 12 GB     | 1분까지의 생성 워크로드에 권장 |
| 16 GB     | 5분 용량 워크로드까지 테스트됨 |

## 사용 방법

데모는 <https://li-na.github.io/minimax-music3-webgpu/>에서 사용할 수 있습니다.

1. 첫 실행 시 약 8GB 모델 파일을 다운로드 받아야 합니다. 브라우저에 캐싱하므로 매번 다운로드 받을 필요는 없습니다.
2. 이후 음악 생성을 위한 프롬프트를 입력하고, 필요한 경우 고급 설정에서 샘플링 값을 조정할 수 있습니다.
3. 곡이 생성되면 WAV로 저장되고 자동으로 재생됩니다. 생성된 곡은 브라우저 내에 저장되며 언제든지 다시 확인하거나 다운로드받을 수 있습니다.

생성된 오디오의 사용 조건은 [라이선스](#라이선스)의 모델 라이선스를 따릅니다.

## 로컬에서 실행

Node.js와 npm이 필요합니다.

```bash
npm install
npm run dev
```

개발 서버는 `artifacts/release/<release>`에 있는 변환된 릴리스를 애플리케이션과 같은 origin의 `/artifacts/<release>/` 경로로 서빙합니다. 별도의 아티팩트 서버나 origin 설정이 필요 없습니다. 변환된 릴리스를 직접 만드는 방법은 [모델 변환](#모델-변환)을 참고하세요.

- 앱은 기본적으로 `artifacts/release/music-variable` 릴리스를 로드합니다. 개발 빌드에서는 `?manifest=<url>`로 다른 릴리스나 호스팅된 미러를 지정할 수 있습니다.
- `/diagnostics.html`은 스테이지별 테스트를 진행하는 별도 진입점입니다. `?release=<name>`으로 대상 릴리스를 선택하며, 프로덕션 빌드에는 포함되지 않습니다.

## 개발

### 스크립트

| 명령                                      | 설명                                                            |
| ----------------------------------------- | --------------------------------------------------------------- |
| `npm run dev`                             | 개발 서버 실행                                                  |
| `npm run build`                           | 타입 검사 후 프로덕션 빌드                                      |
| `npm run lint` / `npm run typecheck`      | ESLint 검사 / TypeScript 타입 검사                              |
| `npm run test`                            | Vitest 단위 테스트                                              |
| `npm run test:browser`                    | Playwright 브라우저 테스트 (아래 모델 변환이 선행되어야 합니다) |
| `npm run format` / `npm run format:check` | Prettier 포맷 적용 / 검사                                       |

### 모델 변환

체크포인트 변환에는 Python 3.11 이상과 [`uv`](https://docs.astral.sh/uv/)가 필요하며, 변환 결과는 Git에서 제외되는 `artifacts/` 아래에 저장됩니다.

```bash
uv sync
uv run music3-convert download-global --artifacts-dir artifacts
uv run music3-convert build-global --artifacts-dir artifacts --layers 36
```

전체 변환 파이프라인과 스테이지별 명령은 [변환 문서](docs/development/conversion.md)를 참고하세요.

## AI 활용 안내

이 프로젝트는 AI를 활용해 개발되었습니다. AI가 어떻게 작업했는지 보려면 `docs` 아래 문서를 참고하세요. 잘못된 내용이 있거나 개선할 부분이 있다면 언제든지 이슈나 PR을 남겨주세요.

## 라이선스

이 저장소의 코드는 [MIT](LICENSE) 라이선스를 따릅니다.

MiniMax Music 3 모델은 [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE)로 배포되는 오픈 웨이트 모델이며, OSI 오픈소스 라이선스가 아닙니다. 변환된 브라우저 아티팩트에도 해당 라이선스와 고지가 유지됩니다. 데모 사용과 생성된 오디오에는 라이선스에 포함된 Acceptable Use Policy를 비롯한 라이선스 조건이 적용되며, 생성된 오디오를 공개적으로 배포할 때는 AI로 생성되었음을 명확히 밝혀야 합니다.

MiniMax Music 3와 관련 자산의 권리는 MiniMax에 있습니다.
