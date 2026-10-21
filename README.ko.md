<p align="center">
  <img src="icon.png" alt="Perch 아이콘" width="160">
</p>

[English](README.md) | [한국어](README.ko.md)

# Perch 브라우저 ONNX SDK

**Perch**는 가벼운 ONNX 모델을 브라우저에서 불러와 실행하는 SDK입니다. ONNX Runtime Web을 사용하며, 가능한 환경에서는 WebGPU를 우선하고 더 넓은 호환성을 위해 WASM도 지원합니다. 모델 파일은 애플리케이션 번들과 분리해 JSON manifest를 통해 불러옵니다.

> 이 저장소는 초기 단계(`0.1.0`)입니다. 실제 제품에 적용하기 전에 대상 모델과 브라우저에서 API 및 런타임 동작을 확인하세요.

## 주요 기능

- manifest URL을 통해 모델과 관련 asset 불러오기
- `auto`, `webgpu`, `wasm` 런타임 선택
- 모델 다운로드 진행률 제공
- 브라우저 Cache API를 이용한 모델 다운로드 캐시
- ONNX tensor 입력 실행 및 softmax·labels를 이용한 선택적 분류 결과 처리

## 요구 사항

- Node.js 및 npm
- 최신 브라우저
- 입력과 출력 동작을 manifest에 기술한 ONNX 변환 모델

## 설치 및 예제 실행

```sh
npm install
npm run dev
```

Vite가 출력하는 로컬 주소를 브라우저에서 여세요. 예제 페이지에서는 포함된 MobileNetV2 모델을 이용한 이미지 분류와 Transformers.js를 사용하는 별도의 텍스트 생성 데모를 제공합니다. 텍스트 생성 경로는 예제 페이지 기능이며 `createLocalAI()` API에는 포함되지 않습니다.

브라우저에서 `/benchmark.html`을 열면 같은 이미지 tensor와 MobileNetV2 모델로 WASM/WebGPU를 비교할 수 있습니다. 모델 초기화 시간과 warm-up 후 추론 p50/p95를 따로 기록하고, 분류 결과의 top-1 일치 여부와 top-K 점수 차이를 검사합니다. 초기화 시간은 다운로드 또는 브라우저 캐시 상태의 영향을 받으므로 결과 JSON의 실행 환경과 함께 해석하세요. 이 검사는 정답 레이블 기반 정확도 평가가 아니라 런타임 간 출력 일치 검사입니다.

## SDK 사용법

```ts
import { createLocalAI } from "@perch-ai/browser";
import { Tensor } from "onnxruntime-web";

const ai = await createLocalAI({
  model: { manifestUrl: "/models/mobilenetv2/manifest.json" },
  runtime: "auto",
  onProgress: ({ loaded, total, phase }) => {
    console.log(`${phase}: ${loaded} / ${total ?? "unknown"} bytes`);
  },
});

try {
  const input = new Tensor("float32", inputData, inputDimensions);
  const result = await ai.run({ input });
  console.log(result);
} finally {
  await ai.dispose();
}
```

ONNX 모델이 요구하는 입력 이름과 차원에 맞는 tensor를 전달해야 합니다. 크기 조정이나 정규화 같은 전처리는 애플리케이션에서 수행해야 합니다. 이미지 전처리 구현은 저장소의 브라우저 예제에서 확인할 수 있습니다.

`createLocalAI()`는 `modelId`, 선택된 `runtime`, `run(input)`, `dispose()`를 반환합니다. 런타임은 `auto`(세션 생성 시 WebGPU를 시도한 뒤 실패하면 WASM), `webgpu`, `wasm` 중 선택할 수 있습니다. `webgpu`를 명시하려면 브라우저가 이를 지원해야 합니다. 현재 자동 fallback은 세션 초기화 단계에서만 동작하며 추론 실패 후에는 동작하지 않습니다.

## 모델 manifest

manifest는 ONNX 모델과 선택적 asset을 지정합니다. 예시는 `public/models/mobilenetv2/manifest.json`입니다.

```json
{
  "schemaVersion": 1,
  "id": "mobilenetv2-imagenet",
  "version": "12.0.0",
  "task": "image-classification",
  "runtime": "onnx",
  "files": {
    "model": "model.onnx",
    "labels": "labels.json"
  },
  "input": { "type": "tensor" },
  "output": {
    "type": "labels",
    "processor": "softmax",
    "topK": 5
  }
}
```

파일 경로는 manifest URL을 기준으로 해석됩니다. 예제 파일은 `public/models/`에 있으며, 개발 중에는 Vite가 `/models/` 경로로 제공합니다. 입력 tensor는 모델이 기대하는 ONNX 입력과 일치해야 합니다. `softmax` 출력 processor를 설정하면 SDK가 점수에 선택적 문자열 배열 labels를 매핑하고 설정된 개수만큼 상위 결과를 반환합니다.

## 개발 명령

```sh
npm run dev        # Vite 개발 서버 실행
npm run build      # 타입 검사 및 JavaScript/선언 파일 빌드
npm test           # 테스트 한 번 실행
npm run test:watch # 테스트 watch 모드 실행
```

## 프로젝트 구조

- `src/index.ts` — SDK 공개 API
- `src/manifest/` — manifest 로드 및 검증
- `src/model/` — 모델 다운로드와 캐시 처리
- `src/runtime/` — ONNX Runtime Web 세션 선택
- `src/pipeline/` — 입력 검사와 출력 처리
- `src/example.ts` — 브라우저 인터랙티브 예제
- `src/benchmark.ts` — 브라우저 WASM/WebGPU 벤치마크
- `public/models/` — 예제 manifest, 모델 파일, labels
- `docs/` — 설계 및 구현 참고 문서

## 현재 범위와 제한 사항

이 프로젝트는 미리 ONNX로 변환된 모델을 실행하며 모델 학습이나 변환은 수행하지 않습니다. SDK는 호출자가 적절한 이름의 ONNX tensor를 전달한다고 가정합니다. 현재 이미지 전처리를 수행하거나 모델 입력과 tensor 차원을 대조 검증하지 않습니다. 브라우저별 런타임 지원 여부와 모델 호환성은 다를 수 있습니다. 구현 세부 사항과 제한 사항은 `docs/`의 설계 및 학습 문서를 참고하세요.

## 라이선스

MIT (`package.json`의 `license` 항목 참고).
