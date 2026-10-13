# 브라우저 ONNX 추론 SDK 설계

## 1. 문서 목적

가벼운 머신러닝 모델을 웹 클라이언트에서 직접 추론할 수 있도록 하는 SDK의 초기 설계를 정의한다.

초기 목표는 대형 언어 모델이 아니라 다음과 같은 소형 모델이다.

- 상품 분류
- 상품 추천 점수 계산
- 텍스트 또는 이미지 임베딩
- 감정·의도 분류
- 가격·품질 예측

모델 실행은 ONNX Runtime Web을 사용하고, 가능한 환경에서는 WebGPU를 사용한다. WebGPU를 사용할 수 없거나 실행에 실패하면 WASM으로 fallback한다.

## 2. 프로젝트 목표

### 목표

1. 개발자가 ONNX 모델을 웹 애플리케이션에 쉽게 등록할 수 있게 한다.
2. 모델 다운로드, 캐시, 실행 장치 선택을 SDK가 처리한다.
3. WebGPU 우선, WASM fallback 정책을 제공한다.
4. 모델별 입력·출력 형식을 명시적으로 관리한다.
5. 모델 실행 계층과 서비스 비즈니스 로직을 분리한다.
6. 정적 파일 CDN만으로 모델을 배포할 수 있게 한다.

### 초기 범위에서 제외

- 모델 학습 및 파인튜닝
- 임의의 PyTorch·TensorFlow 모델 자동 변환
- 대형 LLM 실행
- 서버 추론 플랫폼
- 모델 가중치 보호
- 자동 모델 최적화 및 양자화 파이프라인
- 여러 모델을 조합하는 워크플로 엔진

모델은 SDK가 지원하는 ONNX 형식으로 사전에 변환되어 있어야 한다.

## 3. 사용 예시

```ts
import { createLocalAI } from "@perch-ai/browser";

const ai = await createLocalAI({
  model: {
    manifestUrl: "/models/product-classifier/manifest.json",
  },
  runtime: "auto",
  onProgress: (progress) => {
    console.log(`${progress.loaded}/${progress.total}`);
  },
});

const result = await ai.run({
  text: "거의 새것 같은 아이폰 14입니다.",
});

console.log(result);
```

개발자는 ONNX Runtime Web의 세션 생성, 실행 provider 선택, 모델 캐시, 텐서 변환을 직접 다루지 않는다.

## 4. 전체 구조

```text
웹 애플리케이션
      │
      ▼
@perch-ai/browser SDK
      │
      ├─ Public API
      ├─ Model Manifest Loader
      ├─ Model Cache
      ├─ Capability Detector
      ├─ Input Preprocessor
      ├─ Inference Session Manager
      ├─ Output Postprocessor
      └─ Runtime Adapter
           ├─ ONNX Runtime Web / WebGPU
           └─ ONNX Runtime Web / WASM
      │
      ▼
정적 모델 파일/CDN
  ├─ manifest.json
  ├─ model.onnx 또는 model.onnx.data
  ├─ tokenizer.json 등 전처리 파일
  └─ labels.json 등 출력 메타데이터
```

## 5. 핵심 설계 원칙

### 5.1 실행 장치와 모델 API 분리

사용자는 `webgpu`나 `wasm`을 직접 다루지 않고 `runtime: "auto"`를 사용한다. SDK는 환경을 검사해 적절한 실행 provider를 선택한다.

```text
runtime: auto
  → WebGPU 사용 가능 여부 확인
  → WebGPU 세션 생성
  → 실패 시 WASM 세션 생성
```

명시적인 디버깅과 테스트를 위해 `runtime: "webgpu"` 및 `runtime: "wasm"`도 제공한다.

### 5.2 모델과 애플리케이션 번들 분리

모델 가중치는 JavaScript 번들에 포함하지 않는다. SDK는 manifest URL을 통해 모델 파일을 가져온다.

장점:

- 애플리케이션 번들 크기 증가 방지
- 모델 독립 배포와 버전 관리
- 브라우저 캐시 활용
- 모델을 선택적으로 다운로드
- CDN 사용 가능

### 5.3 명시적 입력·출력 계약

모델마다 입력 텐서의 이름, shape, 타입, 전처리 방식이 다르다. 이를 SDK가 추측하지 않고 manifest에 선언한다.

### 5.4 초기에는 단일 모델·단일 추론 세션

MVP에서는 하나의 `LocalAI` 인스턴스가 하나의 모델을 관리한다. 모델 여러 개의 자동 조합, 세션 풀, 백그라운드 사전 로딩은 이후 범위로 둔다.

## 6. 모델 Manifest

모델 파일과 실행 방법을 설명하는 정적 JSON 파일이다.

```json
{
  "schemaVersion": 1,
  "id": "product-classifier",
  "version": "1.0.0",
  "task": "text-classification",
  "runtime": "onnx",
  "files": {
    "model": "model.onnx",
    "tokenizer": "tokenizer.json",
    "labels": "labels.json"
  },
  "sha256": {
    "model.onnx": "..."
  },
  "input": {
    "type": "text",
    "processor": "wordpiece",
    "maxLength": 128
  },
  "output": {
    "type": "labels",
    "processor": "softmax"
  }
}
```

### 필수 필드

- `schemaVersion`: manifest 형식 버전
- `id`: 모델 식별자
- `version`: 모델 버전
- `task`: 모델 작업 유형
- `runtime`: 현재는 `onnx`만 지원
- `files.model`: ONNX 모델 경로
- `input`: 입력 데이터 계약
- `output`: 출력 데이터 계약

### 무결성 검증

선택적으로 모델 파일의 SHA-256을 manifest에 기록한다. SDK는 다운로드 후 해시를 검증할 수 있다. 검증 실패 시 캐시 파일을 사용하지 않고 오류를 반환한다.

## 7. 실행 흐름

```text
createLocalAI()
  1. manifest 다운로드
  2. manifest schema 검증
  3. 모델 파일 URL 해석
  4. 브라우저 캐시 확인
  5. 실행 runtime 선택
  6. ONNX Runtime Web 세션 생성
  7. 모델 초기화 완료

ai.run(input)
  1. 입력 스키마 검증
  2. 전처리
  3. Tensor 생성
  4. session.run()
  5. 출력 후처리
  6. 일반 JavaScript 값 반환
```

모델 초기화와 추론은 모두 비동기 API로 제공한다.

## 8. Runtime 정책

### 기본 정책

```ts
runtime: "auto"
```

선택 순서:

1. WebGPU API 존재 여부 확인
2. WebGPU adapter/device 생성 시도
3. ONNX WebGPU session 생성 시도
4. 실패 시 WASM session 생성
5. 둘 다 실패하면 명확한 초기화 오류 반환

WebGPU API가 존재하는 것만으로 실행 가능하다고 판단하지 않는다. 실제 device와 session 생성까지 성공해야 WebGPU를 사용한다.

### WASM 설정

WASM은 가장 넓은 호환성을 위한 fallback이다. 초기에는 단일 스레드 또는 ONNX Runtime의 기본 설정으로 시작하고, 성능 측정 후 스레드 설정을 최적화한다.

### 런타임 오류

추론 중 WebGPU 오류가 발생했을 때 같은 세션을 계속 사용하지 않는다. MVP에서는 세션을 폐기하고 WASM으로 한 번 재시도할 수 있다. 동일 입력에 대한 무한 재시도는 금지한다.

## 9. 캐시 설계

초기 구현은 Cache API를 사용한다.

```text
cache key:
  perch-ai-models/{id}/{version}/{file}
```

캐시 정책:

- 모델 버전이 달라지면 새 파일로 취급한다.
- 모델 파일은 다운로드 완료 후에만 캐시에 저장한다.
- 다운로드 중단 파일은 유효한 모델로 취급하지 않는다.
- 캐시 오류는 추론 오류와 구분한다.
- 저장공간 부족 시 캐시를 지우고 재시도하지 않고 명확한 오류를 반환한다.

향후 필요하면 IndexedDB 또는 OPFS를 추가하되, 첫 버전에는 저장소를 여러 개 지원하지 않는다.

## 10. 공개 API 초안

```ts
export type Runtime = "auto" | "webgpu" | "wasm";

export interface CreateLocalAIOptions {
  model: {
    manifestUrl: string;
  };
  runtime?: Runtime;
  onProgress?: (progress: ModelDownloadProgress) => void;
}

export interface ModelDownloadProgress {
  phase: "manifest" | "model" | "asset" | "ready";
  loaded: number;
  total?: number;
  url?: string;
}

export interface LocalAI<Input = unknown, Output = unknown> {
  readonly modelId: string;
  readonly modelVersion: string;
  readonly runtime: "webgpu" | "wasm";

  run(input: Input): Promise<Output>;
  dispose(): Promise<void>;
}

export function createLocalAI<Input = unknown, Output = unknown>(
  options: CreateLocalAIOptions,
): Promise<LocalAI<Input, Output>>;
```

초기 API에는 모델 재로드, 세션 풀, 자동 배치, 스트리밍을 넣지 않는다.

## 11. 모듈 구성

```text
src/
  index.ts
  api.ts
  errors.ts
  manifest/
    schema.ts
    loader.ts
    validator.ts
  model/
    downloader.ts
    cache.ts
    assets.ts
  runtime/
    runtime.ts
    capability.ts
    onnx-session.ts
    webgpu.ts
    wasm.ts
  pipeline/
    input.ts
    output.ts
```

### 모듈 책임

- `api`: 외부 공개 API와 lifecycle 관리
- `manifest`: manifest 로드 및 검증
- `model`: 파일 다운로드, 캐시, 무결성 검증
- `runtime`: 실행 provider 선택과 ONNX Runtime 초기화
- `pipeline`: 입력 전처리와 출력 후처리
- `errors`: 사용자가 처리할 수 있는 오류 타입

## 12. 오류 모델

오류는 문자열이 아니라 분류 가능한 오류 타입으로 제공한다.

```ts
export type LocalAIErrorCode =
  | "MANIFEST_LOAD_FAILED"
  | "INVALID_MANIFEST"
  | "MODEL_DOWNLOAD_FAILED"
  | "MODEL_INTEGRITY_FAILED"
  | "WEBGPU_UNAVAILABLE"
  | "RUNTIME_INIT_FAILED"
  | "INVALID_INPUT"
  | "INFERENCE_FAILED"
  | "CACHE_FAILED";
```

개발자는 오류 코드로 UI와 fallback을 결정할 수 있어야 한다.

## 13. 보안 및 배포 고려사항

- 모델 파일을 제공하는 서버는 CORS를 허용해야 한다.
- 모델 URL은 HTTPS를 사용해야 한다.
- 모델 파일은 신뢰할 수 있는 출처에서만 로드한다.
- SHA-256 검증을 지원해 변조된 모델을 거부한다.
- 클라이언트에 배포된 모델 가중치는 비밀로 간주하지 않는다.
- 사용자 입력은 모델에 전달되기 전에 애플리케이션이 검증한다.
- 모델 실행 결과를 권한 결정이나 결제 승인 같은 보안 경계로 사용하지 않는다.

## 14. 검증 전략

### 단위 테스트

- manifest 필수 필드 검증
- 상대 경로 URL 해석
- 잘못된 버전과 task 거부
- 입력·출력 shape 변환
- 캐시 hit/miss 처리
- 진행률 이벤트
- 오류 코드 매핑

### 브라우저 통합 테스트

- WASM에서 실제 ONNX 모델 실행
- WebGPU 지원 환경에서 실제 ONNX 모델 실행
- WebGPU 초기화 실패 후 WASM fallback
- 첫 다운로드 후 두 번째 호출의 캐시 사용
- `dispose()` 후 세션 리소스 해제

### MVP 검증 모델

처음에는 작은 텍스트 분류 또는 임베딩 모델 하나를 기준 모델로 사용한다.

조건:

- 모델 파일이 작고 다운로드 시간이 짧을 것
- ONNX Runtime Web WASM에서 실행될 것
- 가능하면 WebGPU에서도 같은 결과를 낼 것
- 입력과 출력이 단순하고 검증 가능할 것

## 15. 단계별 구현 순서

### 단계 1: 실행 핵심

- TypeScript 패키지 초기화
- `onnxruntime-web` 연결
- 로컬 ONNX 파일로 WASM 추론
- `createLocalAI()`와 `run()` 구현

### 단계 2: 모델 manifest

- manifest schema 정의
- 원격 manifest와 모델 파일 로드
- 입력·출력 metadata 연결

### 단계 3: WebGPU와 fallback

- WebGPU capability 검사
- WebGPU 우선 세션 생성
- 실패 시 WASM fallback
- runtime 정보 노출

### 단계 4: 캐시와 진행률

- Cache API 도입
- 모델 버전별 key
- 다운로드 진행률 callback
- 무결성 검증

### 단계 5: 예제 애플리케이션

- 상품 분류 예제
- 상품 설명 보조용 입력 UI
- 모델 다운로드 상태 UI
- WebGPU/WASM 상태 표시

## 16. 결정 사항 요약

| 항목 | 결정 |
|---|---|
| 배포 형태 | npm 브라우저 SDK |
| 초기 모델 형식 | ONNX |
| 실행 런타임 | ONNX Runtime Web |
| 기본 장치 | WebGPU 우선 |
| 호환성 fallback | WASM |
| 모델 배포 | 앱 번들과 분리된 정적 파일/CDN |
| 모델 메타데이터 | 버전이 있는 manifest.json |
| 캐시 | 초기에는 Cache API |
| 공개 추론 API | `createLocalAI()` + `run()` |
| 초기 태스크 | 분류·임베딩·추천 계열의 소형 모델 |
| 제외 범위 | 학습, 자동 변환, 대형 LLM |

## 17. 성공 기준

MVP는 다음 조건을 만족하면 성공으로 판단한다.

1. 개발자가 npm 패키지를 설치할 수 있다.
2. manifest URL만 지정해 원격 ONNX 모델을 로드할 수 있다.
3. WebGPU 지원 환경에서는 WebGPU로 실행된다.
4. WebGPU가 없거나 실패하면 WASM으로 실행된다.
5. 두 번째 로드에서는 캐시된 모델을 사용할 수 있다.
6. 모델 입력과 출력이 일반 JavaScript 값으로 노출된다.
7. 다운로드·초기화·추론 실패 원인을 오류 코드로 확인할 수 있다.
8. 작은 실제 모델을 사용한 브라우저 통합 테스트가 통과한다.
