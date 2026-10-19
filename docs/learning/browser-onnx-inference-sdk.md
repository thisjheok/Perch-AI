# Feature Learning Review: 브라우저 ONNX 추론 SDK

## 1. 기능 개요

### 분석 범위와 가정

사용자가 특정 기능명을 지정하지 않았고 현재 체크아웃에 추적되는 Git 변경도 확인되지 않아, 이 저장소의 핵심 기능인 **브라우저에서 ONNX 모델을 불러와 실행하는 SDK** 전체를 대상으로 삼았다. 분석은 `src/` 구현, `src/index.test.ts`, `package.json`, `tsconfig.json`, `public/models/example/manifest.json`, 설계 문서 `docs/onnx-browser-sdk-design.md`에 한정한다. 설계 문서에 적힌 계획과 실제 코드의 동작은 별도로 표시한다.

이 기능은 모델 manifest URL을 받아 브라우저에서 모델 파일을 로드하고, ONNX Runtime Web의 WebGPU 또는 WASM provider로 추론하는 API를 제공한다. 공개 진입점은 `createLocalAI()`이며 생성 결과의 `run()`으로 추론하고 `dispose()`로 세션을 해제한다. 모델 가중치를 앱 번들과 분리하고 브라우저 캐시를 이용해 재사용하려는 목적은 설계 문서에 명시되어 있다.

## 2. 주요 컴포넌트

| 컴포넌트 | 책임과 실제 동작 |
|---|---|
| 공개 API — `src/index.ts`, `createLocalAI()` | manifest 로드, 모델 및 선택적 labels 다운로드, 세션 생성 순서를 조율하고 `modelId`, 실제 `runtime`, `run()`, `dispose()`를 반환한다. 진행률 callback은 모델 다운로드에 연결한다. |
| Manifest 로더 — `src/manifest/loader.ts`, `loadManifest()` | `fetch()`와 JSON 파싱을 수행하고 `validateManifest()`를 호출한다. manifest의 파일 상대 경로를 manifest URL 기준 절대 URL로 변환한다. |
| Manifest 검증 — `src/manifest/validator.ts`, `validateManifest()` | object 형식, schema 버전, 식별 문자열, `runtime: "onnx"`, 지원 task, files/input/output를 검사한다. |
| 파일 다운로드와 캐시 — `src/model/downloader.ts`, `downloadModel()` | Cache API에서 먼저 조회하고 miss 시 `fetch()`한다. response body를 읽으며 진행률을 알리고 완료된 데이터를 캐시에 저장한다. |
| 런타임 선택 — `src/runtime/session.ts`, `createSession()` | `wasm` 직접 선택, `webgpu` 명시 선택, `auto`의 WebGPU 세션 우선 시도를 처리한다. 자동 선택에서 WebGPU 초기화가 실패하면 WASM을 시도한다. |
| 입력 처리 — `src/pipeline/input.ts`, `prepareInput()` | manifest 입력 타입이 `tensor`인지 확인하고 입력이 객체인지 확인한 뒤 `ModelInputs`로 단언해 그대로 전달한다. 실제 tensor 이름·shape·dtype 검증이나 변환은 하지 않는다. |
| 출력 처리 — `src/pipeline/output.ts`, `processOutput()` | 첫 번째 모델 출력 tensor를 취한다. `processor: "softmax"`면 softmax 점수와 labels를 매핑하고 정렬해 `topK`개를 반환한다. 그 외에는 원래 outputs 객체를 반환한다. |
| 오류 타입 — `src/errors.ts`, `LocalAIError` | `code`가 있는 오류를 제공하여 manifest, 다운로드, cache, WebGPU, 세션 초기화, 입력 문제 등을 구분한다. |

사용 라이브러리는 `onnxruntime-web`이며 브라우저 표준 `fetch`, Cache API, `Response`, `ReadableStream`도 사용한다. `package.json`에는 `@huggingface/transformers`도 있으나, 이것은 예제 화면의 별도 텍스트 생성 경로(`src/example.ts`)에서 사용되며 `createLocalAI()` 추론 경로와는 별개다.

## 3. 실행 흐름

### 초기화: `createLocalAI(options)`

1. `src/index.ts`가 manifest URL의 빈 문자열 여부를 검사한다.
2. `loadManifest()`가 manifest를 가져온다. 네트워크 오류, 비정상 HTTP 응답, JSON 파싱 오류는 `MANIFEST_LOAD_FAILED`가 된다. 파싱된 값은 `validateManifest()`에서 검사된다.
3. `loadManifest()`가 `files` 경로를 manifest URL 기준으로 해석해 `modelUrl`과 `assetUrls`를 만든다.
4. `downloadModel(modelUrl, callback)`이 모델을 가져온다. 다운로드 진행 이벤트에는 `phase: "model"`이 추가되어 공개 callback에 전달된다.
5. `labels` asset URL이 있으면 같은 downloader로 추가 다운로드하고 JSON이 문자열 배열인지 확인한다. 형식이 잘못되었거나 JSON이 아니면 `INVALID_MANIFEST` 오류로 처리한다.
6. `createSession()`이 ONNX Runtime Web 세션을 초기화한다. `runtime` 미지정 시 `auto`다. `auto`는 `navigator.gpu` 존재를 확인한 뒤 WebGPU 초기화를 시도하고, 그 시도가 실패하면 WASM을 시도한다.
7. 초기화가 성공하면 실제 선택 provider를 `ai.runtime`에 기록한 `LocalAI` 객체를 돌려준다.

### 추론과 해제

1. `ai.run(input)`은 `prepareInput(input, manifest)`를 호출한다. 현재 구현에서 입력은 manifest 타입이 `tensor`인 object여야 하며, ONNX 입력 map 형태를 변환 없이 넘긴다.
2. `selected.session.run(...)`이 ONNX Runtime Web 세션을 실행한다.
3. 결과가 `processOutput()`으로 전달된다. softmax processor이면 확률화 후 선택적으로 labels를 입히고 상위 결과를 반환한다.
4. `ai.dispose()`는 ONNX 세션의 `release()`를 호출한다.

### 주요 실패 경로와 구현-문서 차이

- manifest 다운로드, HTTP 상태, JSON 형식, schema 필수 필드는 각각 검사되어 분류 오류로 전달된다.
- 모델 다운로드/캐시 저장 실패는 `MODEL_DOWNLOAD_FAILED` 또는 `CACHE_FAILED`로 감싼다.
- 명시적으로 `runtime: "webgpu"`를 요청했는데 `navigator.gpu`가 없으면 `WEBGPU_UNAVAILABLE`이다. provider 세션 생성 실패는 `RUNTIME_INIT_FAILED`이다.
- `auto` fallback은 **세션 생성 단계**의 WebGPU 실패에만 적용된다. 추론 도중의 WebGPU 실패를 감지해 WASM으로 재시도하는 로직은 현재 `run()`에 없다. 설계 문서의 런타임 오류 재시도 방안은 구현된 사실이 아니라 설계 문서상의 계획이다.
- 설계 문서의 SHA-256 검증은 코드에서 수행하지 않는다. manifest에는 `sha256` 타입이 있으나 검증 함수나 해시 확인 경로는 확인되지 않았다.
- 코드의 Cache API key는 모델 ID/버전이 아니라 정규화된 원 URL을 query에 담는 형태다. 따라서 설계 문서의 `id/version/file` key 설명과 다르다.
- `prepareInput()`은 tensor map 검증이나 입력 전처리를 실질적으로 하지 않는다. 따라서 설계 문서의 일반 입력 변환 약속보다 구현 범위가 좁다.
- session `run()`에서 발생한 오류를 `INFERENCE_FAILED`로 감싸는 처리는 없다. 이 코드는 출력이 없을 때와 labels 길이가 불일치할 때 해당 코드를 던진다.

## 4. 핵심 기술 및 설계 결정

### 관찰 가능한 결정

- **모델 파일을 manifest로 간접 지정한다.** manifest URL에서 파일 경로를 resolve하므로 앱 코드와 모델 파일을 분리해 배포할 수 있다. 이는 설계 문서의 CDN/독립 배포 목표와 맞는다.
- **ONNX Runtime Web의 실행 provider를 선택한다.** WebGPU는 지원 환경의 가속 경로이고 WASM은 대체 경로다. `auto`는 WebGPU 초기화에 성공해야 실제 WebGPU를 선택하며, API 존재 확인은 사전 조건일 뿐이다. 다만 capability 검사는 `navigator.gpu` 프로퍼티 존재만 검사하고 adapter/device를 따로 열어보지는 않는다. 실제 가능성 확인은 세션 초기화에 맡긴다.
- **Cache API를 downloader 내부에 둔다.** 동일 URL을 다시 읽을 때 캐시 응답의 ArrayBuffer를 반환해 네트워크 전송을 피할 수 있다. 브라우저가 캐시를 사용할 수 없는 경우(`caches` 미존재)는 네트워크로 진행한다. 저장 실패는 현재 초기화 실패로 처리한다.
- **오류에 안정적인 코드 값을 둔다.** 호출자는 메시지 문구에 의존하지 않고 `LocalAIError.code`로 분기할 수 있다.
- **출력 후처리를 manifest의 processor로 선택한다.** softmax 결과를 classification label/score/index 목록으로 구성하고, processor가 없거나 다른 경우 ONNX 출력 객체를 유지한다.

선택 이유는 설계 문서에서 확인된 목표와 대응시켜 설명했으며, 해당 선택을 한 개인의 의도까지 코드만으로 단정하지 않는다.

### 트레이드오프와 확인할 점

- Cache API는 브라우저 표준 저장소를 활용해 직접 IndexedDB 저장 계층을 작성하지 않아도 되지만, 저장 공간 정책·퇴거·브라우저별 제한은 사용자 코드가 별도로 보장하지 않는다. cache key가 원 URL 중심이므로 URL이 동일한데 파일 내용이 바뀌면 오래된 응답을 재사용할 가능성이 있다. 버전 URL 전략이나 무결성 검증 필요성을 실험할 수 있다.
- WebGPU 우선은 적합한 장치에서 이득을 기대할 수 있지만, API 노출과 모델/provider 호환성은 별개다. WASM fallback은 호환성 범위를 넓히지만 같은 성능을 보장하지 않는다.
- 현재 모델과 labels를 각각 메모리로 읽고, downloader는 스트림 chunk들을 배열에 모은 뒤 전체 버퍼를 새로 할당해 복사한다. 따라서 큰 모델에서는 다운로드 중 chunk와 최종 버퍼가 함께 존재할 수 있다.
- `topK`는 기본값 5, 하한 1을 적용하지만 출력 배열 길이보다 큰 값은 별도 제한하지 않는다. softmax에는 숫자 finite 여부나 빈 tensor 검증이 보이지 않는다.
- 설계 문서에는 입력 전처리, tensor 생성, hash 검증, 추론 중 오류 복구 등 확장 방향이 있지만 모두 현재 구현에서 제공된다고 볼 수 없다.

### 테스트가 실제로 다루는 범위

`src/index.test.ts`는 성공적인 WASM 초기화와 실행, 잘못된 manifest, 404 manifest, WebGPU 초기화 실패 후 WASM fallback, WebGPU 미지원 시 명시 요청 오류를 검증한다. ONNX Runtime은 mock 처리되어 있으므로 실제 브라우저/WebGPU/WASM에서 모델이 실행되는 통합 검증을 대신하지 않는다. 테스트 파일에서 Cache API hit/miss, 캐시 저장 실패, 다운로드 진행률, labels 후처리, malformed tensor, dispose 후 사용, 실제 모델 정확도 검증은 확인되지 않는다.

## 5. 백그라운드 조사 질문

### 질문 1 — `auto` 런타임은 어떤 조건에서 WebGPU에서 WASM으로 전환되는가?

- **질문:** `navigator.gpu`가 존재하지만 특정 모델의 WebGPU session 초기화가 실패하면 현재 SDK가 어떤 호출 순서로 WASM을 선택하며, 추론 중 오류가 나면 같은 fallback이 되는가?
- **중요한 이유:** 기능 탐지와 실제 provider 실행 가능성이 다르다는 점, 초기화 실패와 추론 실패가 서로 다른 오류 경로라는 점을 이해할 수 있다.
- **연관 코드:** `src/runtime/capability.ts:isWebGPUAvailable`, `src/runtime/session.ts:createSession`, `createWithProvider`, `src/index.ts:run`; 검증 테스트 `src/index.test.ts`의 fallback 테스트.
- **실험:** 세션 생성 mock이 WebGPU에서 reject하고 WASM에서 resolve하도록 설정해 호출 순서와 `ai.runtime`을 확인한다. 이어서 생성은 성공시키고 `session.run()`만 reject하게 만들어 자동 fallback 시도가 있는지 확인한다. 두 실험 결과를 테스트와 코드 경로에 대조한다.
- **난이도:** 중급

### 질문 2 — manifest 검증과 입력 계약은 어디까지 모델 호출을 보호하는가?

- **질문:** `validateManifest()`가 확인하는 구조적 조건과 `prepareInput()`이 실제로 확인하는 조건은 무엇이며, 이름이 틀린 tensor나 잘못된 shape/dtype는 어느 계층에서 오류가 나는가?
- **중요한 이유:** 메타데이터 schema 검증과 실제 런타임 입력 검증은 별개의 경계다. 이 경계를 알면 SDK가 오류를 사전에 설명하는지 ONNX Runtime에 위임하는지 판단할 수 있다.
- **연관 코드:** `src/manifest/validator.ts:validateManifest`, `src/manifest/schema.ts:ModelManifest`, `src/pipeline/input.ts:prepareInput`, `src/index.ts:run`.
- **실험:** 유효한 manifest에 대해 null, 배열, 빈 객체, 누락된 tensor 이름, shape가 다른 tensor를 각각 넘기고 어떤 단계에서 어떤 오류가 발생하는지 비교한다. 실행 전후 mock `session.run()` 호출 여부도 관찰한다.
- **난이도:** 기초

### 질문 3 — Cache API key와 모델 버전 변경은 캐시 일관성에 어떤 영향을 주는가?

- **질문:** downloader의 cache key는 어떤 URL로 만들어지며, 같은 URL의 파일 내용이 변경되거나 모델 버전만 바뀔 때 기존 응답이 재사용되는가?
- **중요한 이유:** 캐시는 단순 성능 최적화만이 아니라 사용자가 실행하는 모델의 버전과 무결성에 영향을 준다. 설계 문서의 버전 key와 실제 URL 기반 key의 차이를 추적하면 배포 시 캐시 stale 문제를 이해할 수 있다.
- **연관 코드:** `src/model/downloader.ts:cacheKey`, `downloadModel`, `src/manifest/loader.ts:loadManifest`, `src/manifest/schema.ts:ModelManifest`.
- **실험:** 동일 URL에 서로 다른 응답 bytes를 설정하고 첫 로드 후 두 번째 호출의 네트워크 요청 여부와 반환 bytes를 비교한다. 그 후 manifest의 버전만 바꾸되 model URL은 유지해 결과를 비교한다. 이 검증은 코드/테스트 수준이며 실제 브라우저 Cache API의 보존 정책 검증과 구분한다.
- **난이도:** 중급

### 질문 4 — softmax 후처리는 출력 tensor의 어떤 가정을 필요로 하는가?

- **질문:** `processOutput()`은 여러 출력 중 어떤 tensor를 분류 결과로 사용하며, softmax와 labels/topK 처리에서 빈 출력, labels 개수 불일치, 비정상 score, 큰 topK는 어떻게 되는가?
- **중요한 이유:** 모델 출력 순서와 shape를 암묵적으로 가정하면 모델에 따라 결과가 잘못되거나 오류가 날 수 있다. 후처리의 수학적 의미와 데이터 계약을 동시에 점검할 수 있다.
- **연관 코드:** `src/pipeline/output.ts:softmax`, `processOutput`, `src/index.ts:run`, `src/manifest/validator.ts:validateManifest`.
- **실험:** 여러 output key를 가진 결과와 0개 score, label count mismatch, topK가 score 수보다 큰 manifest를 만들어 반환값/오류를 확인한다. 간단한 logits `[0, 0]`의 반환 score가 합계 1인지도 계산으로 검증한다.
- **난이도:** 중급

### 질문 5 — 스트리밍 다운로드, 진행률, 캐시 저장 실패는 어떻게 관찰되는가?

- **질문:** response body가 있는 경우와 없는 경우 진행률 callback은 어떻게 달라지며, Content-Length가 없거나 Cache API 저장이 실패하면 초기화는 어떤 오류로 끝나는가?
- **중요한 이유:** 브라우저 네트워크 응답은 크기 정보를 항상 제공하지 않으며, 다운로드 데이터의 스트림 처리와 캐시 저장은 서로 다른 실패 지점이다. 대용량 모델 로딩 UX와 메모리 사용 문제의 출발점이다.
- **연관 코드:** `src/model/downloader.ts:readResponse`, `downloadModel`; `src/index.ts:createLocalAI`의 `onProgress` 전달.
- **실험:** ReadableStream 응답을 여러 chunk로 구성하고 Content-Length 유/무를 바꾸어 callback 기록을 확인한다. 이후 `caches.open()` 또는 `cache.put()`을 reject하도록 mock해 `CACHE_FAILED`와 후속 동작을 확인한다. 코드 테스트로 수행할 수 있는 시나리오이며 실제 브라우저의 저장 용량 한계 실험과는 별도다.
- **난이도:** 심화

## 6. 추천 학습 순서

1. **질문 1 — 런타임 선택 및 fallback:** 코드 흐름이 짧고 이미 존재하는 테스트를 출발점으로 삼아 자동 provider 선택과 실패 시 동작을 학습한다.
2. **질문 2 — manifest/input 계약:** 초기화 검증과 런타임 입력 검증의 책임 경계를 비교한다.
3. **질문 4 — softmax 및 후처리:** ONNX tensor를 앱에서 소비할 수 있는 출력으로 바꾸는 과정을 확인한다.
4. **질문 3 — 캐시 일관성:** URL 기반 캐시의 재사용 조건과 모델 업데이트 배포를 연결한다.
5. **질문 5 — 스트리밍과 실패 처리:** 브라우저 스트림, 진행률, 캐시 영속성, 메모리 비용까지 넓힌다.

## 7. 확인되지 않은 사항

- 현재 checkout에서 추적된 Git 변경은 없었고, 특정 기능 지정도 없어 전체 브라우저 ONNX 추론 SDK를 분석 대상으로 가정했다. `.gjc/`의 untracked 항목은 분석 범위가 아니며 수정하지 않았다.
- 실제 `onnxruntime-web` 및 실제 ONNX 모델을 실행하는 브라우저 통합 검증은 이 분석에서 수행하지 않았다. `src/index.test.ts`는 runtime을 mock한다.
- 실행 환경별 WebGPU adapter/device 호환성, 실제 WASM 성능, 브라우저별 Cache API 용량·퇴거 정책은 코드만으로 확정할 수 없다.
- 설계 문서의 `sha256` 필드는 현재 검증에 사용되지 않으며, 문서가 목표로 설명하는 기능 전체가 구현 완료라는 증거는 아니다.
- SDK의 입력 전처리 방식, 모델별 tensor shape 계약, 유효하지 않은 ONNX 파일의 런타임 오류 상세는 구현 및 테스트만으로 충분히 확인되지 않았다.
