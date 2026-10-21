import * as ort from "onnxruntime-web";
import "./benchmark.css";
import { createLocalAI, type LocalAI } from "./index";
import { imageToTensor, type ImagePreprocessingTimings } from "./example-image";

const modelVariants = [
  { variant: "fp32", manifestUrl: "/models/mobilenetv2/manifest.json" },
  { variant: "fp16", manifestUrl: "/models/mobilenetv2/manifest-fp16.json" },
  { variant: "int8", manifestUrl: "/models/mobilenetv2/manifest-int8.json" },
] as const;
const cacheName = "perch-ai-models-v1";
const warmupRuns = 3;
const scoreTolerance = 0.01;
const inputFile = document.querySelector<HTMLInputElement>("#image-file")!;
const profileFiles = document.querySelector<HTMLInputElement>("#profile-files")!;
const iterationsInput = document.querySelector<HTMLInputElement>("#iterations")!;
const trialsInput = document.querySelector<HTMLInputElement>("#trials")!;
const runButton = document.querySelector<HTMLButtonElement>("#run-benchmark")!;
const profileButton = document.querySelector<HTMLButtonElement>("#profile-preprocessing")!;
const status = document.querySelector<HTMLParagraphElement>("#status")!;
const results = document.querySelector<HTMLPreElement>("#results")!;
const profileStatus = document.querySelector<HTMLParagraphElement>("#profile-status")!;
const profileResults = document.querySelector<HTMLPreElement>("#profile-results")!;
const downloadButton = document.querySelector<HTMLButtonElement>("#download-report")!;

interface ClassificationResult {
  label: string;
  index: number;
  score: number;
}

interface RuntimeResult {
  runtime: "wasm" | "webgpu";
  modelId?: string;
  coldInitializationMs?: number;
  warmInitializationMs?: number;
  trialInitializationMs: number[];
  inferenceMs?: { p50: number; p95: number; min: number; max: number };
  successfulTrials: number;
  topResults?: ClassificationResult[];
  errors: string[];
  inferenceSamples?: number[];
}

interface BenchmarkReport {
  startedAt: string;
  preprocessingMs: number;
  preprocessingStagesMs: ImagePreprocessingTimings;
  input: { name: string; bytes: number; dimensions: number[] };
  cache: {
    sdkCacheAvailable: boolean;
    coldCacheResetCount: number;
    note: string;
  };
  settings: { warmupRuns: number; iterationsPerTrial: number; trials: number; scoreTolerance: number };
  environment: {
    userAgent: string;
    platform: string;
    hardwareConcurrency: number | null;
    deviceMemoryGiB: number | null;
    viewport: { width: number; height: number };
  };
  models: ModelVariantResult[];
  unavailableVariants: string[];
  comparison: { comparisons: number; passed: boolean | null };
}

interface ModelAssets {
  modelId: string;
  assetUrls: string[];
  modelBytes: number | null;
}

interface OutputComparison {
  top1Agrees: boolean;
  topKIndicesAgree: boolean;
  matchedTopKCount: number;
  maxAbsoluteScoreDifference: number | null;
  scoreTolerance: number;
  passed: boolean;
}

interface ModelVariantResult {
  variant: string;
  manifestUrl: string;
  modelId?: string;
  modelBytes: number | null;
  sizeReductionPercent?: number | null;
  runtimes: RuntimeResult[];
  runtimeParity?: OutputComparison;
  comparisonToFp32?: Partial<Record<"wasm" | "webgpu", OutputComparison>>;
  measurementOrder: { coldInitialization: string[]; warmInitialization: string[]; trials: string[][] };
}

let selectedFile: File | undefined;
let latestReport: BenchmarkReport | undefined;

profileButton.addEventListener("click", async () => {
  const files = Array.from(profileFiles.files ?? []);
  if (files.length === 0) {
    profileStatus.textContent = "전처리 프로파일링할 이미지들을 선택하세요.";
    return;
  }

  profileButton.disabled = true;
  profileResults.textContent = "";
  try {
    const samples: ImagePreprocessingTimings[] = [];
    for (let index = 0; index < files.length; index++) {
      profileStatus.textContent = `이미지 전처리 측정 중 (${index + 1}/${files.length})...`;
      await nextFrame();
      await imageToTensor(files[index], (timings) => samples.push(timings));
    }
    const stages = Object.fromEntries(
      (Object.keys(samples[0]) as Array<keyof ImagePreprocessingTimings>).map((stage) => [
        stage,
        summarize(samples.map((sample) => sample[stage])),
      ]),
    );
    profileResults.textContent = JSON.stringify({ images: files.length, stages }, null, 2);
    profileStatus.textContent = `완료 — ${files.length}개 이미지, 단계별 p50/p95 기록`;
  } catch (error) {
    profileStatus.textContent = errorMessage(error);
  } finally {
    profileButton.disabled = false;
  }
});

inputFile.addEventListener("change", () => {
  selectedFile = inputFile.files?.[0];
  runButton.disabled = !selectedFile;
  status.textContent = selectedFile ? `${selectedFile.name} 선택됨` : "이미지를 선택하세요.";
});

downloadButton.addEventListener("click", () => {
  if (!latestReport) return;
  const blob = new Blob([JSON.stringify(latestReport, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "perch-browser-benchmark.json";
  link.click();
  URL.revokeObjectURL(url);
});

runButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  const iterations = Number(iterationsInput.value);
  const trialCount = Number(trialsInput.value);
  if (!Number.isInteger(iterations) || iterations < 5 || iterations > 100) {
    status.textContent = "반복 횟수는 5~100 사이의 정수여야 합니다.";
    return;
  }
  if (!Number.isInteger(trialCount) || trialCount < 2 || trialCount > 5) {
    status.textContent = "측정 라운드는 2~5 사이의 정수여야 합니다.";
    return;
  }

  runButton.disabled = true;
  downloadButton.hidden = true;
  results.textContent = "";
  latestReport = undefined;
  const startedAt = new Date().toISOString();
  try {
    status.textContent = "이미지 전처리 중...";
    let preprocessingStagesMs: ImagePreprocessingTimings | undefined;
    const tensor = await imageToTensor(selectedFile, (timings) => {
      preprocessingStagesMs = timings;
    });
    if (!preprocessingStagesMs) throw new Error("Image preprocessing timings were not recorded.");
    const preprocessingMs = preprocessingStagesMs.totalMs;
    const environment = getEnvironment();
    const models: ModelVariantResult[] = [];
    const unavailableVariants: string[] = [];
    let coldCacheResetCount = 0;

    for (const variant of modelVariants) {
      const assets = await getModelAssets(variant.manifestUrl);
      if (!assets) {
        unavailableVariants.push(variant.variant);
        continue;
      }
      const result = await benchmarkVariant(variant, assets, tensor, iterations, trialCount);
      coldCacheResetCount += result.coldCacheResetCount;
      models.push(result.model);
    }

    const baseline = models.find((model) => model.variant === "fp32");
    for (const model of models) {
      model.runtimeParity = compareRuntimeResults(model.runtimes);
      if (!baseline || model.variant === "fp32") continue;
      model.sizeReductionPercent = baseline.modelBytes && model.modelBytes !== null
        ? round((1 - model.modelBytes / baseline.modelBytes) * 100)
        : null;
      const comparisons: ModelVariantResult["comparisonToFp32"] = {};
      for (const runtime of ["wasm", "webgpu"] as const) {
        const baselineResults = baseline.runtimes.find((result) => result.runtime === runtime);
        const variantResults = model.runtimes.find((result) => result.runtime === runtime);
        const comparison = compareOutputs(baselineResults?.topResults, variantResults?.topResults);
        if (comparison) comparisons[runtime] = comparison;
      }
      model.comparisonToFp32 = comparisons;
    }
    const comparisons = models.flatMap((model) => [
      model.runtimeParity,
      ...Object.values(model.comparisonToFp32 ?? {}),
    ]).filter((comparison): comparison is OutputComparison => comparison !== undefined);
    const comparisonPassed = comparisons.length > 0
      ? comparisons.every((comparison) => comparison.passed)
      : null;

    const report: BenchmarkReport = {
      startedAt,
      preprocessingMs,
      preprocessingStagesMs,
      input: {
        name: selectedFile.name,
        bytes: selectedFile.size,
        dimensions: [1, 3, 224, 224],
      },
      cache: {
        sdkCacheAvailable: typeof caches !== "undefined",
        coldCacheResetCount,
        note: typeof caches !== "undefined"
          ? "SDK Cache API entries for each tested model were cleared before cold initialization where reset succeeded; browser HTTP cache is unchanged."
          : "Cache API is unavailable, so SDK cache cold/warm states cannot be controlled.",
      },
      settings: {
        warmupRuns,
        iterationsPerTrial: iterations,
        trials: trialCount,
        scoreTolerance,
      },
      environment,
      models,
      unavailableVariants,
      comparison: { comparisons: comparisons.length, passed: comparisonPassed },
    };
    latestReport = report;
    results.textContent = JSON.stringify(report, null, 2);
    downloadButton.hidden = false;
    status.textContent = report.comparison.passed === null
      ? "완료 — 비교할 출력이 없습니다."
      : `완료 — 출력 비교 ${report.comparison.passed ? "통과" : "불일치"}; 누락 variant: ${unavailableVariants.join(", ") || "없음"}`;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "벤치마크에 실패했습니다.";
  } finally {
    runButton.disabled = !selectedFile;
  }
});

interface InitializedRuntime {
  ai: LocalAI;
  initializationMs: number;
}

interface InferenceMeasurement {
  initializationMs?: number;
  durations: number[];
  modelId?: string;
  topResults?: ClassificationResult[];
  error?: string;
}

async function benchmarkVariant(
  variant: (typeof modelVariants)[number],
  assets: ModelAssets,
  tensor: ort.Tensor,
  iterations: number,
  trialCount: number,
): Promise<{ model: ModelVariantResult; coldCacheResetCount: number }> {
  const runtimes = ["wasm", "webgpu"] as const;
  const runtimeResults = runtimes.map((runtime): RuntimeResult => ({
    runtime,
    trialInitializationMs: [],
    successfulTrials: 0,
    errors: [],
    inferenceSamples: [],
  }));
  const firstRuntime: readonly (typeof runtimes[number])[] = Math.random() < 0.5
    ? runtimes
    : [...runtimes].reverse();
  const coldOrder: string[] = [];
  const warmOrder: string[] = [];
  const trialOrders: string[][] = [];
  let coldCacheResetCount = 0;

  for (const runtime of firstRuntime) {
    status.textContent = `${variant.variant.toUpperCase()} ${runtime} cold-cache 초기화 측정 중...`;
    await nextFrame();
    coldOrder.push(runtime);
    const result = runtimeResults.find((item) => item.runtime === runtime)!;
    try {
      await clearModelCacheEntries(assets.assetUrls);
      if (typeof caches !== "undefined") coldCacheResetCount++;
      const initialized = await initializeRuntime(runtime, variant.manifestUrl);
      result.coldInitializationMs = round(initialized.initializationMs);
      result.modelId = initialized.ai.modelId;
      await initialized.ai.dispose();
    } catch (error) {
      result.errors.push(`Cold initialization: ${errorMessage(error)}`);
    }
  }

  for (const runtime of [...firstRuntime].reverse()) {
    status.textContent = `${variant.variant.toUpperCase()} ${runtime} warm-cache 초기화 측정 중...`;
    await nextFrame();
    warmOrder.push(runtime);
    const result = runtimeResults.find((item) => item.runtime === runtime)!;
    try {
      const initialized = await initializeRuntime(runtime, variant.manifestUrl);
      result.warmInitializationMs = round(initialized.initializationMs);
      result.modelId = initialized.ai.modelId;
      await initialized.ai.dispose();
    } catch (error) {
      result.errors.push(`Warm initialization: ${errorMessage(error)}`);
    }
  }

  for (let trial = 0; trial < trialCount; trial++) {
    const order = trial % 2 === 0 ? firstRuntime : [...firstRuntime].reverse();
    trialOrders.push([...order]);
    for (const runtime of order) {
      status.textContent = `${variant.variant.toUpperCase()} ${runtime} 추론 측정 중 (${trial + 1}/${trialCount})...`;
      await nextFrame();
      const result = runtimeResults.find((item) => item.runtime === runtime)!;
      const measurement = await measureInference(runtime, tensor, iterations, variant.manifestUrl);
      if (measurement.initializationMs !== undefined) {
        result.trialInitializationMs.push(round(measurement.initializationMs));
      }
      if (measurement.error) {
        result.errors.push(`Trial ${trial + 1}: ${measurement.error}`);
        continue;
      }
      result.successfulTrials++;
      result.topResults = measurement.topResults;
      result.inferenceSamples!.push(...measurement.durations);
      result.modelId = measurement.modelId;
    }
  }

  for (const result of runtimeResults) {
    if (result.inferenceSamples?.length) result.inferenceMs = summarize(result.inferenceSamples);
  }
  const model: ModelVariantResult = {
    variant: variant.variant,
    manifestUrl: variant.manifestUrl,
    modelId: runtimeResults.find((result) => result.modelId)?.modelId ?? assets.modelId,
    modelBytes: assets.modelBytes,
    runtimes: runtimeResults.map(({ inferenceSamples: _samples, ...result }) => result),
    measurementOrder: { coldInitialization: coldOrder, warmInitialization: warmOrder, trials: trialOrders },
  };
  return { model, coldCacheResetCount };
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function initializeRuntime(
  runtime: "wasm" | "webgpu",
  manifestUrl: string,
): Promise<InitializedRuntime> {
  const started = performance.now();
  const ai = await createLocalAI({ model: { manifestUrl }, runtime });
  return { ai, initializationMs: performance.now() - started };
}

async function measureInference(
  runtime: "wasm" | "webgpu",
  tensor: ort.Tensor,
  iterations: number,
  manifestUrl: string,
): Promise<InferenceMeasurement> {
  let ai: LocalAI | undefined;
  let initializationMs: number | undefined;
  try {
    const initialized = await initializeRuntime(runtime, manifestUrl);
    ai = initialized.ai;
    initializationMs = initialized.initializationMs;

    for (let index = 0; index < warmupRuns; index++) {
      await ai.run({ input: tensor });
    }

    const durations: number[] = [];
    let finalOutput: unknown;
    for (let index = 0; index < iterations; index++) {
      const started = performance.now();
      finalOutput = await ai.run({ input: tensor });
      durations.push(performance.now() - started);
    }

    return {
      initializationMs,
      modelId: ai.modelId,
      durations,
      topResults: toClassificationResults(finalOutput),
    };
  } catch (error) {
    return {
      initializationMs,
      durations: [],
      error: errorMessage(error),
    };
  } finally {
    await ai?.dispose();
  }
}

async function getModelAssets(manifestUrl: string): Promise<ModelAssets | undefined> {
  const response = await fetch(manifestUrl);
  if (response.status === 404) return undefined;
  if (!response.ok) throw new Error(`Could not load model manifest (${response.status}).`);
  if (!response.headers.get("content-type")?.includes("json")) return undefined;
  const manifest: unknown = await response.json();
  if (typeof manifest !== "object" || manifest === null || !("files" in manifest) || !("id" in manifest)) {
    throw new Error("Model manifest does not define an id and files.");
  }
  const record = manifest as { id: unknown; files: unknown };
  if (typeof record.id !== "string" || record.id.trim() === "") {
    throw new Error("Model manifest id is invalid.");
  }
  const files = record.files;
  if (typeof files !== "object" || files === null || Array.isArray(files)) {
    throw new Error("Model manifest files must be an object.");
  }
  const fileMap = files as Record<string, unknown>;
  const paths = Object.values(fileMap);
  if (typeof fileMap.model !== "string" || paths.length === 0 || !paths.every((path) => typeof path === "string")) {
    throw new Error("Model manifest files must contain string paths.");
  }
  const baseUrl = new URL(manifestUrl, location.href);
  const modelUrl = new URL(fileMap.model, baseUrl).href;
  let modelBytes: number | null = null;
  try {
    const head = await fetch(modelUrl, { method: "HEAD" });
    const contentLength = head.headers.get("content-length");
    if (head.ok && contentLength && Number.isFinite(Number(contentLength))) {
      modelBytes = Number(contentLength);
    }
  } catch {
    // Model size is supplementary; some static hosts do not support HEAD requests.
  }
  return {
    modelId: record.id,
    assetUrls: paths.map((path) => new URL(path, baseUrl).href),
    modelBytes,
  };
}

async function clearModelCacheEntries(assetUrls: string[]): Promise<void> {
  if (typeof caches === "undefined") return;
  const cache = await caches.open(cacheName);
  await Promise.all(assetUrls.map((url) => {
    const key = `https://perch-ai.invalid/model?url=${encodeURIComponent(url)}`;
    return cache.delete(key);
  }));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function toClassificationResults(value: unknown): ClassificationResult[] {
  if (!Array.isArray(value) || !value.every(isClassificationResult)) {
    throw new Error("The benchmark model did not return classification results.");
  }
  return value;
}

function isClassificationResult(value: unknown): value is ClassificationResult {
  if (typeof value !== "object" || value === null) return false;
  const item = value as Partial<ClassificationResult>;
  return typeof item.label === "string"
    && Number.isInteger(item.index)
    && typeof item.score === "number"
    && Number.isFinite(item.score);
}

function summarize(values: number[]): RuntimeResult["inferenceMs"] {
  const sorted = [...values].sort((left, right) => left - right);
  return {
    p50: round(percentile(sorted, 0.5)),
    p95: round(percentile(sorted, 0.95)),
    min: round(sorted[0]),
    max: round(sorted[sorted.length - 1]),
  };
}

function percentile(sorted: number[], quantile: number): number {
  const position = (sorted.length - 1) * quantile;
  const lower = Math.floor(position);
  const fraction = position - lower;
  return sorted[lower] + (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) * fraction;
}

function compareRuntimeResults(results: RuntimeResult[]): OutputComparison | undefined {
  const wasm = results.find((result) => result.runtime === "wasm" && result.successfulTrials > 0);
  const webgpu = results.find((result) => result.runtime === "webgpu" && result.successfulTrials > 0);
  return compareOutputs(wasm?.topResults, webgpu?.topResults);
}

function compareOutputs(
  left: ClassificationResult[] | undefined,
  right: ClassificationResult[] | undefined,
): OutputComparison | undefined {
  if (!left || !right) return undefined;
  const leftByIndex = new Map(left.map((item) => [item.index, item]));
  const rightByIndex = new Map(right.map((item) => [item.index, item]));
  const sameIndices = leftByIndex.size === rightByIndex.size
    && [...leftByIndex.keys()].every((index) => rightByIndex.has(index));
  const sharedIndices = [...leftByIndex].filter(([index]) => rightByIndex.has(index));
  const maxDifference = sharedIndices.length > 0
    ? Math.max(...sharedIndices.map(([index, item]) => Math.abs(item.score - rightByIndex.get(index)!.score)))
    : null;
  const top1Agrees = left[0]?.index === right[0]?.index;

  return {
    top1Agrees,
    topKIndicesAgree: sameIndices,
    matchedTopKCount: sharedIndices.length,
    maxAbsoluteScoreDifference: maxDifference === null ? null : round(maxDifference),
    scoreTolerance,
    passed: top1Agrees && maxDifference !== null && maxDifference <= scoreTolerance,
  };
}

function getEnvironment(): BenchmarkReport["environment"] {
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    hardwareConcurrency: navigator.hardwareConcurrency || null,
    deviceMemoryGiB: nav.deviceMemory ?? null,
    viewport: { width: window.innerWidth, height: window.innerHeight },
  };
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
