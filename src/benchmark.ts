import * as ort from "onnxruntime-web";
import "./benchmark.css";
import { createLocalAI, type LocalAI } from "./index";
import { imageToTensor } from "./example-image";

const manifestUrl = "/models/mobilenetv2/manifest.json";
const cacheName = "perch-ai-models-v1";
const warmupRuns = 3;
const scoreTolerance = 0.01;
const inputFile = document.querySelector<HTMLInputElement>("#image-file")!;
const iterationsInput = document.querySelector<HTMLInputElement>("#iterations")!;
const trialsInput = document.querySelector<HTMLInputElement>("#trials")!;
const runButton = document.querySelector<HTMLButtonElement>("#run-benchmark")!;
const status = document.querySelector<HTMLParagraphElement>("#status")!;
const results = document.querySelector<HTMLPreElement>("#results")!;
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
  manifestUrl: string;
  preprocessingMs: number;
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
  models: Array<{ modelId: string; runtime: string }>;
  measurementOrder: { coldInitialization: string[]; warmInitialization: string[]; trials: string[][] };
  results: RuntimeResult[];
  comparison?: {
    top1Agrees: boolean;
    maxAbsoluteScoreDifference: number | null;
    scoreTolerance: number;
    passed: boolean;
  };
}

let selectedFile: File | undefined;
let latestReport: BenchmarkReport | undefined;

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
    const preprocessingStarted = performance.now();
    const tensor = await imageToTensor(selectedFile);
    const preprocessingMs = round(performance.now() - preprocessingStarted);
    const assetUrls = await getModelAssetUrls();
    const environment = getEnvironment();
    const runtimes = ["wasm", "webgpu"] as const;
    const runtimeResults = runtimes.map((runtime): RuntimeResult => ({
      runtime,
      trialInitializationMs: [],
      successfulTrials: 0,
      errors: [],
      inferenceSamples: [],
    }));
    const models: BenchmarkReport["models"] = [];
    const firstRuntime: readonly (typeof runtimes[number])[] = Math.random() < 0.5
      ? runtimes
      : [...runtimes].reverse();
    const coldOrder: string[] = [];
    const warmOrder: string[] = [];
    const trialOrders: string[][] = [];
    let coldCacheResetCount = 0;

    for (const runtime of firstRuntime) {
      status.textContent = `${runtime} cold-cache 초기화 측정 중...`;
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      coldOrder.push(runtime);
      const result = runtimeResults.find((item) => item.runtime === runtime)!;
      try {
        await clearModelCacheEntries(assetUrls);
        if (typeof caches !== "undefined") coldCacheResetCount++;
        const initialized = await initializeRuntime(runtime);
        result.coldInitializationMs = round(initialized.initializationMs);
        result.modelId = initialized.ai.modelId;
        await initialized.ai.dispose();
      } catch (error) {
        result.errors.push(`Cold initialization: ${errorMessage(error)}`);
      }
    }

    for (const runtime of [...firstRuntime].reverse()) {
      status.textContent = `${runtime} warm-cache 초기화 측정 중...`;
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      warmOrder.push(runtime);
      const result = runtimeResults.find((item) => item.runtime === runtime)!;
      try {
        const initialized = await initializeRuntime(runtime);
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
        status.textContent = `${runtime} 추론 측정 중 (${trial + 1}/${trialCount})...`;
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        const result = runtimeResults.find((item) => item.runtime === runtime)!;
        const measurement = await measureInference(runtime, tensor, iterations);
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
      if (result.inferenceSamples?.length) {
        result.inferenceMs = summarize(result.inferenceSamples);
      }
      if (result.modelId) models.push({ modelId: result.modelId, runtime: result.runtime });
    }
    const reportResults = runtimeResults.map(({ inferenceSamples: _samples, ...result }) => result);

    const report: BenchmarkReport = {
      startedAt,
      manifestUrl,
      preprocessingMs,
      input: {
        name: selectedFile.name,
        bytes: selectedFile.size,
        dimensions: [1, 3, 224, 224],
      },
      cache: {
        sdkCacheAvailable: typeof caches !== "undefined",
        coldCacheResetCount,
        note: typeof caches !== "undefined"
          ? "The SDK Cache API entries for this model were cleared before each cold initialization where reset succeeded; browser HTTP cache is unchanged."
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
      results: reportResults,
      measurementOrder: {
        coldInitialization: coldOrder,
        warmInitialization: warmOrder,
        trials: trialOrders,
      },
    };
    report.comparison = compareResults(runtimeResults);
    latestReport = report;
    results.textContent = JSON.stringify(report, null, 2);
    downloadButton.hidden = false;
    status.textContent = report.comparison
      ? `완료 — 런타임 출력 일치: ${report.comparison.passed ? "통과" : "불일치"}`
      : "완료 — 비교 가능한 추론 결과가 부족합니다.";
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

async function initializeRuntime(runtime: "wasm" | "webgpu"): Promise<InitializedRuntime> {
  const started = performance.now();
  const ai = await createLocalAI({ model: { manifestUrl }, runtime });
  return { ai, initializationMs: performance.now() - started };
}

async function measureInference(
  runtime: "wasm" | "webgpu",
  tensor: ort.Tensor,
  iterations: number,
): Promise<InferenceMeasurement> {
  let ai: LocalAI | undefined;
  let initializationMs: number | undefined;
  try {
    const initialized = await initializeRuntime(runtime);
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

async function getModelAssetUrls(): Promise<string[]> {
  const response = await fetch(manifestUrl);
  if (!response.ok) throw new Error(`Could not load model manifest (${response.status}).`);
  const manifest: unknown = await response.json();
  if (typeof manifest !== "object" || manifest === null || !("files" in manifest)) {
    throw new Error("Model manifest does not define files.");
  }
  const files = (manifest as { files: unknown }).files;
  if (typeof files !== "object" || files === null || Array.isArray(files)) {
    throw new Error("Model manifest files must be an object.");
  }
  const paths = Object.values(files);
  if (paths.length === 0 || !paths.every((path) => typeof path === "string")) {
    throw new Error("Model manifest files must contain string paths.");
  }
  const baseUrl = new URL(manifestUrl, location.href);
  return paths.map((path) => new URL(path, baseUrl).href);
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

function compareResults(results: RuntimeResult[]): NonNullable<BenchmarkReport["comparison"]> | undefined {
  const wasm = results.find((result) => result.runtime === "wasm" && result.successfulTrials > 0);
  const webgpu = results.find((result) => result.runtime === "webgpu" && result.successfulTrials > 0);
  if (!wasm?.topResults || !webgpu?.topResults) return undefined;

  const wasmByIndex = new Map(wasm.topResults.map((item) => [item.index, item]));
  const webgpuByIndex = new Map(webgpu.topResults.map((item) => [item.index, item]));
  const sameIndices = wasmByIndex.size === webgpuByIndex.size
    && [...wasmByIndex.keys()].every((index) => webgpuByIndex.has(index));
  const maxDifference = sameIndices
    ? Math.max(...[...wasmByIndex].map(([index, item]) => Math.abs(item.score - webgpuByIndex.get(index)!.score)))
    : null;
  const top1Agrees = wasm.topResults[0]?.index === webgpu.topResults[0]?.index;

  return {
    top1Agrees,
    maxAbsoluteScoreDifference: maxDifference === null ? null : round(maxDifference),
    scoreTolerance,
    passed: sameIndices && top1Agrees && maxDifference !== null && maxDifference <= scoreTolerance,
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
