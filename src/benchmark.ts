import * as ort from "onnxruntime-web";
import "./benchmark.css";
import { createLocalAI, type LocalAI } from "./index";
import { imageToTensor } from "./example-image";

const manifestUrl = "/models/mobilenetv2/manifest.json";
const warmupRuns = 3;
const scoreTolerance = 0.01;
const inputFile = document.querySelector<HTMLInputElement>("#image-file")!;
const iterationsInput = document.querySelector<HTMLInputElement>("#iterations")!;
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
  status: "ok" | "error";
  modelId?: string;
  initializationMs?: number;
  warmupRuns?: number;
  iterations?: number;
  inferenceMs?: { p50: number; p95: number; min: number; max: number };
  topResults?: ClassificationResult[];
  error?: string;
}

interface BenchmarkReport {
  startedAt: string;
  manifestUrl: string;
  input: { name: string; bytes: number; dimensions: number[] };
  settings: { warmupRuns: number; iterations: number; scoreTolerance: number };
  environment: {
    userAgent: string;
    platform: string;
    hardwareConcurrency: number | null;
    deviceMemoryGiB: number | null;
    viewport: { width: number; height: number };
  };
  models: Array<{ modelId: string; runtime: string }>;
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
  if (!Number.isInteger(iterations) || iterations < 5 || iterations > 100) {
    status.textContent = "반복 횟수는 5~100 사이의 정수여야 합니다.";
    return;
  }

  runButton.disabled = true;
  downloadButton.hidden = true;
  results.textContent = "";
  latestReport = undefined;
  const startedAt = new Date().toISOString();
  try {
    status.textContent = "이미지 전처리 중...";
    const tensor = await imageToTensor(selectedFile);
    const environment = getEnvironment();
    const runtimeResults: RuntimeResult[] = [];
    const models: BenchmarkReport["models"] = [];

    for (const runtime of ["wasm", "webgpu"] as const) {
      status.textContent = `${runtime} 초기화 및 측정 중...`;
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const result = await benchmarkRuntime(runtime, tensor, iterations);
      runtimeResults.push(result);
      if (result.status === "ok" && result.modelId) {
        models.push({ modelId: result.modelId, runtime });
      }
    }

    const report: BenchmarkReport = {
      startedAt,
      manifestUrl,
      input: {
        name: selectedFile.name,
        bytes: selectedFile.size,
        dimensions: [1, 3, 224, 224],
      },
      settings: { warmupRuns, iterations, scoreTolerance },
      environment,
      models,
      results: runtimeResults,
    };
    report.comparison = compareResults(runtimeResults);
    latestReport = report;
    results.textContent = JSON.stringify(report, null, 2);
    downloadButton.hidden = false;
    status.textContent = report.comparison
      ? `완료 — 런타임 출력 일치: ${report.comparison.passed ? "통과" : "불일치"}`
      : "완료 — 성공한 런타임이 하나뿐이므로 비교할 수 없습니다.";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "벤치마크에 실패했습니다.";
  } finally {
    runButton.disabled = !selectedFile;
  }
});

async function benchmarkRuntime(
  runtime: "wasm" | "webgpu",
  tensor: ort.Tensor,
  iterations: number,
): Promise<RuntimeResult> {
  let ai: LocalAI | undefined;
  try {
    const initializationStarted = performance.now();
    ai = await createLocalAI({
      model: { manifestUrl },
      runtime,
    });
    const initializationMs = performance.now() - initializationStarted;

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
      runtime: ai.runtime,
      status: "ok",
      modelId: ai.modelId,
      initializationMs: round(initializationMs),
      warmupRuns,
      iterations,
      inferenceMs: summarize(durations),
      topResults: toClassificationResults(finalOutput),
    };
  } catch (error) {
    return {
      runtime,
      status: "error",
      error: error instanceof Error ? error.message : String(error),
    };
  } finally {
    await ai?.dispose();
  }
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
  const wasm = results.find((result) => result.runtime === "wasm" && result.status === "ok");
  const webgpu = results.find((result) => result.runtime === "webgpu" && result.status === "ok");
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
